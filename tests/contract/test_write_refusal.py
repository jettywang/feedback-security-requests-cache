"""Reusable anti-miswrite contract tests for cache backends.

These tests pin down the contract that a backend wired into ``CachedSession`` must
never persist a response when one of the following refusal conditions applies:

1. The cache is disabled for the request (``session.cache_disabled()``).
2. The response/request is marked non-storable (``Cache-Control: no-store``).
3. A user-supplied ``filter_fn`` rejects the response.

The same suite runs against every backend mixed in via :class:`WriteRefusalContractTest`
(currently in-memory and temporary-file SQLite), and each test observes:

* actual backend writes (a spy on the storage's write path),
* actual backend reads (``get_response`` / ``contains``),
* the observable return value of the request (``from_cache`` / status / body).

Fail-open mode
--------------
Set ``REQUESTS_CACHE_FAIL_OPEN=1`` to temporarily relax the refusal gate. Each test
then fails with a message naming the exact contract clause that was breached,
instead of a generic assertion error. This makes the suite a tripwire when
integrating a custom backend: flip the switch to confirm each clause is enforced
independently, then leave it off (the default) in CI.
"""

import os
from contextlib import nullcontext
from unittest.mock import PropertyMock, patch

import pytest
from requests_mock import ANY as ANY_METHOD
from requests_mock import Adapter

from requests_cache import CachedSession
from requests_cache.backends import BaseCache, SQLiteCache
from tests.conftest import MOCK_PROTOCOLS

# When enabled, the write gate is bypassed to simulate a backend that fails open
FAIL_OPEN = os.getenv('REQUESTS_CACHE_FAIL_OPEN', '').lower() in {'1', 'true', 'yes'}

CONTRACT_URL = 'http+mock://write-refusal.contract/resource'
NO_STORE_REQUEST_URL = 'http+mock://write-refusal.contract/no-store-request'
NO_STORE_RESPONSE_URL = 'http+mock://write-refusal.contract/no-store-response'
FILTERED_URL = 'http+mock://write-refusal.contract/filtered'
UNRELATED_URL = 'http+mock://write-refusal.contract/allowed'

# Stable, human-readable identifiers for each clause. These mirror the
# ``write_criteria`` keys used by ``CacheActions.update_from_response`` so a breach
# message maps directly back to the policy rule that should have fired.
CLAUSE_DISABLED = 'disabled cache'
CLAUSE_NO_STORE = 'disabled by headers'
CLAUSE_FILTER = 'disabled by filter'


class StorageWriteSpy:
    """Record every mutating write to a backend's response storage.

    Some backends persist through ``__setitem__`` only (``DictStorage``), while
    others route through a dedicated ``_write`` hook (``SQLiteDict``). Both entry
    points are observed so the contract holds regardless of backend internals.
    The spy records keys but never changes behavior.
    """

    def __init__(self, storage):
        self.storage = storage
        self.written_keys = []
        storage_type = type(storage)
        self._orig_setitem = storage_type.__setitem__
        self._patchers = [
            patch.object(storage_type, '__setitem__', autospec=True, side_effect=self._on_setitem),
        ]
        self._orig_write = getattr(storage_type, '_write', None)
        if self._orig_write is not None:
            self._patchers.append(
                patch.object(storage_type, '_write', autospec=True, side_effect=self._on_write)
            )

    def _on_setitem(self, storage, key, value):
        self.written_keys.append(key)
        return self._orig_setitem(storage, key, value)

    def _on_write(self, storage, key, value):
        self.written_keys.append(key)
        return self._orig_write(storage, key, value)

    def start(self):
        for patcher in self._patchers:
            patcher.start()
        return self

    def stop(self):
        for patcher in self._patchers:
            patcher.stop()

    @property
    def write_count(self) -> int:
        return len(self.written_keys)


def assert_write_refused(session, spy, clause):
    """Assert nothing reached the backend, and name the breached clause.

    In strict mode a real bypass fails with the specific rule that was lost. In
    fail-open mode the write gate is deliberately held open by the harness, so a
    correct implementation surfaces here as a clause-identified breach -- proving
    the spy can detect (and attribute) a miswrite rather than merely checking flags.
    """
    if spy.write_count != 0:
        pytest.fail(
            f'contract breach ({clause}): {spy.write_count} backend write(s) occurred for keys '
            f'{spy.written_keys!r}, even though the "{clause}" rule requires the response to '
            f'be refused'
        )
    if any(session.cache.get_response(key) is not None for key in spy.written_keys):
        pytest.fail(f'contract breach ({clause}): a refused response is readable from the backend')

    if FAIL_OPEN:
        pytest.fail(
            f'fail-open mode is enabled but no write was observed for "{clause}"; the bypass did '
            f'not reach this code path (set REQUESTS_CACHE_FAIL_OPEN=0 to restore strict mode)'
        )


def assert_no_new_write(session, spy, clause, before):
    """Assert a specific request added no writes, attributed to ``clause``."""
    if spy.write_count != before:
        pytest.fail(
            f'contract breach ({clause}): {spy.write_count - before} new backend write(s) '
            f'occurred for keys {spy.written_keys[before:]!r}, even though the "{clause}" rule '
            f'requires the response to be refused'
        )
    if FAIL_OPEN:
        pytest.fail(
            f'fail-open mode is enabled but no write was observed for "{clause}" (set '
            f'REQUESTS_CACHE_FAIL_OPEN=0 to restore strict mode)'
        )


def build_mock_adapter():
    adapter = Adapter()
    adapter.register_uri(
        ANY_METHOD,
        CONTRACT_URL,
        headers={'Content-Type': 'text/plain'},
        text='contract body',
        status_code=200,
    )
    adapter.register_uri(
        ANY_METHOD,
        NO_STORE_REQUEST_URL,
        headers={'Content-Type': 'text/plain'},
        text='no-store request body',
        status_code=200,
    )
    adapter.register_uri(
        ANY_METHOD,
        NO_STORE_RESPONSE_URL,
        headers={'Content-Type': 'text/plain', 'Cache-Control': 'no-store'},
        text='no-store response body',
        status_code=200,
    )
    adapter.register_uri(
        ANY_METHOD,
        FILTERED_URL,
        headers={'Content-Type': 'text/plain'},
        text='filtered body',
        status_code=200,
    )
    adapter.register_uri(
        ANY_METHOD,
        UNRELATED_URL,
        headers={'Content-Type': 'text/plain'},
        text='allowed body',
        status_code=200,
    )
    return adapter


class WriteRefusalContractTest:
    """Backend-agnostic contract suite.

    Subclasses define ``backend_factory`` (a ``(cache_name) -> BaseCache`` callable)
    and a unique ``backend_label``. Each test gets an isolated session and a single
    write spy covering its whole lifetime.
    """

    backend_factory = None
    backend_label = 'unconfigured'

    @pytest.fixture()
    def harness(self, tmp_path):
        cache_name = str(tmp_path / f'write_refusal_{self.backend_label}')
        backend = self.backend_factory(cache_name)
        session = CachedSession(backend=backend, allowable_methods=('GET', 'POST'))
        adapter = build_mock_adapter()
        for protocol in MOCK_PROTOCOLS:
            session.mount(protocol, adapter)
        session.mock_adapter = adapter

        spy = StorageWriteSpy(session.cache.responses).start()
        # In fail-open mode, hold the write gate open to emulate a backend that
        # ignores every refusal rule. The spy must then observe a real backend
        # write, and each test attributes that miswrite to its specific clause.
        gate = (
            patch(
                'requests_cache.session.CacheActions.skip_write',
                new_callable=PropertyMock,
                return_value=False,
            )
            if FAIL_OPEN
            else nullcontext()
        )
        with gate:
            try:
                yield session, spy
            finally:
                spy.stop()
                session.close()

    # -- Clause 1: disabled cache -------------------------------------------
    def test_disabled_cache_request_is_not_written(self, harness):
        session, spy = harness
        with session.cache_disabled():
            response = session.get(CONTRACT_URL)

        assert response.status_code == 200
        assert response.text == 'contract body'
        assert response.from_cache is False
        assert_write_refused(session, spy, CLAUSE_DISABLED)

        # Cache re-enabled: the earlier response must still be absent, so the
        # repeat request misses and hits the origin again.
        assert session.get(CONTRACT_URL).from_cache is False

    def test_disabled_cache_skips_read_and_write(self, harness):
        session, spy = harness
        with patch.object(session.cache, 'get_response', wraps=session.cache.get_response) as read:
            with session.cache_disabled():
                response = session.get(CONTRACT_URL)
            assert response.from_cache is False
            if read.call_count != 0:
                pytest.fail(
                    f'contract breach ({CLAUSE_DISABLED}): {read.call_count} cache read(s) '
                    f'happened while the cache was disabled'
                )
        assert_write_refused(session, spy, CLAUSE_DISABLED)

    # -- Clause 2: non-storable (no-store) ----------------------------------
    def test_no_store_request_header_is_not_written(self, harness):
        """A request ``Cache-Control: no-store`` is always honored, even without cache_control."""
        session, spy = harness
        response = session.get(NO_STORE_REQUEST_URL, headers={'Cache-Control': 'no-store'})

        assert response.status_code == 200
        assert response.text == 'no-store request body'
        assert response.from_cache is False
        assert_write_refused(session, spy, CLAUSE_NO_STORE)
        repeated = session.get(
            NO_STORE_REQUEST_URL, headers={'Cache-Control': 'no-store'}
        )
        assert repeated.from_cache is False

    def test_no_store_response_header_is_not_written(self, harness):
        """With ``cache_control=True`` a response ``no-store`` must refuse the write."""
        session, spy = harness
        session.settings.cache_control = True
        response = session.get(NO_STORE_RESPONSE_URL)

        assert response.status_code == 200
        assert response.text == 'no-store response body'
        assert response.from_cache is False
        assert_write_refused(session, spy, CLAUSE_NO_STORE)
        assert session.get(NO_STORE_RESPONSE_URL).from_cache is False

    def test_no_store_does_not_poison_neighboring_cacheable_response(self, harness):
        session, spy = harness
        session.settings.cache_control = True
        session.get(UNRELATED_URL)
        assert session.get(UNRELATED_URL).from_cache is True

        before = spy.write_count
        session.get(NO_STORE_RESPONSE_URL)
        assert_no_new_write(session, spy, CLAUSE_NO_STORE, before)
        # The unrelated, cacheable entry must survive untouched.
        assert session.get(UNRELATED_URL).from_cache is True

    # -- Clause 3: rejected filter callback ---------------------------------
    def test_rejected_filter_callback_is_not_written(self, harness):
        session, spy = harness
        session.settings.filter_fn = lambda response: response.url != FILTERED_URL
        response = session.get(FILTERED_URL)

        assert response.status_code == 200
        assert response.text == 'filtered body'
        assert response.from_cache is False
        assert_write_refused(session, spy, CLAUSE_FILTER)
        assert session.get(FILTERED_URL).from_cache is False

    def test_filter_callback_keeps_accepted_responses(self, harness):
        """The filter gate must refuse rejected responses without breaking normal caching."""
        session, spy = harness
        session.settings.filter_fn = lambda response: response.url != FILTERED_URL

        session.get(UNRELATED_URL)
        assert session.get(UNRELATED_URL).from_cache is True

        # The rejected response must not create any new cache entry...
        before = spy.write_count
        session.get(FILTERED_URL)
        assert_no_new_write(session, spy, CLAUSE_FILTER, before)
        assert not session.cache.contains(url=FILTERED_URL)
        # ...and the accepted, previously cached entry stays readable.
        assert session.get(UNRELATED_URL).from_cache is True

    @pytest.mark.skipif(
        FAIL_OPEN,
        reason='exercises the retroactive delete path, which the write-gate bypass does not affect',
    )
    def test_retroactive_filter_deletes_previously_cached_response(self, harness):
        """A response cached before a filter rejects it is served once but deleted on access.

        The read still returns the cached item (``from_cache=True``); the filter runs in
        ``send()`` after the read and removes the now-rejected entry.
        """
        session, spy = harness
        session.get(FILTERED_URL)
        assert session.cache.contains(url=FILTERED_URL)

        session.settings.filter_fn = lambda response: response.url != FILTERED_URL
        response = session.get(FILTERED_URL)
        assert response.from_cache is True
        assert not session.cache.contains(url=FILTERED_URL)
        # With the entry gone, the next request misses and reaches the origin.
        assert session.get(FILTERED_URL).from_cache is False
        assert not session.cache.contains(url=FILTERED_URL)


class TestMemoryWriteRefusal(WriteRefusalContractTest):
    backend_label = 'memory'
    backend_factory = staticmethod(lambda cache_name: BaseCache(cache_name=cache_name))


class TestSQLiteWriteRefusal(WriteRefusalContractTest):
    backend_label = 'sqlite'
    backend_factory = staticmethod(lambda cache_name: SQLiteCache(cache_name))
