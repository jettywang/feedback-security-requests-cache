"""Tests for the restricted, offline cache migration path (``BaseCache.copy_from``)"""

import pickle
from datetime import timedelta
from pickle import PickleError
from unittest.mock import patch

from requests_cache.backends import BaseCache, CacheCopyResult
from requests_cache.backends.sqlite import SQLiteCache
from requests_cache.models import CachedResponse
from requests_cache.policy import utcnow
from tests.conftest import MOCKED_URL, MOCKED_URL_JSON


def make_response(url=MOCKED_URL, no_store=False, expires=None) -> CachedResponse:
    return CachedResponse(status_code=200, url=url, no_store=no_store, expires=expires)


def is_allowed(response: CachedResponse) -> bool:
    return 'reject' not in response.url


def populate_source(cache: BaseCache) -> None:
    future_expires = utcnow() + timedelta(hours=5)
    cache.responses['eligible'] = make_response(MOCKED_URL, expires=future_expires)
    cache.responses['persist_blocked'] = make_response(MOCKED_URL_JSON, no_store=True)
    cache.responses['rejected'] = make_response(f'{MOCKED_URL}?reject=1')
    cache.redirects['redirect_ok'] = 'eligible'
    cache.redirects['redirect_blocked'] = 'persist_blocked'


class TestCopyFromInMemory:
    def test_copies_only_persistable_records(self):
        source = BaseCache()
        populate_source(source)
        dest = BaseCache()

        result = dest.copy_from(source, predicate=is_allowed)

        assert isinstance(result, CacheCopyResult)
        assert (result.copied, result.skipped, result.decode_errors) == (1, 2, 0)
        assert result.total == 3
        assert set(dest.responses) == {'eligible'}
        # Redirects only follow responses that made it into the destination
        assert dict(dest.redirects) == {'redirect_ok': 'eligible'}

    def test_non_persistable_records_never_copied_even_without_predicate(self):
        source = BaseCache()
        source.responses['blocked'] = make_response(no_store=True)
        source.responses['ok'] = make_response()
        dest = BaseCache()

        result = dest.copy_from(source)

        assert (result.copied, result.skipped) == (1, 1)
        assert set(dest.responses) == {'ok'}

    def test_preserves_expiration_time(self):
        original_expires = utcnow() + timedelta(days=2)
        source = BaseCache()
        source.responses['key'] = make_response(expires=original_expires)
        dest = BaseCache()

        dest.copy_from(source)

        copied = dest.responses['key']
        assert copied.expires == original_expires
        assert copied.expires_unix == source.responses['key'].expires_unix

    def test_source_is_not_modified(self):
        source = BaseCache()
        populate_source(source)
        original_keys = set(source.responses)
        original_redirects = dict(source.redirects)
        dest = BaseCache()

        dest.copy_from(source, predicate=is_allowed)

        assert set(source.responses) == original_keys
        assert dict(source.redirects) == original_redirects

    def test_rerun_is_idempotent(self):
        source = BaseCache()
        populate_source(source)
        dest = BaseCache()

        first = dest.copy_from(source, predicate=is_allowed)
        second = dest.copy_from(source, predicate=is_allowed)

        assert first == second == CacheCopyResult(copied=1, skipped=2, decode_errors=0)
        # No duplicate entries after re-running an interrupted migration
        assert len(dest.responses) == 1
        assert len(dest.redirects) == 1

    def test_decode_failure_is_counted_and_left_behind(self):
        source = BaseCache()
        source.responses['good'] = make_response()
        source.responses['undecodable'] = make_response()
        dest = BaseCache()
        original_data = dict(source.responses.data)

        def guarded_getitem(key):
            if key == 'undecodable':
                raise PickleError('corrupt record')
            return original_data[key]

        with patch.object(type(source.responses), '__getitem__', side_effect=guarded_getitem):
            result = dest.copy_from(source)

        assert (result.copied, result.skipped, result.decode_errors) == (1, 0, 1)
        assert set(dest.responses) == {'good'}

    def test_none_deserialization_is_counted_as_decode_failure(self):
        source = BaseCache()
        source.responses['good'] = make_response()
        source.responses['none_record'] = make_response()
        dest = BaseCache()
        original_data = dict(source.responses.data)

        def guarded_getitem(key):
            return None if key == 'none_record' else original_data[key]

        with patch.object(type(source.responses), '__getitem__', side_effect=guarded_getitem):
            result = dest.copy_from(source)

        assert result.decode_errors == 1
        assert result.copied == 1


class TestCopyFromSQLite:
    def test_full_migration_between_persistent_caches(self, tempfile_path):
        source_path = f'{tempfile_path}.source'
        dest_path = f'{tempfile_path}.dest'
        source = SQLiteCache(source_path)
        populate_source(source)
        source.close()

        # Reopen to force a real serialize/deserialize round-trip
        source = SQLiteCache(source_path)
        dest = SQLiteCache(dest_path)
        result = dest.copy_from(source, predicate=is_allowed)
        dest.close()

        assert (result.copied, result.skipped, result.decode_errors) == (1, 2, 0)

        reopened_dest = SQLiteCache(dest_path)
        assert set(reopened_dest.responses) == {'eligible'}
        assert dict(reopened_dest.redirects) == {'redirect_ok': 'eligible'}
        assert reopened_dest.responses['eligible'].no_store is False
        source.close()
        reopened_dest.close()

    def test_no_store_marker_survives_serialization_roundtrip(self, tempfile_path):
        cache = SQLiteCache(tempfile_path, serializer='pickle')
        cache.responses['blocked'] = make_response(no_store=True)
        cache.responses['allowed'] = make_response()
        cache.close()

        reopened = SQLiteCache(tempfile_path, serializer='pickle')
        assert reopened.responses['blocked'].no_store is True
        assert reopened.responses['allowed'].no_store is False

        dest = BaseCache()
        result = dest.copy_from(reopened)
        assert (result.copied, result.skipped) == (1, 1)
        assert set(dest.responses) == {'allowed'}
        reopened.close()

    def test_decode_failure_with_bad_serializer(self, tempfile_path):
        class BadSerializer:
            def dumps(self, value):
                return pickle.dumps(value)

            def loads(self, value):
                response = pickle.loads(value)
                if response.url == MOCKED_URL_JSON:
                    raise PickleError('corrupt record')
                return response

        source = SQLiteCache(f'{tempfile_path}.source', serializer=BadSerializer())
        source.responses['good'] = make_response(MOCKED_URL)
        source.responses['bad'] = make_response(MOCKED_URL_JSON)
        source.close()

        reopened = SQLiteCache(f'{tempfile_path}.source', serializer=BadSerializer())
        dest = SQLiteCache(f'{tempfile_path}.dest')
        result = dest.copy_from(reopened)

        assert (result.copied, result.skipped, result.decode_errors) == (1, 0, 1)
        assert set(dest.responses) == {'good'}
        # The unreadable record remains in the source cache
        assert set(reopened.responses.keys()) == {'good', 'bad'}
        reopened.close()
        dest.close()
