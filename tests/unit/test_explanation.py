"""Tests for redaction-safe cache decision explanations.

The main goal of these tests is parity: for the same batch of requests, the stable
codes returned by the explanation helpers must match what a real CachedSession does
(origin contact, cache hits, storage, and conditional revalidation).
"""

import json
import time
from datetime import timedelta

import pytest
from requests import Request
from requests_mock import Adapter

from requests_cache import (
    DO_NOT_CACHE,
    CachedSession,
    Outcome,
    Reason,
    explain_request,
    explain_response_storage,
)
from requests_cache.policy import utcnow
from tests.conftest import ETAG

URL = 'mock://requests-cache.test/resource'
URL_ETAG = 'mock://requests-cache.test/etag'
URL_VARY = 'mock://requests-cache.test/vary'
URL_404 = 'mock://requests-cache.test/missing'
SECRET = 'super-secret-token-value'


@pytest.fixture
def mock_session(request):
    kwargs = getattr(request, 'param', {})
    session = CachedSession(backend='memory', **kwargs)
    adapter = Adapter()
    session.mount('mock://', adapter)
    adapter.register_uri('GET', URL, text='body', status_code=200)
    adapter.register_uri(
        'GET',
        URL_ETAG,
        [
            {'status_code': 200, 'headers': {'ETag': ETAG}, 'text': 'v1'},
            {'status_code': 304, 'headers': {'ETag': ETAG}},
        ],
    )
    adapter.register_uri(
        'GET', URL_VARY, headers={'Vary': 'Accept'}, text='variant'
    )
    adapter.register_uri('GET', URL_404, status_code=404)
    adapter.register_uri('POST', URL, text='posted')
    yield session, adapter
    session.cache.clear()


def assert_origin_parity(session, adapter, decision, **send_kwargs):
    """Run the real request and check origin-contact behavior against the explanation."""
    before = adapter.call_count
    response = session.request(
        'GET', send_kwargs.pop('url', URL), **send_kwargs
    )
    contacted_origin = adapter.call_count - before

    if decision.outcome in (Outcome.CACHE_HIT, Outcome.NOT_CACHED_504):
        assert contacted_origin == 0
    else:
        assert contacted_origin == 1

    if decision.outcome == Outcome.NOT_CACHED_504:
        assert response.status_code == 504
    elif decision.outcome == Outcome.CACHE_HIT:
        assert response.from_cache is True
    elif decision.outcome in (Outcome.ORIGIN_REQUEST, Outcome.CONDITIONAL_REQUEST):
        assert response.from_cache is False or response.status_code == 304
    return response


# Policy branch: miss, hit, disabled, method, only-if-cached, do-not-cache
# ------------------------------------------------------------------------


def test_explain_miss_then_hit_parity(mock_session):
    session, adapter = mock_session

    first = explain_request(session, 'GET', URL)
    assert first.outcome == Outcome.ORIGIN_REQUEST
    assert first.reason == Reason.CACHE_MISS
    assert first.cache_hit is False and first.would_store is True
    response = assert_origin_parity(session, adapter, first)
    assert response.from_cache is False

    second = explain_request(session, 'GET', URL)
    assert second.outcome == Outcome.CACHE_HIT
    assert second.reason == Reason.FRESH_HIT
    assert second.cache_hit is True and second.stale is False
    assert_origin_parity(session, adapter, second)


def test_explain_disabled_cache_parity(mock_session):
    session, adapter = mock_session
    session.get(URL)

    with session.cache_disabled():
        decision = explain_request(session, 'GET', URL)
        assert decision.skip_read is True
        assert decision.reason == Reason.CACHE_DISABLED
        assert Reason.CACHE_DISABLED in decision.write_reasons
        response = assert_origin_parity(session, adapter, decision)
    assert response.from_cache is False
    assert adapter.call_count == 2


def test_explain_method_not_allowed_parity(mock_session):
    session, adapter = mock_session
    decision = explain_request(session, 'POST', URL)
    assert decision.skip_read is True
    assert decision.reason == Reason.METHOD_NOT_ALLOWED
    assert decision.write_reasons == [Reason.METHOD_NOT_ALLOWED]

    before = adapter.call_count
    session.post(URL)
    session.post(URL)
    assert adapter.call_count - before == 2  # Nothing cached between POSTs


def test_explain_only_if_cached_parity(mock_session):
    session, adapter = mock_session
    decision = explain_request(session, 'GET', URL, only_if_cached=True)
    assert decision.outcome == Outcome.NOT_CACHED_504
    assert decision.reason == Reason.ONLY_IF_CACHED_MISS
    response = assert_origin_parity(
        session, adapter, decision, url=URL, only_if_cached=True
    )
    assert response.status_code == 504


def test_explain_do_not_cache_parity(mock_session):
    session, adapter = mock_session
    decision = explain_request(session, 'GET', URL, expire_after=DO_NOT_CACHE)
    assert decision.skip_read is True
    assert decision.reason == Reason.DO_NOT_CACHE
    # The read path is bypassed entirely, so nothing existing can satisfy the request
    assert decision.outcome == Outcome.ORIGIN_REQUEST

    before = adapter.call_count
    response = session.get(URL, expire_after=DO_NOT_CACHE)
    session.get(URL, expire_after=DO_NOT_CACHE)
    assert adapter.call_count - before == 2  # Cache is never read, so origin is hit each time
    # Parity: the explanation must agree with what the real session did about storage
    storage = explain_response_storage(session, response)
    key = session.cache.create_key(response.request)
    assert storage.would_store == (key in session.cache.responses)


def test_explain_vary_miss_then_match_parity(mock_session):
    session, adapter = mock_session

    session.get(URL_VARY, headers={'Accept': 'application/json'})
    mismatch = explain_request(session, 'GET', URL_VARY, headers={'Accept': 'text/html'})
    assert mismatch.reason == Reason.VARY_MISMATCH
    assert mismatch.outcome == Outcome.ORIGIN_REQUEST
    assert_origin_parity(session, adapter, mismatch, url=URL_VARY, headers={'Accept': 'text/html'})

    match = explain_request(
        session, 'GET', URL_VARY, headers={'Accept': 'application/json'}
    )
    assert match.outcome == Outcome.CACHE_HIT
    assert_origin_parity(
        session, adapter, match, url=URL_VARY, headers={'Accept': 'application/json'}
    )


def test_explain_stale_accepted_with_max_stale(mock_session):
    session, adapter = mock_session
    session.get(URL)
    key = session.cache.create_key(session.prepare_request(Request('GET', URL)))
    session.cache.get_response(key).expires = utcnow() - timedelta(seconds=10)

    decision = explain_request(
        session, 'GET', URL, headers={'Cache-Control': 'max-stale=60'}
    )
    assert decision.outcome == Outcome.CACHE_HIT
    assert decision.reason == Reason.STALE_ACCEPTED
    assert decision.stale is True
    assert_origin_parity(
        session, adapter, decision, url=URL,
        headers={'Cache-Control': 'max-stale=60'},
    )


def test_explain_stale_while_revalidate_background(mock_session):
    session, adapter = mock_session
    session.settings.stale_while_revalidate = True
    session.get(URL)
    key = session.cache.create_key(session.prepare_request(Request('GET', URL)))
    session.cache.get_response(key).expires = utcnow() - timedelta(seconds=10)

    decision = explain_request(session, 'GET', URL)
    assert decision.outcome == Outcome.BACKGROUND_REFRESH
    assert decision.reason == Reason.STALE_WHILE_REVALIDATE
    assert decision.cache_hit is True

    before = adapter.call_count
    response = session.get(URL)
    assert response.from_cache is True  # Stale body returned immediately
    deadline = time.time() + 2
    while adapter.call_count == before and time.time() < deadline:
        time.sleep(0.01)
    assert adapter.call_count - before == 1  # Background refresh still contacted origin


# Skip-storage branch
# -------------------


@pytest.mark.parametrize(
    'kwargs, expected_reasons',
    [
        ({'headers': {'Cache-Control': 'no-store'}}, [Reason.RESPONSE_NO_STORE]),
    ],
)
def test_explain_request_time_skip_storage(mock_session, kwargs, expected_reasons):
    session, _ = mock_session
    decision = explain_request(session, 'GET', URL, **kwargs)
    assert decision.skip_write is True
    assert decision.write_reasons == expected_reasons

    session.get(URL, **kwargs)
    key = session.cache.create_key(
        session.prepare_request(Request('GET', URL, headers=kwargs.get('headers')))
    )
    assert key not in session.cache.responses


def test_explain_read_only_skip_storage(mock_session):
    session, _ = mock_session
    session.settings.read_only = True
    decision = explain_request(session, 'GET', URL)
    assert decision.write_reasons == [Reason.READ_ONLY]
    assert decision.would_store is False

    session.get(URL)
    key = session.cache.create_key(session.prepare_request(Request('GET', URL)))
    assert key not in session.cache.responses


def test_explain_response_storage_parity(mock_session):
    session, adapter = mock_session
    session_adapter = adapter
    session_adapter.register_uri(
        'GET', f'{URL}/no-store', headers={'Cache-Control': 'no-store'}, text='b'
    )

    cases = [
        ('GET', URL, {}),
        ('GET', URL_404, {}),
        ('GET', f'{URL}/no-store', {}),
    ]
    for method, url, kwargs in cases:
        response = session.request(method, url, **kwargs)
        storage = explain_response_storage(session, response)
        key = session.cache.create_key(response.request)
        assert storage.would_store == (key in session.cache.responses)
        if not storage.would_store:
            assert storage.reasons
    assert adapter.call_count == len(cases)


def test_explain_response_storage_status_and_filter(mock_session):
    session, _ = mock_session

    missing = session.get(URL_404)
    decision = explain_response_storage(session, missing)
    assert decision.would_store is False
    assert decision.reasons == [Reason.STATUS_NOT_ALLOWED]

    def filter_fn(response):
        return response.status_code != 200

    session.settings.filter_fn = filter_fn
    filtered = session.get(URL)
    decision = explain_response_storage(session, filtered)
    assert decision.would_store is False
    assert decision.reasons == [Reason.FILTERED]
    key = session.cache.create_key(filtered.request)
    assert key not in session.cache.responses

    session.settings.filter_fn = lambda response: True
    stored = session.get(URL)
    decision = explain_response_storage(session, stored)
    assert decision.would_store is True and decision.reasons == []


def test_explain_expired_without_validator_skip_storage(mock_session):
    session, _ = mock_session
    response = session.get(URL, expire_after=0)
    decision = explain_response_storage(session, response)
    assert decision.would_store is False
    assert decision.reasons == [Reason.EXPIRED_NO_VALIDATOR]
    key = session.cache.create_key(response.request)
    assert key not in session.cache.responses


# Revalidation branch
# -------------------


def _expire_cached(session, url):
    key = session.cache.create_key(session.prepare_request(Request('GET', url)))
    session.cache.get_response(key).expires = utcnow() - timedelta(seconds=10)


def test_explain_conditional_revalidation_parity(mock_session):
    session, adapter = mock_session
    session.get(URL_ETAG)
    _expire_cached(session, URL_ETAG)

    decision = explain_request(session, 'GET', URL_ETAG)
    assert decision.outcome == Outcome.CONDITIONAL_REQUEST
    assert decision.reason == Reason.CONDITIONAL_REVALIDATE
    assert decision.revalidates is True
    assert decision.validation_headers == ['If-None-Match']

    before = adapter.call_count
    response = session.get(URL_ETAG)
    assert adapter.call_count - before == 1
    sent_headers = adapter.last_request.headers
    assert 'If-None-Match' in sent_headers
    # A 304 reuses and refreshes the cached body, so the returned response keeps the
    # original status code but is marked as revalidated
    assert response.from_cache is True
    assert response.revalidated is True


def test_explain_stale_without_validator_sends_plain_request(mock_session):
    session, adapter = mock_session
    session.get(URL)
    _expire_cached(session, URL)

    decision = explain_request(session, 'GET', URL)
    assert decision.outcome == Outcome.ORIGIN_REQUEST
    assert decision.reason == Reason.REVALIDATE
    assert decision.revalidates is False
    assert decision.validation_headers == []

    before = adapter.call_count
    session.get(URL)
    assert adapter.call_count - before == 1
    assert 'If-None-Match' not in adapter.last_request.headers
    assert 'If-Modified-Since' not in adapter.last_request.headers


def test_explain_refresh_and_force_refresh_parity(mock_session):
    session, adapter = mock_session
    session.get(URL_ETAG)  # Fresh entry with ETag validator

    soft = explain_request(session, 'GET', URL_ETAG, refresh=True)
    assert soft.outcome == Outcome.CONDITIONAL_REQUEST
    assert soft.revalidates is True
    before = adapter.call_count
    session.get(URL_ETAG, refresh=True)
    assert adapter.call_count - before == 1
    assert 'If-None-Match' in adapter.last_request.headers

    hard = explain_request(session, 'GET', URL_ETAG, force_refresh=True)
    assert hard.skip_read is True
    assert hard.reason == Reason.REQUEST_NO_CACHE
    before = adapter.call_count
    session.get(URL_ETAG, force_refresh=True)
    assert adapter.call_count - before == 1
    assert 'If-None-Match' not in adapter.last_request.headers


def test_explain_always_revalidate_setting(mock_session):
    session, adapter = mock_session
    session.settings.always_revalidate = True
    session.get(URL_ETAG)  # Fresh, but setting forces revalidation

    decision = explain_request(session, 'GET', URL_ETAG)
    assert decision.outcome == Outcome.CONDITIONAL_REQUEST
    assert decision.revalidates is True
    before = adapter.call_count
    session.get(URL_ETAG)
    assert adapter.call_count - before == 1
    assert 'If-None-Match' in adapter.last_request.headers


# Batch parity + redaction
# ------------------------


def test_decision_parity_for_request_batch(mock_session):
    session, adapter = mock_session
    session.get(URL)  # Seed a fresh hit and an expired validator entry
    session.get(URL_ETAG)
    _expire_cached(session, URL_ETAG)

    batch = [
        ('GET', URL, {}),
        ('GET', URL, {'headers': {'Cache-Control': 'no-store'}}),
        ('GET', URL_ETAG, {}),
        ('GET', URL, {'only_if_cached': True}),
        ('GET', URL, {'expire_after': DO_NOT_CACHE}),
        ('POST', URL, {}),
    ]
    for method, url, kwargs in batch:
        decision = explain_request(session, method, url, **kwargs)
        before = adapter.call_count
        session.request(method, url, **kwargs)
        contacted = adapter.call_count - before
        assert contacted == (0 if decision.outcome in
                             (Outcome.CACHE_HIT, Outcome.NOT_CACHED_504) else 1)


def test_explanation_never_echoes_sensitive_inputs(mock_session):
    session, _ = mock_session
    session.get(URL_ETAG, headers={'Authorization': f'Bearer {SECRET}'})
    _expire_cached(session, URL_ETAG)

    decision = explain_request(
        session,
        'GET',
        URL_ETAG,
        headers={'Authorization': f'Bearer {SECRET}'},
    )
    rendered = json.dumps(decision.to_dict())
    assert SECRET not in rendered
    assert URL_ETAG not in rendered
    assert 'authorization' not in rendered.lower()
    # Header names (not values) are allowed for the revalidation explanation
    assert decision.validation_headers == ['If-None-Match']

    response = session.get(URL_ETAG)
    storage = explain_response_storage(session, response)
    assert SECRET not in json.dumps(storage.to_dict())
