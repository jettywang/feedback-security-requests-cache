#!/usr/bin/env python
"""
Replay revalidation behavior with a local mock adapter (no network required):

1. Normal 304 revalidation: the cached copy is kept and later requests are cache hits.
2. Policy change on revalidation: a 304 response introduces ``Cache-Control: no-store``.
   The caller still receives the revalidated content, but the cached copy is evicted, so the
   next request has to go back to the origin.

Run with: python examples/revalidation_no_store.py
"""
from tempfile import TemporaryDirectory

from requests_mock import Adapter

from requests_cache import CachedSession

URL = 'mock+http://example.com/resource'
ETAG = '"v1"'


def create_session(cache_dir: str, name: str) -> CachedSession:
    # cache_control=True is required for response Cache-Control directives to be honored
    session = CachedSession(f'{cache_dir}/{name}.db', backend='sqlite', cache_control=True)
    adapter = Adapter()
    session.mount('mock+http://', adapter)
    session.mock_adapter = adapter
    return session


def scenario_normal_revalidation(session: CachedSession):
    print('=== Scenario 1: normal 304 revalidation (policy unchanged) ===')
    session.mock_adapter.register_uri(
        'GET', URL, status_code=200, text='v1 payload', headers={'ETag': ETAG}
    )
    first = session.get(URL)
    print(f'1st request : from_cache={first.from_cache}, stored={session.cache.contains(url=URL)}, '
          f'origin calls={session.mock_adapter.call_count}')

    session.mock_adapter.register_uri('GET', URL, status_code=304, headers={'ETag': ETAG})
    revalidated = session.get(URL, refresh=True)
    print(f'2nd request : from_cache={revalidated.from_cache}, returned={revalidated.text!r}, '
          f'stored={session.cache.contains(url=URL)}, origin calls={session.mock_adapter.call_count}')

    next_response = session.get(URL)
    print(f'3rd request : from_cache={next_response.from_cache}, returned={next_response.text!r}, '
          f'stored={session.cache.contains(url=URL)}, origin calls={session.mock_adapter.call_count}')
    assert revalidated.text == first.text
    assert session.cache.contains(url=URL) is True
    assert next_response.from_cache is True


def scenario_no_store_on_revalidation(session: CachedSession):
    print('\n=== Scenario 2: 304 revalidation switches policy to no-store ===')
    session.mock_adapter.register_uri(
        'GET',
        URL,
        status_code=200,
        text='v1 payload',
        headers={'ETag': ETAG, 'Cache-Control': 'max-age=0'},
    )
    first = session.get(URL)
    print(f'1st request : from_cache={first.from_cache}, stored={session.cache.contains(url=URL)}, '
          f'origin calls={session.mock_adapter.call_count}')

    # The origin answers the conditional request with 304 + no-store
    session.mock_adapter.register_uri(
        'GET', URL, status_code=304,
        headers={'ETag': ETAG, 'Cache-Control': 'no-store'},
    )
    revalidated = session.get(URL)
    print(f'2nd request : from_cache={revalidated.from_cache}, returned={revalidated.text!r}, '
          f'stored={session.cache.contains(url=URL)}, origin calls={session.mock_adapter.call_count}')

    next_response = session.get(URL)
    print(f'3rd request : from_cache={next_response.from_cache}, origin calls='
          f'{session.mock_adapter.call_count}, stored={session.cache.contains(url=URL)}')
    # Caller still gets the revalidated content from the in-memory response...
    assert revalidated.text == first.text
    assert revalidated.revalidated is True
    # ...but no copy remains on disk, so the next request reaches the origin again
    assert session.cache.contains(url=URL) is False
    assert next_response.from_cache is False
    assert session.mock_adapter.call_count == 3


def main():
    with TemporaryDirectory() as cache_dir:
        scenario_normal_revalidation(create_session(cache_dir, 'normal'))
        scenario_no_store_on_revalidation(create_session(cache_dir, 'no_store'))
    print('\nAll assertions passed.')


if __name__ == '__main__':
    main()
