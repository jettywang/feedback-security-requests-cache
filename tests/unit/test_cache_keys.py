"""The cache_keys module is mostly covered indirectly via other tests.
This just contains tests for some extra edge cases not covered elsewhere.
"""

from io import BytesIO
import json

import pytest
from requests import Request, Response
from urllib3 import HTTPResponse
from unittest.mock import patch

from requests_cache.cache_keys import (
    MAX_NORM_BODY_SIZE,
    create_key,
    normalize_headers,
    normalize_request,
    project_response,
    redact_response,
)
from requests_cache.models import CachedRequest, CachedResponse
from requests.structures import CaseInsensitiveDict

CACHE_KEY = 'e25f7e6326966e82'


@pytest.mark.parametrize(
    'url, params',
    [
        ('https://example.com?foo=bar&param=1', None),
        ('https://example.com?foo=bar&param=1', {}),
        ('https://example.com/?foo=bar&param=1', {}),
        ('https://example.com?foo=bar&param=1&', {}),
        ('https://example.com?param=1&foo=bar', {}),
        ('https://example.com?param=1', {'foo': 'bar'}),
        ('https://example.com?foo=bar', {'param': '1'}),
        ('https://example.com', {'foo': 'bar', 'param': '1'}),
        ('https://example.com', {'param': '1', 'foo': 'bar'}),
        ('https://example.com', {'foo': 'bar', 'param': 1}),
        ('https://example.com?', {'foo': 'bar', 'param': '1'}),
    ],
)
def test_create_key__normalize_url_params(url, params):
    """All of the above variations should produce the same cache key"""
    request = Request(
        method='GET',
        url=url,
        params=params,
    )
    assert create_key(request) == CACHE_KEY


def test_create_key__normalize_key_only_params():
    request_1 = Request(method='GET', url='https://img.site.com/base/img.jpg?param_1')
    request_2 = Request(method='GET', url='https://img.site.com/base/img.jpg?param_2')
    assert create_key(request_1) != create_key(request_2)

    request_1 = Request(method='GET', url='https://img.site.com/base/img.jpg?k=v&param_1')
    request_2 = Request(method='GET', url='https://img.site.com/base/img.jpg?param_1&k=v')
    assert create_key(request_1) == create_key(request_2)


def test_create_key__normalize_duplicate_params():
    request_1 = Request(method='GET', url='https://img.site.com/base/img.jpg?param_1=a&param_1=b')
    request_2 = Request(method='GET', url='https://img.site.com/base/img.jpg?param_1=a')
    request_3 = Request(method='GET', url='https://img.site.com/base/img.jpg?param_1=b')
    assert create_key(request_1) != create_key(request_2) != create_key(request_3)

    request_1 = Request(
        method='GET', url='https://img.site.com/base/img.jpg?param_1=a&param_1=b&k=v'
    )
    request_2 = Request(
        method='GET',
        url='https://img.site.com/base/img.jpg?param_1=b&param_1=a',
        params={'k': 'v'},
    )
    assert create_key(request_1) == create_key(request_2)


def test_create_key__fips_hash_fallback():
    """Test that if blake2b fails due to FIPS mode, it creates a valid key using a different
    hash function. Fallback on TypeError and ValueError - both are possible in FIPS mode.
    """
    request = Request(method='GET', url='https://example.com')
    key_1 = create_key(request)

    with patch('requests_cache.cache_keys.blake2b') as mock_blake2b:
        mock_blake2b.side_effect = TypeError
        key_2 = create_key(request)
        mock_blake2b.side_effect = ValueError
        key_3 = create_key(request)

    assert key_1 != key_2
    assert key_2 == key_3  # Fallback to the same algorithm


def test_redact_response__escaped_params():
    """Test that redact_response() handles urlescaped request parameters"""
    url = 'https://img.site.com/base/img.jpg?where=code%3D123'
    request = Request(method='GET', url=url).prepare()
    response = Response()
    response.url = url
    response.request = request
    response.raw = HTTPResponse(request_url=url)
    redacted_response = redact_response(response, [])
    assert redacted_response.url == 'https://img.site.com/base/img.jpg?where=code%3D123'
    assert redacted_response.request.url == 'https://img.site.com/base/img.jpg?where=code%3D123'
    assert redacted_response.request.path_url == '/base/img.jpg?where=code%3D123'
    assert (
        redacted_response.raw._request_url == 'https://img.site.com/base/img.jpg?where=code%3D123'
    )
    if hasattr(redacted_response.raw, 'url'):
        assert redacted_response.raw.url == 'https://img.site.com/base/img.jpg?where=code%3D123'


@pytest.mark.parametrize(
    'content_type',
    [
        'application/json',
        'application/json; charset=utf-8',
        'application/vnd.api+json; charset=utf-8',
        'application/any_string+json',
    ],
)
@pytest.mark.parametrize(
    'data',
    [
        b'{"param_1": "value_1", "param_2": "value_2"}',
        b'["param_3", "param_2", "param_1"',
    ],
)
def test_normalize_request__json_body(data, content_type):
    request = Request(
        method='GET',
        url='https://img.site.com/base/img.jpg',
        data=b'{"param_1": "value_1", "param_2": "value_2"}',
        headers={'Content-Type': content_type},
    )
    norm_request = normalize_request(request, ignored_parameters=['param_2'])
    assert norm_request.body == b'{"param_1": "value_1", "param_2": "REDACTED"}'


def test_normalize_request__json_body_list_filtered():
    request = Request(
        method='GET',
        url='https://img.site.com/base/img.jpg',
        data=b'["param_3", "param_2", "param_1"]',
        headers={'Content-Type': 'application/json'},
    )
    norm_request = normalize_request(request, ignored_parameters=['param_2', 'param_1'])
    assert norm_request.body == b'["param_3"]'


def test_normalize_request__json_body_invalid():
    request = Request(
        method='GET',
        url='https://img.site.com/base/img.jpg',
        data=b'invalid JSON!',
        headers={'Content-Type': 'application/json'},
    )
    assert normalize_request(request, ignored_parameters=['param_2']).body == b'invalid JSON!'


def test_normalize_request__json_body_empty():
    request = Request(
        method='GET',
        url='https://img.site.com/base/img.jpg',
        data=b'{}',
        headers={'Content-Type': 'application/json'},
    )
    assert normalize_request(request, ignored_parameters=['param_2']).body == b'{}'


@pytest.mark.parametrize(
    'content_type',
    ['application/octet-stream', None],
)
def test_normalize_request__binary_body(content_type):
    request = Request(
        method='GET',
        url='https://img.site.com/base/img.jpg',
        data=b'some bytes',
        headers={'Content-Type': content_type},
    )
    assert normalize_request(request, ignored_parameters=['param']).body == request.data


def test_normalize_request__oversized_body():
    body = {'param': '1', 'content': '0' * MAX_NORM_BODY_SIZE}
    encoded_body = json.dumps(body).encode('utf-8')

    request = Request(
        method='GET',
        url='https://img.site.com/base/img.jpg',
        json=body,
        headers={'Content-Type': 'application/octet-stream'},
    )
    assert normalize_request(request, ignored_parameters=['param']).body == encoded_body


def test_normalize_request__file_like_body():
    original_body = BytesIO(b'some bytes')
    request = Request(
        method='GET',
        url='https://img.site.com/base/img.jpg',
        data=original_body,
        headers={'Content-Type': 'application/json'},
    )
    assert normalize_request(request).body == b'some bytes'
    assert original_body.read() == b'some bytes'


def test_normalize_request__file_like_reset_fails():
    """Test a request body with a file-like class that doesn't support seek()"""

    class CustomBytesIO:
        def __init__(self, content):
            self.content = content

        def __len__(self):
            return len(self.content)

        def read(self):
            return self.content

    original_body = CustomBytesIO(b'some bytes')
    request = Request(
        method='GET',
        url='https://img.site.com/base/img.jpg',
        data=original_body,
        headers={'Content-Type': 'application/json'},
    )
    assert normalize_request(request).body == b'some bytes'
    assert original_body.read() == b'some bytes'


def test_normalize_headers__single_header_value_as_bytes():
    norm_headers = normalize_headers({'Accept': b'gzip'})
    assert norm_headers == {'Accept': 'gzip'}


def test_normalize_headers__multiple_header_values_as_bytes():
    headers = {'Accept': b'gzip,  deflate,Venmo,  PayPal, '}
    norm_headers = normalize_headers(headers)
    assert norm_headers == {'Accept': 'deflate, gzip, paypal, venmo'}


def test_normalize_headers__single_header_value_as_string():
    norm_headers = normalize_headers({'Accept': 'gzip'})
    assert norm_headers == {'Accept': 'gzip'}


def test_normalize_headers__multiple_header_values_as_string():
    headers = {'Accept': 'gzip,  deflate,Venmo,  PayPal, '}
    norm_headers = normalize_headers(headers)
    assert norm_headers == {'Accept': 'deflate, gzip, paypal, venmo'}


def test_remove_ignored_headers__empty():
    request = Request(
        method='GET',
        url='https://img.site.com/base/img.jpg',
        headers={'foo': 'bar'},
    )
    assert normalize_request(request.prepare(), ignored_parameters=None).headers == request.headers


def _make_cached_response(
    url='https://example.com/test',
    request_headers=None,
    response_headers=None,
    body=b'{"secret": "payload"}',
):
    request = CachedRequest(
        method='POST',
        url=url,
        body=body,
        headers=CaseInsensitiveDict(
            request_headers
            or {
                'Authorization': 'Bearer token',
                'Accept': 'application/json',
                'User-Agent': 'test-client',
            }
        ),
    )
    return CachedResponse(
        status_code=200,
        url=url,
        request=request,
        headers=CaseInsensitiveDict(response_headers or {}),
    )


def test_project_response__no_settings_is_noop():
    response = _make_cached_response()
    projected = project_response(response, None)
    assert projected is response
    assert projected.request.headers['Authorization'] == 'Bearer token'
    assert projected.request.body == b'{"secret": "payload"}'


def test_project_response__keeps_only_selected_fields():
    response = _make_cached_response()
    projected = project_response(response, ['method', 'url'])
    assert projected.request.method == 'POST'
    assert projected.request.url == 'https://example.com/test'
    # Non-selected metadata is dropped from the serialized request
    assert dict(projected.request.headers) == {}
    assert projected.request.body == b''
    assert len(projected.request.cookies) == 0


def test_project_response__retains_vary_headers_without_full_headers():
    # A response varying on Accept must keep Accept on the request even when headers
    # aren't in the selected fields, so Vary validation still works after reopening
    response = _make_cached_response(response_headers={'Vary': 'Accept'})
    projected = project_response(response, ['method', 'url'])
    assert dict(projected.request.headers) == {'Accept': 'application/json'}


def test_project_response__retains_match_headers():
    response = _make_cached_response()
    projected = project_response(response, ['method', 'url'], match_headers=['Accept'])
    assert dict(projected.request.headers) == {'Accept': 'application/json'}


def test_project_response__match_all_headers_retains_everything():
    response = _make_cached_response()
    projected = project_response(response, ['method', 'url'], match_headers=True)
    assert projected.request.headers == CaseInsensitiveDict(
        {
            'Authorization': 'Bearer token',
            'Accept': 'application/json',
            'User-Agent': 'test-client',
        }
    )


def test_project_response__vary_cookie_retains_cookies():
    from requests.cookies import RequestsCookieJar

    jar = RequestsCookieJar()
    jar.set('session', 'abc123', domain='example.com', path='/')
    request = CachedRequest(
        method='GET',
        url='https://example.com/test',
        cookies=jar,
        headers=CaseInsensitiveDict({'User-Agent': 'test-client'}),
    )
    response = CachedResponse(
        status_code=200,
        url='https://example.com/test',
        request=request,
        headers=CaseInsensitiveDict({'Vary': 'Cookie'}),
    )
    projected = project_response(response, ['method', 'url'])
    assert projected.request.cookies.get('session') == 'abc123'
    assert dict(projected.request.headers) == {}


def test_project_response__selected_body_and_headers():
    response = _make_cached_response()
    projected = project_response(response, ['method', 'url', 'body', 'headers'])
    assert projected.request.body == b'{"secret": "payload"}'
    assert projected.request.headers['Authorization'] == 'Bearer token'


def test_project_response__preserves_vary_key_inputs():
    # The retained Vary header must still match on the same terms as the live request,
    # so Vary validation keeps working after the projected response is reopened
    from requests_cache.cache_keys import get_matched_headers

    response = _make_cached_response(response_headers={'Vary': 'Accept'})
    live_request = response.request.copy()
    projected = project_response(response, ['method', 'url'])
    assert get_matched_headers(projected.request.headers, ['accept']) == get_matched_headers(
        live_request.headers, ['accept']
    )


def test_project_response__redirect_history():
    target = _make_cached_response(
        url='https://example.com/target',
        request_headers={'Accept': 'text/html', 'Authorization': 'Bearer token'},
        response_headers={'Vary': 'Accept'},
    )
    redirect = _make_cached_response(
        url='https://example.com/redirect',
        request_headers={'Authorization': 'Bearer token'},
        response_headers={'Location': 'https://example.com/target'},
        body=b'',
    )
    target.history = [redirect]
    projected = project_response(target, ['method', 'url'])
    assert dict(projected.request.headers) == {'Accept': 'text/html'}
    assert dict(projected.history[0].request.headers) == {}
