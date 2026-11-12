# Note: Almost all serializer logic is covered by parametrized integration tests.
# Any additional serializer-specific tests can go here.
import gzip
import pickle
import sys
from importlib import reload
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from cattrs import BaseConverter, GenConverter

from requests_cache import (
    CachedResponse,
    CachedSession,
    CattrStage,
    SerializerPipeline,
    Stage,
    json_serializer,
    safe_pickle_serializer,
    utf8_encoder,
    init_serializer,
)
from tests.conftest import skip_missing_deps


@skip_missing_deps('orjson')
@skip_missing_deps('ujson')
def test_json_aliases():
    assert init_serializer('json', decode_content=True).name == 'json'
    assert init_serializer('orjson', decode_content=True).name == 'orjson'
    assert init_serializer('ujson', decode_content=True).name == 'ujson'


@skip_missing_deps('ujson')
@skip_missing_deps('orjson')
def test_json_explicit_lib():
    from requests_cache.serializers.preconf import (
        json_serializer,
        orjson_serializer,
        ujson_serializer,
    )

    response = CachedResponse(status_code=200)
    for obj in [json_serializer, ujson_serializer, orjson_serializer]:
        assert obj.loads(obj.dumps(response)) == response


def test_optional_dependencies():
    import requests_cache.serializers.preconf

    with patch.dict(
        sys.modules,
        {'bson': None, 'itsdangerous': None, 'yaml': None, 'orjson': None, 'ujson': None},
    ):
        reload(requests_cache.serializers.preconf)

        from requests_cache.serializers.preconf import (
            bson_serializer,
            orjson_serializer,
            safe_pickle_serializer,
            ujson_serializer,
            yaml_serializer,
        )

        for obj in [bson_serializer, yaml_serializer, orjson_serializer, ujson_serializer]:
            print(f'Testing serializer {obj.name}')
            with pytest.raises(ImportError):
                obj.dumps('')
            with pytest.raises(ImportError):
                obj.loads('')

        with pytest.raises(ImportError):
            safe_pickle_serializer('')

    reload(requests_cache.serializers.preconf)


@skip_missing_deps('itsdangerous')
def test_cache_signing(tempfile_path):
    from itsdangerous import Signer

    serializer = safe_pickle_serializer(secret_key=str(uuid4()))
    session = CachedSession(tempfile_path, serializer=serializer)
    assert isinstance(session.cache.responses.serializer.stages[-1].obj, Signer)

    # Simple serialize/deserialize round trip
    response = CachedResponse()
    session.cache.responses['key'] = response
    assert session.cache.responses['key'] == response

    # Without the same signing key, the item shouldn't be considered safe to deserialize,
    # and is treated as a cache miss
    serializer = safe_pickle_serializer(secret_key='a different key')
    session = CachedSession(tempfile_path, serializer=serializer)
    assert session.cache.responses['key'] is None
    assert session.cache.get_response('key') is None


@skip_missing_deps('itsdangerous')
def test_cache_signing_verifies_before_decoding():
    """The signature is validated before the payload is passed to the decoder"""
    from itsdangerous.exc import BadSignature

    serializer = safe_pickle_serializer(secret_key=str(uuid4()))
    # Not a valid signature or pickle payload; the signature check must fail first
    with pytest.raises(BadSignature):
        serializer.loads(b'garbage payload')


@skip_missing_deps('itsdangerous')
def test_cache_signing_key_rotation(tempfile_path):
    """During key rotation, records signed with an old key should still be readable, and new
    records should be signed with the current key"""
    old_key = str(uuid4())
    new_key = str(uuid4())

    # Write a record with the old key
    old_serializer = safe_pickle_serializer(secret_key=old_key)
    session = CachedSession(tempfile_path, serializer=old_serializer)
    old_record = CachedResponse()
    session.cache.responses['old_key_record'] = old_record
    session.close()

    # After rotation, records signed with the old key are still readable
    rotated_serializer = safe_pickle_serializer(secret_key=new_key, old_secret_keys=[old_key])
    session = CachedSession(tempfile_path, serializer=rotated_serializer)
    assert session.cache.responses['old_key_record'] == old_record

    # New records are signed with the current key, not an old one
    new_record = CachedResponse()
    session.cache.responses['new_key_record'] = new_record
    session.close()
    session = CachedSession(tempfile_path, serializer=safe_pickle_serializer(secret_key=new_key))
    assert session.cache.responses['new_key_record'] == new_record
    session = CachedSession(tempfile_path, serializer=safe_pickle_serializer(secret_key=old_key))
    assert session.cache.responses['new_key_record'] is None

    # A failed signature check is equivalent to a cache miss
    session = CachedSession(
        tempfile_path,
        serializer=safe_pickle_serializer(secret_key=new_key, old_secret_keys=[old_key]),
    )
    session.cache.responses['new_key_record'] = new_record
    session.close()
    session = CachedSession(
        tempfile_path, serializer=safe_pickle_serializer(secret_key='an unknown key')
    )
    assert session.cache.responses['new_key_record'] is None
    assert session.cache.get_response('new_key_record') is None


def test_custom_serializer(tempfile_path):
    serializer = SerializerPipeline(
        [
            json_serializer,  # Serialize to a JSON string
            utf8_encoder,  # Encode to bytes
            Stage(dumps=gzip.compress, loads=gzip.decompress),  # Compress
        ]
    )
    session = CachedSession(tempfile_path, serializer=serializer)
    response = CachedResponse()
    session.cache.responses['key'] = response
    assert session.cache.responses['key'] == response


def test_plain_pickle(tempfile_path):
    """`requests.Response` modifies pickling behavior. If plain `pickle` is used as a serializer,
    serializing `CachedResponse` should still work as expected.
    """
    session = CachedSession(tempfile_path, serializer=pickle)

    response = CachedResponse()
    session.cache.responses['key'] = response
    assert session.cache.responses['key'] == response
    assert session.cache.responses['key'].expires is None


def test_cattrs_compat():
    """CattrStage should be compatible with BaseConverter, which doesn't support the omit_if_default
    keyword arg.
    """
    stage_1 = CattrStage()
    assert isinstance(stage_1.converter, GenConverter)

    stage_2 = CattrStage(factory=BaseConverter)
    assert isinstance(stage_2.converter, BaseConverter)


def test_copy():
    stage_1 = CattrStage()
    stage_2 = Stage(MagicMock())
    serializer_1 = SerializerPipeline([stage_1, stage_2], name='test_serializer', is_binary=True)
    serializer_2 = serializer_1.copy()
    serializer_1.set_decode_content(True)

    assert serializer_1.name == serializer_2.name
    assert serializer_1.is_binary == serializer_2.is_binary
    for stage in serializer_1.stages:
        print(stage)
    for stage in serializer_2.stages:
        print(stage)
    assert serializer_1.stages[0].decode_content is True
    assert serializer_2.stages[0].decode_content is False
