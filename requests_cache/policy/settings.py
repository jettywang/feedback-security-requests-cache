from typing import Dict, Iterable, Optional, Union

from attrs import define, field

from .._utils import get_valid_kwargs
from ..models import CachedRequest, RichMixin
from . import ExpirationPattern, ExpirationTime, FilterCallback, KeyCallback

ALL_METHODS = ('GET', 'HEAD', 'OPTIONS', 'POST', 'PUT', 'PATCH', 'DELETE')
DEFAULT_CACHE_NAME = 'http_cache'
DEFAULT_METHODS = ('GET', 'HEAD')
DEFAULT_STATUS_CODES = (200,)

# Fields of CachedRequest that can be selected for serialization via serializable_request_fields
REQUEST_FIELDS = tuple(a.name for a in CachedRequest.__attrs_attrs__)

# Common auth headers and params that are excluded from cache keys and redacted from cached responses
AUTH_HEADERS = (
    'Authorization',
    'Proxy-Authorization',
    'X-API-Key',
    'X-Auth-Token',
    'X-API-Token',
    'X-Access-Token',
)
AUTH_PARAMS = ('access_token', 'api_key', 'apikey')
DEFAULT_IGNORED_PARAMS = AUTH_HEADERS + AUTH_PARAMS


@define(repr=False)
class CacheSettings(RichMixin):
    """Class used internally to store settings that affect caching behavior. This allows settings
    to be used across multiple modules, but exposed to the user in a single property
    (:py:attr:`.CachedSession.settings`). These values can safely be modified after initialization.
    See :py:class:`.CachedSession` and :ref:`user-guide` for usage details.
    """

    allowable_codes: Iterable[int] = field(default=DEFAULT_STATUS_CODES)
    allowable_methods: Iterable[str] = field(default=DEFAULT_METHODS)
    always_revalidate: bool = field(default=False)
    autoclose: bool = field(default=True)
    cache_control: bool = field(default=False)
    content_root_key: Optional[str] = field(default=None)
    disabled: bool = field(default=False)
    expire_after: ExpirationTime = field(default=None)
    filter_fn: FilterCallback = field(default=None)
    ignored_parameters: Iterable[str] = field(default=DEFAULT_IGNORED_PARAMS)
    key_fn: KeyCallback = field(default=None)
    match_headers: Union[Iterable[str], bool] = field(default=False)
    only_if_cached: bool = field(default=False)
    read_only: bool = field(default=False)
    serializable_request_fields: Optional[Iterable[str]] = field(default=None)
    stale_if_error: Union[bool, ExpirationTime] = field(default=False)
    stale_while_revalidate: Union[bool, ExpirationTime] = field(default=False)
    urls_expire_after: Dict[ExpirationPattern, ExpirationTime] = field(factory=dict)

    def __attrs_post_init__(self):
        if self.serializable_request_fields is None:
            return
        invalid = set(self.serializable_request_fields) - set(REQUEST_FIELDS)
        if invalid:
            raise ValueError(
                f'Invalid serializable_request_fields: {sorted(invalid)}. '
                f'Valid fields are: {list(REQUEST_FIELDS)}'
            )
        self.serializable_request_fields = tuple(self.serializable_request_fields)

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Constructor with some additional steps:

        * Handle some deprecated argument names
        * Ignore invalid settings, for easier initialization from mixed ``**kwargs``
        """
        kwargs = cls._rename_kwargs(kwargs)
        kwargs = get_valid_kwargs(cls.__init__, kwargs)
        return cls(**kwargs)

    @staticmethod
    def _rename_kwargs(kwargs):
        if 'old_data_on_error' in kwargs:
            kwargs['stale_if_error'] = kwargs.pop('old_data_on_error')
        if 'include_get_headers' in kwargs:
            kwargs['match_headers'] = kwargs.pop('include_get_headers')
        return kwargs
