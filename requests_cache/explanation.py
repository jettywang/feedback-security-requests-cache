"""Redaction-safe explanations of caching decisions.

These helpers run the same decision path as :class:`.CachedSession` (see
:class:`.CacheActions`) but return a fixed vocabulary of stable reason and outcome
codes instead of live request data. URLs, header values, cache keys, and the inputs
passed to ``filter_fn`` are never included in the output, so an explanation can be
copied directly into a support ticket.
"""

from typing import Any, Dict, List, Optional

from attrs import asdict, define, field
from requests import PreparedRequest, Request, Response

from .policy import (
    DO_NOT_CACHE,
    EXPIRE_IMMEDIATELY,
    CacheActions,
    CacheDirectives,
    CacheSettings,
    set_request_headers,
)
from .policy.actions import _is_method_allowed
from .session import CacheMixin

__all__ = [
    'CacheDecision',
    'Outcome',
    'Reason',
    'StorageDecision',
    'explain_prepared_request',
    'explain_request',
    'explain_response_storage',
]


class Outcome:
    """Stable codes describing what :class:`.CachedSession` will do with a request."""

    CACHE_HIT = 'cache_hit'
    """A cached response satisfies the request; the origin server is not contacted."""

    CONDITIONAL_REQUEST = 'conditional_request'
    """The origin is contacted with validation headers (e.g. ``If-None-Match``);
    the server may answer with a fresh response or ``304 Not Modified``."""

    ORIGIN_REQUEST = 'origin_request'
    """A plain request is sent to the origin because nothing reusable is available."""

    BACKGROUND_REFRESH = 'background_refresh'
    """A stale cached response is returned now while a refresh request is sent."""

    NOT_CACHED_504 = 'not_cached_504'
    """``only-if-cached`` was requested but no usable cached response exists."""


class Reason:
    """Stable codes explaining *why* an outcome or storage decision was reached.

    The codes intentionally describe policy categories only; they never carry request
    URLs, header values, or the response object inspected by ``filter_fn``.
    """

    # Read-side / policy reasons
    CACHE_DISABLED = 'cache_disabled'
    METHOD_NOT_ALLOWED = 'method_not_allowed'
    REQUEST_NO_CACHE = 'request_no_cache'
    DO_NOT_CACHE = 'do_not_cache'
    CACHE_MISS = 'cache_miss'
    VARY_MISMATCH = 'vary_mismatch'
    FRESH_HIT = 'fresh_hit'
    STALE_ACCEPTED = 'stale_accepted'
    STALE_WHILE_REVALIDATE = 'stale_while_revalidate'
    CONDITIONAL_REVALIDATE = 'conditional_revalidate'
    REVALIDATE = 'revalidate'
    ONLY_IF_CACHED_MISS = 'only_if_cached_miss'

    # Write-side / storage reasons
    READ_ONLY = 'read_only'
    STATUS_NOT_ALLOWED = 'status_not_allowed'
    FILTERED = 'filtered'
    RESPONSE_NO_STORE = 'response_no_store'
    EXPIRED_NO_VALIDATOR = 'expired_no_validator'


# Cache-specific keyword arguments understood by CachedSession.request()/send()
_REQUEST_KWARGS = ('expire_after', 'only_if_cached', 'refresh', 'force_refresh')


@define
class CacheDecision:
    """The explained decision for a single request, containing only stable codes.

    Args:
        outcome: One of the :class:`Outcome` codes
        reason: Primary reason code from :class:`Reason`
        cache_hit: Whether the response returned to the caller comes from the cache
        revalidates: Whether the outgoing request carries conditional validation headers
        skip_read: Whether reading from the cache was bypassed by policy
        skip_write: Whether a request-time policy prevents storing the eventual response
        stale: Whether the reusable cached response is already expired, if applicable
        validation_headers: Names (never values) of validation headers that will be sent
        write_reasons: Storage reason codes that can be determined without a response
    """

    outcome: str
    reason: str
    cache_hit: bool
    revalidates: bool
    skip_read: bool
    skip_write: bool
    stale: Optional[bool] = field(default=None)
    validation_headers: List[str] = field(factory=list)
    write_reasons: List[str] = field(factory=list)

    @property
    def would_store(self) -> bool:
        """Whether no request-time policy blocks storing the eventual response.

        Response-dependent checks (status code, ``filter_fn``, response headers) are
        evaluated separately via :func:`explain_response_storage`.
        """
        return not self.skip_write

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable view with no request-specific data."""
        return asdict(self)


@define
class StorageDecision:
    """The explained write decision for a received response, containing only stable codes.

    Args:
        would_store: Whether the response will be saved to the cache
        reasons: Storage reason codes from :class:`Reason`; empty when it will be stored
    """

    would_store: bool
    reasons: List[str] = field(factory=list)

    @property
    def skip_write(self) -> bool:
        return not self.would_store

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def explain_request(
    session: CacheMixin, method: str, url: str, **kwargs
) -> CacheDecision:
    """Explain the cache decision for a request described the same way as
    :meth:`.CachedSession.request`.

    This prepares the request through the session (so session-level headers, cookies,
    auth and parameter normalization match a real request) without sending it or
    modifying the cache.

    Args:
        session: The cached session whose settings and stored responses are evaluated
        method: HTTP method
        url: Request URL; only used internally for key creation, never included in output
        kwargs: Any request or cache keyword arguments accepted by
            :meth:`.CachedSession.request` (``headers``, ``expire_after``,
            ``only_if_cached``, ``refresh``, ``force_refresh``, etc.)
    """
    _require_cached_session(session)
    cache_kwargs = {name: kwargs.pop(name, None) for name in _REQUEST_KWARGS}
    headers = set_request_headers(
        kwargs.pop('headers', None),
        cache_kwargs['expire_after'],
        cache_kwargs['only_if_cached'] or False,
        cache_kwargs['refresh'] or False,
        cache_kwargs['force_refresh'] or False,
    )
    request = session.prepare_request(
        Request(method=method, url=url, headers=headers, **kwargs)
    )
    return explain_prepared_request(session, request)


def explain_prepared_request(
    session: CacheMixin, request: PreparedRequest
) -> CacheDecision:
    """Explain the cache decision for an already-prepared request.

    Mirrors the read path of :meth:`.CachedSession.send` exactly, including the
    secondary ``Vary`` lookup, but performs only cache reads.
    """
    _require_cached_session(session)
    settings = session.settings
    actions = CacheActions.from_request(
        session.cache.create_key(request), request, settings
    )

    cached_response = None
    if not actions.skip_read:
        cached_response = session.cache.get_response(actions.cache_key)
    actions.update_from_cached_response(cached_response, session.cache.create_key)

    # Secondary Vary lookup, mirroring CachedSession.send()
    if actions.vary_cache_key:
        vary_cached = session.cache.get_response(actions.vary_cache_key)
        actions.cache_key = actions.vary_cache_key
        if vary_cached is not None:
            actions.send_request = False
            actions.resend_request = False
            actions.resend_async = False
            actions.error_504 = False
            actions.vary_cache_key = None
            actions._validation_headers = {}
            actions.update_from_cached_response(vary_cached, session.cache.create_key)
            cached_response = vary_cached

    return _build_decision(actions, cached_response, settings)


def explain_response_storage(session: CacheMixin, response: Response) -> StorageDecision:
    """Explain whether a received response would be written to the cache.

    Runs the same write criteria as :meth:`.CacheActions.update_from_response`. The
    session's ``filter_fn`` is invoked as usual, but only the boolean outcome is
    reported; the response passed to it is never echoed.
    """
    _require_cached_session(session)
    if response.request is None:
        raise ValueError('response must be attached to a prepared request')

    settings = session.settings
    actions = CacheActions.from_request(
        session.cache.create_key(response.request), response.request, settings
    )
    actions.update_from_response(response)
    return StorageDecision(
        would_store=not actions.skip_write,
        reasons=_response_write_reasons(actions, response, settings),
    )


def _require_cached_session(session: CacheMixin):
    if not isinstance(session, CacheMixin):
        raise TypeError('explanation helpers require a CachedSession instance')


def _build_decision(
    actions: CacheActions,
    cached_response: Optional['object'],
    settings: CacheSettings,
) -> CacheDecision:
    write_reasons = _request_time_write_reasons(actions, settings)
    validation_headers = sorted(actions._validation_headers)
    revalidates = bool(validation_headers)
    stale = cached_response.is_expired if cached_response is not None else None
    read_skip_reason = _read_skip_reason(actions, settings)

    if actions.error_504:
        outcome, reason, cache_hit = (
            Outcome.NOT_CACHED_504,
            Reason.ONLY_IF_CACHED_MISS,
            False,
        )
        if Reason.ONLY_IF_CACHED_MISS not in write_reasons:
            write_reasons = [Reason.ONLY_IF_CACHED_MISS, *write_reasons]
    elif actions.resend_async:
        outcome, reason, cache_hit = (
            Outcome.BACKGROUND_REFRESH,
            Reason.STALE_WHILE_REVALIDATE,
            True,
        )
    elif actions.resend_request:
        if revalidates:
            outcome, reason, cache_hit = (
                Outcome.CONDITIONAL_REQUEST,
                Reason.CONDITIONAL_REVALIDATE,
                False,
            )
        else:
            outcome, reason, cache_hit = (
                Outcome.ORIGIN_REQUEST,
                Reason.REVALIDATE,
                False,
            )
    elif actions.send_request:
        if read_skip_reason is not None:
            reason = read_skip_reason
        elif not actions._vary_matched:
            reason = Reason.VARY_MISMATCH
        else:
            reason = Reason.CACHE_MISS
        outcome, cache_hit = Outcome.ORIGIN_REQUEST, False
    else:
        outcome = Outcome.CACHE_HIT
        reason = Reason.STALE_ACCEPTED if stale else Reason.FRESH_HIT
        cache_hit = True

    return CacheDecision(
        outcome=outcome,
        reason=reason,
        cache_hit=cache_hit,
        revalidates=revalidates,
        skip_read=actions.skip_read,
        skip_write=bool(write_reasons),
        stale=stale,
        validation_headers=validation_headers,
        write_reasons=write_reasons,
    )


def _is_do_not_cache(actions: CacheActions) -> bool:
    """Mirror the do-not-cache check, which may come from a header or an expiration value."""
    return actions._directives.actual_no_cache or actions.expire_after == DO_NOT_CACHE


def _read_skip_reason(actions: CacheActions, settings: CacheSettings) -> Optional[str]:
    """Mirror the ordered read criteria in CacheActions.from_request()."""
    directives = actions._directives
    if settings.disabled:
        return Reason.CACHE_DISABLED
    if not _is_method_allowed(actions._request, settings):
        return Reason.METHOD_NOT_ALLOWED
    if directives.no_cache or directives.no_store:
        return Reason.REQUEST_NO_CACHE
    if _is_do_not_cache(actions):
        return Reason.DO_NOT_CACHE
    return None


def _request_time_write_reasons(
    actions: CacheActions, settings: CacheSettings
) -> List[str]:
    """Storage blockers that are knowable before a response is received.

    These mirror the request-side inputs of the write criteria in
    :meth:`.CacheActions.update_from_response`, which is only fully evaluable once a
    response exists.
    """
    reasons: List[str] = []
    read_skip_reason = _read_skip_reason(actions, settings)
    if read_skip_reason in (Reason.CACHE_DISABLED, Reason.METHOD_NOT_ALLOWED):
        reasons.append(read_skip_reason)
    if actions._directives.no_store:
        reasons.append(Reason.RESPONSE_NO_STORE)
    if settings.read_only:
        reasons.append(Reason.READ_ONLY)
    return reasons


def _response_write_reasons(
    actions: CacheActions, response: Response, settings: CacheSettings
) -> List[str]:
    """Translate the exact write criteria in :meth:`.CacheActions.update_from_response`
    into stable reason codes, in the same order and using the same boolean inputs.

    The generic ``disabled by headers`` criterion is decomposed so a request-side
    ``no-store`` is not mistaken for a response-side one.
    """
    directives = CacheDirectives.from_headers(response.headers)
    filtered_out = (
        settings.filter_fn is not None and not settings.filter_fn(response)
    )
    skip_stale = (
        actions.expire_after == EXPIRE_IMMEDIATELY and not directives.has_validator
    )
    do_not_cache = actions.expire_after == DO_NOT_CACHE
    header_no_store = directives.no_store or actions._directives.no_store

    criteria = [
        (settings.disabled, Reason.CACHE_DISABLED),
        (not _is_method_allowed(response.request, settings), Reason.METHOD_NOT_ALLOWED),
        (response.status_code not in settings.allowable_codes, Reason.STATUS_NOT_ALLOWED),
        (filtered_out, Reason.FILTERED),
        (header_no_store, Reason.RESPONSE_NO_STORE),
        (do_not_cache, Reason.DO_NOT_CACHE),
        (skip_stale, Reason.EXPIRED_NO_VALIDATOR),
        (settings.read_only, Reason.READ_ONLY),
    ]
    return [reason for matched, reason in criteria if matched]
