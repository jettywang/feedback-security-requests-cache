"""Failure classification for ``stale_if_error`` handling"""

from enum import Enum
from typing import Optional

from requests import HTTPError, RequestException, Response

__all__ = ['FailureKind', 'classify_failure']


class FailureKind(Enum):
    """Categories of request failures that may qualify for serving stale cached data.

    * ``TRANSPORT``: The request raised a transport-level exception (e.g., connection error
      or timeout), so no response was received
    * ``SERVER_ERROR``: The server returned a 5xx response (e.g., the server is down)
    * ``REFUSAL``: The server explicitly refused the request with a 4xx response (e.g., a
      403 for a deactivated account)
    """

    TRANSPORT = 'transport'
    SERVER_ERROR = 'server_error'
    REFUSAL = 'refusal'


def classify_failure(
    error: Optional[Exception] = None, response: Optional[Response] = None
) -> Optional[FailureKind]:
    """Classify a failed request attempt. Returns ``None`` for outcomes that don't fit a
    known failure category (e.g., an unhandled exception).
    """
    if response is None and isinstance(error, HTTPError):
        response = error.response
    if response is not None:
        if 500 <= response.status_code < 600:
            return FailureKind.SERVER_ERROR
        if 400 <= response.status_code < 500:
            return FailureKind.REFUSAL
        return None
    if isinstance(error, RequestException):
        return FailureKind.TRANSPORT
    return None
