"""Which errors from a model provider a search may fall back on.

A provider that could not answer right now (a dropped connection, a timeout,
a rate limit, an overload, a server error) is transient: the SDKs retry these
first, and only then does the search give up and use NoriDFS. Anything else,
such as a bad API key, a missing permission, a malformed request, a missing
SDK or a bug in a custom agent, is a configuration error and must surface
rather than quietly turn every prediction into a NoriDFS one.
"""

from __future__ import annotations

#: SDK exception classes, by name, that mean the provider could not answer.
_TRANSIENT_PROVIDER_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "OverloadedError",
        "RateLimitError",
        "ServiceUnavailableError",
    }
)
#: The SDKs those classes come from.
_PROVIDER_SDK_PACKAGES = frozenset({"anthropic", "openai"})
#: HTTP statuses below 500 for which a later retry could succeed.
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 429})


def is_transient_provider_error(error: BaseException) -> bool:
    """Return whether ``error`` means the model provider could not answer now.

    It recognizes the Anthropic and OpenAI SDKs' connection, timeout, rate
    limit, overload and server errors by class name, without importing either
    SDK, and any error carrying an HTTP ``status_code`` of 408, 409, 429 or 5xx.
    """
    for error_class in type(error).__mro__:
        package = error_class.__module__.split(".", 1)[0]
        if (
            package in _PROVIDER_SDK_PACKAGES
            and error_class.__name__ in _TRANSIENT_PROVIDER_ERROR_NAMES
        ):
            return True
    status_code = getattr(error, "status_code", None)
    return isinstance(status_code, int) and (
        status_code in _TRANSIENT_STATUS_CODES or status_code >= 500
    )


__all__ = ["is_transient_provider_error"]
