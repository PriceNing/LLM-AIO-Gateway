from fastapi import HTTPException
import httpx

from app.services.logger import get_logger

from app.core.policy import RouteTarget
from app.database import find_provider_by_model, get_provider

_app_log = get_logger("app")


def adapter_provider_id(provider_info: dict | None, provider_id: str = "") -> str:
    return (provider_info or {}).get("id") or provider_id or ""


def provider_for_log(provider_info: dict | None, provider_id: str = "") -> str:
    return adapter_provider_id(provider_info, provider_id)


def resolve_provider(model: str, provider_id: str = "") -> dict | None:
    return get_provider(provider_id) if provider_id else find_provider_by_model(model)


def candidate_targets(primary: RouteTarget, fallback_chain: list[RouteTarget] | None = None) -> list[RouteTarget]:
    seen = set()
    targets = []
    for target in [primary, *(fallback_chain or [])]:
        _app_log.debug("[candidate_targets] model=%s provider=%s", target.model, target.provider_id)
        key = (target.model, target.provider_id)
        if not target.model or key in seen:
            continue
        seen.add(key)
        targets.append(target)
    return targets


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def classify_upstream_error(exc: Exception) -> str:
    chain = list(_exception_chain(exc))

    # Prefer timeout/connection signals before HTTP status wrapping.
    # Anthropic/liteLLM adapters often wrap httpx.TimeoutException as HTTP 502,
    # which would otherwise hide the timeout trigger from fallback/retry.
    for item in chain:
        if isinstance(item, (TimeoutError, httpx.TimeoutException)):
            _app_log.debug("[classify_upstream_error] category=timeout exc_type=%s", type(item).__name__)
            return "timeout"
        if not isinstance(item, HTTPException):
            text = str(item).lower()
            exc_name = type(item).__name__.lower()
            if "timeout" in text or "timed out" in text or "timeout" in exc_name:
                _app_log.debug("[classify_upstream_error] category=timeout text_match exc_type=%s", type(item).__name__)
                return "timeout"

    for item in chain:
        if isinstance(item, (httpx.ConnectError, httpx.NetworkError, ConnectionError)):
            _app_log.debug("[classify_upstream_error] category=connection_error exc_type=%s", type(item).__name__)
            return "connection_error"

    for item in chain:
        if isinstance(item, HTTPException):
            if item.status_code == 429:
                _app_log.debug("[classify_upstream_error] category=http_429 exc_type=%s status=%d", type(item).__name__, item.status_code)
                return "http_429"
            if 500 <= item.status_code <= 599:
                _app_log.debug("[classify_upstream_error] category=http_5xx exc_type=%s status=%d", type(item).__name__, item.status_code)
                return "http_5xx"
            if 400 <= item.status_code <= 499:
                _app_log.debug("[classify_upstream_error] category=http_4xx exc_type=%s status=%d", type(item).__name__, item.status_code)
                return "http_4xx"
            _app_log.debug("[classify_upstream_error] category=unknown exc_type=%s status=%d", type(item).__name__, item.status_code)
            return "unknown"

        # httpx.HTTPStatusError keeps the real status on ``response`` rather than
        # on the exception itself.  Read it before falling back to text matching so
        # a 502 is not misleadingly reported as a connection failure.
        status_code = getattr(item, "status_code", None)
        if status_code is None:
            status_code = getattr(getattr(item, "response", None), "status_code", None)
        if status_code is not None and isinstance(status_code, int):
            if status_code == 429:
                _app_log.debug("[classify_upstream_error] category=http_429 exc_type=%s status=%d", type(item).__name__, status_code)
                return "http_429"
            if 500 <= status_code <= 599:
                _app_log.debug("[classify_upstream_error] category=http_5xx exc_type=%s status=%d", type(item).__name__, status_code)
                return "http_5xx"
            if 400 <= status_code <= 499:
                _app_log.debug("[classify_upstream_error] category=http_4xx exc_type=%s status=%d", type(item).__name__, status_code)
                return "http_4xx"

    text = str(exc).lower()
    if "429" in text or "rate limit" in text:
        _app_log.debug("[classify_upstream_error] category=http_429 text_match exc_type=%s", type(exc).__name__)
        return "http_429"
    if any(code in text for code in (" 500", " 502", " 503", " 504", "http 500", "http 502", "http 503", "http 504")):
        _app_log.debug("[classify_upstream_error] category=http_5xx text_match exc_type=%s", type(exc).__name__)
        return "http_5xx"
    _app_log.debug("[classify_upstream_error] category=connection_error fallback exc_type=%s", type(exc).__name__)
    return "connection_error"


_SAME_TARGET_RETRY_TRIGGERS = frozenset({"timeout", "http_5xx", "http_429"})
_CONNECTION_RETRY_TOKENS = (
    "connection",
    "connect",
    "tls",
    "ssl",
    "eof",
    "reset",
    "refused",
    "unreachable",
    "无法连接",
)


def is_same_target_retryable(exc: Exception, trigger: str | None = None) -> bool:
    """Return whether an unused first-byte failure should retry the same target once.

    Unknown RuntimeError values currently classify as connection_error; those must
    not be retried, or every generic fallback test would double-hit the primary.
    Gateway-owned attempt_timeout is also excluded: that budget already decided
    to leave the current attempt, so retrying it would only delay fallback.
    """
    if "fallback attempt timeout" in str(exc).lower():
        return False
    category = trigger or classify_upstream_error(exc)
    if category in _SAME_TARGET_RETRY_TRIGGERS:
        return True
    if category != "connection_error":
        return False
    for item in _exception_chain(exc):
        if isinstance(item, (httpx.ConnectError, httpx.NetworkError, ConnectionError)):
            return True
        text = str(item).lower()
        if any(token in text for token in _CONNECTION_RETRY_TOKENS):
            return True
    return False
