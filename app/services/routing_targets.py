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


def classify_upstream_error(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        if exc.status_code == 429:
            _app_log.debug("[classify_upstream_error] category=http_429 exc_type=%s status=%d", type(exc).__name__, exc.status_code)
            return "http_429"
        if 500 <= exc.status_code <= 599:
            _app_log.debug("[classify_upstream_error] category=http_5xx exc_type=%s status=%d", type(exc).__name__, exc.status_code)
            return "http_5xx"
        if 400 <= exc.status_code <= 499:
            _app_log.debug("[classify_upstream_error] category=http_4xx exc_type=%s status=%d", type(exc).__name__, exc.status_code)
            return "http_4xx"
        _app_log.debug("[classify_upstream_error] category=unknown exc_type=%s status=%d", type(exc).__name__, exc.status_code)
        return "unknown"
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        _app_log.debug("[classify_upstream_error] category=timeout exc_type=%s", type(exc).__name__)
        return "timeout"
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, ConnectionError)):
        _app_log.debug("[classify_upstream_error] category=connection_error exc_type=%s", type(exc).__name__)
        return "connection_error"
    status_code = getattr(exc, "status_code", None)
    if status_code is not None and isinstance(status_code, int):
        if status_code == 429:
            _app_log.debug("[classify_upstream_error] category=http_429 exc_type=%s status=%d", type(exc).__name__, status_code)
            return "http_429"
        if 500 <= status_code <= 599:
            _app_log.debug("[classify_upstream_error] category=http_5xx exc_type=%s status=%d", type(exc).__name__, status_code)
            return "http_5xx"
        if 400 <= status_code <= 499:
            _app_log.debug("[classify_upstream_error] category=http_4xx exc_type=%s status=%d", type(exc).__name__, status_code)
            return "http_4xx"
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        _app_log.debug("[classify_upstream_error] category=timeout text_match exc_type=%s", type(exc).__name__)
        return "timeout"
    if "429" in text or "rate limit" in text:
        _app_log.debug("[classify_upstream_error] category=http_429 text_match exc_type=%s", type(exc).__name__)
        return "http_429"
    if any(code in text for code in (" 500", " 502", " 503", " 504", "http 500", "http 502", "http 503", "http 504")):
        _app_log.debug("[classify_upstream_error] category=http_5xx text_match exc_type=%s", type(exc).__name__)
        return "http_5xx"
    _app_log.debug("[classify_upstream_error] category=connection_error fallback exc_type=%s", type(exc).__name__)
    return "connection_error"
