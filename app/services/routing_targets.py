from fastapi import HTTPException
import httpx

from app.core.policy import RouteTarget
from app.database import find_provider_by_model, get_provider


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
        key = (target.model, target.provider_id)
        if not target.model or key in seen:
            continue
        seen.add(key)
        targets.append(target)
    return targets


def classify_upstream_error(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        if exc.status_code == 429:
            return "http_429"
        if 500 <= exc.status_code <= 599:
            return "http_5xx"
        if 400 <= exc.status_code <= 499:
            return "http_4xx"
        return "unknown"
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return "timeout"
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, ConnectionError)):
        return "connection_error"
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "429" in text or "rate limit" in text:
        return "http_429"
    if any(code in text for code in (" 500", " 502", " 503", " 504", "http 500", "http 502", "http 503", "http 504")):
        return "http_5xx"
    return "connection_error"
