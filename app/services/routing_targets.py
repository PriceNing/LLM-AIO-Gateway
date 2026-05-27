from fastapi import HTTPException

from app.core.policy import RouteTarget
from app.database import find_provider_by_model, get_provider


def adapter_provider_id(provider_info: dict | None, provider_id: str = "") -> str:
    return (provider_info or {}).get("id") or provider_id or ""


def provider_for_log(provider_info: dict | None, provider_id: str = "") -> str:
    return adapter_provider_id(provider_info, provider_id)


def resolve_provider(model: str, provider_id: str = "") -> dict | None:
    return get_provider(provider_id) if provider_id else find_provider_by_model(model)


def candidate_targets(policy, model: str, provider_id: str = "") -> list[RouteTarget]:
    seen = set()
    targets = []
    for target in [RouteTarget(model=model, provider_id=provider_id), *policy.routing.fallbacks]:
        key = (target.model, target.provider_id)
        if not target.model or key in seen:
            continue
        seen.add(key)
        targets.append(target)
    return targets


def is_retryable_endpoint_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPException):
        return exc.status_code >= 500
    return True
