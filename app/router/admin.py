from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from datetime import datetime, timedelta
from app.database import (
    get_providers, get_provider, add_provider, update_provider, delete_provider,
    get_users, get_user, add_user, update_user, delete_user,
    add_user_api_key, update_user_api_key, delete_user_api_key,
    get_routing_rules, get_routing_rule, add_routing_rule, update_routing_rule, delete_routing_rule,
    get_fallback_policies, get_fallback_policy, add_fallback_policy, update_fallback_policy, delete_fallback_policy,
    get_global_stats, reset_global_stats, reset_user_stats,
    get_history_stats,
    find_provider_by_model, parse_model_id,
    get_preprocessors, upsert_preprocessor, delete_preprocessor as delete_preprocessor_config,
)
from app.core.policy import apply_fallback_policy, apply_routing_rules
from app.core.text import mask_key
from app.router.auth import require_admin_session
from app.services.discovery import refresh_provider_models, refresh_all_providers, check_provider_health, check_all_provider_health
from app.services.lite_llm import get_available_models
from app.router.proxy import get_request_log, get_model_stats, clear_request_log, get_timeline_data, get_model_distribution, get_timeline_model_data
from app.services.logger import get_logger
from app.models import ProviderCreate, ProviderUpdate, StatsResponse

router = APIRouter()
_app_log = get_logger("app")


@router.get("/providers")
async def list_providers(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    return get_providers()


@router.post("/providers")
async def create_provider(provider: ProviderCreate, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    existing = get_provider(provider.id)
    if existing:
        raise HTTPException(status_code=400, detail="Provider with this ID already exists")
    return add_provider(provider.model_dump())


@router.put("/providers/{provider_id}")
async def update_provider_endpoint(provider_id: str, updates: ProviderUpdate, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    existing = get_provider(provider_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Provider not found")
    update_data = {k: v for k, v in updates.model_dump().items() if v is not None}
    updated = update_provider(provider_id, update_data)
    return updated


@router.delete("/providers/{provider_id}")
async def delete_provider_endpoint(provider_id: str, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    if not delete_provider(provider_id):
        raise HTTPException(status_code=404, detail="Provider not found")
    return {"status": "deleted", "provider_id": provider_id}


@router.post("/providers/{provider_id}/refresh")
async def refresh_provider(provider_id: str, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    if not get_provider(provider_id):
        raise HTTPException(status_code=404, detail="Provider not found")
    result = await refresh_provider_models(provider_id)
    return result


@router.post("/providers/refresh-all")
async def refresh_all(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    results = await refresh_all_providers()
    return {"results": results}


@router.get("/providers/health-all")
async def provider_health_all(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    return {"results": await check_all_provider_health()}


@router.get("/providers/{provider_id}/health")
async def provider_health(provider_id: str, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    if not get_provider(provider_id):
        raise HTTPException(status_code=404, detail="Provider not found")
    return await check_provider_health(provider_id)


@router.get("/models")
async def list_models(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    models = get_available_models()
    return {"models": models}


@router.get("/users")
async def list_users(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    return {"users": get_users()}


@router.post("/users")
async def create_user(user_info: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    username = (user_info.get("username") or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="username is required")
    if not isinstance(user_info.get("enabled", True), bool) and user_info.get("enabled") is not None:
        raise HTTPException(status_code=400, detail="enabled must be a boolean")
    try:
        return add_user(user_info)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/users/{username}")
async def update_user_endpoint(username: str, updates: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    if "enabled" in updates and not isinstance(updates["enabled"], bool):
        raise HTTPException(status_code=400, detail="enabled must be a boolean")
    user = update_user(username, updates)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.delete("/users/{username}")
async def delete_user_endpoint(username: str, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    if not delete_user(username):
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "deleted"}


@router.post("/users/{username}/api-keys")
async def add_user_api_key_endpoint(username: str, key_info: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    try:
        return add_user_api_key(
            username,
            key_info.get("name", "default"),
            key_info.get("allowed_models")
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/users/{username}/api-keys/{key}")
async def update_user_api_key_endpoint(username: str, key: str, updates: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    result = update_user_api_key(username, key, updates)
    if not result:
        raise HTTPException(status_code=404, detail="API key not found")
    return result


@router.delete("/users/{username}/api-keys/{key}")
async def delete_user_api_key_endpoint(username: str, key: str, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    if not delete_user_api_key(username, key):
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "deleted"}


@router.get("/stats")
async def get_stats(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    stats = get_global_stats()
    total = stats.get("total_calls", 0)
    failed = stats.get("failed_calls", 0)
    success_rate = ((total - failed) / total * 100) if total > 0 else 100.0

    users_summary = []
    for u in get_users():
        u_stats = u.get("stats", {})
        users_summary.append({
            "username": u.get("username", ""),
            "total_calls": u_stats.get("total_calls", 0),
            "failed_calls": u_stats.get("failed_calls", 0),
            "total_tokens": u_stats.get("total_tokens", 0),
        })

    return StatsResponse(
        total_calls=total,
        failed_calls=failed,
        success_rate=round(success_rate, 2),
        last_reset=stats.get("last_reset", ""),
        stats_by_model=get_model_stats(),
        request_log=get_request_log(),
        users=users_summary,
        timeline=get_timeline_data(),
        distribution=get_model_distribution(),
        timeline_models=get_timeline_model_data(),
    )


# -- Routing rules --

@router.get("/routing-rules")
async def list_routing_rules(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    return {"rules": get_routing_rules()}


@router.post("/routing-rules")
async def create_routing_rule(rule: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    try:
        return add_routing_rule(rule)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/routing-rules/{rule_id}")
async def update_routing_rule_endpoint(rule_id: str, updates: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    result = update_routing_rule(rule_id, updates)
    if not result:
        raise HTTPException(status_code=404, detail="Rule not found")
    return result


@router.delete("/routing-rules/{rule_id}")
async def delete_routing_rule_endpoint(rule_id: str, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    if not delete_routing_rule(rule_id):
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"status": "deleted"}


@router.post("/routing-rules/dry-run")
async def dry_run_routing_rule(body: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    requested_model = str(body.get("model") or body.get("requested_model") or "").strip()
    if not requested_model:
        raise HTTPException(status_code=400, detail="model is required")

    username = str(body.get("username") or "").strip()
    api_key_value = str(body.get("api_key") or body.get("api_key_value") or body.get("key") or "")
    resolved_model = str(body.get("resolved_model") or requested_model).strip()
    decision = apply_routing_rules(username, api_key_value, requested_model, resolved_model)

    provider_source = "target_provider" if decision.target_provider else "model_lookup"
    provider = get_provider(decision.target_provider) if decision.target_provider else find_provider_by_model(decision.target_model)
    mid = parse_model_id(requested_model)
    fallback_preview = apply_fallback_policy(
        provider.get("id", "") if provider else decision.target_provider,
        decision.target_model,
        str(body.get("fallback_trigger") or "http_5xx"),
    )

    return {
        "input": {
            "username": username,
            "api_key": mask_key(api_key_value) if api_key_value else "",
            "requested_model": requested_model,
            "resolved_model": resolved_model,
            "model_name": mid.model_name,
            "provider_id": mid.provider_id,
            "is_composite": mid.is_composite,
        },
        "routing": {
            "matched": decision.matched,
            "source": decision.source,
            "rule_id": decision.rule_id,
            "rule_name": decision.rule_name,
            "reason": decision.reason,
            "target_model": decision.target_model,
            "target_provider": decision.target_provider,
        },
        "fallback_preview": {
            "matched": fallback_preview.matched,
            "policy_id": fallback_preview.policy_id,
            "policy_name": fallback_preview.policy_name,
            "trigger": fallback_preview.trigger,
            "reason": fallback_preview.reason,
            "chain": [
                {"model": target.model, "provider_id": target.provider_id}
                for target in fallback_preview.chain
            ],
        },
        "provider": {
            "found": bool(provider),
            "source": provider_source,
            "id": provider.get("id", "") if provider else decision.target_provider,
            "name": provider.get("name", "") if provider else "",
            "provider_type": provider.get("provider_type", "") if provider else "",
            "enabled": bool(provider.get("enabled", False)) if provider else False,
        },
        "effective": {
            "model": decision.target_model,
            "provider_id": provider.get("id", "") if provider else decision.target_provider,
            "provider_type": provider.get("provider_type", "") if provider else "",
        },
    }


# -- Fallback policies --

@router.get("/fallback-policies")
async def list_fallback_policies(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    return {"policies": get_fallback_policies()}


@router.post("/fallback-policies")
async def create_fallback_policy(policy: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    try:
        return add_fallback_policy(policy)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/fallback-policies/{policy_id}")
async def get_fallback_policy_endpoint(policy_id: str, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    policy = get_fallback_policy(policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Fallback policy not found")
    return policy


@router.put("/fallback-policies/{policy_id}")
async def update_fallback_policy_endpoint(policy_id: str, updates: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    policy = update_fallback_policy(policy_id, updates)
    if not policy:
        raise HTTPException(status_code=404, detail="Fallback policy not found")
    return policy


@router.delete("/fallback-policies/{policy_id}")
async def delete_fallback_policy_endpoint(policy_id: str, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    if not delete_fallback_policy(policy_id):
        raise HTTPException(status_code=404, detail="Fallback policy not found")
    return {"status": "deleted", "policy_id": policy_id}


@router.post("/fallback-policies/dry-run")
async def dry_run_fallback_policy(body: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    model = str(body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    provider_id = str(body.get("provider_id") or body.get("provider") or "").strip()
    trigger = str(body.get("trigger") or body.get("error_type") or "http_5xx").strip()
    decision = apply_fallback_policy(provider_id, model, trigger)
    return {
        "input": {"provider_id": provider_id, "model": model, "trigger": trigger},
        "fallback": {
            "matched": decision.matched,
            "policy_id": decision.policy_id,
            "policy_name": decision.policy_name,
            "reason": decision.reason,
            "chain": [
                {"model": target.model, "provider_id": target.provider_id}
                for target in decision.chain
            ],
        },
    }


@router.get("/stats/history")
async def get_stats_history(from_ts: Optional[str] = None, to_ts: Optional[str] = None, granularity: Optional[str] = "day", authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    local_now = datetime.now().astimezone()
    if not to_ts:
        to_ts = local_now.strftime("%Y-%m-%d 23:59:59")
    else:
        to_ts = to_ts + " 23:59:59" if len(to_ts) == 10 else to_ts
    if not from_ts:
        from_ts = (local_now - timedelta(days=30)).strftime("%Y-%m-%d 00:00:00")
    else:
        from_ts = from_ts + " 00:00:00" if len(from_ts) == 10 else from_ts
    if granularity not in ("hour", "day", "week", "month"):
        granularity = "day"
    return get_history_stats(from_ts, to_ts, granularity)


@router.post("/stats/reset")
async def reset_stats(authorization: Optional[str] = Header(None)):
    username = await require_admin_session(authorization)
    reset_global_stats()
    reset_user_stats()
    clear_request_log()
    _app_log.warning("Stats reset by admin '%s'", username)
    return {"status": "ok", "message": "Stats cleared"}


# -- Preprocessor configuration --

@router.get("/preprocessors")
async def list_preprocessors(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    preprocessors = get_preprocessors()
    # Also return model preprocessor flags from DB using composite IDs to avoid same-name ambiguity
    from app.database import get_db
    models = []
    with get_db() as db:
        rows = db.execute(
            "SELECT m.provider_id, m.model_id, m.preprocessor, p.name AS provider_name "
            "FROM provider_models m JOIN providers p ON p.id = m.provider_id "
            "WHERE m.enabled = 1 ORDER BY m.provider_id, m.model_id"
        ).fetchall()
        models = [{"model_id": f"{r['provider_id']}/{r['model_id']}", "provider_id": r["provider_id"],
                    "provider_name": r["provider_name"], "preprocessor": bool(r["preprocessor"])} for r in rows]
    return {"preprocessors": preprocessors, "models": models}


@router.put("/preprocessors/{preprocessor_id}")
async def update_preprocessor(preprocessor_id: str, config: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    try:
        current = upsert_preprocessor(preprocessor_id, config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    current.pop("id", None)
    return {"id": preprocessor_id, "config": current}


@router.delete("/preprocessors/{preprocessor_id}")
async def delete_preprocessor(preprocessor_id: str, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    if delete_preprocessor_config(preprocessor_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Preprocessor not found")


@router.get("/preprocessors/fetch-models")
async def fetch_preprocessor_models(api_base: str, api_key: str = "",
                                     authorization: Optional[str] = Header(None)):
    """Fetch available models from a vision model server."""
    await require_admin_session(authorization)
    from app.services.discovery import model_list_urls, auth_headers
    import httpx
    urls = model_list_urls(api_base, "openai")
    headers_list = auth_headers(api_key, "openai")
    async with httpx.AsyncClient(timeout=10) as client:
        for url in urls:
            for h in headers_list:
                try:
                    resp = await client.get(url, headers=h)
                    resp.raise_for_status()
                    data = resp.json()
                    models = data.get("data") or data.get("models") or []
                    return {"models": [m.get("id") or m.get("name", "?") for m in models if isinstance(m, dict)]}
                except Exception:
                    continue  # skip model that failed to fetch
    raise HTTPException(status_code=502, detail="Failed to fetch models from server")


@router.put("/models/preprocessor")
async def toggle_model_preprocessor(body: dict, authorization: Optional[str] = Header(None)):
    """Toggle preprocessor on/off for a model. body: {"model_id": "provider/model", "enabled": true/false}"""
    await require_admin_session(authorization)
    from app.database import get_db, parse_model_id
    model_id = body.get("model_id", "")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    enabled = body.get("enabled", False)
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="enabled must be a boolean")
    value = "1" if enabled else ""
    # Parse composite ID in provider/model format
    mid = parse_model_id(model_id)
    with get_db() as db:
        if mid.is_composite:
            db.execute(
                "UPDATE provider_models SET preprocessor = ? WHERE provider_id = ? AND model_id = ?",
                (value, mid.provider_id, mid.model_name)
            )
        else:
            db.execute(
                "UPDATE provider_models SET preprocessor = ? WHERE model_id = ?",
                (value, mid.model_name)
            )
    return {"model_id": model_id, "preprocessor": enabled}
