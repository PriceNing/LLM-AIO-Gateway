import time
import json

import anyio
from fastapi import BackgroundTasks
import httpx
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from datetime import datetime, timedelta, timezone
from app.database import (
    get_providers, get_provider, add_provider, update_provider, delete_provider, delete_provider_model,
    get_users, get_user, add_user, update_user, delete_user,
    add_user_api_key, update_user_api_key, delete_user_api_key,
    get_routing_rules, get_routing_rule, add_routing_rule, update_routing_rule, delete_routing_rule,
    get_fallback_policies, get_fallback_policy, add_fallback_policy, update_fallback_policy, delete_fallback_policy,
    get_global_stats, reset_global_stats, reset_user_stats,
    get_history_stats,
    find_provider_by_model, parse_model_id,
    get_preprocessors, upsert_preprocessor, delete_preprocessor as delete_preprocessor_config,
    get_image_generators, upsert_image_generator, delete_image_generator as delete_image_generator_config,
    set_model_image_generation,
)
from app.core.policy import apply_fallback_policy, apply_routing_rules
from app.adapters.anthropic import anthropic_messages_completion_for_internal
from app.adapters.openai import chat_kwargs_from_internal, chat_messages_from_internal
from app.adapters.output import response_to_internal_output
from app.core.text import mask_key
from app.core.types import InternalMessage, InternalRequest, text_part
from app.router.auth import require_admin_session
from app.services.discovery import refresh_provider_models, refresh_all_providers, check_provider_health, check_all_provider_health
from app.services.lite_llm import create_chat_completion, get_available_models
from app.services.routing_targets import provider_for_log, resolve_provider
from app.router.proxy import (
    get_request_log, get_model_stats, clear_request_log,
    get_timeline_data, get_model_distribution, get_timeline_model_data,
)
from app.database import (
    list_request_logs, count_request_logs, get_request_log as db_get_request_log,
    delete_request_log as db_delete_request_log, clear_request_logs,
)
from app.services.logger import available_log_channels, get_logger, list_log_dates, read_log_entries
from app.models import ProviderCreate, ProviderUpdate, StatsResponse

router = APIRouter()
_app_log = get_logger("app")


@router.get("/providers")
async def list_providers(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    return get_providers()


@router.post("/providers")
async def create_provider(provider: ProviderCreate, background_tasks: BackgroundTasks, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    existing = get_provider(provider.id)
    if existing:
        raise HTTPException(status_code=400, detail="Provider with this ID already exists")
    created = add_provider(provider.model_dump())
    return created


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


@router.delete("/providers/{provider_id}/models/{model_id}")
async def delete_provider_model_endpoint(provider_id: str, model_id: str, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    if not get_provider(provider_id):
        raise HTTPException(status_code=404, detail="Provider not found")
    if not delete_provider_model(provider_id, model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deleted", "provider_id": provider_id, "model_id": model_id}


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


def _test_latency_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def _test_preview(output) -> str:
    preview = (getattr(output, "text", "") or getattr(output, "reasoning", "") or "").strip()
    return preview[:500]


def _model_test_request(model_id: str, provider_id: str) -> InternalRequest:
    mid = parse_model_id(model_id)
    return InternalRequest(
        endpoint="chat_completions",
        requested_model=mid.composite,
        target_model=mid.model_name,
        provider_id=provider_id,
        messages=[InternalMessage(role="user", parts=[text_part("请只回复 OK，用于连通性测试。")])],
        stream=False,
        temperature=0,
        max_tokens=16,
        raw_body={"model": mid.composite, "messages": []},
    )


async def _run_model_test(model_id: str) -> dict:
    mid = parse_model_id(model_id)
    if not mid.model_name:
        raise HTTPException(status_code=400, detail="model_id is required")
    provider_info = resolve_provider(mid.model_name, mid.provider_id)
    if not provider_info:
        raise HTTPException(status_code=404, detail="Model provider not found")

    provider_id = provider_for_log(provider_info, mid.provider_id)
    internal = _model_test_request(mid.composite, provider_id)
    provider_type = provider_info.get("provider_type", "openai")
    with anyio.fail_after(20):
        if provider_type == "anthropic":
            output = await anthropic_messages_completion_for_internal(provider_info, internal)
        else:
            response = await anyio.to_thread.run_sync(
                lambda: create_chat_completion(
                    model=internal.target_model,
                    messages=chat_messages_from_internal(internal),
                    provider_id=provider_id,
                    temperature=internal.temperature,
                    max_tokens=internal.max_tokens,
                    **chat_kwargs_from_internal(internal),
                ),
                abandon_on_cancel=True,
            )
            output = response_to_internal_output(response)
    return {
        "status": "ok",
        "provider_id": provider_id,
        "provider_type": provider_type,
        "model": internal.target_model,
        "preview": _test_preview(output),
        "usage": getattr(output, "usage", {}) or {},
    }


@router.post("/models/test")
async def test_model(body: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    model_id = str(body.get("model_id") or "").strip()
    start_time = time.perf_counter()
    try:
        result = await _run_model_test(model_id)
        result["latency_ms"] = _test_latency_ms(start_time)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        _app_log.warning("Model test failed for %s: %s", model_id, exc)
        return {
            "status": "fail",
            "model": model_id,
            "latency_ms": _test_latency_ms(start_time),
            "error": str(exc)[:500],
        }


_PREPROCESSOR_TEST_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _preprocessor_test_url(api_base: str) -> str:
    base = (api_base or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


async def _run_preprocessor_test(preprocessor_id: str) -> dict:
    preprocessors = get_preprocessors()
    preprocessor = preprocessors.get(preprocessor_id)
    if not preprocessor_id:
        raise HTTPException(status_code=400, detail="preprocessor_id is required")
    if not preprocessor:
        raise HTTPException(status_code=404, detail="Preprocessor not found")
    api_base = preprocessor.get("api_base") or ""
    model = preprocessor.get("model") or ""
    if not api_base or not model:
        raise HTTPException(status_code=400, detail="Preprocessor api_base and model are required")

    timeout = min(max(int(preprocessor.get("timeout") or 20), 1), 20)
    headers = {"Content-Type": "application/json"}
    api_key = preprocessor.get("api_key") or ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "这是连通性测试图片，请简短回复 OK 或描述图片。"},
                {"type": "image_url", "image_url": {"url": _PREPROCESSOR_TEST_IMAGE}},
            ],
        }],
        "max_tokens": min(int(preprocessor.get("max_tokens") or 128), 128),
        "temperature": 0,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(_preprocessor_test_url(api_base), headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError((resp.text or f"HTTP {resp.status_code}")[:500])
    data = resp.json()
    message = ((data.get("choices") or [{}])[0].get("message") or {})
    preview = str(message.get("content") or data.get("content") or "").strip()[:500]
    return {
        "status": "ok",
        "preprocessor_id": preprocessor_id,
        "model": model,
        "preview": preview,
        "usage": data.get("usage") or {},
    }


@router.post("/preprocessors/test")
async def test_preprocessor(body: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    preprocessor_id = str(body.get("preprocessor_id") or "").strip()
    start_time = time.perf_counter()
    try:
        result = await _run_preprocessor_test(preprocessor_id)
        result["latency_ms"] = _test_latency_ms(start_time)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        _app_log.warning("Preprocessor test failed for %s: %s", preprocessor_id, exc)
        return {
            "status": "fail",
            "preprocessor_id": preprocessor_id,
            "latency_ms": _test_latency_ms(start_time),
            "error": str(exc)[:500],
        }


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
    total = int(stats.get("total_calls", 0) or 0)
    failed = int(stats.get("failed_calls", 0) or 0)
    degraded = int(stats.get("degraded_calls", 0) or 0)
    rejected = int(stats.get("rejected_calls", 0) or 0)
    cancelled = int(stats.get("cancelled_calls", 0) or 0)
    stateful_fallback_blocked = int(stats.get("stateful_fallback_blocked_calls", 0) or 0)
    image_generation_calls = int(stats.get("image_generation_calls", 0) or 0)
    image_generation_failed_calls = int(stats.get("image_generation_failed_calls", 0) or 0)
    image_generation_images = int(stats.get("image_generation_images", 0) or 0)
    image_generation_bytes = int(stats.get("image_generation_bytes", 0) or 0)
    success_rate = ((total - failed) / total * 100) if total > 0 else 100.0
    # Health rate treats fallback-recovered calls as unhealthy for ops visibility.
    health_rate = ((total - failed - degraded) / total * 100) if total > 0 else 100.0

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
        degraded_calls=degraded,
        rejected_calls=rejected,
        cancelled_calls=cancelled,
        stateful_fallback_blocked_calls=stateful_fallback_blocked,
        image_generation_calls=image_generation_calls,
        image_generation_failed_calls=image_generation_failed_calls,
        image_generation_images=image_generation_images,
        image_generation_bytes=image_generation_bytes,
        success_rate=round(success_rate, 2),
        health_rate=round(health_rate, 2),
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


@router.get("/image-generation")
async def list_image_generation(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    from app.database import get_db
    with get_db() as db:
        rows = db.execute("SELECT m.provider_id, m.model_id, m.image_generation, p.name AS provider_name FROM provider_models m JOIN providers p ON p.id = m.provider_id WHERE m.enabled = 1 ORDER BY m.provider_id, m.model_id").fetchall()
    models = [{
        "model_id": (
            r["model_id"][len(r["provider_id"]) + 1:]
            if r["model_id"].startswith(f"{r['provider_id']}/")
            else r["model_id"]
        ),
        "provider_model": (
            r["model_id"]
            if r["model_id"].startswith(f"{r['provider_id']}/")
            else f"{r['provider_id']}/{r['model_id']}"
        ),
        "provider_id": r["provider_id"],
        "provider_name": r["provider_name"],
        "model_name": (
            r["model_id"][len(r["provider_id"]) + 1:]
            if r["model_id"].startswith(f"{r['provider_id']}/")
            else r["model_id"]
        ),
        "image_generation": bool(r["image_generation"]),
    } for r in rows]
    providers = {}
    for model in models:
        providers.setdefault(model["provider_id"], {"id": model["provider_id"], "name": model["provider_name"], "models": []})["models"].append(model)
    generators = get_image_generators()
    for generator in generators.values():
        generator["has_api_key"] = bool(generator.get("api_key"))
        generator["api_key"] = ""
    return {"generators": generators, "models": models, "providers": list(providers.values())}


@router.post("/image-generation/comfyui/analyze-workflow")
async def analyze_comfyui_workflow(body: dict, authorization: Optional[str] = Header(None)):
    """Inspect API-format workflow JSON and return safe dropdown choices."""
    await require_admin_session(authorization)
    from app.adapters.comfyui import analyze_workflow
    try:
        return analyze_workflow((body or {}).get("workflow"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/image-generation/comfyui/workflows")
async def list_comfyui_workflows(body: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    from app.adapters.comfyui import list_saved_workflows
    body = dict(body or {})
    try:
        workflows = await list_saved_workflows(
            str(body.get("api_base") or ""),
            api_key=str(body.get("api_key") or ""),
            timeout=int(body.get("timeout") or 30),
        )
        return {"workflows": workflows}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/image-generation/comfyui/load-workflow")
async def load_comfyui_workflow(body: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    from app.adapters.comfyui import load_saved_workflow
    body = dict(body or {})
    try:
        workflow = await load_saved_workflow(
            str(body.get("api_base") or ""),
            str(body.get("workflow_name") or ""),
            api_key=str(body.get("api_key") or ""),
            timeout=int(body.get("timeout") or 30),
        )
        return {"workflow": workflow}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/image-generation/test")
async def test_image_generation(body: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    from app.adapters.imagegen import generate_images
    from app.router.proxy import _resolved_image_generator
    generator_id = str(body.get("generator_id") or "").strip()
    generator = get_image_generators().get(generator_id) if generator_id else None
    if not generator:
        raise HTTPException(status_code=404, detail="Image generator not found")
    prompt = str(body.get("prompt") or "A simple red apple on a white background").strip()
    started = time.perf_counter()
    try:
        resolved = _resolved_image_generator({"id": generator_id, **generator})
        results = await generate_images(resolved, prompt=prompt, n=1)
        return {"status": "ok", "generator_id": generator_id, "model": resolved.get("model") or resolved.get("provider_model") or "", "image_count": len(results), "latency_ms": int((time.perf_counter() - started) * 1000)}
    except Exception as exc:
        return {"status": "error", "generator_id": generator_id, "error": str(exc), "latency_ms": int((time.perf_counter() - started) * 1000)}


@router.put("/image-generation/{generator_id}")
async def update_image_generation(generator_id: str, config: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    config = dict(config or {})
    backend_type = str(config.get("backend_type") or "existing_model")
    existing = get_image_generators().get(generator_id) or {}
    if backend_type in {"external_model", "comfyui"} and not str(config.get("api_key") or "").strip():
        # The admin read endpoint never returns stored secrets. An empty
        # password field therefore means "keep the current key", matching
        # provider and preprocessor configuration behavior.
        same_secret_scope = (
            str(existing.get("backend_type") or "") == backend_type
            and str(existing.get("api_base") or "").rstrip("/")
            == str(config.get("api_base") or "").rstrip("/")
        )
        if same_secret_scope:
            config.pop("api_key", None)
        else:
            # Never forward a credential retained from a different backend or
            # host when the administrator changes the generator target.
            config["api_key"] = ""
    provider_model = str(config.get("provider_model") or "").strip()
    if backend_type == "existing_model" and provider_model:
        mid = parse_model_id(provider_model)
        provider = get_provider(mid.provider_id) if mid.provider_id else find_provider_by_model(mid.model_name)
        valid_ids = {
            m.get("id") for m in (provider or {}).get("models", [])
            if m.get("enabled", True)
        }
        if not provider or not ({mid.model_name, f"{mid.provider_id}/{mid.model_name}"} & valid_ids):
            raise HTTPException(status_code=400, detail="provider_model must reference an enabled provider model")
    if backend_type == "existing_model" and not provider_model:
        raise HTTPException(status_code=400, detail="an existing provider model is required")
    if backend_type == "external_model" and (not config.get("api_base") or not config.get("model")):
        raise HTTPException(status_code=400, detail="external model requires api_base and model")
    if backend_type == "comfyui":
        from app.adapters.comfyui import normalize_workflow, validate_mapping
        if not config.get("api_base"):
            raise HTTPException(status_code=400, detail="ComfyUI requires api_base")
        try:
            config["workflow"] = normalize_workflow(config.get("workflow"))
            config["workflow_mapping"] = validate_mapping(
                config["workflow"], config.get("workflow_mapping") or {},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if backend_type not in {"existing_model", "external_model", "comfyui"}:
        raise HTTPException(status_code=400, detail="unsupported image-generation backend type")
    try:
        current = upsert_image_generator(generator_id, config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    current.pop("id", None)
    return {"id": generator_id, "config": current}


@router.delete("/image-generation/{generator_id}")
async def delete_image_generation(generator_id: str, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    if delete_image_generator_config(generator_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Image generator not found")


@router.put("/models/image-generation")
async def toggle_model_image_generation(body: dict, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    model_id = body.get("model_id", "")
    enabled = body.get("enabled", False)
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="enabled must be a boolean")
    if not set_model_image_generation(model_id, enabled):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model_id": model_id, "image_generation": enabled}



# -- Request/Response detail logs --

_VALID_ENDPOINTS = {"chat_completions", "completions", "messages", "responses"}


# -- System log files --

_VALID_LOG_LEVELS = {"", "DEBUG", "INFO", "WARNING", "ERROR"}


@router.get("/system-logs/meta")
async def system_logs_meta(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    return {
        "channels": [
            {"id": channel, "filename": filename}
            for channel, filename in available_log_channels().items()
        ],
        "dates": list_log_dates(),
    }


@router.get("/system-logs")
async def list_system_logs_endpoint(
    date: Optional[str] = None,
    channel: str = "app",
    level: str = "",
    q: str = "",
    limit: int = 200,
    offset: int = 0,
    authorization: Optional[str] = Header(None),
):
    await require_admin_session(authorization)
    dates = list_log_dates()
    selected_date = (date or (dates[0] if dates else datetime.now(timezone.utc).strftime("%Y-%m-%d"))).strip()
    level = str(level or "").upper().strip()
    if level not in _VALID_LOG_LEVELS:
        raise HTTPException(status_code=400, detail="invalid log level")
    try:
        data = read_log_entries(
            selected_date,
            channel,
            limit=max(1, min(int(limit), 1000)),
            offset=max(0, int(offset)),
            level=level,
            q=q,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        **data,
        "date": selected_date,
        "channel": channel,
        "level": level,
        "q": q or "",
    }


@router.get("/request-logs")
async def list_request_logs_endpoint(
    limit: int = 50,
    offset: int = 0,
    endpoint: Optional[str] = None,
    username: Optional[str] = None,
    status: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    await require_admin_session(authorization)
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    if endpoint and endpoint not in _VALID_ENDPOINTS:
        raise HTTPException(status_code=400, detail="invalid endpoint")
    rows = list_request_logs(
        limit=limit,
        offset=offset,
        endpoint=endpoint,
        username=username,
        status=status,
    )
    total = count_request_logs(endpoint=endpoint, username=username, status=status)
    return {"items": rows, "total": total, "limit": limit, "offset": offset}


@router.get("/request-logs/{log_id}")
async def get_request_log_endpoint(log_id: int, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    entry = db_get_request_log(int(log_id))
    if not entry:
        raise HTTPException(status_code=404, detail="request log not found")
    return entry


@router.delete("/request-logs/{log_id}")
async def delete_request_log_endpoint(log_id: int, authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    if not db_delete_request_log(int(log_id)):
        raise HTTPException(status_code=404, detail="request log not found")
    return {"status": "deleted", "log_id": int(log_id)}


@router.post("/request-logs/clear")
async def clear_request_logs_endpoint(authorization: Optional[str] = Header(None)):
    username = await require_admin_session(authorization)
    removed = clear_request_logs()
    _app_log.warning("Request logs cleared by admin '%s' (removed=%d)", username, removed)
    return {"status": "ok", "removed": removed}


# -- Config export / import --

_CONFIG_VERSION = 1
_IMPORT_MODES = {"skip", "replace", "merge"}
_USER_EXPORT_VERSION = 1


def _export_config(include_secrets: bool) -> dict:
    providers = get_providers()
    preprocessors = get_preprocessors()
    image_generators = get_image_generators()
    if not include_secrets:
        for p in providers:
            p["api_key"] = ""
            p.pop("extra_headers", None)
        for p in preprocessors.values():
            p["api_key"] = ""
        for generator in image_generators.values():
            generator["api_key"] = ""
    return {
        "version": _CONFIG_VERSION,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "include_secrets": include_secrets,
        "providers": providers,
        "preprocessors": preprocessors,
        "image_generators": image_generators,
        "routing_rules": get_routing_rules(),
        "fallback_policies": get_fallback_policies(),
    }


def _export_users() -> dict:
    return {
        "version": _USER_EXPORT_VERSION,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "users": get_users(),
    }


@router.get("/users/export")
async def export_users_endpoint(authorization: Optional[str] = Header(None)):
    await require_admin_session(authorization)
    return _export_users()


def _validate_users_payload(payload) -> list:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="users export must be a JSON object")
    users = payload.get("users", [])
    if not isinstance(users, list):
        raise HTTPException(status_code=400, detail="users must be a list")
    for entry in users:
        if not isinstance(entry, dict) or not str(entry.get("username") or "").strip():
            raise HTTPException(status_code=400, detail="each user must be an object with username")
        if "api_keys" in entry and not isinstance(entry.get("api_keys"), list):
            raise HTTPException(status_code=400, detail="api_keys must be a list")
    return users


def _import_user_api_key(username: str, entry: dict, mode: str) -> str:
    key = str(entry.get("key") or "").strip()
    if not key:
        return "skipped"
    allowed_models = entry.get("allowed_models") if isinstance(entry.get("allowed_models"), list) else ["*"]
    name = str(entry.get("name") or "default")
    enabled = bool(entry.get("enabled", True))
    existing_user = get_user(username)
    existing_key = next((item for item in (existing_user or {}).get("api_keys", []) if item.get("key") == key), None)
    if mode == "skip" and existing_key:
        return "skipped"
    if existing_key:
        update_user_api_key(username, key, {"name": name, "allowed_models": allowed_models, "enabled": enabled})
        return "updated"
    from app.database import get_db
    created_at = str(entry.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    stats = entry.get("stats") if isinstance(entry.get("stats"), dict) else {}
    with get_db() as db:
        db.execute(
            """
            INSERT INTO user_api_keys
                (key, username, name, allowed_models, enabled, total_calls, failed_calls, total_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key, username, name, json.dumps(allowed_models, ensure_ascii=False), 1 if enabled else 0,
                int(stats.get("total_calls") or 0), int(stats.get("failed_calls") or 0),
                int(stats.get("total_tokens") or 0), created_at,
            ),
        )
    return "created"


def _import_user(entry: dict, mode: str) -> tuple[str, dict]:
    username = str(entry.get("username") or "").strip()
    existing = get_user(username)
    user_payload = {
        "username": username,
        "display_name": str(entry.get("display_name") or username),
        "enabled": bool(entry.get("enabled", True)),
    }
    if mode == "skip" and existing:
        user_outcome = "skipped"
    elif existing:
        if mode == "merge":
            update_user(username, {k: v for k, v in user_payload.items() if k != "username" and v not in (None, "")})
        else:
            update_user(username, {k: v for k, v in user_payload.items() if k != "username"})
        user_outcome = "updated"
    else:
        add_user(user_payload)
        user_outcome = "created"

    key_summary = {}
    if not (mode == "skip" and existing):
        for key_entry in entry.get("api_keys", []) or []:
            if isinstance(key_entry, dict):
                outcome = _import_user_api_key(username, key_entry, mode)
                key_summary[outcome] = key_summary.get(outcome, 0) + 1
    return user_outcome, key_summary


@router.post("/users/import")
async def import_users_endpoint(payload: dict, authorization: Optional[str] = Header(None)):
    username = await require_admin_session(authorization)
    mode = str(payload.get("mode") or "skip").lower()
    if mode not in _IMPORT_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {sorted(_IMPORT_MODES)}")
    users = _validate_users_payload(payload)
    summary = {"users": {}, "api_keys": {}}
    for entry in users:
        user_outcome, key_summary = _import_user(entry, mode)
        summary["users"][user_outcome] = summary["users"].get(user_outcome, 0) + 1
        for outcome, count in key_summary.items():
            summary["api_keys"][outcome] = summary["api_keys"].get(outcome, 0) + count
    _app_log.info("Users imported by '%s' mode=%s summary=%s", username, mode, summary)
    return {"status": "ok", "mode": mode, "summary": summary}


@router.get("/config/export")
async def export_config_endpoint(
    include_secrets: bool = False,
    authorization: Optional[str] = Header(None),
):
    await require_admin_session(authorization)
    return _export_config(bool(include_secrets))


def _validate_config_payload(payload) -> tuple[list, dict, dict, list, list]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="config must be a JSON object")
    providers = payload.get("providers", [])
    preprocessors = payload.get("preprocessors", {})
    image_generators = payload.get("image_generators", {})
    routing = payload.get("routing_rules", [])
    fallbacks = payload.get("fallback_policies", [])
    if not isinstance(providers, list):
        raise HTTPException(status_code=400, detail="providers must be a list")
    if not isinstance(preprocessors, dict):
        raise HTTPException(status_code=400, detail="preprocessors must be an object")
    if not isinstance(image_generators, dict):
        raise HTTPException(status_code=400, detail="image_generators must be an object")
    if not isinstance(routing, list):
        raise HTTPException(status_code=400, detail="routing_rules must be a list")
    if not isinstance(fallbacks, list):
        raise HTTPException(status_code=400, detail="fallback_policies must be a list")
    for entry in providers:
        if not isinstance(entry, dict) or not entry.get("id"):
            raise HTTPException(status_code=400, detail="each provider must be an object with id")
    for entry in routing:
        if not isinstance(entry, dict) or not entry.get("id"):
            raise HTTPException(status_code=400, detail="each routing_rule must be an object with id")
    for entry in fallbacks:
        if not isinstance(entry, dict) or not entry.get("id"):
            raise HTTPException(status_code=400, detail="each fallback_policy must be an object with id")
    for preprocessor_id, config in preprocessors.items():
        if not str(preprocessor_id or "").strip() or not isinstance(config, dict):
            raise HTTPException(status_code=400, detail="preprocessors must map ids to objects")
    for generator_id, config in image_generators.items():
        if not str(generator_id or "").strip() or not isinstance(config, dict):
            raise HTTPException(status_code=400, detail="image_generators must map ids to objects")
    return providers, preprocessors, image_generators, routing, fallbacks


def _import_provider(entry: dict, mode: str) -> str:
    """Return one of 'created', 'updated', 'skipped'."""
    pid = str(entry.get("id") or "").strip()
    if not pid:
        return "skipped"
    existing = get_provider(pid)
    payload = dict(entry)
    if not payload.get("api_key"):
        payload.pop("api_key", None)
    if mode == "skip" and existing:
        return "skipped"
    if mode == "merge" and existing:
        merged = {**existing, **{k: v for k, v in payload.items() if v not in (None, "", [], {})}}
        update_provider(pid, merged)
        return "updated"
    if existing:
        update_provider(pid, payload)
        return "updated"
    add_provider(payload)
    return "created"


def _import_preprocessor(preprocessor_id: str, config: dict, mode: str) -> str:
    pid = str(preprocessor_id or "").strip()
    if not pid:
        return "skipped"
    existing = get_preprocessors().get(pid)
    payload = dict(config or {})
    if not payload.get("api_key"):
        payload.pop("api_key", None)
    if mode == "skip" and existing:
        return "skipped"
    if mode == "merge" and existing:
        payload = {**existing, **{k: v for k, v in payload.items() if v not in (None, "", [], {})}}
    upsert_preprocessor(pid, payload)
    return "updated" if existing else "created"


def _import_image_generator(generator_id: str, config: dict, mode: str) -> str:
    gid = str(generator_id or "").strip()
    if not gid:
        return "skipped"
    existing = get_image_generators().get(gid)
    payload = dict(config or {})
    if not payload.get("api_key"):
        payload.pop("api_key", None)
    if mode == "skip" and existing:
        return "skipped"
    if mode == "merge" and existing:
        payload = {**existing, **{k: v for k, v in payload.items() if v not in (None, "", [], {})}}
    upsert_image_generator(gid, payload)
    return "updated" if existing else "created"


def _import_routing_rule(entry: dict, mode: str) -> str:
    rid = str(entry.get("id") or "").strip()
    if not rid:
        return "skipped"
    existing = get_routing_rule(rid)
    payload = {k: entry.get(k) for k in (
        "name", "enabled", "username", "api_key_pattern", "match_model", "match_scope", "target_model", "target_provider"
    ) if k in entry}
    if mode == "skip" and existing:
        return "skipped"
    if existing:
        update_routing_rule(rid, payload)
        return "updated"
    add_routing_rule({**payload, "id": rid})
    return "created"


def _import_fallback_policy(entry: dict, mode: str) -> str:
    pid = str(entry.get("id") or "").strip()
    if not pid:
        return "skipped"
    existing = get_fallback_policy(pid)
    payload = {k: entry.get(k) for k in (
        "name", "enabled", "match_provider", "match_model", "triggers", "chain", "attempt_timeout"
    ) if k in entry}
    if mode == "skip" and existing:
        return "skipped"
    if existing:
        update_fallback_policy(pid, payload)
        return "updated"
    add_fallback_policy({**payload, "id": pid})
    return "created"


@router.post("/config/import")
async def import_config_endpoint(payload: dict, authorization: Optional[str] = Header(None)):
    username = await require_admin_session(authorization)
    mode = str(payload.get("mode") or "skip").lower()
    if mode not in _IMPORT_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {sorted(_IMPORT_MODES)}")
    providers, preprocessors, image_generators, routing, fallbacks = _validate_config_payload(payload)

    summary = {"providers": {}, "preprocessors": {}, "image_generators": {}, "routing_rules": {}, "fallback_policies": {}}
    for entry in providers:
        outcome = _import_provider(entry, mode)
        summary["providers"][outcome] = summary["providers"].get(outcome, 0) + 1
    for preprocessor_id, config in preprocessors.items():
        outcome = _import_preprocessor(preprocessor_id, config, mode)
        summary["preprocessors"][outcome] = summary["preprocessors"].get(outcome, 0) + 1
    for generator_id, config in image_generators.items():
        outcome = _import_image_generator(generator_id, config, mode)
        summary["image_generators"][outcome] = summary["image_generators"].get(outcome, 0) + 1
    for entry in routing:
        outcome = _import_routing_rule(entry, mode)
        summary["routing_rules"][outcome] = summary["routing_rules"].get(outcome, 0) + 1
    for entry in fallbacks:
        outcome = _import_fallback_policy(entry, mode)
        summary["fallback_policies"][outcome] = summary["fallback_policies"].get(outcome, 0) + 1

    _app_log.info("Config imported by '%s' mode=%s summary=%s", username, mode, summary)
    return {"status": "ok", "mode": mode, "summary": summary}
