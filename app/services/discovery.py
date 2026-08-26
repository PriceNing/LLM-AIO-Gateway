import asyncio
import time

import httpx
from app.database import get_provider, update_provider, get_providers, get_db
from app.adapters.responses import iter_sse_frames, responses_headers, responses_url, sse_payload


def model_list_urls(api_base: str, provider_type: str) -> list[str]:
    """Return candidate model-list URLs for the given provider.

    For Anthropic-compatible endpoints (non api.anthropic.com), also try
    the parent path - e.g. DeepSeek's /anthropic base has no /models,
    but the root /v1/models works."""
    api_base = api_base.rstrip("/")
    urls = []
    if provider_type == "anthropic":
        if not api_base.endswith("/v1"):
            urls.append(f"{api_base}/v1/models")
        urls.append(f"{api_base}/models")
        # Anthropic-compatible endpoints may host /models on a different base path
        if "api.anthropic.com" not in api_base:
            parent = api_base.rsplit("/", 1)[0]
            if parent and parent != api_base:
                for u in (f"{parent}/v1/models", f"{parent}/models"):
                    if u not in urls:
                        urls.append(u)
    else:
        urls = [f"{api_base}/models"]
    return urls


def auth_headers(api_key: str, provider_type: str) -> list[dict]:
    if api_key:
        headers = [{"Authorization": f"Bearer {api_key}"}]
        if provider_type == "anthropic":
            headers.append({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
    else:
        headers = [{}]
    return headers


def parse_models(data: dict) -> list[dict]:
    raw_models = data.get("data")
    if raw_models is None:
        raw_models = data.get("models", [])
    if not isinstance(raw_models, list):
        return []

    models = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("identifier") or item.get("name")
        if model_id:
            models.append({
                "id": model_id,
                "name": item.get("display_name") or item.get("name") or model_id
            })
    return models


async def discover_models(provider_id: str) -> list[dict]:
    provider = get_provider(provider_id)
    if not provider:
        return []
    if not provider.get("enabled"):
        return []

    api_base = provider["api_base"].rstrip("/")
    api_key = provider["api_key"]
    provider_type = provider["provider_type"]

    last_error = None
    async with httpx.AsyncClient() as client:
        for url in model_list_urls(api_base, provider_type):
            for headers in auth_headers(api_key, provider_type):
                try:
                    models_by_id = {}
                    cursor = ""
                    seen_cursors = set()
                    for _page in range(100):
                        params = None
                        if cursor:
                            params = {"after_id" if provider_type == "anthropic" else "after": cursor}
                        request_kwargs = {"headers": headers, "timeout": 10.0}
                        if params:
                            request_kwargs["params"] = params
                        resp = await client.get(url, **request_kwargs)
                        resp.raise_for_status()
                        payload = resp.json()
                        page_models = parse_models(payload)
                        for model in page_models:
                            models_by_id[str(model["id"])] = model
                        if not payload.get("has_more"):
                            break
                        next_cursor = str(payload.get("last_id") or "").strip()
                        if not next_cursor and page_models:
                            next_cursor = str(page_models[-1]["id"])
                        if not next_cursor or next_cursor in seen_cursors:
                            raise RuntimeError("model discovery returned an invalid pagination cursor")
                        seen_cursors.add(next_cursor)
                        cursor = next_cursor
                    else:
                        raise RuntimeError("model discovery exceeded 100 pages")
                    if models_by_id:
                        return list(models_by_id.values())
                except Exception as exc:
                    last_error = exc

    if last_error:
        raise last_error

    return []


async def refresh_provider_models(provider_id: str) -> dict:
    try:
        discovered = await discover_models(provider_id)
    except Exception as exc:
        return {
            "provider_id": provider_id,
            "discovered": [],
            "count": 0,
            "added": 0,
            "updated": 0,
            "removed": 0,
            "error": str(exc)
        }

    added = 0
    updated = 0
    removed = 0

    if discovered:
        discovered_by_id = {d["id"]: d for d in discovered}
        discovered_ids = set(discovered_by_id)
        with get_db() as db:
            existing_rows = db.execute(
                "SELECT model_id, source FROM provider_models WHERE provider_id = ?",
                (provider_id,),
            ).fetchall()
            existing_ids = {row["model_id"] for row in existing_rows}
            # Manual models are user-owned: refresh never deletes them and never
            # overwrites their display name, even when the same id also appears
            # upstream. Only auto-discovered rows stay in sync with /models.
            auto_ids = {
                row["model_id"]
                for row in existing_rows
                if (row["source"] or "auto") == "auto"
            }

            stale_ids = auto_ids - discovered_ids
            if stale_ids:
                db.executemany(
                    "DELETE FROM provider_models WHERE provider_id = ? AND model_id = ? AND source = 'auto'",
                    [(provider_id, model_id) for model_id in stale_ids],
                )
                removed = len(stale_ids)

            for model_id, model in discovered_by_id.items():
                if model_id in existing_ids:
                    cursor = db.execute(
                        "UPDATE provider_models SET model_name = ? WHERE provider_id = ? AND model_id = ? AND source = 'auto'",
                        (model["name"], provider_id, model_id),
                    )
                    if cursor.rowcount:
                        updated += 1
                else:
                    db.execute(
                        "INSERT OR IGNORE INTO provider_models (provider_id, model_id, model_name, enabled, source) VALUES (?, ?, ?, 1, 'auto')",
                        (provider_id, model_id, model["name"]),
                    )
                    added += 1

    return {
        "provider_id": provider_id,
        "discovered": discovered,
        "count": len(discovered),
        "added": added,
        "updated": updated,
        "removed": removed,
    }


async def refresh_all_providers() -> list[dict]:
    providers = [provider for provider in get_providers() if provider.get("enabled")]
    limiter = asyncio.Semaphore(4)

    async def refresh(provider_id: str) -> dict:
        async with limiter:
            return await refresh_provider_models(provider_id)

    return list(await asyncio.gather(*(refresh(provider["id"]) for provider in providers)))


async def check_provider_health(provider_id: str, timeout: float = 10.0) -> dict:
    provider = get_provider(provider_id)
    if not provider:
        return {"provider_id": provider_id, "ok": False, "status": "not_found", "error": "Provider not found"}
    if not provider.get("enabled"):
        return {"provider_id": provider_id, "ok": False, "status": "disabled", "error": "Provider is disabled"}

    started = time.perf_counter()
    last_error = None
    attempts = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for url in model_list_urls(provider.get("api_base", ""), provider.get("provider_type", "openai")):
            for headers in auth_headers(provider.get("api_key", ""), provider.get("provider_type", "openai")):
                attempt = {"url": url, "ok": False, "status_code": None, "model_count": 0, "error": ""}
                try:
                    resp = await client.get(url, headers=headers)
                    attempt["status_code"] = resp.status_code
                    resp.raise_for_status()
                    models = parse_models(resp.json())
                    attempt["ok"] = True
                    attempt["model_count"] = len(models)
                    attempts.append(attempt)
                    return {
                        "provider_id": provider_id,
                        "ok": True,
                        "status": "ok",
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "checked_url": url,
                        "status_code": resp.status_code,
                        "model_count": len(models),
                        "attempts": attempts,
                    }
                except Exception as exc:
                    last_error = exc
                    attempt["error"] = str(exc)
                    attempts.append(attempt)

    return {
        "provider_id": provider_id,
        "ok": False,
        "status": "error",
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "error": str(last_error) if last_error else "No health endpoint succeeded",
        "attempts": attempts,
    }


async def check_all_provider_health(timeout: float = 10.0) -> list[dict]:
    results = []
    for provider in get_providers():
        results.append(await check_provider_health(provider["id"], timeout=timeout))
    return results
