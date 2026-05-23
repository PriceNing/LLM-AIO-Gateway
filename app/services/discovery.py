import httpx
from app.database import get_provider, update_provider, get_providers, get_db


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
                    resp = await client.get(url, headers=headers, timeout=10.0)
                    resp.raise_for_status()
                    models = parse_models(resp.json())
                    if models:
                        return models
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
            "error": str(exc)
        }

    if discovered:
        existing_provider = get_provider(provider_id)
        existing_ids = {m["id"] for m in existing_provider.get("models", [])}
        new_models = [d for d in discovered if d["id"] not in existing_ids]
        if new_models:
            with get_db() as db:
                for d in new_models:
                    db.execute(
                        "INSERT OR IGNORE INTO provider_models (provider_id, model_id, model_name, enabled) VALUES (?, ?, ?, 1)",
                        (provider_id, d["id"], d["name"])
                    )

    return {"provider_id": provider_id, "discovered": discovered, "count": len(discovered)}


async def refresh_all_providers() -> list[dict]:
    results = []
    for provider in get_providers():
        if provider.get("enabled"):
            result = await refresh_provider_models(provider["id"])
            results.append(result)
    return results
