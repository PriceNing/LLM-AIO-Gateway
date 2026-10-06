import asyncio
import json
import time

import httpx
from app.database import get_provider, update_provider, get_providers, get_db, merge_upstream_model_capabilities
from app.adapters.responses import iter_sse_frames, responses_headers, responses_url, sse_payload
from app.services.url_guard import pinned_request


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


def _pick_positive_int(item: dict, keys: tuple[str, ...]):
    for key in keys:
        value = item.get(key)
        if value is None or isinstance(value, bool):
            # bool 是 int 子类：上游把 context_length 写成 true 时，
            # int(True)==1 会广告成"1 token 上下文"，必须显式拒绝。
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def upstream_capabilities(item: dict) -> dict:
    """从上游 /models 条目提取能力元数据（OpenRouter / new-api / one-api 风格）。

    普通 OpenAI 兼容端点通常只返回 id/created/owned_by，提不到任何字段时
    返回空 dict，由内置启发式与管理员覆盖兜底。
    """
    caps: dict = {}
    top_provider = item.get("top_provider") if isinstance(item.get("top_provider"), dict) else {}
    context_window = _pick_positive_int(item, ("context_length", "context_window", "max_context_tokens", "max_context"))
    if context_window is None:
        context_window = _pick_positive_int(top_provider, ("context_length", "max_context_tokens"))
    if context_window:
        caps["context_window"] = context_window
    max_output = _pick_positive_int(item, ("max_output_tokens", "max_completion_tokens"))
    if max_output is None:
        max_output = _pick_positive_int(top_provider, ("max_output_tokens", "max_completion_tokens"))
    if max_output:
        caps["max_output_tokens"] = max_output

    raw_caps = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
    architecture = item.get("architecture") if isinstance(item.get("architecture"), dict) else {}
    modalities = item.get("input_modalities") or architecture.get("input_modalities")
    if isinstance(modalities, list) and modalities:
        caps["input_modalities"] = [str(m) for m in modalities]

    vision = raw_caps.get("vision")
    if vision is None:
        vision = item.get("supports_vision")
    if vision is None and caps.get("input_modalities"):
        vision = "image" in caps["input_modalities"]
    if vision is not None:
        # 透传原值，由 normalize_capabilities 严格解析：上游用字符串
        # "false"/"0"/"no" 表达否定时，bool() 强转会把它变成 True。
        caps["supports_vision"] = vision

    supported_params = item.get("supported_parameters")
    supported_params = supported_params if isinstance(supported_params, list) else []

    tools = raw_caps.get("function_calling")
    if tools is None:
        tools = item.get("supports_tools")
    if tools is None and supported_params:
        tools = any(p in supported_params for p in ("tools", "tool_use", "function_calling"))
    if tools is not None:
        caps["supports_tools"] = tools

    # 推理能力：OpenRouter 用 supported_parameters 里的 reasoning/include_reasoning
    # 表达；自建网关也可能直接给布尔字段。无法判定时不输出（保持未知）。
    reasoning = raw_caps.get("reasoning")
    if reasoning is None:
        reasoning = item.get("supports_reasoning")
    if reasoning is None and supported_params:
        reasoning = any(p in supported_params for p in ("reasoning", "reasoning_effort", "include_reasoning"))
    if reasoning is not None:
        caps["supports_reasoning"] = reasoning

    pricing = item.get("pricing")
    if isinstance(pricing, dict):
        cleaned = {k: str(v) for k, v in pricing.items() if k in ("prompt", "completion", "image", "request") and isinstance(v, (str, int, float))}
        if cleaned:
            caps["pricing"] = cleaned

    from app.core.model_capabilities import normalize_capabilities
    return normalize_capabilities(caps)


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
            entry = {
                "id": model_id,
                "name": item.get("display_name") or item.get("name") or model_id,
            }
            caps = upstream_capabilities(item)
            if caps:
                entry["capabilities"] = caps
            models.append(entry)
    return models


# 上游 /models 响应体积上限，与 model_registry 的抓取同口径（bug-2026-10-05 L-23）：
# 无上限时异常/恶意上游可用超大 JSON 在 resp.json() 处打爆网关内存。
_MAX_DISCOVERY_BYTES = 10 * 1024 * 1024


async def _get_capped(client, url, **kwargs) -> tuple[int, bytes]:
    """流式读取、带字节上限的 GET，返回 (status_code, body)。

    httpx 非流式 get() 返回时整个 body 已进内存，事后检查 Content-Length
    只能阻止超大 JSON 被解析、不能阻止它被持有（bug-2026-10-05 R-5）。
    流式累计超上限立即中断；header 声明检查保留作快速失败路径。
    不抛状态异常：由调用方按 status 决定（健康检查需要记录状态码）。
    """
    async with client.stream("GET", url, **kwargs) as resp:
        headers = getattr(resp, "headers", None) or {}
        try:
            declared = str(headers.get("content-length") or "")
        except Exception:  # noqa: BLE001 - 假响应对象缺 headers 时不做判定
            declared = ""
        if declared.isdigit() and int(declared) > _MAX_DISCOVERY_BYTES:
            raise RuntimeError(f"upstream models response exceeds {_MAX_DISCOVERY_BYTES} bytes")
        chunks = bytearray()
        async for chunk in resp.aiter_bytes():
            chunks.extend(chunk)
            if len(chunks) > _MAX_DISCOVERY_BYTES:
                raise RuntimeError(f"upstream models response exceeds {_MAX_DISCOVERY_BYTES} bytes")
        return resp.status_code, bytes(chunks)


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
                        # 请求时钉 IP：校验过的地址与实际连接地址一致（R-1）。
                        target = await pinned_request(url)
                        request_kwargs["headers"] = {**headers, **target.headers}
                        if target.extensions:
                            request_kwargs["extensions"] = target.extensions
                        status, body = await _get_capped(client, target.url, **request_kwargs)
                        if status >= 400:
                            raise RuntimeError(f"upstream models endpoint returned HTTP {status}")
                        payload = json.loads(body)
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
                "SELECT model_id, preprocessor, image_generation, responses_status FROM provider_models WHERE provider_id = ?",
                (provider_id,),
            ).fetchall()
            existing_ids = {row["model_id"] for row in existing_rows}

            def _has_admin_config(row) -> bool:
                if str(row["preprocessor"] or "").strip():
                    return True
                if str(row["image_generation"] or "").strip():
                    return True
                # 只有探测成功的结果值得保留（避免重新探测的开销）；
                # error/unsupported 等失败状态不是管理员配置，不应让模型永久滞留。
                return str(row["responses_status"] or "") == "supported"

            # 上游模型列表临时变动（改名/分页/权限）时，直接 DELETE 会丢失
            # 管理员在该模型上配置的 preprocessor / 生图标记 / 能力探测结果；
            # 有配置的过期模型保留，只清理无配置的。
            configured_ids = {row["model_id"] for row in existing_rows if _has_admin_config(row)}
            stale_ids = existing_ids - discovered_ids
            deletable_ids = stale_ids - configured_ids
            if deletable_ids:
                db.executemany(
                    "DELETE FROM provider_models WHERE provider_id = ? AND model_id = ?",
                    [(provider_id, model_id) for model_id in deletable_ids],
                )
                removed = len(deletable_ids)

            for model_id, model in discovered_by_id.items():
                if model_id in existing_ids:
                    db.execute(
                        "UPDATE provider_models SET model_name = ? WHERE provider_id = ? AND model_id = ?",
                        (model["name"], provider_id, model_id),
                    )
                    if model.get("capabilities"):
                        merge_upstream_model_capabilities(db, provider_id, model_id, model["capabilities"])
                    updated += 1
                else:
                    db.execute(
                        "INSERT INTO provider_models (provider_id, model_id, model_name, enabled, capabilities) VALUES (?, ?, ?, 1, ?)",
                        (provider_id, model_id, model["name"], json.dumps(model.get("capabilities") or {}, ensure_ascii=False)),
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
                    target = await pinned_request(url)
                    request_kwargs = {"headers": {**headers, **target.headers}}
                    if target.extensions:
                        request_kwargs["extensions"] = target.extensions
                    status, body = await _get_capped(client, target.url, **request_kwargs)
                    attempt["status_code"] = status
                    if status >= 400:
                        raise RuntimeError(f"health endpoint returned HTTP {status}")
                    models = parse_models(json.loads(body))
                    attempt["ok"] = True
                    attempt["model_count"] = len(models)
                    attempts.append(attempt)
                    return {
                        "provider_id": provider_id,
                        "ok": True,
                        "status": "ok",
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "checked_url": url,
                        "status_code": status,
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


# 健康检查并发上限：无上限时 N 个 provider 同时创建 client、DNS 查询与
# 超时任务，规模化部署下管理端会瞬现连接/任务尖峰（bug-2026-10-05 R-4）。
_HEALTH_CHECK_CONCURRENCY = 8


async def check_all_provider_health(timeout: float = 10.0) -> list[dict]:
    # 并发检查：串行时 N 个 provider 的总耗时是 N 个 10s 超时之和，一个挂死的
    # 上游就能把管理端健康检查拖满整分钟（L-23）。每个 provider 独立 client，
    # 并发无共享状态；semaphore 只限瞬时并发，不改结果集合（R-4）。
    providers = get_providers()
    limiter = asyncio.Semaphore(_HEALTH_CHECK_CONCURRENCY)

    async def guarded(provider_id: str) -> dict:
        async with limiter:
            return await check_provider_health(provider_id, timeout=timeout)

    return list(await asyncio.gather(*(guarded(provider["id"]) for provider in providers)))
