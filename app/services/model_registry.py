"""在线模型能力注册表。

内置家族表（app/core/model_capabilities.py）会随模型迭代过时，权威数据改为
在线获取：默认拉取 OpenRouter ``/api/v1/models``（无需鉴权，包含
context_length / architecture.input_modalities / top_provider /
supported_parameters / pricing），解析为统一能力字典后持久化到 SQLite。

优先级：内置家族表 < 在线注册表（本模块） < 上游 /models 透传 < 管理员覆盖。

离线可用性：首次成功拉取后缓存落库，之后断网也能命中；从未成功拉取时
返回空 dict，由内置表兜底。刷新由 main.py 维护循环按 TTL 触发，也可在
管理面板手动刷新（POST /admin/models/registry/refresh）。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import get_default
from app.services.logger import get_logger

_app_log = get_logger("app")

DEFAULT_REGISTRY_URL = "https://openrouter.ai/api/v1/models"

# 内存缓存：避免每次 /v1/models 都读 DB 并解析 JSON。
_mem_lock = threading.Lock()
_mem_cache: dict[str, Any] = {
    "map": None, "count": 0, "fetched_at": "", "url": "", "loaded_at": 0.0,
    # 失败退避：拉取失败后在窗口内不再重试，避免离线部署每 60s 打外网刷日志。
    "last_attempt_at": 0.0, "last_error": "",
}
_MEM_TTL_SECONDS = 300.0
_FAILURE_BACKOFF_SECONDS = 3600.0
_MAX_REGISTRY_BYTES = 10 * 1024 * 1024
_MAX_REDIRECTS = 2


def _registry_url() -> str:
    url = str(get_default("model_registry_url", DEFAULT_REGISTRY_URL) or DEFAULT_REGISTRY_URL).strip()
    return url or DEFAULT_REGISTRY_URL


def _registry_ttl_seconds() -> int:
    try:
        return max(3600, int(get_default("model_registry_ttl_seconds", 604800)))
    except (TypeError, ValueError):
        return 604800


def registry_enabled() -> bool:
    return bool(get_default("model_registry_enabled", True))


def _slug_keys(model_id: str) -> list[str]:
    """为一个模型 id 生成全部索引/查询键。

    OpenRouter 条目形如 ``anthropic/claude-sonnet-4``，还存在 ``~xxx``（路由
    聚合入口）与 ``:batch`` 等变体后缀；网关侧模型可能是裸名
    ``deepseek-flash`` 或复合名 ``provider/deepseek-flash``。统一小写、
    去掉 ``~`` 前缀与 ``:variant`` 后缀后，同时按全 id 与裸名（最后一段）索引。
    """
    raw = str(model_id or "").strip().lower()
    if not raw:
        return []
    if raw.startswith("~"):
        raw = raw[1:]
    base = raw.split(":", 1)[0]
    keys = [base]
    tail = base.rsplit("/", 1)[-1]
    if tail and tail != base:
        keys.append(tail)
    return keys


def _build_lookup_map(entries: list[dict]) -> dict[str, dict]:
    """把 [{id, capabilities}] 列表压成 {slug: caps}，冲突处理确定性：

    1. 正式条目（无 ``:variant`` 后缀）优先：``:free``/``:extended`` 等变体
       归一后与基础模型同键时，变体能力不得覆盖基础模型；
    2. 同优先级内同键不同能力（如两家厂商同名裸名） → 该键不落地：
       先到者胜会依赖上游返回顺序，非确定；丢弃冲突键则无论顺序如何
       结果一致，未命中的模型回退内置表/上游透传，好过广告错误能力。
    """
    lookup: dict[str, dict] = {}
    conflicted: set[str] = set()
    canonical_keys: set[str] = set()
    for pass_idx in (0, 1):  # 0=正式条目，1=变体条目
        for entry in entries:
            caps = entry.get("capabilities")
            if not isinstance(caps, dict) or not caps:
                continue
            entry_id = str(entry.get("id") or "")
            is_variant = ":" in entry_id
            if (pass_idx == 1) != is_variant:
                continue
            for key in _slug_keys(entry_id):
                if key in conflicted:
                    continue
                existing = lookup.get(key)
                if existing is None:
                    lookup[key] = caps
                    if pass_idx == 0:
                        canonical_keys.add(key)
                elif existing != caps:
                    if pass_idx == 0 or key not in canonical_keys:
                        # 同优先级冲突（正式 vs 正式，或无正式条目时的
                        # 变体 vs 变体）→ 歧义键废弃，不依赖上游返回顺序。
                        conflicted.add(key)
                        lookup.pop(key, None)
                    # 变体与正式条目冲突 → 正式条目胜，跳过
    if conflicted:
        _app_log.debug("[model_registry] dropped %d ambiguous slug keys (e.g. %s)", len(conflicted), sorted(conflicted)[:5])
    return lookup


def _record_attempt_error(message: str) -> None:
    with _mem_lock:
        _mem_cache["last_attempt_at"] = time.monotonic()
        _mem_cache["last_error"] = message[:300]


async def _get_with_guard(client: httpx.AsyncClient, url: str) -> bytes:
    """带 SSRF 校验、受控重定向与体积上限的 GET。"""
    from urllib.parse import urljoin

    from app.services.url_guard import validate_upstream_url_async

    current = url
    for redirect in range(_MAX_REDIRECTS + 1):
        # 与项目安全惯例一致：每一跳都过 url_guard（含 DNS 解析，拦元数据
        # 地址/非法 scheme/解析到被禁 IP 的主机名），并采用其归一化返回值。
        current = await validate_upstream_url_async(current, field="model_registry_url")
        async with client.stream("GET", current, follow_redirects=False) as resp:
            if 300 <= resp.status_code < 400:
                location = resp.headers.get("location")
                if not location or redirect == _MAX_REDIRECTS:
                    raise ValueError(f"registry URL returned an unresolvable redirect (HTTP {resp.status_code})")
                current = urljoin(current, location)
                continue
            resp.raise_for_status()
            chunks = bytearray()
            async for chunk in resp.aiter_bytes():
                chunks.extend(chunk)
                if len(chunks) > _MAX_REGISTRY_BYTES:
                    raise ValueError(f"registry response exceeds {_MAX_REGISTRY_BYTES} bytes")
            return bytes(chunks)
    raise ValueError("registry URL exceeded the redirect limit")


async def fetch_and_store_registry(url: str | None = None, *, timeout: float = 20.0) -> dict:
    """拉取在线注册表并持久化。返回 {status, models, fetched_at} 或 {status: "error", ...}。"""
    from app.database import save_model_registry

    url = (url or _registry_url()).strip()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            body = await _get_with_guard(client, url)
        payload = json.loads(body)
    except Exception as exc:
        message = str(exc)[:300]
        _record_attempt_error(message)
        _app_log.warning("[model_registry] fetch failed url=%s error=%s", url, message[:200])
        return {"status": "error", "error": message, "url": url}

    # 复用 discovery 的上游元数据提取（OpenRouter 字段口径一致）
    from app.services.discovery import upstream_capabilities

    raw_models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list) or not raw_models:
        _record_attempt_error("registry payload has no model list")
        return {"status": "error", "error": "registry payload has no model list", "url": url}

    entries = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "")
        if not model_id:
            continue
        caps = upstream_capabilities(item)
        if caps:
            entries.append({"id": model_id, "capabilities": caps})

    if not entries:
        _record_attempt_error("registry payload contained no capability metadata")
        return {"status": "error", "error": "registry payload contained no capability metadata", "url": url}

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    await asyncio.to_thread(save_model_registry, url, entries, fetched_at)
    with _mem_lock:
        _mem_cache.update({
            "map": _build_lookup_map(entries), "count": len(entries),
            "fetched_at": fetched_at, "url": url, "loaded_at": time.monotonic(),
            "last_attempt_at": time.monotonic(), "last_error": "",
        })
    _app_log.info("[model_registry] refreshed url=%s models=%d", url, len(entries))
    return {"status": "ok", "models": len(entries), "fetched_at": fetched_at, "url": url}


def _load_from_db() -> None:
    from app.database import load_model_registry

    record = load_model_registry()
    with _mem_lock:
        if record and record.get("entries"):
            _mem_cache.update({
                "map": _build_lookup_map(record["entries"]),
                "count": len(record["entries"]),
                "fetched_at": record.get("fetched_at", ""),
                "url": record.get("url", ""),
                "loaded_at": time.monotonic(),
            })
        else:
            _mem_cache.update({"map": {}, "count": 0, "fetched_at": "", "url": "", "loaded_at": time.monotonic()})


def _ensure_loaded() -> dict[str, dict]:
    with _mem_lock:
        fresh = _mem_cache["map"] is not None and (time.monotonic() - _mem_cache["loaded_at"]) < _MEM_TTL_SECONDS
    if not fresh:
        _load_from_db()
    with _mem_lock:
        return _mem_cache["map"] or {}


def registry_lookup(model_id: str, model_name: str = "") -> dict:
    """按模型 id/名称查询在线注册表能力；未命中返回空 dict（不猜测）。"""
    if not registry_enabled():
        return {}
    lookup = _ensure_loaded()
    if not lookup:
        return {}
    for candidate in (model_id, model_name):
        for key in _slug_keys(candidate):
            caps = lookup.get(key)
            if caps:
                return dict(caps)
    return {}


def registry_status() -> dict:
    if not registry_enabled():
        # 保持与启用时相同的字段契约，避免调用方需要分支处理。
        return {
            "enabled": False,
            "url": _registry_url(),
            "fetched_at": "",
            "model_count": 0,
            "ttl_seconds": _registry_ttl_seconds(),
            "stale": False,
            "last_error": "",
        }
    _ensure_loaded()
    with _mem_lock:
        fetched_at = _mem_cache.get("fetched_at") or ""
        url = _mem_cache.get("url") or ""
        model_count = int(_mem_cache.get("count") or 0)
        last_error = str(_mem_cache.get("last_error") or "")
    stale = False
    if fetched_at:
        try:
            parsed = datetime.strptime(fetched_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            stale = (datetime.now(timezone.utc) - parsed).total_seconds() > _registry_ttl_seconds()
        except ValueError:
            stale = True
    return {
        "enabled": True,
        "url": url or _registry_url(),
        "fetched_at": fetched_at,
        "model_count": model_count,
        "ttl_seconds": _registry_ttl_seconds(),
        "stale": stale or not fetched_at,
        "last_error": last_error,
    }


def registry_needs_refresh() -> bool:
    if not registry_enabled():
        return False
    if not registry_status().get("stale"):
        return False
    # 失败退避：上次尝试（无论成败）在窗口内则不重试，避免离线部署
    # 每个维护周期（60s）都发起最长 20s 的外网请求并刷日志。
    with _mem_lock:
        last_attempt = float(_mem_cache.get("last_attempt_at") or 0.0)
    if last_attempt and (time.monotonic() - last_attempt) < min(_FAILURE_BACKOFF_SECONDS, float(_registry_ttl_seconds())):
        return False
    return True


async def refresh_registry_if_stale() -> dict | None:
    """维护循环调用：超过 TTL 才拉取，避免每 60s 打一次外网。

    新鲜度判定内部的 DB 读/JSON 解析移出事件循环，与 admin 端点口径一致。
    """
    if not registry_enabled():
        return None
    if not await asyncio.to_thread(registry_needs_refresh):
        return None
    return await fetch_and_store_registry()
