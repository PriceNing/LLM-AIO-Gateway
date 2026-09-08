"""共享的 httpx 异步客户端池。

Anthropic 与原生 Responses 适配器此前每个请求都 `async with httpx.AsyncClient(...)`，
每次都要重新建立 TCP+TLS 连接；对直连上游这是可以省掉的固定延迟（P8）。

这里按 (scheme+host+port, timeout) 作为键复用客户端：上游配置变更会自然产生
新键，旧客户端在空闲超过 ``idle_ttl_seconds`` 后关闭。

**引用计数是这套池的安全边界**：一个流式响应可能持续数分钟，期间它只在开始
时取过一次客户端。若驱逐只看空闲时间，另一条管理请求新建客户端就会把在飞的
客户端 ``aclose()``，表现为中途断流或 ``RuntimeError: client has been closed``。
因此 ``in_use > 0`` 的客户端绝不关闭；全部在用而池超限时，允许临时超过上限。
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from app.services.logger import get_logger

_app_log = get_logger("app")

_MAX_CLIENTS = 32
_IDLE_TTL_SECONDS = 300.0


@dataclass
class _Entry:
    client: httpx.AsyncClient
    last_used: float
    in_use: int = 0


_entries: dict[str, _Entry] = {}
_lock = asyncio.Lock()


def _pool_key(base_url: str, timeout: float) -> tuple[str, float]:
    parts = urlsplit(str(base_url or ""))
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return (f"{parts.scheme or 'https'}://{parts.hostname or ''}:{port}", float(timeout))


def _evict_locked(now: float, limit: int) -> list[httpx.AsyncClient]:
    """返回待关闭客户端；绝不触碰在用的条目。"""
    victims: list[httpx.AsyncClient] = []
    for key, entry in list(_entries.items()):
        if entry.in_use:
            continue
        if now - entry.last_used > _IDLE_TTL_SECONDS:
            _entries.pop(key, None)
            victims.append(entry.client)

    # 容量回收同样只挑空闲条目；全部在用则临时超限，不牺牲在飞请求。
    while len(_entries) > limit:
        idle = [(entry.last_used, key) for key, entry in _entries.items() if not entry.in_use]
        if not idle:
            _app_log.warning(
                "[http_pool] pool over capacity (%d>%d) but every client is in use; keeping them",
                len(_entries), limit,
            )
            break
        idle.sort()
        _used, oldest_key = idle[0]
        entry = _entries.pop(oldest_key)
        victims.append(entry.client)
    return victims


async def _safe_aclose(client: httpx.AsyncClient) -> None:
    try:
        await client.aclose()
    except Exception as exc:
        _app_log.warning("[http_pool] client close failed: %s", exc)


async def acquire(base_url: str, timeout: float) -> tuple[httpx.AsyncClient, str]:
    """取用客户端并登记引用；必须与 release 成对调用。"""
    key = _pool_key(base_url, timeout)
    now = time.monotonic()
    async with _lock:
        entry = _entries.get(key)
        if entry is None:
            entry = _Entry(client=httpx.AsyncClient(timeout=timeout), last_used=now)
            _entries[key] = entry
        entry.in_use += 1
        entry.last_used = now
        victims = _evict_locked(now, _MAX_CLIENTS)
    for client in victims:
        await _safe_aclose(client)
    return entry.client, key


async def release(key: str) -> None:
    """归还引用；条目归零后从池中摘除则立即关闭。"""
    async with _lock:
        entry = _entries.get(key)
        if entry is None:
            return
        entry.in_use = max(0, entry.in_use - 1)
        entry.last_used = time.monotonic()
        # 超过硬上限且已空闲：直接摘除，避免池无界滞留。
        if len(_entries) > _MAX_CLIENTS and not entry.in_use:
            _entries.pop(key, None)
            victim = entry.client
        else:
            victim = None
    if victim is not None:
        await _safe_aclose(victim)


@asynccontextmanager
async def shared_client(base_url: str, timeout: float):
    """Drop-in replacement for ``async with httpx.AsyncClient(timeout=...)``.

    退出时只归还引用，不关闭客户端；因此调用点无需重新缩进，也不会误关共享连接。
    """
    client, key = await acquire(base_url, timeout)
    try:
        yield client
    finally:
        await release(key)


async def get_shared_client(base_url: str, timeout: float) -> httpx.AsyncClient:
    """不带引用的取用方式，仅供测试与无需长持有的路径使用。"""
    client, _key = await acquire(base_url, timeout)
    # 立即归还，避免调用方忘记 release 造成条目永久免于驱逐。
    await release(_key)
    return client


async def aclose_shared_clients() -> None:
    """Close every pooled client. Called from the application lifespan shutdown."""
    async with _lock:
        clients = [entry.client for entry in _entries.values()]
        _entries.clear()
    for client in clients:
        await _safe_aclose(client)
    _app_log.debug("[http_pool] closed %d shared clients", len(clients))


def shared_client_pool_size() -> int:
    return len(_entries)


def in_use_count() -> int:
    return sum(1 for entry in _entries.values() if entry.in_use)
