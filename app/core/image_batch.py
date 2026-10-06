"""Short-lived in-process idempotency for generated-image invocations."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImageInvocationClaim:
    key: str
    owner: bool
    future: concurrent.futures.Future


@dataclass
class _Entry:
    future: concurrent.futures.Future
    created_at: float
    completed_at: float | None = None


class ImageInvocationCache:
    """Coordinate identical in-flight calls and briefly reuse completed artifacts."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def claim(self, key: str, *, ttl_seconds: int, max_entries: int, inflight_max_age_seconds: float = 0) -> ImageInvocationClaim:
        now = time.monotonic()
        ttl = max(1, int(ttl_seconds))
        limit = max(1, int(max_entries))
        with self._lock:
            expired = [
                entry_key for entry_key, entry in self._entries.items()
                if entry.completed_at is not None and now - entry.completed_at > ttl
            ]
            # in-flight 条目同样要有寿命上限：owner 在 claim 后、resolve/reject 前
            # 因未预期路径退出时，旧实现里该条目永不超时（过期清理只认
            # completed_at，容量逐出也只弹已完成条目），等待者只能靠自身超时
            # 脱身（bug-2026-10-05 M-24）。超过 owner 合法寿命上限即视为遗弃。
            if inflight_max_age_seconds > 0:
                abandoned = [
                    entry_key for entry_key, entry in self._entries.items()
                    if entry.completed_at is None and now - entry.created_at > inflight_max_age_seconds
                ]
            else:
                abandoned = []
            for entry_key in abandoned:
                dropped = self._entries.pop(entry_key, None)
                if dropped is not None and not dropped.future.done():
                    dropped.future.set_exception(
                        RuntimeError("image generation owner was abandoned; retry the request")
                    )
            for entry_key in expired:
                self._entries.pop(entry_key, None)

            existing = self._entries.get(key)
            if existing is not None:
                return ImageInvocationClaim(key=key, owner=False, future=existing.future)

            completed = sorted(
                (
                    (entry.completed_at or entry.created_at, entry_key)
                    for entry_key, entry in self._entries.items()
                    if entry.completed_at is not None
                ),
                key=lambda item: item[0],
            )
            while len(self._entries) >= limit and completed:
                _, entry_key = completed.pop(0)
                self._entries.pop(entry_key, None)
            if len(self._entries) >= limit:
                raise RuntimeError("too many concurrent image-generation invocations")

            future: concurrent.futures.Future[Any] = concurrent.futures.Future()
            # A failed owner may have no waiter. Consume the exception so the
            # future does not emit an unhandled-exception warning at GC time.
            future.add_done_callback(
                lambda completed: completed.exception()
                if not completed.cancelled() else None
            )
            self._entries[key] = _Entry(future=future, created_at=now)
            return ImageInvocationClaim(key=key, owner=True, future=future)

    def resolve(self, claim: ImageInvocationClaim, value: Any) -> None:
        with self._lock:
            entry = self._entries.get(claim.key)
            if entry is None or entry.future is not claim.future:
                return
            entry.completed_at = time.monotonic()
            if not claim.future.done():
                claim.future.set_result(value)

    def reject(self, claim: ImageInvocationClaim, exc: BaseException) -> None:
        with self._lock:
            entry = self._entries.get(claim.key)
            if entry is not None and entry.future is claim.future:
                self._entries.pop(claim.key, None)
            if not claim.future.done():
                if isinstance(exc, asyncio.CancelledError):
                    # 不能 cancel()：concurrent.futures.CancelledError 与
                    # asyncio.CancelledError 是同一个类，等待者在线程里调
                    # future.result() 时会被 anyio 当成"自己的任务被取消"，
                    # 导致连接正常的等待者无端失败。改用普通异常，让等待者
                    # 拿到可处理的错误（缓存条目已移除，重试可重新发起）。
                    claim.future.set_exception(
                        RuntimeError("image generation owner was cancelled; retry the request")
                    )
                else:
                    claim.future.set_exception(exc)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


image_invocation_cache = ImageInvocationCache()
