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

    def claim(self, key: str, *, ttl_seconds: int, max_entries: int) -> ImageInvocationClaim:
        now = time.monotonic()
        ttl = max(1, int(ttl_seconds))
        limit = max(1, int(max_entries))
        with self._lock:
            expired = [
                entry_key for entry_key, entry in self._entries.items()
                if entry.completed_at is not None and now - entry.completed_at > ttl
            ]
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
                    claim.future.cancel()
                else:
                    claim.future.set_exception(exc)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


image_invocation_cache = ImageInvocationCache()
