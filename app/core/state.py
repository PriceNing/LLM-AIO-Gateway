import hashlib
import json
import threading
import time

from app.config import get_default
from app.core.text import message_text
from app.core.types import InternalMessage


class TTLDict:
    """Thread-safe dict with TTL expiration and max-size eviction."""

    def __init__(self, ttl_seconds: int = 1800, max_size: int = 1000):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._data: dict = {}
        self._timestamps: dict = {}
        self._lock = threading.Lock()

    def _expired(self, key: str) -> bool:
        return time.time() - self._timestamps.get(key, 0) > self.ttl

    def _drop_locked(self, key: str) -> None:
        self._data.pop(key, None)
        self._timestamps.pop(key, None)

    def drop(self, key: str) -> None:
        with self._lock:
            self._drop_locked(key)

    def increment(self, key: str, delta: int = 1) -> int:
        with self._lock:
            if key in self._data:
                if self._expired(key):
                    self._drop_locked(key)
                    val = delta
                else:
                    val = self._data[key] + delta
            else:
                if len(self._data) >= self.max_size:
                    self._evict_expired()
                    if len(self._data) >= self.max_size and self._timestamps:
                        oldest = min(self._timestamps, key=lambda k: self._timestamps[k])
                        self._drop_locked(oldest)
                val = delta
            self._data[key] = val
            self._timestamps[key] = time.time()
            return val

    def reset(self, key: str) -> None:
        with self._lock:
            if key in self._data and not self._expired(key):
                self._data[key] = 0
                self._timestamps[key] = time.time()

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, ts in self._timestamps.items() if now - ts > self.ttl]
        for k in expired:
            self._drop_locked(k)

    def get(self, key: str, default=None):
        with self._lock:
            if key not in self._data:
                return default
            if self._expired(key):
                self._drop_locked(key)
                return default
            return self._data[key]

    def __setitem__(self, key: str, value) -> None:
        with self._lock:
            if len(self._data) >= self.max_size and key not in self._data:
                self._evict_expired()
                if len(self._data) >= self.max_size and self._timestamps:
                    oldest = min(self._timestamps, key=lambda k: self._timestamps[k])
                    self._drop_locked(oldest)
            self._data[key] = value
            self._timestamps[key] = time.time()

    def __getitem__(self, key: str):
        with self._lock:
            if key not in self._data:
                raise KeyError(key)
            if self._expired(key):
                self._drop_locked(key)
                raise KeyError(key)
            return self._data[key]

    def keys(self):
        with self._lock:
            self._evict_expired()
            return list(self._data.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


TOOL_ONLY_LIMIT = get_default("tool_only_limit", 20)

tool_only_turns = TTLDict(
    ttl_seconds=get_default("tool_only_turns_ttl", 600),
    max_size=get_default("tool_only_turns_max_size", 2000),
)
reasoning_cache = TTLDict(
    ttl_seconds=get_default("reasoning_cache_ttl", 1800),
    max_size=get_default("reasoning_cache_max_size", 1000),
)
reasoning_tool_cache = TTLDict(
    ttl_seconds=get_default("reasoning_cache_ttl", 1800),
    max_size=get_default("reasoning_cache_max_size", 1000),
)
reasoning_tool_global_cache = TTLDict(
    ttl_seconds=get_default("reasoning_cache_ttl", 1800),
    max_size=get_default("reasoning_cache_max_size", 1000),
)
response_chain_cache = TTLDict(
    ttl_seconds=get_default("reasoning_cache_ttl", 1800),
    max_size=get_default("reasoning_cache_max_size", 1000),
)


def ir_tool_message_count(messages: list[InternalMessage]) -> int:
    return sum(1 for msg in messages if any(part.kind == "tool_call" for part in msg.parts))


def ir_reasoning_message_count(messages: list[InternalMessage]) -> int:
    return sum(1 for msg in messages if any(part.kind == "reasoning" for part in msg.parts))


def remember_response_chain_key(response_id: str, conv_key: str) -> None:
    if response_id and conv_key:
        response_chain_cache[str(response_id)] = conv_key


def conversation_cache_key(api_key: str, messages: list, response_chain_id: str = "") -> str:
    if response_chain_id:
        chained_key = response_chain_cache.get(str(response_chain_id))
        if chained_key:
            return chained_key
    user_fingerprints = []
    for msg in messages:
        if isinstance(msg, InternalMessage):
            if msg.role != "user":
                continue
            text = ir_text_for_cache(msg)
            if not text and msg.parts:
                text = json.dumps([part.raw if part.raw is not None else part.kind for part in msg.parts], sort_keys=True, default=str)
            if text:
                user_fingerprints.append(text[:200])
            continue
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        text = message_text(content)
        if not text and isinstance(content, list):
            text = json.dumps(content, sort_keys=True, default=str)
        if text:
            user_fingerprints.append(text[:200])
    fingerprint = "\n---\n".join(user_fingerprints)[:2000]
    conv_hash = hashlib.md5(fingerprint.encode()).hexdigest()[:16]
    return f"{api_key}:{conv_hash}"


def ir_text_for_cache(message: InternalMessage) -> str:
    parts = []
    for part in message.parts:
        if part.kind == "text" and part.text:
            parts.append(part.text)
    return "\n".join(parts)


def remember_reasoning_content(conv_key: str, reasoning_content: str, tool_call_ids=None) -> None:
    if not reasoning_content:
        return
    reasoning_cache[conv_key] = reasoning_content
    ids = [str(tid) for tid in (tool_call_ids or []) if tid]
    if not ids:
        return
    tool_map = dict(reasoning_tool_cache.get(conv_key, {}) or {})
    for tid in ids:
        tool_map[tid] = reasoning_content
        reasoning_tool_global_cache[tid] = reasoning_content
    while len(tool_map) > 200:
        oldest = next(iter(tool_map))
        tool_map.pop(oldest, None)
    reasoning_tool_cache[conv_key] = tool_map


def reasoning_context(conv_key: str, messages: list[InternalMessage] | None = None) -> tuple[str | None, dict]:
    tool_map = reasoning_tool_cache.get(conv_key, {}) or {}
    if messages:
        tool_map = merge_global_reasoning_context(messages, tool_map)
    return reasoning_cache.get(conv_key), tool_map


def merge_global_reasoning_context(messages: list[InternalMessage], tool_map: dict) -> dict:
    merged = dict(tool_map or {})
    for msg in messages or []:
        if not isinstance(msg, InternalMessage):
            continue
        for part in msg.parts:
            if part.kind != "tool_result" or not part.tool_call_id or part.tool_call_id in merged:
                continue
            rc = reasoning_tool_global_cache.get(part.tool_call_id)
            if rc:
                merged[part.tool_call_id] = rc
    return merged
