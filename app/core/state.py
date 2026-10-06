import hashlib
import json
import threading
import time

from app.config import get_default
from app.core.runtime_config import register_runtime_hook
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

    def reserve(self, key: str, limit: int, count: int) -> int:
        """单次加锁内完成 check-and-add：返回实际预留量（0..count）。

        charge_image_generation_budget 旧实现是 get+increment 两次独立加锁，
        并发请求可同时通过检查造成预算双花（bug-2026-10-05 M-15）。
        """
        if count <= 0:
            return 0
        with self._lock:
            if key in self._data and self._expired(key):
                self._drop_locked(key)
            used = int(self._data.get(key) or 0) if key in self._data else 0
            allowed = min(count, max(0, int(limit) - used))
            if allowed:
                self._increment_locked(key, allowed)
            return allowed

    def _increment_locked(self, key: str, delta: int) -> int:
        if key in self._data:
            val = self._data[key] + delta
        else:
            if len(self._data) >= self.max_size:
                self._evict_expired_locked()
                if len(self._data) >= self.max_size and self._timestamps:
                    oldest = min(self._timestamps, key=lambda k: self._timestamps[k])
                    self._drop_locked(oldest)
            val = delta
        self._data[key] = val
        self._timestamps[key] = time.time()
        return val

    def increment(self, key: str, delta: int = 1) -> int:
        """累加计数并刷新时间戳——TTL 是**滑动**的（bug-2026-10-05 L-14）。

        对 tool-only 断路器的含义：持续 tool-only 循环会不断续期计数器，
        永不过期；达阈值后该会话一直被剥工具，直到出现文本回合（reset）或
        一个完整 TTL 静默期。这是有意语义（活跃循环不该被 TTL 洗白），
        改动前先确认调用方预期。
        """
        with self._lock:
            return self._increment_locked(key, delta)

    def reset(self, key: str) -> None:
        with self._lock:
            if key in self._data and not self._expired(key):
                self._data[key] = 0
                self._timestamps[key] = time.time()

    def _evict_expired(self) -> None:
        with self._lock:
            self._evict_expired_locked()

    def _evict_expired_locked(self) -> None:
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
                self._evict_expired_locked()
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
            self._evict_expired_locked()
            return list(self._data.keys())

    def configure(self, ttl_seconds: int, max_size: int) -> bool:
        """按新上限重配自ttl/容量；容量调小时立即淘汰到新的上限。

        返回是否发生变化。该属性本来就是普通字段，提供显式入口是为了让“配置写入
        后推送新值”有一个确定的地方，而不是在热路径上每次现读。
        """
        with self._lock:
            changed = self.ttl != ttl_seconds or self.max_size != max_size
            self.ttl = ttl_seconds
            self.max_size = max_size
            if len(self._data) > max_size:
                self._evict_expired_locked()
                while len(self._data) > max_size and self._timestamps:
                    oldest = min(self._timestamps, key=lambda k: self._timestamps[k])
                    self._drop_locked(oldest)
            return changed

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def tool_only_limit() -> int:
    """工具调用循环断路器阈值（每次现读，设置页改完立即生效）。"""
    try:
        return max(0, int(get_default("tool_only_limit", 20)))
    except (TypeError, ValueError):
        return 20


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
image_generation_budget = TTLDict(
    ttl_seconds=get_default("image_generation_budget_ttl", 3600),
    max_size=get_default("image_generation_budget_max_size", 1000),
)


# 各 TTL 缓存与其配置键的对应关系：(缓存, ttl 键, max_size 键, 默认 ttl, 默认容量)。
# 新增带 TTL/容量配置的缓存时在这里登记，不要另写一份推送逻辑。
_CACHE_LIMITS: tuple[tuple["TTLDict", str, str, int, int], ...] = (
    (tool_only_turns, "tool_only_turns_ttl", "tool_only_turns_max_size", 600, 2000),
    (reasoning_cache, "reasoning_cache_ttl", "reasoning_cache_max_size", 1800, 1000),
    (reasoning_tool_cache, "reasoning_cache_ttl", "reasoning_cache_max_size", 1800, 1000),
    (reasoning_tool_global_cache, "reasoning_cache_ttl", "reasoning_cache_max_size", 1800, 1000),
    (response_chain_cache, "reasoning_cache_ttl", "reasoning_cache_max_size", 1800, 1000),
    (image_generation_budget, "image_generation_budget_ttl", "image_generation_budget_max_size", 3600, 1000),
)


def _limit_value(defaults: dict, key: str, fallback: int) -> int:
    raw = defaults.get(key, fallback)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


def configure_caches(defaults: dict | None = None) -> list[str]:
    """把配置里的 ttl/max_size 推送到各 TTL 缓存，返回实际发生变化的缓存键。

    ttl 与容量在构造时定死，所以“改配置就生效”依赖调用方（启动时的
    core.runtime_config.apply_runtime_config、以及管理页写入后的同一次推送）。
    """
    defaults = defaults if isinstance(defaults, dict) else {}
    changed: list[str] = []
    for cache, ttl_key, size_key, ttl_fallback, size_fallback in _CACHE_LIMITS:
        if cache.configure(
            _limit_value(defaults, ttl_key, ttl_fallback),
            _limit_value(defaults, size_key, size_fallback),
        ):
            changed.append(f"{ttl_key}/{size_key}")
    return changed


def _reconfigure_caches_from_config(config: dict) -> None:
    """runtime hook：把整份配置的 defaults 段推送到各 TTL 缓存。"""
    defaults = config.get("defaults") if isinstance(config, dict) else None
    configure_caches(defaults if isinstance(defaults, dict) else {})


register_runtime_hook("state_caches", _reconfigure_caches_from_config)


def charge_image_generation_budget(conv_key: str, count: int) -> int:
    """Reserve up to ``count`` images from the conversation's time-window budget.

    Returns the number of images actually allowed (0..count).  This is the
    post-hoc cost control (modern-harness budget/rate-limit pattern): the
    bridge tool stays resident and the model decides when to call it; the
    budget only caps how many images may be generated per window, it never
    decides whether the tool is available.

    check-and-add 在单次锁内完成（TTLDict.reserve），并发请求不再能双花
    （bug-2026-10-05 M-15）。
    """
    if count <= 0:
        return 0
    limit = int(get_default("image_generation_budget_limit", 20) or 0)
    if not conv_key or limit <= 0:
        return count
    return image_generation_budget.reserve(conv_key, limit, count)


def refund_image_generation_budget(conv_key: str, count: int) -> None:
    """把未成功生成的张数退回会话预算（后端失败不应消耗窗口配额，M-15）。

    下限 0：退多于已扣（如 TTL 窗口已重置）时钉回 0 而不是变负数。
    已过期条目先丢弃再跳过退款：窗口重置后旧请求的退款对新窗口没有
    意义，写回只会把陈旧计数塞进下一个窗口（bug-2026-10-05 R-3）。
    """
    if not conv_key or count <= 0:
        return
    with image_generation_budget._lock:
        if conv_key in image_generation_budget._data and image_generation_budget._expired(conv_key):
            image_generation_budget._drop_locked(conv_key)
            return
        current = int(image_generation_budget._data.get(conv_key) or 0)
        if current:
            image_generation_budget._data[conv_key] = max(0, current - count)
            image_generation_budget._timestamps[conv_key] = time.time()


def ir_tool_message_count(messages: list[InternalMessage]) -> int:
    return sum(1 for msg in messages if any(part.kind == "tool_call" for part in msg.parts))


def ir_reasoning_message_count(messages: list[InternalMessage]) -> int:
    return sum(1 for msg in messages if any(part.kind == "reasoning" for part in msg.parts))


def _principal_of(conv_key: str) -> str:
    """取会话键的调用方段（第一个冒号前），用于全局缓存的隔离前缀。"""
    return conv_key.split(":", 1)[0] if conv_key else ""


def _global_reasoning_key(conv_key: str, tool_call_id: str) -> str:
    """reasoning_tool_global_cache 的键。

    旧实现用裸 tool_call_id 做键：不同用户/会话的上游若发出相同 id（尤其
    是旧版合成 id “前缀+秒级时间戳”形态）就会把别人的 reasoning 注入本会话
    （bug-2026-10-05 H-1）。现在用调用方段做前缀：同一 API key 内仍可跨
    conv_key 漂移回查（设计初衷），跨用户永久隔离。
    """
    return f"{_principal_of(conv_key)}\x00{tool_call_id}"


def remember_response_chain_key(response_id: str, conv_key: str) -> None:
    if response_id and conv_key:
        response_chain_cache[str(response_id)] = conv_key


def conversation_cache_key(api_key: str, messages: list, response_chain_id: str = "") -> str:
    if response_chain_id:
        chained_key = response_chain_cache.get(str(response_chain_id))
        # 链式命中也要过调用方段校验：response id 是 uuid 不可猜，旧实现直接
        # 返回存储键被报告评为低风险，但"绕过 api_key 段"这件事本身不该存在
        # （bug-2026-10-05 M-23 附带项）。principal 不匹配视为无效链。
        principal = hashlib.sha256(api_key.encode()).hexdigest()[:16] if api_key else "legacy"
        if chained_key and str(chained_key).split(":", 1)[0] == principal:
            return chained_key
    # 指纹：取前 3 条用户消息的全文摘要。
    # 旧实现把每条消息截到 200 字符、拼接后再截 2000 字符：前缀相同的两个
    # 会话（同一开场白很常见）得同键，reasoning/tool-only/生图预算跨会话
    # 串流（bug-2026-10-05 M-23）。全文摘要仍保持“会话内稳定”（历史只追加），
    # 但不再丢弃第 200 字符之后的内容。限定前 3 条是为了键在轮次间可重现。
    digests = []
    for msg in messages:
        if isinstance(msg, InternalMessage):
            if msg.role != "user":
                continue
            text = ir_text_for_cache(msg)
            if not text and msg.parts:
                text = json.dumps([part.raw if part.raw is not None else part.kind for part in msg.parts], sort_keys=True, default=str)
            if text:
                digests.append(hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest())
                if len(digests) >= 3:
                    break
            continue
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        text = message_text(content)
        if not text and isinstance(content, list):
            text = json.dumps(content, sort_keys=True, default=str)
        if text:
            digests.append(hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest())
            if len(digests) >= 3:
                break
    fingerprint = "\n---\n".join(digests)
    conv_hash = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    # 调用方段用 API key 的摘要而非明文：旧键把完整明文 key（共 50 字符）
    # 嵌进 conv_key，而多处日志打 conv_key[:60]/全文，密钥直接落盘
    # （bug-2026-10-05 M-5）。
    principal = hashlib.sha256(api_key.encode()).hexdigest()[:16] if api_key else "legacy"
    return f"{principal}:{conv_hash}"


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
        reasoning_tool_global_cache[_global_reasoning_key(conv_key, tid)] = reasoning_content
    while len(tool_map) > 200:
        oldest = next(iter(tool_map))
        tool_map.pop(oldest, None)
    reasoning_tool_cache[conv_key] = tool_map


def reasoning_context(conv_key: str, messages: list[InternalMessage] | None = None) -> tuple[str | None, dict]:
    tool_map = reasoning_tool_cache.get(conv_key, {}) or {}
    if messages:
        tool_map = merge_global_reasoning_context(conv_key, messages, tool_map)
    return reasoning_cache.get(conv_key), tool_map


def merge_global_reasoning_context(conv_key: str, messages: list[InternalMessage], tool_map: dict) -> dict:
    merged = dict(tool_map or {})
    if not _principal_of(conv_key):
        return merged
    for msg in messages or []:
        if not isinstance(msg, InternalMessage):
            continue
        for part in msg.parts:
            if part.kind != "tool_result" or not part.tool_call_id or part.tool_call_id in merged:
                continue
            rc = reasoning_tool_global_cache.get(_global_reasoning_key(conv_key, part.tool_call_id))
            if rc:
                merged[part.tool_call_id] = rc
    return merged
