import hashlib
import json
import time
import re
import queue
import threading
from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from typing import Optional
import anyio
from collections import deque
from app.database import (
    get_providers, find_user_by_api_key, get_routing_rules,
    increment_global_stats, increment_user_usage, get_db,
    parse_model_id, add_request_record,
)
from app.services.lite_llm import create_chat_completion, create_completion, create_chat_completion_stream, create_completion_stream
from app.services.preprocessing import preprocess_messages
from app.services.logger import get_logger
from app.config import get_default

_access_log = get_logger("access")
_error_log = get_logger("error")
_tool_log = get_logger("tool_calls")
_req_log = get_logger("request")
_app_log = get_logger("app")

router = APIRouter()

_STREAM_SENTINEL = object()  # Sentinel value to signal end of streaming queue

# Rolling log of recent requests for the admin stats dashboard
_request_log = deque(maxlen=get_default("request_log_max", 200))
_request_log_lock = threading.Lock()
_max_log_len = get_default("request_log_max", 200)


# ── 上游错误消息映射 ──

# (匹配模式, 友好消息) 列表，按优先级排列
_UPSTREAM_ERROR_MAP = [
    ("output new_sensitive (1027)", "内容被上游安全策略拦截（输出端）"),
    ("input new_sensitive", "内容被上游安全策略拦截（输入端）"),
    ("content_filter", "内容被上游安全策略拦截"),
    ("content_policy_violation", "内容违反上游使用策略"),
    ("safety_rating", "内容未通过上游安全评级"),
    ("No endpoints found that support image input", "该模型不支持图像输入，请在管理面板开启图像预处理"),
]


def _strip_billing_header(text):
    """Remove Anthropic billing header (x-anthropic-billing-header) injected by Claude Code.

    Claude Code 2.1.37+ injects a random `cch=xxxxx` value into the system prompt as an
    x-anthropic-billing-header text block.  Since `cch` changes on every request, this
    breaks DeepSeek's prefix cache (request body becomes non-deterministic).
    Strip lines matching the pattern so the system prompt stays stable across requests.

    Handles both string and Anthropic array format (list of content blocks).
    """
    if not text:
        if isinstance(text, list):
            return []
        return ""
    if isinstance(text, list):
        cleaned = []
        for block in text:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                stripped = re.sub(
                    r'^\s*x-anthropic-billing-header:\s*cc_version=[^;]+;\s*cc_entrypoint=[^;]+;\s*cch=[^;]+;?\s*$',
                    '', t, flags=re.MULTILINE
                ).strip()
                cleaned.append({"type": "text", "text": stripped} if stripped else None)
            else:
                cleaned.append(block)
        return [b for b in cleaned if b is not None]
    return re.sub(
        r'^\s*x-anthropic-billing-header:\s*cc_version=[^;]+;\s*cc_entrypoint=[^;]+;\s*cch=[^;]+;?\s*$',
        '', text, flags=re.MULTILINE
    ).strip()


def _friendly_error_msg(e: Exception) -> str:
    """将已知的上游错误映射为用户可读的友好消息，未匹配则返回原始错误。"""
    msg = str(e)
    for pattern, friendly in _UPSTREAM_ERROR_MAP:
        if pattern in msg:
            return f"{friendly}（原始: {msg[:120]}）"
    return msg


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return key
    return key[:4] + "..." + key[-4:]


def _fix_tool_args(tc_dict: dict) -> None:
    """Fix malformed tool-call arguments like MiniMax sending url:undefined."""
    func = tc_dict.get("function")
    if not func or not isinstance(func, dict):
        return
    args = func.get("arguments", "")
    if args and "undefined" in args:
        func["arguments"] = _sanitize_args(args)


def _sanitize_args(args: str) -> str:
    """Replace bare `undefined` values (MiniMax emits url:undefined in JSON)."""
    out = []
    in_str = False
    i = 0
    n = len(args)
    while i < n:
        c = args[i]
        if c == '"' and (i == 0 or args[i-1] != '\\'):
            in_str = not in_str
        if not in_str and args[i:i+9] == 'undefined':
            end = i + 9
            if end >= n or args[end] in ',}]\n\r\t ':
                out.append('""')
                i = end
                continue
        out.append(c)
        i += 1
    return ''.join(out)


def _log_request(username: str, api_key: str, model: str, provider_id: str,
                 endpoint: str, success: bool, tokens: int,
                 requested_model: str = "") -> None:
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "full_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "username": username,
        "api_key": _mask_key(api_key),
        "model": model,
        "requested_model": requested_model or model,
        "provider": provider_id or "",
        "endpoint": endpoint,
        "success": success,
        "tokens": tokens,
    }
    with _request_log_lock:
        _request_log.appendleft(entry)
    # Also write to structured access log
    if success:
        _access_log.info("[OK] %s user=%s model=%s provider=%s tokens=%d",
                         endpoint, username, model, provider_id or "-", tokens)
    else:
        _access_log.warning("[FAIL] %s user=%s model=%s provider=%s",
                           endpoint, username, model, provider_id or "-")
    # Write to persistent history for stats
    try:
        add_request_record(model=requested_model or model, username=username, success=success, tokens=tokens)
    except Exception:
        pass  # Never let DB errors block the response


def _log_request_body(username: str, model: str, endpoint: str, body: dict) -> None:
    """Log request metadata for debugging (truncated body, DEBUG level by default)."""
    _req_log.debug(
        "[%s] user=%s model=%s stream=%s tools=%d msgs=%d body_len=%d",
        endpoint, username, model,
        body.get("stream", False),
        len(body.get("tools", [])),
        len(body.get("messages", [])),
        len(json.dumps(body, ensure_ascii=False, default=str)),
    )


async def _maybe_preprocess(messages: list, model: str, provider_id: str = "", requested_model: str = ""):
    """Replace images with text descriptions if a vision preprocessor is configured.

    视觉描述决策依据 requested_model（未传则回退到 model），
    确保路由规则对用户透明——路由到哪个目标模型不影响是否生成描述。

    Returns (messages, modified: bool).
    """
    from app.services.lite_llm import _has_image_content
    has_img = _has_image_content(messages)
    check_model = requested_model or model
    _app_log.debug("[preprocess] CHECK req_model=%s target_model=%s has_image=%s msg_count=%d",
                   check_model, model, has_img, len(messages))
    # 用请求模型查 preprocessor（而非路由后的目标模型）
    mid = parse_model_id(check_model)
    with get_db() as db:
        if mid.provider_id:
            row = db.execute(
                "SELECT preprocessor FROM provider_models WHERE provider_id = ? AND model_id = ? AND enabled = 1 LIMIT 1",
                (mid.provider_id, mid.model_name)
            ).fetchone()
        else:
            row = db.execute(
                "SELECT preprocessor FROM provider_models WHERE model_id = ? AND enabled = 1 ORDER BY provider_id LIMIT 1",
                (mid.model_name,)
            ).fetchone()
    _app_log.info("[preprocess] DB lookup model='%s' -> row=%s", check_model, dict(row) if row else None)
    if not row or not row["preprocessor"]:
        if has_img:
            _app_log.warning("[preprocess] images detected for model=%s but preprocessor not enabled", check_model)
        return messages, False
    # Auto-detect first enabled preprocessor from config.json
    from app.config import get_config
    cfg = get_config().config.get("preprocessors", {})
    preprocessor_config = None
    preprocessor_id = ""
    for pid, pcfg in cfg.items():
        if isinstance(pcfg, dict) and pcfg.get("enabled", True):
            preprocessor_config = dict(pcfg)
            preprocessor_id = pid
            break
    if not preprocessor_config:
        _app_log.warning("[preprocess] no enabled preprocessor in config.json")
        return messages, False
    preprocessor_config["id"] = preprocessor_id
    had_images = _has_image_content(messages)
    result = await preprocess_messages(messages, preprocessor_config)
    return result, had_images


def get_request_log() -> list:
    with _request_log_lock:
        return list(_request_log)


def clear_request_log() -> None:
    with _request_log_lock:
        _request_log.clear()


def get_timeline_data() -> dict:
    """Aggregate requests by minute for timeline chart. Zero-fills gaps."""
    with _request_log_lock:
        snapshot = list(_request_log)
    if not snapshot:
        return {"labels": [], "success": [], "failed": []}
    buckets: dict[str, dict] = {}
    for entry in snapshot:
        minute = entry.get("full_time", entry["time"])[:16]
        if minute not in buckets:
            buckets[minute] = {"label": minute[-5:], "success": 0, "failed": 0}
        if entry["success"]:
            buckets[minute]["success"] += 1
        else:
            buckets[minute]["failed"] += 1
    sorted_keys = sorted(buckets.keys())
    # Zero-fill gaps between first and last minute
    if len(sorted_keys) >= 2:
        from datetime import datetime, timedelta
        def _parse_minute(s):
            for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"):
                try:
                    return datetime.strptime(s[:16].replace("T", " "), "%Y-%m-%d %H:%M")
                except ValueError:
                    continue
            return datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
        start = _parse_minute(sorted_keys[0])
        end = _parse_minute(sorted_keys[-1])
        cur = start
        while cur <= end:
            key = cur.strftime("%Y-%m-%d %H:%M")
            if key not in buckets:
                label = cur.strftime("%H:%M")
                buckets[key] = {"label": label, "success": 0, "failed": 0}
            cur += timedelta(minutes=1)
    sorted_buckets = sorted(buckets.items(), key=lambda x: x[0])
    return {
        "labels": [b["label"] for _, b in sorted_buckets],
        "success": [b["success"] for _, b in sorted_buckets],
        "failed": [b["failed"] for _, b in sorted_buckets],
    }


def get_model_distribution() -> dict:
    """Model usage distribution for pie chart."""
    with _request_log_lock:
        snapshot = list(_request_log)
    counts: dict[str, int] = {}
    for entry in snapshot:
        model = entry["model"]
        counts[model] = counts.get(model, 0) + 1
    sorted_models = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return {
        "labels": [m for m, _ in sorted_models],
        "counts": [c for _, c in sorted_models],
    }


def get_model_stats() -> dict:
    """Aggregate per-model stats from recent request log."""
    with _request_log_lock:
        snapshot = list(_request_log)
    models = {}
    for entry in snapshot:
        mid = entry["model"]
        if mid not in models:
            models[mid] = {"total": 0, "failed": 0, "tokens": 0}
        models[mid]["total"] += 1
        if not entry["success"]:
            models[mid]["failed"] += 1
        models[mid]["tokens"] += entry["tokens"]
    return models


def get_timeline_model_data() -> dict:
    """Per-model per-minute breakdown from request log for stacked bar chart. Zero-fills gaps."""
    with _request_log_lock:
        snapshot = list(_request_log)
    if not snapshot:
        return {"labels": [], "models": [], "calls": [], "tokens": []}
    buckets: dict[str, dict] = {}
    for entry in snapshot:
        minute = entry.get("full_time", entry["time"])[:16]
        if minute not in buckets:
            buckets[minute] = {}
        model = entry["model"]
        if model not in buckets[minute]:
            buckets[minute][model] = {"total": 0, "tokens": 0}
        buckets[minute][model]["total"] += 1
        buckets[minute][model]["tokens"] += entry["tokens"]
    sorted_keys = sorted(buckets.keys())
    # Zero-fill gaps between first and last minute
    if len(sorted_keys) >= 2:
        from datetime import datetime, timedelta
        def _parse_minute_ts(s):
            return datetime.strptime(s[:16].replace("T", " "), "%Y-%m-%d %H:%M")
        start = _parse_minute_ts(sorted_keys[0])
        end = _parse_minute_ts(sorted_keys[-1])
        cur = start
        while cur <= end:
            key = cur.strftime("%Y-%m-%d %H:%M")
            if key not in buckets:
                buckets[key] = {}
            cur += timedelta(minutes=1)
        sorted_keys = sorted(buckets.keys())
    all_models = sorted({m for b in buckets.values() for m in b})
    return {
        "labels": [k[-5:] for k in sorted_keys],
        "models": all_models,
        "calls": [[buckets[k].get(m, {}).get("total", 0) for k in sorted_keys] for m in all_models],
        "tokens": [[buckets[k].get(m, {}).get("tokens", 0) for k in sorted_keys] for m in all_models],
    }


class TTLDict:
    """Dict with TTL-based expiration and max-size eviction. Thread-safe.

    Prevents unbounded memory growth for caches of short-lived conversations.
    Entries older than ttl_seconds are lazily evicted on access; when max_size
    is exceeded the oldest entry is evicted.
    """

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
        """Public method to remove a cache entry."""
        with self._lock:
            self._drop_locked(key)

    def increment(self, key: str, delta: int = 1) -> int:
        """Atomically increment a counter and return the new value."""
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
        """Atomically reset a counter to 0."""
        with self._lock:
            if key in self._data and not self._expired(key):
                self._data[key] = 0
                self._timestamps[key] = time.time()

    def _evict_expired(self) -> None:
        """Internal — caller must hold _lock."""
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


# Tool-call circuit breaker: track consecutive tool-only turns per conversation
_tool_only_turns = TTLDict(
    ttl_seconds=get_default("tool_only_turns_ttl", 600),
    max_size=get_default("tool_only_turns_max_size", 2000)
)
TOOL_ONLY_LIMIT = get_default("tool_only_limit", 20)

# Reasoning content cache: DeepSeek requires reasoning_content echoed back in multi-turn
# Keyed by "{api_key}:{conversation_hash}", stores reasoning_content for replay
_reasoning_cache = TTLDict(
    ttl_seconds=get_default("reasoning_cache_ttl", 1800),
    max_size=get_default("reasoning_cache_max_size", 1000)
)




def _debug_msg_types(messages: list) -> None:
    """Log the content types present in each message to diagnose image format issues."""
    for i, msg in enumerate(messages[-4:]):  # Last 4 messages only
        content = msg.get("content")
        role = msg.get("role", "?")
        if isinstance(content, list):
            types = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("type", "?")
                    if t == "text" and isinstance(part.get("text"), str) and len(part["text"]) > 100:
                        types.append(f"text(len={len(part['text'])})")
                    elif t == "text":
                        types.append(f"text={part.get('text','')[:50]}")
                    elif t in ("image_url", "input_image"):
                        url = part.get("image_url", "")
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        types.append(f"{t}(url_prefix={str(url)[:60]})")
                    else:
                        types.append(f"{t}")
                elif isinstance(part, str):
                    types.append(f"str(len={len(part)})")
            _app_log.debug("[debug] msg[%d] role=%s content_types=%s", i, role, types)
        elif isinstance(content, str):
            _app_log.debug("[debug] msg[%d] role=%s content=str(len=%d) prefix=%s",
                         i, role, len(content), content[:80])
        else:
            _app_log.debug("[debug] msg[%d] role=%s content_type=%s", i, role, type(content).__name__)


def _attr(obj, key: str, default=None):
    """从对象（getattr）或 dict（.get）中取值，兼容两种类型。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _conversation_cache_key(api_key: str, messages: list) -> str:
    """Build a cache key that isolates different conversations sharing the same API key."""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if user_msgs:
        first_content = user_msgs[0].get("content")
        first_content_list = None
        if isinstance(first_content, list):
            # Multimodal content — extract text parts for fingerprint
            first_content_list = first_content
            text_parts = []
            for p in first_content:
                if isinstance(p, dict) and p.get("type") == "text":
                    text_parts.append(p.get("text", ""))
                elif isinstance(p, str):
                    text_parts.append(p)
            first_content = "\n".join(text_parts)
            if not first_content and first_content_list:
                first_content = json.dumps(first_content_list, sort_keys=True, default=str)[:200]
        fingerprint = (first_content or "")[:200]
    else:
        fingerprint = ""
    conv_hash = hashlib.md5(fingerprint.encode()).hexdigest()[:16]
    return f"{api_key}:{conv_hash}"

def verify_api_key(authorization: Optional[str] = Header(None)) -> tuple[dict, dict]:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")

    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")

    user_match = find_user_by_api_key(token)
    if user_match:
        return user_match

    raise HTTPException(status_code=401, detail="Invalid API key")

def allowed_models_for(user: dict, api_key: dict) -> list:
    # Only key-level allowed_models matters. User is just enable/disable.
    key_models = api_key.get("allowed_models")
    if key_models is None:
        return ["*"]  # not configured → unrestricted
    if "*" in key_models:
        return ["*"]
    return key_models  # explicit list, empty = deny all

def ensure_model_allowed(user: dict, api_key: dict, model: str) -> None:
    allowed = allowed_models_for(user, api_key)
    if "*" in allowed:
        return
    # ModelId.__eq__ 自动处理复合/简单四种方向的匹配
    if parse_model_id(model) in allowed:
        return
    raise HTTPException(status_code=403, detail=f"Model '{model}' is not allowed for this API key")

def usage_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        "prompt_cache_hit_tokens": getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
        "prompt_cache_miss_tokens": getattr(usage, "prompt_cache_miss_tokens", 0) or 0,
    }

@router.get("/models")
def list_models(authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)
    allowed = allowed_models_for(user, api_key)
    models = []

    for provider in get_providers():
        if provider.get("enabled"):
            for model in provider.get("models", []):
                if model.get("enabled"):
                    composite_id = f"{provider['id']}/{model['id']}"
                    # 检查允许列表：支持复合 ID、简单 model_id、通配符
                    if "*" not in allowed and model["id"] not in allowed and composite_id not in allowed:
                        continue
                    entry = {
                        "id": composite_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": provider["name"],
                        "provider": provider["id"]
                    }
                    # 启用了视觉预处理器的模型声明支持图片，让客户端(Codex/OpenCode等)
                    # 愿意发送图片数据，由网关拦截后交给视觉模型描述。
                    # 尝试多种常用字段名，因为不同客户端检查不同的 key。
                    if model.get("preprocessor"):
                        entry["supports_vision"] = True
                        entry["image_support"] = True
                        entry["multimodal"] = True
                    models.append(entry)

    return {"object": "list", "data": models}

async def _iter_stream_async(stream_func):
    """Iterate a sync generator in a background thread to avoid blocking the event loop.

    Routes stream chunks through a thread-safe queue so the event loop thread
    never blocks on network I/O, allowing concurrent streaming requests.

    When the client disconnects, we close the stream generator to release the
    underlying HTTP connection, then use PyThreadState_SetAsyncExc to forcibly
    interrupt the background thread in case it's stuck in a blocking read (e.g.
    llamacpp prefill taking minutes).
    """
    import ctypes

    chunk_queue = queue.Queue()
    error = None
    done = threading.Event()
    cancel = threading.Event()
    stream_gen = None
    bg_thread = None

    def _run():
        nonlocal error, stream_gen
        try:
            stream_gen = stream_func()
            _chunk_idx = 0
            for chunk in stream_gen:
                if cancel.is_set():
                    break
                _chunk_idx += 1
                chunk_queue.put(chunk)
            _app_log.debug("[_iter_stream_async] generator finished, total_chunks=%d", _chunk_idx)
        except GeneratorExit:
            pass
        except Exception as e:
            import traceback as _tb
            try:
                _error_log.error("[_iter_stream_async] type=%s msg=%s", type(e).__name__, str(e)[:200])
                _error_log.error("[_iter_stream_async] %s", _tb.format_exc())
            except Exception:
                pass
            error = e
        finally:
            if stream_gen is not None:
                try:
                    stream_gen.close()
                except Exception:
                    pass
            chunk_queue.put(_STREAM_SENTINEL)
            done.set()

    bg_thread = threading.Thread(target=_run, daemon=True)
    bg_thread.start()

    try:
        while True:
            try:
                chunk = chunk_queue.get(timeout=0.01)
                if chunk is _STREAM_SENTINEL:
                    break
                yield chunk
            except queue.Empty:
                if done.is_set():
                    break
                await anyio.sleep(0)

        if error:
            raise error
    finally:
        cancel.set()
        # Force-interrupt the background thread if it's stuck in a blocking
        # read (e.g. llamacpp processing a large prompt). This causes httpx's
        # read() to abort via GeneratorExit and the HTTP connection to close.
        if bg_thread is not None and bg_thread.is_alive():
            try:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(bg_thread.ident),
                    ctypes.py_object(GeneratorExit)
                )
            except Exception:
                pass


async def _stream_chat(model, messages, provider_id, temperature, max_tokens, username, api_key_value, requested_model="", conv_key="", **extra):
    chat_id = f"chatcmpl-{int(time.time())}"
    if not conv_key:
        conv_key = _conversation_cache_key(api_key_value, messages)

    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    error_msg = None
    has_text_content = False
    has_tool_calls = False
    accumulated_reasoning = ""
    think_stripped = False  # Set to True once </think> is seen
    think_buf = ""          # Buffer content while inside <think> block
    cache_hit = 0
    cache_miss = 0

    # Tool-call circuit breaker: strip tools if too many consecutive tool-only turns
    if _tool_only_turns.get(conv_key, 0) >= TOOL_ONLY_LIMIT:
        extra.pop("tools", None)
        extra.pop("tool_choice", None)

    try:
        stream_func = lambda: create_chat_completion_stream(
            model=model,
            messages=messages,
            provider_id=provider_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra
        )

        async for chunk in _iter_stream_async(stream_func):
            choice = chunk.choices[0] if chunk.choices else None
            delta = {}
            finish_reason = None

            if choice:
                finish_reason = getattr(choice, "finish_reason", None) or None
                message = getattr(choice, "delta", None)
                if message:
                    content = getattr(message, "content", None)
                    if content:
                        if not think_stripped:
                            think_buf += content
                            if '</think>' in think_buf:
                                # <think> block(s) completed — extract thinking, flush rest
                                think_buf, think_content = _extract_and_strip_think(think_buf)
                                if think_content:
                                    accumulated_reasoning = think_content
                                think_stripped = True
                                if think_buf:
                                    delta["content"] = think_buf
                                    has_text_content = True
                            elif '<think>' in think_buf or think_buf.lstrip().startswith('<think'):
                                # If <think> never closes after buffering much content,
                                # treat it as plain text so users aren't stuck waiting.
                                if len(think_buf) >= 200:
                                    think_buf = think_buf.replace('<think>', '', 1)
                                    think_stripped = True
                                    delta["content"] = think_buf
                                    has_text_content = True
                                    think_buf = ""
                                # else: Still inside a <think> block — suppress for now
                            else:
                                # No <think> tag — not a thinking-mode response.
                                # Flush buffer as regular text immediately.
                                think_stripped = True
                                delta["content"] = think_buf
                                has_text_content = True
                                think_buf = ""
                        else:
                            delta["content"] = content
                            has_text_content = True
                    reasoning = getattr(message, "reasoning_content", None)
                    if reasoning:
                        delta["reasoning_content"] = reasoning
                        accumulated_reasoning += reasoning
                    role = getattr(message, "role", None)
                    if role:
                        delta["role"] = role
                    tool_calls = getattr(message, "tool_calls", None)
                    if tool_calls:
                        _tool_log.info("[_stream_chat] raw tool_calls count=%d model=%s", len(tool_calls), model)
                        serialized = []
                        for tc in tool_calls:
                            _tool_log.debug("[_stream_chat] raw tc type=%s repr=%s", type(tc).__name__, repr(tc))
                            if hasattr(tc, "model_dump"):
                                tc_dict = tc.model_dump(exclude_none=True)
                            elif isinstance(tc, dict):
                                tc_dict = dict(tc)
                            else:
                                tc_dict = {
                                    "index": getattr(tc, "index", 0),
                                    "id": getattr(tc, "id", None),
                                    "type": getattr(tc, "type", "function"),
                                    "function": {
                                        "name": getattr(tc.function, "name", None) if hasattr(tc, "function") and tc.function else None,
                                        "arguments": getattr(tc.function, "arguments", "") if hasattr(tc, "function") and tc.function else ""
                                    }
                                }
                            _tool_log.info("[_stream_chat] after dict convert: %s", json.dumps(tc_dict, ensure_ascii=False, default=str))
                            # Coerce id to str (MiniMax returns bare integers)
                            if tc_dict.get("id") is not None and not isinstance(tc_dict["id"], str):
                                tc_dict["id"] = str(tc_dict["id"])
                            tc_idx = int(tc_dict.get("index", 0))
                            # Filter out spurious tool-calls from non-standard content
                            # blocks (e.g. MiniMax "thinking") misread as empty tool uses.
                            # Only filter when index<0.  id=None with non-empty arguments
                            # is a valid arguments delta chunk — must NOT be filtered.
                            if tc_idx < 0:
                                _tool_log.info("[_stream_chat] FILTERED spurious: id=%s idx=%s", tc_dict.get("id"), tc_idx)
                                continue
                            # Fix malformed tool-call arguments — MiniMax may emit
                            # invalid JSON with bare `undefined` values (e.g. url:undefined)
                            _fix_tool_args(tc_dict)
                            _tool_log.info("[_stream_chat] after _fix_tool_args: %s", json.dumps(tc_dict, ensure_ascii=False, default=str))
                            serialized.append(tc_dict)
                        if serialized:
                            has_tool_calls = True
                            delta["tool_calls"] = serialized
                            _tool_log.info("[_stream_chat] SSE delta payload: %s", json.dumps(serialized, ensure_ascii=False, default=str))
                        else:
                            _tool_log.info("[_stream_chat] all tool_calls filtered out, no tool_calls in delta")

            sse_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason
                }]
            }
            yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

            if hasattr(chunk, "usage") and chunk.usage:
                total_tokens = getattr(chunk.usage, "total_tokens", 0) or 0
                input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
                cache_hit = getattr(chunk.usage, "prompt_cache_hit_tokens", 0) or 0
                cache_miss = getattr(chunk.usage, "prompt_cache_miss_tokens", 0) or 0
        if accumulated_reasoning:
            _reasoning_cache[conv_key] = accumulated_reasoning
            _app_log.info("[chat_stream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d", conv_key[:40], len(accumulated_reasoning), cache_hit, cache_miss)
        else:
            _app_log.info("[chat_stream] no reasoning accumulated (think_stripped=%s)", think_stripped)

        # Update tool-only counter for circuit breaker
        if has_tool_calls and not has_text_content:
            _tool_only_turns.increment(conv_key)
        else:
            _tool_only_turns.reset(conv_key)

        _log_request(username, api_key_value, model, provider_id or "", "chat_completions", True, total_tokens, requested_model)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, total_tokens)
        increment_global_stats(success=True)

    except Exception as e:
        error_msg = _friendly_error_msg(e)
        _error_log.error("[chat_stream] %s", str(e))
        _log_request(username, api_key_value, requested_model, provider_id or "", "chat_completions", False, 0, requested_model)
        _tool_only_turns.reset(conv_key)
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)

    finally:
        if error_msg:
            yield f"data: {json.dumps({'error': {'message': error_msg, 'type': 'server_error'}})}\n\n"
        else:
            # When thinking mode consumed all tokens before generating visible content,
            # fall back to reasoning_content as the response so the client shows something.
            if not has_text_content and accumulated_reasoning:
                yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': accumulated_reasoning}, 'finish_reason': 'stop'}]})}\n\n"
            # If think_buf has buffered content but </think> was never seen (model
            # doesn't use <think> tags — e.g. DeepSeek via Anthropic endpoint), flush
            # the buffer as regular text so the client receives the response.
            elif not has_text_content and think_buf:
                yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': think_buf}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"


async def _stream_completions(model, prompt, provider_id, temperature, max_tokens, username, api_key_value, requested_model="", **extra):
    """Stream a text completion response as SSE."""
    cmpl_id = f"cmpl-{int(time.time())}"

    total_tokens = 0
    error_msg = None

    try:
        stream_func = lambda: create_completion_stream(
            model=model,
            prompt=prompt,
            provider_id=provider_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra
        )

        async for chunk in _iter_stream_async(stream_func):
            choice = chunk.choices[0] if chunk.choices else None
            text = ""
            finish_reason = None

            if choice:
                finish_reason = getattr(choice, "finish_reason", None) or None
                text = _strip_think_tags(getattr(choice, "text", "") or "")

            sse_data = {
                "id": cmpl_id,
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": text,
                    "finish_reason": finish_reason
                }]
            }
            yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

            if hasattr(chunk, "usage") and chunk.usage:
                total_tokens = getattr(chunk.usage, "total_tokens", 0) or 0
                cache_hit = getattr(chunk.usage, "prompt_cache_hit_tokens", 0) or 0
                cache_miss = getattr(chunk.usage, "prompt_cache_miss_tokens", 0) or 0

        _log_request(username, api_key_value, model, provider_id or "", "completions", True, total_tokens, requested_model)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, total_tokens)
        increment_global_stats(success=True)

    except Exception as e:
        error_msg = _friendly_error_msg(e)
        _error_log.error("[completions_stream] %s", str(e))
        _log_request(username, api_key_value, requested_model, provider_id or "", "completions", False, 0, requested_model)
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)

    finally:
        if error_msg:
            yield f"data: {json.dumps({'error': {'message': error_msg, 'type': 'server_error'}})}\n\n"
        else:
            yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)

    body = await request.json()
    model = body.get("model")
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = get_default("max_tokens", 16384)
    provider_id = body.get("provider_id")
    stream = body.get("stream", False)

    username = user.get("username", "legacy")
    api_key_value = api_key.get("key", "")
    _log_request_body(username, model, "chat", body)

    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    ensure_model_allowed(user, api_key, model)

    # Debug: log content types in messages to diagnose image format issues
    _debug_msg_types(messages)

    requested_model = model
    route_model, route_provider = _apply_routing_rules(username, api_key_value, model, model)
    if route_model != model:
        _app_log.info("[chat] ROUTED model=%s -> %s", model, route_model)
        model = route_model
    if route_provider:
        provider_id = route_provider

    # Merge consecutive same-role messages for providers that require alternation
    messages = _normalize_messages(messages)

    # Compute conversation cache key BEFORE preprocessing, since preprocessing
    # modifies the first user message (images → text) which changes the key.
    # All cache operations (reasoning, tool counter) must use this stable key.
    conv_key = _conversation_cache_key(api_key_value, messages)

    # Preprocessor: replace images with text
    msg, modified = await _maybe_preprocess(messages, model, provider_id, requested_model=requested_model)
    messages = msg
    if modified:
        _reasoning_cache.drop(conv_key)

    try:
        allowed_params = {
            "top_p", "presence_penalty", "frequency_penalty", "stop",
            "tools", "tool_choice", "response_format", "user"
        }
        extra = {key: body[key] for key in allowed_params if key in body}

        # DeepSeek reasoning_content cache injection for multi-turn continuity
        cached_rc = _reasoning_cache.get(conv_key)
        if cached_rc is not None:
            injected = 0
            for msg in messages:
                # Only inject reasoning_content into assistant messages that made tool calls.
                # Per DeepSeek docs: reasoning_content only needs to be passed back when
                # the previous turn included a tool call. Messages without tool_calls don't
                # need it — injecting would only break prefix cache.
                if (msg.get("role") == "assistant"
                    and not msg.get("reasoning_content")
                    and msg.get("tool_calls")):
                    msg["reasoning_content"] = cached_rc
                    injected += 1
            _app_log.info("[chat] INJECTED rc key=%s len=%d into %d asst-with-tool msgs", conv_key[:40], len(cached_rc), injected)
        else:
            # First turn: inject empty reasoning_content only into assistant messages
            # with tool_calls, to satisfy DeepSeek's requirement without breaking cache.
            for msg in messages:
                if (msg.get('role') == 'assistant'
                    and 'reasoning_content' not in msg
                    and msg.get('tool_calls')):
                    msg['reasoning_content'] = ''

        # Tool-call circuit breaker: strip tools if too many consecutive tool-only turns
        if _tool_only_turns.get(conv_key, 0) >= TOOL_ONLY_LIMIT:
            extra.pop("tools", None)
            extra.pop("tool_choice", None)

        if stream:
            return StreamingResponse(
                _stream_chat(
                    model=model,
                    messages=messages,
                    provider_id=provider_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    username=username,
                    api_key_value=api_key_value,
                    requested_model=requested_model,
                    conv_key=conv_key,
                    **extra
                ),
                media_type="text/event-stream"
            )

        response = await anyio.to_thread.run_sync(
            lambda: create_chat_completion(
                model=model,
                messages=messages,
                provider_id=provider_id,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra
            )
        )
        choice = response.choices[0]
        message = getattr(choice, "message", {})
        content = _strip_think_tags(getattr(message, "content", None) or "")
        tool_calls = getattr(message, "tool_calls", None)
        reasoning_content = getattr(message, "reasoning_content", None)

        # Store reasoning_content for next turn
        if reasoning_content:
            _reasoning_cache[conv_key] = reasoning_content
            usage = usage_dict(response)
            _app_log.info("[chat_nonstream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d",
                          conv_key[:40], len(reasoning_content),
                          usage.get("prompt_cache_hit_tokens", 0), usage.get("prompt_cache_miss_tokens", 0))

        # Update tool-only counter for circuit breaker
        valid_tool_calls = []
        if tool_calls:
            _tool_log.info("[chat_completions] raw tool_calls count=%d model=%s", len(tool_calls), model)
            for tc in tool_calls:
                _tool_log.debug("[chat_completions] raw tc type=%s repr=%s", type(tc).__name__, repr(tc))
                tc_id = getattr(tc, "id", None) if hasattr(tc, "id") else tc.get("id")
                tc_idx = getattr(tc, "index", 0) if hasattr(tc, "index") else tc.get("index", 0)
                if int(tc_idx) >= 0:
                    valid_tool_calls.append(tc)
                else:
                    _tool_log.info("[chat_completions] FILTERED spurious in valid check: id=%s idx=%s", tc_id, tc_idx)
            tool_calls = valid_tool_calls if valid_tool_calls else None
        if tool_calls and not content:
            _tool_only_turns.increment(conv_key)
        else:
            _tool_only_turns.reset(conv_key)

        usage = usage_dict(response)
        _log_request(username, api_key_value, model, provider_id or "", "chat_completions", True, usage.get("total_tokens", 0), requested_model)
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, usage.get("total_tokens", 0))
        # When thinking/reasoning mode consumes all tokens before generating content,
        # fall back to reasoning_content as the visible response.
        if not content and reasoning_content:
            content = reasoning_content
        resp_msg = {"role": "assistant", "content": content}
        if reasoning_content:
            resp_msg["reasoning_content"] = reasoning_content
        if tool_calls:
            serialized = []
            for tc in tool_calls:
                if hasattr(tc, "model_dump"):
                    tc_dict = tc.model_dump(exclude_none=True)
                elif isinstance(tc, dict):
                    tc_dict = dict(tc)
                else:
                    tc_dict = {
                        "index": getattr(tc, "index", 0),
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(tc.function, "name", None) if hasattr(tc, "function") and tc.function else None,
                            "arguments": getattr(tc.function, "arguments", "") if hasattr(tc, "function") and tc.function else ""
                        }
                    }
                _tool_log.info("[chat_completions] after dict convert: %s", json.dumps(tc_dict, ensure_ascii=False, default=str))
                if tc_dict.get("id") is not None and not isinstance(tc_dict["id"], str):
                    tc_dict["id"] = str(tc_dict["id"])
                if int(tc_dict.get("index", 0)) < 0:
                    _tool_log.info("[chat_completions] FILTERED spurious: id=%s idx=%s", tc_dict.get("id"), tc_dict.get("index", 0))
                    continue
                _fix_tool_args(tc_dict)
                _tool_log.info("[chat_completions] after _fix_tool_args: %s", json.dumps(tc_dict, ensure_ascii=False, default=str))
                serialized.append(tc_dict)
            if serialized:
                resp_msg["tool_calls"] = serialized
                _tool_log.info("[chat_completions] FINAL tool_calls: %s", json.dumps(serialized, ensure_ascii=False, default=str))
            else:
                _tool_log.info("[chat_completions] all tool_calls filtered out")
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": resp_msg,
                "finish_reason": getattr(choice, "finish_reason", "stop") or "stop"
            }],
            "usage": usage
        }
    except Exception as e:
        _error_log.error("[chat] %s", str(e))
        _log_request(username, api_key_value, requested_model, provider_id or "", "chat_completions", False, 0, requested_model)
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        raise HTTPException(status_code=500, detail=_friendly_error_msg(e))

@router.post("/completions")
async def completions(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)

    body = await request.json()
    model = body.get("model")
    prompt = body.get("prompt", "")
    provider_id = body.get("provider_id")
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = get_default("max_tokens", 16384)

    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    ensure_model_allowed(user, api_key, model)

    username = user.get("username", "legacy")
    _log_request_body(username, model, "completions", body)
    api_key_value = api_key.get("key", "")
    requested_model = model
    route_model, route_provider = _apply_routing_rules(username, api_key_value, model, model)
    if route_model != model:
        _app_log.info("[completions] ROUTED model=%s -> %s", model, route_model)
        model = route_model
    if route_provider:
        provider_id = route_provider

    # Preprocessor: wrap prompt for image processing
    msgs = [{"role": "user", "content": prompt}]
    msgs, _ = await _maybe_preprocess(msgs, model, provider_id, requested_model=requested_model)
    prompt = msgs[0].get("content", "") if msgs else prompt

    try:
        if stream:
            return StreamingResponse(
                _stream_completions(
                    model=model,
                    prompt=prompt,
                    provider_id=provider_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    username=username,
                    api_key_value=api_key_value,
                    requested_model=requested_model
                ),
                media_type="text/event-stream"
            )

        response = await anyio.to_thread.run_sync(
            lambda: create_completion(
                model=model,
                prompt=prompt,
                provider_id=provider_id,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )
        choice = response.choices[0]
        message = getattr(choice, "message", {})

        usage = usage_dict(response)
        _log_request(username, api_key_value, model, provider_id or "", "completions", True, usage.get("total_tokens", 0), requested_model)
    
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, usage.get("total_tokens", 0))
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "text": getattr(message, "content", ""),
                "index": 0,
                "finish_reason": getattr(choice, "finish_reason", "stop") or "stop"
            }],
            "usage": usage
        }
    except Exception as e:
        _log_request(username, api_key_value, "-", provider_id or "", "completions", False, 0, requested_model)

        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", str(e))
        raise HTTPException(status_code=500, detail=_friendly_error_msg(e))

# ── Anthropic Messages API conversion ──

def _anthropic_content_to_openai(content) -> list:
    """Convert Anthropic content blocks to OpenAI chat content format."""
    if isinstance(content, str):
        return [content]  # plain string, return as-is for backward compat
    if not isinstance(content, list):
        return [str(content)]
    parts = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
        elif block.get("type") == "text":
            text = block.get("text", "")
            if text:  # 过滤空文本块，避免下游模型拒绝
                parts.append(text)
        elif block.get("type") == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                media = source.get("media_type", "image/png")
                data = source.get("data", "")
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media};base64,{data}"}
                })
        elif block.get("type") == "tool_use":
            # Tool_use in conversation history means it was an assistant tool call
            # Already handled by _anthropic_to_openai_messages, skip here
            pass
        elif block.get("type") == "tool_result":
            # Tool_result in conversation — handled by message role conversion
            pass
    # Return string if only text, list if has images
    if all(isinstance(p, str) for p in parts):
        return parts  # will be joined if needed
    return parts


def _anthropic_to_openai_messages(anthropic_msgs: list, system_prompt: str = "") -> tuple:
    """Convert Anthropic Messages API input to OpenAI Chat Completions messages.
    Also extracts tool definitions from the request for later use.
    Returns (openai_messages, has_tools)."""
    openai_msgs = []
    has_tools = False
    if system_prompt:
        # Anthropic system field 可以是字符串或 content block 数组（含 cache_control 等）
        if isinstance(system_prompt, list):
            text_parts = []
            for block in system_prompt:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        text_parts.append(text)
            system_prompt = "\n".join(text_parts)
        if system_prompt:
            openai_msgs.append({"role": "system", "content": system_prompt})
    for m in anthropic_msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            openai_content = None
            tool_calls = []
            thinking_parts = []
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            text_parts.append(text)
                    elif block.get("type") == "tool_use":
                        has_tools = True
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {}))
                            }
                        })
                    elif block.get("type") == "thinking":
                        th_text = block.get("thinking", "")
                        if th_text:
                            thinking_parts.append(th_text)
                openai_content = "\n".join(text_parts) or None
            else:
                openai_content = str(content) if content else None
            msg = {"role": "assistant", "content": openai_content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            # Preserve reasoning_content for DeepSeek thinking mode multi-turn continuity.
            # Anthropic format stores thinking in content blocks; OpenAI format uses
            # the reasoning_content field. Without this, DeepSeek re-thinks from scratch
            # and may enter empty-output loops (content_chars=0, finish_reason=stop).
            rc = m.get("reasoning_content")
            if rc:
                msg["reasoning_content"] = rc
            elif thinking_parts:
                msg["reasoning_content"] = "\n".join(thinking_parts)
            openai_msgs.append(msg)
        elif role == "user":
            parts = _anthropic_content_to_openai(content)
            # Check for tool_result blocks embedded in user messages (Anthropic format)
            tool_results = []
            other_parts = []
            for block in (content if isinstance(content, list) else []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_results.append(block)
                else:
                    other_parts.append(block)
            # Tool results MUST come first (before user text) — DeepSeek and other
            # strict providers require tool messages immediately after assistant tool_calls.
            # See: "An assistant message with 'tool_calls' must be followed by tool messages"
            for tr in tool_results:
                tool_use_id = tr.get("tool_use_id", "")
                result_content = tr.get("content", "")
                if isinstance(result_content, list):
                    text_parts = []
                    image_parts = []
                    for block in result_content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    text_parts.append(text)
                            elif block.get("type") == "image":
                                source = block.get("source", {})
                                if source.get("type") == "base64":
                                    media = source.get("media_type", "image/png")
                                    data = source.get("data", "")
                                    image_parts.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}})
                    if image_parts:
                        result_content = [{"type": "text", "text": "\n".join(text_parts) or "(tool output)"}] + image_parts
                    else:
                        result_content = "\n".join(text_parts)
                elif isinstance(result_content, str):
                    result_content = result_content
                else:
                    result_content = str(result_content)
                openai_msgs.append({"role": "tool", "tool_call_id": tool_use_id, "content": result_content})
            # Add non-tool parts as user message AFTER tool messages
            if other_parts:
                non_tool = _anthropic_content_to_openai(other_parts)
                if len(non_tool) == 1 and isinstance(non_tool[0], str):
                    openai_msgs.append({"role": "user", "content": non_tool[0]})
                else:
                    formatted = []
                    for p in non_tool:
                        if isinstance(p, str) and p:
                            formatted.append({"type": "text", "text": p})
                        elif isinstance(p, dict):
                            formatted.append(p)
                    openai_msgs.append({"role": "user", "content": formatted})
            elif not tool_results:
                # No tool_results, handle normally
                if len(parts) == 1 and isinstance(parts[0], str):
                    openai_msgs.append({"role": "user", "content": parts[0]})
                else:
                    formatted = []
                    for p in parts:
                        if isinstance(p, str) and p:
                            formatted.append({"type": "text", "text": p})
                        elif isinstance(p, dict):
                            formatted.append(p)
                    openai_msgs.append({"role": "user", "content": formatted})
        elif role == "tool_result":
            # Claude Code sends tool results (like file_read output) — convert to tool role
            tool_use_id = m.get("tool_use_id", "")
            result_content = m.get("content", "")
            # Handle Anthropic tool_result content format (string or array of blocks)
            if isinstance(result_content, list):
                text_parts = []
                image_parts = []
                for block in result_content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                text_parts.append(text)
                        elif block.get("type") == "image":
                            source = block.get("source", {})
                            if source.get("type") == "base64":
                                media = source.get("media_type", "image/png")
                                data = source.get("data", "")
                                image_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{media};base64,{data}"}
                                })
                if image_parts:
                    result_content = [{"type": "text", "text": "\n".join(text_parts) or "(tool output)"}] + image_parts
                else:
                    result_content = "\n".join(text_parts) or "(tool output)"
            elif isinstance(result_content, str):
                result_content = result_content
            else:
                result_content = str(result_content)
            openai_msgs.append({"role": "tool", "tool_call_id": tool_use_id, "content": result_content})
    _app_log.info("[messages] converted %d msgs has_tools=%s",
                 len(openai_msgs),
                 has_tools)
    return openai_msgs, has_tools


def _openai_to_anthropic_content(message: dict) -> list:
    """Convert OpenAI assistant message to Anthropic content blocks."""
    content_blocks = []
    content = message.get("content", "")
    if content:
        content_blocks.append({"type": "text", "text": content})
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc_id,
                "name": func.get("name", ""),
                "input": args
            })
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    return content_blocks


def _openai_messages_to_anthropic(messages: list, system_prompt: str = "") -> tuple:
    """将 OpenAI Chat Completions 消息转换为 Anthropic Messages 格式。
    返回 (anthropic_messages, system_prompt, tools) 三元组。
    """
    anthropic_msgs = []
    system_texts = []
    tools_from_messages = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            system_texts.append(_strip_billing_header(content if isinstance(content, str) else str(content)))
            continue

        if role == "user":
            parts = []
            if isinstance(content, str):
                parts = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        if part.get("type") == "image_url":
                            img = part.get("image_url", {})
                            url = img.get("url", "") if isinstance(img, dict) else img
                            if url.startswith("data:"):
                                parts.append({"type": "image", "source": {
                                    "type": "base64",
                                    "media_type": url.split(";")[0].replace("data:", ""),
                                    "data": url.split(",", 1)[1] if "," in url else ""
                                }})
                        elif part.get("type") == "input_image":
                            parts.append(part)
                        else:
                            parts.append(part)
            anthropic_msgs.append({"role": "user", "content": parts})

        elif role == "assistant":
            content_blocks = _openai_to_anthropic_content(msg)
            anthropic_msgs.append({"role": "assistant", "content": content_blocks})
            # 从消息中提取 tool_calls 信息（已在 _openai_to_anthropic_content 中处理）

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_content = content if isinstance(content, str) else str(content)
            anthropic_msgs.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": tool_content}]
            })

    # 合并 system 文本
    if system_prompt:
        system_texts.insert(0, system_prompt)
    final_system = "\n\n".join(system_texts) if system_texts else ""

    return anthropic_msgs, final_system


def _map_stop_reason(finish_reason: str) -> str:
    return {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}.get(
        finish_reason, "end_turn")


async def _stream_anthropic_messages(model, messages, provider_id, temperature,
                                      max_tokens, username, api_key_value,
                                      requested_model="", system_prompt="", conv_key="", **extra):
    """Stream OpenAI chat chunks as Anthropic-formatted SSE events."""
    if not conv_key:
        conv_key = _conversation_cache_key(api_key_value, messages)
    msg_id = f"msg_{int(time.time())}"
    total_tokens = 0
    cache_hit = 0
    cache_miss = 0
    output_tokens = 0
    error_msg = None
    finish_reason = None
    _flushed = False  # guard: finish_reason flush runs only once
    accumulated_text = ""
    text_buffer = ""
    text_content_started = False
    block_index = 0
    tool_uses = {}
    accumulated_reasoning = ""  # index -> {id, name, arguments_buffer, started}

    try:
        # Diagnostic: log message structure summary
        msg_roles = [m.get("role", "?") if isinstance(m, dict) else "?" for m in messages]
        _app_log.debug("[messages_stream] START model=%s msg_count=%d roles=%s max_tokens=%s tools=%s",
                      model, len(messages), str(msg_roles), str(max_tokens),
                      str(extra.get("tools", [])[:10]) if extra.get("tools") else "none")
        stream_func = lambda: create_chat_completion_stream(
            model=model, messages=messages, provider_id=provider_id,
            temperature=temperature, max_tokens=max_tokens, **extra
        )
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

        chunk_count = 0
        content_total = 0
        async for chunk in _iter_stream_async(stream_func):
            chunk_count += 1
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = getattr(choice, "delta", None)
            chunk_finish = getattr(choice, "finish_reason", None)
            if chunk_finish:
                finish_reason = chunk_finish  # save first finish_reason, don't overwrite
                _app_log.debug(
                    "[messages_stream] GOT finish_reason=%s at chunk=%d content_chars=%d",
                    finish_reason, chunk_count, content_total
                )

            if delta:
                content_delta = getattr(delta, "content", None)
                if content_delta:
                    content_total += len(content_delta)
                    accumulated_text += content_delta
                    text_buffer += content_delta
                    if not text_content_started:
                        text_content_started = True
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    if len(text_buffer) >= 16:
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': text_buffer}})}\n\n"
                        text_buffer = ""
                # Capture reasoning_content for DeepSeek multi-turn replay
                reasoning_delta = getattr(delta, "reasoning_content", None)
                if reasoning_delta:
                    accumulated_reasoning += reasoning_delta

                # Handle tool call deltas (OpenAI streaming format → Anthropic tool_use events)
                tool_calls_delta = getattr(delta, "tool_calls", None)
                if tool_calls_delta:
                    for tc in tool_calls_delta:
                        idx = getattr(tc, "index", 0) if hasattr(tc, "index") else tc.get("index", 0)
                        tc_id = getattr(tc, "id", "") if hasattr(tc, "id") else tc.get("id", "")
                        tc_func = getattr(tc, "function", None) if hasattr(tc, "function") else tc.get("function", {})
                        tc_name = getattr(tc_func, "name", "") if hasattr(tc_func, "name") else tc_func.get("name", "")
                        tc_args = getattr(tc_func, "arguments", "") if hasattr(tc_func, "arguments") else tc_func.get("arguments", "")

                        if idx not in tool_uses:
                            tu_block_idx = block_index if not text_content_started else block_index + 1
                            tu_id = tc_id if tc_id else f"toolu_{idx}"
                            tool_uses[idx] = {"id": tu_id, "name": tc_name, "arguments": tc_args, "started": False, "block_index": tu_block_idx}
                        else:
                            tool_uses[idx]["arguments"] += tc_args
                            if tc_name:
                                tool_uses[idx]["name"] = tc_name
                            if tc_id:
                                tool_uses[idx]["id"] = tc_id
                            # Send delta for accumulating arguments
                            if tc_args and tool_uses[idx]["started"]:
                                try:
                                    json.loads(tool_uses[idx]["arguments"])
                                except Exception:
                                    pass  # still building, don't output incomplete JSON

            if finish_reason and not _flushed:
                _flushed = True
                # Flush text
                if text_buffer:
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': text_buffer}})}\n\n"
                    text_buffer = ""
                if text_content_started:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"

                # Emit tool_use blocks
                for idx, tu in sorted(tool_uses.items()):
                    tu_block_idx = tu["block_index"]
                    tu_id = tu["id"] or f"toolu_{idx}"
                    try:
                        tu_input = json.loads(tu["arguments"]) if tu["arguments"] else {}
                    except json.JSONDecodeError:
                        tu_input = {}
                    sse_data = {
                        "type": "content_block_start",
                        "index": tu_block_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tu_id,
                            "name": tu["name"],
                            "input": {}
                        }
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(sse_data)}\n\n"
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': tu_block_idx, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(tu_input, ensure_ascii=False)}})}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tu_block_idx})}\n\n"

            if hasattr(chunk, "usage") and chunk.usage:
                total_tokens = getattr(chunk.usage, "total_tokens", 0) or 0
                output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
                cache_hit = getattr(chunk.usage, "prompt_cache_hit_tokens", 0) or 0
                cache_miss = getattr(chunk.usage, "prompt_cache_miss_tokens", 0) or 0

        # Fallback: if thinking consumed all tokens with no text/tool output,
        # render reasoning_content as the visible response (same as _stream_chat fallback).
        if content_total == 0 and not tool_uses and accumulated_reasoning:
            _app_log.info("[messages_stream] fallback: reasoning_content (%s chars) as response", len(accumulated_reasoning))
            if not text_content_started:
                block_index += 1
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': accumulated_reasoning}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"

        stop_reason = _map_stop_reason(finish_reason) if finish_reason else "end_turn"
        _app_log.debug(
            "[messages_stream] DONE finish_reason=%s stop_reason=%s total_chunks=%d content_chars=%d accumulated_text_len=%d text_buffer_len=%d tool_uses_count=%d reasoning_chars=%d max_tokens_param=%s",
            finish_reason or "None", stop_reason, chunk_count, content_total,
            len(accumulated_text), len(text_buffer), len(tool_uses),
            len(accumulated_reasoning), str(max_tokens)
        )
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        _log_request(username, api_key_value, model, provider_id or "", "messages", True, total_tokens, requested_model)
        if accumulated_reasoning:
            _reasoning_cache[conv_key] = accumulated_reasoning
            _app_log.info("[messages_stream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d", conv_key[:60], len(accumulated_reasoning), cache_hit, cache_miss)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, total_tokens)
        increment_global_stats(success=True)

    except Exception as e:
        error_msg = _friendly_error_msg(e)
        _error_log.error("[messages_stream] %s", str(e))
        _log_request(username, api_key_value, "-", provider_id or "", "messages", False, 0, requested_model)
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)

    finally:
        if error_msg:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'server_error', 'message': error_msg}})}\n\n"


async def _anthropic_passthrough(provider_info: dict, messages: list,
                                  body: dict, max_tokens: int,
                                  temperature, model: str) -> dict:
    """Direct HTTP call to an Anthropic-compatible endpoint, bypassing liteLLM conversion."""
    import httpx
    api_base = (provider_info.get("api_base") or "").rstrip("/")
    api_key = provider_info.get("api_key") or "sk-no-auth"
    # 从复合 ID 提取实际模型名（上游不认识 provider/model 格式）
    mid = parse_model_id(model)
    upstream_model = mid.model_name
    # Build Anthropic-format request — use routed model, not body.get("model")
    req_body = {"model": upstream_model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        req_body["temperature"] = temperature
    system = _strip_billing_header(body.get("system"))
    if system:
        req_body["system"] = system
    _app_log.info("[anthropic_passthrough] filtered system=%s", req_body.get("system"))
    tools = body.get("tools")
    if tools:
        req_body["tools"] = [{"name": t["name"], "description": t.get("description", ""),
                             "input_schema": t.get("input_schema", {"type": "object", "properties": {}})}
                            for t in tools if isinstance(t, dict) and t.get("name")]
    # Per-provider thinking mode: configured via provider_info.extra_headers.
    extra_headers = provider_info.get("extra_headers", {}) or {}
    thinking = extra_headers.get("thinking")
    if thinking in ("enabled", "disabled"):
        req_body["thinking"] = {"type": thinking}
    _app_log.info("[anthropic_passthrough] request body=%s", json.dumps(req_body, ensure_ascii=False))
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{api_base}/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"},
            json=req_body,
        )
        if resp.status_code != 200:
            try:
                err_body = resp.json()
                err_msg = err_body.get("error", {}).get("message", resp.text[:300])
            except Exception:
                err_msg = resp.text[:300] or f"HTTP {resp.status_code}"
            raise HTTPException(status_code=502, detail=f"Upstream: {err_msg}")
        data = resp.json()
        # Wrap in OpenAI-compatible format for downstream processing
        content_blocks = data.get("content", [])
        text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
        tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]
        # Build minimal OpenAI-compatible response
        openai_message = {"role": "assistant", "content": "\n".join(text_parts) or None}
        if tool_uses:
            openai_message["tool_calls"] = [{
                "id": tu["id"], "type": "function",
                "function": {"name": tu.get("name", ""), "arguments": json.dumps(tu.get("input", {}))}
            } for tu in tool_uses]
        usage = data.get("usage", {})
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": openai_message,
                "finish_reason": data.get("stop_reason", "end_turn")
            })],
            "usage": type("Usage", (), {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            })
        })()


async def _stream_anthropic_passthrough(provider_info, messages, body, max_tokens, temperature,
                                         username, api_key_value, model, requested_model):
    """Stream SSE events directly from an Anthropic-compatible upstream."""
    import httpx
    api_base = (provider_info.get("api_base") or "").rstrip("/")
    api_key = provider_info.get("api_key") or "sk-no-auth"
    # 从复合 ID 提取实际模型名（上游不认识 provider/model 格式）
    mid = parse_model_id(model)
    upstream_model = mid.model_name
    req_body = {"model": upstream_model, "messages": messages, "max_tokens": max_tokens, "stream": True}
    if temperature is not None:
        req_body["temperature"] = temperature
    system = _strip_billing_header(body.get("system"))
    if system:
        req_body["system"] = system
    tools = body.get("tools")
    if tools:
        # Strip type field — Anthropic-compatible endpoints (DeepSeek) reject type:"custom"
        req_body["tools"] = [{k: v for k, v in t.items() if k != "type"} for t in tools if isinstance(t, dict)]
    # Per-provider thinking mode: configured via provider_info.extra_headers.
    extra_headers = provider_info.get("extra_headers", {}) or {}
    thinking = extra_headers.get("thinking")
    if thinking in ("enabled", "disabled"):
        req_body["thinking"] = {"type": thinking}
    total_tokens = 0
    provider_id = provider_info.get("id", "")
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST", f"{api_base}/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json=req_body,
            ) as resp:
                if resp.status_code != 200:
                    try:
                        err_body = await resp.aread()
                        err_data = json.loads(err_body)
                        err_msg = err_data.get("error", {}).get("message", str(err_body)[:300])
                    except Exception:
                        err_msg = f"HTTP {resp.status_code}"
                    _log_request(username, api_key_value, "-", provider_id, "messages", False, 0, requested_model)
                    increment_global_stats(success=False)
                    if username != "legacy":
                        increment_user_usage(username, api_key_value, False, 0)
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': f'Upstream {resp.status_code}: {err_msg}'}})}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if line:
                        # Track usage from Anthropic SSE events for gateway stats
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("type") == "message_start":
                                    usage = data.get("message", {}).get("usage", {})
                                    total_tokens += usage.get("input_tokens", 0)
                                elif data.get("type") == "message_delta":
                                    usage = data.get("usage", {})
                                    total_tokens += usage.get("output_tokens", 0)
                            except Exception:
                                pass
                        yield line + "\n"
        _log_request(username, api_key_value, model, provider_id, "messages", True, total_tokens, requested_model)
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, total_tokens)
    except Exception as e:
        _log_request(username, api_key_value, "-", provider_id, "messages", False, 0, requested_model)
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("[anthropic_passthrough_stream] %s", e)
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"


@router.post("/messages")
async def anthropic_messages(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)

    body = await request.json()
    model = body.get("model")
    anthropic_msgs = body.get("messages", [])
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = get_default("max_tokens", 16384)
    provider_id = body.get("provider_id")
    stream = body.get("stream", False)
    system_prompt = _strip_billing_header(body.get("system", ""))
    temperature = body.get("temperature")

    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    ensure_model_allowed(user, api_key, model)

    username = user.get("username", "legacy")
    api_key_value = api_key.get("key", "")
    requested_model = model
    route_model, route_provider = _apply_routing_rules(username, api_key_value, model, model)
    if route_model != model:
        _app_log.info("[messages] ROUTED model=%s -> %s", model, route_model)
        model = route_model
    if route_provider:
        provider_id = route_provider

    # Check if target provider is Anthropic-native — pass through directly
    # to avoid double-formatting which loses tool_use IDs
    from app.database import get_provider as _get_prov, find_provider_by_model as _find
    if provider_id:
        provider_info = _get_prov(provider_id)
    else:
        provider_info = _find(model)
    if provider_info and provider_info.get("provider_type") == "anthropic":
        is_anthropic_provider = True
        messages = anthropic_msgs
    else:
        is_anthropic_provider = False
        messages, _ = _anthropic_to_openai_messages(anthropic_msgs, system_prompt)
        # Diagnostic: log converted message summary to find truncation root cause
        _app_log.debug(
            "[messages] CONVERTED anthropic(%d msgs) -> openai(%d msgs) system_prompt_len=%d tools=%s stream=%s max_tokens=%s model=%s",
            len(anthropic_msgs), len(messages), len(system_prompt) if system_prompt else 0,
            str(body.get("tools", [])[:10]) if body.get("tools") else "none",
            str(body.get("stream")), str(max_tokens), model
        )
        system_prompt = ""  # already embedded in messages

    # Compute conversation cache key BEFORE preprocessing (see chat_completions for rationale)
    conv_key = _conversation_cache_key(api_key_value, messages)

    # Preprocessor: replace images with text descriptions
    msg_list, modified = await _maybe_preprocess(messages, model, requested_model=requested_model)
    messages = msg_list
    if modified:
        _reasoning_cache.drop(conv_key)

    try:
        # Collect Anthropic tool definitions: top-level tools + tools from body
        tools = body.get("tools")
        if tools and isinstance(tools, list):
            converted = []
            for t in tools:
                if isinstance(t, dict) and t.get("name"):
                    converted.append({
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("input_schema", {"type": "object", "properties": {}})
                        }
                    })
            tools = converted if converted else None
        else:
            tools = body.get("tools")  # pass through if not our format

        # Non-Anthropic providers: strip tool_choice that will cause rejection.
        # MiniMax rejects "auto" (error 2013), and Anthropic dict-format
        # tool_choice ({type: "auto"}) has no OpenAI equivalent.
        tool_choice = body.get("tool_choice")
        if not is_anthropic_provider:
            if isinstance(tool_choice, dict) and not tool_choice.get("name"):
                tool_choice = None  # Anthropic "auto"/"any" → let provider default
            elif tool_choice == "auto":
                tool_choice = None  # MiniMax rejects "auto"

        # Stream or non-stream
        # Inject DeepSeek reasoning_content for multi-turn continuity
        if not is_anthropic_provider:
            cached_rc = _reasoning_cache.get(conv_key)
            if cached_rc is not None:
                count = 0
                for msg in messages:
                    if (msg.get("role") == "assistant"
                        and not msg.get("reasoning_content")
                        and msg.get("tool_calls")):
                        msg["reasoning_content"] = cached_rc
                        count += 1
                _app_log.info("[messages] INJECTED rc injected=%d asst-with-tool msgs len=%d conv_key=%s", count, len(cached_rc), conv_key)
            else:
                # First turn: inject empty reasoning_content only into assistant
                # messages with tool_calls.
                for msg in messages:
                    if (msg.get('role') == 'assistant'
                        and 'reasoning_content' not in msg
                        and msg.get('tool_calls')):
                        msg['reasoning_content'] = ''

        if stream:
            if provider_info and provider_info.get("provider_type") == "anthropic":
                return StreamingResponse(
                    _stream_anthropic_passthrough(
                        provider_info, messages, body, max_tokens, temperature,
                        username, api_key_value, model, requested_model
                    ),
                    media_type="text/event-stream"
                )
            return StreamingResponse(
                _stream_anthropic_messages(
                    model=model, messages=messages, provider_id=provider_id,
                    temperature=temperature, max_tokens=max_tokens,
                    username=username, api_key_value=api_key_value,
                    requested_model=requested_model, system_prompt=system_prompt,
                    tools=tools, tool_choice=tool_choice,
                    conv_key=conv_key,
                ),
                media_type="text/event-stream"
            )

        # Non-streaming: use Anthropic passthrough for native providers
        if provider_info and provider_info.get("provider_type") == "anthropic":
            response = await _anthropic_passthrough(
                provider_info, messages, body, max_tokens, temperature, model)
        else:
            response = await anyio.to_thread.run_sync(
                lambda: create_chat_completion(
                    model=model, messages=messages, provider_id=provider_id,
                    max_tokens=max_tokens, temperature=temperature,
                    tools=tools, tool_choice=tool_choice,
                )
            )
        choice = response.choices[0]
        message = getattr(choice, "message", {})
        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"

        # Fallback: if thinking consumed all tokens with no text/tool output,
        # render reasoning_content as the visible response.
        reasoning_content = getattr(message, "reasoning_content", None)
        if not message.get("content") and not message.get("tool_calls") and reasoning_content:
            message["content"] = reasoning_content

        # Capture cache stats before converting message
        usage = usage_dict(response)
        cache_hit = usage.get("prompt_cache_hit_tokens", 0)
        cache_miss = usage.get("prompt_cache_miss_tokens", 0)

        if reasoning_content:
            _reasoning_cache[conv_key] = reasoning_content
            _app_log.info("[messages_nonstream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d", conv_key[:60], len(reasoning_content), cache_hit, cache_miss)
        # Convert OpenAI response to Anthropic format
        content_blocks = _openai_to_anthropic_content(message)
        _log_request(username, api_key_value, model, provider_id or "", "messages", True, usage.get("total_tokens", 0), requested_model)
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, usage.get("total_tokens", 0))

        return {
            "id": f"msg_{int(time.time())}",
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": model,
            "stop_reason": _map_stop_reason(finish_reason),
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0)
            }
        }
    except Exception as e:
        _log_request(username, api_key_value, "-", provider_id or "", "messages", False, 0, requested_model)
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", str(e))
        raise HTTPException(status_code=500, detail=_friendly_error_msg(e))


async def _stream_responses(model, messages, provider_id, temperature, max_tokens, username, api_key_value, instructions, requested_model="", conv_key="", **extra):
    resp_id = f"resp_{int(time.time())}"
    msg_id = f"msg_{int(time.time())}"
    created_at = int(time.time())

    if not conv_key:
        conv_key = _conversation_cache_key(api_key_value, messages)

    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    error_msg = None
    cache_hit = 0
    cache_miss = 0
    finish_reason = None
    _flushed = False  # guard: finish_reason flush runs only once
    text_item_added = False
    text_content_added = False
    accumulated_text = ""
    accumulated_reasoning = ""
    text_buffer = ""
    think_stripped = False  # True once </think> has been passed
    tool_calls_state = {}  # index -> {id, name, arguments_buffer, item_added, output_index}
    output_index_counter = 0

    # Tool-call circuit breaker: strip tools if too many consecutive tool-only turns
    if _tool_only_turns.get(conv_key, 0) >= TOOL_ONLY_LIMIT:
        extra.pop("tools", None)
        extra.pop("tool_choice", None)

    # Strip OpenAI-specific fields from tool function definitions that
    # non-OpenAI providers (MiniMax) reject as "invalid chat setting (2013)".
    for tool in extra.get("tools", []):
        fn = tool.get("function")
        if isinstance(fn, dict):
            fn.pop("strict", None)
            fn.pop("additionalProperties", None)
            params = fn.get("parameters")
            if isinstance(params, dict):
                params.pop("additionalProperties", None)
                for prop in params.get("properties", {}).values():
                    if isinstance(prop, dict):
                        prop.pop("additionalProperties", None)

    try:
        _app_log.info("[responses_stream] START model=%s", model)
        stream_func = lambda: create_chat_completion_stream(
            model=model,
            messages=messages,
            provider_id=provider_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra
        )

        response_base = {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "status": "in_progress",
            "model": model,
            "output": [],
            "previous_response_id": None,
            "metadata": {},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        }

        yield f"data: {json.dumps({'type': 'response.created', 'response': response_base})}\n\n"
        yield f"data: {json.dumps({'type': 'response.in_progress', 'response': response_base})}\n\n"

        chunk_count = 0
        async for chunk in _iter_stream_async(stream_func):
            chunk_count += 1
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = getattr(choice, "delta", None)
            chunk_finish = getattr(choice, "finish_reason", None)
            if chunk_finish:
                finish_reason = chunk_finish  # save first finish_reason, don't overwrite

            if delta:
                # --- Handle text content ---
                content_delta = getattr(delta, "content", None)
                if content_delta:
                    accumulated_text += content_delta
                    if not think_stripped:
                        # MiniMax/DeepSeek emit <think>...</think> inline.
                        # Buffer until </think> is seen, then extract & strip.
                        if '</think>' in accumulated_text:
                            accumulated_text, think_content = _extract_and_strip_think(accumulated_text)
                            if think_content:
                                accumulated_reasoning = think_content
                            think_stripped = True
                            text_buffer = accumulated_text  # seed with cleaned text
                        elif '<think>' in accumulated_text or accumulated_text.lstrip().startswith('<think'):
                            # If <think> never closes after buffering much content,
                            # treat it as plain text so users aren't stuck waiting.
                            if len(accumulated_text) >= 200:
                                accumulated_text = accumulated_text.replace('<think>', '', 1)
                                think_stripped = True
                                text_buffer = accumulated_text
                            # else: still inside <think> block — don't yield yet
                        elif len(accumulated_text) >= 5:
                            # No think tag detected — model doesn't use thinking mode
                            think_stripped = True
                            text_buffer = accumulated_text
                    else:
                        text_buffer += content_delta

                    if not text_item_added:
                        yield f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': output_index_counter, 'item': {'type': 'message', 'id': msg_id, 'status': 'in_progress', 'role': 'assistant', 'content': []}})}\n\n"
                        text_item_added = True
                        text_output_index = output_index_counter
                        output_index_counter += 1
                    if think_stripped:
                        if not text_content_added:
                            yield f"data: {json.dumps({'type': 'response.content_part.added', 'output_index': text_output_index, 'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}})}\n\n"
                            text_content_added = True
                        if len(text_buffer) >= 16:
                            yield f"data: {json.dumps({'type': 'response.output_text.delta', 'output_index': text_output_index, 'content_index': 0, 'delta': text_buffer})}\n\n"
                            text_buffer = ""

                # --- Capture reasoning content for DeepSeek multi-turn replay ---
                reasoning_delta = getattr(delta, "reasoning_content", None)
                if reasoning_delta:
                    accumulated_reasoning += reasoning_delta

                # --- Handle tool calls ---
                tool_calls_delta = getattr(delta, "tool_calls", None)
                if tool_calls_delta:
                    for tc in tool_calls_delta:
                        _tool_log.debug("[_stream_responses] raw tc type=%s repr=%s", type(tc).__name__, repr(tc))
                        idx = getattr(tc, "index", 0) if hasattr(tc, "index") else tc.get("index", 0)
                        tc_id = getattr(tc, "id", "") if hasattr(tc, "id") else tc.get("id", "")
                        # Coerce id to str (MiniMax returns bare integers)
                        if tc_id is not None and not isinstance(tc_id, str):
                            tc_id = str(tc_id)
                        # Skip spurious tool-calls from non-standard content blocks (e.g. MiniMax "thinking")
                        # Only filter when index<0.  id=None is normal for arguments-only delta chunks.
                        if int(idx) < 0:
                            _tool_log.info("[_stream_responses] FILTERED spurious: id=%s idx=%s", tc_id, idx)
                            continue
                        tc_func = getattr(tc, "function", None) if hasattr(tc, "function") else tc.get("function", {})

                        if idx not in tool_calls_state:
                            # New tool call
                            fn_name = getattr(tc_func, "name", "") if hasattr(tc_func, "name") else tc_func.get("name", "")
                            tc_output_index = output_index_counter
                            output_index_counter += 1
                            call_id = tc_id if tc_id and tc_id.startswith("call_") else (f"call_{tc_id}" if tc_id else f"call_{int(time.time())}_{idx}")
                            tool_calls_state[idx] = {
                                "id": tc_id or f"fc_{int(time.time())}_{idx}",
                                "call_id": call_id,
                                "name": fn_name,
                                "arguments_buffer": "",
                                "item_added": False,
                                "output_index": tc_output_index
                            }
                            _tool_log.info("[_stream_responses] NEW tool_call idx=%d name=%s id=%s", idx, fn_name, tc_id)

                        state = tool_calls_state[idx]
                        args_chunk = getattr(tc_func, "arguments", "") if hasattr(tc_func, "arguments") else tc_func.get("arguments", "")
                        if args_chunk:
                            _tool_log.info("[_stream_responses] args_chunk RAW: %s", args_chunk)
                            state["arguments_buffer"] += args_chunk
                            # Fix malformed JSON (e.g. MiniMax sends url:undefined) — both in buffer and in the chunk being sent
                            if "undefined" in args_chunk:
                                args_chunk = _sanitize_args(args_chunk)
                                _tool_log.info("[_stream_responses] args_chunk SANITIZED -> %s", args_chunk)
                            if "undefined" in state["arguments_buffer"]:
                                state["arguments_buffer"] = _sanitize_args(state["arguments_buffer"])
                                _tool_log.info("[_stream_responses] buffer SANITIZED -> %s", state["arguments_buffer"])
                            if not state["item_added"]:
                                yield f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': state['output_index'], 'item': {'type': 'function_call', 'id': state['id'], 'call_id': state['call_id'], 'name': state['name'], 'arguments': '', 'status': 'in_progress'}})}\n\n"
                                state["item_added"] = True
                            yield f"data: {json.dumps({'type': 'response.function_call_arguments.delta', 'output_index': state['output_index'], 'call_id': state['call_id'], 'delta': args_chunk})}\n\n"

            # --- Handle finish ---
            if finish_reason and not _flushed:
                _flushed = True
                # Flush text buffer
                if text_buffer:
                    yield f"data: {json.dumps({'type': 'response.output_text.delta', 'output_index': text_output_index, 'content_index': 0, 'delta': text_buffer})}\n\n"
                    text_buffer = ""
                if text_content_added:
                    yield f"data: {json.dumps({'type': 'response.output_text.done', 'output_index': text_output_index, 'content_index': 0, 'text': accumulated_text})}\n\n"
                    yield f"data: {json.dumps({'type': 'response.content_part.done', 'output_index': text_output_index, 'content_index': 0, 'part': {'type': 'output_text', 'text': accumulated_text, 'annotations': []}})}\n\n"
                if text_item_added:
                    output_content = [{'type': 'output_text', 'text': accumulated_text, 'annotations': []}]
                    msg_item = {'type': 'message', 'id': msg_id, 'status': 'completed', 'role': 'assistant', 'content': output_content}
                    if accumulated_reasoning:
                        msg_item['reasoning_content'] = accumulated_reasoning
                    yield f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': text_output_index, 'item': msg_item})}\n\n"

                # Finalize tool calls
                for idx, state in sorted(tool_calls_state.items()):
                    if state["item_added"]:
                        yield f"data: {json.dumps({'type': 'response.function_call_arguments.done', 'output_index': state['output_index'], 'call_id': state['call_id'], 'arguments': state['arguments_buffer']})}\n\n"
                        yield f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': state['output_index'], 'item': {'type': 'function_call', 'id': state['id'], 'call_id': state['call_id'], 'name': state['name'], 'arguments': state['arguments_buffer'], 'status': 'completed'}})}\n\n"

                # Reset handled

            if hasattr(chunk, "usage") and chunk.usage:
                total_tokens = getattr(chunk.usage, "total_tokens", 0) or 0
                input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
                cache_hit = getattr(chunk.usage, "prompt_cache_hit_tokens", 0) or 0
                cache_miss = getattr(chunk.usage, "prompt_cache_miss_tokens", 0) or 0

        _app_log.info("[responses_stream] TOTAL chunks=%d text=%d tools=%d reasoning=%d", chunk_count, len(accumulated_text), len(tool_calls_state), len(accumulated_reasoning))

        # Store reasoning_content for next turn replay (DeepSeek thinking mode requires echo-back)
        if accumulated_reasoning:
            _reasoning_cache[conv_key] = accumulated_reasoning
            _app_log.info("[responses_stream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d", conv_key, len(accumulated_reasoning), cache_hit, cache_miss)

        # Update tool-only counter for circuit breaker
        has_text = len(accumulated_text.strip()) > 0
        has_tools = len(tool_calls_state) > 0
        if has_tools and not has_text:
            _tool_only_turns.increment(conv_key)
        else:
            _tool_only_turns.reset(conv_key)

        _log_request(username, api_key_value, model, provider_id or "", "responses", True, total_tokens, requested_model)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, total_tokens)
        increment_global_stats(success=True)

    except Exception as e:
        import traceback
        error_msg = _friendly_error_msg(e)
        _app_log.error("[responses_stream] type=%s msg=%s", type(e).__name__, str(e))
        _app_log.error("[responses_stream] %s", traceback.format_exc())
        _error_log.error("[responses_stream] type=%s msg=%s", type(e).__name__, str(e))
        _error_log.error("[responses_stream] %s", traceback.format_exc())
        _log_request(username, api_key_value, "-", provider_id or "", "responses", False, 0, requested_model)
        _error_log.error("[responses_stream] %s", error_msg)
        _tool_only_turns.reset(conv_key)  # Reset on error
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)

    finally:
        if error_msg:
            yield f"data: {json.dumps({'type': 'error', 'error': {'message': error_msg, 'type': 'server_error'}})}\n\n"
        completion_output = []
        if not error_msg:
            if text_item_added:
                msg_out = {'type': 'message', 'id': msg_id, 'status': 'completed', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': accumulated_text, 'annotations': []}]}
                if accumulated_reasoning:
                    msg_out['reasoning_content'] = accumulated_reasoning
                completion_output.append(msg_out)
            elif accumulated_reasoning:
                # Thinking mode consumed all tokens before generating visible content.
                # Fall back to reasoning_content as the response.
                completion_output.append({'type': 'message', 'id': msg_id, 'status': 'completed', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': accumulated_reasoning, 'annotations': []}]})
            elif accumulated_text:
                # <think> opened but never closed before stream ended, or no
                # thinking tags at all. Flush whatever was buffered as the response.
                flushed = accumulated_text.replace('<think>', '', 1) if '<think>' in accumulated_text else accumulated_text
                if flushed.strip():
                    completion_output.append({'type': 'message', 'id': msg_id, 'status': 'completed', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': flushed, 'annotations': []}]})
            for idx in sorted(tool_calls_state.keys()):
                state = tool_calls_state[idx]
                if state["item_added"]:
                    completion_output.append({'type': 'function_call', 'id': state['id'], 'call_id': state['call_id'], 'name': state['name'], 'arguments': state['arguments_buffer'], 'status': 'completed'})
        _app_log.info("[responses_stream] SENT response.completed error=%s output_items=%d", str(error_msg is not None), len(completion_output))
        response_completed = {
            'type': 'response.completed',
            'response': {
                'id': resp_id, 'object': 'response', 'created_at': created_at,
                'status': 'failed' if error_msg else 'completed',
                'model': model, 'output': completion_output,
                'previous_response_id': None, 'metadata': {},
                'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total_tokens}
            }
        }
        if error_msg:
            response_completed['response']['status_details'] = {
                'error': {'type': 'server_error', 'message': error_msg}
            }
        yield f"data: {json.dumps(response_completed)}\n\n"
        yield "data: [DONE]\n\n"


def _normalize_messages(messages: list) -> list:
    """Merge consecutive same-role messages into one.

    Many OpenAI-compatible providers (MiniMax, DeepSeek, etc.) require strict
    user/assistant/tool alternation and only one system message at the start.
    Consecutive same-role messages cause "invalid chat setting" errors.

    Also strips Anthropic billing headers (x-anthropic-billing-header; cch=xxx)
    from system message content — injected by Claude Code since v2.1.37, these
    contain a random `cch` value that breaks DeepSeek's prefix cache.
    """
    if not messages:
        return messages

    merged = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Strip billing header from system message text
        if role == "system" and isinstance(content, str):
            msg["content"] = _strip_billing_header(content)
            content = msg["content"]

        if merged and merged[-1].get("role") == role and role in ("system", "user"):
            # Merge content into previous same-role message
            prev = merged[-1]
            prev_content = prev.get("content", "")
            if isinstance(prev_content, str) and isinstance(content, str):
                prev["content"] = prev_content + "\n\n" + content
            elif isinstance(prev_content, list) and isinstance(content, str):
                prev_content.append({"type": "text", "text": content})
            elif isinstance(prev_content, str) and isinstance(content, list):
                prev["content"] = [{"type": "text", "text": prev_content}] + content
            elif isinstance(prev_content, list) and isinstance(content, list):
                prev["content"] = prev_content + content
            # Preserve other fields from the later message (reasoning_content, etc.)
            for k, v in msg.items():
                if k not in ("role", "content") and v:
                    if k not in prev or not prev.get(k):
                        prev[k] = v
        else:
            merged.append(dict(msg))

    return merged


def _extract_and_strip_think(text: str) -> tuple[str, str]:
    """Extract and remove <think>...</think> blocks with correct nesting support.

    Uses a depth counter rather than a non-greedy regex so that nested
    <think> blocks (e.g. <think>A<think>B</think>C</think>) are handled
    correctly — the outermost block is extracted whole.
    """
    if not text:
        return text, ""
    think_parts = []
    result = []
    i = 0
    while i < len(text):
        start = text.find("<think>", i)
        if start == -1:
            result.append(text[i:])
            break
        result.append(text[i:start])
        # Find matching </think> using depth counter
        depth = 1
        pos = start + 7  # len("<think>") == 7
        while depth > 0 and pos < len(text):
            next_open = text.find("<think>", pos)
            next_close = text.find("</think>", pos)
            if next_close == -1:
                # Unclosed <think> — treat rest as think content
                pos = -1
                break
            if next_open != -1 and next_open < next_close:
                depth += 1
                pos = next_open + 7
            else:
                depth -= 1
                if depth == 0:
                    # Extract content between <think> and </think>
                    think_parts.append(text[start + 7:next_close])
                pos = next_close + 8  # len("</think>") == 8
        if pos == -1:
            result.append(text[start:])
            break
        # Skip whitespace after </think> to match old regex \s* behavior
        while pos < len(text) and text[pos] in " \t\n\r\f":
            pos += 1
        i = pos
    return "".join(result).strip(), "\n".join(think_parts)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks, discarding the content."""
    cleaned, _ = _extract_and_strip_think(text)
    return cleaned


def _convert_responses_input(input_list: list) -> list:
    """Convert OpenAI Responses API input format to Chat Completions messages format.

    Handles: message, function_call, function_call_output, reasoning.
    """
    messages = []

    def _convert_content(content_parts):
        """Convert Responses API content parts to Chat Completions content format.

        Returns a plain string when text-only (backward compatible), or a list of
        content part dicts when images are present (OpenAI Chat Completions format).
        """
        if isinstance(content_parts, str):
            return content_parts
        if isinstance(content_parts, list):
            has_visual = False
            parts = []
            for part in content_parts:
                if isinstance(part, dict):
                    if part.get("type") in ("input_text", "output_text", "text"):
                        parts.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") == "input_image":
                        has_visual = True
                        image_url = part.get("image_url", "")
                        if isinstance(image_url, dict):
                            image_url = image_url.get("url", "")
                        detail = part.get("detail", "auto")
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": detail}
                        })
                elif isinstance(part, str):
                    parts.append({"type": "text", "text": part})
            if has_visual:
                return parts
            text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
            return text
        return ""

    i = 0
    while i < len(input_list):
        item = input_list[i]
        if not isinstance(item, dict):
            i += 1
            continue

        item_type = item.get("type", "")

        if item_type == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            content = _convert_content(item.get("content", []))
            msg = {"role": role, "content": content}
            # Preserve reasoning_content for DeepSeek multi-turn continuity
            rc = item.get("reasoning_content")
            if role == "assistant" and rc:
                msg["reasoning_content"] = rc
            messages.append(msg)
            i += 1

        elif item_type == "function_call":
            # Collect consecutive function_call items into one assistant message
            tool_calls = []
            while i < len(input_list) and isinstance(input_list[i], dict) and input_list[i].get("type") == "function_call":
                fc = input_list[i]
                tc_id = fc.get("call_id", "")
                tc_name = fc.get("name", "")
                tc_args = fc.get("arguments", "")
                tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": tc_name, "arguments": tc_args}
                })
                i += 1
            msg = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            # Check any function_call for reasoning_content (Codex may preserve it)
            for fc_item in input_list[i - len(tool_calls):i]:
                if isinstance(fc_item, dict) and fc_item.get("reasoning_content"):
                    msg["reasoning_content"] = fc_item["reasoning_content"]
                    break
            messages.append(msg)

        elif item_type == "function_call_output":
            fc_output = item
            output = fc_output.get("output", "")
            # Diagnostic: show output format for debugging image preprocessing
            if isinstance(output, str):
                has_data_uri = "data:image" in output
                _app_log.info("[responses FCO] call_id=%s output=str(len=%d, data_uri=%s, preview=%s)",
                             fc_output.get('call_id','?')[:30], len(output), has_data_uri, repr(output[:200]))
            elif isinstance(output, list):
                types = [(p.get("type","?"), len(str(p)[:80])) for p in output if isinstance(p, dict)]
                _app_log.info("[responses FCO] call_id=%s output=list(len=%d, types=%s)",
                             fc_output.get('call_id','?')[:30], len(output), types)
            else:
                _app_log.info("[responses FCO] call_id=%s output=%s",
                             fc_output.get('call_id','?')[:30], type(output).__name__)
            # Convert content parts (input_image → image_url) so preprocessing can
            # detect and describe images embedded in tool call outputs.
            converted_output = _convert_content(output)
            messages.append({
                "role": "tool",
                "tool_call_id": fc_output.get("call_id", ""),
                "content": converted_output
            })
            i += 1

        elif item_type == "reasoning":
            # Skip reasoning items — they are summaries, not full conversation context
            i += 1

        elif item_type == "input_image":
            # Top-level input_image (Responses API): attach to the nearest preceding
            # user message, or create a new user message if none exists yet.
            image_url = item.get("image_url", "")
            if isinstance(image_url, dict):
                image_url = image_url.get("url", "")
            detail = item.get("detail", "auto")
            image_part = {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
            # Walk backwards to find last user message
            attached = False
            for m in reversed(messages):
                if m.get("role") == "user":
                    if isinstance(m["content"], str):
                        m["content"] = [{"type": "text", "text": m["content"]}]
                    m["content"].append(image_part)
                    attached = True
                    break
            if not attached:
                messages.append({"role": "user", "content": [image_part]})
            i += 1

        elif "role" in item:
            # Fallback for simple message format
            messages.append({"role": item["role"], "content": item.get("content", "")})
            i += 1
        else:
            i += 1

    return messages


def _convert_responses_tools(tools: list) -> list:
    """Convert tools from Responses API format to Chat Completions format.

    Responses API: {"type": "function", "name": "...", "description": "...", "parameters": {...}}
    Chat Completions: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

    Only keeps type='function' tools; filters out web_search, custom, and other built-in types
    that are Codex-specific and not supported by Chat Completions API.
    """
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        # Already in Chat Completions format (has nested 'function' key)
        if "function" in tool:
            converted.append(tool)
            continue
        tool_type = tool.get("type", "")
        # Only pass through function-type tools; skip built-in types like web_search, custom
        if tool_type != "function":
            continue
        # Responses API flat format — wrap non-type fields under 'function'
        # Strip OpenAI-specific fields (strict, additionalProperties) that
        # non-OpenAI providers (MiniMax) reject as "invalid chat setting (2013)".
        # Always copy nested structures before modifying to avoid mutating the
        # original request body.
        function_fields = {k: v for k, v in tool.items() if k not in ("type", "strict", "additionalProperties")}
        params = function_fields.get("parameters")
        if isinstance(params, dict):
            params = dict(params)  # shallow copy to avoid mutating original
            params.pop("additionalProperties", None)
            props = params.get("properties", {})
            if isinstance(props, dict):
                cleaned = {}
                for key, prop in props.items():
                    if isinstance(prop, dict):
                        prop = dict(prop)  # copy before modifying
                        prop.pop("additionalProperties", None)
                    cleaned[key] = prop
                params["properties"] = cleaned
            function_fields["parameters"] = params
        converted.append({"type": tool_type, "function": function_fields})
    return converted



def _apply_routing_rules(username: str, api_key_value: str, requested_model: str, resolved_model: str) -> tuple[str, str]:
    """Apply user-defined routing rules. Returns (final_model, provider_id)."""

    rules = get_routing_rules()
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        # Match username: empty = all, or exact match
        rule_user = rule.get("username", "")
        if rule_user and rule_user != username:
            continue
        # Match API key: empty = all, or substring match
        key_pat = rule.get("api_key_pattern", "")
        if key_pat and key_pat not in api_key_value:
            continue
        # Match model: the requested_model (before resolution), supports * wildcard
        # 同时尝试匹配复合 ID 和简单 model_id，兼容两种格式
        match_model = rule.get("match_model", "")
        if not match_model:
            continue
        mid = parse_model_id(requested_model)
        if not (_wildcard_match(match_model, requested_model) or
                (mid.is_composite and _wildcard_match(match_model, mid.model_name))):
            continue
        # Rule matched — return target
        target = rule.get("target_model", resolved_model)
        provider = rule.get("target_provider", "")
        if target and target != resolved_model:
            _app_log.info("[routing] rule='%s' matched: %s@%s requested '%s', routing to '%s'",
                          rule.get("name", ""), username, _mask_key(api_key_value),
                          requested_model, target)
        return target or resolved_model, provider or ""
    return resolved_model, ""


def _wildcard_match(pattern: str, value: str) -> bool:
    """Simple glob-style wildcard matching: * matches any sequence."""
    regex = re.escape(pattern).replace(r"\*", ".*")
    return bool(re.fullmatch(regex, value, re.IGNORECASE))


@router.post("/responses")
async def responses_endpoint(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)

    body = await request.json()
    model = body.get("model")
    input_data = body.get("input", "")
    instructions = body.get("instructions", "")
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = get_default("max_tokens", 16384)
    provider_id = body.get("provider_id")
    stream = body.get("stream", False)

    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if not input_data:
        raise HTTPException(status_code=400, detail="input is required")

    # Log Codex request details for debugging
    tools_count = len(body.get("tools", []))
    input_len = len(json.dumps(body.get("input", ""), ensure_ascii=False))
    instructions_len = len(body.get("instructions", ""))
    # Log input item types for debugging tool loop
    if isinstance(body.get("input"), list):
        item_types = {}
        for item in body["input"]:
            t = item.get("type", "unknown") if isinstance(item, dict) else "non-dict"
            item_types[t] = item_types.get(t, 0) + 1
        _app_log.info("[responses] model=%s stream=%s tools=%d input_len=%d instructions_len=%d input_types=%s", model, stream, tools_count, input_len, instructions_len, str(item_types))
    else:
        _app_log.info("[responses] model=%s stream=%s tools=%d input_len=%d instructions_len=%d", model, stream, tools_count, input_len, instructions_len)

    # Check permission on requested model BEFORE routing
    requested_model = model
    ensure_model_allowed(user, api_key, requested_model)
    # Apply user-defined routing rules
    username = user.get("username", "legacy")
    api_key_value = api_key.get("key", "")
    route_model, route_provider = _apply_routing_rules(username, api_key_value, requested_model, model)
    if route_model != model:
        _app_log.info("[responses] ROUTED model=%s -> %s", model, route_model)
        model = route_model
    if route_provider:
        provider_id = route_provider

    if isinstance(input_data, str):
        if instructions:
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_data}
            ]
        else:
            messages = [{"role": "user", "content": input_data}]
    elif isinstance(input_data, list):
        messages = _convert_responses_input(input_data)
        if instructions:
            messages.insert(0, {"role": "system", "content": instructions})
    else:
        raise HTTPException(status_code=400, detail="input must be a string or list of messages")

    # Merge consecutive same-role messages so the array alternates role correctly.
    # Non-OpenAI providers (MiniMax) require strict user/assistant alternation and
    # reject consecutive same-role messages as "invalid chat setting (2013)".
    _pre_norm = len(messages)
    messages = _normalize_messages(messages)
    _app_log.info("[responses NORM] messages %d -> %d roles=%s", _pre_norm, len(messages), [m['role'] for m in messages])

    # Compute conversation cache key BEFORE preprocessing (see chat_completions for rationale)
    conv_key = _conversation_cache_key(api_key_value, messages)

    # Preprocessor: replace images with text
    msg, modified = await _maybe_preprocess(messages, model, provider_id, requested_model=requested_model)
    messages = msg
    if modified:
        _reasoning_cache.drop(conv_key)

    # Inject cached reasoning_content into ALL assistant messages missing it (DeepSeek requirement)
    if isinstance(input_data, list):
        cached_rc = _reasoning_cache.get(conv_key)
        if cached_rc is not None:
            injected_count = 0
            for msg in messages:
                if (msg.get("role") == "assistant"
                    and not msg.get("reasoning_content")
                    and msg.get("tool_calls")):
                    msg["reasoning_content"] = cached_rc
                    injected_count += 1
            if injected_count:
                _app_log.info("[responses] INJECTED rc key=%s into %d asst-with-tool msgs len=%d", conv_key, injected_count, len(cached_rc))
        else:
            for msg in messages:
                if (msg.get('role') == 'assistant'
                    and 'reasoning_content' not in msg
                    and msg.get('tool_calls')):
                    msg['reasoning_content'] = ''
            _app_log.debug("[responses] CACHE MISS key=%s available_keys=%s", conv_key, str(list(_reasoning_cache.keys())))

    try:
        allowed_params = {
            "top_p", "presence_penalty", "frequency_penalty", "stop",
            "tools", "tool_choice", "response_format", "user"
        }
        extra = {key: body[key] for key in allowed_params if key in body}
        # Convert tools from Responses API format to Chat Completions format
        if "tools" in extra and isinstance(extra["tools"], list):
            extra["tools"] = _convert_responses_tools(extra["tools"])
        # MiniMax rejects tool_choice="auto" (error 2013). Strip it for
        # providers that don't support this parameter.
        if extra.get("tool_choice") == "auto":
            extra.pop("tool_choice")

        # 检测目标提供商是否为 Anthropic 类型，若是则使用直通避开 liteLLM 双重格式转换
        from app.database import get_provider as _get_prov, find_provider_by_model as _find
        if provider_id:
            provider_info = _get_prov(provider_id)
        else:
            provider_info = _find(model)
        is_anthropic = provider_info and provider_info.get("provider_type") == "anthropic"

        if is_anthropic:
            # 将 Chat Completions 消息转换为 Anthropic 原生格式
            anthropic_msgs, system_text = _openai_messages_to_anthropic(messages, instructions or "")
            # 构建 Anthropic 请求体
            anthropic_body = {"system": system_text} if system_text else {}
            anthropic_tools = extra.get("tools")
            if anthropic_tools and isinstance(anthropic_tools, list):
                anthropic_body["tools"] = [
                    {"name": t["function"]["name"],
                     "description": t["function"].get("description", ""),
                     "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}})}
                    for t in anthropic_tools
                    if isinstance(t, dict) and t.get("function", {}).get("name")
                ]

            # 流式：liteLLM 的 Anthropic 路径会输出 OpenAI 格式 chunk，
            # _stream_responses 能正确处理。直通输出的是 Anthropic SSE 而非
            # Responses SSE，Codex 无法解析。
            if stream:
                return StreamingResponse(
                    _stream_responses(
                        model=model,
                        messages=messages,
                        provider_id=provider_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        username=username,
                        api_key_value=api_key_value,
                        instructions=instructions,
                        requested_model=requested_model,
                        conv_key=conv_key,
                        **extra
                    ),
                    media_type="text/event-stream"
                )

            # 非流式：直接 HTTP 调用 Anthropic 端点，避开 liteLLM 双重格式转换
            response = await _anthropic_passthrough(
                provider_info, anthropic_msgs, anthropic_body,
                max_tokens, temperature, model
            )
        else:
            if stream:
                return StreamingResponse(
                    _stream_responses(
                        model=model,
                        messages=messages,
                        provider_id=provider_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        username=username,
                        api_key_value=api_key_value,
                        instructions=instructions,
                        requested_model=requested_model,
                        conv_key=conv_key,
                        **extra
                    ),
                    media_type="text/event-stream"
                )

            response = await anyio.to_thread.run_sync(
                lambda: create_chat_completion(
                    model=model,
                    messages=messages,
                    provider_id=provider_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra
                )
            )
        choice = response.choices[0]
        message = getattr(choice, "message", {})
        # _anthropic_passthrough 返回 dict 类型 message，liteLLM 返回对象类型，
        # 统一用 _attr 兼容两种取值方式
        content = _strip_think_tags(_attr(message, "content", "") or "")
        reasoning_content = _attr(message, "reasoning_content", None)
        tool_calls = _attr(message, "tool_calls", None)
        usage = usage_dict(response)
        _log_request(username, api_key_value, model, provider_id or "", "responses", True, usage.get("total_tokens", 0), requested_model)
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, usage.get("total_tokens", 0))

        # Cache reasoning_content for multi-turn replay (DeepSeek thinking mode requires echo-back)
        if reasoning_content:
            _reasoning_cache[conv_key] = reasoning_content
            _app_log.info("[responses_nonstream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d",
                          conv_key, len(reasoning_content),
                          usage.get("prompt_cache_hit_tokens", 0), usage.get("prompt_cache_miss_tokens", 0))

        resp_id = f"resp_{int(time.time())}"
        msg_id = f"msg_{int(time.time())}"
        output = []
        # When thinking mode consumed all tokens before generating visible content,
        # fall back to reasoning_content as the response.
        if not content and reasoning_content:
            content = reasoning_content
        if content:
            msg_out = {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}]
            }
            if reasoning_content:
                msg_out["reasoning_content"] = reasoning_content
            output.append(msg_out)
        if tool_calls:
            _tool_log.info("[responses_endpoint] raw tool_calls count=%d model=%s", len(tool_calls), model)
            for tc in tool_calls:
                _tool_log.debug("[responses_endpoint] raw tc type=%s repr=%s", type(tc).__name__, repr(tc))
                tc_id = getattr(tc, "id", "") if hasattr(tc, "id") else tc.get("id", "")
                if tc_id is not None and not isinstance(tc_id, str):
                    tc_id = str(tc_id)
                tc_idx = getattr(tc, "index", 0) if hasattr(tc, "index") else tc.get("index", 0)
                if int(tc_idx) < 0:
                    _tool_log.info("[responses_endpoint] FILTERED spurious: id=%s idx=%s", tc_id, tc_idx)
                    continue
                tc_func = getattr(tc, "function", None) if hasattr(tc, "function") else tc.get("function", {})
                fn_name = getattr(tc_func, "name", "") if hasattr(tc_func, "name") else tc_func.get("name", "")
                fn_args = getattr(tc_func, "arguments", "") if hasattr(tc_func, "arguments") else tc_func.get("arguments", "")
                _tool_log.info("[responses_endpoint] fn_name=%s args_raw=%s", fn_name, fn_args)
                if fn_args and "undefined" in fn_args:
                    fn_args = _sanitize_args(fn_args)
                    _tool_log.info("[responses_endpoint] args SANITIZED -> %s", fn_args)
                output.append({
                    "type": "function_call",
                    "id": tc_id or f"fc_{int(time.time())}",
                    "call_id": tc_id or f"call_{int(time.time())}",
                    "name": fn_name,
                    "arguments": fn_args,
                    "status": "completed"
                })
        return {
            "id": resp_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": model,
            "output": output,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
        }
    except Exception as e:
        _log_request(username, api_key_value, "-", provider_id or "", "responses", False, 0, requested_model)
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", str(e))
        raise HTTPException(status_code=500, detail=_friendly_error_msg(e))
