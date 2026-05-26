import hashlib
import json
import threading
import time
import uuid
from collections import deque
from typing import Optional

import anyio
from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from app.database import (
    get_providers, find_user_by_api_key,
    increment_global_stats, increment_user_usage, get_db,
    parse_model_id, add_request_record,
)
from app.core.text import friendly_error_msg, mask_key, message_text
from app.protocols.ingress import (
    anthropic_messages_to_internal,
    chat_completions_to_internal,
    completions_to_internal,
    responses_to_internal,
)
from app.core.types import InternalMessage
from app.core.policy import apply_routing_rules, prepare_request_policy
from app.adapters.anthropic import (
    anthropic_body_from_internal,
    anthropic_messages_completion_for_internal,
)
from app.adapters.openai import chat_kwargs_from_internal, chat_messages_from_internal
from app.adapters.output import response_to_internal_output
from app.adapters.anthropic_streaming import iter_anthropic_output_events
from app.adapters.openai_streaming import iter_openai_chat_output_events
from app.protocols.egress import (
    render_anthropic_messages_sse,
    render_anthropic_message,
    render_chat_completion,
    render_chat_completions_sse,
    render_completion,
    render_completions_sse,
    render_response,
    render_responses_error_sse,
    render_responses_sse,
)
from app.services.lite_llm import create_chat_completion
from app.services.preprocessing import has_image_content, preprocess_messages
from app.services.logger import get_logger
from app.config import get_default

_access_log = get_logger("access")
_error_log = get_logger("error")
_tool_log = get_logger("tool_calls")
_req_log = get_logger("request")
_app_log = get_logger("app")

router = APIRouter()

# Rolling log of recent requests for the admin stats dashboard
_request_log = deque(maxlen=get_default("request_log_max", 200))
_request_log_lock = threading.Lock()


def _log_request(username: str, api_key: str, model: str, provider_id: str,
                 endpoint: str, success: bool, tokens: int,
                 requested_model: str = "") -> None:
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "full_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "username": username,
        "api_key": mask_key(api_key),
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


async def _policy_preprocess_request(internal, model: str, provider_id: str, requested_model: str):
    check_model = requested_model or model
    has_img = has_image_content(internal.messages)
    _app_log.debug("[preprocess] CHECK req_model=%s target_model=%s has_image=%s msg_count=%d",
                   check_model, model, has_img, len(internal.messages))

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
    _app_log.debug("[preprocess] DB lookup model='%s' -> row=%s", check_model, dict(row) if row else None)
    if not row or not row["preprocessor"]:
        if has_img:
            _app_log.warning("[preprocess] images detected for model=%s but preprocessor not enabled", check_model)
        return False

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
        return False
    preprocessor_config["id"] = preprocessor_id
    await preprocess_messages(internal.messages, preprocessor_config)
    return has_img


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
        """Internal - caller must hold _lock."""
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
_reasoning_tool_cache = TTLDict(
    ttl_seconds=get_default("reasoning_cache_ttl", 1800),
    max_size=get_default("reasoning_cache_max_size", 1000)
)
_reasoning_tool_global_cache = TTLDict(
    ttl_seconds=get_default("reasoning_cache_ttl", 1800),
    max_size=get_default("reasoning_cache_max_size", 1000)
)
_response_chain_cache = TTLDict(
    ttl_seconds=get_default("reasoning_cache_ttl", 1800),
    max_size=get_default("reasoning_cache_max_size", 1000)
)
def _ir_tool_message_count(messages: list[InternalMessage]) -> int:
    return sum(1 for msg in messages if any(part.kind == "tool_call" for part in msg.parts))


def _ir_reasoning_message_count(messages: list[InternalMessage]) -> int:
    return sum(1 for msg in messages if any(part.kind == "reasoning" for part in msg.parts))


def _remember_response_chain_key(response_id: str, conv_key: str) -> None:
    if response_id and conv_key:
        _response_chain_cache[str(response_id)] = conv_key


def _conversation_cache_key(api_key: str, messages: list, response_chain_id: str = "") -> str:
    """Build a cache key that isolates different conversations sharing the same API key."""
    if response_chain_id:
        chained_key = _response_chain_cache.get(str(response_chain_id))
        if chained_key:
            return chained_key
    user_fingerprints = []
    for msg in messages:
        if isinstance(msg, InternalMessage):
            if msg.role != "user":
                continue
            text = _ir_text_for_cache(msg)
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


def _ir_text_for_cache(message: InternalMessage) -> str:
    parts = []
    for part in message.parts:
        if part.kind == "text" and part.text:
            parts.append(part.text)
    return "\n".join(parts)


def _remember_reasoning_content(conv_key: str, reasoning_content: str, tool_call_ids=None) -> None:
    if not reasoning_content:
        return
    _reasoning_cache[conv_key] = reasoning_content
    ids = [str(tid) for tid in (tool_call_ids or []) if tid]
    if not ids:
        return
    tool_map = dict(_reasoning_tool_cache.get(conv_key, {}) or {})
    for tid in ids:
        tool_map[tid] = reasoning_content
        _reasoning_tool_global_cache[tid] = reasoning_content
    while len(tool_map) > 200:
        oldest = next(iter(tool_map))
        tool_map.pop(oldest, None)
    _reasoning_tool_cache[conv_key] = tool_map


def _reasoning_context(conv_key: str, messages: list[InternalMessage] | None = None) -> tuple[str | None, dict]:
    tool_map = _reasoning_tool_cache.get(conv_key, {}) or {}
    if messages:
        tool_map = _merge_global_reasoning_context(messages, tool_map)
    return _reasoning_cache.get(conv_key), tool_map


def _merge_global_reasoning_context(messages: list[InternalMessage], tool_map: dict) -> dict:
    merged = dict(tool_map or {})
    for msg in messages or []:
        if not isinstance(msg, InternalMessage):
            continue
        for part in msg.parts:
            if part.kind != "tool_result" or not part.tool_call_id or part.tool_call_id in merged:
                continue
            rc = _reasoning_tool_global_cache.get(part.tool_call_id)
            if rc:
                merged[part.tool_call_id] = rc
    return merged


async def _record_streaming_events(
    events,
    *,
    conv_key: str,
    tool_only_turns=None,
    remember_reasoning_content=None,
):
    accumulated_reasoning = ""
    has_text = False
    has_tools = False
    tool_ids = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}

    def finalize() -> None:
        nonlocal accumulated_reasoning
        if accumulated_reasoning and remember_reasoning_content is not None:
            remember_reasoning_content(conv_key, accumulated_reasoning, tool_ids)
            _app_log.debug("[stream_policy] STORED rc key=%s len=%d tool_ids=%d", conv_key[:60], len(accumulated_reasoning), len(tool_ids))
            accumulated_reasoning = ""
        if tool_only_turns is not None:
            if has_tools and not has_text:
                count = tool_only_turns.increment(conv_key)
                _app_log.debug("[stream_policy] tool_only_increment key=%s count=%d", conv_key[:60], count)
            else:
                tool_only_turns.reset(conv_key)
                _app_log.debug("[stream_policy] tool_only_reset key=%s has_tools=%s has_text=%s", conv_key[:60], has_tools, has_text)

    finalized = False
    async for event in events:
        if event.kind == "text_delta" and event.text:
            has_text = True
        elif event.kind == "reasoning_delta" and event.reasoning:
            accumulated_reasoning += event.reasoning
        elif event.kind == "tool_call_start":
            has_tools = True
            if event.tool_call_id:
                tool_ids.append(event.tool_call_id)
        elif event.kind == "usage":
            usage.update(event.usage)
        elif event.kind == "message_done":
            finalize()
            finalized = True
        yield event

    if not finalized:
        finalize()


async def _stream_internal_output(
    *,
    events,
    endpoint: str,
    model: str,
    username: str,
    api_key_value: str,
    provider_id: str,
    requested_model: str,
    previous_response_id: str | None = None,
    conv_key: str = "",
    remember_response_chain_key=None,
    remember_reasoning_content=None,
    tool_only_turns=None,
):
    total_tokens = 0
    _app_log.debug(
        "[stream_orchestrator] START endpoint=%s provider=%s model=%s requested=%s conv_key=%s previous_response_id=%s",
        endpoint,
        provider_id or "",
        model,
        requested_model,
        conv_key[:60],
        previous_response_id or "",
    )

    async def metered_events():
        nonlocal total_tokens
        async for event in _record_streaming_events(
            events,
            conv_key=conv_key,
            remember_reasoning_content=remember_reasoning_content,
            tool_only_turns=tool_only_turns,
        ):
            if event.kind == "usage":
                total_tokens = event.usage.get("total_tokens", total_tokens) or total_tokens
            yield event

    response_id = f"resp_{uuid.uuid4().hex}" if endpoint == "responses" else None
    if endpoint == "responses" and remember_response_chain_key is not None and conv_key:
        remember_response_chain_key(response_id, conv_key)

    try:
        if endpoint == "chat_completions":
            async for line in render_chat_completions_sse(metered_events(), model=model):
                yield line
        elif endpoint == "completions":
            async for line in render_completions_sse(metered_events(), model=model):
                yield line
        elif endpoint == "messages":
            async for line in render_anthropic_messages_sse(metered_events(), model=model):
                yield line
        elif endpoint == "responses":
            async for line in render_responses_sse(
                metered_events(),
                model=model,
                previous_response_id=previous_response_id,
                response_id=response_id,
            ):
                yield line
        _log_request(username, api_key_value, model, provider_id or "", endpoint, True, total_tokens, requested_model)
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, total_tokens)
        _app_log.debug("[stream_orchestrator] DONE endpoint=%s provider=%s model=%s total_tokens=%d", endpoint, provider_id or "", model, total_tokens)
    except Exception as e:
        error_msg = friendly_error_msg(e)
        _error_log.error("[%s_stream] %s", endpoint, str(e))
        _log_request(username, api_key_value, "-", provider_id or "", endpoint, False, 0, requested_model)
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        if tool_only_turns is not None and conv_key:
            tool_only_turns.reset(conv_key)
        if endpoint == "responses":
            async for line in render_responses_error_sse(model=model, message=error_msg, previous_response_id=previous_response_id):
                yield line
        elif endpoint == "messages":
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'server_error', 'message': error_msg}})}\n\n"
        else:
            yield f"data: {json.dumps({'error': {'message': error_msg, 'type': 'server_error'}})}\n\n"
            yield "data: [DONE]\n\n"

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
        return ["*"]  # not configured -> unrestricted
    if "*" in key_models:
        return ["*"]
    return key_models  # explicit list, empty = deny all

def ensure_model_allowed(user: dict, api_key: dict, model: str) -> None:
    allowed = allowed_models_for(user, api_key)
    if "*" in allowed:
        return
    # ModelId.__eq__ handles composite/simple matching in both directions
    if parse_model_id(model) in allowed:
        return
    raise HTTPException(status_code=403, detail=f"Model '{model}' is not allowed for this API key")

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
                    # Check allow-list support for composite IDs, simple model IDs, and wildcard
                    if "*" not in allowed and model["id"] not in allowed and composite_id not in allowed:
                        continue
                    entry = {
                        "id": composite_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": provider["name"],
                        "provider": provider["id"]
                    }
                    # Models with a vision preprocessor advertise image support so clients such as Codex/OpenCode
                    # will send images, which the gateway intercepts and describes with a vision model.
                    # Set several common fields because different clients check different keys.
                    if model.get("preprocessor"):
                        entry["supports_vision"] = True
                        entry["image_support"] = True
                        entry["multimodal"] = True
                    models.append(entry)

    return {"object": "list", "data": models}

@router.post("/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)

    body = await request.json()
    internal = chat_completions_to_internal(body)
    model = internal.target_model
    temperature = internal.temperature
    max_tokens = internal.max_tokens
    provider_id = internal.provider_id
    stream = internal.stream

    username = user.get("username", "legacy")
    api_key_value = api_key.get("key", "")
    _log_request_body(username, model, "chat", body)

    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if not internal.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    ensure_model_allowed(user, api_key, model)

    requested_model = model
    policy = await prepare_request_policy(
        internal,
        username=username,
        api_key_value=api_key_value,
        preprocess_request=_policy_preprocess_request,
        conversation_cache_key=_conversation_cache_key,
        reasoning_context=_reasoning_context,
        tool_only_turns=_tool_only_turns,
        tool_only_limit=TOOL_ONLY_LIMIT,
        log_label="chat",
    )
    model = internal.target_model
    provider_id = internal.provider_id
    conv_key = policy.conv_key

    try:
        extra = chat_kwargs_from_internal(internal)
        from app.database import get_provider as _get_prov, find_provider_by_model as _find
        if provider_id:
            provider_info = _get_prov(provider_id)
        else:
            provider_info = _find(model)
        is_anthropic = provider_info and provider_info.get("provider_type") == "anthropic"
        adapter_provider_id = provider_info.get("id", provider_id or "") if provider_info else provider_id

        if stream:
            if is_anthropic:
                anthropic_msgs, anthropic_body = anthropic_body_from_internal(internal)
                adapter_provider_id = provider_info.get("id", provider_id or "")
                events = iter_anthropic_output_events(
                    provider_info=provider_info,
                    messages=anthropic_msgs,
                    body=anthropic_body,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model=model,
                )
            else:
                events = iter_openai_chat_output_events(
                    model=model,
                    messages=chat_messages_from_internal(internal),
                    provider_id=provider_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra=extra,
                )
            return StreamingResponse(
                _stream_internal_output(
                    events=events,
                    endpoint="chat_completions",
                    model=model,
                    username=username,
                    api_key_value=api_key_value,
                    provider_id=adapter_provider_id,
                    requested_model=requested_model,
                    conv_key=conv_key,
                    remember_reasoning_content=_remember_reasoning_content,
                    tool_only_turns=_tool_only_turns,
                ),
                media_type="text/event-stream"
            )

        if is_anthropic:
            output = await anthropic_messages_completion_for_internal(provider_info, internal)
            adapter_provider_id = provider_info.get("id", provider_id or "")
        else:
            response = await anyio.to_thread.run_sync(
                lambda: create_chat_completion(
                    model=model,
                    messages=chat_messages_from_internal(internal),
                    provider_id=provider_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra
                )
            )
            output = response_to_internal_output(response)
        if output.reasoning:
            _remember_reasoning_content(conv_key, output.reasoning, [tool.id for tool in output.tool_calls])
            _app_log.debug("[chat_nonstream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d",
                          conv_key[:40], len(output.reasoning),
                          output.usage.get("prompt_cache_hit_tokens", 0), output.usage.get("prompt_cache_miss_tokens", 0))

        if output.tool_calls and not output.text:
            _tool_only_turns.increment(conv_key)
        else:
            _tool_only_turns.reset(conv_key)

        _log_request(username, api_key_value, model, adapter_provider_id or "", "chat_completions", True, output.usage.get("total_tokens", 0), requested_model)
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, output.usage.get("total_tokens", 0))
        return render_chat_completion(output, model=model)
    except Exception as e:
        _error_log.error("[chat] %s", str(e))
        _log_request(username, api_key_value, requested_model, provider_id or "", "chat_completions", False, 0, requested_model)
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        raise HTTPException(status_code=500, detail=friendly_error_msg(e))

@router.post("/completions")
async def completions(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)

    body = await request.json()
    internal = completions_to_internal(body)
    model = internal.target_model
    provider_id = internal.provider_id
    stream = internal.stream
    temperature = internal.temperature
    max_tokens = internal.max_tokens

    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    ensure_model_allowed(user, api_key, model)

    username = user.get("username", "legacy")
    _log_request_body(username, model, "completions", body)
    api_key_value = api_key.get("key", "")
    requested_model = model

    policy = await prepare_request_policy(
        internal,
        username=username,
        api_key_value=api_key_value,
        preprocess_request=_policy_preprocess_request,
        conversation_cache_key=_conversation_cache_key,
        reasoning_context=None,
        normalize=True,
        log_label="completions",
    )
    model = internal.target_model
    provider_id = internal.provider_id
    conv_key = policy.conv_key

    try:
        from app.database import get_provider as _get_prov, find_provider_by_model as _find
        if provider_id:
            provider_info = _get_prov(provider_id)
        else:
            provider_info = _find(model)
        is_anthropic = provider_info and provider_info.get("provider_type") == "anthropic"
        adapter_provider_id = provider_info.get("id", provider_id or "") if provider_info else provider_id

        if stream:
            if is_anthropic:
                anthropic_msgs, anthropic_body = anthropic_body_from_internal(internal)
                adapter_provider_id = provider_info.get("id", provider_id or "")
                events = iter_anthropic_output_events(
                    provider_info=provider_info,
                    messages=anthropic_msgs,
                    body=anthropic_body,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model=model,
                )
            else:
                events = iter_openai_chat_output_events(
                    model=model,
                    messages=chat_messages_from_internal(internal),
                    provider_id=adapter_provider_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra=chat_kwargs_from_internal(internal),
                )
            return StreamingResponse(
                _stream_internal_output(
                    events=events,
                    endpoint="completions",
                    model=model,
                    username=username,
                    api_key_value=api_key_value,
                    provider_id=adapter_provider_id,
                    requested_model=requested_model,
                    conv_key=conv_key,
                ),
                media_type="text/event-stream"
            )

        if is_anthropic:
            output = await anthropic_messages_completion_for_internal(provider_info, internal)
            adapter_provider_id = provider_info.get("id", provider_id or "")
        else:
            response = await anyio.to_thread.run_sync(
                lambda: create_chat_completion(
                    model=model,
                    messages=chat_messages_from_internal(internal),
                    provider_id=adapter_provider_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **chat_kwargs_from_internal(internal),
                )
            )
            output = response_to_internal_output(response)
        _log_request(username, api_key_value, model, adapter_provider_id or "", "completions", True, output.usage.get("total_tokens", 0), requested_model)
    
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, output.usage.get("total_tokens", 0))
        return render_completion(output, model=model)
    except Exception as e:
        _log_request(username, api_key_value, "-", provider_id or "", "completions", False, 0, requested_model)

        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", str(e))
        raise HTTPException(status_code=500, detail=friendly_error_msg(e))

@router.post("/messages")
async def anthropic_messages(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)

    body = await request.json()
    model = body.get("model")
    anthropic_msgs = body.get("messages", [])
    provider_id = body.get("provider_id")
    stream = body.get("stream", False)
    previous_response_id = body.get("previous_response_id") or ""
    internal = anthropic_messages_to_internal({**body, "provider_id": provider_id})
    system_prompt = internal.system
    _app_log.debug("[ANTHRO_ENTRY] model=%s msgs=%d system=%s tools=%s",
                  model, len(anthropic_msgs),
                  "yes" if system_prompt else "no",
                  "yes" if body.get("tools") else "no")
    temperature = body.get("temperature")

    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    ensure_model_allowed(user, api_key, model)

    username = user.get("username", "legacy")
    api_key_value = api_key.get("key", "")
    requested_model = model

    route_model, route_provider = apply_routing_rules(username, api_key_value, model, model)
    routed_model = route_model
    routed_provider_id = route_provider or provider_id

    # Resolve the upstream adapter after ingress has normalized the client protocol.
    from app.database import get_provider as _get_prov, find_provider_by_model as _find
    if routed_provider_id:
        provider_info = _get_prov(routed_provider_id)
    else:
        provider_info = _find(routed_model)
    internal.target_model = routed_model
    if routed_provider_id:
        internal.provider_id = routed_provider_id
    policy = await prepare_request_policy(
        internal,
        username=username,
        api_key_value=api_key_value,
        preprocess_request=_policy_preprocess_request,
        conversation_cache_key=_conversation_cache_key,
        reasoning_context=_reasoning_context,
        normalize=False,
        log_label="messages",
    )
    model = internal.target_model
    provider_id = internal.provider_id
    previous_response_id = internal.previous_response_id
    max_tokens = internal.max_tokens
    temperature = internal.temperature
    system_prompt = internal.system
    _app_log.debug(
        "[messages] NORMALIZED anthropic(%d msgs) -> internal(%d msgs) system_prompt_len=%d tools=%s stream=%s max_tokens=%s model=%s provider_type=%s",
        len(anthropic_msgs), len(internal.messages), len(system_prompt) if system_prompt else 0,
        str(body.get("tools", [])[:10]) if body.get("tools") else "none",
        str(body.get("stream")), str(max_tokens), model,
        provider_info.get("provider_type") if provider_info else "unknown",
    )

    conv_key = policy.conv_key

    try:
        if stream:
            if provider_info and provider_info.get("provider_type") == "anthropic":
                anthropic_adapter_messages, anthropic_adapter_body = anthropic_body_from_internal(internal)
                events = iter_anthropic_output_events(
                    provider_info=provider_info,
                    messages=anthropic_adapter_messages,
                    body=anthropic_adapter_body,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model=model,
                )
                return StreamingResponse(
                    _stream_internal_output(
                        events=events,
                        endpoint="messages",
                        model=model,
                        username=username,
                        api_key_value=api_key_value,
                        provider_id=provider_id,
                        requested_model=requested_model,
                        conv_key=conv_key,
                        remember_reasoning_content=_remember_reasoning_content,
                    ),
                    media_type="text/event-stream"
                )
            adapter_messages = chat_messages_from_internal(internal)
            adapter_extra = chat_kwargs_from_internal(internal)
            events = iter_openai_chat_output_events(
                model=model,
                messages=adapter_messages,
                provider_id=provider_id,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=adapter_extra,
                strip_thinking=False,
            )
            return StreamingResponse(
                _stream_internal_output(
                    events=events,
                    endpoint="messages",
                    model=model,
                    username=username,
                    api_key_value=api_key_value,
                    provider_id=provider_id,
                    requested_model=requested_model,
                    conv_key=conv_key,
                    remember_reasoning_content=_remember_reasoning_content,
                ),
                media_type="text/event-stream"
            )

        if provider_info and provider_info.get("provider_type") == "anthropic":
            output = await anthropic_messages_completion_for_internal(provider_info, internal)
        else:
            adapter_messages = chat_messages_from_internal(internal)
            adapter_extra = chat_kwargs_from_internal(internal)
            _app_log.debug("[OPENAI_EXIT] model=%s msgs=%d tools=%s tool_choice=%s conv_key=%s",
                          model, len(adapter_messages), "yes" if adapter_extra.get("tools") else "no",
                          str(adapter_extra.get("tool_choice")), conv_key)
            response = await anyio.to_thread.run_sync(
                lambda: create_chat_completion(
                    model=model, messages=adapter_messages, provider_id=provider_id,
                    max_tokens=max_tokens, temperature=temperature,
                    **adapter_extra,
                )
            )
            output = response_to_internal_output(response)
        if output.reasoning:
            _remember_reasoning_content(conv_key, output.reasoning, [tool.id for tool in output.tool_calls])
            _app_log.debug("[messages_nonstream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d",
                          conv_key[:60], len(output.reasoning),
                          output.usage.get("prompt_cache_hit_tokens", 0), output.usage.get("prompt_cache_miss_tokens", 0))
        _log_request(username, api_key_value, model, provider_id or "", "messages", True, output.usage.get("total_tokens", 0), requested_model)
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, output.usage.get("total_tokens", 0))
        return render_anthropic_message(output, model=model)
    except Exception as e:
        _log_request(username, api_key_value, "-", provider_id or "", "messages", False, 0, requested_model)
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", str(e))
        raise HTTPException(status_code=500, detail=friendly_error_msg(e))


@router.post("/responses")
async def responses_endpoint(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)

    body = await request.json()
    internal = responses_to_internal(body)
    model = internal.target_model
    input_data = body.get("input", "")
    instructions = internal.metadata.get("instructions", "")
    temperature = internal.temperature
    max_tokens = internal.max_tokens
    provider_id = internal.provider_id
    stream = internal.stream
    previous_response_id = internal.previous_response_id

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
        _app_log.debug("[responses] model=%s stream=%s tools=%d input_len=%d instructions_len=%d input_types=%s", model, stream, tools_count, input_len, instructions_len, str(item_types))
    else:
        _app_log.debug("[responses] model=%s stream=%s tools=%d input_len=%d instructions_len=%d", model, stream, tools_count, input_len, instructions_len)

    # Check permission on requested model BEFORE routing
    requested_model = model
    ensure_model_allowed(user, api_key, requested_model)
    username = user.get("username", "legacy")
    api_key_value = api_key.get("key", "")

    if isinstance(input_data, str):
        pass
    elif isinstance(input_data, list):
        _app_log.debug(
            "[responses CONVERT] input_items=%d ir_messages=%d roles=%s tool_msgs=%d rc_msgs=%d",
            len(input_data),
            len(internal.messages),
            [m.role for m in internal.messages],
            _ir_tool_message_count(internal.messages),
            _ir_reasoning_message_count(internal.messages),
        )
    else:
        raise HTTPException(status_code=400, detail="input must be a string or list of messages")

    policy = await prepare_request_policy(
        internal,
        username=username,
        api_key_value=api_key_value,
        preprocess_request=_policy_preprocess_request,
        conversation_cache_key=_conversation_cache_key,
        reasoning_context=_reasoning_context if isinstance(input_data, list) else None,
        log_label="responses",
    )
    model = internal.target_model
    provider_id = internal.provider_id
    conv_key = policy.conv_key

    if isinstance(input_data, list):
        _app_log.debug(
            "[responses REASONING] injected=%d ir_messages=%d tool_msgs=%d rc_msgs=%d conv_key=%s",
            policy.reasoning_injected,
            len(internal.messages),
            _ir_tool_message_count(internal.messages),
            _ir_reasoning_message_count(internal.messages),
            conv_key[:60],
        )

    try:
        extra = dict(internal.extra)

        # Choose an upstream adapter after policy has finalized the internal request.
        from app.database import get_provider as _get_prov, find_provider_by_model as _find
        if provider_id:
            provider_info = _get_prov(provider_id)
        else:
            provider_info = _find(model)
        is_anthropic = provider_info and provider_info.get("provider_type") == "anthropic"

        if is_anthropic:
            anthropic_msgs, anthropic_body = anthropic_body_from_internal(internal)

            if stream:
                events = iter_anthropic_output_events(
                    provider_info=provider_info,
                    messages=anthropic_msgs,
                    body=anthropic_body,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model=model,
                )
                return StreamingResponse(
                    _stream_internal_output(
                        events=events,
                        endpoint="responses",
                        model=model,
                        username=username,
                        api_key_value=api_key_value,
                        provider_id=provider_id,
                        requested_model=requested_model,
                        previous_response_id=previous_response_id,
                        conv_key=conv_key,
                        remember_response_chain_key=_remember_response_chain_key,
                        remember_reasoning_content=_remember_reasoning_content,
                        tool_only_turns=_tool_only_turns,
                    ),
                    media_type="text/event-stream"
                )

            output = await anthropic_messages_completion_for_internal(provider_info, internal)
        else:
            adapter_messages = chat_messages_from_internal(internal)
            adapter_extra = chat_kwargs_from_internal(internal)
            if stream:
                events = iter_openai_chat_output_events(
                    model=model,
                    messages=adapter_messages,
                    provider_id=provider_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra=adapter_extra,
                )
                return StreamingResponse(
                    _stream_internal_output(
                        events=events,
                        endpoint="responses",
                        model=model,
                        username=username,
                        api_key_value=api_key_value,
                        provider_id=provider_id,
                        requested_model=requested_model,
                        previous_response_id=previous_response_id,
                        conv_key=conv_key,
                        remember_response_chain_key=_remember_response_chain_key,
                        remember_reasoning_content=_remember_reasoning_content,
                        tool_only_turns=_tool_only_turns,
                    ),
                    media_type="text/event-stream"
                )

            response = await anyio.to_thread.run_sync(
                lambda: create_chat_completion(
                    model=model,
                    messages=adapter_messages,
                    provider_id=provider_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **adapter_extra
                )
            )
            output = response_to_internal_output(response)
        if output.reasoning:
            _remember_reasoning_content(conv_key, output.reasoning, [tool.id for tool in output.tool_calls])
            _app_log.debug("[responses_nonstream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d",
                          conv_key, len(output.reasoning),
                          output.usage.get("prompt_cache_hit_tokens", 0), output.usage.get("prompt_cache_miss_tokens", 0))

        resp_id = f"resp_{uuid.uuid4().hex}"
        _remember_response_chain_key(resp_id, conv_key)
        _log_request(username, api_key_value, model, provider_id or "", "responses", True, output.usage.get("total_tokens", 0), requested_model)
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, output.usage.get("total_tokens", 0))
        return render_response(output, model=model, previous_response_id=previous_response_id, response_id=resp_id)
    except Exception as e:
        _log_request(username, api_key_value, "-", provider_id or "", "responses", False, 0, requested_model)
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", str(e))
        raise HTTPException(status_code=500, detail=friendly_error_msg(e))
