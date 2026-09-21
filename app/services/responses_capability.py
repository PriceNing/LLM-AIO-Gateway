"""Native Responses 上游能力状态管理（模型能力边界）。

从 router/proxy.py 迁入：/responses 支持度的探测、TTL 新鲜度、负向缓存与
降级判断属于上游模型能力管理（与 services/discovery、database 的
model_responses_capability 表配套），不属于端点编排。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.config import get_default
from app.core.policy import RouteTarget
from app.core.text import error_detail_for_log
from app.database import get_model_responses_capability, set_model_responses_capability
from app.protocols.ingress import responses_to_internal
from app.services.routing_targets import provider_for_log, resolve_provider
from app.adapters.responses import post_native_response

RESPONSES_CAPABILITY_PROBE_MARKER = "auto_probe_v2"

def responses_capability_is_fresh(capability: dict | None) -> bool:
    if not capability or not capability.get("responses_expires_at"):
        return False
    try:
        return datetime.fromisoformat(capability["responses_expires_at"]) > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return False



def responses_tools_capability_is_fresh(capability: dict | None) -> bool:
    if not capability or not capability.get("responses_tools_expires_at"):
        return False
    try:
        return datetime.fromisoformat(capability["responses_tools_expires_at"]) > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return False



def responses_capability_expiry(status: str) -> str:
    ttl_key = {
        "supported": "responses_capability_supported_ttl",
        "unsupported": "responses_capability_unsupported_ttl",
    }.get(status, "responses_capability_transient_ttl")
    fallback = 604800 if status == "supported" else 21600 if status == "unsupported" else 300
    return (datetime.now(timezone.utc) + timedelta(seconds=max(0, int(get_default(ttl_key, fallback))))).isoformat()



def mark_model_responses_unknown(provider_id: str, model: str, error: Exception | str = "") -> None:
    """Invalidate native capability after a transient upstream failure.

    错误文本用 error_detail_for_log（含上游响应体），与 request_logs 的
    native_failure_message 同口径；否则 responses_error 只剩 httpx 摘要，排障时
    看不到真正的拒绝原因（审查 14 轮 #4）。
    """
    set_model_responses_capability(
        provider_id, model, status="unknown",
        expires_at=responses_capability_expiry("transient"),
        error=error_detail_for_log(error),
    )



def native_error_is_explicitly_unsupported(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    if status not in {400, 404, 405, 422, 501}:
        return False
    try:
        detail = response.text.lower()
    except Exception:
        detail = str(exc).lower()
    mentions_responses = any(marker in detail for marker in ("/responses", "responses api", "response api", "responses endpoint", "native responses"))
    rejects_protocol = any(marker in detail for marker in (
        "not supported", "unsupported", "not implemented", "unknown endpoint",
        "method not allowed", "unprocessable", "invalid request",
    ))
    if status in {404, 405, 501}:
        return mentions_responses or rejects_protocol
    if status == 422:
        return mentions_responses or rejects_protocol or not detail.strip()
    return mentions_responses and rejects_protocol



def native_response_target_supported(target: RouteTarget, *, stream: bool, required_tool_types: set[str], is_primary: bool, has_tools: bool = False) -> tuple[dict | None, str]:
    provider = resolve_provider(target.model, target.provider_id)
    if not provider or provider.get("provider_type") != "openai":
        return None, ""
    capability = get_model_responses_capability(provider.get("id") or target.provider_id, target.model)
    # Only a fresh explicit negative result prevents a real user request from
    # attempting native Responses. Unknown, expired, and transient results are
    # deliberately request-driven rechecks.
    if responses_capability_is_fresh(capability) and capability.get("responses_status") == "unsupported":
        return None, ""
    # Tool-shape negative: a request carrying tools skips native while a fresh
    # tool-level negative is cached; text/stream requests stay on native.
    if has_tools and responses_tools_capability_is_fresh(capability) and capability.get("responses_tools_status") == "unsupported":
        return None, ""
    # ``responses_tool_types`` is learned from successful response output.  It is
    # therefore positive evidence, not an exhaustive declaration of what an
    # upstream can do.  Treating an absent entry as unsupported prevented a newly
    # discovered native fallback from ever handling Codex tools (the generic
    # capability probe deliberately does not execute tools).
    #
    # Keep accepting ``required_tool_types`` here so callers document why they
    # selected native dispatch; explicit negative capability data can be added
    # later without changing this boundary.
    del stream, required_tool_types, is_primary
    return provider, provider_for_log(provider, target.provider_id)



async def probe_model_responses_capability(provider: dict, model: str) -> bool:
    """Probe native Responses support and cache the result per provider/model.

    A normal OpenAI-compatible provider is not proof of Responses support.  The
    probe intentionally requires a valid Responses object.  Explicit protocol
    rejections (and incomplete 422s) are cached as unsupported.  Generic 400s
    and transient 5xx/network failures stay unknown so a request-level validation
    error cannot disable native Responses for hours.
    """
    provider_id = str(provider.get("id") or "")
    probe = responses_to_internal({
        "model": model,
        "input": "capability probe",
        "stream": False,
        "max_output_tokens": max(1, int(get_default("responses_capability_probe_max_output_tokens", 16))),
    })
    probe.target_model = model
    probe.provider_id = provider_id
    try:
        probe_provider = dict(provider)
        probe_provider["request_timeout"] = max(1, int(get_default("responses_capability_probe_timeout", 8)))
        payload = await post_native_response(probe_provider, probe)
        supported = isinstance(payload, dict) and payload.get("object") == "response"
        if supported:
            set_model_responses_capability(
                provider_id, model, status="supported",
                streaming=False, streaming_status="unknown",
                error=RESPONSES_CAPABILITY_PROBE_MARKER,
                expires_at=responses_capability_expiry("supported"),
            )
        return supported
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
        # Incomplete 422s are a practical unsupported signal for Chat-only or
        # partial Responses proxies.  Generic 400s are not: request-level
        # validation errors must not be cached as a missing protocol.  Other
        # 4xx responses are only negative when the body explicitly identifies
        # the Responses endpoint/protocol as unsupported.
        explicitly_unsupported = native_error_is_explicitly_unsupported(exc)
        if explicitly_unsupported or status == 422:
            set_model_responses_capability(
                provider_id, model, status="unsupported",
                expires_at=responses_capability_expiry("unsupported"),
                error=error_detail_for_log(exc),
            )
        else:
            mark_model_responses_unknown(provider_id, model, exc)
        return False



async def native_capability_for_request(provider: dict | None, model: str, has_tools: bool = False) -> bool:
    if not provider or provider.get("provider_type") != "openai":
        return False
    if provider.get("force_chat_completions"):
        return False
    provider_id = str(provider.get("id") or "")
    capability = get_model_responses_capability(provider_id, model)
    # 工具形态级负向：含 tools 的请求直接走 Chat（兼容性路径会按策略处理工具），
    # 文本/流式请求不受影响，继续原生。
    if has_tools and responses_tools_capability_is_fresh(capability) and capability.get("responses_tools_status") == "unsupported":
        return False
    if responses_capability_is_fresh(capability):
        status = capability.get("responses_status")
        if status == "supported":
            # Legacy optimistic supported rows are revalidated once.
            return capability.get("responses_error") == RESPONSES_CAPABILITY_PROBE_MARKER
        if status == "unsupported":
            return False
        if status == "unknown":
            # A recent real request or capability probe already failed for a
            # transient/request-specific reason. Honor the transient TTL as a
            # native retry backoff and use the Chat compatibility path meanwhile.
            return False
    return await probe_model_responses_capability(provider, model)

