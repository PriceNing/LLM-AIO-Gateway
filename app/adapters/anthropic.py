import json
import asyncio

import httpx
from fastapi import HTTPException

from app.core.output import InternalOutputMessage, InternalToolCallOutput
from app.core.types import InternalRequest
from app.database import parse_model_id
from app.protocols.ir import ir_to_anthropic_messages
from app.core.text import strip_billing_header
from app.config import get_default
from app.services.logger import get_logger


_app_log = get_logger("app")


def _anthropic_message_url(api_base: str) -> str:
    base = (api_base or "").rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _build_anthropic_request_body(
    provider_info: dict,
    messages: list,
    body: dict,
    max_tokens: int,
    temperature,
    model: str,
    *,
    stream: bool = False,
    tool_format: str = "normalized",
) -> dict:
    mid = parse_model_id(model)
    req_body = {
        "model": mid.model_name,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if stream:
        req_body["stream"] = True
    if temperature is not None:
        req_body["temperature"] = temperature

    system = strip_billing_header(body.get("system"))
    if system:
        req_body["system"] = system

    tools = body.get("tools")
    if tools:
        if tool_format == "native_strip_type":
            req_body["tools"] = [
                {key: value for key, value in tool.items() if key != "type"}
                for tool in tools
                if isinstance(tool, dict)
            ]
        else:
            req_body["tools"] = [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("input_schema", {"type": "object", "properties": {}}),
                }
                for tool in tools
                if isinstance(tool, dict) and tool.get("name")
            ]

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in ("auto", "any"):
            req_body["tool_choice"] = {"type": choice_type}
        elif choice_type == "tool" and tool_choice.get("name"):
            req_body["tool_choice"] = {"type": "tool", "name": tool_choice["name"]}

    provider_options = provider_info.get("provider_options", {}) or {}
    thinking = provider_options.get("thinking")
    if thinking == "disabled":
        req_body["thinking"] = {"type": "disabled"}
    elif thinking == "enabled":
        thinking_config = _enabled_thinking_config(max_tokens, provider_options)
        if thinking_config:
            req_body["thinking"] = thinking_config
            req_body.pop("temperature", None)
    return req_body


def _enabled_thinking_config(max_tokens: int, provider_options: dict) -> dict | None:
    try:
        budget = int(provider_options.get("thinking_budget_tokens") or get_default("anthropic_thinking_budget_tokens", 1024))
    except (TypeError, ValueError):
        budget = 1024
    budget = max(1024, budget)
    try:
        token_limit = int(max_tokens or 0)
    except (TypeError, ValueError):
        token_limit = 0
    if token_limit <= 1024:
        _app_log.warning(
            "[anthropic] skipping thinking: max_tokens=%s is too small for budget_tokens>=1024",
            token_limit,
        )
        return None
    if budget >= token_limit:
        budget = token_limit - 1
    return {"type": "enabled", "budget_tokens": budget}


def _http_exception_from_upstream(status_code: int, message: str) -> HTTPException:
    if status_code == 429:
        mapped = 429
    elif 400 <= status_code < 500:
        mapped = status_code
    else:
        mapped = 502
    return HTTPException(status_code=mapped, detail=f"Upstream: {message}")


def _anthropic_headers(provider_info: dict) -> dict:
    return {
        "x-api-key": provider_info.get("api_key") or "sk-no-auth",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }


def provider_request_timeout(provider_info: dict, default: int = 120) -> int:
    try:
        return max(1, min(3600, int(provider_info.get("request_timeout") or default)))
    except (TypeError, ValueError):
        return default


def provider_retry_count(provider_info: dict) -> int:
    try:
        return max(0, min(10, int(provider_info.get("retry_count") or 0)))
    except (TypeError, ValueError):
        return 0


def provider_retry_backoff(provider_info: dict) -> float:
    try:
        return max(0.0, min(60.0, float(provider_info.get("retry_backoff") or 0.5)))
    except (TypeError, ValueError):
        return 0.5


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def _upstream_error_message(resp) -> str:
    try:
        err_body = resp.json()
        return err_body.get("error", {}).get("message", resp.text[:300])
    except Exception:
        return resp.text[:300] or f"HTTP {resp.status_code}"


def _finish_reason_from_anthropic(stop_reason: str) -> str:
    return {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls"}.get(stop_reason or "", "stop")


def _anthropic_response_to_internal(data: dict) -> InternalOutputMessage:
    content_blocks = data.get("content", [])
    text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
    reasoning_parts = []
    for block in content_blocks:
        block_type = block.get("type")
        if block_type in ("thinking", "redacted_thinking", "reasoning"):
            reasoning_text = block.get("thinking") or block.get("text") or block.get("reasoning") or ""
            if reasoning_text:
                reasoning_parts.append(reasoning_text)
    top_level_reasoning = data.get("reasoning_content") or data.get("thinking") or ""
    if top_level_reasoning:
        reasoning_parts.append(str(top_level_reasoning))
    tool_uses = [block for block in content_blocks if block.get("type") == "tool_use"]
    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    tool_calls = []
    for tool_use in tool_uses:
        tool_id = str(tool_use.get("id", ""))
        tool_calls.append(InternalToolCallOutput(
            id=tool_id,
            call_id=tool_id if tool_id.startswith("call_") else (f"call_{tool_id}" if tool_id else ""),
            name=tool_use.get("name", ""),
            arguments=json.dumps(tool_use.get("input", {}), ensure_ascii=False),
            raw=dict(tool_use),
        ))
    text = "\n".join(text_parts)
    reasoning = "\n".join(reasoning_parts)
    reasoning_signature = ""
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "thinking" and block.get("signature"):
            reasoning_signature = str(block.get("signature") or "")
            break
    if not text and not tool_calls:
        _app_log.warning(
            "[anthropic_output_adapter] empty output stop_reason=%s content_types=%s raw_keys=%s",
            data.get("stop_reason", ""),
            [block.get("type") for block in content_blocks if isinstance(block, dict)],
            list(data.keys()),
        )
    return InternalOutputMessage(
        role="assistant",
        text=text,
        reasoning=reasoning,
        reasoning_signature=reasoning_signature,
        tool_calls=tool_calls,
        finish_reason=_finish_reason_from_anthropic(data.get("stop_reason", "end_turn")),
        usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        raw=data,
    )


def anthropic_body_from_internal(internal: InternalRequest) -> tuple[list, dict]:
    """Convert an internal request into native Anthropic messages and request extras."""
    if not internal.messages:
        raise ValueError("InternalRequest.messages is required for Anthropic adapter")
    anthropic_messages, system_text = ir_to_anthropic_messages(internal.messages)
    body = {"system": system_text} if system_text else {}
    tools = internal.anthropic_tools()
    if tools:
        body["tools"] = tools
    if internal.tool_choice is not None:
        body["tool_choice"] = internal.tool_choice
    return anthropic_messages, body


async def anthropic_messages_completion_for_internal(provider_info: dict, internal: InternalRequest):
    messages, body = anthropic_body_from_internal(internal)
    return await anthropic_messages_completion(
        provider_info,
        messages,
        body,
        internal.max_tokens,
        internal.temperature,
        internal.target_model,
    )


async def anthropic_messages_completion(
    provider_info: dict,
    messages: list,
    body: dict,
    max_tokens: int,
    temperature,
    model: str,
):
    """Call an Anthropic-compatible Messages endpoint and return internal output."""
    req_body = _build_anthropic_request_body(
        provider_info,
        messages,
        body,
        max_tokens,
        temperature,
        model,
    )
    timeout = provider_request_timeout(provider_info, 120)
    retries = provider_retry_count(provider_info)
    backoff = provider_retry_backoff(provider_info)
    last_exc = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(retries + 1):
            try:
                resp = await client.post(
                    _anthropic_message_url(provider_info.get("api_base") or ""),
                    headers=_anthropic_headers(provider_info),
                    json=req_body,
                )
                if resp.status_code == 200:
                    return _anthropic_response_to_internal(resp.json())
                if attempt < retries and _is_retryable_status(resp.status_code):
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue
                raise _http_exception_from_upstream(resp.status_code, _upstream_error_message(resp))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < retries:
                    await asyncio.sleep(backoff * (2 ** attempt))
                    continue
                raise
    raise last_exc or RuntimeError("Anthropic upstream request failed")
