import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.output import InternalOutputEvent
from app.core.text import friendly_error_msg
from app.database import increment_global_stats, increment_user_usage
from app.protocols.egress import (
    render_anthropic_messages_sse,
    render_chat_completions_sse,
    render_completions_sse,
    render_responses_error_sse,
    render_responses_sse,
)
from app.services.logger import get_logger


_app_log = get_logger("app")
_error_log = get_logger("error")

RequestLogger = Callable[..., None]
RememberResponseChainKey = Callable[[str, str], None]
RememberReasoningContent = Callable[[str, str, Any], None]


async def record_streaming_events(
    events,
    *,
    conv_key: str,
    tool_only_turns=None,
    remember_reasoning_content: RememberReasoningContent | None = None,
):
    accumulated_reasoning = ""
    has_text = False
    has_tools = False
    tool_ids = []

    def finalize() -> None:
        nonlocal accumulated_reasoning
        if accumulated_reasoning and remember_reasoning_content is not None:
            remember_reasoning_content(conv_key, accumulated_reasoning, tool_ids)
            _app_log.debug(
                "[stream_policy] STORED rc key=%s len=%d tool_ids=%d",
                conv_key[:60],
                len(accumulated_reasoning),
                len(tool_ids),
            )
            accumulated_reasoning = ""
        if tool_only_turns is not None:
            if has_tools and not has_text:
                count = tool_only_turns.increment(conv_key)
                _app_log.debug("[stream_policy] tool_only_increment key=%s count=%d", conv_key[:60], count)
            else:
                tool_only_turns.reset(conv_key)
                _app_log.debug(
                    "[stream_policy] tool_only_reset key=%s has_tools=%s has_text=%s",
                    conv_key[:60],
                    has_tools,
                    has_text,
                )

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
        elif event.kind == "message_done":
            finalize()
            finalized = True
        yield event

    if not finalized:
        finalize()


async def stream_internal_output(
    *,
    events,
    endpoint: str,
    model: str,
    username: str,
    api_key_value: str,
    provider_id: str,
    requested_model: str,
    log_request: RequestLogger,
    previous_response_id: str | None = None,
    conv_key: str = "",
    remember_response_chain_key: RememberResponseChainKey | None = None,
    remember_reasoning_content: RememberReasoningContent | None = None,
    tool_only_turns=None,
):
    total_tokens = 0
    final_model = model
    final_provider_id = provider_id or ""
    visible_output_started = False
    stream_details: dict[str, Any] = {"stream": True, "fallback_status": "unused"}
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
        nonlocal total_tokens, final_model, final_provider_id, visible_output_started, stream_details
        async for event in record_streaming_events(
            events,
            conv_key=conv_key,
            remember_reasoning_content=remember_reasoning_content,
            tool_only_turns=tool_only_turns,
        ):
            if event.kind == "metadata":
                final_model = str(event.metadata.get("model") or final_model)
                final_provider_id = str(event.metadata.get("provider_id") or final_provider_id)
                for key in (
                    "stream",
                    "attempt_index",
                    "fallback_status",
                    "fallback_reason",
                    "error_trigger",
                    "error_stage",
                    "attempted_model",
                    "attempted_provider",
                ):
                    if key in event.metadata:
                        stream_details[key] = event.metadata[key]
                continue
            if event.kind == "usage":
                total_tokens = event.usage.get("total_tokens", total_tokens) or total_tokens
            if event.kind in ("text_delta", "reasoning_delta"):
                if event.text or event.reasoning:
                    visible_output_started = True
            elif event.kind in ("tool_call_start", "tool_call_arguments_delta", "tool_call_done", "message_done"):
                visible_output_started = True
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
        log_request(
            username,
            api_key_value,
            final_model,
            final_provider_id,
            endpoint,
            True,
            total_tokens,
            requested_model,
            details={**stream_details, "status": "ok", "partial_output": False},
        )
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, total_tokens)
        _app_log.debug(
            "[stream_orchestrator] DONE endpoint=%s provider=%s model=%s total_tokens=%d",
            endpoint,
            final_provider_id,
            final_model,
            total_tokens,
        )
    except Exception as exc:
        error_msg = friendly_error_msg(exc)
        exc_details = getattr(exc, "request_details", None)
        if not isinstance(exc_details, dict):
            exc_details = {}
        partial_output = bool(exc_details.get("partial_output", visible_output_started))
        failure_details = {
            **stream_details,
            **exc_details,
            "stream": True,
            "status": "partial" if partial_output else "fail",
            "partial_output": partial_output,
            "error_message": exc_details.get("error_message") or error_msg,
        }
        logged_model = str(
            failure_details.get("attempted_model")
            or final_model
            or model
            or "-"
        )
        logged_provider = str(
            failure_details.get("attempted_provider")
            or final_provider_id
            or provider_id
            or ""
        )
        _error_log.error("[%s_stream] %s", endpoint, str(exc))
        log_request(
            username,
            api_key_value,
            logged_model,
            logged_provider,
            endpoint,
            False,
            total_tokens,
            requested_model,
            details=failure_details,
        )
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
