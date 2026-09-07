import json
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import asyncio

from app.core.outcome import (
    apply_outcome_to_details,
    is_client_disconnect_error,
    stats_counters_for_status,
)
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
RequestDetailRecorder = Callable[..., None]
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
    record_request_log: RequestDetailRecorder | None = None,
    previous_response_id: str | None = None,
    conv_key: str = "",
    remember_response_chain_key: RememberResponseChainKey | None = None,
    remember_reasoning_content: RememberReasoningContent | None = None,
    tool_only_turns=None,
    base_details: dict[str, Any] | None = None,
    render_extra: dict[str, Any] | None = None,
):
    total_tokens = 0
    final_model = model
    final_provider_id = provider_id or ""
    visible_output_started = False
    stream_details: dict[str, Any] = {"stream": True, "fallback_status": "unused"}
    if base_details:
        stream_details.update(base_details)
        stream_details.setdefault("stream", True)
        stream_details.setdefault("fallback_status", "unused")
    streamed_text_parts: list[str] = []
    streamed_reasoning_parts: list[str] = []
    streamed_tool_calls: list[dict[str, Any]] = []
    current_tool: dict[str, Any] | None = None
    streamed_usage: dict[str, Any] = {}
    stream_started_at = time.monotonic()
    first_output_at: float | None = None
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
        nonlocal streamed_text_parts, streamed_reasoning_parts, streamed_tool_calls, current_tool, streamed_usage
        nonlocal first_output_at
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
                    "fallback_attempts",
                    "routing_matched",
                    "routing_rule_id",
                    "routing_rule_name",
                    "routing_reason",
                    "routed_model",
                    "routed_provider",
                    "upstream_endpoint",
                    "responses_mode",
                    "native_attempted",
                    "native_failure_endpoint",
                    "native_failure_status",
                    "native_failure_reason",
                    "native_failure_message",
                    "native_attempts",
                ):
                    if key in event.metadata:
                        stream_details[key] = event.metadata[key]
                continue
            if event.kind == "usage":
                total_tokens = event.usage.get("total_tokens", total_tokens) or total_tokens
                streamed_usage.update(event.usage or {})
            if event.kind in ("text_delta", "reasoning_delta"):
                if event.text or event.reasoning:
                    visible_output_started = True
                    if first_output_at is None:
                        first_output_at = time.monotonic()
                if event.kind == "text_delta" and event.text:
                    streamed_text_parts.append(event.text)
                elif event.kind == "reasoning_delta" and event.reasoning:
                    streamed_reasoning_parts.append(event.reasoning)
            elif event.kind == "tool_call_start":
                visible_output_started = True
                if first_output_at is None:
                    first_output_at = time.monotonic()
                if event.tool_call_id or event.name:
                    current_tool = {
                        "id": event.tool_call_id or "",
                        "name": event.name or "",
                        "arguments": "",
                    }
            elif event.kind == "tool_call_arguments_delta":
                if current_tool is None:
                    current_tool = {
                        "id": event.tool_call_id or "",
                        "name": event.name or "",
                        "arguments": "",
                    }
                if event.name:
                    current_tool["name"] = event.name
                if event.tool_call_id:
                    current_tool["id"] = event.tool_call_id
                current_tool["arguments"] = event.arguments or (
                    (current_tool.get("arguments") or "") + (event.arguments_delta or "")
                )
            elif event.kind == "tool_call_done" and current_tool is not None:
                if event.name:
                    current_tool["name"] = event.name
                if event.tool_call_id:
                    current_tool["id"] = event.tool_call_id
                streamed_tool_calls.append(current_tool)
                current_tool = None
            elif event.kind == "message_done":
                visible_output_started = True
                if first_output_at is None:
                    first_output_at = time.monotonic()
                if current_tool is not None:
                    streamed_tool_calls.append(current_tool)
                    current_tool = None
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
                extra=render_extra,
            ):
                yield line
        success_details = apply_outcome_to_details(
            {**stream_details, "partial_output": False},
            success=True,
            partial_output=False,
        )
        _attach_stream_performance(success_details, streamed_usage, first_output_at, stream_started_at)
        counters = stats_counters_for_status(success_details.get("status", "ok"))
        log_request(
            username,
            api_key_value,
            final_model,
            final_provider_id,
            endpoint,
            counters.hard_success,
            total_tokens,
            requested_model,
            details=success_details,
        )
        _invoke_record_request_log(
            record_request_log,
            success=counters.hard_success,
            status=success_details.get("status", "ok"),
            tokens=total_tokens,
            details=success_details,
            streamed_text_parts=streamed_text_parts,
            streamed_reasoning_parts=streamed_reasoning_parts,
            streamed_tool_calls=streamed_tool_calls,
            streamed_usage=streamed_usage,
            final_model=final_model,
            final_provider_id=final_provider_id,
            visible_output_started=visible_output_started,
            generation_started_at=first_output_at or stream_started_at,
            request_started_at=stream_started_at,
        )
        increment_global_stats(
            counters.hard_success,
            degraded=counters.degraded,
            rejected=counters.rejected,
            cancelled=counters.cancelled,
        )
        if username != "legacy":
            increment_user_usage(username, api_key_value, counters.hard_success, total_tokens)
        _app_log.debug(
            "[stream_orchestrator] DONE endpoint=%s provider=%s model=%s total_tokens=%d status=%s",
            endpoint,
            final_provider_id,
            final_model,
            total_tokens,
            success_details.get("status"),
        )
    except BaseException as exc:
        if is_client_disconnect_error(exc):
            cancel_details = apply_outcome_to_details(
                {
                    **stream_details,
                    "stream": True,
                    "client_disconnected": True,
                    "error_message": "client disconnected",
                    "status": "cancelled",
                },
                success=False,
                partial_output=visible_output_started,
            )
            cancel_details["status"] = "cancelled"
            _attach_stream_performance(cancel_details, streamed_usage, first_output_at, stream_started_at)
            logged_model = str(final_model or model or "-")
            logged_provider = str(final_provider_id or provider_id or "")
            _app_log.warning(
                "[%s_stream.cancelled] provider=%s model=%s partial=%s tokens=%d",
                endpoint,
                logged_provider or "-",
                logged_model,
                visible_output_started,
                total_tokens,
            )
            log_request(
                username,
                api_key_value,
                logged_model,
                logged_provider,
                endpoint,
                False,
                total_tokens,
                requested_model,
                details=cancel_details,
            )
            _invoke_record_request_log(
                record_request_log,
                success=False,
                status="cancelled",
                tokens=total_tokens,
                details=cancel_details,
                streamed_text_parts=streamed_text_parts,
                streamed_reasoning_parts=streamed_reasoning_parts,
                streamed_tool_calls=streamed_tool_calls,
                streamed_usage=streamed_usage,
                final_model=logged_model,
                final_provider_id=logged_provider,
                visible_output_started=visible_output_started,
                generation_started_at=first_output_at or stream_started_at,
                request_started_at=stream_started_at,
                error_message="client disconnected",
            )
            increment_global_stats(False, cancelled=True)
            if username != "legacy":
                increment_user_usage(username, api_key_value, False, 0)
            if tool_only_turns is not None and conv_key:
                tool_only_turns.reset(conv_key)
            if isinstance(exc, Exception) and not isinstance(exc, asyncio.CancelledError):
                return
            raise

        if not isinstance(exc, Exception):
            raise

        error_msg = friendly_error_msg(exc)
        exc_details = getattr(exc, "request_details", None)
        if not isinstance(exc_details, dict):
            exc_details = {}
        partial_output = bool(exc_details.get("partial_output", visible_output_started))
        failure_details = apply_outcome_to_details(
            {
                **stream_details,
                **exc_details,
                "stream": True,
                "error_message": exc_details.get("error_message") or error_msg,
            },
            success=False,
            partial_output=partial_output,
        )
        _attach_stream_performance(failure_details, streamed_usage, first_output_at, stream_started_at)
        counters = stats_counters_for_status(failure_details.get("status", "fail"))
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
        _invoke_record_request_log(
            record_request_log,
            success=False,
            status=failure_details.get("status", "fail"),
            tokens=total_tokens,
            details=failure_details,
            streamed_text_parts=streamed_text_parts,
            streamed_reasoning_parts=streamed_reasoning_parts,
            streamed_tool_calls=streamed_tool_calls,
            streamed_usage=streamed_usage,
            final_model=logged_model,
            final_provider_id=logged_provider,
            visible_output_started=visible_output_started,
            generation_started_at=first_output_at or stream_started_at,
            request_started_at=stream_started_at,
            error_message=failure_details.get("error_message"),
        )
        increment_global_stats(
            False,
            degraded=counters.degraded,
            rejected=counters.rejected,
            cancelled=counters.cancelled,
            stateful_fallback_blocked=bool(failure_details.get("stateful_fallback_blocked")),
        )
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


def _invoke_record_request_log(
    recorder: RequestDetailRecorder | None,
    *,
    success: bool,
    status: str,
    tokens: int,
    details: dict[str, Any],
    streamed_text_parts: list[str],
    streamed_reasoning_parts: list[str],
    streamed_tool_calls: list[dict[str, Any]],
    streamed_usage: dict[str, Any],
    final_model: str,
    final_provider_id: str,
    visible_output_started: bool,
    error_message: str | None = None,
    generation_started_at: float | None = None,
    request_started_at: float | None = None,
) -> None:
    if recorder is None:
        return
    try:
        recorder(
            success=success,
            status=status,
            tokens=tokens,
            details=details,
            streamed_text="".join(streamed_text_parts),
            streamed_reasoning="".join(streamed_reasoning_parts),
            streamed_tool_calls=streamed_tool_calls,
            usage=streamed_usage,
            final_model=final_model,
            final_provider_id=final_provider_id,
            partial_output=visible_output_started,
            error_message=error_message,
            generation_started_at=generation_started_at,
            request_started_at=request_started_at,
        )
    except Exception as exc:
        _app_log.warning("record_request_log callback failed: %s", exc)


def _attach_stream_performance(
    details: dict[str, Any],
    usage: dict[str, Any],
    first_output_at: float | None,
    stream_started_at: float,
) -> None:
    now = time.monotonic()
    duration_s = max(0.0, now - stream_started_at)
    generation_s = max(0.0, now - (first_output_at or stream_started_at))
    details["duration_ms"] = round(duration_s * 1000)
    details["generation_ms"] = round(generation_s * 1000)
    try:
        completion_tokens = max(0, int(
            (usage or {}).get("completion_tokens")
            or (usage or {}).get("output_tokens")
            or 0
        ))
    except (TypeError, ValueError):
        completion_tokens = 0
    details["completion_tokens"] = completion_tokens
    details["tps"] = round(completion_tokens / generation_s, 2) if generation_s > 0 else None
