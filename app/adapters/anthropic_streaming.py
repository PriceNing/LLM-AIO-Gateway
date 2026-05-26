import json
import time

import httpx
from fastapi import HTTPException

from app.adapters.anthropic import _anthropic_headers, _anthropic_message_url, _build_anthropic_request_body
from app.core.output import InternalOutputEvent
from app.core.text import friendly_error_msg
from app.services.logger import get_logger


_app_log = get_logger("app")
_tool_log = get_logger("tool_calls")


async def iter_anthropic_output_events(
    *,
    provider_info: dict,
    messages: list,
    body: dict,
    max_tokens: int,
    temperature,
    model: str,
):
    """Adapt native Anthropic Messages SSE into internal output events."""
    req_body = _build_anthropic_request_body(
        provider_info,
        messages,
        body,
        max_tokens,
        temperature,
        model,
        stream=True,
        tool_format="native_strip_type",
    )
    block_states: dict[int, dict] = {}
    input_tokens = 0
    output_tokens = 0
    finish_reason = "stop"
    provider_id = provider_info.get("id", "")

    _app_log.debug(
        "[anthropic_stream_adapter] START provider=%s model=%s messages=%d tools=%d max_tokens=%s stream=true",
        provider_id,
        model,
        len(messages),
        len(body.get("tools") or []),
        max_tokens,
    )

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                _anthropic_message_url(provider_info.get("api_base") or ""),
                headers=_anthropic_headers(provider_info),
                json=req_body,
            ) as resp:
                if resp.status_code != 200:
                    try:
                        err_body = await resp.aread()
                        err_data = json.loads(err_body)
                        err_msg = err_data.get("error", {}).get("message", str(err_body)[:300])
                    except Exception:
                        err_msg = f"HTTP {resp.status_code}"
                    raise HTTPException(status_code=502, detail=f"Upstream {resp.status_code}: {err_msg}")

                _app_log.debug("[anthropic_stream_adapter] CONNECTED provider=%s model=%s status=%d", provider_id, model, resp.status_code)

                current_event = None
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event: "):
                        current_event = line[7:].strip()
                        continue
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                    except Exception:
                        continue
                    event_type = current_event or data.get("type")
                    current_event = None

                    if event_type == "message_start":
                        usage = data.get("message", {}).get("usage", {}) or {}
                        input_tokens = usage.get("input_tokens", input_tokens) or 0
                        _app_log.debug("[anthropic_stream_adapter] message_start input_tokens=%d", input_tokens)
                        yield InternalOutputEvent(kind="message_start", role="assistant", raw=data)
                    elif event_type == "content_block_start":
                        block_index = int(data.get("index", 0))
                        block = data.get("content_block", {}) or {}
                        block_type = block.get("type", "")
                        block_states[block_index] = {
                            "type": block_type,
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "arguments": "",
                        }
                        _app_log.debug("[anthropic_stream_adapter] block_start index=%d type=%s", block_index, block_type)
                        if block_type == "tool_use":
                            tool_id = block.get("id", "") or f"toolu_{block_index}"
                            call_id = tool_id if str(tool_id).startswith("call_") else f"call_{tool_id}"
                            _tool_log.debug(
                                "[anthropic_stream_adapter] tool_start index=%d id=%s name=%s",
                                block_index,
                                tool_id,
                                block.get("name", ""),
                            )
                            yield InternalOutputEvent(
                                kind="tool_call_start",
                                tool_index=block_index,
                                tool_call_id=tool_id,
                                call_id=call_id,
                                name=block.get("name", ""),
                                raw=data,
                            )
                    elif event_type == "content_block_delta":
                        block_index = int(data.get("index", 0))
                        delta = data.get("delta", {}) or {}
                        delta_type = delta.get("type", "")
                        state = block_states.get(block_index, {})
                        if delta_type == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                _app_log.debug("[anthropic_stream_adapter] text_delta index=%d chars=%d", block_index, len(text))
                                yield InternalOutputEvent(kind="text_delta", text=text, raw=data)
                        elif delta_type == "input_json_delta":
                            partial = delta.get("partial_json", "")
                            if partial:
                                state["arguments"] = state.get("arguments", "") + partial
                                tool_id = state.get("id") or f"toolu_{block_index}"
                                call_id = tool_id if str(tool_id).startswith("call_") else f"call_{tool_id}"
                                _tool_log.debug(
                                    "[anthropic_stream_adapter] tool_args_delta index=%d id=%s chars=%d total_chars=%d",
                                    block_index,
                                    tool_id,
                                    len(partial),
                                    len(state["arguments"]),
                                )
                                yield InternalOutputEvent(
                                    kind="tool_call_arguments_delta",
                                    tool_index=block_index,
                                    tool_call_id=tool_id,
                                    call_id=call_id,
                                    name=state.get("name", ""),
                                    arguments_delta=partial,
                                    arguments=state["arguments"],
                                    raw=data,
                                )
                        elif delta_type in ("thinking_delta", "redacted_thinking_delta"):
                            thinking = delta.get("thinking", "") or delta.get("text", "")
                            if thinking:
                                _app_log.debug("[anthropic_stream_adapter] reasoning_delta index=%d chars=%d", block_index, len(thinking))
                                yield InternalOutputEvent(kind="reasoning_delta", reasoning=thinking, raw=data)
                    elif event_type == "content_block_stop":
                        block_index = int(data.get("index", 0))
                        state = block_states.get(block_index, {})
                        if state.get("type") == "tool_use":
                            tool_id = state.get("id") or f"toolu_{block_index}"
                            call_id = tool_id if str(tool_id).startswith("call_") else f"call_{tool_id}"
                            _tool_log.debug(
                                "[anthropic_stream_adapter] tool_done index=%d id=%s name=%s args_chars=%d",
                                block_index,
                                tool_id,
                                state.get("name", ""),
                                len(state.get("arguments", "")),
                            )
                            yield InternalOutputEvent(
                                kind="tool_call_done",
                                tool_index=block_index,
                                tool_call_id=tool_id,
                                call_id=call_id,
                                name=state.get("name", ""),
                                arguments=state.get("arguments", ""),
                                raw=data,
                            )
                    elif event_type == "message_delta":
                        delta = data.get("delta", {}) or {}
                        stop_reason = delta.get("stop_reason")
                        if stop_reason == "tool_use":
                            finish_reason = "tool_calls"
                        elif stop_reason == "max_tokens":
                            finish_reason = "length"
                        elif stop_reason:
                            finish_reason = "stop"
                        usage = data.get("usage", {}) or {}
                        output_tokens = usage.get("output_tokens", output_tokens) or output_tokens
                        _app_log.debug(
                            "[anthropic_stream_adapter] message_delta stop_reason=%s finish_reason=%s output_tokens=%d",
                            stop_reason,
                            finish_reason,
                            output_tokens,
                        )
                        yield InternalOutputEvent(
                            kind="usage",
                            usage={
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens,
                            },
                            raw=data,
                        )
                    elif event_type == "message_stop":
                        break
    except Exception as exc:
        _app_log.debug("[anthropic_stream_adapter] ERROR provider=%s model=%s error=%s", provider_id, model, friendly_error_msg(exc))
        raise HTTPException(status_code=502, detail=friendly_error_msg(exc)) from exc

    _app_log.debug(
        "[anthropic_stream_adapter] DONE provider=%s model=%s finish_reason=%s input_tokens=%d output_tokens=%d blocks=%d",
        provider_id,
        model,
        finish_reason,
        input_tokens,
        output_tokens,
        len(block_states),
    )
    yield InternalOutputEvent(
        kind="usage",
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
    )
    yield InternalOutputEvent(kind="message_done", finish_reason=finish_reason)
