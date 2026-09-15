import json
import time
import asyncio

import httpx
from fastapi import HTTPException

from app.adapters.anthropic import (
    _anthropic_headers,
    _anthropic_message_url,
    _build_anthropic_request_body,
    _http_exception_from_upstream,
    _is_retryable_status,
    provider_request_timeout,
    provider_retry_backoff,
    provider_retry_count,
)
from app.core.output import InternalOutputEvent
from app.core.text import client_status_for_upstream_error, error_detail_for_log, friendly_error_msg
from app.services.logger import get_logger
from app.services.http_pool import shared_client


_app_log = get_logger("app")
_tool_log = get_logger("tool_calls")


def _usage_value(usage: dict, current: int, *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if value is not None:
            return value or 0
    return current


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
    cache_hit_tokens = 0
    cache_miss_tokens = 0
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

    timeout = provider_request_timeout(provider_info, 300)
    retries = provider_retry_count(provider_info)
    backoff = provider_retry_backoff(provider_info)

    try:
        async with shared_client(provider_info.get("api_base") or "", timeout) as client:
            for attempt in range(retries + 1):
                # 只要已向下游 yield 过任何事件，就绝不允许重试：重新请求会把已输出
                # 的文本/工具参数再发一遍，造成客户端内容重复。重试仅限首字节前失败。
                emitted = False
                try:
                    async for event in _iter_anthropic_stream_once(
                        client=client,
                        provider_info=provider_info,
                        req_body=req_body,
                        provider_id=provider_id,
                        model=model,
                        block_states=block_states,
                    ):
                        if event.kind == "usage" and event.usage:
                            input_tokens = _usage_value(event.usage, input_tokens, "input_tokens", "prompt_tokens")
                            output_tokens = _usage_value(event.usage, output_tokens, "output_tokens", "completion_tokens")
                            cache_hit_tokens = _usage_value(event.usage, cache_hit_tokens, "prompt_cache_hit_tokens")
                            cache_miss_tokens = _usage_value(event.usage, cache_miss_tokens, "prompt_cache_miss_tokens")
                        if event.kind == "message_done":
                            finish_reason = event.finish_reason or finish_reason
                            continue
                        emitted = True
                        yield event
                    break
                except HTTPException as exc:
                    status_code = getattr(exc, "status_code", 0)
                    if not emitted and attempt < retries and _is_retryable_status(status_code):
                        block_states.clear()
                        await asyncio.sleep(backoff * (2 ** attempt))
                        continue
                    raise
                except (httpx.TimeoutException, httpx.TransportError):
                    if not emitted and attempt < retries:
                        block_states.clear()
                        await asyncio.sleep(backoff * (2 ** attempt))
                        continue
                    raise
    except HTTPException:
        raise
    except (httpx.TimeoutException, TimeoutError, httpx.TransportError):
        raise
    except Exception as exc:
        _app_log.debug("[anthropic_stream_adapter] ERROR provider=%s model=%s error=%s", provider_id, model, error_detail_for_log(exc))
        raise HTTPException(status_code=client_status_for_upstream_error(exc), detail=friendly_error_msg(exc)) from exc

    _app_log.debug(
        "[anthropic_stream_adapter] DONE provider=%s model=%s finish_reason=%s input_tokens=%d output_tokens=%d blocks=%d",
        provider_id,
        model,
        finish_reason,
        input_tokens,
        output_tokens,
        len(block_states),
    )
    final_usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    # 与 OpenAI 路径保持同名字段；为零时不携带，避免破坏对 usage 的精确断言。
    if cache_hit_tokens:
        final_usage["prompt_cache_hit_tokens"] = cache_hit_tokens
    if cache_miss_tokens:
        final_usage["prompt_cache_miss_tokens"] = cache_miss_tokens
    yield InternalOutputEvent(kind="usage", usage=final_usage)
    yield InternalOutputEvent(kind="message_done", finish_reason=finish_reason)


async def _iter_anthropic_stream_once(
    *,
    client: httpx.AsyncClient,
    provider_info: dict,
    req_body: dict,
    provider_id: str,
    model: str,
    block_states: dict[int, dict],
):
    input_tokens = 0
    output_tokens = 0
    cache_creation = 0
    cache_read = 0
    finish_reason = "stop"
    saw_message_delta = False
    saw_message_stop = False
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
            raise _http_exception_from_upstream(resp.status_code, err_msg)

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

            if event_type == "error":
                err = data.get("error", {}) or {}
                err_msg = err.get("message") or json.dumps(data, ensure_ascii=False)
                # 上游原文只进日志：客户端必须拿统一分类器的安全文案（本项目
                # “客户端只拿安全消息”契约；原文可能含内部模型名/配额/后端 URL）。
                _app_log.warning(
                    "[anthropic_stream_adapter] UPSTREAM SSE ERROR provider=%s model=%s error=%s",
                    provider_id, model, err_msg,
                )
                carrier = RuntimeError(f"anthropic upstream error event: {err_msg}")
                # “已确认源自上游”的语义交给分类器（confirmed_upstream），调用方不
                # 得对分类器结果做本地二次改写（审查报告四轮 #1）。
                raise HTTPException(
                    status_code=client_status_for_upstream_error(carrier, confirmed_upstream=True),
                    detail=friendly_error_msg(carrier),
                )
            if event_type == "message_start":
                usage = data.get("message", {}).get("usage", {}) or data.get("usage", {}) or {}
                input_tokens = _usage_value(usage, input_tokens, "input_tokens", "prompt_tokens")
                cache_creation = _usage_value(usage, cache_creation, "cache_creation_input_tokens")
                cache_read = _usage_value(usage, cache_read, "cache_read_input_tokens")
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
                    "signature": block.get("signature", "") or "",
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
                state = block_states.setdefault(block_index, {"type": "", "id": "", "name": "", "arguments": ""})
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
                elif delta_type == "signature_delta":
                    signature = delta.get("signature", "") or ""
                    if signature:
                        state["signature"] = signature
                        yield InternalOutputEvent(
                            kind="reasoning_delta",
                            reasoning_signature=signature,
                            raw=data,
                        )
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
                saw_message_delta = True
                usage = data.get("usage", {}) or {}
                input_tokens = _usage_value(usage, input_tokens, "input_tokens", "prompt_tokens")
                output_tokens = _usage_value(usage, output_tokens, "output_tokens", "completion_tokens")
                cache_creation = _usage_value(usage, cache_creation, "cache_creation_input_tokens")
                cache_read = _usage_value(usage, cache_read, "cache_read_input_tokens")
                _app_log.debug(
                    "[anthropic_stream_adapter] message_delta stop_reason=%s finish_reason=%s input_tokens=%d output_tokens=%d",
                    stop_reason,
                    finish_reason,
                    input_tokens,
                    output_tokens,
                )
                usage_payload = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
                if cache_read:
                    usage_payload["prompt_cache_hit_tokens"] = cache_read
                if cache_creation:
                    usage_payload["prompt_cache_miss_tokens"] = cache_creation
                yield InternalOutputEvent(kind="usage", usage=usage_payload, raw=data)
            elif event_type == "message_stop":
                saw_message_stop = True
                break
    if not saw_message_delta and not saw_message_stop:
        # 上游既没发 message_delta 也没发 message_stop 就关闭了连接：内容被截断，
        # 不能伪装成 finish_reason=stop 的正常结束。
        raise HTTPException(status_code=502, detail="Upstream: anthropic stream closed before message completion")
    yield InternalOutputEvent(kind="message_done", finish_reason=finish_reason)
