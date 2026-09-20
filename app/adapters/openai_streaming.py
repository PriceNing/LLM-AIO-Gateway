import time

from app.adapters.streaming import iter_stream_async
from app.core.output import InternalOutputEvent
from app.core.think import extract_and_strip_think
from app.core.tool_args import fix_tool_args, sanitize_args
from app.services.lite_llm import create_chat_completion_stream
from app.services.logger import get_logger


_app_log = get_logger("app")
_tool_log = get_logger("tool_calls")


def _tool_call_to_dict(tool_call) -> dict:
    if hasattr(tool_call, "model_dump"):
        data = tool_call.model_dump(exclude_none=True)
    elif isinstance(tool_call, dict):
        data = dict(tool_call)
    else:
        fn = getattr(tool_call, "function", None)
        data = {
            "index": getattr(tool_call, "index", 0),
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", "function"),
            "function": {
                "name": getattr(fn, "name", None) if fn else None,
                "arguments": getattr(fn, "arguments", "") if fn else "",
            },
        }
    if data.get("id") is not None and not isinstance(data["id"], str):
        data["id"] = str(data["id"])
    return data


def _infer_tool_index(tc_dict: dict, tool_states: dict[int, dict]) -> int:
    """部分 OpenAI 兼容上游的流式 tool_calls 不带 index。

    可达性：``_tool_call_to_dict`` 对 Pydantic 对象总是产出 ``index``（默认 0），
    因此本函数仅在“上游直接发送缺 index 的裸 dict”时被调用；若全部默认 0，
    多个并行工具调用会被合并串参。

    推断规则：已知 id 匹配现有状态则沿用；新 id 分配下一个紧凑下标；
    无 id 则延续最后一个（增量上下文）。
    """
    tc_id = str(tc_dict.get("id") or "")
    if tc_id:
        for idx, state in tool_states.items():
            if state.get("id") == tc_id:
                return idx
        # 用 len() 而非 max()+1：下标只作为内部键与 SSE index，紧凑分配可
        # 避免上游乱序 id 造成的空洞让客户端看到不连续的 tool_calls index。
        # 与显式 index 混用时 len() 可能撞上已占用的键（如 {0,2} 时 len=2），
        # 逐个跳过已占用槽位，保证不会把新调用合并进无关状态。
        candidate = len(tool_states)
        while candidate in tool_states:
            candidate += 1
        return candidate
    return max(tool_states) if tool_states else 0


async def iter_openai_chat_output_events(
    *,
    model,
    messages,
    provider_id,
    temperature,
    max_tokens,
    extra=None,
    strip_thinking=True,
):
    """Adapt liteLLM/OpenAI-compatible chat chunks into internal output events."""
    extra = dict(extra or {})
    stream_func = lambda: create_chat_completion_stream(
        model=model,
        messages=messages,
        provider_id=provider_id,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra,
    )

    _app_log.debug(
        "[openai_stream_adapter] START provider=%s model=%s messages=%d tools=%d max_tokens=%s strip_thinking=%s",
        provider_id or "",
        model,
        len(messages),
        len(extra.get("tools") or []),
        max_tokens,
        strip_thinking,
    )

    yield InternalOutputEvent(kind="message_start", role="assistant")

    think_state = {"stripped": not strip_thinking, "buf": ""}
    tool_states: dict[int, dict] = {}
    finish_reason = None
    saw_output = False
    saw_finish = False
    saw_answer_output = False
    tolerated_tail_error = False

    try:
        async for chunk in iter_stream_async(stream_func):
            async for event in _events_from_openai_chunk(
                chunk,
                model=model,
                tool_states=tool_states,
                think_state=think_state,
            ):
                if event.kind in ("text_delta", "reasoning_delta", "tool_call_start", "tool_call_arguments_delta", "usage"):
                    saw_output = True
                if event.kind in ("text_delta", "tool_call_start", "tool_call_arguments_delta"):
                    saw_answer_output = True
                if event.kind == "message_delta" and event.finish_reason:
                    finish_reason = event.finish_reason
                    saw_finish = True
                    continue
                if event.kind == "message_done":
                    finish_reason = event.finish_reason or finish_reason
                    saw_finish = saw_finish or bool(event.finish_reason)
                    continue
                yield event
    except Exception as exc:
        if saw_output and _is_litellm_tail_chunk_builder_error(exc):
            tolerated_tail_error = True
            _app_log.warning(
                "[openai_stream_adapter] tolerated liteLLM tail chunk-builder error provider=%s model=%s error=%s",
                provider_id or "",
                model,
                str(exc)[:240],
            )
        else:
            raise

    if think_state["buf"]:
        yield InternalOutputEvent(kind="text_delta", text=think_state["buf"])
        saw_answer_output = True

    for idx, state in sorted(tool_states.items()):
        _tool_log.debug(
            "[openai_stream_adapter] tool_done index=%d id=%s name=%s args_chars=%d",
            idx,
            state["id"],
            state["name"],
            len(state["arguments"]),
        )
        yield InternalOutputEvent(
            kind="tool_call_done",
            tool_index=idx,
            tool_call_id=state["id"],
            call_id=state["call_id"],
            name=state["name"],
            arguments=state["arguments"],
        )
    if not saw_finish and not saw_answer_output and not tool_states:
        # 静默截断守卫：上游从未发送 finish_reason，也没有产出任何可见回答
        # （正文/工具调用）。此前这里会合成 message_done(stop) 并记 ok，
        # 下游 harness（pi/Codex）把被服务端 watchdog 掉断的流当成正常
        # 空回合结束 → 任务未完成但 agent 停止（SuperGrok 2026-09-16 事故）。
        # 改为报错：客户端能看到 error 并可重试，而不是静默停住。
        _app_log.warning(
            "[openai_stream_adapter] silent truncation provider=%s model=%s stream closed without finish_reason or answer output (tolerated_tail_error=%s)",
            provider_id or "",
            model,
            tolerated_tail_error,
        )
        raise _silent_truncation_error(
            model=model,
            provider_id=provider_id or "",
            # 走到这里已经没有回答；打上 empty_stream_response 让 fallback 层
            # 在尚未向客户端可见输出时保留同目标退化重试。reasoning-only
            # 已 emitted，不会换模型，此标记无效。
            empty_stream=True,
        )

    _app_log.debug(
        "[openai_stream_adapter] DONE provider=%s model=%s finish_reason=%s tool_calls=%d tolerated_tail_error=%s",
        provider_id or "",
        model,
        finish_reason or "stop",
        len(tool_states),
        tolerated_tail_error,
    )
    yield InternalOutputEvent(kind="message_done", finish_reason=finish_reason or "stop")


async def _events_from_openai_chunk(chunk, *, model, tool_states: dict[int, dict], think_state: dict):
    finish_reason = None
    choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
    if choice:
        finish_reason = getattr(choice, "finish_reason", None) or None
        delta = getattr(choice, "delta", None)
        if delta:
            role = getattr(delta, "role", None)
            if role:
                yield InternalOutputEvent(kind="message_start", role=role, raw=chunk)

            content = getattr(delta, "content", None)
            if content:
                _app_log.debug("[openai_stream_adapter] text_chunk chars=%d", len(content))
                if not think_state["stripped"]:
                    think_state["buf"] += content
                    if "</think>" in think_state["buf"]:
                        visible, thinking = extract_and_strip_think(think_state["buf"])
                        if thinking:
                            _app_log.debug("[openai_stream_adapter] think_extracted chars=%d", len(thinking))
                            yield InternalOutputEvent(kind="reasoning_delta", reasoning=thinking, raw=chunk)
                        think_state["stripped"] = True
                        think_state["buf"] = ""
                        if visible:
                            yield InternalOutputEvent(kind="text_delta", text=visible, raw=chunk)
                    elif "<think>" in think_state["buf"] or think_state["buf"].lstrip().startswith("<think"):
                        if len(think_state["buf"]) >= 200:
                            visible = think_state["buf"].replace("<think>", "", 1)
                            think_state["stripped"] = True
                            think_state["buf"] = ""
                            if visible:
                                yield InternalOutputEvent(kind="text_delta", text=visible, raw=chunk)
                    else:
                        think_state["stripped"] = True
                        visible = think_state["buf"]
                        think_state["buf"] = ""
                        yield InternalOutputEvent(kind="text_delta", text=visible, raw=chunk)
                else:
                    yield InternalOutputEvent(kind="text_delta", text=content, raw=chunk)

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                _app_log.debug("[openai_stream_adapter] reasoning_delta chars=%d", len(reasoning))
                yield InternalOutputEvent(kind="reasoning_delta", reasoning=reasoning, raw=chunk)

            tool_calls = getattr(delta, "tool_calls", None)
            if tool_calls:
                _tool_log.debug("[openai_stream_adapter] raw tool_calls count=%d model=%s", len(tool_calls), model)
                for tc in tool_calls:
                    tc_dict = _tool_call_to_dict(tc)
                    idx = int(tc_dict.get("index", 0)) if "index" in tc_dict else _infer_tool_index(tc_dict, tool_states)
                    if idx < 0:
                        _tool_log.debug("[openai_stream_adapter] FILTERED spurious: id=%s idx=%s", tc_dict.get("id"), idx)
                        continue
                    fix_tool_args(tc_dict)
                    fn = tc_dict.get("function") or {}
                    tc_id = tc_dict.get("id") or ""
                    name = fn.get("name") or ""
                    args_delta = fn.get("arguments") or ""
                    if idx not in tool_states:
                        call_id = tc_id if str(tc_id).startswith("call_") else (f"call_{tc_id}" if tc_id else f"call_{int(time.time())}_{idx}")
                        tool_states[idx] = {
                            "id": tc_id or f"fc_{int(time.time())}_{idx}",
                            "call_id": call_id,
                            "name": name,
                            "arguments": "",
                        }
                        _tool_log.debug(
                            "[openai_stream_adapter] tool_start index=%d id=%s name=%s",
                            idx,
                            tool_states[idx]["id"],
                            name,
                        )
                        yield InternalOutputEvent(
                            kind="tool_call_start",
                            tool_index=idx,
                            tool_call_id=tool_states[idx]["id"],
                            call_id=tool_states[idx]["call_id"],
                            name=name,
                            raw=tc_dict,
                        )
                    state = tool_states[idx]
                    if tc_id:
                        state["id"] = tc_id
                    if name:
                        state["name"] = name
                    if args_delta:
                        if "undefined" in args_delta:
                            args_delta = sanitize_args(args_delta)
                        state["arguments"] += args_delta
                        if "undefined" in state["arguments"]:
                            state["arguments"] = sanitize_args(state["arguments"])
                        _tool_log.debug(
                            "[openai_stream_adapter] tool_args_delta index=%d id=%s chars=%d total_chars=%d",
                            idx,
                            state["id"],
                            len(args_delta),
                            len(state["arguments"]),
                        )
                        yield InternalOutputEvent(
                            kind="tool_call_arguments_delta",
                            tool_index=idx,
                            tool_call_id=state["id"],
                            call_id=state["call_id"],
                            name=state["name"],
                            arguments_delta=args_delta,
                            arguments=state["arguments"],
                            raw=tc_dict,
                        )

    usage_obj = getattr(chunk, "usage", None)
    if usage_obj:
        _app_log.debug(
            "[openai_stream_adapter] usage prompt=%d completion=%d total=%d cache_hit=%d cache_miss=%d",
            getattr(usage_obj, "prompt_tokens", 0) or 0,
            getattr(usage_obj, "completion_tokens", 0) or 0,
            getattr(usage_obj, "total_tokens", 0) or 0,
            getattr(usage_obj, "prompt_cache_hit_tokens", 0) or 0,
            getattr(usage_obj, "prompt_cache_miss_tokens", 0) or 0,
        )
        yield InternalOutputEvent(
            kind="usage",
            usage={
                "input_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
                "prompt_cache_hit_tokens": getattr(usage_obj, "prompt_cache_hit_tokens", 0) or 0,
                "prompt_cache_miss_tokens": getattr(usage_obj, "prompt_cache_miss_tokens", 0) or 0,
            },
            raw=chunk,
        )

    if finish_reason:
        yield InternalOutputEvent(kind="message_delta", finish_reason=finish_reason)


def _is_litellm_tail_chunk_builder_error(exc: Exception) -> bool:
    msg = str(exc)
    return "Error building chunks for logging/streaming usage calculation" in msg


def _silent_truncation_error(*, model: str, provider_id: str, empty_stream: bool) -> RuntimeError:
    """Confirmed upstream failure: stream closed with no finish and no answer."""
    exc = RuntimeError("upstream stream ended without a finish reason before sending any answer output")
    exc.confirmed_upstream = True
    exc.attempted_model = model
    exc.attempted_provider = provider_id or ""
    # 空流才允许 fallback 层做同目标退化重试；reasoning-only 不算 empty。
    exc.empty_stream_response = bool(empty_stream)
    return exc
