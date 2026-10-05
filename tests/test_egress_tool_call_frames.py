"""Chat Completions 工具调用增量帧协议回归测试。"""
import json

import pytest

from app.core.output import InternalOutputEvent
from app.protocols.egress import render_chat_completions_sse


async def _events(items):
    for item in items:
        yield item


async def _frames(items):
    return [line async for line in render_chat_completions_sse(_events(items), model="m")]


def _tool_chunks(frames):
    result = []
    for frame in frames:
        if not frame.startswith("data: ") or "[DONE]" in frame:
            continue
        payload = json.loads(frame[len("data: "):])
        for choice in payload.get("choices") or []:
            result.extend((choice.get("delta") or {}).get("tool_calls") or [])
    return result


@pytest.mark.asyncio
async def test_tool_call_name_and_id_appear_once():
    frames = await _frames([
        InternalOutputEvent(kind="tool_call_start", tool_index=0, tool_call_id="call_1", name="get_weather"),
        InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=0, tool_call_id="call_1", name="get_weather", arguments_delta="{"),
        InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=0, tool_call_id="call_1", name="get_weather", arguments_delta='"city":"上海"'),
        InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=0, tool_call_id="call_1", name="get_weather", arguments_delta="}"),
        InternalOutputEvent(kind="message_done", finish_reason="tool_calls"),
    ])
    calls = _tool_chunks(frames)
    assert sum(bool(call.get("function", {}).get("name")) for call in calls) == 1
    assert sum(bool(call.get("id")) for call in calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert "".join(call.get("function", {}).get("arguments", "") for call in calls) == '{"city":"上海"}'


@pytest.mark.asyncio
async def test_parallel_tools_each_has_one_header():
    frames = await _frames([
        InternalOutputEvent(kind="tool_call_start", tool_index=0, tool_call_id="call_0", name="a"),
        InternalOutputEvent(kind="tool_call_start", tool_index=1, tool_call_id="call_1", name="b"),
        InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=0, tool_call_id="call_0", name="a", arguments_delta="{}"),
        InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=1, tool_call_id="call_1", name="b", arguments_delta="{}"),
        InternalOutputEvent(kind="message_done", finish_reason="tool_calls"),
    ])
    calls = _tool_chunks(frames)
    for index in (0, 1):
        indexed = [call for call in calls if call["index"] == index]
        assert sum(bool(call.get("function", {}).get("name")) for call in indexed) == 1
        assert sum(bool(call.get("id")) for call in indexed) == 1


@pytest.mark.asyncio
async def test_arguments_only_first_frame_gets_header():
    frames = await _frames([
        InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=0, tool_call_id="call_1", name="run", arguments_delta="{}"),
        InternalOutputEvent(kind="message_done", finish_reason="tool_calls"),
    ])
    calls = _tool_chunks(frames)
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"]["name"] == "run"
