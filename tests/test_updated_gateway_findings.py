"""附录 A R1-R4 的本地回归测试。"""
import json
from types import SimpleNamespace

import pytest

from app.adapters import openai_streaming
from app.core.output import InternalOutputEvent
from app.protocols.egress import render_completions_sse, render_responses_sse


class _AsyncEvents:
    def __init__(self, items):
        self.items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration


def _chunk(content=None, finish=None):
    delta = SimpleNamespace(content=content, reasoning_content=None, role=None, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


@pytest.mark.asyncio
async def test_think_open_tag_split_across_chunks_stays_reasoning(monkeypatch):
    monkeypatch.setattr(
        openai_streaming,
        "create_chat_completion_stream",
        lambda **_: iter([
            _chunk("<"), _chunk("thin"), _chunk("k"), _chunk(">"),
            _chunk("内部思考"), _chunk("</think>"), _chunk("答案"), _chunk(finish="stop"),
        ]),
    )
    events = []
    async for event in openai_streaming.iter_openai_chat_output_events(
        model="m", messages=[], provider_id="p", temperature=0.7, max_tokens=100,
    ):
        events.append(event)
    assert "".join(e.reasoning for e in events if e.kind == "reasoning_delta") == "内部思考"
    assert "".join(e.text for e in events if e.kind == "text_delta") == "答案"


@pytest.mark.asyncio
async def test_reasoning_and_text_output_indices_match_final_order():
    events = [
        InternalOutputEvent(kind="reasoning_delta", reasoning="plan"),
        InternalOutputEvent(kind="text_delta", text="answer"),
        InternalOutputEvent(kind="message_done", finish_reason="stop"),
    ]
    frames = [line async for line in render_responses_sse(_AsyncEvents(events), model="m")]
    added = []
    completed = None
    done = []
    for line in frames:
        if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
            continue
        payload = json.loads(line[6:])
        if payload["type"] == "response.output_item.added":
            added.append((payload["output_index"], payload["item"]["type"]))
        elif payload["type"] == "response.output_item.done":
            done.append((payload["output_index"], payload["item"]["type"]))
        elif payload["type"] == "response.completed":
            completed = payload["response"]
    assert added == [(0, "reasoning"), (1, "message")]
    assert done == [(0, "reasoning"), (1, "message")]
    assert [item["type"] for item in completed["output"]] == ["reasoning", "message"]


@pytest.mark.asyncio
async def test_completions_tool_call_is_not_reported_as_normal_stop(caplog):
    events = [
        InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=2, arguments_delta="{}"),
        InternalOutputEvent(kind="message_done", finish_reason=None),
    ]
    frames = [line async for line in render_completions_sse(_AsyncEvents(events), model="m")]
    payloads = [json.loads(line[6:]) for line in frames if line.startswith("data: ") and line[6:].strip() != "[DONE]"]
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
