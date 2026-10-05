"""已确认协议/流式问题的回归测试。"""
import json
from types import SimpleNamespace

import pytest

from app.adapters import openai_streaming
from app.adapters.anthropic import _build_anthropic_request_body, anthropic_body_from_internal
from app.core.output import InternalOutputEvent, InternalOutputMessage
from app.core.types import InternalRequest
from app.protocols.egress import render_responses_sse, render_response
from app.protocols.ingress import anthropic_messages_to_internal, responses_to_internal
from app.protocols.ir import ir_to_anthropic_messages, openai_messages_to_ir


def _chunk(content=None, finish=None):
    delta = SimpleNamespace(content=content, reasoning_content=None, role=None, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


@pytest.mark.asyncio
async def test_streaming_long_think_chunks_remain_reasoning(monkeypatch):
    words = ("The user wants a long answer and I should reason carefully about the "
             "constraints before producing the final response. " * 5).split()
    chunks = [_chunk("<think>")] + [_chunk(word + " ") for word in words]
    chunks += [_chunk("</think>"), _chunk("final"), _chunk(finish="stop")]

    monkeypatch.setattr(openai_streaming, "create_chat_completion_stream", lambda **_: iter(chunks))
    events = []
    async for event in openai_streaming.iter_openai_chat_output_events(
        model="m", messages=[], provider_id="p", temperature=0.7, max_tokens=1000,
    ):
        events.append(event)

    reasoning = "".join(e.reasoning for e in events if e.kind == "reasoning_delta")
    text = "".join(e.text for e in events if e.kind == "text_delta")
    assert len(reasoning) > 200
    assert text == "final"
    assert "</think>" not in text


class _AsyncEvents:
    def __init__(self, events):
        self.events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration:
            raise StopAsyncIteration


async def _response_frames(events):
    return [line async for line in render_responses_sse(_AsyncEvents(events), model="m")]


def _payload(line):
    return json.loads(line.removeprefix("data: ").strip())


@pytest.mark.asyncio
async def test_responses_stream_preserves_reasoning_only_output():
    frames = await _response_frames([
        InternalOutputEvent(kind="message_start"),
        InternalOutputEvent(kind="reasoning_delta", reasoning="plan"),
        InternalOutputEvent(kind="message_done", finish_reason="stop"),
    ])
    completed = next(_payload(line)["response"] for line in frames if "response.completed" in line)
    assert completed["output"]
    assert completed["output"][0]["type"] == "reasoning"
    assert completed["output"][0]["summary"][0]["text"] == "plan"


@pytest.mark.asyncio
async def test_responses_stream_marks_length_as_incomplete():
    frames = await _response_frames([
        InternalOutputEvent(kind="text_delta", text="partial"),
        InternalOutputEvent(kind="message_done", finish_reason="length"),
    ])
    completed = next(_payload(line)["response"] for line in frames if "response.completed" in line)
    assert completed["status"] == "incomplete"
    assert completed["incomplete_details"] == {"reason": "max_output_tokens"}


def test_anthropic_tool_input_is_object_for_non_object_json():
    messages = openai_messages_to_ir([{
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "run", "arguments": "[1, 2]",
        }}],
    }])
    block = ir_to_anthropic_messages(messages)[0][0]["content"][0]
    assert block["input"] == {"value": [1, 2]}


def test_anthropic_sampling_fields_survive_ingress_and_projection():
    internal = anthropic_messages_to_internal({
        "model": "m", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}],
        "stop_sequences": ["END"], "top_p": 0.2, "top_k": 5,
    })
    _, body = anthropic_body_from_internal(internal)
    request_body = _build_anthropic_request_body({}, [{"role": "user", "content": "hi"}], body, 100, 0.7, "m")
    assert request_body["stop_sequences"] == ["END"]
    assert request_body["top_p"] == 0.2
    assert request_body["top_k"] == 5


def test_responses_max_output_tokens_is_explicit_internal_budget():
    internal = responses_to_internal({"model": "m", "input": "hi", "max_output_tokens": 50})
    assert internal.max_tokens == 50
    assert internal.metadata["max_tokens_specified"] is True


def test_chat_reasoning_list_is_normalized_as_text():
    messages = openai_messages_to_ir([{
        "role": "assistant", "content": "answer",
        "reasoning_content": [{"type": "reasoning_text", "text": "deep thought"}],
    }])
    assert messages[0].parts[0].kind == "reasoning"
    assert messages[0].parts[0].text == "deep thought"


def test_responses_nonstream_marks_length_incomplete_and_keeps_reasoning():
    output = InternalOutputMessage(text="partial", reasoning="plan", finish_reason="length")
    result = render_response(output, model="m")
    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}
    assert [item["type"] for item in result["output"]] == ["reasoning", "message"]
    reasoning = result["output"][0]
    assert reasoning["summary"][0]["text"] == "plan"
