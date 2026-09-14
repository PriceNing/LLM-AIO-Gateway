"""针对协议层/适配器审查修复的回归测试。

覆盖：Anthropic URL 图片丢失、developer 角色、include_usage 转发、
Anthropic SSE block 乱序、无签名 thinking、tool_result.is_error、
Responses reasoning item 归属、错误帧 response id、cache token 统计、
缺失 tool index 推断、空 choices、中途重试不重复、截断流不伪装正常结束。
"""
import json

import httpx
import pytest
from fastapi import HTTPException

from app.adapters import anthropic_streaming
from app.adapters.anthropic import _anthropic_response_to_internal
from app.adapters.anthropic_streaming import iter_anthropic_output_events
from app.adapters.openai_streaming import _infer_tool_index
from app.adapters.output import response_to_internal_output
from app.core.output import InternalOutputEvent
from app.protocols.egress import (
    render_anthropic_messages_sse,
    render_chat_completions_sse,
    render_responses_error_sse,
)
from app.protocols.ir import (
    anthropic_messages_to_ir,
    ir_to_anthropic_messages,
    ir_to_openai_messages,
    openai_messages_to_ir,
    responses_input_to_ir,
)


# ---------------------------------------------------------------------------
# #2 Anthropic url 图片源保留
# ---------------------------------------------------------------------------

def test_anthropic_url_image_source_preserved_in_ir():
    messages = anthropic_messages_to_ir([
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
        ]},
    ])
    image = messages[0].parts[0]
    assert image.kind == "image"
    assert image.source["kind"] == "url"
    assert image.source["url"] == "https://example.com/a.png"

    openai_messages = ir_to_openai_messages(messages)
    assert openai_messages[0]["content"][0]["image_url"]["url"] == "https://example.com/a.png"


# ---------------------------------------------------------------------------
# #3 developer 角色映射为 system
# ---------------------------------------------------------------------------

def test_openai_developer_role_maps_to_system():
    messages = openai_messages_to_ir([
        {"role": "developer", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ])
    assert messages[0].role == "system"
    assert messages[0].parts[0].text == "be brief"

    openai_out = ir_to_openai_messages(messages)
    assert openai_out[0]["role"] == "system"


# ---------------------------------------------------------------------------
# #4 Chat SSE 转发 usage（include_usage）
# ---------------------------------------------------------------------------

async def _chat_sse_events():
    yield InternalOutputEvent(kind="text_delta", text="hi")
    yield InternalOutputEvent(kind="usage", usage={
        "input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
        "prompt_cache_hit_tokens": 1,
    })
    yield InternalOutputEvent(kind="message_done", finish_reason="stop")


@pytest.mark.asyncio
async def test_chat_sse_emits_usage_chunk_when_requested():
    frames = []
    async for line in render_chat_completions_sse(_chat_sse_events(), model="m", include_usage=True):
        frames.append(line)
    data_frames = [json.loads(f[len("data: "):]) for f in frames if f.startswith("data: ") and "[DONE]" not in f]
    usage_frames = [f for f in data_frames if f.get("choices") == [] and "usage" in f]
    assert len(usage_frames) == 1
    assert usage_frames[0]["usage"]["prompt_tokens"] == 3
    assert usage_frames[0]["usage"]["completion_tokens"] == 2
    assert usage_frames[0]["usage"]["total_tokens"] == 5
    assert usage_frames[0]["usage"]["prompt_tokens_details"]["cached_tokens"] == 1
    # usage 块必须在 finish 块之后
    finish_idx = next(i for i, f in enumerate(data_frames) if f["choices"] and f["choices"][0].get("finish_reason"))
    usage_idx = data_frames.index(usage_frames[0])
    assert usage_idx > finish_idx


@pytest.mark.asyncio
async def test_chat_sse_omits_usage_chunk_by_default():
    frames = []
    async for line in render_chat_completions_sse(_chat_sse_events(), model="m"):
        frames.append(line)
    joined = "".join(frames)
    assert '"usage"' not in joined


# ---------------------------------------------------------------------------
# #5 Anthropic SSE content block index 单调
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_sse_block_indices_monotonic_when_text_follows_tool_start():
    async def events():
        yield InternalOutputEvent(kind="tool_call_start", tool_index=0, tool_call_id="toolu_1", name="run")
        yield InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=0, tool_call_id="toolu_1", name="run", arguments_delta='{"x":1}', arguments='{"x":1}')
        yield InternalOutputEvent(kind="text_delta", text="done")
        yield InternalOutputEvent(kind="message_done", finish_reason="tool_calls")

    starts = []
    async for line in render_anthropic_messages_sse(events(), model="claude-test"):
        if line.startswith("event: content_block_start"):
            starts.append(json.loads(line.split("data: ", 1)[1]))

    indices = [s["index"] for s in starts]
    assert indices == sorted(indices), f"block index 不单调: {indices}"
    types = [s["content_block"]["type"] for s in starts]
    assert types == ["text", "tool_use"]


# ---------------------------------------------------------------------------
# #6 无签名 thinking 块不发给 Anthropic 上游
# ---------------------------------------------------------------------------

def test_unsigned_thinking_dropped_for_anthropic_upstream():
    messages = openai_messages_to_ir([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "hidden"},
        {"role": "user", "content": "q2"},
    ])
    anthropic_messages, _ = ir_to_anthropic_messages(messages)
    for block in anthropic_messages[1]["content"]:
        assert block.get("type") != "thinking"


def test_signed_thinking_kept_for_anthropic_upstream():
    messages = anthropic_messages_to_ir([
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan", "signature": "sig_ok"},
            {"type": "text", "text": "answer"},
        ]},
    ])
    anthropic_messages, _ = ir_to_anthropic_messages(messages)
    assert anthropic_messages[0]["content"][0] == {"type": "thinking", "thinking": "plan", "signature": "sig_ok"}


# ---------------------------------------------------------------------------
# #7 tool_result.is_error 保留
# ---------------------------------------------------------------------------

def test_tool_result_is_error_preserved_roundtrip():
    messages = anthropic_messages_to_ir([
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "run", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "boom", "is_error": True},
        ]},
    ])
    assert messages[1].parts[0].extensions.get("is_error") is True

    anthropic_messages, _ = ir_to_anthropic_messages(messages)
    block = anthropic_messages[1]["content"][0]
    assert block["is_error"] is True
    assert block["tool_use_id"] == "toolu_1"


# ---------------------------------------------------------------------------
# #8 Responses reasoning item 归属下一组
# ---------------------------------------------------------------------------

def test_responses_reasoning_before_function_call_attaches_to_that_group():
    result = ir_to_openai_messages(responses_input_to_ir([
        {"type": "message", "role": "user", "content": "q"},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "first plan"}]},
        {"type": "function_call", "call_id": "c1", "name": "search", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "second plan"}]},
        {"type": "function_call", "call_id": "c2", "name": "search", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c2", "output": "ok"},
    ]))
    assistants = [m for m in result if m["role"] == "assistant"]
    assert assistants[0].get("reasoning_content") == "first plan"
    assert assistants[1].get("reasoning_content") == "second plan"


# ---------------------------------------------------------------------------
# #10 Responses 错误帧复用 response id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_responses_error_sse_reuses_response_id():
    frames = []
    async for line in render_responses_error_sse(model="m", message="boom", response_id="resp_fixed"):
        frames.append(line)
    joined = "".join(frames)
    completed = json.loads([f for f in frames if "response.completed" in f][0][len("data: "):])
    assert completed["response"]["id"] == "resp_fixed"
    assert "resp_fixed" in joined


# ---------------------------------------------------------------------------
# #11 Anthropic cache token 统计
# ---------------------------------------------------------------------------

def test_anthropic_nonstream_usage_includes_cache_tokens():
    output = _anthropic_response_to_internal({
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10, "output_tokens": 4,
            "cache_creation_input_tokens": 3, "cache_read_input_tokens": 7,
        },
    })
    assert output.usage["prompt_cache_hit_tokens"] == 7
    assert output.usage["prompt_cache_miss_tokens"] == 3


# ---------------------------------------------------------------------------
# #12 缺失 index 的工具增量推断
# ---------------------------------------------------------------------------

def test_infer_tool_index_by_id():
    states = {0: {"id": "call_a"}}
    assert _infer_tool_index({"id": "call_a"}, states) == 0
    assert _infer_tool_index({"id": "call_b"}, states) == 1
    assert _infer_tool_index({}, states) == 0
    assert _infer_tool_index({"id": "call_x"}, {}) == 0


# ---------------------------------------------------------------------------
# #13 空 choices 友好报错
# ---------------------------------------------------------------------------

def test_empty_choices_raises_friendly_error():
    class FakeResponse:
        choices = []

    with pytest.raises(ValueError):
        response_to_internal_output(FakeResponse())


# ---------------------------------------------------------------------------
# #1 Anthropic 中途失败不重试（不重复输出）
# ---------------------------------------------------------------------------

class _MidStreamErrorStream:
    status_code = 200

    def __init__(self):
        self.calls = 0

    async def __aenter__(self):
        self.calls += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        yield "event: message_start"
        yield 'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}'
        yield "event: content_block_start"
        yield 'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}'
        yield "event: content_block_delta"
        yield 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}'
        raise httpx.ReadError("connection reset by peer")


@pytest.mark.asyncio
async def test_anthropic_stream_no_retry_after_partial_output(monkeypatch):
    stream = _MidStreamErrorStream()

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            return stream

    monkeypatch.setattr(anthropic_streaming, "shared_client", lambda *args, **kwargs: FakeClient())

    texts = []
    with pytest.raises((httpx.ReadError, HTTPException)):
        async for event in iter_anthropic_output_events(
            provider_info={"id": "anth", "api_base": "https://a.example", "api_key": "k", "retry_count": 3},
            messages=[{"role": "user", "content": "hi"}],
            body={},
            max_tokens=16,
            temperature=0.7,
            model="claude-test",
        ):
            if event.kind == "text_delta":
                texts.append(event.text)
    # 关键断言：已输出的内容绝不重复，且不会发起第二次请求
    assert texts == ["hello"]
    assert stream.calls == 1


@pytest.mark.asyncio
async def test_anthropic_stream_retries_before_first_output(monkeypatch):
    class FailThenOkStream:
        status_code = 200

        def __init__(self, fail: bool):
            self.fail = fail

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            if self.fail:
                raise httpx.ConnectError("connection refused")
                yield  # pragma: no cover
            yield "event: message_start"
            yield 'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}'
            yield "event: content_block_delta"
            yield 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}'
            yield "event: message_delta"
            yield 'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}'
            yield "event: message_stop"
            yield 'data: {"type":"message_stop"}'

    attempts = {"n": 0}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            attempts["n"] += 1
            return FailThenOkStream(fail=attempts["n"] == 1)

    monkeypatch.setattr(anthropic_streaming, "shared_client", lambda *args, **kwargs: FakeClient())

    texts = []
    async for event in iter_anthropic_output_events(
        provider_info={"id": "anth", "api_base": "https://a.example", "api_key": "k", "retry_count": 1, "retry_backoff": 0},
        messages=[{"role": "user", "content": "hi"}],
        body={},
        max_tokens=16,
        temperature=0.7,
        model="claude-test",
    ):
        if event.kind == "text_delta":
            texts.append(event.text)
    assert texts == ["ok"]
    assert attempts["n"] == 2


# ---------------------------------------------------------------------------
# #14 截断流不伪装正常结束
# ---------------------------------------------------------------------------

class _TruncatedStream:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        yield "event: message_start"
        yield 'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}'
        yield "event: content_block_delta"
        yield 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}'
        # 没有 message_delta / message_stop，连接直接关闭


@pytest.mark.asyncio
async def test_anthropic_truncated_stream_raises(monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            return _TruncatedStream()

    monkeypatch.setattr(anthropic_streaming, "shared_client", lambda *args, **kwargs: FakeClient())

    with pytest.raises(HTTPException) as exc_info:
        async for _ in iter_anthropic_output_events(
            provider_info={"id": "anth", "api_base": "https://a.example", "api_key": "k"},
            messages=[{"role": "user", "content": "hi"}],
            body={},
            max_tokens=16,
            temperature=0.7,
            model="claude-test",
        ):
            pass
    assert exc_info.value.status_code == 502
