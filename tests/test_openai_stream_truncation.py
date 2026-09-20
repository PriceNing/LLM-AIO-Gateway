"""静默截断守卫：上游不发 finish_reason 且无回答输出时必须报错，不得合成正常结束。

背景：SuperGrok 2026-09-16 22:43 事故——服务端在 ~162s 掐流，只流出 311 字符
reasoning，网关合成 message_done(stop) 记 ok，pi/Codex 误判回合正常结束而停住。
"""
import pytest
from types import SimpleNamespace

from app.adapters.openai_streaming import iter_openai_chat_output_events
from app.core.text import client_status_for_upstream_error, friendly_error_msg


def _chunk(content=None, reasoning=None, finish=None, tool_calls=None, usage=None):
    delta = SimpleNamespace(
        role=None,
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage)


def _stream(*chunks):
    def factory(**kwargs):
        def gen():
            yield from chunks
        return gen()
    return factory


def _stream_then_error(*chunks, error):
    def factory(**kwargs):
        def gen():
            yield from chunks
            raise error
        return gen()
    return factory


async def _collect(events):
    out = []
    async for e in events:
        out.append(e)
    return out


def _patch(monkeypatch, factory):
    monkeypatch.setattr("app.adapters.openai_streaming.create_chat_completion_stream", factory)


def _assert_silent_truncation(exc):
    assert "without a finish reason" in str(exc)
    assert getattr(exc, "confirmed_upstream", False) is True
    assert getattr(exc, "empty_stream_response", False) is True
    assert client_status_for_upstream_error(exc) == 502
    assert "LEAK" not in friendly_error_msg(exc)
    assert "without a finish reason" not in friendly_error_msg(exc)


@pytest.mark.asyncio
async def test_reasoning_only_without_finish_raises(monkeypatch):
    _patch(monkeypatch, _stream(
        _chunk(reasoning="thinking hard..."),
        _chunk(reasoning="still thinking..."),
    ))
    with pytest.raises(RuntimeError, match="without a finish reason") as excinfo:
        await _collect(iter_openai_chat_output_events(
            model="m", messages=[], provider_id="p", temperature=0.7, max_tokens=64,
        ))
    _assert_silent_truncation(excinfo.value)


@pytest.mark.asyncio
async def test_empty_stream_without_finish_raises(monkeypatch):
    _patch(monkeypatch, _stream())
    with pytest.raises(RuntimeError, match="without a finish reason") as excinfo:
        await _collect(iter_openai_chat_output_events(
            model="m", messages=[], provider_id="p", temperature=0.7, max_tokens=64,
        ))
    _assert_silent_truncation(excinfo.value)


@pytest.mark.asyncio
async def test_usage_only_without_finish_raises(monkeypatch):
    usage = SimpleNamespace(
        prompt_tokens=3, completion_tokens=0, total_tokens=3,
        prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=0,
    )
    _patch(monkeypatch, _stream(_chunk(usage=usage)))
    with pytest.raises(RuntimeError, match="without a finish reason") as excinfo:
        await _collect(iter_openai_chat_output_events(
            model="m", messages=[], provider_id="p", temperature=0.7, max_tokens=64,
        ))
    _assert_silent_truncation(excinfo.value)


@pytest.mark.asyncio
async def test_reasoning_only_tail_error_still_raises_truncation(monkeypatch):
    _patch(monkeypatch, _stream_then_error(
        _chunk(reasoning="thinking hard..."),
        error=Exception("litellm.APIError: Error building chunks for logging/streaming usage calculation"),
    ))
    with pytest.raises(RuntimeError, match="without a finish reason") as excinfo:
        await _collect(iter_openai_chat_output_events(
            model="m", messages=[], provider_id="p", temperature=0.7, max_tokens=64,
        ))
    _assert_silent_truncation(excinfo.value)


@pytest.mark.asyncio
async def test_reasoning_with_finish_completes(monkeypatch):
    _patch(monkeypatch, _stream(
        _chunk(reasoning="thinking hard..."),
        _chunk(finish="stop"),
    ))
    events = await _collect(iter_openai_chat_output_events(
        model="m", messages=[], provider_id="p", temperature=0.7, max_tokens=64,
    ))
    assert events[-1].kind == "message_done"
    assert events[-1].finish_reason == "stop"
    assert any(e.kind == "reasoning_delta" for e in events)


@pytest.mark.asyncio
async def test_text_output_without_finish_still_completes(monkeypatch):
    # 宽松但可用的上游：有正文没 finish → 维持既有合成行为，不报错
    _patch(monkeypatch, _stream(_chunk(content="hello")))
    events = await _collect(iter_openai_chat_output_events(
        model="m", messages=[], provider_id="p", temperature=0.7, max_tokens=64,
    ))
    assert events[-1].kind == "message_done"
    assert events[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_finish_without_content_is_legit_empty(monkeypatch):
    _patch(monkeypatch, _stream(_chunk(finish="stop")))
    events = await _collect(iter_openai_chat_output_events(
        model="m", messages=[], provider_id="p", temperature=0.7, max_tokens=64,
    ))
    assert events[-1].kind == "message_done"


@pytest.mark.asyncio
async def test_tool_call_without_finish_still_completes(monkeypatch):
    tc = SimpleNamespace(index=0, id="call_1", type="function",
                         function=SimpleNamespace(name="exec", arguments='{"a":1}'))
    _patch(monkeypatch, _stream(_chunk(tool_calls=[tc])))
    events = await _collect(iter_openai_chat_output_events(
        model="m", messages=[], provider_id="p", temperature=0.7, max_tokens=64,
    ))
    assert events[-1].kind == "message_done"
    assert any(e.kind == "tool_call_done" for e in events)
