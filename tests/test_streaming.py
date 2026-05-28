"""Tests for app/core/streaming.py ¡ª record_streaming_events and stream_internal_output."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.output import InternalOutputEvent
from app.core.streaming import record_streaming_events, stream_internal_output


async def _collect_events(events):
    """Collect all events from an async generator."""
    result = []
    async for e in events:
        result.append(e)
    return result


async def _make_events(*kinds_and_data):
    """Helper: yield InternalOutputEvent from (kind, field, value) triples."""
    for kind, field, value in kinds_and_data:
        yield InternalOutputEvent(kind=kind, **{field: value})


# --- record_streaming_events ---

@pytest.mark.asyncio
async def test_record_text_delta_sets_has_text():
    """Text delta events should be tracked."""
    gen = _make_events(
        ("text_delta", "text", "hello"),
        ("message_done", "finish_reason", "stop"),
    )
    reasoning_tool_ids = []
    tool_only = MagicMock()
    tool_only.reset = MagicMock()

    events = await _collect_events(
        record_streaming_events(gen, conv_key="test", tool_only_turns=tool_only)
    )
    # tool_only_turns.reset should be called because has_text=True
    tool_only.reset.assert_called_once_with("test")


@pytest.mark.asyncio
async def test_record_reasoning_delta_accumulates():
    """Reasoning deltas should accumulate and be stored on message_done."""
    stored = {}

    def fake_remember(conv_key, reasoning, tool_ids):
        stored["key"] = conv_key
        stored["reasoning"] = reasoning
        stored["tool_ids"] = tool_ids

    gen = _make_events(
        ("reasoning_delta", "reasoning", "think "),
        ("reasoning_delta", "reasoning", "more"),
        ("message_done", "finish_reason", "stop"),
    )
    events = await _collect_events(
        record_streaming_events(
            gen,
            conv_key="conv1",
            remember_reasoning_content=fake_remember,
        )
    )
    assert stored["key"] == "conv1"
    assert stored["reasoning"] == "think more"


@pytest.mark.asyncio
async def test_record_tool_only_turn_increments():
    """Tool-only turn (tools but no text) should increment counter."""
    increment_mock = MagicMock(return_value=3)
    tool_only = MagicMock()
    tool_only.increment = increment_mock
    tool_only.reset = MagicMock()

    gen = _make_events(
        ("tool_call_start", "tool_call_id", "tc_1"),
        ("message_done", "finish_reason", "stop"),
    )
    events = await _collect_events(
        record_streaming_events(gen, conv_key="conv2", tool_only_turns=tool_only)
    )
    tool_only.increment.assert_called_once_with("conv2")
    tool_only.reset.assert_not_called()


@pytest.mark.asyncio
async def test_record_mixed_text_and_tools_resets():
    """Both text and tools should reset tool-only counter."""
    tool_only = MagicMock()
    tool_only.reset = MagicMock()

    gen = _make_events(
        ("text_delta", "text", "hi"),
        ("tool_call_start", "tool_call_id", "tc_1"),
        ("message_done", "finish_reason", "stop"),
    )
    events = await _collect_events(
        record_streaming_events(gen, conv_key="conv3", tool_only_turns=tool_only)
    )
    tool_only.reset.assert_called_once_with("conv3")


@pytest.mark.asyncio
async def test_record_finalize_on_stream_end_without_message_done():
    """If stream ends without message_done, finalize should still run."""
    stored = {}

    def fake_remember(conv_key, reasoning, tool_ids):
        stored["reasoning"] = reasoning

    gen = _make_events(
        ("reasoning_delta", "reasoning", "partial"),
    )
    events = await _collect_events(
        record_streaming_events(
            gen,
            conv_key="conv4",
            remember_reasoning_content=fake_remember,
        )
    )
    assert stored["reasoning"] == "partial"


@pytest.mark.asyncio
async def test_record_no_finalize_called_twice():
    """Finalize should only run once even if message_done fires."""
    call_count = [0]

    def counting_remember(conv_key, reasoning, tool_ids):
        call_count[0] += 1

    gen = _make_events(
        ("reasoning_delta", "reasoning", "thinking"),
        ("message_done", "finish_reason", "stop"),
    )
    events = await _collect_events(
        record_streaming_events(
            gen,
            conv_key="conv5",
            remember_reasoning_content=counting_remember,
        )
    )
    # message_done triggers finalize; no second call at stream end
    assert call_count[0] == 1


@pytest.mark.asyncio
async def test_record_no_remember_when_no_reasoning():
    """If no reasoning events fire, remember should not be called."""
    call_count = [0]

    def counting_remember(conv_key, reasoning, tool_ids):
        call_count[0] += 1

    gen = _make_events(
        ("text_delta", "text", "hello"),
        ("message_done", "finish_reason", "stop"),
    )
    events = await _collect_events(
        record_streaming_events(
            gen,
            conv_key="conv6",
            remember_reasoning_content=counting_remember,
        )
    )
    assert call_count[0] == 0


@pytest.mark.asyncio
async def test_record_no_tool_only_turns_when_none():
    """If tool_only_turns is None, no error should occur."""
    gen = _make_events(
        ("text_delta", "text", "hi"),
        ("message_done", "finish_reason", "stop"),
    )
    events = await _collect_events(
        record_streaming_events(gen, conv_key="conv7", tool_only_turns=None)
    )
    assert len(events) == 2


# --- stream_internal_output (focused on metadata/usage routing) ---

@pytest.mark.asyncio
async def test_stream_internal_output_extracts_usage():
    """Usage events should be captured for request logging."""
    logged = {}

    def fake_log(user, key, model, provider, endpoint, success, tokens, requested, **kwargs):
        logged["tokens"] = tokens
        logged["success"] = success

    async def _events():
        yield InternalOutputEvent(kind="text_delta", text="hi")
        yield InternalOutputEvent(kind="usage", usage={"total_tokens": 42})
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")
    events = _events()

    result = []
    async for line in stream_internal_output(
        events=events,
        endpoint="chat_completions",
        model="test-model",
        username="testuser",
        api_key_value="sk-test",
        provider_id="p1",
        requested_model="test-model",
        log_request=fake_log,
        conv_key="conv8",
    ):
        result.append(line)

    assert logged["tokens"] == 42
    assert logged["success"] is True


@pytest.mark.asyncio
async def test_stream_internal_output_updates_model_from_metadata():
    """Metadata events should update the final model used for logging."""
    logged = {}

    def fake_log(user, key, model, provider, endpoint, success, tokens, requested, **kwargs):
        logged["model"] = model
        logged["provider"] = provider

    async def _events():
        yield InternalOutputEvent(kind="metadata", metadata={"model": "real-model", "provider_id": "real-provider"})
        yield InternalOutputEvent(kind="text_delta", text="hi")
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")
    events = _events()

    result = []
    async for line in stream_internal_output(
        events=events,
        endpoint="chat_completions",
        model="original-model",
        username="u",
        api_key_value="k",
        provider_id="p1",
        requested_model="orig",
        log_request=fake_log,
        conv_key="conv9",
    ):
        result.append(line)

    assert logged["model"] == "real-model"
    assert logged["provider"] == "real-provider"


@pytest.mark.asyncio
async def test_stream_internal_output_logs_partial_failure_after_visible_output():
    """Failures after SSE output starts should keep the attempted target and mark partial output."""
    logged = {}

    def fake_log(user, key, model, provider, endpoint, success, tokens, requested, **kwargs):
        logged["model"] = model
        logged["provider"] = provider
        logged["success"] = success
        logged["tokens"] = tokens
        logged["details"] = kwargs.get("details") or {}

    async def _events():
        yield InternalOutputEvent(kind="metadata", metadata={"model": "real-model", "provider_id": "real-provider"})
        yield InternalOutputEvent(kind="text_delta", text="hi")
        exc = RuntimeError("upstream closed")
        exc.request_details = {
            "fallback_status": "skipped",
            "fallback_reason": "client_output_started",
            "error_trigger": "http_5xx",
            "attempted_model": "real-model",
            "attempted_provider": "real-provider",
            "partial_output": True,
        }
        raise exc

    result = []
    async for line in stream_internal_output(
        events=_events(),
        endpoint="responses",
        model="original-model",
        username="u",
        api_key_value="k",
        provider_id="p1",
        requested_model="orig",
        log_request=fake_log,
        conv_key="conv10",
    ):
        result.append(line)

    assert logged["success"] is False
    assert logged["model"] == "real-model"
    assert logged["provider"] == "real-provider"
    assert logged["details"]["status"] == "partial"
    assert logged["details"]["partial_output"] is True
    assert logged["details"]["fallback_status"] == "skipped"
