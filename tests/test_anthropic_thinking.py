from fastapi import HTTPException
import pytest

from app.adapters.anthropic import (
    _anthropic_response_to_internal,
    _build_anthropic_request_body,
    _http_exception_from_upstream,
)
from app.core.output import InternalOutputEvent
from app.protocols.egress import render_anthropic_message, render_anthropic_messages_sse
from app.protocols.ingress import anthropic_messages_to_internal
from app.protocols.ir import ir_to_anthropic_messages


def test_enabled_thinking_includes_budget_and_drops_temperature():
    body = _build_anthropic_request_body(
        {"provider_options": {"thinking": "enabled"}},
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        {},
        4096,
        0.7,
        "claude-sonnet-4-6",
    )
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "temperature" not in body


def test_enabled_thinking_respects_custom_budget_and_caps_below_max_tokens():
    body = _build_anthropic_request_body(
        {"provider_options": {"thinking": "enabled", "thinking_budget_tokens": 8000}},
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        {},
        2048,
        1,
        "claude-sonnet-4-6",
    )
    assert body["thinking"]["budget_tokens"] == 2047


def test_thinking_signature_survives_ir_roundtrip():
    req = anthropic_messages_to_internal({
        "model": "claude-test",
        "messages": [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hidden", "signature": "sig_keep"},
                {"type": "text", "text": "answer"},
            ],
        }],
    })
    messages, _ = ir_to_anthropic_messages(req.messages)
    assert messages[0]["content"][0] == {
        "type": "thinking",
        "thinking": "hidden",
        "signature": "sig_keep",
    }


def test_redacted_thinking_survives_ir_roundtrip():
    req = anthropic_messages_to_internal({
        "model": "claude-test",
        "messages": [{
            "role": "assistant",
            "content": [{"type": "redacted_thinking", "data": "opaque"}],
        }],
    })
    messages, _ = ir_to_anthropic_messages(req.messages)
    assert messages[0]["content"][0] == {"type": "redacted_thinking", "data": "opaque"}


def test_anthropic_client_error_keeps_4xx_status():
    exc = _http_exception_from_upstream(400, "invalid thinking payload")
    assert isinstance(exc, HTTPException)
    assert exc.status_code == 400


def test_anthropic_server_error_stays_502():
    exc = _http_exception_from_upstream(503, "unavailable")
    assert exc.status_code == 502


def test_render_anthropic_message_emits_thinking_block():
    output = _anthropic_response_to_internal({
        "content": [
            {"type": "thinking", "thinking": "plan", "signature": "sig_out"},
            {"type": "text", "text": "done"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    })
    rendered = render_anthropic_message(output, model="claude-test")
    assert rendered["content"][0] == {"type": "thinking", "thinking": "plan", "signature": "sig_out"}
    assert rendered["content"][1] == {"type": "text", "text": "done"}


@pytest.mark.asyncio
async def test_anthropic_messages_sse_emits_thinking_block():
    async def events():
        yield InternalOutputEvent(kind="reasoning_delta", reasoning="plan")
        yield InternalOutputEvent(kind="reasoning_delta", reasoning_signature="sig_stream")
        yield InternalOutputEvent(kind="text_delta", text="done")
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    chunks = []
    async for line in render_anthropic_messages_sse(events(), model="claude-test"):
        chunks.append(line)
    joined = "".join(chunks)
    assert '"type": "thinking"' in joined
    assert '"type": "thinking_delta"' in joined
    assert '"thinking": "plan"' in joined
    assert '"type": "signature_delta"' in joined
    assert '"signature": "sig_stream"' in joined
    assert '"type": "text_delta"' in joined
    assert joined.index("thinking_delta") < joined.index("text_delta")
