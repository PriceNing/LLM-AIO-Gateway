"""Chat Completions image-bridge tests (Phase B).

The gateway image bridge was lifted out of ``/responses`` into
``app/core/image_orchestration.run_image_bridge`` so ``/chat/completions``
can share the same model-driven image generation. These tests lock that
behavior for the chat wire format (markdown with inline data-URI previews
plus HTTP download links).
"""

import json

import pytest
from fastapi.testclient import TestClient

from main import app
from app.database import (
    add_admin,
    add_provider,
    add_user,
    add_user_api_key,
    init_db,
    list_request_logs,
    set_model_image_generation,
    upsert_image_generator,
)
from app.security import hash_password
from app.adapters.imagegen import ImageGenerationResult
from app.core.output import InternalOutputEvent, InternalOutputMessage, InternalToolCallOutput
from app.core.image_bridge import IMAGE_BRIDGE_TOOL_NAME, GATEWAY_IMAGE_ASSET_MARKER


@pytest.fixture
def chat_image_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "gateway.db")
    init_db(db_path)
    add_admin("admin", hash_password("secret"), "Admin")
    add_user({"username": "alice", "display_name": "Alice", "enabled": True})
    key = add_user_api_key("alice", "default", ["chat/chat-model"])["key"]
    add_provider({
        "id": "chat", "name": "Chat", "api_base": "http://chat.test/v1",
        "models": [{"id": "chat-model", "name": "Chat Model"}],
    })
    set_model_image_generation("chat/chat-model", True)
    upsert_image_generator(
        "default", {"api_base": "http://image.test/v1", "model": "image-model", "enabled": True}
    )
    image_dir = tmp_path / "generated-images"
    monkeypatch.setattr("app.core.image_results.image_result_directory", lambda: image_dir)
    conv_key = "test:conv:key"
    monkeypatch.setattr(
        "app.router.proxy._conversation_cache_key",
        lambda *args, **kwargs: conv_key,
    )
    return {
        "headers": {"Authorization": f"Bearer {key}"},
        "image_dir": image_dir,
        "conv_key": conv_key,
    }


def test_chat_model_driven_bridge_generates_after_model_tool_call(chat_image_db, monkeypatch):
    calls = []

    async def fake_planner(*args, **kwargs):
        return InternalOutputMessage(
            tool_calls=[InternalToolCallOutput(
                id="call_image", call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME,
                arguments=json.dumps({"prompt": "生成一个苹果的图像"}, ensure_ascii=False),
            )],
            finish_reason="tool_calls", usage={"total_tokens": 7},
        ), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        calls.append(kwargs["prompt"])
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "messages": [{"role": "user", "content": "生成一个苹果的图像"}]},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    # Private bridge markers must never reach the client.
    assert GATEWAY_IMAGE_ASSET_MARKER not in content
    # Chat wire format: download URL only, NO inline base64 data URI. Inlining
    # data URIs inflates the assistant message text (a 1MB PNG becomes ~250KB
    # of base64); the agent loop then echoes the whole blob back into the next
    # request's history, where it triggers the "model says task done" empty
    # response that leads to correction+502. Codex /responses still uses the
    # data-URI form because that protocol requires it.
    assert "data:image/png;base64" not in content
    assert "[`generated-asset-1.png` — download original](http://testserver/v1/image-results/" in content
    assert len(list(chat_image_db["image_dir"].glob("*.png"))) == 1
    assert calls == ["生成一个苹果的图像"]


def test_chat_ordinary_request_injects_bridge_but_does_not_generate(chat_image_db, monkeypatch):
    """现代 harness 约定：生图模型上桥接工具常驻，普通请求由模型自主不调用。"""
    calls = 0

    async def fake_planner(policy, internal, **kwargs):
        nonlocal calls
        calls += 1
        # The bridge tool stays resident on image-capable models; the model
        # simply does not call it for ordinary requests.
        assert any(tool.name == IMAGE_BRIDGE_TOOL_NAME for tool in internal.tools)
        return InternalOutputMessage(text="Reviewed the code.", usage={"total_tokens": 3}), {"id": "chat"}, "chat"

    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "messages": [{"role": "user", "content": "Review the code and explain the failing test."}]},
    )
    assert response.status_code == 200, response.text
    assert calls == 1
    assert "Reviewed the code." in response.text
    assert not list(chat_image_db["image_dir"].glob("*.png"))


def test_chat_image_intent_model_without_capability_does_not_bridge(chat_image_db, monkeypatch):
    set_model_image_generation("chat/chat-model", False)
    calls = 0

    async def fake_planner(policy, internal, **kwargs):
        nonlocal calls
        calls += 1
        assert all(tool.name != IMAGE_BRIDGE_TOOL_NAME for tool in internal.tools)
        return InternalOutputMessage(text="I cannot generate images.", usage={"total_tokens": 3}), {"id": "chat"}, "chat"

    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "messages": [{"role": "user", "content": "Generate an image of an apple"}]},
    )
    assert response.status_code == 200, response.text
    assert calls == 1
    assert "I cannot generate images." in response.text
    assert not list(chat_image_db["image_dir"].glob("*.png"))


def test_chat_image_bridge_passthrough_when_model_makes_other_tool_call(chat_image_db, monkeypatch):
    """When the planner makes a non-image tool call, the bridge falls through to
    normal rendering and never exposes the private tool name."""
    calls = 0

    async def fake_planner(policy, internal, **kwargs):
        nonlocal calls
        calls += 1
        # Model chooses a client-owned tool, not the private image bridge.
        return InternalOutputMessage(
            tool_calls=[InternalToolCallOutput(
                id="call_search", call_id="call_search", name="web_search",
                arguments=json.dumps({"query": "apple"}),
            )],
            finish_reason="tool_calls", usage={"total_tokens": 4},
        ), {"id": "chat"}, "chat"

    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "messages": [{"role": "user", "content": "Generate an image of an apple"}]},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    message = payload["choices"][0]["message"]
    # The client-owned tool call is preserved, the private bridge tool is not.
    tool_calls = message.get("tool_calls") or []
    assert any(tc["function"]["name"] == "web_search" for tc in tool_calls)
    assert IMAGE_BRIDGE_TOOL_NAME not in response.text
    assert not list(chat_image_db["image_dir"].glob("*.png"))


def test_chat_image_bridge_empty_response_passes_through(chat_image_db, monkeypatch):
    """纯模型驱动：模型没调用生图工具（甚至完全空响应）时不 502、不纠正，
    原样 passthrough。"""
    async def fake_planner(policy, internal, **kwargs):
        # Completely empty: no text, no tool calls.
        return InternalOutputMessage(usage={"total_tokens": 0}), {"id": "chat"}, "chat"

    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "messages": [{"role": "user", "content": "Generate an image of an apple"}]},
    )
    assert response.status_code == 200, response.text
    log = list_request_logs(limit=1)[0]
    assert log["status"] == "ok"
    assert log["details"].get("image_correction_applied") in (None, False)


def test_chat_stream_image_intent_generates(chat_image_db, monkeypatch):
    """Streaming chat with an image request now runs the bridge: it buffers the
    upstream stream, detects the image tool call, generates, and streams the
    result back as chat SSE (the private bridge marker never leaks)."""
    seen = {}
    calls = []

    async def fake_stream(internal, **kwargs):
        seen["tools"] = [tool.name for tool in internal.tools]
        yield InternalOutputEvent(kind="tool_call_start", tool_index=0, tool_call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME)
        yield InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=0, tool_call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME, arguments_delta='{"prompt": "apple"}')
        yield InternalOutputEvent(kind="tool_call_done", tool_index=0, tool_call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME, arguments='{"prompt": "apple"}')
        yield InternalOutputEvent(kind="message_done", finish_reason="tool_calls")

    async def fake_continuation(*args, **kwargs):
        return InternalOutputMessage(text="Here is your apple.", finish_reason="stop", usage={"total_tokens": 5}), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        calls.append(kwargs["prompt"])
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy._stream_events_with_fallbacks", fake_stream)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_continuation)
    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "stream": True, "messages": [{"role": "user", "content": "Generate an image of an apple"}]},
    )
    assert response.status_code == 200, response.text
    assert IMAGE_BRIDGE_TOOL_NAME in seen["tools"]
    # Chat wire format: download URL only, NO inline data URI -- see
    # ``test_chat_model_driven_bridge_generates_after_model_tool_call`` for
    # why inlining the base64 blows up the agent loop's history.
    assert "data:image/png;base64,AAAA" not in response.text
    assert "/v1/image-results/" in response.text
    # Private bridge marker must never reach the client.
    assert GATEWAY_IMAGE_ASSET_MARKER not in response.text
    assert calls == ["apple"]
    assert len(list(chat_image_db["image_dir"].glob("*.png"))) == 1
    # M5: a progress notice is streamed first so the client is not left on a
    # silent stream for the whole out-of-band generation.
    assert "正在生成图片" in response.text


def test_chat_stream_bridge_output_is_url_only_no_inline_data_uri(chat_image_db, monkeypatch):
    """Lock the chat wire format: bridge output is download URLs only, never
    inline base64 data URIs. Inlining the base64 inflates the assistant
    message text (a 1MB PNG becomes ~250KB of base64); the agent loop then
    echoes the whole blob back into the next request's history, where it
    triggers the "model says task done" empty response that leads to
    correction+502. Codex /responses still uses the data-URI form because
    that protocol requires it."""

    async def fake_stream(internal, **kwargs):
        yield InternalOutputEvent(kind="tool_call_start", tool_index=0, tool_call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME)
        yield InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=0, tool_call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME, arguments_delta='{"prompt": "red apple"}')
        yield InternalOutputEvent(kind="tool_call_done", tool_index=0, tool_call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME, arguments='{"prompt": "red apple"}')
        yield InternalOutputEvent(kind="message_done", finish_reason="tool_calls")

    async def fake_continuation(*args, **kwargs):
        return InternalOutputMessage(text="Here is your apple.", finish_reason="stop", usage={"total_tokens": 5}), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        # Even with a large data URI here, the chat output must NOT include
        # the base64 -- only the download URL.
        big = "A" * 300_000
        return [ImageGenerationResult(f"data:image/png;base64,{big}")]

    monkeypatch.setattr("app.router.proxy._stream_events_with_fallbacks", fake_stream)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_continuation)
    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "stream": True, "messages": [{"role": "user", "content": "Generate an image of an apple"}]},
    )
    assert response.status_code == 200, response.text
    assert "data:image/png;base64" not in response.text, "data URI leaked into chat response -- would bloat history"
    # The download URL is present.
    assert "/v1/image-results/" in response.text
    # The response is small (URL-only) regardless of the source image size.
    assert len(response.text) < 5_000, f"chat response too large: {len(response.text)} bytes"


def test_chat_stream_text_only_response_passes_through(chat_image_db, monkeypatch):
    """纯模型驱动：模型只输出文本（不调用生图工具）时，缓冲的流原样
    passthrough，不追加指令、不强制 tool_choice。"""

    async def fake_stream(internal, **kwargs):
        yield InternalOutputEvent(kind="text_delta", text="好的")
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    async def fake_nonstream(policy, internal, **kwargs):
        raise AssertionError("纯模型驱动：不允许 correction 重调用")

    monkeypatch.setattr("app.router.proxy._stream_events_with_fallbacks", fake_stream)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_nonstream)
    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "stream": True, "messages": [{"role": "user", "content": "Generate an image of an apple"}]},
    )
    assert response.status_code == 200, response.text
    # 模型文本原样返回
    assert "好的" in response.text
    # No "image generation" notice -- nothing was generated.
    assert "正在生成图片" not in response.text
    # The private bridge marker must never reach the client.
    assert GATEWAY_IMAGE_ASSET_MARKER not in response.text


def test_chat_stream_passthrough_updates_tool_only_circuit_breaker(chat_image_db, monkeypatch):
    """When the bridge passthrough contains only tool_calls (no text), the
    tool-only circuit breaker counter must increment -- otherwise a runaway
    agent loop in passthrough mode would never trip the bridge limiter."""

    async def fake_stream(internal, **kwargs):
        # 模型只调用非生图工具（bash）-> 缓冲路径 passthrough
        yield InternalOutputEvent(kind="tool_call_start", tool_index=0, tool_call_id="call_bash", name="bash")
        yield InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=0, tool_call_id="call_bash", name="bash", arguments_delta='{"command": "ls"}')
        yield InternalOutputEvent(kind="tool_call_done", tool_index=0, tool_call_id="call_bash", name="bash", arguments='{"command": "ls"}')
        yield InternalOutputEvent(kind="message_done", finish_reason="tool_calls")

    async def fake_nonstream(policy, internal, **kwargs):
        raise AssertionError("纯模型驱动：不允许 correction 重调用")

    monkeypatch.setattr("app.router.proxy._stream_events_with_fallbacks", fake_stream)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_nonstream)
    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "stream": True, "messages": [{"role": "user", "content": "Generate an image"}]},
    )
    assert response.status_code == 200, response.text
    # tool_only counter for this conversation key must have been incremented
    # -- otherwise the circuit breaker never fires for a passthrough loop.
    from app.core.state import tool_only_turns as _state
    counter = _state.get(chat_image_db["conv_key"], 0)
    assert counter >= 1, f"tool_only counter not bumped: {counter}"


def test_chat_stream_ordinary_request_keeps_true_streaming(chat_image_db, monkeypatch):
    """普通流式请求保持真流式：桥接工具常驻注入，但不进入缓冲 bridge 循环，
    客户端流里不出现私有工具调用事件。"""
    seen = {}

    async def fake_stream(internal, **kwargs):
        seen["tools"] = [tool.name for tool in internal.tools]
        yield InternalOutputEvent(kind="text_delta", text="hello")
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    monkeypatch.setattr("app.router.proxy._stream_events_with_fallbacks", fake_stream)
    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "stream": True, "messages": [{"role": "user", "content": "Say hello"}]},
    )
    assert response.status_code == 200, response.text
    # 工具常驻注入（现代 harness 约定），但走真流式路径而非缓冲 bridge 循环
    assert IMAGE_BRIDGE_TOOL_NAME in seen["tools"]
    assert "hello" in response.text


def test_chat_stream_followup_after_generated_image_enters_bridge(chat_image_db, monkeypatch):
    """历史里有网关生图结果（/v1/image-results/ 链接）时，跟进修改消息
    （无生图关键词）也进入缓冲 bridge 路径，而不是真流式。"""
    seen = {}

    async def fake_stream(internal, **kwargs):
        seen["tools"] = [tool.name for tool in internal.tools]
        yield InternalOutputEvent(kind="text_delta", text="ok")
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    monkeypatch.setattr("app.router.proxy._stream_events_with_fallbacks", fake_stream)
    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "stream": True, "messages": [
            {"role": "user", "content": "Generate an image"},
            {"role": "assistant", "content": "Generated image: [`a.png` — download original](/v1/image-results/abc123)"},
            {"role": "user", "content": "把背景改成蓝色"},
        ]},
    )
    assert response.status_code == 200, response.text
    # 注入了 bridge 工具且走缓冲路径（fake_stream 即缓冲路径的上游调用）
    assert IMAGE_BRIDGE_TOOL_NAME in seen["tools"]
    # 模型文本原样 passthrough
    assert "ok" in response.text


def test_chat_stream_spontaneous_bridge_tool_call_is_executed(chat_image_db, monkeypatch):
    """纯模型驱动：模型自发调用 bridge 工具（无意图关键词、无生图历史）时，
    缓冲路径照常执行生图，私有工具名不泄漏给客户端。"""

    async def fake_stream(internal, **kwargs):
        yield InternalOutputEvent(kind="tool_call_start", tool_index=0, tool_call_id="call_x", name=IMAGE_BRIDGE_TOOL_NAME)
        yield InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=0, tool_call_id="call_x", name=IMAGE_BRIDGE_TOOL_NAME, arguments_delta='{"prompt": "a cat"}')
        yield InternalOutputEvent(kind="tool_call_done", tool_index=0, tool_call_id="call_x", name=IMAGE_BRIDGE_TOOL_NAME, arguments='{"prompt": "a cat"}')
        yield InternalOutputEvent(kind="message_done", finish_reason="tool_calls")

    async def fake_continuation(*args, **kwargs):
        return InternalOutputMessage(text="Here is the cat.", finish_reason="stop", usage={"total_tokens": 5}), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy._stream_events_with_fallbacks", fake_stream)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_continuation)
    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "stream": True, "messages": [{"role": "user", "content": "给我来点视觉素材"}]},
    )
    assert response.status_code == 200, response.text
    assert "/v1/image-results/" in response.text
    assert IMAGE_BRIDGE_TOOL_NAME not in response.text
