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
    return {"headers": {"Authorization": f"Bearer {key}"}, "image_dir": image_dir}


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
    # Option (a): inline data-URI preview plus an HTTP download link.
    assert "![Generated image](data:image/png;base64,AAAA)" in content
    assert "[`generated-asset-1.png` — download original](http://testserver/v1/image-results/" in content
    assert content.index("Original:") < content.index("data:image/png;base64")
    assert len(list(chat_image_db["image_dir"].glob("*.png"))) == 1
    assert calls == ["生成一个苹果的图像"]


def test_chat_ordinary_request_does_not_enter_image_bridge(chat_image_db, monkeypatch):
    calls = 0

    async def fake_planner(policy, internal, **kwargs):
        nonlocal calls
        calls += 1
        # No image intent -> the private bridge tool must not be injected.
        assert all(tool.name != IMAGE_BRIDGE_TOOL_NAME for tool in internal.tools)
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


def test_chat_image_bridge_correction_failure_is_logged(chat_image_db, monkeypatch):
    """M4: when the model never invokes the image tool, the 502 is recorded as a
    failure (request log + stats), not silently passed through."""
    async def fake_planner(policy, internal, **kwargs):
        # Decline the image tool on both the initial and correction rounds.
        return InternalOutputMessage(text="I will not generate an image.", usage={"total_tokens": 5}), {"id": "chat"}, "chat"

    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "messages": [{"role": "user", "content": "Generate an image of an apple"}]},
    )
    assert response.status_code == 502, response.text
    assert "did not invoke the image-generation tool" in response.json()["detail"]
    log = list_request_logs(limit=1)[0]
    assert log["status"] == "fail"
    assert log["details"]["request_kind"] == "image_generation"
    assert log["details"].get("image_correction_applied") is True


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
    # The generated image is streamed back with the inline data-URI preview.
    assert "data:image/png;base64,AAAA" in response.text
    # Private bridge marker must never reach the client.
    assert GATEWAY_IMAGE_ASSET_MARKER not in response.text
    assert calls == ["apple"]
    assert len(list(chat_image_db["image_dir"].glob("*.png"))) == 1
    # M5: a progress notice is streamed first so the client is not left on a
    # silent stream for the whole out-of-band generation.
    assert "正在生成图片" in response.text


def test_chat_stream_ordinary_request_has_no_bridge(chat_image_db, monkeypatch):
    """A streaming chat request without image intent does not get the bridge
    tool and streams normally."""
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
    assert IMAGE_BRIDGE_TOOL_NAME not in seen["tools"]
    assert "hello" in response.text
