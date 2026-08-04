import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from main import app
from app.database import add_admin, add_provider, add_user, add_user_api_key
from app.security import create_session, hash_password

from app.adapters.imagegen import (
    ImageGenerationResult, generate_images, image_results_bytes, images_url,
)
from app.database import (
    add_provider,
    get_enabled_image_generator,
    get_global_stats,
    get_model_image_generation,
    init_db,
    list_request_logs,
    set_model_image_generation,
    upsert_image_generator,
)
from app.protocols.egress import render_response, render_responses_image_generation
from app.protocols.ingress import responses_to_internal
from app.core.image_intent import is_image_generation_intent, latest_user_text
from app.core.image_bridge import (
    GATEWAY_IMAGE_DISPLAY_CALL_PREFIX,
    IMAGE_BRIDGE_MARKER,
    IMAGE_BRIDGE_TOOL_NAME,
    has_codex_generated_image_exec_tool,
    has_codex_image_function_tool,
    image_call_arguments_from_exec,
    inject_hosted_image_capability,
)
from app.core.image_results import find_image_result, image_preview_data_uri, store_image_results
from app.core.output import InternalOutputMessage, InternalToolCallOutput


@pytest.fixture
def image_app_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "gateway.db")
    init_db(db_path)
    add_admin("admin", hash_password("secret"), "Admin")
    add_user({"username": "alice", "display_name": "Alice", "enabled": True})
    key = add_user_api_key("alice", "default", ["chat/chat-model"])["key"]
    add_provider({"id": "chat", "name": "Chat", "api_base": "http://chat.test/v1", "models": [{"id": "chat-model", "name": "Chat Model"}]})
    set_model_image_generation("chat/chat-model", True)
    upsert_image_generator("default", {"api_base": "http://image.test/v1", "model": "image-model", "enabled": True})
    image_dir = tmp_path / "generated-images"
    monkeypatch.setattr("app.core.image_results.image_result_directory", lambda: image_dir)
    return {"headers": {"Authorization": f"Bearer {key}"}, "image_dir": image_dir}


def test_images_url_normalizes_common_api_base_forms():
    assert images_url("https://example.test") == "https://example.test/v1/images/generations"
    assert images_url("https://example.test/v1") == "https://example.test/v1/images/generations"
    assert images_url("https://example.test/v1/images/generations") == "https://example.test/v1/images/generations"


def test_image_results_bytes_counts_decoded_payload():
    assert image_results_bytes([
        ImageGenerationResult("data:image/png;base64,QUJDRA=="),
        ImageGenerationResult("data:image/jpeg;base64,RUY="),
    ]) == 6


def test_image_preview_is_jpeg_and_bounded(tmp_path, monkeypatch):
    from PIL import Image
    from io import BytesIO

    source = BytesIO()
    Image.new("RGBA", (2400, 1600), (40, 120, 220, 255)).save(source, format="PNG")
    result = ImageGenerationResult("data:image/png;base64," + base64.b64encode(source.getvalue()).decode())
    monkeypatch.setattr("app.core.image_results.get_default", lambda key, fallback=None: {
        "image_preview_enabled": True,
        "image_preview_max_dimension": 640,
        "image_preview_quality": 82,
        "image_preview_max_bytes": 200000,
    }.get(key, fallback))
    preview = image_preview_data_uri(result)
    assert preview.startswith("data:image/jpeg;base64,")
    preview_bytes = base64.b64decode(preview.split(",", 1)[1])
    assert len(preview_bytes) <= 200000
    with Image.open(BytesIO(preview_bytes)) as image:
        assert image.format == "JPEG"
        assert max(image.size) <= 640


def test_image_generator_config_and_model_flag(tmp_path):
    init_db(str(tmp_path / "images.db"))
    add_provider({"id": "chat", "name": "Chat", "models": [{"id": "chat-model", "name": "Chat Model"}]})
    upsert_image_generator("default", {"api_base": "https://example.test/v1", "model": "image-model", "enabled": True})
    assert get_enabled_image_generator()["model"] == "image-model"
    assert set_model_image_generation("chat/chat-model", True) is True
    assert get_model_image_generation("chat", "chat-model") is True


@pytest.mark.anyio
async def test_generate_images_normalizes_b64_json(monkeypatch):
    raw = b"fake-png"

    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"b64_json": base64.b64encode(raw).decode("ascii"), "revised_prompt": "revised"}]}

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return MockResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: MockClient())
    result = await generate_images({"api_base": "http://image.test/v1", "model": "image-model"}, prompt="an apple")
    assert result[0].data_uri == "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    assert result[0].revised_prompt == "revised"


@pytest.mark.anyio
async def test_generate_images_detects_jpeg_b64(monkeypatch):
    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"b64_json": base64.b64encode(b"\xff\xd8\xfffake").decode("ascii")}]}

    class MockClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def post(self, *args, **kwargs): return MockResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: MockClient())
    result = await generate_images({"api_base": "http://image.test/v1", "model": "image-model"}, prompt="an apple")
    assert result[0].mime_type == "image/jpeg"


@pytest.mark.anyio
async def test_generate_images_preserves_upstream_error_body(monkeypatch):
    class MockResponse:
        status_code = 400
        is_error = True
        text = '{"error":"Argument not supported: size"}'

    class MockClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def post(self, *args, **kwargs): return MockResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: MockClient())
    with pytest.raises(ValueError, match="Argument not supported: size"):
        await generate_images({"api_base": "http://image.test/v1", "model": "image-model"}, prompt="an apple")


def test_render_responses_image_generation():
    response = render_responses_image_generation([ImageGenerationResult("data:image/png;base64,AAAA", size="1024x1024")], model="chat-model")
    assert response["object"] == "response"
    assert response["model"] == "chat-model"
    assert response["output"][0]["type"] == "image_generation_call"
    assert response["output"][0]["result"] == "AAAA"
    assert response["completed_at"] == response["created_at"]
    assert response["parallel_tool_calls"] is True
    assert response["tool_usage"]["image_gen"]["total_tokens"] == 0


@pytest.mark.anyio
async def test_render_responses_image_generation_sse_contains_image_events():
    from app.protocols.egress import render_responses_image_generation_sse
    events = [event async for event in render_responses_image_generation_sse(
        [ImageGenerationResult("data:image/png;base64,AAAA")], model="chat-model"
    )]
    payloads = [json.loads(event.split("data: ", 1)[1]) for event in events if "data: {" in event]
    assert all(event.startswith("data: {") for event in events if "data: {" in event)
    assert [item["type"] for item in payloads] == [
        "response.created", "response.in_progress", "response.output_item.done", "response.completed",
    ]
    assert payloads[2]["item"]["type"] == "image_generation_call"
    completed_response = next(item for item in payloads if item["type"] == "response.completed")["response"]
    assert completed_response["output"][0]["type"] == "image_generation_call"
    assert completed_response["output"][0]["status"] == "completed"
    assert completed_response["output"][0]["result"] == "AAAA"


def test_image_generation_intent_is_conservative():
    assert is_image_generation_intent("生成一个苹果的图像") is True
    assert is_image_generation_intent("请画一张赛博朋克城市图片") is True
    assert is_image_generation_intent("generate an image of an apple") is True
    assert is_image_generation_intent("这张图片为什么打不开") is False
    assert is_image_generation_intent("请解释 image_generation 工具") is False


def test_responses_model_driven_bridge_generates_after_model_tool_call(image_app_db, monkeypatch):
    calls = []

    async def fake_planner(*args, **kwargs):
        return InternalOutputMessage(tool_calls=[InternalToolCallOutput(
            id="call_image", call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME,
            arguments=json.dumps({"prompt": "生成一个苹果的图像"}, ensure_ascii=False),
        )], finish_reason="tool_calls", usage={"total_tokens": 7}), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        calls.append(kwargs["prompt"])
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    client = TestClient(app)
    response = client.post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": "生成一个苹果的图像",
        "size": "1024x1024",
    })
    assert response.status_code == 200
    item = response.json()["output"][0]
    assert item["type"] == "message"
    assert item["status"] == "completed"
    text = item["content"][0]["text"]
    assert "![Generated image](data:image/png;base64,AAAA)" in text
    assert "[Open generated image](http://testserver/v1/image-results/" in text
    assert len(list(image_app_db["image_dir"].glob("*.png"))) == 1
    assert calls == ["生成一个苹果的图像"]


def test_responses_api_key_codex_streams_image_through_generated_image_exec(image_app_db, monkeypatch):
    async def fake_planner(*args, **kwargs):
        return InternalOutputMessage(tool_calls=[InternalToolCallOutput(
            id="call_image", call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME,
            arguments='{"prompt":"paint a red apple"}',
        )], finish_reason="tool_calls", usage={"total_tokens": 7}), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    client = TestClient(app)
    response = client.post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "stream": True,
        "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Generate an image of an apple"}]},
            {"type": "additional_tools", "tools": [{
                "type": "custom", "name": "exec",
                "description": "Run JavaScript. generatedImage(result) appends an image-generation result.",
                "format": {"type": "text"},
            }]},
        ],
    })
    assert response.status_code == 200
    assert '"type": "custom_tool_call"' in response.text
    assert '"name": "exec"' in response.text
    assert "generatedImage({ image_url:" in response.text
    assert "data:image/png;base64,AAAA" in response.text
    assert '"type": "image_generation_call"' not in response.text
    assert "image-results" not in response.text
    assert len(list(image_app_db["image_dir"].glob("*.png"))) == 1
    log = list_request_logs(limit=1)[0]
    assert log["request_kind"] == "image_generation"
    assert log["image_bytes"] == 3
    assert get_global_stats()["image_generation_calls"] == 1


def test_generated_image_exec_followup_completes_without_regenerating(image_app_db, monkeypatch):
    async def fail_generate(*args, **kwargs):
        raise AssertionError("display tool follow-up must not generate another image")

    monkeypatch.setattr("app.router.proxy.generate_images", fail_generate)
    call_id = GATEWAY_IMAGE_DISPLAY_CALL_PREFIX + "abc"
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "stream": True,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Generate an image"}]},
            {"type": "custom_tool_call", "call_id": call_id, "name": "exec", "input": "generatedImage(...);"},
            {"type": "custom_tool_call_output", "call_id": call_id, "output": [{
                "type": "input_image", "image_url": "data:image/png;base64,AAAA",
            }]},
        ],
    })
    assert response.status_code == 200
    assert '"type": "response.completed"' in response.text
    assert '"output": []' in response.text


def test_responses_codex_intent_bridge_excludes_large_context(image_app_db, monkeypatch):
    huge_context = "system/tool definition " * 10000
    input_data = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": huge_context}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "生成一只红色苹果"}]},
    ]
    assert latest_user_text(input_data) == "生成一只红色苹果"
    from app.router.proxy import _responses_image_prompt
    assert _responses_image_prompt(input_data, huge_context) == "生成一只红色苹果"


def test_responses_normal_chat_does_not_trigger_codex_bridge(image_app_db, monkeypatch):
    async def fake_planner(*args, **kwargs):
        return InternalOutputMessage(text="普通文本回答", usage={"total_tokens": 3}), {"id": "chat"}, "chat"

    async def fail_generate(*args, **kwargs):
        raise AssertionError("normal chat must not call image backend")

    monkeypatch.setattr("app.router.proxy.generate_images", fail_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": "请解释一下图片生成模型和视觉模型的区别",
    })
    assert response.status_code != 500


def test_responses_explicit_image_words_require_model_tool_call(image_app_db, monkeypatch):
    async def fake_planner(*args, **kwargs):
        return InternalOutputMessage(
            text="I will continue the task without generating an image.",
            usage={"total_tokens": 3},
        ), {"id": "chat"}, "chat"

    async def fail_generate(*args, **kwargs):
        raise AssertionError("natural-language intent must not directly execute image generation")

    monkeypatch.setattr("app.router.proxy.generate_images", fail_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": "Build the UI and create an image upload component",
    })
    assert response.status_code == 200
    assert response.json()["output"][0]["type"] == "message"
    assert list_request_logs(limit=1)[0]["request_kind"] == "text_generation"


def test_responses_image_tool_declaration_alone_does_not_generate(image_app_db, monkeypatch):
    calls = []

    async def fail_generate(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("a capability declaration is not an invocation")

    async def fake_planner(*args, **kwargs):
        return InternalOutputMessage(text="苹果公司历史", usage={"total_tokens": 4}), {"id": "chat"}, "chat"

    monkeypatch.setattr("app.router.proxy.generate_images", fail_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "请介绍一下苹果公司的历史"}]},
            {"type": "additional_tools", "tools": [{"type": "image_generation"}]},
        ],
    })
    assert calls == []


def test_image_generation_admin_menu_api(image_app_db):
    admin = {"Authorization": f"Bearer {create_session('admin')}"}
    response = TestClient(app).get("/admin/image-generation", headers=admin)
    assert response.status_code == 200
    model = response.json()["models"][0]
    assert model["image_generation"] is True
    assert model["model_id"] == "chat-model"
    assert model["provider_model"] == "chat/chat-model"


def test_image_generation_admin_menu_masks_external_api_key(image_app_db):
    upsert_image_generator("default", {
        "backend_type": "external_model",
        "api_base": "https://images.example/v1",
        "model": "image-model",
        "api_key": "secret-image-key",
    })
    admin = {"Authorization": f"Bearer {create_session('admin')}"}
    payload = TestClient(app).get("/admin/image-generation", headers=admin).json()
    assert payload["generators"]["default"]["api_key"] == ""
    assert payload["generators"]["default"]["has_api_key"] is True


def test_image_generation_toggle_accepts_stale_duplicate_provider_prefix(image_app_db):
    admin = {"Authorization": f"Bearer {create_session('admin')}"}
    response = TestClient(app).put(
        "/admin/models/image-generation", headers=admin,
        json={"model_id": "chat/chat/chat-model", "enabled": False},
    )
    assert response.status_code == 200
    assert get_model_image_generation("chat", "chat-model") is False


def test_image_generation_admin_normalizes_legacy_prefixed_model_id(image_app_db):
    from app.database import get_db
    with get_db() as db:
        db.execute(
            "UPDATE provider_models SET model_id = ? WHERE provider_id = ? AND model_id = ?",
            ("chat/chat-model", "chat", "chat-model"),
        )
    admin = {"Authorization": f"Bearer {create_session('admin')}"}
    client = TestClient(app)
    model = client.get("/admin/image-generation", headers=admin).json()["models"][0]
    assert model["model_id"] == "chat-model"
    assert model["provider_model"] == "chat/chat-model"
    response = client.put(
        "/admin/models/image-generation", headers=admin,
        json={"model_id": model["provider_model"], "enabled": False},
    )
    assert response.status_code == 200


def test_image_generation_backend_validation(image_app_db):
    admin = {"Authorization": f"Bearer {create_session('admin')}"}
    response = TestClient(app).put("/admin/image-generation/default", headers=admin, json={
        "backend_type": "existing_model", "provider_model": "missing/nope",
    })
    assert response.status_code == 400
    response = TestClient(app).put("/admin/image-generation/default", headers=admin, json={
        "backend_type": "comfyui", "api_base": "http://comfy.test", "enabled": True,
    })
    assert response.status_code == 400
    assert "not implemented" in response.json()["detail"]


def test_responses_image_generation_uses_configured_backend(image_app_db, monkeypatch):
    async def fake_planner(*args, **kwargs):
        return InternalOutputMessage(tool_calls=[InternalToolCallOutput(
            id="call_image", call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME,
            arguments='{"prompt":"draw an apple"}',
        )], finish_reason="tool_calls"), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        assert config["model"] == "image-model"
        assert kwargs["prompt"] == "draw an apple"
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": "draw an apple",
        "tools": [{"type": "image_generation"}],
    })
    assert response.status_code == 200
    item = response.json()["output"][0]
    assert item["type"] == "message"
    assert "![Generated image](data:image/png;base64,AAAA)" in item["content"][0]["text"]
    assert "[Open generated image](http://testserver/v1/image-results/" in item["content"][0]["text"]
    assert len(list(image_app_db["image_dir"].glob("*.png"))) == 1


def test_codex_bridge_injects_hosted_tool_choice_and_instructions_once():
    body = {"model": "chat-model", "input": "hello", "tools": [{"type": "web_search"}]}
    assert inject_hosted_image_capability(body) is True
    assert body["tools"] == [
        {"type": "web_search"},
        {"type": "image_generation", "output_format": "png"},
    ]
    assert body["tool_choice"] == "auto"
    assert IMAGE_BRIDGE_MARKER in body["instructions"]
    assert inject_hosted_image_capability(body) is False
    assert sum(tool.get("type") == "image_generation" for tool in body["tools"]) == 1


def test_codex_bridge_preserves_client_image_gen_function_tool():
    body = {"tools": [{"type": "function", "name": "image_gen.imagegen", "parameters": {}}]}
    assert has_codex_image_function_tool(body) is True
    assert inject_hosted_image_capability(body) is False
    assert body["tools"] == [{"type": "function", "name": "image_gen.imagegen", "parameters": {}}]


def test_codex_exec_image_call_is_consumed_by_gateway():
    arguments = json.dumps({
        "input": 'const r = await tools.llm_aio_image_generation({"prompt":"draw an apple","size":"1024x1024"}); text(r);',
    })
    assert image_call_arguments_from_exec(arguments) == {
        "prompt": "draw an apple", "size": "1024x1024",
    }


def test_image_result_storage_rejects_invalid_tokens_and_serves_exact_bytes(tmp_path):
    stored = store_image_results(
        [ImageGenerationResult("data:image/png;base64," + base64.b64encode(b"png-bytes").decode())],
        directory=tmp_path,
    )
    found = find_image_result(stored[0].token, directory=tmp_path)
    assert found is not None
    assert found.path.read_bytes() == b"png-bytes"
    assert find_image_result("../" + stored[0].token, directory=tmp_path) is None


def test_codex_image_namespace_round_trips_as_client_tool_call():
    body = {
        "model": "chat/chat-model",
        "input": "Generate an image",
        "tools": [{
            "type": "namespace",
            "name": "image_gen",
            "description": "Image tools",
            "tools": [{
                "type": "function",
                "name": "imagegen",
                "description": "Generate an image",
                "parameters": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                },
            }],
        }],
    }
    assert has_codex_image_function_tool(body) is True
    internal = responses_to_internal(body)
    assert [tool.name for tool in internal.tools] == ["image_gen-imagegen"]

    rendered = render_response(InternalOutputMessage(tool_calls=[InternalToolCallOutput(
        id="fc_image", call_id="call_image", name="image_gen-imagegen",
        arguments='{"prompt":"paint a blue whale"}',
    )]), model="chat/chat-model", extra=internal.extra)
    item = rendered["output"][0]
    assert item["type"] == "function_call"
    assert item["namespace"] == "image_gen"
    assert item["name"] == "imagegen"


def test_codex_lite_additional_tools_namespace_is_client_owned():
    body = {
        "input": [
            {"type": "additional_tools", "tools": [{"type": "namespace", "name": "image_gen"}]},
        ],
    }
    assert has_codex_image_function_tool(body) is True


def test_codex_generated_image_exec_capability_requires_explicit_helper():
    body = {"input": [{"type": "additional_tools", "tools": [{
        "type": "custom", "name": "exec",
        "description": "generatedImage(result) appends an image-generation result.",
    }]}]}
    assert has_codex_generated_image_exec_tool(body) is True
    body["input"][0]["tools"][0]["description"] = "Run JavaScript"
    assert has_codex_generated_image_exec_tool(body) is False


def test_flattened_codex_image_namespace_is_client_owned():
    body = {
        "input": [
            {"type": "additional_tools", "tools": [{
                "type": "function", "namespace": "image_gen", "name": "imagegen",
            }]},
        ],
    }
    assert has_codex_image_function_tool(body) is True


def test_codex_client_owned_image_tool_returns_function_call_without_gateway_bridge(image_app_db, monkeypatch):
    async def fake_planner(*args, **kwargs):
        return InternalOutputMessage(tool_calls=[InternalToolCallOutput(
            id="fc_image", call_id="call_image", name="image_gen-imagegen",
            arguments='{"prompt":"draw an apple"}',
        )], finish_reason="tool_calls", usage={"total_tokens": 4}), {"id": "chat"}, "chat"

    async def fail_generate(*args, **kwargs):
        raise AssertionError("client-owned Codex tool must call /images/generations after this response")

    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    monkeypatch.setattr("app.router.proxy.generate_images", fail_generate)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": "Generate an image of an apple",
        "tools": [{
            "type": "namespace", "name": "image_gen", "tools": [{
                "type": "function", "name": "imagegen",
                "parameters": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"],
                },
            }],
        }],
    })
    assert response.status_code == 200
    item = response.json()["output"][0]
    assert item["type"] == "function_call"
    assert item["namespace"] == "image_gen"
    assert item["name"] == "imagegen"


def test_images_generation_uses_configured_backend(image_app_db, monkeypatch):
    async def fake_generate(config, **kwargs):
        return [ImageGenerationResult("data:image/jpeg;base64,AAAA", mime_type="image/jpeg")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    response = TestClient(app).post("/v1/images/generations", headers=image_app_db["headers"], json={
        "model": "chat/chat-model", "prompt": "draw an apple",
    })
    assert response.status_code == 200
    assert response.json()["data"][0]["mime_type"] == "image/jpeg"
    log = list_request_logs(limit=1)[0]
    assert log["request_kind"] == "image_generation"
    assert log["image_model"] == "image-model"
    assert log["image_count"] == 1
    stats = get_global_stats()
    assert stats["image_generation_calls"] == 1
    assert stats["image_generation_images"] == 1


def test_codex_images_generation_accepts_fixed_image_model(image_app_db, monkeypatch):
    captured = {}

    async def fake_generate(config, **kwargs):
        captured.update(kwargs)
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    response = TestClient(app).post("/v1/images/generations", headers=image_app_db["headers"], json={
        "model": "gpt-image-2",
        "prompt": "paint a blue whale",
        "background": "auto",
        "quality": "auto",
        "size": "auto",
    })
    assert response.status_code == 200
    assert captured["model"] == "image-model"
    assert captured["size"] == "auto"


def test_grok_image_options_ignore_codex_auto_size():
    from app.adapters.imagegen import _grok_image_options

    assert _grok_image_options("auto") == {}
