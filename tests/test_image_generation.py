import base64
import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from main import app
from app.database import add_admin, add_provider, add_user, add_user_api_key
from app.security import create_session, hash_password

from app.adapters.imagegen import (
    ImageGenerationResult, _download_image, _is_public_address, generate_images,
    image_results_bytes, images_url,
)
from app.adapters.comfyui import (
    analyze_workflow, convert_ui_workflow, generate_comfyui_images, list_saved_workflows,
    load_saved_workflow, validate_mapping, workflow_userdata_url,
)
from app.database import (
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
    GATEWAY_IMAGE_ASSET_MARKER,
    GATEWAY_IMAGE_DISPLAY_CALL_PREFIX,
    GATEWAY_IMAGE_RESULT_MARKER,
    IMAGE_BRIDGE_CORRECTION_MARKER,
    IMAGE_BRIDGE_MARKER,
    IMAGE_BRIDGE_TOOL_NAME,
    configure_internal_image_bridge,
    has_codex_generated_image_exec_tool,
    has_codex_image_function_tool,
    gateway_generated_image_asset_context,
    has_gateway_generated_image_history,
    image_call_arguments_from_exec,
    image_call_arguments_list_from_exec,
    inject_hosted_image_capability,
    sanitize_gateway_image_display_followup,
    sanitize_gateway_generated_image_history,
)
from app.core.image_results import (
    find_image_result, image_preview_data_uri, remove_stored_image_results, store_image_results,
)
from app.core.output import InternalOutputEvent, InternalOutputMessage, InternalToolCallOutput
from app.core.image_batch import ImageInvocationCache


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


def test_image_download_address_policy_rejects_private_networks():
    assert _is_public_address("8.8.8.8") is True
    assert _is_public_address("127.0.0.1") is False
    assert _is_public_address("10.0.0.1") is False
    assert _is_public_address("169.254.169.254") is False
    assert _is_public_address("::1") is False


def _mock_dns(monkeypatch, address):
    monkeypatch.setattr(
        "app.adapters.imagegen.socket.getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", (address, 443))],
    )


@pytest.mark.asyncio
async def test_image_download_accepts_public_host(monkeypatch):
    _mock_dns(monkeypatch, "8.8.8.8")
    png = b"\x89PNG\r\n\x1a\nimage-data"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=png, headers={"content-type": "image/png"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        data_uri, mime_type = await _download_image(client, "https://images.example/result.png")
    assert mime_type == "image/png"
    assert base64.b64decode(data_uri.split(",", 1)[1]) == png


@pytest.mark.asyncio
async def test_image_download_rejects_private_dns_result(monkeypatch):
    _mock_dns(monkeypatch, "10.0.0.8")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None)) as client:
        with pytest.raises(ValueError, match="private or unsafe"):
            await _download_image(client, "https://images.example/result.png")


@pytest.mark.asyncio
async def test_image_download_rejects_url_credentials():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None)) as client:
        with pytest.raises(ValueError, match="must not contain credentials"):
            await _download_image(client, "https://user:password@images.example/result.png")


@pytest.mark.asyncio
async def test_image_download_rejects_declared_length_over_limit(monkeypatch):
    _mock_dns(monkeypatch, "8.8.8.8")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-length": "70000", "content-type": "image/png"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError, match="configured size limit"):
            await _download_image(client, "https://images.example/result.png", max_bytes=65536)


@pytest.mark.asyncio
async def test_image_download_rejects_stream_over_limit(monkeypatch):
    _mock_dns(monkeypatch, "8.8.8.8")
    payload = b"\x89PNG\r\n\x1a\n" + (b"x" * 65536)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=payload, headers={"content-type": "image/png"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError, match="configured size limit"):
            await _download_image(client, "https://images.example/result.png", max_bytes=65536)


@pytest.mark.asyncio
async def test_image_download_private_host_requires_explicit_override(monkeypatch):
    _mock_dns(monkeypatch, "127.0.0.1")
    png = b"\x89PNG\r\n\x1a\nimage-data"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=png, headers={"content-type": "image/png"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        data_uri, mime_type = await _download_image(
            client,
            "http://127.0.0.1/result.png",
            allow_private_hosts=True,
        )
    assert mime_type == "image/png"
    assert base64.b64decode(data_uri.split(",", 1)[1]) == png


def test_image_invocation_cache_rejects_new_key_when_all_entries_are_active():
    cache = ImageInvocationCache()
    first = cache.claim("first", ttl_seconds=60, max_entries=1)
    assert first.owner is True
    with pytest.raises(RuntimeError, match="too many concurrent"):
        cache.claim("second", ttl_seconds=60, max_entries=1)
    duplicate = cache.claim("first", ttl_seconds=60, max_entries=1)
    assert duplicate.owner is False
    assert duplicate.future is first.future


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


def test_image_preview_reduces_dimensions_until_byte_limit(monkeypatch):
    from PIL import Image
    from io import BytesIO

    source = BytesIO()
    Image.effect_noise((2048, 2048), 100).convert("RGB").save(source, format="PNG")
    result = ImageGenerationResult(
        "data:image/png;base64," + base64.b64encode(source.getvalue()).decode()
    )
    monkeypatch.setattr("app.core.image_results.get_default", lambda key, fallback=None: {
        "image_preview_enabled": True,
        "image_preview_max_dimension": 2048,
        "image_preview_quality": 82,
        "image_preview_max_bytes": 70000,
    }.get(key, fallback))

    preview = image_preview_data_uri(result)
    preview_bytes = base64.b64decode(preview.split(",", 1)[1])
    assert preview.startswith("data:image/jpeg;base64,")
    assert len(preview_bytes) <= 70000
    with Image.open(BytesIO(preview_bytes)) as image:
        assert max(image.size) < 2048


def test_store_image_results_rolls_back_partial_write(tmp_path, monkeypatch):
    from pathlib import Path

    original_write_bytes = Path.write_bytes
    writes = 0

    def fail_second_write(path, data):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("disk full")
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_second_write)
    results = [
        ImageGenerationResult("data:image/png;base64," + base64.b64encode(value).decode())
        for value in (b"first", b"second")
    ]
    with pytest.raises(OSError, match="disk full"):
        store_image_results(results, directory=tmp_path)
    assert list(tmp_path.iterdir()) == []


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


@pytest.mark.anyio
async def test_generate_images_retries_transient_504(monkeypatch):
    calls = 0

    class MockResponse:
        headers = {}

        def __init__(self, status_code, text="", body=None):
            self.status_code = status_code
            self.text = text
            self._body = body or {}

        def json(self):
            return self._body

    class MockClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False

        async def post(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return MockResponse(504, '{"error":"timeout"}')
            return MockResponse(200, body={
                "data": [{"b64_json": base64.b64encode(b"image").decode()}],
            })

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: MockClient())
    monkeypatch.setattr("app.adapters.imagegen.asyncio.sleep", no_sleep)
    result = await generate_images(
        {"api_base": "http://image.test/v1", "model": "image-model", "max_retries": 1},
        prompt="an apple",
    )
    assert calls == 2
    assert result[0].backend_attempts == 2


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
    assert is_image_generation_intent("Build the UI and create an image upload component") is False
    assert is_image_generation_intent("不要生成图像，只解释接口") is False
    assert is_image_generation_intent("请不要调用任何图像生成") is False
    assert is_image_generation_intent("禁止使用生图工具，只检查代码") is False
    assert is_image_generation_intent("制作一个流程图并写进文档") is False
    assert is_image_generation_intent("生成一张图") is True


def test_image_generation_intent_ignores_tool_outputs_and_instructions():
    input_data = [
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "review 代码并补充文档"},
        ]},
        {"type": "function_call_output", "call_id": "call_1",
         "output": "日志包含：生成一张诊断图片"},
        {"type": "custom_tool_call_output", "call_id": "call_2",
         "output": "image generation is available"},
    ]
    assert latest_user_text(input_data) == "review 代码并补充文档"
    assert is_image_generation_intent(input_data) is False
    assert is_image_generation_intent(
        input_data, instructions="When useful, generate an image with the image tool."
    ) is False


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
    assert GATEWAY_IMAGE_ASSET_MARKER not in text
    assert GATEWAY_IMAGE_RESULT_MARKER not in text
    assert "download these URLs into the project workspace" not in text
    assert "[`generated-asset-1.png` — download original](http://testserver/v1/image-results/" in text
    assert text.index("Original:") < text.index("data:image/png;base64")
    assert "![Generated image](data:image/png;base64,AAAA)" in text
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
    assert "image-results" in response.text
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    completed = next(item for item in payloads if item["type"] == "response.completed")
    messages = [item for item in completed["response"]["output"] if item["type"] == "message"]
    assert all(GATEWAY_IMAGE_ASSET_MARKER not in json.dumps(item) for item in messages)
    assert len(list(image_app_db["image_dir"].glob("*.png"))) == 1
    log = list_request_logs(limit=1)[0]
    assert log["request_kind"] == "image_generation"
    assert log["image_bytes"] == 3
    assert get_global_stats()["image_generation_calls"] == 1


def test_responses_image_bridge_preserves_other_agent_tools(image_app_db, monkeypatch):
    async def fake_planner(*args, **kwargs):
        return InternalOutputMessage(tool_calls=[
            InternalToolCallOutput(
                id="call_image", call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME,
                arguments='{"prompt":"paint a gomoku board"}',
            ),
            InternalToolCallOutput(
                id="call_plan", call_id="call_plan", name="update_plan",
                arguments='{"plan":[{"step":"Build the game","status":"in_progress"}]}',
            ),
        ], finish_reason="tool_calls", usage={"total_tokens": 9}), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "stream": True,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Build a Gomoku game and generate its art"}]},
            {"type": "additional_tools", "tools": [
                {
                    "type": "custom", "name": "exec",
                    "description": "Run JavaScript. generatedImage(result) appends an image-generation result.",
                    "format": {"type": "text"},
                },
                {
                    "type": "custom", "name": "update_plan",
                    "description": "Update the task plan.", "format": {"type": "text"},
                },
            ]},
        ],
    })
    assert response.status_code == 200
    assert response.text.count('"type": "custom_tool_call"') >= 2
    assert '"name": "exec"' in response.text
    assert '"name": "update_plan"' in response.text
    assert "generatedImage({ image_url:" in response.text
    assert IMAGE_BRIDGE_TOOL_NAME not in response.text


def test_responses_multiple_image_calls_continue_to_agent_tools(image_app_db, monkeypatch):
    planner_calls = 0
    generated_prompts = []

    async def fake_planner(policy, internal, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            return InternalOutputMessage(tool_calls=[
                InternalToolCallOutput(
                    id=f"call_image_{index}", call_id=f"call_image_{index}",
                    name=IMAGE_BRIDGE_TOOL_NAME,
                    arguments=json.dumps({
                        "prompt": prompt,
                        "filename": f"{prompt.replace(' ', '-')}.png",
                    }),
                )
                for index, prompt in enumerate(("board", "black stone", "white stone"), start=1)
            ], finish_reason="tool_calls", usage={"total_tokens": 9}), {"id": "chat"}, "chat"
        assert any(tool.name == IMAGE_BRIDGE_TOOL_NAME for tool in internal.tools)
        result_text = "\n".join(
            nested.text
            for message in internal.messages
            for part in message.parts
            if part.kind == "tool_result"
            for nested in part.parts
            if nested.kind == "text"
        )
        assert "download original" in result_text
        assert "http://testserver/v1/image-results/" in result_text
        assert "`board.png`" in result_text
        assert "data:image/" not in result_text
        return InternalOutputMessage(tool_calls=[InternalToolCallOutput(
            id="call_build", call_id="call_build", name="update_plan",
            arguments='{"plan":[{"step":"Build game files","status":"in_progress"}]}',
        )], finish_reason="tool_calls", usage={"total_tokens": 4}), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        generated_prompts.append(kwargs["prompt"])
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": "Build a Gomoku game and generate its visual assets first",
    })
    assert response.status_code == 200, response.text
    assert planner_calls == 2
    assert generated_prompts == ["board", "black stone", "white stone"]
    output = response.json()["output"]
    assert output[0]["type"] == "message"
    assert output[0]["content"][0]["text"].count("![Generated image") == 3
    assert "`board.png`" in output[0]["content"][0]["text"]
    assert "`black-stone.png`" in output[0]["content"][0]["text"]
    assert output[1]["type"] == "function_call"
    assert output[1]["name"] == "update_plan"


def test_responses_duplicate_asset_filenames_are_made_unique(image_app_db, monkeypatch):
    planner_calls = 0

    async def fake_planner(policy, internal, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            return InternalOutputMessage(tool_calls=[
                InternalToolCallOutput(
                    id=f"call_image_{index}", call_id=f"call_image_{index}",
                    name=IMAGE_BRIDGE_TOOL_NAME,
                    arguments=json.dumps({"prompt": prompt, "filename": "asset.png"}),
                )
                for index, prompt in enumerate(("first", "second"), start=1)
            ], finish_reason="tool_calls"), {"id": "chat"}, "chat"
        return InternalOutputMessage(text="continue"), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model", "input": "Generate two assets",
    })
    assert response.status_code == 200
    text = response.json()["output"][0]["content"][0]["text"]
    assert "`asset.png`" in text
    assert "`asset-2.png`" in text


def test_responses_multi_image_invocation_uses_distinct_fallback_names(
    image_app_db, monkeypatch
):
    planner_calls = 0

    async def fake_planner(policy, internal, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            return InternalOutputMessage(tool_calls=[InternalToolCallOutput(
                id="call_image", call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME,
                arguments=json.dumps({"prompt": "two variants", "filename": "ignored.png"}),
            )], finish_reason="tool_calls"), {"id": "chat"}, "chat"
        return InternalOutputMessage(text="continue"), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        return [
            ImageGenerationResult("data:image/png;base64,AAAA"),
            ImageGenerationResult("data:image/png;base64,BBBB"),
        ]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model", "input": "Generate two variants",
    })
    assert response.status_code == 200
    text = response.json()["output"][0]["content"][0]["text"]
    assert "`generated-asset-1.png`" in text
    assert "`generated-asset-2.png`" in text


def test_responses_multi_image_markdown_caps_inline_previews(image_app_db, monkeypatch):
    planner_calls = 0

    async def fake_planner(*args, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            return InternalOutputMessage(tool_calls=[
                InternalToolCallOutput(
                    id=f"call_image_{index}", call_id=f"call_image_{index}",
                    name=IMAGE_BRIDGE_TOOL_NAME,
                    arguments=json.dumps({"prompt": f"asset {index}"}),
                )
                for index in range(1, 6)
            ], finish_reason="tool_calls"), {"id": "chat"}, "chat"
        return InternalOutputMessage(text="continue"), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model", "input": "Generate five assets",
    })
    assert response.status_code == 200
    text = response.json()["output"][0]["content"][0]["text"]
    assert text.count("![Generated image") == 4
    assert "1 additional generated image preview(s) were omitted" in text
    assert text.count("download original]") == 5


def test_responses_failed_image_batch_preserves_success_and_retries_failed_item(image_app_db, monkeypatch):
    calls = 0

    async def fake_planner(*args, **kwargs):
        return InternalOutputMessage(tool_calls=[
            InternalToolCallOutput(
                id=f"call_image_{index}", call_id=f"call_image_{index}",
                name=IMAGE_BRIDGE_TOOL_NAME, arguments=json.dumps({"prompt": prompt}),
            )
            for index, prompt in enumerate(("first", "second"), start=1)
        ], finish_reason="tool_calls"), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second image failed")
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model", "input": "Generate two assets",
    })
    assert response.status_code == 200
    assert len(list(image_app_db["image_dir"].glob("*"))) == 2
    assert response.json()["output"][0]["content"][0]["text"].count("![Generated image") == 2
    log = list_request_logs(limit=1)[0]
    assert log["status"] in {"ok", "degraded"}
    assert log["image_count"] == 2


def test_responses_failed_continuation_returns_preserved_images_as_degraded(
    image_app_db, monkeypatch
):
    planner_calls = 0

    async def fake_planner(*args, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            return InternalOutputMessage(tool_calls=[InternalToolCallOutput(
                id="call_image", call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME,
                arguments='{"prompt":"board texture"}',
            )], finish_reason="tool_calls"), {"id": "chat"}, "chat"
        raise RuntimeError("continuation failed")

    async def fake_generate(config, **kwargs):
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model", "input": "Build a game with a generated board texture",
    })

    assert response.status_code == 200
    assert len(list(image_app_db["image_dir"].glob("*"))) == 1
    text = response.json()["output"][0]["content"][0]["text"]
    assert "![Generated image" in text
    assert "agent continuation failed" in text
    log = list_request_logs(limit=1)[0]
    assert log["request_kind"] == "image_generation"
    assert log["image_count"] == 1
    assert log["status"] == "degraded"


def test_responses_staged_image_batches_then_continue_to_code(image_app_db, monkeypatch):
    planner_calls = 0
    generated_prompts = []

    async def fake_planner(policy, internal, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            prompts = ("board background", "wood texture")
        elif planner_calls == 2:
            prompts = ("black stone", "white stone")
        else:
            return InternalOutputMessage(tool_calls=[InternalToolCallOutput(
                id="call_code", call_id="call_code", name="exec_command",
                arguments='{"cmd":"write index.html"}',
            )], finish_reason="tool_calls", usage={"total_tokens": 5}), {"id": "chat"}, "chat"
        return InternalOutputMessage(tool_calls=[
            InternalToolCallOutput(
                id=f"call_image_{planner_calls}_{index}",
                call_id=f"call_image_{planner_calls}_{index}",
                name=IMAGE_BRIDGE_TOOL_NAME,
                arguments=json.dumps({"prompt": prompt}),
            )
            for index, prompt in enumerate(prompts, start=1)
        ], finish_reason="tool_calls", usage={"total_tokens": 4}), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        generated_prompts.append(kwargs["prompt"])
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": "Build a Gomoku game with generated board and stone assets",
    })
    assert response.status_code == 200
    assert planner_calls == 3
    assert generated_prompts == ["board background", "wood texture", "black stone", "white stone"]
    output = response.json()["output"]
    assert output[0]["content"][0]["text"].count("![Generated image") == 4
    assert output[1]["type"] == "function_call"
    assert output[1]["name"] == "exec_command"
    assert response.json()["usage"]["total_tokens"] == 13


def test_codex_long_task_downloads_gateway_asset_then_continues(
    image_app_db, monkeypatch, tmp_path
):
    """Simulate the critical Codex asset handoff across two Responses turns."""
    planner_calls = 0
    image_calls = 0

    async def fake_planner(policy, internal, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            return InternalOutputMessage(tool_calls=[InternalToolCallOutput(
                id="call_image", call_id="call_image", name=IMAGE_BRIDGE_TOOL_NAME,
                arguments=json.dumps({
                    "prompt": "seamless modern wooden Gomoku board texture, no stones",
                    "filename": "board-texture.png",
                }),
            )], finish_reason="tool_calls", usage={"total_tokens": 8}), {"id": "chat"}, "chat"
        context = "\n".join(
            nested.text
            for message in internal.messages
            for part in message.parts
            for nested in ([part] if part.kind == "text" else part.parts)
            if nested.kind == "text"
        )
        assert "board-texture.png" in context
        assert "/v1/image-results/" in context
        assert "data:image/" not in context
        if planner_calls == 2:
            url = re.search(r"https?://[^)\s]+/v1/image-results/[0-9a-f]{32}", context).group(0)
            return InternalOutputMessage(tool_calls=[InternalToolCallOutput(
                id="call_download", call_id="call_download", name="exec_command",
                arguments=json.dumps({
                    "cmd": f"Invoke-WebRequest -Uri '{url}' -OutFile 'assets/board-texture.png'"
                }),
            )], finish_reason="tool_calls", usage={"total_tokens": 5}), {"id": "chat"}, "chat"
        assert any(tool.name == IMAGE_BRIDGE_TOOL_NAME for tool in internal.tools)
        return InternalOutputMessage(
            text="Game implementation completed using assets/board-texture.png.",
            finish_reason="stop", usage={"total_tokens": 4},
        ), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        nonlocal image_calls
        image_calls += 1
        return [ImageGenerationResult(
            "data:image/png;base64," + base64.b64encode(b"gateway-original-image").decode()
        )]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    client = TestClient(app)
    first = client.post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": "Build a complete Gomoku game and generate its board asset",
    })
    assert first.status_code == 200
    first_output = first.json()["output"]
    manifest_text = first_output[0]["content"][0]["text"]
    download_call = first_output[1]
    command = json.loads(download_call["arguments"])["cmd"]
    url = re.search(r"https?://[^']+/v1/image-results/[0-9a-f]{32}", command).group(0)

    # Execute the material part of Codex's proposed download against the same app.
    downloaded = client.get(url.replace("http://testserver", ""))
    assert downloaded.status_code == 200
    asset = tmp_path / "project" / "assets" / "board-texture.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(downloaded.content)
    assert asset.read_bytes() == b"gateway-original-image"

    second = client.post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": [
            {"type": "message", "role": "user", "content": [{
                "type": "input_text",
                "text": "Build a complete Gomoku game and generate its board asset",
            }]},
            {"type": "message", "role": "assistant", "content": [{
                "type": "output_text", "text": manifest_text,
            }]},
            {
                "type": "function_call", "call_id": "call_download",
                "name": "exec_command", "arguments": download_call["arguments"],
            },
            {
                "type": "function_call_output", "call_id": "call_download",
                "output": "Downloaded assets/board-texture.png and verified it exists.",
            },
        ],
    })
    assert second.status_code == 200
    assert "completed using assets/board-texture.png" in second.json()["output"][0]["content"][0]["text"]
    assert image_calls == 1
    assert planner_calls == 3


def test_generated_image_exec_followup_continues_without_regenerating(image_app_db, monkeypatch):
    async def fail_generate(*args, **kwargs):
        raise AssertionError("display tool follow-up must not generate another image")

    async def fake_planner(policy, internal, **kwargs):
        assert any(tool.name == IMAGE_BRIDGE_TOOL_NAME for tool in internal.tools)
        return InternalOutputMessage(
            text="The image is displayed; continuing with the game implementation.",
            usage={"total_tokens": 5},
        ), {"id": "chat"}, "chat"

    monkeypatch.setattr("app.router.proxy.generate_images", fail_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    call_id = GATEWAY_IMAGE_DISPLAY_CALL_PREFIX + "abc"
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "stream": False,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Generate an image"}]},
            {"type": "custom_tool_call", "call_id": call_id, "name": "exec", "input": "generatedImage({ image_url: 'data:image/png;base64,AAAA' });"},
            {"type": "custom_tool_call_output", "call_id": call_id, "output": [{
                "type": "input_image", "image_url": "data:image/png;base64,AAAA",
            }]},
        ],
    })
    assert response.status_code == 200
    item = response.json()["output"][0]
    assert item["type"] == "message"
    assert "continuing with the game implementation" in item["content"][0]["text"]


def test_gateway_image_display_followup_sanitizes_image_bytes():
    call_id = GATEWAY_IMAGE_DISPLAY_CALL_PREFIX + "abc"
    input_data = [
        {"type": "custom_tool_call", "call_id": call_id, "name": "exec", "input": "data:image/png;base64,AAAA"},
        {"type": "custom_tool_call_output", "call_id": call_id, "output": [{
            "type": "input_image", "image_url": "data:image/png;base64,BBBB",
        }]},
    ]
    assert sanitize_gateway_image_display_followup(input_data) is True
    assert "AAAA" not in input_data[0]["input"]
    assert input_data[1]["output"] == "The generated image was displayed successfully."


def test_gateway_generated_image_history_detection_is_assistant_scoped():
    assert has_gateway_generated_image_history([{
        "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": f"{GATEWAY_IMAGE_RESULT_MARKER}\n![Generated image](data:image/png;base64,AAAA)"}],
    }]) is True
    assert has_gateway_generated_image_history([{
        "type": "message", "role": "user",
        "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
    }]) is False
    assert has_gateway_generated_image_history([
        {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": f"{GATEWAY_IMAGE_RESULT_MARKER}\nold image"}],
        },
        {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "Generate a different image"}],
        },
    ]) is False
    assert has_gateway_generated_image_history([{
        "type": "message", "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": "See /image-results/example and ![Generated image](https://example.test/a.png)",
        }],
    }]) is False


def test_gateway_generated_image_history_compacts_only_gateway_assistant_previews():
    input_data = [
        {
            "type": "message", "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": (
                    f"{GATEWAY_IMAGE_ASSET_MARKER}\n"
                    "[download original](http://gateway.test/v1/image-results/abc)\n\n"
                    f"{GATEWAY_IMAGE_RESULT_MARKER}\n"
                    "![Generated image](data:image/png;base64,QUJDRA==)"
                ),
            }],
        },
        {
            "type": "message", "role": "user",
            "content": [{"type": "input_image", "image_url": "data:image/png;base64,VVNFUg=="}],
        },
    ]
    assert sanitize_gateway_generated_image_history(input_data) is True
    assistant_text = input_data[0]["content"][0]["text"]
    assert "QUJDRA==" not in assistant_text
    assert "http://gateway.test/v1/image-results/abc" in assistant_text
    assert input_data[1]["content"][0]["image_url"].endswith("VVNFUg==")


def test_marker_free_gateway_image_history_is_compacted_and_rehydrates_assets():
    input_data = [{
        "type": "message", "role": "assistant", "content": [{
            "type": "output_text",
            "text": (
                "Original: [test-apple.png](http://gateway.test/v1/image-results/abc)\n\n"
                "![Generated image](data:image/png;base64,QUJDRA==)"
            ),
        }],
    }]
    assert has_gateway_generated_image_history(input_data) is True
    context = gateway_generated_image_asset_context(input_data)
    assert GATEWAY_IMAGE_ASSET_MARKER in context
    assert "test-apple.png: http://gateway.test/v1/image-results/abc" in context
    assert sanitize_gateway_generated_image_history(input_data) is True
    assert "QUJDRA==" not in input_data[0]["content"][0]["text"]


def test_gateway_generated_image_asset_context_keeps_manifest_without_preview():
    input_data = [{
        "type": "message", "role": "assistant", "content": [{
            "type": "output_text",
            "text": (
                f"{GATEWAY_IMAGE_ASSET_MARKER}\nasset URL\n\n"
                f"{GATEWAY_IMAGE_RESULT_MARKER}\n"
                "![Generated image](data:image/png;base64,QUJDRA==)"
            ),
        }],
    }]
    context = gateway_generated_image_asset_context(input_data)
    assert "asset URL" in context
    assert GATEWAY_IMAGE_RESULT_MARKER not in context
    assert "QUJDRA==" not in context


def test_gateway_generated_image_asset_context_reads_hidden_exec_manifest():
    input_data = [{
        "type": "custom_tool_call",
        "call_id": f"{GATEWAY_IMAGE_DISPLAY_CALL_PREFIX}abc",
        "name": "exec",
        "input": (
            'generatedImage({ image_url: "data:image/png;base64,QUJDRA==" });\n'
            f"/*\n{GATEWAY_IMAGE_ASSET_MARKER}\n"
            "asset.png: http://gateway.test/v1/image-results/abc\n*/"
        ),
    }]
    context = gateway_generated_image_asset_context(input_data)
    assert GATEWAY_IMAGE_ASSET_MARKER in context
    assert "http://gateway.test/v1/image-results/abc" in context
    assert "QUJDRA==" not in context
    assert "*/" not in context


def test_prior_gateway_image_keeps_optional_bridge_on_later_tool_turn(image_app_db, monkeypatch):
    async def fake_planner(policy, internal, **kwargs):
        assert any(tool.name == IMAGE_BRIDGE_TOOL_NAME for tool in internal.tools)
        history_text = "\n".join(
            part.text
            for message in internal.messages
            for part in message.parts
            if part.kind == "text"
        )
        assert "http://testserver/v1/image-results/abc" in history_text
        assert "QUJDRA==" not in history_text
        return InternalOutputMessage(tool_calls=[InternalToolCallOutput(
            id="call_build", call_id="call_build", name="exec_command",
            arguments='{"cmd":"write game files"}',
        )], finish_reason="tool_calls", usage={"total_tokens": 4}), {"id": "chat"}, "chat"

    async def fail_generate(*args, **kwargs):
        raise AssertionError("an available bridge must not generate unless the model calls it")

    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    monkeypatch.setattr("app.router.proxy.generate_images", fail_generate)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Build a game and generate its assets"}]},
            {"type": "message", "role": "assistant", "content": [{
                "type": "output_text",
                "text": (
                    f"{GATEWAY_IMAGE_ASSET_MARKER}\n"
                    "[download original](http://testserver/v1/image-results/abc)\n\n"
                    f"{GATEWAY_IMAGE_RESULT_MARKER}\n"
                    "![Generated image](data:image/png;base64,QUJDRA==)"
                ),
            }]},
            {"type": "function_call", "call_id": "call_plan", "name": "update_plan", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_plan", "output": "ok"},
        ],
    })
    assert response.status_code == 200
    assert response.json()["output"][-1]["name"] == "exec_command"


def test_generated_image_exec_stream_followup_uses_compatibility_loop(image_app_db, monkeypatch):
    async def fail_native(*args, **kwargs):
        raise AssertionError("gateway display follow-up must not switch into native Responses")

    async def fake_planner(policy, internal, **kwargs):
        assert any(tool.name == IMAGE_BRIDGE_TOOL_NAME for tool in internal.tools)
        return InternalOutputMessage(
            text="Continuing the agent task.", finish_reason="stop",
            usage={"total_tokens": 3},
        ), {"id": "chat"}, "chat"

    monkeypatch.setattr("app.router.proxy._native_response_with_fallbacks", fail_native)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    call_id = GATEWAY_IMAGE_DISPLAY_CALL_PREFIX + "stream"
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "stream": True,
        "input": [
            {"type": "custom_tool_call", "call_id": call_id, "name": "exec", "input": "generatedImage(...);"},
            {"type": "custom_tool_call_output", "call_id": call_id, "output": "ok"},
        ],
    })
    assert response.status_code == 200
    assert "Continuing the agent task." in response.text
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    completed = next(item for item in payloads if item["type"] == "response.completed")
    assert completed["response"]["output"][0]["type"] == "message"


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


def test_responses_codex_exec_planner_is_not_forced_into_image_generation(image_app_db, monkeypatch):
    planner_calls = 0
    generated = []

    async def fake_planner(policy, internal, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            return InternalOutputMessage(
                tool_calls=[InternalToolCallOutput(
                    id="call_exec", call_id="call_exec", name="exec",
                    arguments=json.dumps({"input": "Get-ChildItem -Force"}),
                )], finish_reason="tool_calls", usage={"total_tokens": 3},
            ), {"id": "chat"}, "chat"
        assert any(
            IMAGE_BRIDGE_CORRECTION_MARKER in part.text
            for message in internal.messages
            for part in message.parts
            if part.kind == "text"
        )
        assert internal.tool_choice == {
            "type": "function",
            "function": {"name": IMAGE_BRIDGE_TOOL_NAME},
        }
        assert "tool_choice" in internal.extra["allowed_openai_params"]
        exec_tool = next(tool for tool in internal.tools if tool.name == "exec")
        assert "tools.llm_aio_image_generation" in exec_tool.description
        return InternalOutputMessage(
            tool_calls=[InternalToolCallOutput(
                id="call_image", call_id="call_image", name="exec",
                arguments=json.dumps({
                    "input": 'const r = await tools.llm_aio_image_generation({prompt:"board texture",filename:"board.png"}); text(r);',
                }),
            )], finish_reason="tool_calls", usage={"total_tokens": 4},
        ), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        generated.append(kwargs["prompt"])
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "stream": True,
        "input": [
            {"type": "message", "role": "user", "content": [{
                "type": "input_text", "text": "实现网页游戏，素材文件通过图像生成功能自动生成",
            }]},
            {"type": "additional_tools", "tools": [{
                "type": "custom", "name": "exec", "description": "Run commands and generatedImage(...).",
            }]},
        ],
    })
    assert response.status_code == 200, response.text
    assert planner_calls == 1
    assert generated == []
    assert IMAGE_BRIDGE_CORRECTION_MARKER not in response.text
    assert "Get-ChildItem -Force" in response.text


def test_display_followup_rejecting_gateway_for_local_key_is_not_forced_back_to_bridge(
    image_app_db, monkeypatch,
):
    planner_calls = 0
    generated = []

    async def fake_planner(policy, internal, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        assert any(tool.name == IMAGE_BRIDGE_TOOL_NAME for tool in internal.tools)
        if planner_calls == 1:
            assert internal.tool_choice is None
            return InternalOutputMessage(
                text="Please configure local OPENAI_API_KEY before I generate the remaining posters.",
                usage={"total_tokens": 3},
            ), {"id": "chat"}, "chat"
        assert internal.tool_choice == {
            "type": "function",
            "function": {"name": IMAGE_BRIDGE_TOOL_NAME},
        }
        assert "tool_choice" in internal.extra["allowed_openai_params"]
        return InternalOutputMessage(
            tool_calls=[InternalToolCallOutput(
                id="call_poster", call_id="call_poster", name=IMAGE_BRIDGE_TOOL_NAME,
                arguments=json.dumps({
                    "prompt": "original anime character poster on transparent background",
                    "filename": "character-poster-2.png",
                }),
            )],
            finish_reason="tool_calls",
            usage={"total_tokens": 4},
        ), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        generated.append(kwargs["prompt"])
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    call_id = GATEWAY_IMAGE_DISPLAY_CALL_PREFIX + "first"
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "stream": True,
        "input": [
            {"type": "message", "role": "user", "content": [{
                "type": "input_text",
                "text": "生成木质背景与 5 张原创角色海报，并完成 HTML 页面",
            }]},
            {"type": "additional_tools", "tools": [{
                "type": "custom", "name": "exec",
                "description": "Run JavaScript; generatedImage(result) displays generated images.",
            }]},
            {"type": "custom_tool_call", "call_id": call_id, "name": "exec",
             "input": "generatedImage({ image_url: 'data:image/png;base64,AAAA' });"},
            {"type": "custom_tool_call_output", "call_id": call_id,
             "output": "The generated image was displayed successfully."},
        ],
    })
    assert response.status_code == 200, response.text
    assert planner_calls == 1
    assert generated == []
    assert "OPENAI_API_KEY" in response.text
    assert IMAGE_BRIDGE_CORRECTION_MARKER not in response.text


def test_codex_exec_batch_generates_every_nested_image_call(image_app_db, monkeypatch):
    generated = []

    async def fake_planner(policy, internal, **kwargs):
        script = (
            "const r = await Promise.all(["
            "tools.llm_aio_image_generation({filename:'aria.png',prompt:'original Aria poster'}),"
            "tools.llm_aio_image_generation({filename:'ren.png',prompt:'original Ren poster'})"
            "]); text(JSON.stringify(r));"
        )
        return InternalOutputMessage(
            tool_calls=[InternalToolCallOutput(
                id="call_batch", call_id="call_batch", name="exec",
                arguments=json.dumps({"input": script}),
            )],
            finish_reason="tool_calls",
            usage={"total_tokens": 5},
        ), {"id": "chat"}, "chat"

    async def fake_generate(config, **kwargs):
        generated.append(kwargs["prompt"])
        encoded = base64.b64encode(kwargs["prompt"].encode()).decode()
        return [ImageGenerationResult(f"data:image/png;base64,{encoded}")]

    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "stream": True,
        "input": [
            {"type": "message", "role": "user", "content": [{
                "type": "input_text", "text": "Generate two original character posters",
            }]},
            {"type": "additional_tools", "tools": [{
                "type": "custom", "name": "exec",
                "description": "Run JavaScript; generatedImage(result) displays generated images.",
            }]},
        ],
    })
    assert response.status_code == 200, response.text
    assert generated == ["original Aria poster", "original Ren poster"]
    # Responses SSE repeats the completed tool input in several lifecycle
    # events; assert both distinct generated payloads are present instead of
    # counting serialized occurrences.
    assert base64.b64encode(b"original Aria poster").decode() in response.text
    assert base64.b64encode(b"original Ren poster").decode() in response.text
    logs = list_request_logs(limit=5)
    image_log = next(item for item in logs if item["details"].get("request_kind") == "image_generation")
    assert image_log["details"]["image_requested_count"] == 2
    assert image_log["details"]["image_succeeded_count"] == 2


def test_responses_codex_system_turn_does_not_inject_or_correct_image_bridge(image_app_db, monkeypatch):
    planner_calls = 0

    async def fake_planner(policy, internal, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        assert all(tool.name != IMAGE_BRIDGE_TOOL_NAME for tool in internal.tools)
        assert IMAGE_BRIDGE_MARKER not in internal.system
        return InternalOutputMessage(
            text='{"title":"实现五子棋","description":"自动生成素材"}',
            usage={"total_tokens": 3},
        ), {"id": "chat"}, "chat"

    async def fail_generate(*args, **kwargs):
        raise AssertionError("a Codex system turn must not invoke the image backend")

    monkeypatch.setattr("app.router.proxy.generate_images", fail_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": [{
            "type": "message", "role": "user", "content": [{
                "type": "input_text",
                "text": "Create a task title for: 生成一个游戏背景图像素材",
            }],
        }],
        "client_metadata": {
            "x-codex-turn-metadata": json.dumps({
                "request_kind": "turn", "thread_source": "system",
            }),
        },
        "text": {"format": {"type": "json_schema", "name": "codex_output_schema"}},
    })
    assert response.status_code == 200
    assert planner_calls == 1
    assert IMAGE_BRIDGE_CORRECTION_MARKER not in response.text


def test_responses_ordinary_exec_is_not_replanned_as_image_generation(image_app_db, monkeypatch):
    planner_calls = 0

    async def fake_planner(policy, internal, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            return InternalOutputMessage(
                text="I will inspect the workspace first.",
                tool_calls=[InternalToolCallOutput(
                    id="call_original", call_id="call_original", name="exec",
                    arguments=json.dumps({"input": "const r = await tools.shell_command({command:\"Get-ChildItem -Force\"}); text(r)"}),
                )],
                finish_reason="tool_calls", usage={"total_tokens": 3},
            ), {"id": "chat"}, "chat"
        return InternalOutputMessage(
            text="I will generate the requested asset.",
            tool_calls=[InternalToolCallOutput(
                id="call_empty", call_id="call_empty", name="exec", arguments="{}",
            )],
            finish_reason="tool_calls", usage={"total_tokens": 4},
        ), {"id": "chat"}, "chat"

    async def fail_generate(*args, **kwargs):
        raise AssertionError("an empty exec is not an image invocation")

    monkeypatch.setattr("app.router.proxy.generate_images", fail_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": "实现网页游戏，素材文件通过图像生成功能自动生成",
    })
    assert response.status_code == 200
    assert planner_calls == 1
    assert "call_original" in response.text
    assert "call_empty" not in response.text
    assert IMAGE_BRIDGE_CORRECTION_MARKER not in response.text


def test_responses_allows_required_imagegen_skill_read_before_correction(image_app_db, monkeypatch):
    planner_calls = 0

    async def fake_planner(*args, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        return InternalOutputMessage(
            tool_calls=[InternalToolCallOutput(
                id="call_skill", call_id="call_skill", name="exec",
                arguments=json.dumps({
                    "input": "const r = await tools.shell_command({command:"
                    "\"Get-Content C:/Users/NRC/.codex/skills/.system/imagegen/SKILL.md\"}); text(r);",
                }),
            )], finish_reason="tool_calls", usage={"total_tokens": 3},
        ), {"id": "chat"}, "chat"

    async def fail_generate(*args, **kwargs):
        raise AssertionError("skill discovery turn must reach Codex before image correction")

    monkeypatch.setattr("app.router.proxy.generate_images", fail_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": "生成一个游戏背景图像素材",
    })
    assert response.status_code == 200
    assert planner_calls == 1
    assert "SKILL.md" in response.text
    assert IMAGE_BRIDGE_CORRECTION_MARKER not in response.text


def test_responses_ordinary_exec_does_not_trigger_image_skill_correction(image_app_db, monkeypatch):
    planner_calls = 0

    async def fake_planner(*args, **kwargs):
        nonlocal planner_calls
        planner_calls += 1
        if planner_calls == 1:
            script = "const r = await tools.shell_command({command:\"Get-ChildItem -Force\"}); text(r)"
        else:
            script = (
                "const r = await tools.shell_command({command:"
                "\"Get-Content C:/Users/NRC/.codex/skills/.system/imagegen/SKILL.md\"}); text(r)"
            )
        return InternalOutputMessage(
            tool_calls=[InternalToolCallOutput(
                id=f"call_{planner_calls}", call_id=f"call_{planner_calls}", name="exec",
                arguments=json.dumps({"input": script}),
            )], finish_reason="tool_calls", usage={"total_tokens": 3},
        ), {"id": "chat"}, "chat"

    async def fail_generate(*args, **kwargs):
        raise AssertionError("skill read is not an image backend invocation")

    monkeypatch.setattr("app.router.proxy.generate_images", fail_generate)
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_planner)
    response = TestClient(app).post("/v1/responses", headers=image_app_db["headers"], json={
        "model": "chat/chat-model",
        "input": "生成一个游戏背景图像素材",
    })
    assert response.status_code == 200
    assert planner_calls == 1
    assert "Get-ChildItem" in response.text
    assert IMAGE_BRIDGE_CORRECTION_MARKER not in response.text


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
    payload = response.json()
    model = payload["models"][0]
    assert model["image_generation"] is True
    assert model["model_id"] == "chat-model"
    assert model["provider_model"] == "chat/chat-model"
    assert payload["providers"] == [{
        "id": "chat",
        "name": "Chat",
        "models": [model],
    }]


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


def test_image_generation_admin_does_not_reuse_key_across_backend_hosts(image_app_db):
    upsert_image_generator("default", {
        "backend_type": "external_model",
        "api_base": "https://images.example/v1",
        "model": "image-model",
        "api_key": "secret-image-key",
    })
    admin = {"Authorization": f"Bearer {create_session('admin')}"}
    workflow = _comfy_workflow()
    mapping = analyze_workflow(workflow)["suggestions"]
    response = TestClient(app).put("/admin/image-generation/default", headers=admin, json={
        "backend_type": "comfyui", "api_base": "http://comfy.test",
        "api_key": "", "workflow": workflow, "workflow_mapping": mapping,
        "enabled": True,
    })
    assert response.status_code == 200
    assert get_enabled_image_generator()["api_key"] == ""


def test_image_generation_admin_connection_test(image_app_db, monkeypatch):
    async def fake_generate(config, **kwargs):
        assert config["model"] == "image-model"
        assert kwargs["prompt"]
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.adapters.imagegen.generate_images", fake_generate)
    admin = {"Authorization": f"Bearer {create_session('admin')}"}
    response = TestClient(app).post(
        "/admin/image-generation/test",
        headers=admin,
        json={"generator_id": "default"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["generator_id"] == "default"
    assert response.json()["model"] == "image-model"
    assert response.json()["image_count"] == 1


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
        "workflow": {}, "workflow_mapping": {},
    })
    assert response.status_code == 400
    assert "workflow" in response.json()["detail"]


def _comfy_workflow():
    return {
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive", "clip": ["4", 1]}, "_meta": {"title": "Positive Prompt"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative", "clip": ["4", 1]}, "_meta": {"title": "Negative Prompt"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 7.0, "latent_image": ["5", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}},
    }


def _comfy_ui_workflow():
    return {
        "id": "demo",
        "nodes": [
            {"id": 4, "type": "CLIPTextEncode", "title": "CLIP Text Encode (Positive)", "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 19}, {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None}], "widgets_values": ["a cat"]},
            {"id": 5, "type": "CLIPTextEncode", "title": "CLIP Text Encode (Negative)", "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 20}, {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None}], "widgets_values": ["blurry"]},
            {"id": 6, "type": "EmptyLatentImage", "mode": 0, "inputs": [{"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None}, {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None}, {"name": "batch_size", "type": "INT", "widget": {"name": "batch_size"}, "link": None}], "widgets_values": [896, 1344, 1]},
            {"id": 7, "type": "KSampler", "mode": 0, "inputs": [{"name": "model", "type": "MODEL", "link": 22}, {"name": "positive", "type": "CONDITIONING", "link": 1}, {"name": "negative", "type": "CONDITIONING", "link": 2}, {"name": "latent_image", "type": "LATENT", "link": 3}, {"name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None}, {"name": "steps", "type": "INT", "widget": {"name": "steps"}, "link": None}, {"name": "cfg", "type": "FLOAT", "widget": {"name": "cfg"}, "link": None}, {"name": "sampler_name", "type": "COMBO", "widget": {"name": "sampler_name"}, "link": None}, {"name": "scheduler", "type": "COMBO", "widget": {"name": "scheduler"}, "link": None}, {"name": "denoise", "type": "FLOAT", "widget": {"name": "denoise"}, "link": None}], "widgets_values": [43593584383551, "randomize", 10, 1, "er_sde", "simple", 1]},
            {"id": 8, "type": "VAEDecode", "mode": 0, "inputs": [{"name": "samples", "type": "LATENT", "link": 4}, {"name": "vae", "type": "VAE", "link": 21}], "widgets_values": []},
            {"id": 9, "type": "SaveImage", "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 9}, {"name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None}], "widgets_values": ["output"]},
            {"id": 16, "type": "CheckpointLoaderSimple", "mode": 0, "inputs": [{"name": "ckpt_name", "type": "COMBO", "widget": {"name": "ckpt_name"}, "link": None}], "widgets_values": ["krea.safetensors"]},
        ],
        "links": [[1, 4, 0, 7, 1, "CONDITIONING"], [2, 5, 0, 7, 2, "CONDITIONING"], [3, 6, 0, 7, 3, "LATENT"], [4, 7, 0, 8, 0, "LATENT"], [9, 8, 0, 9, 0, "IMAGE"], [19, 16, 1, 4, 0, "CLIP"], [20, 16, 1, 5, 0, "CLIP"], [21, 16, 2, 8, 1, "VAE"], [22, 16, 0, 7, 0, "MODEL"]],
    }


def test_comfyui_workflow_analysis_suggests_dropdown_mappings():
    analysis = analyze_workflow(_comfy_workflow())
    assert analysis["node_count"] == 5
    assert analysis["suggestions"]["prompt"] == {"node_id": "6", "input": "text"}
    assert analysis["suggestions"]["negative_prompt"] == {"node_id": "7", "input": "text"}
    assert analysis["suggestions"]["width"] == {"node_id": "5", "input": "width"}
    assert analysis["suggestions"]["output_node_id"] == "9"


def test_comfyui_regular_ui_workflow_is_converted_and_analyzed():
    converted = convert_ui_workflow(_comfy_ui_workflow())
    assert converted["4"]["inputs"] == {"clip": ["16", 1], "text": "a cat"}
    assert converted["6"]["inputs"] == {"width": 896, "height": 1344, "batch_size": 1}
    assert converted["7"]["inputs"]["seed"] == 43593584383551
    assert converted["7"]["inputs"]["steps"] == 10
    assert converted["7"]["inputs"]["cfg"] == 1
    assert converted["7"]["inputs"]["sampler_name"] == "er_sde"
    analysis = analyze_workflow(_comfy_ui_workflow())
    assert analysis["converted"] is True
    assert analysis["source_format"] == "ui"
    assert analysis["suggestions"]["prompt"] == {"node_id": "4", "input": "text"}
    assert analysis["suggestions"]["negative_prompt"] == {"node_id": "5", "input": "text"}


def test_comfyui_mapping_rejects_unknown_node_inputs():
    with pytest.raises(ValueError, match="unknown node input"):
        validate_mapping(_comfy_workflow(), {"prompt": {"node_id": "404", "input": "text"}})


def test_comfyui_workflow_userdata_url_encodes_nested_path():
    assert workflow_userdata_url("http://comfy.test/", "folder/demo") == (
        "http://comfy.test/api/userdata/workflows%2Ffolder%2Fdemo.json"
    )


@pytest.mark.asyncio
async def test_comfyui_saved_workflow_discovery(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url, params=None):
            if params:
                assert params == {"dir": "workflows", "recurse": "true"}
                return httpx.Response(200, json=["krea2_demo.json", "folder/other.json", "notes.txt"], request=httpx.Request("GET", url))
            assert url.endswith("/api/userdata/workflows%2Fkrea2_demo.json")
            return httpx.Response(200, json={"nodes": []}, request=httpx.Request("GET", url))

    monkeypatch.setattr("app.adapters.comfyui.httpx.AsyncClient", FakeClient)
    assert await list_saved_workflows("http://comfy.test") == ["folder/other.json", "krea2_demo.json"]
    assert await load_saved_workflow("http://comfy.test", "krea2_demo") == {"nodes": []}


def test_comfyui_workflow_admin_analysis_and_save(image_app_db):
    admin = {"Authorization": f"Bearer {create_session('admin')}"}
    client = TestClient(app)
    analysis = client.post(
        "/admin/image-generation/comfyui/analyze-workflow", headers=admin,
        json={"workflow": _comfy_workflow()},
    )
    assert analysis.status_code == 200
    mapping = analysis.json()["suggestions"]
    response = client.put("/admin/image-generation/default", headers=admin, json={
        "backend_type": "comfyui", "api_base": "http://comfy.test",
        "workflow": _comfy_workflow(), "workflow_mapping": mapping,
        "timeout": 300, "poll_interval": 0.5, "enabled": True,
    })
    assert response.status_code == 200
    stored = get_enabled_image_generator()
    assert stored["backend_type"] == "comfyui"
    assert stored["workflow"]["6"]["class_type"] == "CLIPTextEncode"
    assert stored["workflow_mapping"]["prompt"] == {"node_id": "6", "input": "text"}
    assert stored["poll_interval"] == 0.5


@pytest.mark.asyncio
async def test_comfyui_adapter_submits_polls_and_downloads(monkeypatch):
    png = b"\x89PNG\r\n\x1a\n" + b"test-image"
    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json):
            calls.append(("post", url, json))
            return httpx.Response(200, json={"prompt_id": "job-1"})
        async def get(self, url, params=None):
            calls.append(("get", url, params))
            if url.endswith("/history/job-1"):
                return httpx.Response(200, json={"job-1": {"status": {"completed": True}, "outputs": {"9": {"images": [{"filename": "result.png", "subfolder": "", "type": "output"}]}}}})
            return httpx.Response(200, content=png, headers={"content-type": "image/png"})

    monkeypatch.setattr("app.adapters.comfyui.httpx.AsyncClient", FakeClient)
    mapping = analyze_workflow(_comfy_workflow())["suggestions"]
    results = await generate_comfyui_images({
        "api_base": "http://comfy.test", "workflow": _comfy_workflow(),
        "workflow_mapping": mapping, "timeout": 10,
    }, prompt="draw an apple", n=1, size="768x1024", extra={"steps": 30})
    submitted = calls[0][2]["prompt"]
    assert submitted["6"]["inputs"]["text"] == "draw an apple"
    assert submitted["5"]["inputs"]["width"] == 768
    assert submitted["5"]["inputs"]["height"] == 1024
    assert submitted["3"]["inputs"]["steps"] == 30
    assert results[0].mime_type == "image/png"
    assert results[0].data_uri.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_comfyui_adapter_repeats_jobs_when_workflow_has_no_batch_mapping(monkeypatch):
    workflow = _comfy_workflow()
    del workflow["5"]["inputs"]["batch_size"]
    mapping = analyze_workflow(workflow)["suggestions"]
    calls = {"post": 0}

    class FakeClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json):
            calls["post"] += 1
            return httpx.Response(200, json={"prompt_id": f"job-{calls['post']}"})
        async def get(self, url, params=None):
            if "/history/" in url:
                prompt_id = url.rsplit("/", 1)[-1]
                return httpx.Response(200, json={prompt_id: {"status": {"completed": True}, "outputs": {"9": {"images": [{"filename": f"{prompt_id}.png"}]}}}})
            return httpx.Response(200, content=b"\x89PNG\r\n\x1a\nimage", headers={"content-type": "image/png"})

    monkeypatch.setattr("app.adapters.comfyui.httpx.AsyncClient", FakeClient)
    results = await generate_comfyui_images({
        "api_base": "http://comfy.test", "workflow": workflow,
        "workflow_mapping": mapping, "timeout": 10,
    }, prompt="three images", n=3)
    assert calls["post"] == 3
    assert len(results) == 3


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
    assert "download original](http://testserver/v1/image-results/" in item["content"][0]["text"]
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


def test_codex_exec_image_call_accepts_luna_javascript_object_literal():
    arguments = json.dumps({
        "input": (
            'const r = await tools.llm_aio_image_generation({'
            'filename:"gomoku-board-texture.png",size:"1024x1024",quality:"high",'
            'prompt:"dark texture with literal {accent:blue} text"}); text(r);'
        ),
    })
    assert image_call_arguments_from_exec(arguments) == {
        "filename": "gomoku-board-texture.png",
        "size": "1024x1024",
        "quality": "high",
        "prompt": "dark texture with literal {accent:blue} text",
    }


def test_codex_exec_image_call_accepts_multiple_single_quoted_javascript_literals():
    arguments = json.dumps({
        "input": (
            "const r = await Promise.all(["
            "tools.llm_aio_image_generation({filename:'aria.png',prompt:'Aria with {gold} light'}),"
            "tools.llm_aio_image_generation({filename:'ren.png',size:'1024x1536',prompt:'Ren poster'})"
            "]); text(JSON.stringify(r));"
        ),
    })
    assert image_call_arguments_list_from_exec(arguments) == [
        {"filename": "aria.png", "prompt": "Aria with {gold} light"},
        {"filename": "ren.png", "size": "1024x1536", "prompt": "Ren poster"},
    ]
    assert image_call_arguments_from_exec(arguments) == {
        "filename": "aria.png", "prompt": "Aria with {gold} light",
    }


def test_codex_exec_description_advertises_gateway_virtual_image_tool():
    body = {
        "model": "chat/chat-model",
        "input": [{
            "type": "additional_tools",
            "tools": [{"type": "custom", "name": "exec", "description": "Run JavaScript."}],
        }],
    }
    internal = responses_to_internal(body)
    inject_hosted_image_capability(body)
    configure_internal_image_bridge(internal, body)
    exec_tool = next(tool for tool in internal.tools if tool.name == "exec")
    assert "tools.llm_aio_image_generation" in exec_tool.description
    assert "absent from ALL_TOOLS" in exec_tool.description
    assert "OPENAI_API_KEY" in internal.system
    assert "do not inspect" in internal.system.lower()
    assert "scripts/image_gen.py" in internal.system


def test_image_result_storage_rejects_invalid_tokens_and_serves_exact_bytes(tmp_path):
    stored = store_image_results(
        [ImageGenerationResult("data:image/png;base64," + base64.b64encode(b"png-bytes").decode())],
        directory=tmp_path,
    )
    found = find_image_result(stored[0].token, directory=tmp_path)
    assert found is not None
    assert found.path.read_bytes() == b"png-bytes"
    assert find_image_result("../" + stored[0].token, directory=tmp_path) is None
    remove_stored_image_results(stored)
    assert not stored[0].path.exists()


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
