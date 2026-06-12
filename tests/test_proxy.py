import pytest
from fastapi.testclient import TestClient
from main import app
from app.config import load_config
from app.database import init_db, add_provider, add_user, add_user_api_key, upsert_preprocessor

client = TestClient(app)
headers = {"Authorization": "Bearer user-key"}


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Use a temporary database and config for each test."""
    db_path = str(tmp_path / "test.db")
    config_path = str(tmp_path / "config.json")
    config = load_config(config_path, force_reload=True)
    config.config = {
        "host": "0.0.0.0",
        "port": 8000,
        "database": db_path,
        "logging": {"enabled": False, "level": "INFO", "log_dir": "logs", "retention_days": 30, "console": False}
    }
    config.save()
    init_db(db_path)

    # Seed test data
    add_provider({
        "id": "test-provider",
        "name": "Test Provider",
        "provider_type": "openai",
        "api_base": "https://api.test.com/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [
            {"id": "allowed-model", "name": "Allowed", "enabled": True},
            {"id": "blocked-model", "name": "Blocked", "enabled": True}
        ]
    })
    add_user({
        "username": "alice",
        "display_name": "Alice",
        "enabled": True
    })
    add_user_api_key("alice", "default", ["allowed-model"])
    # Override the key to a known value for tests
    from app.database import get_db
    with get_db() as db:
        db.execute("UPDATE user_api_keys SET key = 'user-key' WHERE username = 'alice'")

    yield config


def test_list_models_filters_by_user_key():
    response = client.get("/v1/models", headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert [item["id"] for item in data] == ["test-provider/allowed-model"]


def test_list_models_advertises_native_vision_without_preprocessor():
    from app.database import get_db
    add_provider({
        "id": "pixel-api",
        "name": "pixel-api",
        "provider_type": "anthropic",
        "api_base": "https://ai-pixel.online",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "gpt-5.5", "name": "gpt-5.5", "enabled": True}],
    })
    with get_db() as db:
        db.execute("UPDATE user_api_keys SET allowed_models = ? WHERE key = 'user-key'", ('["pixel-api/gpt-5.5"]',))

    response = client.get("/v1/models", headers=headers)

    assert response.status_code == 200
    model = response.json()["data"][0]
    assert model["id"] == "pixel-api/gpt-5.5"
    assert model["supports_vision"] is True
    assert model["image_support"] is True
    assert model["multimodal"] is True
    assert "preprocessor" not in model


def test_list_models_unauthorized():
    response = client.get("/v1/models")
    assert response.status_code == 401


def test_chat_completion_unauthorized():
    response = client.post("/v1/chat/completions", json={
        "model": "allowed-model",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response.status_code == 401


def test_chat_completion_forbidden_model():
    response = client.post("/v1/chat/completions", headers=headers, json={
        "model": "blocked-model",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response.status_code == 403


def test_chat_completion_requires_provider_prefix_when_only_composite_model_allowed():
    from app.database import get_db
    with get_db() as db:
        db.execute("UPDATE user_api_keys SET allowed_models = ? WHERE key = 'user-key'", ('["test-provider/allowed-model"]',))

    response = client.post("/v1/chat/completions", headers=headers, json={
        "model": "allowed-model",
        "messages": [{"role": "user", "content": "Hello"}]
    })

    assert response.status_code == 403
    assert "provider-qualified" in response.json()["detail"]


def test_chat_completion_accepts_request_without_previous_response_id(monkeypatch):
    from app.router import proxy

    class Usage:
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2

    class Message:
        content = "ok"
        tool_calls = None
        reasoning_content = None

    class Choice:
        message = Message()
        finish_reason = "stop"

    class Response:
        choices = [Choice()]
        usage = Usage()

    monkeypatch.setattr(proxy, "create_chat_completion", lambda **kwargs: Response())

    response = client.post("/v1/chat/completions", headers=headers, json={
        "model": "allowed-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    })

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"


def test_anthropic_messages_accepts_request_without_previous_response_id(monkeypatch):
    from app.adapters import openai_streaming

    def fake_stream(**kwargs):
        class Delta:
            content = "ok"
            reasoning_content = None
            tool_calls = None

        class Choice:
            delta = Delta()
            finish_reason = "stop"

        class Chunk:
            choices = [Choice()]
            usage = None

        yield Chunk()

    monkeypatch.setattr(openai_streaming, "create_chat_completion_stream", fake_stream)

    with client.stream("POST", "/v1/messages", headers=headers, json={
        "model": "allowed-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "max_tokens": 32,
    }) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert "message_start" in body
    assert "message_stop" in body


@pytest.mark.asyncio
async def test_openai_stream_tolerates_litellm_tail_chunk_builder_error_after_output(monkeypatch):
    from app.adapters import openai_streaming

    class Delta:
        content = "ok"
        reasoning_content = None
        tool_calls = None

    class Choice:
        delta = Delta()
        finish_reason = None

    class Chunk:
        choices = [Choice()]
        usage = None

    def fake_stream(**kwargs):
        yield Chunk()
        raise Exception("litellm.APIError: Error building chunks for logging/streaming usage calculation")

    monkeypatch.setattr(openai_streaming, "create_chat_completion_stream", fake_stream)

    events = []
    async for event in openai_streaming.iter_openai_chat_output_events(
        model="allowed-model",
        messages=[{"role": "user", "content": "Hello"}],
        provider_id="test-provider",
        temperature=0.7,
        max_tokens=32,
    ):
        events.append(event)

    assert [event.kind for event in events] == ["message_start", "text_delta", "message_done"]
    assert events[1].text == "ok"
    assert events[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_stream_propagates_litellm_tail_chunk_builder_error_before_output(monkeypatch):
    from app.adapters import openai_streaming

    def fake_stream(**kwargs):
        raise Exception("litellm.APIError: Error building chunks for logging/streaming usage calculation")
        yield

    monkeypatch.setattr(openai_streaming, "create_chat_completion_stream", fake_stream)

    events = openai_streaming.iter_openai_chat_output_events(
        model="allowed-model",
        messages=[{"role": "user", "content": "Hello"}],
        provider_id="test-provider",
        temperature=0.7,
        max_tokens=32,
    )

    with pytest.raises(Exception, match="Error building chunks"):
        async for _event in events:
            pass


def test_static_admin_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "LLM AIO Gateway" in response.text


# Test section

@pytest.fixture
def preprocess_db(tmp_path):
    """Test behavior."""
    db_path = str(tmp_path / "test_preprocess.db")
    config_path = str(tmp_path / "config_preprocess.json")
    config = load_config(config_path, force_reload=True)
    config.config = {
        "host": "0.0.0.0",
        "port": 8000,
        "database": db_path,
        "logging": {"enabled": False, "level": "INFO", "log_dir": "logs", "retention_days": 30, "console": False},
    }
    config.save()
    init_db(db_path)
    upsert_preprocessor("test-vision", {
        "api_base": "http://127.0.0.1:8080/v1",
        "model": "test-vision",
        "api_key": "k",
        "timeout": 30,
        "max_images": 20,
        "prompt": "Describe this image.",
        "enabled": True,
    })
    # Test section
    add_provider({
        "id": "native-provider",
        "name": "Native Provider",
        "provider_type": "openai",
        "api_base": "https://api.native.com/v1",
        "api_key": "k",
        "enabled": True,
        "models": [
            {"id": "native-model", "name": "Native Model", "enabled": True},
            {"id": "inject-model", "name": "Injected Model", "enabled": True},
            {"id": "noimage-model", "name": "No Image Model", "enabled": True},
        ]
    })
    # Test section
    from app.database import get_db
    with get_db() as db:
        db.execute("UPDATE provider_models SET preprocessor = '' WHERE model_id = 'native-model'")
        db.execute("UPDATE provider_models SET preprocessor = '1' WHERE model_id = 'inject-model'")
        db.execute("UPDATE provider_models SET preprocessor = '1' WHERE model_id = 'noimage-model'")
    yield config


@pytest.mark.asyncio
async def test_policy_preprocess_native_model_images_preserved(preprocess_db):
    """Test behavior."""
    from app.adapters.openai import chat_messages_from_internal
    from app.protocols.ingress import chat_completions_to_internal
    from app.router.proxy import _policy_preprocess_request

    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Describe:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
    ]}]
    original = [{"role": "user", "content": [
        {"type": "text", "text": "Describe:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
    ]}]

    internal = chat_completions_to_internal({"model": "native-model", "messages": msgs})
    modified = await _policy_preprocess_request(internal, "native-model", "", "native-model")
    result = chat_messages_from_internal(internal)

    assert modified is False
    assert result == original
    # Test section
    content = result[0]["content"]
    assert isinstance(content, list)
    assert any(p.get("type") == "image_url" for p in content)


@pytest.mark.asyncio
async def test_policy_preprocess_inject_model_images_stripped(preprocess_db):
    """Test behavior."""
    from app.adapters.openai import chat_messages_from_internal
    from app.protocols.ingress import chat_completions_to_internal
    from app.router.proxy import _policy_preprocess_request
    from unittest.mock import AsyncMock, MagicMock, patch

    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Describe:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/inject-test.jpg"}}
    ]}]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "A landscape photo"}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        internal = chat_completions_to_internal({"model": "inject-model", "messages": msgs})
        modified = await _policy_preprocess_request(internal, "inject-model", "", "inject-model")
        result = chat_messages_from_internal(internal)

    assert modified is True
    content = result[0]["content"]
    assert isinstance(content, str)
    assert "A landscape photo" in content
    # Test section
    assert "inject-test.jpg" not in content


@pytest.mark.asyncio
async def test_policy_preprocess_inject_model_no_images_skips(preprocess_db):
    """Test behavior."""
    from app.adapters.openai import chat_messages_from_internal
    from app.protocols.ingress import chat_completions_to_internal
    from app.router.proxy import _policy_preprocess_request

    msgs = [{"role": "user", "content": "Just a text question"}]
    original = [{"role": "user", "content": "Just a text question"}]

    internal = chat_completions_to_internal({"model": "noimage-model", "messages": msgs})
    modified = await _policy_preprocess_request(internal, "noimage-model", "", "noimage-model")
    result = chat_messages_from_internal(internal)

    assert modified is False
    assert result == original


@pytest.mark.asyncio
async def test_policy_preprocess_model_not_in_db(preprocess_db):
    """Test behavior."""
    from app.adapters.openai import chat_messages_from_internal
    from app.protocols.ingress import chat_completions_to_internal
    from app.router.proxy import _policy_preprocess_request

    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/unknown.jpg"}}
    ]}]
    original = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/unknown.jpg"}}
    ]}]

    internal = chat_completions_to_internal({"model": "unknown-model", "messages": msgs})
    modified = await _policy_preprocess_request(internal, "unknown-model", "", "unknown-model")
    result = chat_messages_from_internal(internal)

    assert modified is False
    assert result == original


@pytest.mark.asyncio
async def test_policy_preprocess_respects_requested_model(preprocess_db):
    """Test behavior."""
    """Test behavior."""
