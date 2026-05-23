import pytest
from fastapi.testclient import TestClient
from main import app
from app.config import load_config
from app.database import init_db, add_provider, add_user, add_user_api_key

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
        "preprocessors": {
            "test-vision": {
                "api_base": "http://127.0.0.1:8080/v1",
                "model": "test-vision",
                "api_key": "k",
                "timeout": 30,
                "max_images": 20,
                "prompt": "Describe this image.",
                "enabled": True,
            }
        }
    }
    config.save()
    init_db(db_path)
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
async def test_maybe_preprocess_native_model_images_preserved(preprocess_db):
    """Test behavior."""
    from app.router.proxy import _maybe_preprocess

    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Describe:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
    ]}]
    original = [{"role": "user", "content": [
        {"type": "text", "text": "Describe:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
    ]}]

    result, modified = await _maybe_preprocess(msgs, "native-model")

    assert modified is False
    assert result == original
    # Test section
    content = result[0]["content"]
    assert isinstance(content, list)
    assert any(p.get("type") == "image_url" for p in content)


@pytest.mark.asyncio
async def test_maybe_preprocess_inject_model_images_stripped(preprocess_db):
    """Test behavior."""
    from app.router.proxy import _maybe_preprocess
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
        result, modified = await _maybe_preprocess(msgs, "inject-model")

    assert modified is True
    content = result[0]["content"]
    assert isinstance(content, str)
    assert "A landscape photo" in content
    # Test section
    assert "inject-test.jpg" not in content


@pytest.mark.asyncio
async def test_maybe_preprocess_inject_model_no_images_skips(preprocess_db):
    """Test behavior."""
    from app.router.proxy import _maybe_preprocess

    msgs = [{"role": "user", "content": "Just a text question"}]
    original = [{"role": "user", "content": "Just a text question"}]

    result, modified = await _maybe_preprocess(msgs, "noimage-model")

    assert modified is False
    assert result == original


@pytest.mark.asyncio
async def test_maybe_preprocess_model_not_in_db(preprocess_db):
    """Test behavior."""
    from app.router.proxy import _maybe_preprocess

    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/unknown.jpg"}}
    ]}]
    original = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/unknown.jpg"}}
    ]}]

    result, modified = await _maybe_preprocess(msgs, "unknown-model")

    assert modified is False
    assert result == original


@pytest.mark.asyncio
async def test_maybe_preprocess_respects_requested_model(preprocess_db):
    """Test behavior."""
    """Test behavior."""
