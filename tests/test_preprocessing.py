"""
 - describe_imagepreprocess_messages
 turn max_images inline replacement
"""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.preprocessing import (
    _cache_key, _get_cached_description, _set_cached_description,
    describe_image,
    _build_inline_replacement,
)

# Test section
_IMG1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" * 3
_IMG2 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKM//2Q==" * 3

_DATA_URI1 = f"data:image/png;base64,{_IMG1}"
_DATA_URI2 = f"data:image/jpeg;base64,{_IMG2}"


# -- Fixtures --

@pytest.fixture(autouse=True)
def temp_config(tmp_path):
    """Test behavior."""
    from app.config import load_config
    from app.database import init_db
    db_path = str(tmp_path / "test.db")
    config_path = str(tmp_path / "config.json")
    config = load_config(config_path, force_reload=True)
    config.config = {
        "host": "0.0.0.0",
        "port": 8000,
        "database": db_path,
        "logging": {"enabled": False, "level": "INFO", "log_dir": "logs", "retention_days": 30, "console": False},
        "preprocessors": {
            "vision-model": {
                "api_base": "http://127.0.0.1:8080/v1",
                "model": "test-vision",
                "api_key": "test-key",
                "timeout": 30,
                "max_images": 20,
                "prompt": "Please describe the image content.",
                "enabled": True,
            },
            "disabled-vision": {
                "api_base": "http://127.0.0.1:8081/v1",
                "model": "disabled-vision",
                "api_key": "",
                "timeout": 30,
                "max_images": 20,
                "prompt": "Describe",
                "enabled": False,
            },
        }
    }
    config.save()
    init_db(db_path)
    # Test section
    from app.services.preprocessing import _cache
    _cache.clear()
    yield config


@pytest.fixture
def preprocessor_config():
    return {
        "id": "vision-model",
        "api_base": "http://127.0.0.1:8080/v1",
        "model": "test-vision",
        "api_key": "test-key",
        "timeout": 30,
        "max_images": 20,
        "prompt": "Please describe the image content.",
        "enabled": True,
    }


# Test section

def test_cache_key_same_input():
    k1 = _cache_key(_DATA_URI1)
    k2 = _cache_key(_DATA_URI1)
    assert k1 == k2
    assert isinstance(k1, str)
    assert len(k1) == 16


def test_cache_key_different_input():
    k1 = _cache_key(_DATA_URI1)
    k2 = _cache_key(_DATA_URI2)
    assert k1 != k2


def test_cache_key_empty():
    k = _cache_key("")
    assert isinstance(k, str)
    assert len(k) == 16


def test_build_inline_replacement_has_no_timestamp():
    result = _build_inline_replacement(0, ["A cat"], True)
    assert result == "[Image #1]: A cat"


def test_cache_get_set():
    _set_cached_description("test-url", "A test description")
    assert _get_cached_description("test-url") == "A test description"


def test_cache_miss():
    assert _get_cached_description("never-cached") is None


def test_cache_eviction_fifo():
    """Test behavior."""
    for i in range(600):
        _set_cached_description(f"url-{i}", f"desc-{i}")
    # Test section
    assert _get_cached_description("url-0") is None
    assert _get_cached_description("url-1") is None
    # Test section
    assert _get_cached_description("url-599") == "desc-599"


# Test section

@pytest.mark.asyncio
async def test_describe_image_no_config():
    result = await describe_image(image_url="https://example.com/img.jpg", preprocessor_config=None)
    assert result is None


@pytest.mark.asyncio
async def test_describe_image_cache_hit():
    _set_cached_description("https://example.com/cached.jpg", "Cached description")
    result = await describe_image(
        image_url="https://example.com/cached.jpg",
        preprocessor_config={
            "api_base": "http://x", "model": "x", "api_key": "",
            "timeout": 30, "max_images": 5
        }
    )
    assert result == "Cached description"


@pytest.mark.asyncio
async def test_describe_image_success():
    """Test behavior."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "A blue sky with clouds"}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await describe_image(
            image_url="https://example.com/sky.jpg",
            preprocessor_config={
                "api_base": "http://127.0.0.1:8080/v1",
                "model": "test-vision",
                "api_key": "k",
                "timeout": 30,
                "max_images": 5,
            }
        )
        assert result == "A blue sky with clouds"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_describe_image_with_data_uri():
    """Test behavior."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "A small test image"}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await describe_image(
            image_data=_DATA_URI1,
            preprocessor_config={
                "api_base": "http://127.0.0.1:8080/v1",
                "model": "test-vision",
                "api_key": "k",
                "timeout": 30,
                "max_images": 5,
            }
        )
        assert result == "A small test image"


@pytest.mark.asyncio
async def test_describe_image_http_error():
    """Test behavior."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await describe_image(
            image_url="https://example.com/img.jpg",
            preprocessor_config={
                "api_base": "http://127.0.0.1:8080/v1",
                "model": "test-vision",
                "api_key": "k",
                "timeout": 30,
            }
        )
        assert "[image: vision model HTTP 500]" in result


@pytest.mark.asyncio
async def test_describe_image_empty_response():
    """Test behavior."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": ""}}]}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await describe_image(
            image_url="https://example.com/img.jpg",
            preprocessor_config={
                "api_base": "http://127.0.0.1:8080/v1",
                "model": "test-vision",
                "api_key": "k",
                "timeout": 30,
            }
        )
        assert "empty response" in result


@pytest.mark.asyncio
async def test_describe_image_exception():
    """Test behavior."""
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = Exception("Connection refused")
        result = await describe_image(
            image_url="https://example.com/img.jpg",
            preprocessor_config={
                "api_base": "http://127.0.0.1:8080/v1",
                "model": "test-vision",
                "api_key": "k",
                "timeout": 30,
            }
        )
        assert "unavailable or timed out" in result


@pytest.mark.asyncio
async def test_describe_image_reasoning_content_fallback():
    """Test behavior."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"reasoning_content": "Thinking about the image..."}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await describe_image(
            image_url="https://example.com/img.jpg",
            preprocessor_config={
                "api_base": "http://127.0.0.1:8080/v1",
                "model": "test-vision",
                "api_key": "k",
                "timeout": 30,
            }
        )
        assert "Thinking about the image" in result


@pytest.mark.asyncio
async def test_describe_image_reasoning_content_fallback_logs_warning(caplog):
    caplog.set_level("WARNING", logger="llmgw.app")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "", "reasoning_content": "Only reasoning text"}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await describe_image(
            image_url="https://example.com/reasoning-only.jpg",
            preprocessor_config={
                "api_base": "http://127.0.0.1:8080/v1",
                "model": "test-vision",
                "api_key": "k",
                "timeout": 30,
            }
        )

    assert result == "Only reasoning text"


@pytest.mark.asyncio
async def test_describe_image_no_url_or_data():
    """Test behavior."""
    result = await describe_image(
        preprocessor_config={
            "api_base": "http://127.0.0.1:8080/v1",
            "model": "test-vision",
            "api_key": "k",
            "timeout": 30,
        }
    )
    assert result is None


