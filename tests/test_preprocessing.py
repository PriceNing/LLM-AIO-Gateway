"""
图像预处理管道测试 — 缓存、describe_image、preprocess_messages、
多轮对话 turn 边界检测、去重、max_images 限制、inline replacement。
"""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.preprocessing import (
    _cache_key, _get_cached_description, _set_cached_description,
    _get_preprocessor_config, describe_image, preprocess_messages,
    _strip_all_images, _strip_images_with_descriptions,
    _build_inline_replacement, _join_text_parts,
)

# 足够长的 base64 图片数据（>100 字符，满足最小长度检测）
_IMG1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" * 3
_IMG2 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKM//2Q==" * 3

_DATA_URI1 = f"data:image/png;base64,{_IMG1}"
_DATA_URI2 = f"data:image/jpeg;base64,{_IMG2}"


# ── Fixtures ──

@pytest.fixture(autouse=True)
def temp_config(tmp_path):
    """每个测试使用临时配置和数据库。"""
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
                "prompt": "请描述图片中的内容。",
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
    # 清空预处理缓存，避免测试间缓存污染
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
        "prompt": "请描述图片中的内容。",
        "enabled": True,
    }


# ── 缓存测试 ──

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


def test_cache_get_set():
    _set_cached_description("test-url", "A test description")
    assert _get_cached_description("test-url") == "A test description"


def test_cache_miss():
    assert _get_cached_description("never-cached") is None


def test_cache_eviction_fifo():
    """缓存超过 MAX_CACHE_SIZE 时 FIFO 淘汰最旧的条目。"""
    for i in range(600):
        _set_cached_description(f"url-{i}", f"desc-{i}")
    # 最早的一些应该被淘汰
    assert _get_cached_description("url-0") is None
    assert _get_cached_description("url-1") is None
    # 最新的应该还在
    assert _get_cached_description("url-599") == "desc-599"


# ── describe_image 测试 ──

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
    """模拟视觉模型成功返回描述。"""
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
    """使用 data URI 而非 URL 描述图片。"""
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
    """视觉模型返回 HTTP 错误时应返回错误占位符。"""
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
    """视觉模型返回空内容时应返回空响应占位符。"""
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
    """视觉模型不可达时应返回不可用占位符。"""
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
    """部分模型返回 reasoning_content 而非 content。"""
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
async def test_describe_image_no_url_or_data():
    """既没有 URL 也没有 data URI 时应返回 None。"""
    result = await describe_image(
        preprocessor_config={
            "api_base": "http://127.0.0.1:8080/v1",
            "model": "test-vision",
            "api_key": "k",
            "timeout": 30,
        }
    )
    assert result is None


# ── preprocess_messages 测试 ──

@pytest.mark.asyncio
async def test_preprocess_no_config():
    """无预处理器配置时应跳过并返回原消息。"""
    msgs = [{"role": "user", "content": "Hello"}]
    result = await preprocess_messages(msgs, preprocessor_config=None)
    assert result == msgs


@pytest.mark.asyncio
async def test_preprocess_disabled():
    """预处理器被禁用时应跳过。"""
    msgs = [{"role": "user", "content": "Hello"}]
    config = {"id": "test", "api_base": "http://x", "model": "x", "enabled": False}
    result = await preprocess_messages(msgs, preprocessor_config=config)
    assert result == msgs


@pytest.mark.asyncio
async def test_preprocess_no_images():
    """无图片的消息应跳过处理。"""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the weather?"},
    ]
    config = {"id": "test", "api_base": "http://x", "model": "x", "api_key": "", "enabled": True}
    result = await preprocess_messages(msgs, preprocessor_config=config)
    assert result == msgs  # 无变化


@pytest.mark.asyncio
async def test_preprocess_single_image_current_turn():
    """当前轮次有一张图片 — 应描述并内联替换。"""
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
        ]}
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "A beautiful sunset over the ocean"}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await preprocess_messages(
            msgs,
            preprocessor_config={
                "id": "vision", "api_base": "http://x/v1", "model": "v",
                "api_key": "", "timeout": 30, "max_images": 20, "enabled": True,
            }
        )

    # 图片应该被移除，描述应该被插入
    content = result[0]["content"]
    assert isinstance(content, str)
    assert "A beautiful sunset over the ocean" in content
    assert "photo.jpg" not in content  # URL 已移除


@pytest.mark.asyncio
async def test_preprocess_images_in_history_only():
    """图片仅在历史消息中（当前轮次之前）— 只 strip，不描述。"""
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "Old image:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/old.jpg"}}
        ]},
        {"role": "assistant", "content": "I see an old photo."},  # ← turn 边界
        {"role": "user", "content": "What about this new text?"},  # 当前轮次，无图片
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Description should not be called"}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await preprocess_messages(
            msgs,
            preprocessor_config={
                "id": "vision", "api_base": "http://x/v1", "model": "v",
                "api_key": "", "timeout": 30, "max_images": 20, "enabled": True,
            }
        )

    # 历史图片应被 strip 但不应该调用视觉模型
    mock_post.assert_not_called()
    # 历史消息中的图片应被替换为占位符
    old_content = result[0]["content"]
    assert isinstance(old_content, str)
    assert "[image: removed]" in old_content
    # 当前轮次消息应保持不变
    assert result[2]["content"] == "What about this new text?"


@pytest.mark.asyncio
async def test_preprocess_multi_turn_with_description_tag():
    """包含 <image_description> 的用户消息标记 turn 边界。"""
    msgs = [
        {"role": "user", "content": "Look at this image"},
        {"role": "assistant", "content": "I'll describe it."},
        {"role": "user", "content": "[Image #1 at 12:00:00]: A cat"},  # 已处理过
        {"role": "assistant", "content": "That's a cute cat."},
        {"role": "user", "content": [
            {"type": "text", "text": "What about this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/dog.jpg"}}
        ]},  # 新轮次，有新图片
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "A golden retriever"}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await preprocess_messages(
            msgs,
            preprocessor_config={
                "id": "vision", "api_base": "http://x/v1", "model": "v",
                "api_key": "", "timeout": 30, "max_images": 20, "enabled": True,
            }
        )

    # 仅应对新图片调用一次视觉模型
    assert mock_post.call_count == 1
    # 新的 user 消息应包含新描述
    new_content = result[4]["content"]
    assert isinstance(new_content, str)
    assert "A golden retriever" in new_content


@pytest.mark.asyncio
async def test_preprocess_turn_boundary_assistant_with_text():
    """有文本但无 tool_calls 的 assistant 消息标记上一轮结束。"""
    msgs = [
        {"role": "user", "content": "Old request"},
        {"role": "assistant", "content": "Old reply with text"},  # ← turn 边界（有文本、无 tool_calls）
        {"role": "user", "content": [
            {"type": "text", "text": "New request"},
            {"type": "image_url", "image_url": {"url": "https://example.com/new.jpg"}}
        ]},
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "A new image description"}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await preprocess_messages(
            msgs,
            preprocessor_config={
                "id": "vision", "api_base": "http://x/v1", "model": "v",
                "api_key": "", "timeout": 30, "max_images": 20, "enabled": True,
            }
        )

    assert mock_post.call_count == 1


@pytest.mark.asyncio
async def test_preprocess_turn_boundary_assistant_tool_calls_not_boundary():
    """有 tool_calls 的 assistant 消息不作为 turn 边界（后面还有 tool 消息）。"""
    msgs = [
        {"role": "user", "content": "Search for images"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}}
        ]},
        {"role": "assistant", "content": "Here are the results:"},  # ← turn 边界（有文本、无 tool_calls）
        {"role": "user", "content": [
            {"type": "text", "text": "New request"},
            {"type": "image_url", "image_url": {"url": "https://example.com/new.jpg"}}
        ]},
    ]

    # 工具结果中的图片在历史中，不应触发描述
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "New image desc"}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await preprocess_messages(
            msgs,
            preprocessor_config={
                "id": "vision", "api_base": "http://x/v1", "model": "v",
                "api_key": "", "timeout": 30, "max_images": 20, "enabled": True,
            }
        )

    # 只有新轮次的图片被描述（历史 tool_result 中的图片只 strip）
    assert mock_post.call_count == 1


@pytest.mark.asyncio
async def test_preprocess_deduplicate_same_turn():
    """同一轮次内相同图片在 user 消息和 tool_result 中都出现时应去重。
    没有文本回复的 assistant（只有 tool_calls）不作为 turn 边界，
    后续的 tool 消息和 assistant 响应都属于同一轮次。"""
    same_data_uri = _DATA_URI1
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Look at this: {same_data_uri}"},
        ]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": [
            {"type": "text", "text": "File read result:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}}
        ]},
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "A test pattern"}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        await preprocess_messages(
            msgs,
            preprocessor_config={
                "id": "vision", "api_base": "http://x/v1", "model": "v",
                "api_key": "", "timeout": 30, "max_images": 20, "enabled": True,
            }
        )

    # 相同图片仅应调用一次视觉模型（去重）
    assert mock_post.call_count == 1


@pytest.mark.asyncio
async def test_preprocess_max_images_limit():
    """超过 max_images 限制的图片应被跳过（只描述前 N 张）。"""
    msgs = [{"role": "user", "content": []}]
    for i in range(7):
        msgs[0]["content"].append({"type": "image_url", "image_url": {"url": f"https://example.com/img{i}.jpg"}})

    call_count = 0

    async def mock_describe(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return f"Description {call_count}"

    with patch("app.services.preprocessing.describe_image", side_effect=mock_describe):
        result = await preprocess_messages(
            msgs,
            preprocessor_config={
                "id": "vision", "api_base": "http://x/v1", "model": "v",
                "api_key": "", "timeout": 30, "max_images": 3, "enabled": True,
            }
        )

    # 只描述前 3 张（max_images=3）
    assert call_count == 3
    content = result[0]["content"]
    assert "Description 1" in content
    assert "Description 2" in content
    assert "Description 3" in content
    assert "Description 4" not in content


@pytest.mark.asyncio
async def test_preprocess_images_in_tool_result_current_turn():
    """当前轮次中 tool_result 包含嵌套图片 — 应描述并替换。"""
    msgs = [
        {"role": "user", "content": "Read the file"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": [
            {"type": "text", "text": "File contents:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}}
        ]},
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "A screenshot showing code"}}]
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await preprocess_messages(
            msgs,
            preprocessor_config={
                "id": "vision", "api_base": "http://x/v1", "model": "v",
                "api_key": "", "timeout": 30, "max_images": 20, "enabled": True,
            }
        )

    # tool_result 中的图片应被替换为描述
    tool_content = result[2]["content"]
    assert isinstance(tool_content, str)
    assert "A screenshot showing code" in tool_content


@pytest.mark.asyncio
async def test_preprocess_mixed_content_types():
    """混合内容：文本 + 多张图片 — 应全部描述并替换。"""
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Compare these:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/mix-a.jpg"}},
        {"type": "text", "text": "and this:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/mix-b.jpg"}},
    ]}]

    call_count = 0

    async def mock_describe(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return f"Image #{call_count}"

    with patch("app.services.preprocessing.describe_image", side_effect=mock_describe):
        result = await preprocess_messages(
            msgs,
            preprocessor_config={
                "id": "vision", "api_base": "http://x/v1", "model": "v",
                "api_key": "", "timeout": 30, "max_images": 20, "enabled": True,
            }
        )

    assert call_count == 2
    content = result[0]["content"]
    assert isinstance(content, str)
    assert "Image #1" in content
    assert "Image #2" in content
    assert "https://example.com/mix-a.jpg" not in content
    assert "https://example.com/mix-b.jpg" not in content


# ── _strip_all_images 测试 ──

def test_strip_all_images_image_url():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Look:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
    ]}]
    _strip_all_images(msgs)
    content = msgs[0]["content"]
    assert isinstance(content, str)
    assert "[image: removed]" in content
    assert "img.jpg" not in content


def test_strip_all_images_input_image():
    msgs = [{"role": "user", "content": [
        {"type": "input_image", "image_url": {"url": "https://example.com/img.jpg"}}
    ]}]
    _strip_all_images(msgs)
    assert "[image: removed]" in msgs[0]["content"]


def test_strip_all_images_anthropic_image():
    msgs = [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}}
    ]}]
    _strip_all_images(msgs)
    assert "[image: removed]" in msgs[0]["content"]


def test_strip_all_images_nested_in_tool_result():
    msgs = [{"role": "tool", "tool_call_id": "c1", "content": [
        {"type": "text", "text": "Result:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}}
    ]}]
    _strip_all_images(msgs)
    content = msgs[0]["content"]
    assert isinstance(content, str)
    assert "Result:" in content
    assert "[image: removed]" in content


def test_strip_all_images_data_uri_in_string():
    msgs = [{"role": "user", "content": f"Check: {_DATA_URI1}"}]
    _strip_all_images(msgs)
    assert _DATA_URI1 not in msgs[0]["content"]
    assert "[image: removed]" in msgs[0]["content"]


def test_strip_all_images_text_only_unchanged():
    msgs = [{"role": "user", "content": "Just text"}]
    original = msgs[0]["content"]
    _strip_all_images(msgs)
    assert msgs[0]["content"] == original


def test_strip_all_images_empty_list():
    msgs = [{"role": "user", "content": []}]
    _strip_all_images(msgs)
    assert msgs[0]["content"] == ""


# ── _build_inline_replacement 测试 ──

def test_build_inline_replacement_current_turn():
    descriptions = ["A cat on a sofa", "A dog in the park"]
    result = _build_inline_replacement(0, descriptions, is_current=True)
    assert "[image: removed]" not in result  # 有描述时不再附 removed 标记
    assert "[Image #1" in result
    assert "A cat on a sofa" in result


def test_build_inline_replacement_history():
    descriptions = ["Should not appear"]
    result = _build_inline_replacement(0, descriptions, is_current=False)
    assert result == "[image: removed]"
    assert "Should not appear" not in result


def test_build_inline_replacement_out_of_range():
    descriptions = ["Only one"]
    result = _build_inline_replacement(5, descriptions, is_current=True)
    assert result == "[image: removed]"


# ── _join_text_parts 测试 ──

def test_join_text_parts_all_text_dicts():
    parts = [
        {"type": "text", "text": "Part 1"},
        {"type": "text", "text": "Part 2"},
    ]
    result = _join_text_parts(parts)
    assert result == "Part 1\nPart 2"


def test_join_text_parts_mixed():
    parts = [
        {"type": "text", "text": "Text part"},
        {"type": "image_url", "image_url": {"url": "x"}},
    ]
    result = _join_text_parts(parts)
    assert isinstance(result, list)
    assert len(result) == 2


def test_join_text_parts_empty():
    assert _join_text_parts([]) == ""


def test_join_text_parts_strings_and_dicts():
    parts = ["Plain string", {"type": "text", "text": "Dict text"}]
    result = _join_text_parts(parts)
    assert result == "Plain string\nDict text"


# ── _get_preprocessor_config 测试 ──

def test_get_preprocessor_config_exists(temp_config):
    cfg = _get_preprocessor_config("vision-model")
    assert cfg is not None
    assert cfg["model"] == "test-vision"
    assert cfg["enabled"] is True


def test_get_preprocessor_config_not_exists():
    cfg = _get_preprocessor_config("nonexistent")
    assert cfg is None


def test_get_preprocessor_config_disabled(temp_config):
    cfg = _get_preprocessor_config("disabled-vision")
    assert cfg is not None
    assert cfg["enabled"] is False


# ── _strip_images_with_descriptions 测试 ──

def test_strip_images_with_descriptions_current_and_history():
    """当前轮次图片获得描述，历史图片只 strip。"""
    descriptions = ["Cat description", "Dog description"]
    msgs = [
        # 历史消息（index 0, < turn_start=2）
        {"role": "user", "content": [
            {"type": "text", "text": "Old:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/old.jpg"}}
        ]},
        {"role": "assistant", "content": "I see it."},
        # 当前轮次（index 2, >= turn_start=2）
        {"role": "user", "content": [
            {"type": "text", "text": "New:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/new1.jpg"}}
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "Also:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/new2.jpg"}}
        ]},
    ]

    _strip_images_with_descriptions(msgs, descriptions, turn_start=2)

    # 历史消息：只有占位符，无描述
    hist_content = msgs[0]["content"]
    assert isinstance(hist_content, str)
    assert "[image: removed]" in hist_content
    assert "Cat description" not in hist_content

    # 当前轮次第一张：占位符 + 描述 #1
    curr1 = msgs[2]["content"]
    assert isinstance(curr1, str)
    assert "[Image #1" in curr1
    assert "Cat description" in curr1

    # 当前轮次第二张：占位符 + 描述 #2
    curr2 = msgs[3]["content"]
    assert isinstance(curr2, str)
    assert "[Image #2" in curr2
    assert "Dog description" in curr2
