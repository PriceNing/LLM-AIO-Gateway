"""
Unit tests for app.services.lite_llm — model name routing, image handling, content normalization.
"""
import pytest
from app.services.lite_llm import (
    get_litellm_model_name,
    _extract_image_data_uris,
    _has_image_content,
    _normalize_image_content,
)

# ── Model name routing ──

def test_get_litellm_model_name_deepseek():
    """DeepSeek with openai type gets openai/ prefix — liteLLM detects from api_base."""
    provider = {"id": "deepseek", "provider_type": "openai", "api_base": "https://api.deepseek.com"}
    assert get_litellm_model_name("deepseek-v4-pro", provider) == "openai/deepseek-v4-pro"


def test_get_litellm_model_name_openai_standard():
    provider = {"id": "openai", "provider_type": "openai", "api_base": "https://api.openai.com/v1"}
    assert get_litellm_model_name("gpt-4", provider) == "gpt-4"


def test_get_litellm_model_name_openai_custom_endpoint():
    """Provider with OpenAI type but non-OpenAI api_base gets openai/ prefix."""
    provider = {"id": "custom", "provider_type": "openai", "api_base": "https://custom.api.com/v1"}
    assert get_litellm_model_name("my-model", provider) == "openai/my-model"


def test_get_litellm_model_name_anthropic():
    provider = {"id": "minimax", "provider_type": "anthropic", "api_base": "https://api.minimaxi.com/v1"}
    assert get_litellm_model_name("MiniMax-M2.7-highspeed", provider) == "anthropic/MiniMax-M2.7-highspeed"


def test_get_litellm_model_name_already_prefixed():
    """含 / 的模型名：liteLLM 格式不变，复合 ID 提取纯模型名后重建。"""
    provider = {"id": "any", "provider_type": "openai", "api_base": "https://api.test.com/v1"}
    assert get_litellm_model_name("openai/my-model", provider) == "openai/my-model"
    # 复合 ID deepseek/v4 → parse → v4 → 非 OpenAI host 加 openai/ 前缀
    assert get_litellm_model_name("deepseek/v4", provider) == "openai/v4"


# ── Image data URI extraction ──

# Valid base64 image data (must be >100 chars to pass the minimum length check)
_IMG1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" * 3  # ~300 chars
_IMG2 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKM//2Q==" * 3  # ~300 chars


def test_extract_image_data_uris_from_string():
    content = f"Look at this: data:image/png;base64,{_IMG1}"
    results = _extract_image_data_uris(content)
    assert len(results) == 1
    assert results[0][0] == "image/png"


def test_extract_image_data_uris_multiple():
    content = f"First: data:image/jpeg;base64,{_IMG2} Second: data:image/png;base64,{_IMG1}"
    results = _extract_image_data_uris(content)
    assert len(results) == 2


def test_extract_image_data_uris_ignores_short_data():
    """Very short base64 strings (<100 chars) are not real images."""
    content = "data:image/png;base64,abc"
    results = _extract_image_data_uris(content)
    assert len(results) == 0


def test_extract_image_data_uris_no_images():
    content = "Just plain text without any data URIs."
    results = _extract_image_data_uris(content)
    assert len(results) == 0


def test_extract_image_data_uris_non_string():
    assert _extract_image_data_uris(None) == []
    assert _extract_image_data_uris(123) == []
    assert _extract_image_data_uris({"key": "value"}) == []


# ── Image content detection ──

def test_has_image_content_with_image_url_part():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
    ]}]
    assert _has_image_content(messages) is True


def test_has_image_content_text_only():
    messages = [{"role": "user", "content": "Just text"}]
    assert _has_image_content(messages) is False


def test_has_image_content_with_data_uri_in_string():
    messages = [{"role": "user", "content": f"Look: data:image/png;base64,{_IMG1}"}]
    assert _has_image_content(messages) is True


def test_has_image_content_empty_messages():
    assert _has_image_content([]) is False


# ── Content normalization ──

def test_normalize_image_content_extracts_data_uris():
    messages = [{"role": "user", "content": f"Look: data:image/png;base64,{_IMG1}"}]
    normalized = _normalize_image_content(messages)
    content = normalized[0]["content"]
    assert isinstance(content, list)  # Should be converted to list of parts
    assert any(p.get("type") == "image_url" for p in content)


def test_normalize_image_content_preserves_no_image_messages():
    messages = [{"role": "user", "content": "Just text"}]
    normalized = _normalize_image_content(messages)
    assert normalized[0]["content"] == "Just text"


def test_normalize_image_content_handles_list_content():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"Look: data:image/png;base64,{_IMG1}"}
    ]}]
    normalized = _normalize_image_content(messages)
    parts = normalized[0]["content"]
    assert any(p.get("type") == "image_url" for p in parts)


# ── 原生多模态路径：图片保留验证 ──

def test_normalize_image_content_preserves_image_url_parts():
    """原生多模态：已有的 image_url 部分应保持原样，不被移除。"""
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
    ]}]
    normalized = _normalize_image_content(messages)
    parts = normalized[0]["content"]
    assert len(parts) == 2
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == "https://example.com/photo.jpg"


def test_normalize_image_content_multiple_images_in_list():
    """原生多模态：列表中已有的多张 image_url 应全部保留。"""
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/b.jpg"}},
    ]}]
    normalized = _normalize_image_content(messages)
    parts = normalized[0]["content"]
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 2


def test_normalize_image_content_mixed_data_uri_and_image_url():
    """原生多模态：字符串中的 data URI 被提取为 image_url，
    已有的 image_url 保持原样 — 两条路径共存。"""
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/existing.jpg"}},
        {"type": "text", "text": f"Also inline: data:image/png;base64,{_IMG1}"},
    ]}]
    normalized = _normalize_image_content(messages)
    parts = normalized[0]["content"]
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 2  # 原有的 + 从 data URI 提取的
    # 验证原有图片 URL 保留
    urls = [p["image_url"]["url"] for p in image_parts]
    assert "https://example.com/existing.jpg" in urls


def test_normalize_image_content_cleans_text_after_extraction():
    """原生多模态：data URI 从字符串提取后，剩余文本应清理干净。"""
    messages = [{"role": "user", "content": f"Before data:image/png;base64,{_IMG1} After"}]
    normalized = _normalize_image_content(messages)
    content = normalized[0]["content"]
    # 应包含清理后的文本和提取的图片部分
    text_parts = [p for p in content if p.get("type") == "text"]
    image_parts = [p for p in content if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert len(text_parts) == 1
    assert "Before" in text_parts[0]["text"]
    assert "After" in text_parts[0]["text"]
    assert "data:image" not in text_parts[0]["text"]


def test_normalize_image_content_pure_image_no_text():
    """原生多模态：纯图片消息（无文本）— 应正确保留。"""
    messages = [{"role": "user", "content": f"data:image/png;base64,{_IMG1}"}]
    normalized = _normalize_image_content(messages)
    content = normalized[0]["content"]
    assert isinstance(content, list)
    # 纯图片消息提取后应只剩 image_url 部分（文本为空被省略）
    image_parts = [p for p in content if p.get("type") == "image_url"]
    assert len(image_parts) == 1


# ── _has_image_content 原生路径补充 ──

def test_has_image_content_input_image_type():
    """Anthropic/Claude 的 input_image 类型也应被检测。"""
    messages = [{"role": "user", "content": [
        {"type": "input_image", "image_url": {"url": "https://example.com/img.jpg"}}
    ]}]
    assert _has_image_content(messages) is True


def test_has_image_content_anthropic_image_block():
    """Anthropic 原生的 image 内容块应被检测。"""
    messages = [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}}
    ]}]
    assert _has_image_content(messages) is True


def test_has_image_content_nested_tool_result():
    """嵌套在 tool_result 中的图片应被检测。"""
    messages = [{"role": "tool", "tool_call_id": "c1", "content": [
        {"type": "text", "text": "Result:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}}
    ]}]
    assert _has_image_content(messages) is True


def test_has_image_content_tool_result_string_with_data_uri():
    """tool_result 的字符串 content 中包含 data URI 也应被检测。"""
    messages = [{"role": "tool", "tool_call_id": "c1",
                 "content": f"Result with image: data:image/png;base64,{_IMG1}"}]
    assert _has_image_content(messages) is True
