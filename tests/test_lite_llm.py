"""
Unit tests for liteLLM OpenAI-compatible routing and image normalization helpers.
"""
import pytest
from app.core.images import extract_image_data_uris, has_image_content, normalize_image_content
from app.services.lite_llm import _disable_thinking_when_tools_forced, get_litellm_model_name

# -- Model name routing --

def test_get_litellm_model_name_deepseek():
    """DeepSeek with openai type gets openai/ prefix - liteLLM detects from api_base."""
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
    with pytest.raises(ValueError, match="OpenAI-compatible"):
        get_litellm_model_name("MiniMax-M2.7-highspeed", provider)


def test_get_litellm_model_name_already_prefixed():
    """Composite model IDs are reduced to the provider-local model name."""
    provider = {"id": "any", "provider_type": "openai", "api_base": "https://api.test.com/v1"}
    assert get_litellm_model_name("openai/my-model", provider) == "openai/my-model"
    assert get_litellm_model_name("deepseek/v4", provider) == "openai/v4"


def test_disable_thinking_when_tools_forced_only_with_tools():
    kwargs = {"extra_body": {"thinking": {"type": "enabled"}}}
    _disable_thinking_when_tools_forced(kwargs)
    assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}

    kwargs["tools"] = [{"type": "function", "function": {"name": "run"}}]
    _disable_thinking_when_tools_forced(kwargs)
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}


# -- Image data URI extraction --

# Valid base64 image data (must be >100 chars to pass the minimum length check)
_IMG1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" * 3  # ~300 chars
_IMG2 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKM//2Q==" * 3  # ~300 chars


def test_extract_image_data_uris_from_string():
    content = f"Look at this: data:image/png;base64,{_IMG1}"
    results = extract_image_data_uris(content)
    assert len(results) == 1
    assert results[0][0] == "image/png"


def test_extract_image_data_uris_multiple():
    content = f"First: data:image/jpeg;base64,{_IMG2} Second: data:image/png;base64,{_IMG1}"
    results = extract_image_data_uris(content)
    assert len(results) == 2


def test_extract_image_data_uris_ignores_short_data():
    """Very short base64 strings (<100 chars) are not real images."""
    content = "data:image/png;base64,abc"
    results = extract_image_data_uris(content)
    assert len(results) == 0


def test_extract_image_data_uris_no_images():
    content = "Just plain text without any data URIs."
    results = extract_image_data_uris(content)
    assert len(results) == 0


def test_extract_image_data_uris_non_string():
    assert extract_image_data_uris(None) == []
    assert extract_image_data_uris(123) == []
    assert extract_image_data_uris({"key": "value"}) == []


# -- Image content detection --

def test_has_image_content_with_image_url_part():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
    ]}]
    assert has_image_content(messages) is True


def test_has_image_content_text_only():
    messages = [{"role": "user", "content": "Just text"}]
    assert has_image_content(messages) is False


def test_has_image_content_with_data_uri_in_string():
    messages = [{"role": "user", "content": f"Look: data:image/png;base64,{_IMG1}"}]
    assert has_image_content(messages) is True


def test_has_image_content_empty_messages():
    assert has_image_content([]) is False


# -- Content normalization --

def test_normalize_image_content_extracts_data_uris():
    messages = [{"role": "user", "content": f"Look: data:image/png;base64,{_IMG1}"}]
    normalized = normalize_image_content(messages)
    content = normalized[0]["content"]
    assert isinstance(content, list)  # Should be converted to list of parts
    assert any(p.get("type") == "image_url" for p in content)


def test_normalize_image_content_preserves_no_image_messages():
    messages = [{"role": "user", "content": "Just text"}]
    normalized = normalize_image_content(messages)
    assert normalized[0]["content"] == "Just text"


def test_normalize_image_content_handles_list_content():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"Look: data:image/png;base64,{_IMG1}"}
    ]}]
    normalized = normalize_image_content(messages)
    parts = normalized[0]["content"]
    assert any(p.get("type") == "image_url" for p in parts)


def test_normalize_image_content_preserves_image_url_parts():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
    ]}]
    normalized = normalize_image_content(messages)
    parts = normalized[0]["content"]
    assert len(parts) == 2
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == "https://example.com/photo.jpg"


def test_normalize_image_content_multiple_images_in_list():
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/b.jpg"}},
    ]}]
    normalized = normalize_image_content(messages)
    parts = normalized[0]["content"]
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(image_parts) == 2


def test_normalize_image_content_mixed_data_uri_and_image_url():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"Embedded: data:image/png;base64,{_IMG1}"},
        {"type": "image_url", "image_url": {"url": "https://example.com/existing.jpg"}},
    ]}]
    normalized = normalize_image_content(messages)
    parts = normalized[0]["content"]
    image_urls = [p.get("image_url", {}).get("url") for p in parts if p.get("type") == "image_url"]
    assert f"data:image/png;base64,{_IMG1}" in image_urls
    assert "https://example.com/existing.jpg" in image_urls
