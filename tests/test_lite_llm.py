"""
Unit tests for liteLLM OpenAI-compatible routing and image normalization helpers.
"""
import pytest
from app.core.images import extract_image_data_uris, has_image_content, normalize_image_content
from tests.image_fixtures import JPEG_B64, png_b64
from app.services.lite_llm import (
    _disable_thinking_for_missing_reasoning,
    _disable_thinking_when_tools_forced,
    _system_messages_first,
    apply_temperature_lock,
    build_completion_args,
    get_litellm_model_name,
    model_temperature_locks,
)
from app.core.model_capabilities import builtin_capabilities

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


def test_system_messages_first_collapses_to_one_leading_system():
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "extra"},
        {"role": "assistant", "content": "ok"},
    ]
    assert _system_messages_first(messages) == [
        {"role": "system", "content": "base\n\nextra"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]


def test_disable_thinking_when_tools_forced_only_with_tools():
    kwargs = {"extra_body": {"thinking": {"type": "enabled"}}}
    _disable_thinking_when_tools_forced(kwargs)
    assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}

    kwargs["tools"] = [{"type": "function", "function": {"name": "run"}}]
    _disable_thinking_when_tools_forced(kwargs)
    assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}

    kwargs["tool_choice"] = "required"
    _disable_thinking_when_tools_forced(kwargs)
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}


def test_disable_thinking_for_missing_reasoning_only_when_marked():
    kwargs = {"extra_body": {"thinking": {"type": "enabled"}}}
    _disable_thinking_for_missing_reasoning(kwargs)
    assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}

    kwargs["disable_thinking_for_missing_reasoning"] = True
    _disable_thinking_for_missing_reasoning(kwargs)
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
    assert "disable_thinking_for_missing_reasoning" not in kwargs


def test_build_completion_args_forwards_only_transport_headers(monkeypatch):
    provider = {
        "id": "headers", "provider_type": "openai", "api_base": "https://example.test/v1",
        "api_key": "test-key", "enabled": True,
        "provider_options": {
            "thinking": "enabled",
            "thinking_budget_tokens": 8000,
        },
        "upstream_headers": {
            "User-Agent": "gateway-test/1.0",
        },
    }
    monkeypatch.setattr("app.services.lite_llm.get_provider", lambda _provider_id: provider)

    _model, params = build_completion_args("test-model", "headers")

    assert params["extra_headers"] == {"User-Agent": "gateway-test/1.0"}
    assert params["extra_body"] == {"thinking": {"type": "enabled"}}


# -- Temperature lock：由模型能力驱动，条件里不得出现模型名 --

def test_temperature_lock_coerces_when_capability_declares_fixed():
    kwargs = {"temperature": 0}
    apply_temperature_lock({"fixed_temperature": 1}, kwargs)
    assert kwargs["temperature"] == 1


def test_temperature_lock_keeps_value_when_it_already_matches():
    kwargs = {"temperature": 1}
    apply_temperature_lock({"fixed_temperature": 1}, kwargs)
    assert kwargs["temperature"] == 1


def test_temperature_lock_with_reasoning_only_applies_with_reasoning():
    plain = {"temperature": 0}
    apply_temperature_lock({"fixed_temperature_with_reasoning": 1}, plain)
    assert plain["temperature"] == 0

    for effort in ("none", None):
        kw = {"temperature": 0.7, "reasoning_effort": effort}
        apply_temperature_lock({"fixed_temperature_with_reasoning": 1}, kw)
        assert kw["temperature"] == 0.7, effort

    kw = {"temperature": 0.7, "reasoning_effort": "medium"}
    apply_temperature_lock({"fixed_temperature_with_reasoning": 1}, kw)
    assert kw["temperature"] == 1


def test_temperature_lock_without_capability_leaves_request_untouched():
    kwargs = {"temperature": 0}
    apply_temperature_lock({}, kwargs)
    assert kwargs["temperature"] == 0


def test_temperature_lock_ignores_requests_without_temperature():
    kwargs = {"reasoning_effort": "high"}
    apply_temperature_lock({"fixed_temperature": 1}, kwargs)
    assert "temperature" not in kwargs


# 行为等价：以下三组断言覆盖旧版基于 startswith("gpt-5") 的全部判定分支。

@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-codex", "gpt-5.6", "openai/gpt-5.6"])
def test_gpt5_family_locks_temperature_via_builtin_capability(model):
    kwargs = {"temperature": 0}
    apply_temperature_lock(builtin_capabilities(model), kwargs)
    assert kwargs["temperature"] == 1


def test_gpt51_locks_only_when_reasoning_is_enabled():
    caps = builtin_capabilities("gpt-5.1")
    assert caps.get("fixed_temperature") is None, "gpt-5.1 必须落在 gpt-5 之前命中"

    plain = {"temperature": 0}
    apply_temperature_lock(caps, plain)
    assert plain["temperature"] == 0

    with_reasoning = {"temperature": 0.7, "reasoning_effort": "medium"}
    apply_temperature_lock(caps, with_reasoning)
    assert with_reasoning["temperature"] == 1


def test_other_model_families_are_left_alone():
    for model in ("gpt-4.1", "gpt-6", "claude-sonnet-4", "mimo-v2.6-pro"):
        kwargs = {"temperature": 0, "reasoning_effort": "medium"}
        apply_temperature_lock(builtin_capabilities(model), kwargs)
        assert kwargs["temperature"] == 0, model


def test_stored_override_wins_over_builtin_family(monkeypatch):
    """管理员/上游覆盖能改变锁，不需要改代码。"""
    monkeypatch.setattr(
        "app.services.lite_llm.get_model_stored_capabilities",
        lambda provider_id, model: {"fixed_temperature": 0.7},
    )
    monkeypatch.setattr("app.services.lite_llm.registry_lookup", lambda *args: {})

    caps = model_temperature_locks("openai/gpt-5.6-terra", "PixelAPI")
    assert caps["fixed_temperature"] == 0.7
    kwargs = {"temperature": 0}
    apply_temperature_lock(caps, kwargs)
    assert kwargs["temperature"] == 0.7


def test_lock_still_applies_when_no_provider_model_row(monkeypatch):
    """未同步的模型不得因查不到 DB 行而丢失锁（回退到内置家族表）。"""
    monkeypatch.setattr(
        "app.services.lite_llm.get_model_stored_capabilities",
        lambda provider_id, model: {},
    )
    monkeypatch.setattr("app.services.lite_llm.registry_lookup", lambda *args: {})

    caps = model_temperature_locks("gpt-5.6-terra", "PixelAPI")
    assert caps.get("fixed_temperature") == 1


# -- Image data URI extraction --

# 真实、完整可解码的图片载荷（网关会校验 base64 与容器完整性，
# 不再接受“把一段 base64 重复几遍凑长度”的伪图片）。
_IMG1 = png_b64(16, 16)
_IMG2 = JPEG_B64


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


def test_compatibility_patches_are_active():
    """补丁依赖 litellm 内部实现；升级后失效必须被测试捕获（Q3）。"""
    from app.services.lite_llm import compatibility_patch_status

    status = compatibility_patch_status()
    assert status == {"fields": True, "converter": True}, (
        f"liteLLM 兼容补丁未完全生效：{status}。"
        "reasoning_content 与 prompt cache 统计会在多轮工具调用中丢失。"
    )
