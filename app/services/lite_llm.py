import logging
import re
import litellm
from litellm import completion
from typing import Optional, Any
from app.database import get_providers, get_provider, find_provider_by_model
from app.services.logger import get_logger
from app.config import get_default

# ── Monkey-patch: MiniMax Anthropic responses may have non-string `id` fields ──
# liteLLM's AnthropicResponse and related models require `id: str`, but MiniMax's
# Anthropic-compatible endpoint returns `id` as a bare integer.  Coerce to str.
try:
    from typing import Annotated
    from pydantic import BeforeValidator
    from litellm.llms.anthropic.chat import (
        AnthropicResponse,
        AnthropicResponseContentBlockToolUse,
    )
    from litellm.types.utils import ModelResponse, Message, Delta
    from litellm.types.llms.openai import ChatCompletionChunk

    _coerce_id = Annotated[str, BeforeValidator(lambda x: str(x) if x is not None else x)]
    for _model in (
        AnthropicResponse,
        AnthropicResponseContentBlockToolUse,
        ModelResponse,
        ChatCompletionChunk,
    ):
        _model.model_fields["id"].annotation = _coerce_id
        _model.model_rebuild(force=True)

    # Add reasoning_content to Message and Delta models so liteLLM preserves it
    # when providers (llama.cpp, DeepSeek, etc.) include it in the response.
    from pydantic import Field
    for _model in (Message, Delta):
        if "reasoning_content" not in _model.model_fields:
            _model.model_fields["reasoning_content"] = Field(default=None)
            _model.model_rebuild(force=True)
except Exception:
    logging.getLogger("llmgw.app").warning("liteLLM monkey-patch for MiniMax id coercion failed — MiniMax Anthropic responses may have type errors")

# ── Monkey-patch: preserve reasoning_content in liteLLM responses ──
# liteLLM's convert_to_model_response_object (utils.py:5755) constructs Message
# objects from the OpenAI response dict but only extracts known fields (content,
# role, function_call, tool_calls).  reasoning_content is dropped even though the
# raw API response includes it.  We wrap the converter to inject reasoning_content
# back into each choice's message.
try:
    import litellm.utils as _litellm_utils
    _original_convert = _litellm_utils.convert_to_model_response_object

    def _patched_convert(response_object=None, model_response_object=None, **kwargs):
        result = _original_convert(
            response_object=response_object,
            model_response_object=model_response_object,
            **kwargs
        )
        if (response_object and isinstance(response_object, dict)
                and isinstance(result, ModelResponse)):
            for i, choice in enumerate(result.choices):
                if i < len(response_object.get("choices", [])):
                    rc = (response_object["choices"][i]
                          .get("message", {})
                          .get("reasoning_content"))
                    if rc:
                        choice.message.reasoning_content = rc
        return result

    _litellm_utils.convert_to_model_response_object = _patched_convert
except Exception:
    logging.getLogger("llmgw.app").warning(
        "liteLLM monkey-patch for reasoning_content failed"
    )

# ── Monkey-patch removed: thinking mode is now configured via
# provider.extra_headers in build_completion_args and passthrough paths.
# The AnthropicChatCompletion path (used by MiniMax etc.) no longer needs
# a monkey-patch — each provider's extra_headers controls thinking.

# ── Monkey-patch: convert reasoning_content → thinking_blocks for Anthropic messages ──
# liteLLM's anthropic_messages_pt (factory.py:2558) only looks for "thinking_blocks"
# in assistant messages, ignoring "reasoning_content".  When the gateway injects
# cached reasoning_content for multi-turn continuity, liteLLM drops it during
# OpenAI→Anthropic message conversion.  Intercept the function to synthesize
# thinking_blocks from reasoning_content before liteLLM processes the messages.
try:
    from litellm.llms.prompt_templates import factory as _pt_factory
    _original_anthropic_messages_pt = _pt_factory.anthropic_messages_pt

    def _patched_anthropic_messages_pt(messages, model, llm_provider):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                if not msg.get("thinking_blocks") and msg.get("reasoning_content"):
                    msg["thinking_blocks"] = [
                        {"type": "thinking", "thinking": msg["reasoning_content"]}
                    ]
        return _original_anthropic_messages_pt(messages, model, llm_provider)

    _pt_factory.anthropic_messages_pt = _patched_anthropic_messages_pt
except Exception:
    logging.getLogger("llmgw.app").warning(
        "liteLLM monkey-patch for anthropic_messages_pt (reasoning_content→thinking) failed"
    )

litellm.drop_params = False  # Allow provider-specific params like DeepSeek's 'thinking'
litellm.add_function_to_prompt = False

OPENAI_HOSTS = ("api.openai.com", "azure.com")

# Minimum max_tokens for requests containing images, to accommodate thinking/reasoning
# tokens that consume the budget before visible content is generated.
MIN_IMAGE_MAX_TOKENS = get_default("min_image_max_tokens", 2000)


def _extract_image_data_uris(content) -> list:
    """Extract image data URIs from string content, returning list of (mime_type, data_uri)."""
    if not isinstance(content, str):
        return []
    # Match data:image/<type>;base64,<data>
    pattern = r'data:image/(\w+);base64,([A-Za-z0-9+/=]+)'
    matches = re.findall(pattern, content)
    result = []
    for mime_subtype, data in matches:
        if len(data) > 100:  # Minimum length to be a real image
            result.append((f'image/{mime_subtype}', f'data:image/{mime_subtype};base64,{data}'))
    return result


def _normalize_image_content(messages: list) -> list:
    """Convert image data URIs found in text content to proper image_url content parts.

    OpenCode and other agent SDKs may include image data as data URIs within text
    content (e.g. from file-read tools) rather than as image_url content parts.
    Without image_url parts, multimodal models won't activate their vision encoder.
    """
    fixed = False
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    image_uris = _extract_image_data_uris(text)
                    if image_uris:
                        fixed = True
                        # Remove the data URI from text to avoid duplication
                        cleaned = re.sub(
                            r'data:image/\w+;base64,[A-Za-z0-9+/=]{100,}',
                            '',
                            text
                        ).strip()
                        if cleaned:
                            new_parts.append({"type": "text", "text": cleaned})
                        for mime_type, data_uri in image_uris:
                            new_parts.append({
                                "type": "image_url",
                                "image_url": {"url": data_uri}
                            })
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            msg["content"] = new_parts
        elif isinstance(content, str):
            image_uris = _extract_image_data_uris(content)
            if image_uris:
                fixed = True
                cleaned = re.sub(
                    r'data:image/\w+;base64,[A-Za-z0-9+/=]{100,}',
                    '',
                    content
                ).strip()
                new_parts = []
                if cleaned:
                    new_parts.append({"type": "text", "text": cleaned})
                for mime_type, data_uri in image_uris:
                    new_parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    })
                msg["content"] = new_parts
    if fixed:
        logging.getLogger("llmgw.app").info("Normalized image data URIs to image_url content parts")
    return messages


def _has_image_content(messages: list) -> bool:
    """Check if any message contains image content (including nested in tool_result)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in ("image_url", "input_image", "image"):
                        return True
                    # Anthropic tool_result may contain nested image blocks
                    if part.get("type") == "tool_result":
                        inner = part.get("content")
                        if isinstance(inner, list):
                            for ip in inner:
                                if isinstance(ip, dict) and ip.get("type") == "image":
                                    return True
                        elif isinstance(inner, str):
                            if _extract_image_data_uris(inner):
                                return True
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        if _extract_image_data_uris(part["text"]):
                            return True
        elif isinstance(content, str):
            if _extract_image_data_uris(content):
                return True
    return False


def get_litellm_model_name(model: str, provider: dict) -> str:
    """Build the correct model name for liteLLM based on provider type."""
    provider_type = provider.get("provider_type", "openai")
    provider_id = provider.get("id", "")
    api_base = provider.get("api_base", "")

    # 提取纯模型名（parse_model_id 自动处理简单/复合两种格式）
    from app.database import parse_model_id
    model = parse_model_id(model).model_name

    if provider_type == "openai":
        if api_base and not any(host in api_base for host in OPENAI_HOSTS):
            return f"openai/{model}"
        return model
    elif provider_type == "anthropic":
        return f"anthropic/{model}"
    else:
        return model


def build_completion_args(model: str, provider_id: Optional[str] = None) -> tuple[str, dict[str, Any]]:
    provider = get_provider(provider_id) if provider_id else find_provider_by_model(model)
    if not provider:
        raise ValueError(f"No provider found for model '{model}'")
    if not provider.get("enabled"):
        raise ValueError(f"Provider '{provider['id']}' is disabled")

    params: dict[str, Any] = {"api_key": provider.get("api_key") or "sk-no-auth"}
    api_base = provider.get("api_base", "").rstrip("/")
    if api_base:
        # Anthropic-compatible providers expect the official API path /v1/messages
        # appended to the configured base URL (consistent with passthrough behavior).
        if provider.get("provider_type") == "anthropic" and not api_base.endswith(("/v1/messages", "/messages")):
            params["api_base"] = api_base + "/v1/messages"
        else:
            params["api_base"] = api_base

    litellm_model = get_litellm_model_name(model, provider)
    # Per-provider thinking mode: configured via provider.extra_headers.
    # If not set, no thinking parameter is sent (each provider defaults).
    extra_headers = provider.get("extra_headers", {}) or {}
    thinking = extra_headers.get("thinking")
    if thinking in ("enabled", "disabled"):
        params.setdefault("extra_body", {})
        params["extra_body"]["thinking"] = {"type": thinking}
    get_logger("app").debug("route model=%s provider_type=%s api_base=%s -> litellm_model=%s",
                           model, provider.get("provider_type"), api_base, litellm_model)
    return litellm_model, params


def clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove None values and provider-unsafe params that cause 400 errors."""
    cleaned = {key: value for key, value in params.items() if value is not None}
    # Empty extra_body dictionary is rejected by some providers
    if isinstance(cleaned.get("extra_body"), dict) and not cleaned["extra_body"]:
        cleaned.pop("extra_body")
    return cleaned


def create_chat_completion(
    model: str,
    messages: list,
    provider_id: Optional[str] = None,
    **kwargs
) -> dict:
    litellm_model, extra_params = build_completion_args(model, provider_id)
    kwargs.update(extra_params)
    _normalize_image_content(messages)
    if _has_image_content(messages):
        kwargs["max_tokens"] = max(kwargs.get("max_tokens", 0), MIN_IMAGE_MAX_TOKENS)
    response = completion(model=litellm_model, messages=messages, **clean_params(kwargs))
    return response


def create_chat_completion_stream(
    model: str,
    messages: list,
    provider_id: Optional[str] = None,
    **kwargs
):
    litellm_model, extra_params = build_completion_args(model, provider_id)
    kwargs.update(extra_params)
    kwargs["stream"] = True
    if "stream_options" not in kwargs:
        kwargs["stream_options"] = {"include_usage": True}
    _normalize_image_content(messages)
    if _has_image_content(messages):
        kwargs["max_tokens"] = max(kwargs.get("max_tokens", 0), MIN_IMAGE_MAX_TOKENS)
    return completion(model=litellm_model, messages=messages, **clean_params(kwargs))


def create_completion(
    model: str,
    prompt: str,
    provider_id: Optional[str] = None,
    **kwargs
) -> dict:
    litellm_model, extra_params = build_completion_args(model, provider_id)
    kwargs.update(extra_params)
    messages = [{"role": "user", "content": prompt}]
    _normalize_image_content(messages)
    if _has_image_content(messages):
        kwargs["max_tokens"] = max(kwargs.get("max_tokens", 0), MIN_IMAGE_MAX_TOKENS)
    response = completion(model=litellm_model, messages=messages, **clean_params(kwargs))
    return response


def create_completion_stream(
    model: str,
    prompt: str,
    provider_id: Optional[str] = None,
    **kwargs
):
    litellm_model, extra_params = build_completion_args(model, provider_id)
    kwargs.update(extra_params)
    kwargs["stream"] = True
    if "stream_options" not in kwargs:
        kwargs["stream_options"] = {"include_usage": True}
    messages = [{"role": "user", "content": prompt}]
    _normalize_image_content(messages)
    if _has_image_content(messages):
        kwargs["max_tokens"] = max(kwargs.get("max_tokens", 0), MIN_IMAGE_MAX_TOKENS)
    return completion(model=litellm_model, messages=messages, **clean_params(kwargs))


def get_available_models(provider_id: Optional[str] = None) -> list:
    models = []
    if provider_id:
        providers = [get_provider(provider_id)]
    else:
        providers = get_providers()

    for provider in providers:
        if provider and provider.get("enabled"):
            for model in provider.get("models", []):
                if model.get("enabled"):
                    models.append({
                        "id": f"{provider['id']}/{model['id']}",
                        "name": model.get("name", model["id"]),
                        "provider": provider["id"],
                        "provider_name": provider["name"],
                        "provider_type": provider["provider_type"]
                    })
    return models
