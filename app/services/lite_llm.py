import json
import logging
import litellm
from litellm import completion
from typing import Optional, Any
from pydantic import Field
from litellm.types.utils import ModelResponse, Message, Delta
from app.database import get_providers, get_provider, find_provider_by_model
from app.services.logger import get_logger
from app.config import get_default
from app.core.images import has_image_content, normalize_image_content

# -- liteLLM compatibility: expose reasoning_content on response models --
# Several OpenAI-compatible providers return reasoning_content, but some liteLLM
# versions do not include it on Message/Delta. The policy layer needs the field
# for multi-turn reasoning continuity.
try:
    for _model in (Message, Delta):
        if "reasoning_content" not in _model.model_fields:
            _model.model_fields["reasoning_content"] = Field(default=None)
            _model.model_rebuild(force=True)
except Exception:
    logging.getLogger("llmgw.app").warning(
        "liteLLM compatibility patch for reasoning_content fields failed"
    )

# -- liteLLM compatibility: preserve reasoning_content in responses --
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
            # Inject cache stats into usage from raw response
            if hasattr(result, "usage") and result.usage:
                usage_raw = response_object.get("usage", {})
                if isinstance(usage_raw, dict):
                    hit = usage_raw.get("prompt_cache_hit_tokens")
                    miss = usage_raw.get("prompt_cache_miss_tokens")
                    if hit is not None:
                        result.usage.prompt_cache_hit_tokens = hit
                    if miss is not None:
                        result.usage.prompt_cache_miss_tokens = miss
        return result

    _litellm_utils.convert_to_model_response_object = _patched_convert
except Exception:
    logging.getLogger("llmgw.app").warning(
        "liteLLM compatibility patch for preserving reasoning_content failed"
    )

litellm.drop_params = False  # Allow provider-specific params like DeepSeek's 'thinking'
litellm.add_function_to_prompt = False

OPENAI_HOSTS = ("api.openai.com", "azure.com")

# Minimum max_tokens for requests containing images, to accommodate thinking/reasoning
# tokens that consume the budget before visible content is generated.
MIN_IMAGE_MAX_TOKENS = get_default("min_image_max_tokens", 2000)


def get_litellm_model_name(model: str, provider: dict) -> str:
    """Build the liteLLM model name for OpenAI-compatible providers."""
    provider_type = provider.get("provider_type", "openai")
    api_base = provider.get("api_base", "")
    if provider_type != "openai":
        raise ValueError("liteLLM adapter only supports OpenAI-compatible providers")

    # Extract the plain model name; parse_model_id handles simple and composite formats
    from app.database import parse_model_id
    model = parse_model_id(model).model_name

    if api_base and not any(host in api_base for host in OPENAI_HOSTS):
        return f"openai/{model}"
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
        params["api_base"] = api_base

    litellm_model = get_litellm_model_name(model, provider)
    # Per-provider thinking mode: configured via provider.extra_headers.
    # If not set, no thinking parameter is sent (each provider defaults).
    extra_headers = provider.get("extra_headers", {}) or {}
    if isinstance(extra_headers, str):
        try:
            extra_headers = json.loads(extra_headers)
        except (json.JSONDecodeError, TypeError):
            extra_headers = {}
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
    normalize_image_content(messages)
    if has_image_content(messages):
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
    normalize_image_content(messages)
    if has_image_content(messages):
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
