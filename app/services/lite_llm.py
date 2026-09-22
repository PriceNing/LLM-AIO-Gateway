import json
import logging
import litellm
from litellm import completion
from typing import Optional, Any
from pydantic import Field
from litellm.types.utils import ModelResponse, Message, Delta
from app.database import get_providers, get_provider, find_provider_by_model, get_model_stored_capabilities, parse_model_id
from app.services.logger import get_logger
from app.config import get_default
from app.core.images import has_image_content, normalize_image_content
from app.core.model_capabilities import resolve_model_capabilities
from app.services.model_registry import registry_lookup

# -- liteLLM compatibility: expose reasoning_content on response models --
# Several OpenAI-compatible providers return reasoning_content, but some liteLLM
# versions do not include it on Message/Delta. The policy layer needs the field
# for multi-turn reasoning continuity.
_PATCH_STATE = {"fields": False, "converter": False}

try:
    for _model in (Message, Delta):
        if "reasoning_content" not in _model.model_fields:
            _model.model_fields["reasoning_content"] = Field(default=None)
            _model.model_rebuild(force=True)
    _PATCH_STATE["fields"] = all("reasoning_content" in m.model_fields for m in (Message, Delta))
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
    _PATCH_STATE["converter"] = (
        getattr(_litellm_utils, "convert_to_model_response_object", None) is _patched_convert
    )
except Exception:
    logging.getLogger("llmgw.app").warning(
        "liteLLM compatibility patch for preserving reasoning_content failed"
    )


def compatibility_patch_status() -> dict:
    """Report whether the liteLLM compatibility patches are still in place.

    补丁依赖 litellm 内部函数名，升级后可能静默失效导致 reasoning_content
    丢失、多轮 thinking 断裂，因此必须可观测（Q3）。
    """
    return dict(_PATCH_STATE)


def _log_compatibility_patch_state() -> None:
    logger = logging.getLogger("llmgw.app")
    if _PATCH_STATE["fields"] and _PATCH_STATE["converter"]:
        logger.debug("liteLLM compatibility patches active")
        return
    logger.warning(
        "liteLLM compatibility patches INCOMPLETE: %s; reasoning_content continuity "
        "and prompt-cache usage may be dropped. litellm version=%s",
        _PATCH_STATE,
        getattr(litellm, "__version__", "unknown"),
    )


_log_compatibility_patch_state()

litellm.drop_params = False  # Allow provider-specific params like DeepSeek's 'thinking'
litellm.add_function_to_prompt = False
# Cap liteLLM's global request timeout. Individual provider timeouts are passed
# per-call but liteLLM's own default (6000 s) governs the connect phase and can
# cause multi-minute hangs when an upstream is unreachable.
litellm.request_timeout = get_default("litellm_request_timeout", 120)

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
    params["timeout"] = provider.get("request_timeout", 120)
    params["num_retries"] = provider.get("retry_count", 0)

    litellm_model = get_litellm_model_name(model, provider)
    # 不再向 litellm.model_cost 注入 supports_vision 占位条目：
    # liteLLM 1.83.14 只在 anthropic / gemini 的 prompt_factory 分支读 supports_vision()，
    # openai/ 前缀直接透传 messages 不做本地视觉校验；实测注册与否行为一致。
    # 保留该注入反而会往 model_cost 里塞入缺字段条目（max_output_tokens=None 等）。
    # Provider options affect protocol payloads; upstream headers affect HTTP
    # transport only. Keep these configuration channels deliberately separate.
    provider_options = provider.get("provider_options", {}) or {}
    if isinstance(provider_options, str):
        try:
            provider_options = json.loads(provider_options)
        except (json.JSONDecodeError, TypeError):
            provider_options = {}
    thinking = provider_options.get("thinking")
    if thinking in ("enabled", "disabled"):
        params.setdefault("extra_body", {})
        params["extra_body"]["thinking"] = {"type": thinking}
    upstream_headers = provider.get("upstream_headers", {}) or {}
    if isinstance(upstream_headers, str):
        try:
            upstream_headers = json.loads(upstream_headers)
        except (json.JSONDecodeError, TypeError):
            upstream_headers = {}
    transport_headers = {
        str(key): str(value)
        for key, value in upstream_headers.items()
        if value not in (None, "")
    }
    if transport_headers:
        params["extra_headers"] = transport_headers
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


def model_temperature_locks(model: str, provider_id: Optional[str] = None) -> dict[str, Any]:
    """解析该模型的 temperature 锁能力。

    来源顺序与 /v1/models 一致：内置家族表 < 在线注册表 < 已存储（上游透传 + 管理员覆盖）。
    provider_models 行不存在时仍会回退到家族表，因此行为不依赖数据库里是否有该模型行。
    """
    mid = parse_model_id(model)
    name = mid.model_name
    resolved_provider = provider_id or mid.provider_id
    stored = get_model_stored_capabilities(resolved_provider, name) if resolved_provider else {}
    remote = registry_lookup(name, name)
    return resolve_model_capabilities(
        {"id": name, "name": name, "capabilities": stored}, remote=remote
    )


def apply_temperature_lock(caps: dict[str, Any], kwargs: dict[str, Any]) -> None:
    """把模型不接受的 temperature 归一为它接受的值。

    纯函数（能力入参、kwargs 出参），条件里不出现任何模型/厂商名称：是否锁、
    锁到多少完全来自能力数据（见 core/model_capabilities.py 的 fixed_temperature）。
    """
    if kwargs.get("temperature") is None:
        return
    locked = caps.get("fixed_temperature")
    if locked is None and kwargs.get("reasoning_effort") not in (None, "none"):
        locked = caps.get("fixed_temperature_with_reasoning")
    if locked is None:
        return
    try:
        current, target = float(kwargs["temperature"]), float(locked)
    except (TypeError, ValueError):
        return
    if current == target:
        return
    kwargs["temperature"] = int(target) if target.is_integer() else target
    get_logger("app").debug(
        "Locking temperature=%s to %s by model capability fixed_temperature",
        current, kwargs["temperature"],
    )


def _forced_tool_choice(tool_choice: Any) -> bool:
    if tool_choice in ("required", "none"):
        return True
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type") or "")
        return choice_type in {"function", "tool", "required", "none"} or bool(tool_choice.get("name") or tool_choice.get("function"))
    return False


def _disable_thinking_for_missing_reasoning(kwargs: dict[str, Any]) -> None:
    if not kwargs.pop("disable_thinking_for_missing_reasoning", False):
        return
    extra_body = kwargs.get("extra_body")
    if not isinstance(extra_body, dict):
        return
    thinking = extra_body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        extra_body["thinking"] = {"type": "disabled"}


def _disable_thinking_when_tools_forced(kwargs: dict[str, Any]) -> None:
    """DeepSeek historically rejected forced tool_choice while thinking was enabled.

    Official DeepSeek V3.2+ docs allow tools together with thinking. Only disable
    thinking for an explicitly forced tool_choice, not merely because tools exist.
    """
    if not kwargs.get("tools") or not _forced_tool_choice(kwargs.get("tool_choice")):
        return
    extra_body = kwargs.get("extra_body")
    if not isinstance(extra_body, dict):
        return
    thinking = extra_body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        extra_body["thinking"] = {"type": "disabled"}


def _merge_system_contents(contents: list) -> Any:
    if not contents:
        return ""
    if all(isinstance(item, str) for item in contents):
        return "\n\n".join(item for item in contents if item)
    merged: list = []
    for item in contents:
        if isinstance(item, list):
            merged.extend(item)
        elif isinstance(item, str) and item:
            merged.append({"type": "text", "text": item})
        elif item not in (None, ""):
            merged.append(item)
    return merged or ""


def _system_messages_first(messages: list) -> list:
    """Guarantee a single leading system message for llama.cpp/Qwen templates."""
    if not isinstance(messages, list):
        return messages
    systems = [item for item in messages if isinstance(item, dict) and item.get("role") == "system"]
    if not systems:
        return messages
    conversation = [item for item in messages if not (isinstance(item, dict) and item.get("role") == "system")]
    content = _merge_system_contents([item.get("content") for item in systems])
    return [{"role": "system", "content": content}] + conversation


def create_chat_completion(
    model: str,
    messages: list,
    provider_id: Optional[str] = None,
    **kwargs
) -> dict:
    litellm_model, extra_params = build_completion_args(model, provider_id)
    kwargs.update(extra_params)
    apply_temperature_lock(model_temperature_locks(model, provider_id), kwargs)
    _disable_thinking_for_missing_reasoning(kwargs)
    _disable_thinking_when_tools_forced(kwargs)
    messages = _system_messages_first(messages)
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
    apply_temperature_lock(model_temperature_locks(model, provider_id), kwargs)
    _disable_thinking_for_missing_reasoning(kwargs)
    _disable_thinking_when_tools_forced(kwargs)
    kwargs["stream"] = True
    if "stream_options" not in kwargs:
        kwargs["stream_options"] = {"include_usage": True}
    messages = _system_messages_first(messages)
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
                    resolved_caps = resolve_model_capabilities(model, remote=registry_lookup(model["id"], model.get("name", "")))
                    models.append({
                        "id": f"{provider['id']}/{model['id']}",
                        "name": model.get("name", model["id"]),
                        "provider": provider["id"],
                        "provider_name": provider["name"],
                        "provider_type": provider["provider_type"],
                        # 内置家族表 < 在线注册表 < 上游透传 < 管理员覆盖。
                        "capabilities": resolved_caps,
                        # 与实际生效值取交集：历史脏行的 admin_keys 可能指向
                        # 已被读取侧归一化丢弃的键，不得在面板上虚报"已覆盖"。
                        "capabilities_overridden": sorted(
                            k for k in ((model.get("capabilities") or {}).get("admin_keys") or [])
                            if k in resolved_caps
                        ),
                    })
    return models
