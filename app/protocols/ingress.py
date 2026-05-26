from typing import Any

from app.config import get_default
from app.core.types import InternalRequest, tools_from_chat
from app.protocols.ir import (
    anthropic_messages_to_ir,
    openai_messages_to_ir,
    responses_input_to_ir,
)
from app.core.text import strip_billing_header


def _max_tokens_from_body(body: dict[str, Any]) -> int:
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = get_default("max_tokens", 16384)
    return max_tokens


def _temperature_from_body(body: dict[str, Any]):
    if "temperature" in body:
        return body.get("temperature")
    return get_default("temperature", 0.7)


def _completion_prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        return "\n".join(str(item) for item in prompt)
    if prompt is None:
        return ""
    return str(prompt)


def responses_tools_to_chat_tools(tools: list[Any] | None) -> list[dict[str, Any]]:
    converted = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if "function" in tool:
            converted.append(tool)
            continue
        if tool.get("type") != "function":
            continue
        function_fields = {
            key: value
            for key, value in tool.items()
            if key not in ("type", "strict", "additionalProperties")
        }
        params = function_fields.get("parameters")
        if isinstance(params, dict):
            params = dict(params)
            params.pop("additionalProperties", None)
            props = params.get("properties", {})
            if isinstance(props, dict):
                cleaned = {}
                for key, prop in props.items():
                    if isinstance(prop, dict):
                        prop = dict(prop)
                        prop.pop("additionalProperties", None)
                    cleaned[key] = prop
                params["properties"] = cleaned
            function_fields["parameters"] = params
        converted.append({"type": "function", "function": function_fields})
    return converted


def chat_completions_to_internal(body: dict[str, Any]) -> InternalRequest:
    model = body.get("model")
    extra_keys = {"top_p", "presence_penalty", "frequency_penalty", "stop", "response_format", "user"}
    extra = {key: body[key] for key in extra_keys if key in body}
    tool_choice = body.get("tool_choice")
    if tool_choice is not None:
        extra["tool_choice"] = tool_choice
    messages = openai_messages_to_ir(body.get("messages", []))
    return InternalRequest(
        endpoint="chat_completions",
        requested_model=model,
        target_model=model,
        messages=messages,
        provider_id=body.get("provider_id", ""),
        tools=tools_from_chat(body.get("tools")),
        tool_choice=tool_choice,
        stream=body.get("stream", False),
        temperature=_temperature_from_body(body),
        max_tokens=_max_tokens_from_body(body),
        previous_response_id=body.get("previous_response_id") or "",
        extra=extra,
        raw_body=body,
    )


def completions_to_internal(body: dict[str, Any]) -> InternalRequest:
    model = body.get("model")
    prompt = _completion_prompt_to_text(body.get("prompt", ""))
    extra_keys = {"top_p", "presence_penalty", "frequency_penalty", "stop", "suffix", "echo", "logprobs", "user"}
    extra = {key: body[key] for key in extra_keys if key in body}
    messages = openai_messages_to_ir([{"role": "user", "content": prompt}])
    return InternalRequest(
        endpoint="completions",
        requested_model=model,
        target_model=model,
        messages=messages,
        provider_id=body.get("provider_id", ""),
        stream=body.get("stream", False),
        temperature=_temperature_from_body(body),
        max_tokens=_max_tokens_from_body(body),
        extra=extra,
        raw_body=body,
        metadata={"prompt": prompt},
    )


def anthropic_messages_to_internal(body: dict[str, Any]) -> InternalRequest:
    model = body.get("model")
    system_prompt = strip_billing_header(body.get("system", ""))
    anthropic_messages = body.get("messages", [])
    messages = anthropic_messages_to_ir(anthropic_messages, system_prompt)

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") in ("auto",):
        tool_choice = None
    elif tool_choice == "auto":
        tool_choice = None

    return InternalRequest(
        endpoint="messages",
        requested_model=model,
        target_model=model,
        messages=messages,
        provider_id=body.get("provider_id", ""),
        system="",
        tools=tools_from_chat(body.get("tools")),
        tool_choice=tool_choice,
        stream=body.get("stream", False),
        temperature=_temperature_from_body(body),
        max_tokens=_max_tokens_from_body(body),
        previous_response_id=body.get("previous_response_id") or "",
        raw_body=body,
        metadata={"anthropic_input_count": len(anthropic_messages)},
    )


def responses_to_internal(body: dict[str, Any]) -> InternalRequest:
    model = body.get("model")
    input_data = body.get("input", "")
    instructions = body.get("instructions", "")
    messages = responses_input_to_ir(input_data, instructions)

    extra_keys = {"top_p", "presence_penalty", "frequency_penalty", "stop", "response_format", "user"}
    extra = {key: body[key] for key in extra_keys if key in body}
    if body.get("previous_response_id"):
        extra["previous_response_id"] = body.get("previous_response_id")
    tool_choice = body.get("tool_choice")
    if tool_choice is not None:
        extra["tool_choice"] = tool_choice
    if tool_choice == "auto":
        tool_choice = None
        extra.pop("tool_choice", None)

    converted_tools = responses_tools_to_chat_tools(body.get("tools", [])) if isinstance(body.get("tools"), list) else []
    if converted_tools:
        extra["tools"] = converted_tools

    return InternalRequest(
        endpoint="responses",
        requested_model=model,
        target_model=model,
        messages=messages,
        provider_id=body.get("provider_id", ""),
        system=instructions,
        tools=tools_from_chat(converted_tools),
        tool_choice=tool_choice,
        stream=body.get("stream", False),
        temperature=_temperature_from_body(body),
        max_tokens=_max_tokens_from_body(body),
        previous_response_id=body.get("previous_response_id") or "",
        extra=extra,
        raw_body=body,
        metadata={"input_is_list": isinstance(input_data, list), "instructions": instructions},
    )
