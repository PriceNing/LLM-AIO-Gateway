import copy
import json
from typing import Any

from app.config import get_default
from app.services.logger import get_logger

_app_log = get_logger("app")
from app.core.types import InternalRequest, tools_from_chat
from app.protocols.ir import (
    anthropic_messages_to_ir,
    openai_messages_to_ir,
    responses_input_to_ir,
)
from app.core.text import strip_billing_header


_INCOMPLETE_JSON_PREFIXES = ("{", "[", '"')


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


def _strip_schema_extra_fields(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    cleaned = {key: _strip_schema_extra_fields(value) for key, value in schema.items() if key != "additionalProperties"}
    return cleaned


def _custom_tool_argument_field(tool: dict[str, Any]) -> str:
    name = str(tool.get("name") or "")
    if name == "apply_patch":
        return "patch"
    return "input"


def _custom_tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    argument_field = _custom_tool_argument_field(tool)
    return {
        "type": "object",
        "properties": {
            argument_field: {
                "type": "string",
                "description": "Raw custom tool input.",
            }
        },
        "required": [argument_field],
    }


def _namespace_tool_name(namespace: str, name: str) -> str:
    """Return the Chat-facing name for a Responses namespace function.

    Keep the original function name so Chat Completions models continue to see
    Codex tools such as spawn_agent or imagegen. Namespace is restored on egress
    through responses_namespace_tools, not by rewriting the tool identity.
    """
    return name


def _namespace_map_key(namespace: str, name: str) -> str:
    if namespace and name:
        return f"{namespace}.{name}"
    return name


def responses_tools_to_chat_tools(tools: list[Any] | None) -> list[dict[str, Any]]:
    converted = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if "function" in tool:
            converted.append(tool)
            continue
        tool_type = tool.get("type")
        if tool_type == "namespace":
            namespace = str(tool.get("name") or "")
            for sub_tool in tool.get("tools") or []:
                if isinstance(sub_tool, dict) and sub_tool.get("type") == "function":
                    converted.append(_responses_function_tool_to_chat_tool(
                        sub_tool,
                        override_name=_namespace_tool_name(namespace, str(sub_tool.get("name") or "")),
                    ))
            continue
        if tool_type == "custom":
            name = str(tool.get("name") or "")
            if not name:
                continue
            converted.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description") or "Provide the raw custom tool input.",
                    "parameters": _custom_tool_parameters(tool),
                },
            })
            continue
        if tool_type != "function":
            continue
        converted.append(_responses_function_tool_to_chat_tool(tool))
    return converted


def _responses_function_tool_to_chat_tool(tool: dict[str, Any], *, override_name: str | None = None) -> dict[str, Any]:
    function_fields = {
        key: copy.deepcopy(value)
        for key, value in tool.items()
        if key not in ("type", "strict", "additionalProperties")
    }
    if override_name:
        function_fields["name"] = override_name
    params = function_fields.get("parameters")
    if isinstance(params, dict):
        function_fields["parameters"] = _strip_schema_extra_fields(params)
    return {"type": "function", "function": function_fields}


def _responses_request_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    tools = []
    if isinstance(body.get("tools"), list):
        tools.extend(tool for tool in body["tools"] if isinstance(tool, dict))
    input_data = body.get("input")
    if isinstance(input_data, list):
        tools.extend(_responses_additional_tools_from_input(input_data))
    return tools


def responses_tool_maps(tools: list[Any] | None) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    namespace_tools: dict[str, dict[str, str]] = {}
    custom_tools: dict[str, dict[str, str]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        name = str(tool.get("name") or "")
        if tool_type == "namespace" and name:
            for sub_tool in tool.get("tools") or []:
                if not isinstance(sub_tool, dict) or sub_tool.get("type") != "function":
                    continue
                sub_name = str(sub_tool.get("name") or "")
                if sub_name:
                    mapped = {"namespace": name, "name": sub_name}
                    namespace_tools[sub_name] = mapped
                    namespace_tools[_namespace_map_key(name, sub_name)] = mapped
                    namespace_tools[f"{name}-{sub_name}"] = mapped
        elif tool_type == "custom" and name:
            custom_tools[name] = {"name": name, "argument_field": _custom_tool_argument_field(tool)}
        elif tool_type == "function" and name == "apply_patch":
            custom_tools[name] = {"name": name, "argument_field": _single_string_parameter_name(tool) or _custom_tool_argument_field(tool)}
    return namespace_tools, custom_tools


def _single_string_parameter_name(tool: dict[str, Any]) -> str | None:
    params = tool.get("parameters")
    if not isinstance(params, dict):
        return None
    props = params.get("properties")
    if not isinstance(props, dict) or len(props) != 1:
        return None
    name, schema = next(iter(props.items()))
    if isinstance(schema, dict) and schema.get("type") == "string":
        return str(name)
    return None


def custom_tool_input_from_arguments(arguments: Any, argument_field: str) -> str:
    if isinstance(arguments, dict):
        value = arguments.get(argument_field)
        return value if isinstance(value, str) else json.dumps(arguments, ensure_ascii=False)
    raw = str(arguments or "")
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        if raw.lstrip().startswith(_INCOMPLETE_JSON_PREFIXES):
            return ""
        return raw
    if isinstance(parsed, dict) and isinstance(parsed.get(argument_field), str):
        return parsed[argument_field]
    return raw


def _responses_additional_tools_from_input(input_data: Any) -> list[dict[str, Any]]:
    additional_tools = []
    if not isinstance(input_data, list):
        return additional_tools

    for item in input_data:
        if not isinstance(item, dict) or item.get("type") != "additional_tools":
            continue
        tools = item.get("tools")
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if isinstance(tool, dict):
                additional_tools.append(dict(tool))
    return additional_tools


def chat_completions_to_internal(body: dict[str, Any]) -> InternalRequest:
    model = body.get("model")
    _app_log.debug("[ingress] chat_completions model=%s stream=%s msgs=%d tools=%s",
                   model, body.get("stream"), len(body.get("messages", [])), bool(body.get("tools")))
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
    _app_log.debug("[ingress] completions model=%s stream=%s prompt_len=%d",
                   model, body.get("stream"), len(prompt))
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
    _app_log.debug("[ingress] messages model=%s stream=%s msgs=%d tools=%s",
                   model, body.get("stream"), len(body.get("messages", [])), bool(body.get("tools")))
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
    input_count = len(input_data) if isinstance(input_data, list) else 0
    _app_log.debug("[ingress] responses model=%s stream=%s input_items=%d instructions_len=%d",
                   model, body.get("stream"), input_count, len(instructions) if instructions else 0)
    request_tools = _responses_request_tools(body)
    namespace_tools, custom_tools = responses_tool_maps(request_tools)
    messages = responses_input_to_ir(input_data, instructions, custom_tools=custom_tools)

    # Only Chat-compatible fields go in extra. The full cleaned body is retained
    # separately for a native Responses upstream and must never leak into Chat.
    extra_keys = {"top_p", "presence_penalty", "frequency_penalty", "stop", "response_format", "user", "parallel_tool_calls"}
    extra = {key: body[key] for key in extra_keys if key in body}

    if namespace_tools:
        extra["responses_namespace_tools"] = namespace_tools
    if custom_tools:
        extra["responses_custom_tools"] = custom_tools

    tool_choice = body.get("tool_choice")
    if tool_choice is not None:
        extra["tool_choice"] = tool_choice
    if tool_choice == "auto":
        tool_choice = None
        extra.pop("tool_choice", None)

    converted_tools = responses_tools_to_chat_tools(request_tools)
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
        metadata={"input_is_list": isinstance(input_data, list), "instructions": instructions, "responses_native": {"request_body": copy.deepcopy(body)}},
    )
