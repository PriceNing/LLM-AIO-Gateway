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


def thinking_fields_from_body(body: dict[str, Any] | None) -> dict[str, Any]:
    """Whitelist thinking controls from Chat, Completions, Messages, or Responses.

    Chat clients send top-level reasoning_effort. Responses clients such as
    Codex send nested reasoning.effort. Both must land in InternalRequest.extra
    and request-log details without leaking the rest of raw_body.
    """
    if not isinstance(body, dict):
        return {}
    fields: dict[str, Any] = {}
    for key in ("reasoning_effort", "chat_template_kwargs", "enable_thinking"):
        if key in body:
            fields[key] = body[key]
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and "reasoning_effort" not in fields:
        effort = reasoning.get("effort")
        if effort is not None and effort != "":
            fields["reasoning_effort"] = effort
    if "enable_thinking" not in fields:
        template_kwargs = fields.get("chat_template_kwargs")
        if isinstance(template_kwargs, dict) and "enable_thinking" in template_kwargs:
            fields["enable_thinking"] = template_kwargs["enable_thinking"]
        elif "reasoning_effort" in fields:
            fields["enable_thinking"] = str(fields["reasoning_effort"]).lower() not in (
                "", "none", "off", "false", "0"
            )
    return fields


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


def _chat_function_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """Make only the function-schema root Chat-compatible.

    Keep nested JSON Schema semantics intact.  Some OpenAI-compatible
    providers reject a missing root ``required`` as though it were null, and
    require the root itself to declare ``type: object`` instead of an
    ``anyOf``/``oneOf`` of object variants.
    """
    cleaned = _strip_schema_extra_fields(copy.deepcopy(schema))

    def resolve_ref(node: dict[str, Any]) -> dict[str, Any] | None:
        ref = node.get("$ref")
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return None
        target: Any = cleaned
        for part in ref[2:].split("/"):
            if not isinstance(target, dict):
                return None
            target = target.get(part.replace("~1", "/").replace("~0", "~"))
        return target if isinstance(target, dict) else None

    def root_properties(node: Any, seen: set[int] | None = None) -> dict[str, Any]:
        """Collect properties through root combiners without walking property values."""
        if not isinstance(node, dict):
            return {}
        seen = seen or set()
        marker = id(node)
        if marker in seen:
            return {}
        seen.add(marker)

        properties: dict[str, Any] = {}

        def merge(incoming: dict[str, Any]) -> None:
            for name, definition in incoming.items():
                definition = copy.deepcopy(definition)
                current = properties.get(name)
                if current is None:
                    properties[name] = definition
                elif current != definition:
                    variants = current.get("anyOf") if isinstance(current, dict) and set(current) == {"anyOf"} else None
                    if not isinstance(variants, list):
                        variants = [current]
                    if definition not in variants:
                        properties[name] = {"anyOf": [*variants, definition]}

        if isinstance(node.get("properties"), dict):
            merge(node["properties"])
        referenced = resolve_ref(node)
        if referenced is not None:
            merge(root_properties(referenced, seen))
        for combiner in ("allOf", "anyOf", "oneOf"):
            branches = node.get(combiner)
            if isinstance(branches, list):
                for branch in branches:
                    merge(root_properties(branch, seen))
        return properties

    # A JSON Schema may declare ``type: object`` and still put the actual
    # variants in a root anyOf/oneOf.  Chat providers validate that union as
    # the function root and reject it, so root combiners take precedence over
    # the otherwise-valid object fast path.
    has_root_union = any(isinstance(cleaned.get(key), list) for key in ("anyOf", "oneOf"))
    if has_root_union:
        result = {
            key: copy.deepcopy(value)
            for key, value in cleaned.items()
            if key not in ("type", "properties", "required", "anyOf", "oneOf", "allOf", "additionalProperties")
        }
        result.update({"type": "object", "properties": root_properties(cleaned), "required": []})
        return result

    if cleaned.get("type") == "object" or isinstance(cleaned.get("properties"), dict):
        cleaned["type"] = "object"
        if not isinstance(cleaned.get("required"), list):
            cleaned["required"] = []
        return cleaned

    # Function arguments must have an object root in Chat Completions.  This
    # also handles a root $ref/allOf whose referenced schema omits an explicit
    # object type.
    properties = root_properties(cleaned)
    result = {
        key: copy.deepcopy(value)
        for key, value in cleaned.items()
        if key not in ("type", "properties", "required", "$ref", "allOf", "additionalProperties")
    }
    result.update({"type": "object", "properties": properties, "required": []})
    return result


def _normalize_chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(tool)
    function = normalized.get("function")
    if isinstance(function, dict) and isinstance(function.get("parameters"), dict):
        function["parameters"] = _chat_function_parameters(function["parameters"])
    return normalized


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


def _responses_custom_tool_to_chat_tool(tool: dict[str, Any], *, override_name: str | None = None) -> dict[str, Any] | None:
    name = override_name or str(tool.get("name") or "")
    if not name:
        return None
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description") or "Provide the raw custom tool input.",
            "parameters": _custom_tool_parameters(tool),
        },
    }


def responses_tools_to_chat_tools(tools: list[Any] | None) -> list[dict[str, Any]]:
    converted = []
    names: set[str] = set()

    def append_unique(tool: dict[str, Any]) -> None:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = str(function.get("name") or "")
        if name and name in names:
            return
        if name:
            names.add(name)
        converted.append(tool)

    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if "function" in tool:
            append_unique(_normalize_chat_tool(tool))
            continue
        tool_type = tool.get("type")
        if tool_type == "namespace":
            namespace = str(tool.get("name") or "")
            for sub_tool in tool.get("tools") or []:
                if not isinstance(sub_tool, dict):
                    continue
                sub_type = sub_tool.get("type")
                sub_name = _namespace_tool_name(namespace, str(sub_tool.get("name") or ""))
                if sub_type == "function":
                    append_unique(_responses_function_tool_to_chat_tool(
                        sub_tool,
                        override_name=sub_name,
                    ))
                elif sub_type == "custom":
                    converted_custom = _responses_custom_tool_to_chat_tool(
                        sub_tool,
                        override_name=sub_name,
                    )
                    if converted_custom:
                        append_unique(converted_custom)
            continue
        if tool_type == "custom":
            converted_custom = _responses_custom_tool_to_chat_tool(tool)
            if converted_custom:
                append_unique(converted_custom)
            continue
        if tool_type != "function":
            continue
        append_unique(_responses_function_tool_to_chat_tool(tool))
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
        function_fields["parameters"] = _chat_function_parameters(params)
    return {"type": "function", "function": function_fields}


def _responses_request_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    tools = []
    if isinstance(body.get("tools"), list):
        tools.extend(tool for tool in body["tools"] if isinstance(tool, dict))
    input_data = body.get("input")
    if isinstance(input_data, list):
        tools.extend(_responses_additional_tools_from_input(input_data))
    return tools


def _register_custom_tool_aliases(custom_tools: dict[str, dict[str, str]], tool: dict[str, Any], *, namespace: str = "") -> None:
    name = str(tool.get("name") or "")
    if not name:
        return
    mapped = {"name": name, "argument_field": _custom_tool_argument_field(tool)}
    custom_tools[name] = mapped
    if namespace:
        custom_tools[_namespace_map_key(namespace, name)] = mapped
        custom_tools[f"{namespace}-{name}"] = mapped


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
                if not isinstance(sub_tool, dict):
                    continue
                sub_type = sub_tool.get("type")
                sub_name = str(sub_tool.get("name") or "")
                if not sub_name:
                    continue
                if sub_type == "custom":
                    _register_custom_tool_aliases(custom_tools, sub_tool, namespace=name)
                    continue
                if sub_type != "function":
                    continue
                mapped = {"namespace": name, "name": sub_name}
                namespace_tools[sub_name] = mapped
                namespace_tools[_namespace_map_key(name, sub_name)] = mapped
                namespace_tools[f"{name}-{sub_name}"] = mapped
        elif tool_type == "custom" and name:
            _register_custom_tool_aliases(custom_tools, tool)
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
    extra_keys = {
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "response_format",
        "user",
    }
    extra = {key: body[key] for key in extra_keys if key in body}
    extra.update(thinking_fields_from_body(body))
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
    extra_keys = {
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "suffix",
        "echo",
        "logprobs",
        "user",
    }
    extra = {key: body[key] for key in extra_keys if key in body}
    extra.update(thinking_fields_from_body(body))
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

    # Preserve optional thinking controls when this endpoint is routed to an
    # OpenAI-compatible upstream. Responses-style reasoning.effort is mapped
    # onto reasoning_effort by thinking_fields_from_body().
    extra = thinking_fields_from_body(body)

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
        extra=extra,
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
    extra_keys = {
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "response_format",
        "user",
        "parallel_tool_calls",
    }
    extra = {key: body[key] for key in extra_keys if key in body}
    extra.update(thinking_fields_from_body(body))

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
