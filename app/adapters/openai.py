from typing import Any

from app.core.types import InternalRequest
from app.services.logger import get_logger

_app_log = get_logger("app")
from app.protocols.ir import ir_to_openai_messages


def _chat_tool_choice(tool_choice: Any) -> Any:
    """Project provider-neutral or client-protocol tool_choice to OpenAI Chat shape."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return tool_choice

    choice_type = tool_choice.get("type")
    if choice_type in ("auto", "none", "required"):
        return choice_type
    if choice_type == "any":
        return "required"
    if choice_type == "function":
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return {"type": "function", "function": {"name": function["name"]}}
        if tool_choice.get("name"):
            return {"type": "function", "function": {"name": tool_choice["name"]}}
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return tool_choice


def chat_messages_from_internal(internal: InternalRequest) -> list[dict[str, Any]]:
    if not internal.messages:
        raise ValueError("InternalRequest.messages is required for OpenAI adapter")
    return ir_to_openai_messages(internal.messages)


def chat_kwargs_from_internal(internal: InternalRequest) -> dict[str, Any]:
    kwargs = dict(internal.extra)
    if internal.tools:
        kwargs["tools"] = internal.chat_tools()
    raw_tool_choice = internal.tool_choice if internal.tool_choice is not None else kwargs.get("tool_choice")
    projected_tool_choice = _chat_tool_choice(raw_tool_choice)
    if projected_tool_choice is None:
        kwargs.pop("tool_choice", None)
    else:
        kwargs["tool_choice"] = projected_tool_choice
    _app_log.debug(
        "[openai_adapter] tools=%d tool_choice=%s -> %s extra_keys=%s",
        len(internal.tools or []),
        raw_tool_choice,
        projected_tool_choice,
        list(internal.extra.keys()),
    )
    return kwargs
