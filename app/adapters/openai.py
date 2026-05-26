from typing import Any

from app.core.types import InternalRequest
from app.protocols.ir import ir_to_openai_messages


def chat_messages_from_internal(internal: InternalRequest) -> list[dict[str, Any]]:
    if not internal.messages:
        raise ValueError("InternalRequest.messages is required for OpenAI adapter")
    return ir_to_openai_messages(internal.messages)


def chat_kwargs_from_internal(internal: InternalRequest) -> dict[str, Any]:
    kwargs = dict(internal.extra)
    if internal.tools:
        kwargs["tools"] = internal.chat_tools()
    if internal.tool_choice is not None and "tool_choice" not in kwargs:
        kwargs["tool_choice"] = internal.tool_choice
    return kwargs
