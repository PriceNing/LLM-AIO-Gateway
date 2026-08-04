"""Codex Responses image-generation capability bridge.

The public side follows the hosted ``image_generation`` tool contract used by
OpenAI Responses.  Chat-compatible upstreams receive an internal function tool
instead; the gateway consumes that function call and executes the configured
image backend without exposing the proxy function to the client.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from app.core.types import InternalMessage, InternalRequest, InternalTool, text_part


IMAGE_BRIDGE_MARKER = "<llm-aio-codex-image-generation>"
IMAGE_BRIDGE_TOOL_NAME = "llm_aio_image_generation"
CODEX_IMAGE_FUNCTION_NAME = "image_gen.imagegen"
GATEWAY_IMAGE_DISPLAY_CALL_PREFIX = "call_gateway_image_display_"
IMAGE_BRIDGE_INSTRUCTIONS = (
    f"{IMAGE_BRIDGE_MARKER}\n"
    "When the user asks for raster image generation, call the attached image-generation "
    "tool with a concise standalone prompt. The gateway will execute the configured image "
    "backend. Do not claim image generation is unavailable merely because a local image_gen "
    "namespace is absent. Do not call the tool for image analysis or ordinary discussion "
    "about images.\n"
    "</llm-aio-codex-image-generation>"
)


def _tool_function_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    tool_type = str(tool.get("type") or "")
    if tool_type == "function":
        if str(tool.get("namespace") or "") == "image_gen" and str(tool.get("name") or "") in {"", "imagegen", "image_gen.imagegen"}:
            return CODEX_IMAGE_FUNCTION_NAME
        function = tool.get("function")
        if isinstance(function, dict):
            if str(function.get("namespace") or tool.get("namespace") or "") == "image_gen" and str(function.get("name") or tool.get("name") or "") in {"", "imagegen", "image_gen.imagegen"}:
                return CODEX_IMAGE_FUNCTION_NAME
            return str(function.get("name") or "")
        return str(tool.get("name") or "")
    # Codex Lite declares built-in namespaces in input[].additional_tools.  In
    # the real client payload the namespace may be only a declaration (without
    # the child schema), so the namespace name itself is authoritative.  Some
    # compatible clients flatten the same declaration and keep namespace on
    # the function item; accept that form as well.
    if tool_type == "namespace" and str(tool.get("name") or "") == "image_gen":
        return CODEX_IMAGE_FUNCTION_NAME
    if str(tool.get("namespace") or "") == "image_gen" and str(tool.get("name") or "") in {"", "imagegen", "image_gen.imagegen"}:
        return CODEX_IMAGE_FUNCTION_NAME
    if str(tool.get("name") or "") == CODEX_IMAGE_FUNCTION_NAME:
        return CODEX_IMAGE_FUNCTION_NAME
    return ""


def responses_request_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    tools = [tool for tool in body.get("tools") or [] if isinstance(tool, dict)]
    for item in body.get("input") or []:
        if not isinstance(item, dict) or item.get("type") != "additional_tools":
            continue
        tools.extend(tool for tool in item.get("tools") or [] if isinstance(tool, dict))
    return tools


def has_codex_image_function_tool(body: dict[str, Any]) -> bool:
    return any(_tool_function_name(tool) == CODEX_IMAGE_FUNCTION_NAME for tool in responses_request_tools(body))


def has_codex_generated_image_exec_tool(body: dict[str, Any]) -> bool:
    """Detect Codex's code-mode exec surface that owns generatedImage()."""
    for tool in responses_request_tools(body):
        if str(tool.get("type") or "") != "custom" or str(tool.get("name") or "") != "exec":
            continue
        if "generatedImage" in str(tool.get("description") or ""):
            return True
    return False


def is_gateway_image_display_followup(input_data: Any) -> bool:
    """Recognize the client tool-result turn for a gateway-created display call."""
    if not isinstance(input_data, list):
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("type") or "") in {"custom_tool_call", "custom_tool_call_output"}
        and str(item.get("call_id") or "").startswith(GATEWAY_IMAGE_DISPLAY_CALL_PREFIX)
        for item in input_data
    )


def has_hosted_image_tool(body: dict[str, Any]) -> bool:
    return any(str(tool.get("type") or "") == "image_generation" for tool in responses_request_tools(body))


def inject_hosted_image_capability(body: dict[str, Any]) -> bool:
    """Apply the same public request transform as sub2api.

    A client-owned ``image_gen.imagegen`` tool wins because Codex can execute it
    itself.  Otherwise add one hosted image tool, preserve existing tools and
    tool_choice, and append the bridge instructions exactly once.
    """
    if has_codex_image_function_tool(body):
        return False
    changed = False
    if not has_hosted_image_tool(body):
        tools = body.get("tools")
        if not isinstance(tools, list):
            tools = []
        body["tools"] = [*tools, {"type": "image_generation", "output_format": "png"}]
        changed = True
    if "tool_choice" not in body:
        body["tool_choice"] = "auto"
        changed = True
    existing = str(body.get("instructions") or "").rstrip()
    if IMAGE_BRIDGE_MARKER not in existing:
        body["instructions"] = f"{existing}\n\n{IMAGE_BRIDGE_INSTRUCTIONS}" if existing else IMAGE_BRIDGE_INSTRUCTIONS
        changed = True
    return changed


def configure_internal_image_bridge(internal: InternalRequest, body: dict[str, Any]) -> None:
    """Project the hosted capability to a provider-neutral function tool."""
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Standalone image-generation prompt."},
            "size": {"type": "string", "description": "Optional WIDTHxHEIGHT output size."},
            "quality": {"type": "string"},
            "background": {"type": "string"},
            "output_format": {"type": "string"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }
    raw = {
        "type": "function",
        "function": {
            "name": IMAGE_BRIDGE_TOOL_NAME,
            "description": "Generate a raster image using the gateway's configured image backend.",
            "parameters": copy.deepcopy(parameters),
        },
    }
    internal.tools = [tool for tool in internal.tools if tool.name != IMAGE_BRIDGE_TOOL_NAME]
    internal.tools.append(InternalTool(
        name=IMAGE_BRIDGE_TOOL_NAME,
        description=raw["function"]["description"],
        parameters=parameters,
        raw=raw,
    ))
    internal.system = str(body.get("instructions") or "")
    internal.metadata["instructions"] = internal.system
    if not any(
        IMAGE_BRIDGE_MARKER in part.text
        for message in internal.messages
        for part in message.parts
        if part.kind == "text"
    ):
        internal.messages.insert(0, InternalMessage(role="system", parts=[text_part(IMAGE_BRIDGE_INSTRUCTIONS)]))
    internal.extra["tools"] = internal.chat_tools()
    internal.tool_choice = None
    internal.extra.pop("tool_choice", None)


def image_call_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(str(arguments or "{}"))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


_EXEC_IMAGE_CALL_RE = re.compile(r"(?:tools\.)?llm_aio_image_generation\s*\(")


def image_call_arguments_from_exec(arguments: Any) -> dict[str, Any]:
    """Extract a gateway image call nested in Codex's JavaScript exec tool.

    Some compatible models follow Codex's instruction to execute capabilities
    through ``exec`` instead of emitting the private bridge function directly.
    The client cannot execute that private function, so the gateway consumes it
    before returning the exec call to Codex.
    """
    outer = image_call_arguments(arguments)
    script = outer.get("input")
    if not isinstance(script, str):
        return {}
    match = _EXEC_IMAGE_CALL_RE.search(script)
    if not match:
        return {}
    start = match.end() - 1
    depth = 0
    quote = ""
    escaped = False
    end = None
    for index in range(start, len(script)):
        char = script[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in "'\"`":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end is None:
        return {}
    return image_call_arguments(script[start + 1:end])
