"""Codex Responses image-generation capability bridge.

The public side follows the hosted ``image_generation`` tool contract used by
OpenAI Responses.  Chat-compatible upstreams receive an internal function tool
instead; the gateway consumes that function call and executes the configured
image backend without exposing the proxy function to the client.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from typing import Any

from app.core.types import InternalMessage, InternalRequest, InternalTool, text_part


IMAGE_BRIDGE_MARKER = "<llm-aio-codex-image-generation>"
IMAGE_BRIDGE_CORRECTION_MARKER = "<llm-aio-codex-image-generation-required>"
IMAGE_BRIDGE_TOOL_NAME = "llm_aio_image_generation"
CODEX_IMAGE_FUNCTION_NAME = "image_gen.imagegen"
GATEWAY_IMAGE_DISPLAY_CALL_PREFIX = "call_gateway_image_display_"
GATEWAY_IMAGE_RESULT_MARKER = "<!-- llm-aio-generated-image -->"
GATEWAY_IMAGE_ASSET_MARKER = "<!-- llm-aio-image-assets -->"
IMAGE_BRIDGE_INSTRUCTIONS = (
    f"{IMAGE_BRIDGE_MARKER}\n"
    "When the user asks for raster image generation, call the attached image-generation "
    "tool with a concise standalone prompt. The gateway will execute the configured image "
    "backend using server-side provider credentials. Do not inspect, request, or require a local "
    "OPENAI_API_KEY, and do not use the local imagegen CLI or scripts/image_gen.py as a fallback. "
    "For coding or design tasks, give each requested asset a short unique filename "
    "and make the prompt suitable for direct project use (for example, request a transparent "
    "background for an isolated sprite or a seamless texture for a background). The tool result "
    "will contain gateway download URLs; download those originals into the user's workspace and "
    "reference the downloaded files in the project. Generated images are not automatically saved "
    "in the agent's workspace. When writing project files on Windows, preserve Unicode by using "
    "apply_patch or an explicitly UTF-8-safe writer; do not pipe non-ASCII PowerShell here-strings "
    "into native processes. Verify interactive projects in a real browser when browser tooling is "
    "available instead of treating static checks or mocked DOM tests as end-to-end validation. "
    "Codex code-mode agents may call capabilities through the custom exec tool. In that mode, "
    "invoke this gateway capability directly with JavaScript such as: "
    "const result = await tools.llm_aio_image_generation({\"prompt\":\"...\","
    "\"filename\":\"asset.png\"}); text(result); "
    "This is a gateway-owned virtual nested tool: it may be absent from the client-side "
    "ALL_TOOLS listing, and the gateway consumes the call before it reaches the client. Do not "
    "use the absence of a local imagegen namespace or an ALL_TOOLS entry to decide whether this "
    "attached capability exists. If client policy requires reading an imagegen skill, follow that "
    "policy and then invoke this gateway tool. Do not claim image generation is unavailable merely because a local "
    "image_gen namespace is absent. Do not call the tool for image analysis or ordinary discussion "
    "about images.\n"
    "</llm-aio-codex-image-generation>"
)

IMAGE_BRIDGE_CORRECTION_INSTRUCTIONS = (
    f"{IMAGE_BRIDGE_CORRECTION_MARKER}\n"
    "The current user explicitly requires newly generated raster images or project assets, but "
    "no image-generation call was made. Before continuing with shell, file, browser, or other "
    "project work, generate the required assets now. Call llm_aio_image_generation once for each "
    "distinct required asset, or in Codex exec mode emit an exec script that calls "
    "tools.llm_aio_image_generation with a JSON object containing prompt and filename. Do not "
    "treat an absent ALL_TOOLS entry as an error. Gateway provider credentials are server-side: "
    "do not inspect or request a local OPENAI_API_KEY and do not switch to scripts/image_gen.py. "
    "After the gateway returns the asset manifest, "
    "download the original into the workspace and continue the user's complete task.\n"
    "</llm-aio-codex-image-generation-required>"
)

_CODEX_EXEC_IMAGE_CAPABILITY = (
    " Gateway virtual nested capability: when raster image generation is required, you may emit "
    "an exec script `const result = await tools.llm_aio_image_generation({\"prompt\":\"...\","
    "\"filename\":\"asset.png\"}); text(result);`. It is valid even when absent from ALL_TOOLS; "
    "the gateway intercepts it before client execution."
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


def has_gateway_generated_image_history(input_data: Any) -> bool:
    """Detect a gateway image result after the latest real user message."""
    if not isinstance(input_data, list):
        return False
    latest_user_index = -1
    for index, item in enumerate(input_data):
        if (
            isinstance(item, dict)
            and str(item.get("type") or "") == "message"
            and str(item.get("role") or "") == "user"
        ):
            latest_user_index = index
    for item in input_data[latest_user_index + 1:]:
        if not isinstance(item, dict):
            continue
        call_id = str(item.get("call_id") or "")
        if call_id.startswith(GATEWAY_IMAGE_DISPLAY_CALL_PREFIX):
            return True
        if str(item.get("type") or "") != "message" or str(item.get("role") or "") != "assistant":
            continue
        serialized = json.dumps(item, ensure_ascii=False).lower()
        if (
            GATEWAY_IMAGE_RESULT_MARKER in serialized
            or GATEWAY_IMAGE_ASSET_MARKER in serialized
            or _is_public_gateway_image_text(serialized)
        ):
            return True
    return False


def sanitize_gateway_image_display_followup(input_data: Any) -> bool:
    """Keep the display tool result while removing image bytes from model context."""
    if not isinstance(input_data, list):
        return False
    changed = False
    for item in input_data:
        if not isinstance(item, dict):
            continue
        call_id = str(item.get("call_id") or "")
        if not call_id.startswith(GATEWAY_IMAGE_DISPLAY_CALL_PREFIX):
            continue
        item_type = str(item.get("type") or "")
        if item_type == "custom_tool_call":
            item["input"] = 'generatedImage({ image_url: "[displayed]" });'
            changed = True
        elif item_type == "custom_tool_call_output":
            item["output"] = "The generated image was displayed successfully."
            changed = True
    return changed


_GENERATED_IMAGE_DATA_URI_RE = re.compile(
    r"data:image/[a-zA-Z0-9.+-]+;base64,[a-zA-Z0-9+/=]+"
)
_GATEWAY_IMAGE_RESULT_URL_RE = re.compile(
    r"https?://[^\s)]+/(?:v1/)?image-results/[a-zA-Z0-9._~-]+"
)
_GATEWAY_ASSET_LINK_RE = re.compile(
    r"\[([^]\r\n]+)\]\((https?://[^\s)]+/(?:v1/)?image-results/[a-zA-Z0-9._~-]+)\)"
)


def _is_public_gateway_image_text(value: str) -> bool:
    """Recognize marker-free generated-image Markdown from this gateway."""
    return bool(
        _GATEWAY_IMAGE_RESULT_URL_RE.search(value)
        and ("download original" in value.lower() or "original:" in value.lower())
        and "![generated image" in value.lower()
    )


def sanitize_gateway_generated_image_history(input_data: Any) -> bool:
    """Remove gateway preview bytes while preserving its artifact manifest.

    Codex sends rendered assistant Markdown back on every tool round. Keeping a
    Base64 preview in that history can consume tens of thousands of tokens and,
    when clients truncate it, can hide the original download URL that follows
    the preview. Only gateway-marked assistant messages are changed; user image
    inputs remain untouched.
    """
    if not isinstance(input_data, list):
        return False
    changed = False
    for item in input_data:
        if (
            not isinstance(item, dict)
            or str(item.get("type") or "") != "message"
            or str(item.get("role") or "") != "assistant"
        ):
            continue
        content = item.get("content")
        blocks = content if isinstance(content, list) else []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            value = block.get("text")
            if not isinstance(value, str) or (
                GATEWAY_IMAGE_RESULT_MARKER not in value
                and not _is_public_gateway_image_text(value)
            ):
                continue
            compact = _GENERATED_IMAGE_DATA_URI_RE.sub(
                "[generated-image-preview-omitted]", value
            )
            if compact != value:
                block["text"] = compact
                changed = True
    return changed


def gateway_generated_image_asset_context(input_data: Any) -> str:
    """Return compact asset manifests created after the latest user message."""
    if not isinstance(input_data, list):
        return ""
    latest_user_index = -1
    for index, item in enumerate(input_data):
        if (
            isinstance(item, dict)
            and str(item.get("type") or "") == "message"
            and str(item.get("role") or "") == "user"
        ):
            latest_user_index = index
    manifests = []
    for item in input_data[latest_user_index + 1:]:
        if not isinstance(item, dict):
            continue
        if (
            str(item.get("type") or "") == "custom_tool_call"
            and str(item.get("call_id") or "").startswith(GATEWAY_IMAGE_DISPLAY_CALL_PREFIX)
        ):
            value = str(item.get("input") or "")
            marker_index = value.find(GATEWAY_IMAGE_ASSET_MARKER)
            if marker_index >= 0:
                compact = _GENERATED_IMAGE_DATA_URI_RE.sub(
                    "[generated-image-preview-omitted]", value[marker_index:]
                )
                # The hidden manifest is stored in a trailing JavaScript block
                # comment. Remove only that wrapper before injecting it into
                # the model's private system context.
                compact = compact.rsplit("*/", 1)[0].strip()
                manifests.append(compact)
            continue
        if (
            str(item.get("type") or "") != "message"
            or str(item.get("role") or "") != "assistant"
        ):
            continue
        for block in item.get("content") or []:
            if not isinstance(block, dict):
                continue
            value = block.get("text")
            if not isinstance(value, str):
                continue
            if GATEWAY_IMAGE_ASSET_MARKER in value:
                compact = _GENERATED_IMAGE_DATA_URI_RE.sub(
                    "[generated-image-preview-omitted]", value
                )
                # Legacy preview blocks included the private manifest in the
                # assistant body. Keep reading those histories during upgrade.
                manifests.append(compact.split(GATEWAY_IMAGE_RESULT_MARKER, 1)[0].strip())
                continue
            if not _is_public_gateway_image_text(value):
                continue
            links = _GATEWAY_ASSET_LINK_RE.findall(value)
            if links:
                lines = [GATEWAY_IMAGE_ASSET_MARKER, "Generated project assets:"]
                lines.extend(f"{name}: {url}" for name, url in links)
                lines.append("Download these originals into the workspace before using them.")
                manifests.append("\n".join(lines))
    return "\n\n".join(dict.fromkeys(value for value in manifests if value))


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
            "filename": {
                "type": "string",
                "description": "Suggested unique project filename, for example board-texture.png.",
            },
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
    for tool in internal.tools:
        if tool.name != "exec" or _CODEX_EXEC_IMAGE_CAPABILITY.strip() in tool.description:
            continue
        tool.description = f"{tool.description.rstrip()}{_CODEX_EXEC_IMAGE_CAPABILITY}"
        raw_tool = tool.raw if isinstance(tool.raw, dict) else None
        raw_function = raw_tool.get("function") if raw_tool else None
        if isinstance(raw_function, dict):
            raw_function["description"] = tool.description
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


def _quote_javascript_object_keys(source: str) -> str:
    """Convert ordinary JS identifier keys to JSON keys without evaluating code.

    Luna commonly emits ``{prompt:\"...\", filename:\"...\"}`` inside the
    Codex exec script. Values remain strict JSON; only keys immediately after
    an object opener or comma and immediately before a colon are quoted. The
    scanner deliberately ignores key-like text inside strings.
    """
    result: list[str] = []
    index = 0
    quote = ""
    escaped = False
    while index < len(source):
        char = source[index]
        if quote:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            index += 1
            continue
        if char in "\"'`":
            quote = char
            result.append(char)
            index += 1
            continue
        if char not in "{,":
            result.append(char)
            index += 1
            continue

        result.append(char)
        index += 1
        whitespace_start = index
        while index < len(source) and source[index].isspace():
            index += 1
        result.append(source[whitespace_start:index])
        identifier_start = index
        if index < len(source) and (source[index].isalpha() or source[index] in "_$"):
            index += 1
            while index < len(source) and (source[index].isalnum() or source[index] in "_$"):
                index += 1
            identifier = source[identifier_start:index]
            spacing_start = index
            while index < len(source) and source[index].isspace():
                index += 1
            if index < len(source) and source[index] == ":":
                result.append(json.dumps(identifier))
                result.append(source[spacing_start:index + 1])
                index += 1
                continue
            result.append(source[identifier_start:index])
    return "".join(result)


def _image_call_arguments_from_javascript(source: str) -> dict[str, Any]:
    value = image_call_arguments(source)
    if value:
        return value
    normalized = _quote_javascript_object_keys(source)
    value = image_call_arguments(normalized)
    if value:
        return value
    # Codex commonly emits JavaScript object literals with single-quoted
    # strings. ``ast.literal_eval`` accepts that data-only subset without
    # executing JavaScript or Python code.
    try:
        value = ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _exec_image_call_sources(script: str) -> list[str]:
    """Extract every balanced gateway-image argument expression from exec JS."""
    sources: list[str] = []
    search_from = 0
    while match := _EXEC_IMAGE_CALL_RE.search(script, search_from):
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
            break
        sources.append(script[start + 1:end])
        search_from = end + 1
    return sources


def image_call_arguments_list_from_exec(arguments: Any) -> list[dict[str, Any]]:
    """Extract all gateway image invocations nested in one Codex exec call."""
    outer = image_call_arguments(arguments)
    script = outer.get("input")
    if not isinstance(script, str):
        return []
    return [
        parsed
        for source in _exec_image_call_sources(script)
        if (parsed := _image_call_arguments_from_javascript(source))
    ]


def image_call_arguments_from_exec(arguments: Any) -> dict[str, Any]:
    """Extract a gateway image call nested in Codex's JavaScript exec tool.

    Some compatible models follow Codex's instruction to execute capabilities
    through ``exec`` instead of emitting the private bridge function directly.
    The client cannot execute that private function, so the gateway consumes it
    before returning the exec call to Codex.
    """
    invocations = image_call_arguments_list_from_exec(arguments)
    return invocations[0] if invocations else {}
