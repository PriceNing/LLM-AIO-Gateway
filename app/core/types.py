from dataclasses import dataclass, field
from typing import Any, Literal


EndpointName = Literal["chat_completions", "completions", "messages", "responses"]
InternalRole = Literal["system", "user", "assistant", "tool"]
PartKind = Literal["text", "image", "tool_call", "tool_result", "reasoning", "unknown"]


@dataclass(slots=True)
class InternalTool:
    """Provider-neutral tool definition stored in function schema shape."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InternalPart:
    kind: PartKind
    text: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""
    name: str = ""
    arguments: Any = None
    raw_arguments: Any = None
    parts: list["InternalPart"] = field(default_factory=list)
    raw: Any = None
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InternalMessage:
    role: InternalRole
    parts: list[InternalPart] = field(default_factory=list)
    name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)


def text_part(text: Any, *, raw: Any = None, extensions: dict[str, Any] | None = None) -> InternalPart:
    return InternalPart(kind="text", text="" if text is None else str(text), raw=raw, extensions=extensions or {})


def _system_text_already_present(haystack: str, value: str) -> bool:
    """Skip only exact instruction blocks, not short substring matches."""
    existing = haystack or ""
    needle = value.strip()
    if not needle:
        return True
    if existing.strip() == needle:
        return True
    return any(block.strip() == needle for block in existing.split("\n\n"))


def prepend_system_text(messages: list[InternalMessage], text: Any, *, raw: Any = None) -> None:
    """Keep a single leading system message instead of inserting another one."""
    value = "" if text is None else str(text)
    if not value.strip():
        return
    if messages and messages[0].role == "system":
        first = messages[0]
        if any(part.kind == "text" and _system_text_already_present(part.text or "", value) for part in first.parts):
            return
        if first.parts and first.parts[0].kind == "text":
            existing = first.parts[0].text or ""
            first.parts[0].text = f"{value}\n\n{existing}" if existing else value
            return
        first.parts.insert(0, text_part(value, raw=raw))
        return
    messages.insert(0, InternalMessage(role="system", parts=[text_part(value, raw=raw)]))


def append_system_text(messages: list[InternalMessage], text: Any, *, raw: Any = None) -> None:
    """Fold extra system instructions into the existing leading system message."""
    value = "" if text is None else str(text)
    if not value.strip():
        return
    for message in messages:
        if message.role != "system":
            continue
        if any(part.kind == "text" and _system_text_already_present(part.text or "", value) for part in message.parts):
            return
        if message.parts and message.parts[-1].kind == "text":
            existing = message.parts[-1].text or ""
            message.parts[-1].text = f"{existing}\n\n{value}" if existing else value
            return
        message.parts.append(text_part(value, raw=raw))
        return
    messages.insert(0, InternalMessage(role="system", parts=[text_part(value, raw=raw)]))


def image_part(source: dict[str, Any], *, raw: Any = None, extensions: dict[str, Any] | None = None) -> InternalPart:
    return InternalPart(kind="image", source=source, raw=raw, extensions=extensions or {})


def tool_call_part(
    tool_call_id: str,
    name: str,
    arguments: Any,
    *,
    raw_arguments: Any = None,
    raw: Any = None,
    extensions: dict[str, Any] | None = None,
) -> InternalPart:
    return InternalPart(
        kind="tool_call",
        tool_call_id=str(tool_call_id or ""),
        name=name or "",
        arguments=arguments,
        raw_arguments=raw_arguments,
        raw=raw,
        extensions=extensions or {},
    )


def tool_result_part(
    tool_call_id: str,
    parts: list[InternalPart],
    *,
    raw: Any = None,
    extensions: dict[str, Any] | None = None,
) -> InternalPart:
    return InternalPart(
        kind="tool_result",
        tool_call_id=str(tool_call_id or ""),
        parts=parts,
        raw=raw,
        extensions=extensions or {},
    )


def reasoning_part(text: Any, *, raw: Any = None, extensions: dict[str, Any] | None = None) -> InternalPart:
    return InternalPart(kind="reasoning", text="" if text is None else str(text), raw=raw, extensions=extensions or {})


def unknown_part(raw: Any, *, extensions: dict[str, Any] | None = None) -> InternalPart:
    return InternalPart(kind="unknown", raw=raw, extensions=extensions or {})


@dataclass(slots=True)
class InternalRequest:
    """Normalized request passed between protocol, policy, and adapter layers."""

    endpoint: EndpointName
    requested_model: str
    target_model: str
    messages: list[InternalMessage] = field(default_factory=list)
    provider_id: str = ""
    system: Any = ""
    tools: list[InternalTool] = field(default_factory=list)
    tool_choice: Any = None
    stream: bool = False
    temperature: Any = None
    max_tokens: int = 0
    previous_response_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    raw_body: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def chat_tools(self) -> list[dict[str, Any]] | None:
        if not self.tools:
            return None
        result = []
        for tool in self.tools:
            raw = tool.raw if isinstance(tool.raw, dict) else {}
            if raw.get("type") == "function" and isinstance(raw.get("function"), dict):
                result.append(dict(raw))
                continue
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return result

    def anthropic_tools(self) -> list[dict[str, Any]] | None:
        if not self.tools:
            return None
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in self.tools
        ]


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(function.get("name") or "")


def _tool_description(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(function.get("description") or "")


def _tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    params = function.get("parameters") or function.get("input_schema")
    return params if isinstance(params, dict) else {"type": "object", "properties": {}}


# tool_choice 的合法形状（协议事实，不含任何厂商/模型身份）：
# * None
# * "auto" / "none" / "required"
# * 带合法 type 的 dict；function/tool 还必须能解析出工具名
TOOL_CHOICE_STRINGS = frozenset({"auto", "none", "required"})
TOOL_CHOICE_DICT_TYPES = frozenset({"auto", "none", "required", "any", "function", "tool"})


def is_valid_tool_choice(tool_choice: Any, *, strict_dict_types: bool = True) -> bool:
    """Whether a client ``tool_choice`` value has a projectable shape.

    非法值以前被原样透传给下游客户端库，库侧校验异常被归类为 upstream
    failure（500「请联系管理员」），但请求根本没有发出去。入口据此判定 400，
    适配器据此拒绝投影，两处必须用同一条规则。

    ``strict_dict_types=False`` 给 Responses 入口用：它的 tool_choice 值空间比
    Chat 大得多（file_search / web_search / mcp / image_generation 等），且原生
    路径会把整个 body 回放给上游、由上游校验。那里只拦“不可能是任何协议的
    合法形状”的值（非 None/str/dict，以及不在白名单里的字符串）。
    """
    if tool_choice is None:
        return True
    if isinstance(tool_choice, str):
        return tool_choice in TOOL_CHOICE_STRINGS
    if not isinstance(tool_choice, dict):
        return False
    if not strict_dict_types:
        return True
    choice_type = tool_choice.get("type")
    if choice_type is None:
        # 容忍旧形 {"function": {"name": ...}}（无 type 字段）：以前是透传且
        # 上游可用，改成 400 属于无谓收紧。仍能解析出工具名就放行。
        function = tool_choice.get("function")
        return isinstance(function, dict) and bool(function.get("name"))
    if choice_type not in TOOL_CHOICE_DICT_TYPES:
        return False
    if choice_type in ("function", "tool"):
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return True
        return bool(tool_choice.get("name"))
    return True


def tools_from_chat(tools: list[Any] | None) -> list[InternalTool]:
    normalized = []
    if not isinstance(tools, list):
        # 入口已对非 list 的 tools 返回 400；这里只做防御，保证诊断/归一化
        # 路径不会因为一个标量而抛 TypeError（`tools or []` 对 123 不生效）。
        return normalized
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = _tool_name(tool)
        if not name:
            continue
        normalized.append(
            InternalTool(
                name=name,
                description=_tool_description(tool),
                parameters=_tool_parameters(tool),
                raw=dict(tool),
            )
        )
    return normalized
