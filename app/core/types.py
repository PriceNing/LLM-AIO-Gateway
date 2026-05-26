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
class ModelRef:
    requested: str = ""
    target: str = ""
    provider_id: str = ""


@dataclass(slots=True)
class GenerationOptions:
    stream: bool = False
    temperature: Any = None
    max_tokens: int = 0
    tool_choice: Any = None
    previous_response_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


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

    @property
    def model_ref(self) -> ModelRef:
        return ModelRef(self.requested_model or "", self.target_model or "", self.provider_id or "")

    @property
    def options(self) -> GenerationOptions:
        return GenerationOptions(
            stream=self.stream,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tool_choice=self.tool_choice,
            previous_response_id=self.previous_response_id,
            extra=dict(self.extra),
        )

    def chat_tools(self) -> list[dict[str, Any]] | None:
        if not self.tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools
        ]

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


def tools_from_chat(tools: list[Any] | None) -> list[InternalTool]:
    normalized = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not name:
            continue
        normalized.append(
            InternalTool(
                name=name,
                description=fn.get("description", ""),
                parameters=fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}},
                raw=dict(tool),
            )
        )
    return normalized
