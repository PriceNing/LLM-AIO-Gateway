from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal


OutputEventKind = Literal[
    "message_start",
    "text_delta",
    "reasoning_delta",
    "tool_call_start",
    "tool_call_arguments_delta",
    "tool_call_done",
    "message_delta",
    "message_done",
    "usage",
    "metadata",
]


@dataclass(slots=True)
class InternalOutputEvent:
    """Provider-neutral output event used between upstream adapters and endpoint renderers."""

    kind: OutputEventKind
    text: str = ""
    reasoning: str = ""
    role: str = "assistant"
    finish_reason: str | None = None
    stop_reason: str | None = None
    output_index: int = 0
    content_index: int = 0
    tool_index: int = 0
    tool_call_id: str = ""
    call_id: str = ""
    name: str = ""
    arguments_delta: str = ""
    arguments: str = ""
    reasoning_signature: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass(slots=True)
class InternalToolCallOutput:
    id: str = ""
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    raw: Any = None


@dataclass(slots=True)
class InternalOutputMessage:
    role: str = "assistant"
    text: str = ""
    reasoning: str = ""
    reasoning_signature: str = ""
    tool_calls: list[InternalToolCallOutput] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


async def aclose_async_iterator(iterator) -> None:
    """Deterministically close an async iterator, never masking the caller's error.

    上游流适配器内部用 ``async with client.stream(...)`` 持有连接；只有生成器
    被真正关闭时 ``__aexit__`` 才会执行。依赖 GC 与 finalizer 的时机不是释放
    保证，因此所有提前放弃迭代的路径（超时、回退、客户端断开）都应显式关闭
    （见「当前问题.md」S5）。
    """
    aclose = getattr(iterator, "aclose", None)
    if not callable(aclose):
        return
    with suppress(Exception):
        await aclose()
