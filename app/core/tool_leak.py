"""泄漏工具调用修复（协议无关，IR 层）。

部分本地推理框架（llama.cpp 等）会**概率性**没能把模型按对话模板原生格式
（XML 块）输出的工具调用解析成结构化 tool_calls，于是原始标记文本原样躺在
message 正文里下发。下游 harness（Codex/Claude Code 等）只认结构化工具
调用，收到这种正文会认为"模型只是说了段话"，回合直接结束。

本模块在 IR 输出规范化点做一次保守抢救：仅当

1. 本轮请求声明了工具（无工具则永不触发，纯聊天零开销）；
2. 正文（去除首尾空白后）**整体**由 1..N 个结构完整、彼此紧邻的工具调用块
   组成，块之间只允许空白——嵌在散文/代码围栏里的"讨论语法"不满足此条；
3. 每个块的工具名命中声明集（含 namespace 展平别名）；
4. 每个参数名都在该工具 schema 的 properties 中，且值可按 schema 类型解析；
5. 块嵌套深度与数量在上限内；

才把正文替换为结构化工具调用。任何一条不满足即放弃修复、原样透传——
宁可漏修，绝不误伤。

命中通过 ``app.services.logger`` 的 ``tool_calls`` 通道计数，上游修复后
该指标归零即可退役本层。
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

# 标记字面量用拼接构造，避免本文件自身被任何工具调用解析器误识别。
_OPEN = "<" + "tool_call" + ">"
_CLOSE = "<" + "/" + "tool_call" + ">"
_FUNC_OPEN = "<" + "function="
_FUNC_CLOSE = "<" + "/" + "function" + ">"
_PARAM_OPEN = "<" + "parameter="
_PARAM_CLOSE = "<" + "/" + "parameter" + ">"

MAX_BLOCKS = 8          # 单条消息最多抢救的工具调用数（并行调用上限）
MAX_DEPTH_TAGS = 32     # 未闭合标记数量上限，防畸形输入放大扫描
MAX_TEXT_CHARS = 64_000 # 超过此长度的正文直接放弃（不是泄漏形态）

_BLOCK_RE = re.compile(
    r"\s*"
    + re.escape(_OPEN)
    + r"\s*"
    + re.escape(_FUNC_OPEN)
    + r"([^\s<>]+)\s*>"
    + r"(.*?)"
    + re.escape(_FUNC_CLOSE)
    + r"\s*"
    + re.escape(_CLOSE),
    re.DOTALL,
)

_PARAM_RE = re.compile(
    re.escape(_PARAM_OPEN)
    + r"([^\s<>]+)\s*>"
    + r"(.*?)"
    + re.escape(_PARAM_CLOSE),
    re.DOTALL,
)

_FENCE = "```"


def _iter_tool_index(tools: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """把声明工具索引为 name -> parameters-schema。

    工具名同时收录原名与 namespace 展平后的裸名（``ns.tool`` / ``ns-tool``），
    与 ingress 的 Chat 投影保持同一套命名约定。
    """
    index: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        name = str(getattr(tool, "name", "") or "")
        if not name:
            continue
        params = getattr(tool, "parameters", None)
        if not isinstance(params, dict):
            params = {}
        index[name] = params
        # namespace 展平别名：a.b -> b；a-b -> b
        for sep in (".", "-"):
            if sep in name:
                bare = name.rsplit(sep, 1)[-1]
                if bare:
                    index.setdefault(bare, params)
    return index


def _coerce(value: str, schema: Any) -> Any:
    """按 schema 类型解析文本值；失败返回 _FAIL。"""
    raw = value.strip("\r\n")
    typ = schema.get("type") if isinstance(schema, dict) else None
    if typ in (None, "string"):
        return raw
    if typ in ("integer", "number"):
        try:
            return int(raw) if typ == "integer" else (int(raw) if re.fullmatch(r"[-+]?\d+", raw) else float(raw))
        except ValueError:
            return _FAIL
    if typ == "boolean":
        low = raw.lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        return _FAIL
    if typ in ("object", "array"):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return _FAIL
    return _FAIL


class _Fail:
    pass


_FAIL = _Fail()


def parse_leaked_tool_calls(text: str, tools: Iterable[Any]) -> list[dict[str, Any]] | None:
    """尝试把正文解析为结构化工具调用。

    返回 [{"name": str, "arguments": dict}]；不符合泄漏形态或任何校验失败时
    返回 None（调用方保持原文）。
    """
    if not text or len(text) > MAX_TEXT_CHARS:
        return None
    if _FENCE in text:
        return None
    index = _iter_tool_index(tools)
    if not index:
        return None

    body = text.strip()
    if not body.startswith(_OPEN):
        return None
    if body.count(_OPEN) > MAX_DEPTH_TAGS or body.count(_OPEN) > MAX_BLOCKS:
        return None

    blocks: list[dict[str, Any]] = []
    pos = 0
    while pos < len(body):
        match = _BLOCK_RE.match(body, pos)
        if match is None:
            return None
        name = match.group(1)
        inner = match.group(2)
        params_schema = index.get(name)
        if params_schema is None:
            return None
        props = params_schema.get("properties") if isinstance(params_schema, dict) else None
        if not isinstance(props, dict):
            props = {}
        # inner 必须整体被参数块消费，只允许空白残留
        consumed = 0
        arguments: dict[str, Any] = {}
        for pmatch in _PARAM_RE.finditer(inner):
            if inner[consumed:pmatch.start()].strip():
                return None
            key = pmatch.group(1)
            if key not in props:
                return None
            value = _coerce(pmatch.group(2), props.get(key))
            if value is _FAIL:
                return None
            arguments[key] = value
            consumed = pmatch.end()
        if inner[consumed:].strip():
            return None
        blocks.append({"name": name, "arguments": arguments})
        pos = match.end()

    if not blocks or len(blocks) > MAX_BLOCKS:
        return None
    return blocks


def repair_output(output: Any, tools: Iterable[Any]) -> int:
    """就地修复一个 InternalOutputMessage：命中返回块数，未命中返回 0。

    命中时：output.text 置空、finish_reason 改为 tool_calls、
    output.tool_calls 追加解析结果（arguments 序列化为 JSON 字符串，
    与适配器产物保持同一形状）。
    """
    text = getattr(output, "text", "") or ""
    blocks = parse_leaked_tool_calls(text, tools)
    if not blocks:
        return 0
    from app.core.output import InternalToolCallOutput
    import uuid

    for block in blocks:
        arguments = json.dumps(block["arguments"], ensure_ascii=False)
        call_id = "call_" + uuid.uuid4().hex[:24]
        output.tool_calls.append(InternalToolCallOutput(
            id=call_id,
            call_id=call_id,
            name=block["name"],
            arguments=arguments,
            raw={"repaired_from_text": True},
        ))
    output.text = ""
    output.finish_reason = "tool_calls"
    return len(blocks)
