"""Responses 客户端 wire 格式的请求特征分析（协议边界）。

这些判断只依赖客户端原始 body，属于 Responses 协议入口语义。统一在这里
计算并由 ``ingress.responses_to_internal`` 写入 ``InternalRequest.metadata``，
端点/策略代码只读 IR metadata，不再直接解析 wire 格式。

metadata 键（全部 JSON 可序列化）：
- ``requires_native_responses``: list[str]  Chat 无法忠实表达的特性
- ``client_owned_tool_markers``: list[str]  Codex 私有 custom/namespace 工具
- ``stateful_tool_markers``: list[str]      跨 provider 有状态标记
- ``required_tool_types``: list[str]        请求声明的工具类型
- ``incomplete_tool_history``: bool         存在未闭合的 tool call
- ``is_system_turn``: bool                  Codex 后台/system 轮
- ``has_prior_assistant``: bool             input 中已有 assistant 轮
- ``image_generation_tool``: dict | None    客户端声明的 image_generation 工具
- ``has_codex_image_function_tool``: bool   客户端自带 image_gen 函数工具
- ``has_codex_generated_image_exec_tool``: bool  客户端声明 generatedImage exec 助手
- ``latest_user_prompt``: str               仅当前用户输入（不含历史）
"""

from __future__ import annotations

import json
from typing import Any

from app.core.image_bridge import (
    has_codex_generated_image_exec_tool,
    has_codex_image_function_tool,
)
from app.core.image_intent import latest_user_text


def client_owned_tool_markers(body: dict) -> list[str]:
    """Identify Codex-owned Responses tools that Chat rewrite cannot preserve faithfully."""
    found: list[str] = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "")
        name = str(tool.get("name") or "")
        if tool_type == "custom" and name and f"custom:{name}" not in found:
            found.append(f"custom:{name}")
        elif tool_type == "namespace" and name and f"namespace:{name}" not in found:
            found.append(f"namespace:{name}")
    input_data = body.get("input")
    if isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict) or item.get("type") != "additional_tools":
                continue
            for tool in item.get("tools") or []:
                if not isinstance(tool, dict):
                    continue
                tool_type = str(tool.get("type") or "")
                name = str(tool.get("name") or "")
                if tool_type == "custom" and name and f"custom:{name}" not in found:
                    found.append(f"custom:{name}")
                elif tool_type == "namespace" and name and f"namespace:{name}" not in found:
                    found.append(f"namespace:{name}")
    return found


def requires_native(body: dict) -> list[str]:
    """Return request features that cannot be faithfully represented by Chat."""
    required = []
    chat_safe_fields = {
        "model", "input", "instructions", "tools", "tool_choice", "stream",
        "temperature", "top_p", "presence_penalty", "frequency_penalty", "stop",
        "user", "previous_response_id", "provider_id", "parallel_tool_calls",
        "max_output_tokens", "max_completion_tokens",
        # Responses metadata/options that have a reasonable IR/Chat
        # compatibility equivalent (or can safely be ignored by the adapter).
        # These must not force an unsupported provider down the native-only
        # path; Codex commonly sends them on every request.
        "reasoning", "text", "store", "metadata", "truncation", "include",
        "background", "service_tier", "safety_identifier", "prompt_cache_key",
        "prompt_cache_retention", "max_tool_calls", "top_logprobs", "logprobs",
        "client_metadata",
    }
    for field, value in body.items():
        if field not in chat_safe_fields and value not in (None, False, "", [], {}):
            required.append(field)
    # Hosted Responses tools are intentionally filtered by ingress when a
    # provider uses the compatibility path.  They must not turn an otherwise
    # compatible Codex request into a native-only request.
    chat_tool_types = {"function", "custom", "namespace", "web_search"}
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type") not in chat_tool_types:
            required.append(f"tool:{tool.get('type') or 'unknown'}")
    return required


def image_generation_tool(body: dict) -> dict | None:
    """Return the client-declared image_generation tool, if any."""
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type") == "image_generation":
            return tool
    for item in body.get("input") or []:
        if not isinstance(item, dict) or item.get("type") != "additional_tools":
            continue
        for tool in item.get("tools") or []:
            if isinstance(tool, dict) and tool.get("type") == "image_generation":
                return tool
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "image_generation":
        return choice
    return None


def is_system_turn(body: dict) -> bool:
    """Return whether Codex identified this as an app-owned background turn.

    Codex creates auxiliary Responses requests for task titles, ambient
    suggestion safety, and other UI metadata.  Their wrapped user prompt can
    mention image generation even though the request itself must only produce
    structured metadata.  Honor ``thread_source=system`` and ambient
    suggestion markers instead of guessing from prompt wording or schemas.
    """
    metadata = body.get("client_metadata")
    if not isinstance(metadata, dict):
        return False
    turn_metadata = metadata.get("x-codex-turn-metadata")
    if isinstance(turn_metadata, str):
        try:
            turn_metadata = json.loads(turn_metadata)
        except (TypeError, ValueError):
            return False
    if not isinstance(turn_metadata, dict):
        return False
    source = str(turn_metadata.get("thread_source") or "").strip().lower()
    trigger = str(turn_metadata.get("turn_trigger") or "").strip().lower()
    request_kind = str(turn_metadata.get("request_kind") or "").strip().lower()
    return (
        source in {"system", "ambient", "ambient_suggestion_safety", "thread_title"}
        or trigger.startswith("ambient")
        or trigger in {"thread_title"}
        or request_kind.startswith("ambient")
        or request_kind in {"thread_title"}
    )


def has_prior_assistant(input_data: Any) -> bool:
    """True when a prior assistant turn is already present in the input."""
    if not isinstance(input_data, list):
        return False
    return any(isinstance(item, dict) and item.get("role") == "assistant" for item in input_data)


def required_tool_types(body: dict) -> set[str]:
    return {str(tool.get("type") or "") for tool in body.get("tools") or [] if isinstance(tool, dict) and tool.get("type")}


def stateful_tool_markers(body: dict) -> list[str]:
    """Collect prior Responses tool/agent markers for logging and fallback policy.

    Only ``previous_response_id`` is provider-bound and blocks cross-provider
    native fallback. Explicit tool outputs remain eligible so a failed primary
    can still reach another native-capable target; dialect incompatibilities
    are filtered separately.
    """
    input_data = body.get("input")
    found = []
    if body.get("previous_response_id"):
        found.append("previous_response_id")
    if not isinstance(input_data, list):
        return found
    marker_types = {
        "custom_tool_call_output",
        "function_call_output",
        "computer_call_output",
    }
    for item in input_data:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type in marker_types and item_type not in found:
            found.append(item_type)
    return found


def cross_provider_incompatible_reasons(body: dict | None) -> list[str]:
    """Shapes that commonly 400 when a Codex native body is forwarded to another vendor."""
    if not isinstance(body, dict):
        return []
    reasons: list[str] = []
    input_data = body.get("input")
    call_ids: set[str] = set()
    output_ids: set[str] = set()
    if isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type in {"function_call", "custom_tool_call"}:
                call_id = str(item.get("call_id") or item.get("id") or "")
                if call_id:
                    call_ids.add(call_id)
            elif item_type in {"function_call_output", "custom_tool_call_output"}:
                call_id = str(item.get("call_id") or "")
                if call_id:
                    output_ids.add(call_id)
    if call_ids - output_ids:
        reasons.append("unpaired_tool_call")
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("encrypted_content") and "encrypted_reasoning" not in reasons:
        reasons.append("encrypted_reasoning")
    return reasons


def incomplete_tool_history(body: dict | None) -> bool:
    """True when Responses input has tool calls without matching outputs.

    Chat Completions cannot repair that shape. Downgrading it only repeats the
    same 400 across fallback providers.
    """
    return "unpaired_tool_call" in cross_provider_incompatible_reasons(body)


def image_prompt(input_data: Any, instructions: Any = "") -> str:
    """Extract the current user request without forwarding conversation history."""
    prompt = latest_user_text(input_data).strip()
    if prompt:
        return prompt
    return str(instructions or "").strip()


def request_flags(body: dict) -> dict:
    """Compute all Responses wire-level request flags for IR metadata.

    Called once by ``responses_to_internal``; endpoint/policy code must read
    the resulting ``InternalRequest.metadata`` instead of re-parsing the wire
    body.
    """
    input_data = body.get("input")
    instructions = body.get("instructions")
    return {
        "requires_native_responses": requires_native(body),
        "client_owned_tool_markers": client_owned_tool_markers(body),
        "stateful_tool_markers": stateful_tool_markers(body),
        "required_tool_types": sorted(required_tool_types(body)),
        "incomplete_tool_history": incomplete_tool_history(body),
        "is_system_turn": is_system_turn(body),
        "has_prior_assistant": has_prior_assistant(input_data),
        "image_generation_tool": image_generation_tool(body),
        "has_codex_image_function_tool": has_codex_image_function_tool(body),
        "has_codex_generated_image_exec_tool": has_codex_generated_image_exec_tool(body),
        "latest_user_prompt": image_prompt(input_data, instructions),
    }
