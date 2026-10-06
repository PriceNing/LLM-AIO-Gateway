from typing import Any

from app.core.types import InternalRequest
from app.services.logger import get_logger

_app_log = get_logger("app")
from app.protocols.ir import ir_to_openai_messages


_OPENAI_COMPAT_REASONING_PARAMS = {
    "reasoning_effort",
    "chat_template_kwargs",
    "enable_thinking",
}


def _chat_tool_choice(tool_choice: Any) -> Any:
    """Project provider-neutral or client-protocol tool_choice to OpenAI Chat shape."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        # 标量/列表不可能是任何协议的合法 tool_choice。以前原样透传给下游
        # 客户端库，库侧校验异常被归类为 upstream failure（500「请联系管理员」），
        # 但请求根本没有发出去，会把排查引向上游并可能触发 fallback 无意义重试。
        # 入口（protocols.ingress）已拦成 400，这里只是最后一道防线。
        # dict 仍然透传：Responses 的 tool_choice 值空间更大（file_search /
        # web_search / mcp 等），交给上游用正确的 4xx 拒绝，不在这里猜。
        raise ValueError(f"unsupported tool_choice shape: {type(tool_choice).__name__}")

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
    if choice_type is None:
        # 旧形：无 type 但能解析出工具名（以前靠透传生效，丢弃会改变行为）。
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return {"type": "function", "function": {"name": function["name"]}}
        if tool_choice.get("name"):
            return {"type": "function", "function": {"name": tool_choice["name"]}}
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    # 不可投影的 dict（Responses 专有 file_search/web_search/mcp/allowed_tools 等）：
    # 本函数只在投影成 Chat Completions 的路径上被调用，透传会让 openai SDK 在
    # 建连前本地校验失败，客户端输入问题被误报成上游故障 500（0.0s，请求根本没
    # 发出），还会触发无意义的同目标重试（GATEWAY-F3-FINDING / bug-2026-10-05 M-21）。
    # 丢弃并记 WARNING：实测上游对缺失 tool_choice 正常返回，只是失去强制/限定
    # 工具语义——该语义在此路径上本来也无法送达上游。原生 Responses 回放不经
    # 本函数，仍按 F2 约定原样透传由上游用 4xx 拒绝。
    _app_log.warning(
        "[openai adapter] tool_choice type=%r cannot be projected to Chat Completions; dropped",
        choice_type if isinstance(choice_type, str) else type(choice_type).__name__,
    )
    return None


def chat_messages_from_internal(internal: InternalRequest) -> list[dict[str, Any]]:
    if not internal.messages:
        raise ValueError("InternalRequest.messages is required for OpenAI adapter")
    from app.core.model_capabilities import resolve_request_capabilities

    # 能力驱动的消息投影（bug-2026-10-05 L-1）：只有显式声明
    # accepts_reasoning_content=false 的上游才剥离消息级 reasoning 回写；
    # 缺省（未知/接受）保持 DeepSeek 系既有透传行为。
    caps = resolve_request_capabilities(internal.target_model, internal.provider_id)
    include_reasoning = caps.get("accepts_reasoning_content", True) is not False
    return ir_to_openai_messages(internal.messages, include_reasoning_content=include_reasoning)


def chat_kwargs_from_internal(internal: InternalRequest) -> dict[str, Any]:
    kwargs = dict(internal.extra)
    # 网关内部映射表不得透传给 litellm：litellm 的 get_optional_params 对未知
    # 参数直接抛 UnsupportedParamsError(500)，导致 Responses→Chat 降级路径
    # 请求未发出即失败（bug-2026-10-05 H-2）。egress 仍从 internal.extra 读它们。
    for key in [k for k in kwargs if str(k).startswith("responses_")]:
        kwargs.pop(key, None)
    reasoning_params = _OPENAI_COMPAT_REASONING_PARAMS.intersection(kwargs)
    if reasoning_params:
        allowed = list(kwargs.get("allowed_openai_params") or [])
        for param in sorted(reasoning_params):
            if param not in allowed:
                allowed.append(param)
        kwargs["allowed_openai_params"] = allowed
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
