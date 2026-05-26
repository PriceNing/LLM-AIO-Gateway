import json

from app.core.output import InternalOutputMessage, InternalToolCallOutput
from app.core.text import attr
from app.core.think import strip_think_tags
from app.core.tool_args import fix_tool_args
from app.services.logger import get_logger


_app_log = get_logger("app")
_tool_log = get_logger("tool_calls")


def usage_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        "prompt_cache_hit_tokens": getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
        "prompt_cache_miss_tokens": getattr(usage, "prompt_cache_miss_tokens", 0) or 0,
    }


def response_to_internal_output(response) -> InternalOutputMessage:
    choice = response.choices[0]
    message = getattr(choice, "message", {})
    content = strip_think_tags(attr(message, "content", "") or "")
    reasoning = attr(message, "reasoning_content", None) or ""
    if not content and reasoning:
        content = reasoning

    tool_outputs = []
    for tc in attr(message, "tool_calls", None) or []:
        tc_dict = _tool_call_to_dict(tc)
        if int(tc_dict.get("index", 0)) < 0:
            _tool_log.debug("[output_adapter] FILTERED spurious tool_call id=%s idx=%s", tc_dict.get("id"), tc_dict.get("index", 0))
            continue
        fix_tool_args(tc_dict)
        fn = tc_dict.get("function") or {}
        tc_id = tc_dict.get("id") or ""
        _tool_log.debug(
            "[output_adapter] tool_call id=%s name=%s args_chars=%d",
            tc_id,
            fn.get("name", ""),
            len(fn.get("arguments", "") or ""),
        )
        tool_outputs.append(
            InternalToolCallOutput(
                id=tc_id,
                call_id=tc_id if str(tc_id).startswith("call_") else (f"call_{tc_id}" if tc_id else ""),
                name=fn.get("name", ""),
                arguments=fn.get("arguments", "") or "",
                raw=tc_dict,
            )
        )

    usage = usage_dict(response)
    _app_log.debug(
        "[output_adapter] nonstream_output finish_reason=%s text_chars=%d reasoning_chars=%d tool_calls=%d total_tokens=%d cache_hit=%d cache_miss=%d",
        getattr(choice, "finish_reason", "stop") or "stop",
        len(content or ""),
        len(reasoning or ""),
        len(tool_outputs),
        usage.get("total_tokens", 0),
        usage.get("prompt_cache_hit_tokens", 0),
        usage.get("prompt_cache_miss_tokens", 0),
    )

    return InternalOutputMessage(
        role="assistant",
        text=content,
        reasoning=reasoning,
        tool_calls=tool_outputs,
        finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
        usage=usage,
        raw=response,
    )


def _tool_call_to_dict(tool_call) -> dict:
    if hasattr(tool_call, "model_dump"):
        data = tool_call.model_dump(exclude_none=True)
    elif isinstance(tool_call, dict):
        data = dict(tool_call)
    else:
        fn = getattr(tool_call, "function", None)
        data = {
            "index": getattr(tool_call, "index", 0),
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", "function"),
            "function": {
                "name": getattr(fn, "name", None) if fn else None,
                "arguments": getattr(fn, "arguments", "") if fn else "",
            },
        }
    if data.get("id") is not None and not isinstance(data["id"], str):
        data["id"] = str(data["id"])
    return data


def tool_arguments_to_input(arguments: str):
    try:
        return json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return {}
