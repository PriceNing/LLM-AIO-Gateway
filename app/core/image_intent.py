"""Helpers for extracting the latest explicit user input (Responses clients).

生图工具注入与调用时机已改为纯模型驱动（见 image_bridge.should_inject_image_bridge），
本模块不再包含任何意图检测逻辑，只保留 latest_user_text 提取工具。
"""

from __future__ import annotations

import re
from typing import Any


_HARNESS_WRAPPER_TAGS = (
    "environment_context",
    "thread_title",
    "agent_skills",
    "permissions_instructions",
    "skills_instructions",
    "collaboration_mode",
)
_HARNESS_WRAPPER_TAG_RE = "|".join(_HARNESS_WRAPPER_TAGS)
_WRAPPED_USER_PROMPT_RE = re.compile(
    r"(?im)^(?:#{1,6}\s*)?(?:User\s+(?:prompt|message|input)|用户(?:提示|输入)|Prompt)\s*[:：]?\s*"
)
_LEADING_HARNESS_WRAPPER_RE = re.compile(
    rf"^(?:<({_HARNESS_WRAPPER_TAG_RE})\b[^>]*>[\s\S]*?</\1>\s*)+",
    re.IGNORECASE,
)
_PSEUDO_USER_WRAPPER_RE = re.compile(
    rf"^(?:<({_HARNESS_WRAPPER_TAG_RE})\b[^>]*>[\s\S]*?</\1>\s*)+$",
    re.IGNORECASE,
)


def _text_from_item(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    value = item.get("text")
    if isinstance(value, str):
        return value.strip()
    content = item.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"].strip())
        return "\n".join(part for part in parts if part)
    return ""


def _unwrap_harness_text(text: str) -> str:
    """Keep only the user payload after Codex title/metadata markers."""
    if not text:
        return ""
    stripped = _LEADING_HARNESS_WRAPPER_RE.sub("", text).strip()
    parts = _WRAPPED_USER_PROMPT_RE.split(stripped)
    if len(parts) > 1:
        return parts[-1].strip()
    return stripped


def _is_pseudo_user_wrapper(text: str) -> bool:
    """True when a role=user item is only a Codex XML envelope."""
    stripped = (text or "").strip()
    return bool(stripped) and bool(_PSEUDO_USER_WRAPPER_RE.fullmatch(stripped))


def latest_user_text(input_data: Any) -> str:
    """Return only the latest explicit user input, excluding tool output.

    Responses tool-result items normally omit ``role``.  Treating a missing
    role as a user role lets command output, logs, or injected context become
    user text on later agent turns.
    """
    if isinstance(input_data, str):
        return _unwrap_harness_text(input_data)
    if not isinstance(input_data, list):
        return ""
    for item in reversed(input_data):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item.get("role") == "user" or (
            not item.get("role") and item_type in {"input_text", "input_message"}
        ):
            text = _text_from_item(item)
            if not text or _is_pseudo_user_wrapper(text):
                continue
            unwrapped = _unwrap_harness_text(text)
            if unwrapped:
                return unwrapped
    return ""
