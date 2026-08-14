"""Conservative image-generation intent detection for Responses clients."""

from __future__ import annotations

import re
from typing import Any


_CN_IMAGE_WORDS = ("图片", "图像", "照片", "插画", "海报", "头像", "壁纸", "素材")
_CN_ACTION_WORDS = ("生成", "画", "绘制", "制作", "创建", "做一张", "出图", "生图")
_EN_IMAGE_WORDS = (
    "image", "picture", "photo", "illustration", "poster",
    "wallpaper", "avatar", "art", "asset", "assets",
    "visual asset", "visuals", "texture", "textures", "variant",
    "variants",
)
_EN_ACTION_RE = re.compile(r"\b(?:generate|generated|draw|create|make|render|illustrate)\b", re.IGNORECASE)
_NEGATED_IMAGE_REQUEST_RE = re.compile(
    r"(?:不要|无需|不需要|禁止|别)\s*(?:生成|画|绘制|制作|创建|生图)|"
    r"(?:不要|无需|不需要|禁止|别).{0,12}(?:调用|使用|进入|触发).{0,12}(?:图像生成|图片生成|生图)(?:功能|工具|服务)?|"
    r"\b(?:do\s+not|don't|never|without)\s+(?:generate|draw|create|make|render|illustrate)\b",
    re.IGNORECASE,
)
_IMAGE_DISCUSSION_RE = re.compile(
    r"(?:如何|怎么|为什么|为何|是否|能否|可否|请解释|介绍一下).{0,24}(?:图片|图像|照片|生图)|"
    r"\b(?:how|why|whether|explain|describe|documentation|api)\b.{0,40}"
    r"(?:image|picture|photo|illustration|image[ _-]?generation)",
    re.IGNORECASE,
)
_IMAGE_UI_COMPONENT_RE = re.compile(
    r"\bimage\s+(?:upload|input|picker|viewer|preview|editor|component|field|button|endpoint|api|model|tool)\b",
    re.IGNORECASE,
)
_CN_STANDALONE_IMAGE_RE = re.compile(
    r"(?:生成|画|绘制|制作|创建|做|出)\s*(?:一|两|三|四|五|六|七|八|九|十|\d+)?\s*"
    r"(?:张|幅|个)?\s*图(?:\s|$|[，。！？,.!?])"
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


def latest_user_text(input_data: Any) -> str:
    """Return only the latest explicit user input, excluding tool output.

    Responses tool-result items normally omit ``role``.  Treating a missing
    role as a user role lets command output, logs, or injected context become
    an image-generation intent on later agent turns.
    """
    if isinstance(input_data, str):
        return input_data.strip()
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
            if text:
                return text
    return ""


def is_image_generation_intent(input_data: Any, instructions: Any = "") -> bool:
    """Detect an explicit request to create an image, not image discussion."""
    text = latest_user_text(input_data)
    # System/developer instructions describe agent capabilities and repository
    # policy; they are never user authorization to invoke image generation.
    del instructions
    if not text:
        return False
    if _NEGATED_IMAGE_REQUEST_RE.search(text) or _IMAGE_DISCUSSION_RE.search(text):
        return False
    lowered = text.lower()
    if (
        any(action in text for action in _CN_ACTION_WORDS)
        and (any(word in text for word in _CN_IMAGE_WORDS) or _CN_STANDALONE_IMAGE_RE.search(text))
    ):
        return True
    return bool(
        _EN_ACTION_RE.search(text)
        and any(word in lowered for word in _EN_IMAGE_WORDS)
        and not _IMAGE_UI_COMPONENT_RE.search(text)
    )
