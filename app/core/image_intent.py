"""Conservative image-generation intent detection for Responses clients."""

from __future__ import annotations

import re
from typing import Any


_CN_IMAGE_WORDS = ("图片", "图像", "照片", "插画", "海报", "头像", "壁纸", "图")
_CN_ACTION_WORDS = ("生成", "画", "绘制", "制作", "创建", "做一张", "出图", "生图")
_EN_IMAGE_WORDS = ("image", "picture", "photo", "illustration", "poster", "wallpaper", "avatar")
_EN_ACTION_RE = re.compile(r"\b(?:generate|draw|create|make|render|illustrate)\b", re.IGNORECASE)


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
    """Return only the latest user-facing text, excluding historical turns."""
    if isinstance(input_data, str):
        return input_data.strip()
    if not isinstance(input_data, list):
        return ""
    for item in reversed(input_data):
        if not isinstance(item, dict):
            continue
        if item.get("role") in {None, "user"}:
            text = _text_from_item(item)
            if text:
                return text
    return ""


def is_image_generation_intent(input_data: Any, instructions: Any = "") -> bool:
    """Detect an explicit request to create an image, not image discussion."""
    text = latest_user_text(input_data)
    if isinstance(instructions, str) and instructions.strip():
        text = f"{instructions.strip()}\n{text}" if text else instructions.strip()
    if not text:
        return False
    lowered = text.lower()
    if any(action in text for action in _CN_ACTION_WORDS) and any(word in text for word in _CN_IMAGE_WORDS):
        return True
    return bool(_EN_ACTION_RE.search(text) and any(word in lowered for word in _EN_IMAGE_WORDS))
