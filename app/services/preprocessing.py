"""
Image preprocessing pipeline - intercepts image content, sends to a vision model
for description, then replaces the image with text so non-multimodal models can
"see" the image.

Configuration (in config.json):
  "preprocessors": {
    "vision-model": {
      "type": "vision",
      "api_base": "http://127.0.0.1:8080/v1",
      "model": "llamacpp",
      "api_key": ""
    }
  }

Database: config.json preprocessors section. The first enabled entry is auto-applied to all requests.
"""
import hashlib
import logging
import threading
from typing import Optional

import httpx

from app.config import get_default
from app.core.types import InternalMessage, InternalPart, text_part, tool_result_part
from app.services.lite_llm import _extract_image_data_uris

_log = logging.getLogger("llmgw.app")

# Cache image descriptions to avoid redundant vision calls.
# Key: md5(image_content), Value: description text
_cache: dict[str, str] = {}
_cache_lock = threading.Lock()
MAX_CACHE_SIZE = get_default("image_cache_max_size", 500)


def _cache_key(image_url: str, image_data: str = "") -> str:
    """Derive a stable cache key from an image URL or data URI."""
    raw = image_url or image_data or ""
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _get_cached_description(url_or_data: str) -> Optional[str]:
    with _cache_lock:
        return _cache.get(_cache_key(url_or_data))


def _set_cached_description(url_or_data: str, description: str) -> None:
    with _cache_lock:
        if len(_cache) >= MAX_CACHE_SIZE:
            # Evict oldest (simple FIFO via dict pop)
            _cache.pop(next(iter(_cache)), None)
        _cache[_cache_key(url_or_data)] = description


async def describe_image(
    image_url: str = "",
    image_data: str = "",
    preprocessor_config: Optional[dict] = None,
) -> Optional[str]:
    """Send an image to the vision model and return its description.

    Args:
        image_url: HTTP(S) URL to the image.
        image_data: Base64 data URI (data:image/...;base64,...).
        preprocessor_config: Preprocessor config dict from config.json.
    """
    if not preprocessor_config:
        return None

    cached = _get_cached_description(image_url or image_data)
    if cached:
        _log.info("[preprocess] cache HIT image=%s", (image_url or image_data)[:80])
        return cached

    api_base = (preprocessor_config.get("api_base") or "").rstrip("/")
    api_key = preprocessor_config.get("api_key") or "sk-no-auth"
    vision_model = preprocessor_config.get("model")

    if not api_base or not vision_model:
        _log.error("[preprocess] invalid preprocessor config: api_base=%s model=%s", api_base, vision_model)
        return None

    # Build image content for the vision model
    image_text = preprocessor_config.get("prompt",
        "Please describe this image in detail, including all visible text, objects, people, colors, and layout.")
    image_content: list[dict] = [{"type": "text", "text": image_text}]

    if image_data and image_data.startswith("data:image"):
        image_content.append({"type": "image_url", "image_url": {"url": image_data}})
    elif image_url:
        image_content.append({"type": "image_url", "image_url": {"url": image_url}})
    else:
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    max_tokens = preprocessor_config.get("max_tokens", 1024)
    body = {
        "model": vision_model,
        "messages": [{"role": "user", "content": image_content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    try:
        async with httpx.AsyncClient(timeout=preprocessor_config.get("timeout", 60)) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=body,
            )
            if resp.status_code == 200:
                data = resp.json()
                message = data.get("choices", [{}])[0].get("message", {})
                description = message.get("content", "") or message.get("reasoning_content", "")
                if description:
                    _set_cached_description(image_url or image_data, description)
                    _log.info("[preprocess] vision OK model=%s desc_len=%d", vision_model, len(description))
                    return description.strip()
                else:
                    _log.warning("[preprocess] vision returned empty content")
                    return "[image: vision model returned empty response]"
            else:
                _log.error("[preprocess] vision call failed status=%d body=%s",
                          resp.status_code, resp.text[:300])
                return f"[image: vision model HTTP {resp.status_code}]"
    except Exception as e:
        _log.error("[preprocess] vision call exception type=%s msg=%s", type(e).__name__, str(e))
        return "[image: vision model unavailable or timed out]"


async def preprocess_messages(
    messages: list[InternalMessage],
    preprocessor_config: Optional[dict] = None,
) -> list[InternalMessage]:
    """Preprocess images in internal messages."""
    if not preprocessor_config:
        _log.info("[preprocess:v2] SKIP no config dict")
        return messages
    if not messages:
        return messages

    if preprocessor_config.get("enabled") is False:
        _log.info("[preprocess:v2] SKIP preprocessor disabled - stripping images anyway")
        _strip_all_images(messages)
        return messages

    if not has_image_content(messages):
        _log.info("[preprocess:v2] no images detected, skipping")
        return messages

    preprocessor_id = preprocessor_config.get("id", "unknown")
    new_turn_start = _new_turn_start(messages)
    _log.info("[preprocess:v2] new_turn_start=%d total_msgs=%d", new_turn_start, len(messages))

    images_to_describe = _collect_images(messages, new_turn_start)
    if not images_to_describe:
        _log.info("[preprocess:v2] images only in history (before turn %d), strip only", new_turn_start)
        _strip_all_images(messages)
        return messages

    seen_keys = set()
    unique_images = []
    for img in images_to_describe:
        key = _cache_key(img.get("url", ""), img.get("data", ""))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_images.append(img)

    _log.info("[preprocess:v2] processing %d new image(s) via preprocessor=%s (turn_start=%d)",
              len(unique_images), preprocessor_id, new_turn_start)

    descriptions = []
    for img in unique_images[:preprocessor_config.get("max_images", 5)]:
        desc = await describe_image(
            image_url=img.get("url", ""),
            image_data=img.get("data", ""),
            preprocessor_config=preprocessor_config,
        )
        descriptions.append(desc or "[image: could not be described]")

    _strip_images_with_descriptions(messages, descriptions, new_turn_start)
    return messages


def has_image_content(messages: list[InternalMessage]) -> bool:
    for msg in messages:
        for part in msg.parts:
            if part.kind == "image":
                return True
            if part.kind == "tool_result" and has_image_content([InternalMessage(role="tool", parts=part.parts)]):
                return True
            if part.kind == "text" and _extract_image_data_uris(part.text):
                return True
    return False


def _new_turn_start(messages: list[InternalMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.role == "assistant":
            has_text = any(part.kind == "text" and part.text.strip() for part in msg.parts)
            has_tool_calls = any(part.kind == "tool_call" for part in msg.parts)
            if has_text and not has_tool_calls:
                return i + 1
        elif msg.role == "user":
            if any(part.kind == "text" and "<image_description" in part.text for part in msg.parts):
                return i + 1
    return 0


def _collect_images(messages: list[InternalMessage], turn_start: int) -> list[dict]:
    images = []
    for msg in messages[turn_start:]:
        if msg.role not in ("user", "tool"):
            continue
        _collect_images_from_parts(msg.parts, images)
    return images


def _collect_images_from_parts(parts: list[InternalPart], out: list[dict]) -> None:
    for part in parts:
        if part.kind == "image":
            out.append(_image_payload(part))
        elif part.kind == "tool_result":
            _collect_images_from_parts(part.parts, out)
        elif part.kind == "text":
            for _mime, data_uri in _extract_image_data_uris(part.text):
                out.append({"url": "", "data": data_uri})


def _image_payload(part: InternalPart) -> dict:
    source = part.source or {}
    if source.get("kind") == "url":
        return {"url": source.get("url", ""), "data": ""}
    if source.get("kind") == "base64":
        media = source.get("media_type") or "image/png"
        return {"url": "", "data": f"data:{media};base64,{source.get('data', '')}"}
    return {"url": source.get("url", ""), "data": source.get("data", "")}


def _strip_all_images(messages: list[InternalMessage]) -> None:
    for msg in messages:
        msg.parts = _replace_images_in_parts(msg.parts, [], False, 0)[0]


def _strip_images_with_descriptions(messages: list[InternalMessage], descriptions: list, turn_start: int) -> None:
    desc_idx = 0
    for idx, msg in enumerate(messages):
        msg.parts, desc_idx = _replace_images_in_parts(msg.parts, descriptions, idx >= turn_start, desc_idx)


def _replace_images_in_parts(parts: list[InternalPart], descriptions: list, is_current: bool, desc_idx: int):
    replaced = []
    for part in parts:
        if part.kind == "image":
            payload = _image_payload(part)
            replaced.append(text_part(_build_inline_replacement(desc_idx, descriptions, is_current, payload.get("url", ""), payload.get("data", ""))))
            if is_current:
                desc_idx += 1
        elif part.kind == "tool_result":
            inner, desc_idx = _replace_images_in_parts(part.parts, descriptions, is_current, desc_idx)
            replaced.append(tool_result_part(part.tool_call_id, inner, raw=part.raw, extensions=part.extensions))
        elif part.kind == "text":
            text = part.text
            for _mime, data_uri in _extract_image_data_uris(text):
                text = text.replace(data_uri, _build_inline_replacement(desc_idx, descriptions, is_current, image_data=data_uri))
                if is_current:
                    desc_idx += 1
            replaced.append(text_part(text, raw=part.raw, extensions=part.extensions))
        else:
            replaced.append(part)
    return replaced, desc_idx


def _cached_image_replacement(image_url: str = "", image_data: str = "") -> str:
    """Return a cached image description replacement, or the removal marker."""
    cached = _get_cached_description(image_url or image_data)
    if cached:
        return f"[Image cached]: {cached}"
    return "[image: removed]"


def _build_inline_replacement(desc_idx: int, descriptions: list, is_current: bool,
                              image_url: str = "", image_data: str = "") -> str:
    """Build replacement text for an image. Current-turn images get description
    immediately after the placeholder so the LLM doesn't need remote index mapping."""
    if is_current and desc_idx < len(descriptions):
        desc = descriptions[desc_idx]
        return f"[Image #{desc_idx + 1}]: {desc}"
    return _cached_image_replacement(image_url, image_data)
