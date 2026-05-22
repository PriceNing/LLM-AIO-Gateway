"""
Image preprocessing pipeline — intercepts image content, sends to a vision model
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
import json
import logging
import threading
import time
from typing import Optional

import httpx

from app.config import get_config, get_default
from app.services.lite_llm import _extract_image_data_uris, _has_image_content

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


def _get_preprocessor_config(preprocessor_id: str) -> Optional[dict]:
    """Load preprocessor configuration from config.json."""
    cfg = get_config()
    preprocessors = cfg.config.get("preprocessors", {})
    return preprocessors.get(preprocessor_id)


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
    messages: list,
    preprocessor_config: Optional[dict] = None,
) -> list:
    """Intercept images in messages, describe them via vision model, replace with text."""
    if not preprocessor_config:
        _log.info("[preprocess] SKIP no config dict")
        return messages

    if preprocessor_config.get("enabled") is False:
        _log.info("[preprocess] SKIP preprocessor disabled — stripping images anyway")
        _strip_all_images(messages)
        return messages

    preprocessor_id = preprocessor_config.get("id", "unknown")

    if not _has_image_content(messages):
        _log.info("[preprocess] no images detected, skipping")
        return messages

    # 找到"当前 turn"的起始位置：向前查找最后一个有文本内容且无 tool_calls
    # 的 assistant 消息（即上一轮的最终回复），该位置之后的消息属于当前轮次。
    # 只对当前轮次的新图片进行描述；历史消息中的旧图片只做 strip 处理，
    # 避免旧图片描述被重复追加到最新 user message 中造成模型混淆。
    new_turn_start = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                has_text = bool(content.strip())
            elif isinstance(content, list):
                has_text = any(
                    p.get("type") == "text" and p.get("text", "").strip()
                    for p in content if isinstance(p, dict)
                )
            else:
                has_text = False
            has_tool_calls = bool(msg.get("tool_calls"))
            # 有文本但无 tool_calls → 上一轮的最终回复 → 以此为界
            if has_text and not has_tool_calls:
                new_turn_start = i + 1
                break
        elif msg.get("role") == "user":
            # 如果消息中已经包含 <image_description>（之前预处理过的标记），
            # 说明这是上一轮已经处理过的 user 消息，以此为界。
            c = msg.get("content", "")
            if isinstance(c, str) and "<image_description" in c:
                new_turn_start = i + 1
                break
            elif isinstance(c, list):
                has_desc = any(
                    isinstance(p, dict) and p.get("type") == "text"
                    and "<image_description" in p.get("text", "")
                    for p in c
                )
                if has_desc:
                    new_turn_start = i + 1
                    break

    _log.info("[preprocess] new_turn_start=%d total_msgs=%d", new_turn_start, len(messages))

    # 只扫描当前轮次的消息中是否有图片
    has_new_images = False
    for i in range(new_turn_start, len(messages)):
        msg = messages[i]
        role = msg.get("role")
        if role not in ("user", "tool"):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("image_url", "input_image", "image"):
                    has_new_images = True
                    break
                if part.get("type") == "tool_result":
                    inner = part.get("content")
                    if isinstance(inner, list):
                        for ip in inner:
                            if isinstance(ip, dict) and ip.get("type") == "image":
                                has_new_images = True
                                break
                if part.get("type") == "text":
                    if _extract_image_data_uris(part.get("text", "")):
                        has_new_images = True
                        break
            if has_new_images:
                break
        elif isinstance(content, str):
            if _extract_image_data_uris(content):
                has_new_images = True
                break

    if not has_new_images:
        # 当前轮次无新图片，仅清理历史消息中的残留图片
        if _has_image_content(messages):
            _log.info("[preprocess] images only in history (before turn %d), strip only", new_turn_start)
        else:
            _log.info("[preprocess] no images detected, skipping")
        _strip_all_images(messages)
        return messages

    # 从当前轮次消息中提取图片
    images_to_describe: list[dict] = []
    last_user_msg_idx = None

    for i in range(new_turn_start, len(messages)):
        msg = messages[i]
        role = msg.get("role")
        if role not in ("user", "tool"):
            continue
        if role == "user":
            last_user_msg_idx = i
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("image_url", "input_image"):
                    url = part.get("image_url", {})
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    elif isinstance(url, str):
                        pass
                    else:
                        url = ""
                    images_to_describe.append({"url": url, "data": ""})
                elif part.get("type") == "image":
                    source = part.get("source", {})
                    if source.get("type") == "base64":
                        media = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        images_to_describe.append({"url": "", "data": f"data:{media};base64,{data}"})
                elif part.get("type") == "tool_result":
                    inner = part.get("content")
                    if isinstance(inner, list):
                        for ip in inner:
                            if isinstance(ip, dict) and ip.get("type") == "image":
                                source = ip.get("source", {})
                                if source.get("type") == "base64":
                                    media = source.get("media_type", "image/png")
                                    data = source.get("data", "")
                                    images_to_describe.append({"url": "", "data": f"data:{media};base64,{data}"})
                elif part.get("type") == "text":
                    uris = _extract_image_data_uris(part.get("text", ""))
                    for mime, data_uri in uris:
                        images_to_describe.append({"url": "", "data": data_uri})
        elif isinstance(content, str):
            uris = _extract_image_data_uris(content)
            for mime, data_uri in uris:
                images_to_describe.append({"url": "", "data": data_uri})

    if not images_to_describe:
        _strip_all_images(messages)
        return messages

    # 同一轮内去重：同一张图可能同时出现在 user message 和 tool_result 中
    seen_keys = set()
    unique_images: list[dict] = []
    for img in images_to_describe:
        key = _cache_key(img.get("url", ""), img.get("data", ""))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_images.append(img)
    images_to_describe = unique_images

    _log.info("[preprocess] processing %d new image(s) via preprocessor=%s (turn_start=%d)",
             len(images_to_describe), preprocessor_id, new_turn_start)

    descriptions = []
    for img in images_to_describe[:preprocessor_config.get("max_images", 5)]:
        desc = await describe_image(
            image_url=img.get("url", ""),
            image_data=img.get("data", ""),
            preprocessor_config=preprocessor_config,
        )
        if desc:
            descriptions.append(desc)
        else:
            descriptions.append("[image: could not be described]")

    _log.info("[preprocess] described %d images", len(descriptions))

    # 将描述内联插入到每张图片的原位置之后，保留文件名/上下文关联
    _strip_images_with_descriptions(messages, descriptions, new_turn_start)

    return messages


def _join_text_parts(parts: list) -> str | list:
    """If all items are text blocks (type=text dicts or plain strings), join into a single string.
    Otherwise return the list unchanged."""
    if not parts:
        return ""
    all_text = True
    text_values = []
    for p in parts:
        if isinstance(p, str):
            text_values.append(p)
        elif isinstance(p, dict) and p.get("type") == "text":
            text_values.append(p.get("text", ""))
        else:
            all_text = False
            break
    if all_text:
        return "\n".join(text_values)
    return parts


def _build_inline_replacement(desc_idx: int, descriptions: list, is_current: bool) -> str:
    """Build replacement text for an image. Current-turn images get description
    immediately after the placeholder so the LLM doesn't need remote index mapping."""
    if is_current and desc_idx < len(descriptions):
        desc = descriptions[desc_idx]
        ts = time.strftime("%H:%M:%S")
        return f"[Image #{desc_idx + 1} at {ts}]: {desc}"
    return "[image: removed]"


def _strip_images_with_descriptions(messages: list, descriptions: list, turn_start: int) -> None:
    """Strip all images from messages. In the current turn (>= turn_start), each image
    is replaced with a placeholder followed immediately by its description, preserving
    positional context with nearby filename references.

    History messages (< turn_start) only get stripped without descriptions."""
    desc_idx = 0
    for mi, msg in enumerate(messages):
        is_current = mi >= turn_start
        content = msg.get("content")
        if isinstance(content, list):
            stripped = []
            for part in content:
                if not isinstance(part, dict):
                    stripped.append(part)
                    continue

                if part.get("type") in ("image_url", "input_image", "image"):
                    replacement = _build_inline_replacement(desc_idx, descriptions, is_current)
                    stripped.append({"type": "text", "text": replacement})
                    if is_current:
                        desc_idx += 1
                    _log.debug("[preprocess] stripped image %s", "with desc" if is_current else "history")

                elif part.get("type") == "tool_result":
                    inner = part.get("content")
                    if isinstance(inner, list):
                        clean_inner = []
                        for ip in inner:
                            if isinstance(ip, dict) and ip.get("type") == "image":
                                replacement = _build_inline_replacement(desc_idx, descriptions, is_current)
                                clean_inner.append({"type": "text", "text": replacement})
                                if is_current:
                                    desc_idx += 1
                            else:
                                clean_inner.append(ip)
                        part = dict(part)
                        part["content"] = _join_text_parts(clean_inner)
                    stripped.append(part)
                else:
                    stripped.append(part)
            msg["content"] = _join_text_parts(stripped)

        elif isinstance(content, str):
            uris = _extract_image_data_uris(content)
            cleaned = content
            for _mime, data_uri in uris:
                replacement = _build_inline_replacement(desc_idx, descriptions, is_current)
                cleaned = cleaned.replace(data_uri, replacement)
                if is_current:
                    desc_idx += 1
            if cleaned != content:
                msg["content"] = cleaned
                _log.debug("[preprocess] stripped data URI from text")


def _strip_all_images(messages: list) -> None:
    """Remove all image_url, input_image, and Anthropic image content parts.
    Replaces them with '[image: removed]' text placeholders to preserve
    conversation flow for non-multimodal models."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            stripped = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in ("image_url", "input_image", "image"):
                        stripped.append({"type": "text", "text": "[image: removed]"})
                        _log.debug("[preprocess] stripped image from message")
                    elif part.get("type") == "tool_result":
                        # Strip nested images from tool_result content
                        inner = part.get("content")
                        if isinstance(inner, list):
                            clean_inner = []
                            for ip in inner:
                                if isinstance(ip, dict) and ip.get("type") == "image":
                                    clean_inner.append({"type": "text", "text": "[image: removed]"})
                                else:
                                    clean_inner.append(ip)
                            part = dict(part)
                            part["content"] = _join_text_parts(clean_inner)
                        stripped.append(part)
                    else:
                        stripped.append(part)
                else:
                    stripped.append(part)
            msg["content"] = _join_text_parts(stripped)
        elif isinstance(content, str):
            # Use _extract_image_data_uris to find all embedded data URIs
            uris = _extract_image_data_uris(content)
            cleaned = content
            for _mime, data_uri in uris:
                cleaned = cleaned.replace(data_uri, "[image: removed]")
            if cleaned != content:
                msg["content"] = cleaned
                _log.debug("[preprocess] stripped data URI from text")
