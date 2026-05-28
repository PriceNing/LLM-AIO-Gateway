import re

from app.services.logger import get_logger

_app_log = get_logger("app")


_DATA_IMAGE_RE = re.compile(r"data:image/(\w+);base64,([A-Za-z0-9+/=]+)")
_LONG_DATA_IMAGE_RE = re.compile(r"data:image/\w+;base64,[A-Za-z0-9+/=]{100,}")


def extract_image_data_uris(content) -> list:
    """Extract data:image/... URIs from string content."""
    if not isinstance(content, str):
        return []
    result = []
    for mime_subtype, data in _DATA_IMAGE_RE.findall(content):
        if len(data) > 100:
            result.append((f"image/{mime_subtype}", f"data:image/{mime_subtype};base64,{data}"))
    return result


def normalize_image_content(messages: list) -> list:
    """Convert inline image data URIs in text into image_url content parts."""
    fixed = False
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    image_uris = extract_image_data_uris(text)
                    if image_uris:
                        fixed = True
                        cleaned = _LONG_DATA_IMAGE_RE.sub("", text).strip()
                        if cleaned:
                            new_parts.append({"type": "text", "text": cleaned})
                        for _mime_type, data_uri in image_uris:
                            new_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            msg["content"] = new_parts
        elif isinstance(content, str):
            image_uris = extract_image_data_uris(content)
            if image_uris:
                fixed = True
                cleaned = _LONG_DATA_IMAGE_RE.sub("", content).strip()
                new_parts = []
                if cleaned:
                    new_parts.append({"type": "text", "text": cleaned})
                for _mime_type, data_uri in image_uris:
                    new_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                msg["content"] = new_parts
    if fixed:
        _app_log.info("[images] normalized data URIs to image_url content parts")
    return messages


def has_image_content(messages: list) -> bool:
    """Check if any message contains image content, including nested tool results."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in ("image_url", "input_image", "image"):
                        return True
                    if part.get("type") == "tool_result":
                        inner = part.get("content")
                        if isinstance(inner, list):
                            for ip in inner:
                                if isinstance(ip, dict) and ip.get("type") == "image":
                                    return True
                        elif isinstance(inner, str) and extract_image_data_uris(inner):
                            return True
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        if extract_image_data_uris(part["text"]):
                            return True
        elif isinstance(content, str) and extract_image_data_uris(content):
            return True
    return False
