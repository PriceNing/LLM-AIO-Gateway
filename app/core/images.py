import base64
import binascii
import re

from app.services.logger import get_logger

_app_log = get_logger("app")


_DATA_IMAGE_RE = re.compile(r"data:image/(\w+);base64,([A-Za-z0-9+/=]+)")

# 与旧的 _LONG_DATA_IMAGE_RE 保持同一门槛：短于此长度的 base64 不可能是真实图片。
_MIN_IMAGE_BASE64_CHARS = 100

# 位图容器签名：(起始魔数, 结束标记)。
# 这里只描述图像格式规范本身的事实，不出现任何 provider/model 名称：
# 校验的目的不是"猜某个上游喜欢什么"，而是"网关不主动制造非法载荷"。
# 只有能证明"完整"的格式才会被提升为附件；证明不了的格式一律原样不动。
_IMAGE_SIGNATURES = {
    "png": ((b"\x89PNG\r\n\x1a\n",), (b"IEND\xaeB`\x82",)),
    "jpeg": ((b"\xff\xd8\xff",), (b"\xff\xd9",)),
    "jpg": ((b"\xff\xd8\xff",), (b"\xff\xd9",)),
    "gif": ((b"GIF87a", b"GIF89a"), (b"\x3b",)),
}
# TIFF 等没有结束标记、也没有可靠总长度字段的格式不在表内：无法证明完整即不提升。


def _webp_is_complete(raw: bytes) -> bool:
    """RIFF 容器自带总长度：size 字段 = 除前 8 字节外的全部字节数。

    结构：``RIFF``(0:4) size(4:8) ``WEBP``(8:12) chunk FourCC(12:16) ...
    """
    if len(raw) < 16 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
        return False
    if raw[12:16] not in {b"VP8 ", b"VP8L", b"VP8X", b"ANIM"}:
        return False
    declared = int.from_bytes(raw[4:8], "little")
    # RIFF 允许奇数长度补一个填充字节，故 len-8 或 len-9 都算自洽。
    return declared in (len(raw) - 8, len(raw) - 9)


def _bmp_is_complete(raw: bytes) -> bool:
    """BITMAPFILEHEADER.bfSize(2:6) 必须等于实际字节数；bfSize=0 视为无法证明。"""
    if len(raw) < 6 or raw[:2] != b"BM":
        return False
    declared = int.from_bytes(raw[2:6], "little")
    return declared == len(raw)


_LENGTH_CHECKED = {"webp": _webp_is_complete, "bmp": _bmp_is_complete}


def _matches_signature(mime_subtype: str, raw: bytes) -> bool:
    length_check = _LENGTH_CHECKED.get(mime_subtype)
    if length_check is not None:
        return length_check(raw)
    signature = _IMAGE_SIGNATURES.get(mime_subtype)
    if signature is None:
        # 未知/非位图（svg、tiff、avif、heic 等）：无法证明是完整位图，一律不动。
        return False
    starts, ends = signature
    return raw.startswith(starts) and raw.endswith(ends)


def _is_complete_bitmap(mime_subtype: str, data: str) -> bool:
    """Return True only when *data* is a complete, decodable bitmap payload.

    截断的工具输出、被引号切断的字符串、或只是"长得像"图片的文本都会在这里被拒绝，
    因此网关不会把它们提升成上游必然拒绝的 image_url 附件。
    """
    if len(data) % 4:
        return False
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return False
    if not raw:
        return False
    return _matches_signature(mime_subtype, raw)


def _scan(content: str) -> tuple[list, list]:
    """Single pass over *content*.

    Returns ``(verified, candidate_spans)`` where each verified entry is
    ``(mime, uri, span)`` — the span is what lets deletion stay positional
    instead of substring-based (see ``_strip_extracted``).
    """
    verified: list = []
    candidate_spans: list = []
    if not isinstance(content, str):
        return verified, candidate_spans
    seen: set[str] = set()
    for match in _DATA_IMAGE_RE.finditer(content):
        mime_subtype, data = match.group(1), match.group(2)
        if len(data) < _MIN_IMAGE_BASE64_CHARS:
            continue
        candidate_spans.append(match.span())
        if not _is_complete_bitmap(mime_subtype, data):
            continue
        uri = match.group(0)
        if uri in seen:
            continue  # 同一份载荷只提升一次（与预处理层的同轮去重口径一致）
        seen.add(uri)
        verified.append((f"image/{mime_subtype}", uri, match.span()))
    return verified, candidate_spans


def extract_image_data_uris(content) -> list:
    """Extract complete, decodable ``data:image/...;base64,`` URIs from string content.

    载荷必须通过 base64 与容器完整性校验；未通过的候选会原样留在文本里，
    既不会被提升为图片附件，也不会被删除。
    """
    return [(mime, uri) for mime, uri, _span in _scan(content)[0]]


def _strip_extracted(text: str, spans: list) -> str:
    """Remove exactly the URIs that were promoted, by position.

    按位置从后往前删，而不是 str.replace：一个已验证的短 URI 可能恰好是另一个
    未验证长载荷的前缀（base64 字符类包含 ``=``），子串替换会把长载荷挖成碎片。
    """
    out = text
    for start, end in sorted(spans, reverse=True):
        out = out[:start] + out[end:]
    return out.strip()


def _parts_from_text(verified: list, text: str) -> list:
    cleaned = _strip_extracted(text, [span for _mime, _uri, span in verified])
    parts = []
    if cleaned:
        parts.append({"type": "text", "text": cleaned})
    for _mime_type, uri, _span in verified:
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    return parts


def normalize_image_content(messages: list) -> list:
    """Promote inline bitmap data URIs in *user* messages into ``image_url`` parts.

    边界（刻意为之，放宽前请先想清楚归属）：

    * 只改写 ``role == "user"`` 的消息。assistant / tool 文本里出现的 data URI
      通常是工具日志或被长度上限切断的返回值，不是"要给模型看的图片"。
    * 只提升能被证明完整的位图。校验失败的载荷保持原样：不猜测、不删除、不替换。
    * 只删除确实已经被提取成附件的那段 URI（按位置删），避免同一份数据被计费两次。
    """
    promoted = 0
    unverified = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            verified, candidates = _scan(content)
            unverified += len(candidates) - len({span for _m, _u, span in verified})
            if verified:
                promoted += len(verified)
                msg["content"] = _parts_from_text(verified, content)
        elif isinstance(content, list):
            new_parts = []
            changed = False
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "") or ""
                    verified, candidates = _scan(text)
                    unverified += len(candidates) - len({span for _m, _u, span in verified})
                    if verified:
                        promoted += len(verified)
                        new_parts.extend(_parts_from_text(verified, text))
                        changed = True
                        continue
                new_parts.append(part)
            if changed:
                msg["content"] = new_parts
    if promoted or unverified:
        _app_log.info(
            "[images] normalized data URIs to image_url content parts promoted=%d skipped_unverified=%d",
            promoted,
            unverified,
        )
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
