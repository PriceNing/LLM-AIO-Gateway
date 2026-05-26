import re


_BILLING_HEADER_RE = re.compile(r'^\s*x-anthropic-billing-header:.*(?:\r?\n)?', re.IGNORECASE | re.MULTILINE)
_UPSTREAM_ERROR_MAP = [
    ("output new_sensitive (1027)", "Content blocked by upstream safety policy on output"),
    ("input new_sensitive", "Content blocked by upstream safety policy on input"),
    ("content_filter", "Content blocked by upstream safety policy"),
    ("content_policy_violation", "Content violates upstream usage policy"),
    ("safety_rating", "Content failed upstream safety rating"),
    ("No endpoints found that support image input", "This model does not support image input. Enable image preprocessing in the admin panel."),
]


def strip_billing_header(text):
    if not text:
        if isinstance(text, list):
            return []
        return ""
    if isinstance(text, list):
        cleaned = []
        for block in text:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                stripped = _BILLING_HEADER_RE.sub('', t).strip()
                if stripped:
                    cleaned_block = dict(block)
                    cleaned_block["text"] = stripped
                    cleaned.append(cleaned_block)
            else:
                cleaned.append(block)
        return cleaned
    return _BILLING_HEADER_RE.sub('', text).strip()


def friendly_error_msg(e: Exception) -> str:
    msg = str(e)
    for pattern, friendly in _UPSTREAM_ERROR_MAP:
        if pattern in msg:
            return f"{friendly} (original: {msg[:120]})"
    return msg


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return key
    return key[:4] + "..." + key[-4:]


def message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


def attr(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
