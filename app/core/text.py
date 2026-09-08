import re

from app.services.logger import get_request_id


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


# 未命中映射表时的兜底消息。上游原文绝不进入客户端响应：原文可能包含
# 上游 URL、内部路径、供应商响应体甚至凭据片段（见「当前问题.md」S2）。
_GENERIC_UPSTREAM_FAILURE = "Upstream request failed. Please retry later or contact the administrator."


def _with_trace_hint(message: str) -> str:
    """Append the current request id so support can correlate the client error
    with the full upstream detail that stays in the server-side error log."""
    rid = get_request_id()
    return f"{message} (request_id: {rid})" if rid else message


def friendly_error_msg(e: Exception) -> str:
    """Return a client-safe message for an upstream failure.

    The raw upstream text is intentionally NOT included. Callers that need it
    for logs must use ``error_detail_for_log`` (or ``str(exc)``) instead.
    """
    msg = str(e)
    lowered = msg.lower()
    for pattern, friendly in _UPSTREAM_ERROR_MAP:
        if pattern.lower() in lowered:
            return _with_trace_hint(friendly)
    return _with_trace_hint(_GENERIC_UPSTREAM_FAILURE)


def error_detail_for_log(e: BaseException, *, max_chars: int = 2000) -> str:
    """Full upstream error text for server-side logging only."""
    parts = [str(e)]
    response = getattr(e, "response", None)
    body = ""
    if response is not None:
        try:
            body = (getattr(response, "text", None) or "").strip()
        except Exception:
            body = ""
    if body and body not in parts[0]:
        parts.append(body)
    return " | ".join(part for part in parts if part)[:max_chars]


def mask_key(key: str) -> str:
    """Always redact: short secrets are masked too, never echoed verbatim."""
    text = str(key or "")
    if not text:
        return ""
    if len(text) <= 8:
        return "*" * len(text)
    return text[:4] + "..." + text[-4:]


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
