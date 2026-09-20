import re

import httpx

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
_TIMEOUT_FAILURE = "Upstream request timed out. Please retry."
_RATE_LIMIT_FAILURE = "Upstream rate limited the request. Please retry later."
_BALANCE_FAILURE = "Upstream account balance is exhausted."
_CONNECTION_FAILURE = "Unable to reach the upstream provider. Please retry later."
_UPSTREAM_REJECTION = "Upstream rejected the request. Check model and request parameters; retrying unchanged will fail again."
_UPSTREAM_CREDENTIAL_FAILURE = "Gateway upstream credentials failed. Contact the administrator to update the provider API key."
_BALANCE_FAILURE_MARKERS = (
    "insufficient balance",
    "insufficient credit",
    "insufficient funds",
    "insufficient_quota",
    "account balance",
    "credit balance",
    "balance is too low",
    "balance is low",
    "balance exhausted",
    "exceeded your current quota",
    "arrears",
    "欠费",
    "余额不足",
    "额度不足",
)


def _with_trace_hint(message: str) -> str:
    """Append the current request id so support can correlate the client error
    with the full upstream detail that stays in the server-side error log."""
    rid = get_request_id()
    return f"{message} (request_id: {rid})" if rid else message


def friendly_error_msg(e: Exception) -> str:
    """Return a client-safe message for an upstream failure.

    The raw upstream text is intentionally NOT included. Callers that need it
    for logs must use ``error_detail_for_log`` (or ``str(exc)``) instead.

    文案与客户端状态码同源：先过内容安全/余额等模式表（仅覆盖文案），其余
    一律取自 ``classify_for_client``，不得另搭一套判定顺序。
    """
    msg = str(e)
    lowered = msg.lower()
    for pattern, friendly in _UPSTREAM_ERROR_MAP:
        if pattern.lower() in lowered:
            return _with_trace_hint(friendly)
    if any(marker in lowered for marker in _BALANCE_FAILURE_MARKERS):
        return _with_trace_hint(_BALANCE_FAILURE)
    _status, message = classify_for_client(e)
    return _with_trace_hint(message)


def _flagged_confirmed_upstream(exc: Exception) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if bool(getattr(current, "confirmed_upstream", False)):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _looks_like_connection_error(exc: Exception, lowered: str) -> bool:
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError, ConnectionError)):
        return True
    tokens = ("connection", "connect", "tls", "ssl", "eof", "reset", "refused", "unreachable", "无法连接")
    return any(token in lowered for token in tokens)


def classify_for_client(e: Exception, *, confirmed_upstream: bool = False) -> tuple[int, str]:
    """单一事实来源：把上游异常映射为 (客户端状态码, 客户端安全文案)。

    判定顺序（状态码与文案永远同源，不得在调用方各自重拼）：
    1. 异常链上 isinstance 级硬超时（TimeoutError / httpx.TimeoutException）→ 504；
    2. 权威状态码（链上 status_code / response.status_code）：401/403 → 502+凭据文案
       （网关侧上游凭据问题，不得伪装成客户端 key 失效）；408 → 504；429 → 429+限流文案；
       其余 4xx → 保留原状态码+“修正请求”文案；5xx 及其他非 4xx 的权威状态码（1xx/2xx/3xx，
       必然来自上游）→ 502；
    3. 无权威状态码时：文本型 timeout → 504；二次确认的连接失败 → 502；
    4. 其余：confirmed_upstream=True（调用方已确定异常来自上游，如 Anthropic SSE error 事件），
       或异常链上带 ``confirmed_upstream`` 标记（空流 / 静默截断等网关确认的上游失败）
       → 502；否则保守 500（网关内部错）。
    文本启发式（如消息里出现 "429"/"500"）永远不得在存在权威状态码时否决它，
    也不得推出比状态码更具体的文案。
    """
    # 延迟导入：routing_targets 导入 core.policy，而 core.policy 导入本模块。
    from app.services.routing_targets import (
        classify_upstream_error, has_hard_timeout, upstream_status_code,
    )

    if has_hard_timeout(e):
        return 504, _TIMEOUT_FAILURE
    status = upstream_status_code(e)
    if status is not None:
        if status in (401, 403):
            return 502, _UPSTREAM_CREDENTIAL_FAILURE
        if status == 408:
            return 504, _TIMEOUT_FAILURE
        if status == 429:
            return 429, _RATE_LIMIT_FAILURE
        if 400 <= status <= 499:
            return status, _UPSTREAM_REJECTION
        # 5xx 及其他非 4xx 的权威状态码（1xx/2xx/3xx）都必然来自上游，归 502，
        # 不得把上游异常伪装成网关内部错（审查报告四轮 #4）。
        return 502, _GENERIC_UPSTREAM_FAILURE
    trigger = classify_upstream_error(e)
    if trigger == "timeout":
        return 504, _TIMEOUT_FAILURE
    # classify 的兜底会把未知异常归为 connection_error，需二次确认才给 502，
    # 否则网关内部 bug 会被伪装成上游网络故障。
    if trigger == "connection_error" and _looks_like_connection_error(e, str(e).lower()):
        return 502, _CONNECTION_FAILURE
    if confirmed_upstream or _flagged_confirmed_upstream(e):
        # 调用方已确认异常源自上游（如上游 SSE error 事件、空流、静默截断）：
        # 无法归类时归 502，不伪装成网关内部错；避免调用方对分类器结果做本地
        # 二次改写（审查报告四轮 #1）。
        return 502, _GENERIC_UPSTREAM_FAILURE
    return 500, _GENERIC_UPSTREAM_FAILURE


def client_status_for_upstream_error(e: Exception, *, confirmed_upstream: bool = False) -> int:
    """客户端 HTTP 状态码；与 ``friendly_error_msg`` 同源，见 ``classify_for_client``。"""
    status, _message = classify_for_client(e, confirmed_upstream=confirmed_upstream)
    return status


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
