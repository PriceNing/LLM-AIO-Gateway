import re

from fastapi import HTTPException
import httpx

from app.services.logger import get_logger

from app.core.policy import RouteTarget
from app.database import find_provider_by_model, get_provider

_app_log = get_logger("app")


def adapter_provider_id(provider_info: dict | None, provider_id: str = "") -> str:
    return (provider_info or {}).get("id") or provider_id or ""


def provider_for_log(provider_info: dict | None, provider_id: str = "") -> str:
    return adapter_provider_id(provider_info, provider_id)


def resolve_provider(model: str, provider_id: str = "") -> dict | None:
    return get_provider(provider_id) if provider_id else find_provider_by_model(model)


def candidate_targets(primary: RouteTarget, fallback_chain: list[RouteTarget] | None = None) -> list[RouteTarget]:
    seen = set()
    targets = []
    for target in [primary, *(fallback_chain or [])]:
        _app_log.debug("[candidate_targets] model=%s provider=%s", target.model, target.provider_id)
        if not target.model:
            continue
        # 去重键必须按“解析后的物理目标”归一（bug-2026-10-05 M-8）：
        # primary 入链前已被解析成具体 provider_id，而 chain 条目常保留管理员
        # 配置的空 provider 或复合 "provider/model" 写法；旧键
        # (model, provider_id) 把同一上游当成两个目标重试两次，放大故障时
        # 的上游压力并浪费 attempt_timeout 预算。
        key = target_identity(target)
        if key in seen:
            continue
        seen.add(key)
        targets.append(target)
    return targets


def target_identity(target: RouteTarget) -> tuple[str, str]:
    """(裸模型名, 解析后 provider_id)：同一物理上游的多种写法归一到同一键。"""
    from app.database import parse_model_id

    ref = parse_model_id(target.model)
    provider_id = target.provider_id or ref.provider_id
    if not provider_id:
        resolved = resolve_provider(ref.model_name, "")
        provider_id = (resolved or {}).get("id") or ""
    return (ref.model_name, provider_id)


def _exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def has_hard_timeout(exc: BaseException) -> bool:
    """异常链上是否存在 isinstance 级的超时证据（非文本推断）。

    classify_upstream_error 的 timeout 判定含文本匹配；客户端状态码/文案需要
    区分“真超时”与“文本里碰巧带 timeout 字样的上游 4xx 拒绝”，避免 400 被
    改写成可重试的 504（与 client_status_for_upstream_error 的权威优先口径对齐）。
    """
    return any(isinstance(item, (TimeoutError, httpx.TimeoutException)) for item in _exception_chain(exc))


def upstream_status_code(exc: BaseException) -> int | None:
    """从异常链中提取第一个可用的上游 HTTP 状态码。

    覆盖 httpx.HTTPStatusError（状态在 .response.status_code）、
    liteLLM 异常（直接挂 .status_code）以及 OpenAI SDK 的 APIStatusError。
    """
    for item in _exception_chain(exc):
        status = getattr(item, "status_code", None)
        if status is None:
            status = getattr(getattr(item, "response", None), "status_code", None)
        if isinstance(status, int) and 100 <= status <= 599:
            return status
    return None


def classify_upstream_error(exc: Exception) -> str:
    chain = list(_exception_chain(exc))

    # Prefer timeout/connection signals before HTTP status wrapping.
    # Anthropic/liteLLM adapters often wrap httpx.TimeoutException as HTTP 502,
    # which would otherwise hide the timeout trigger from fallback/retry.
    for item in chain:
        if isinstance(item, (TimeoutError, httpx.TimeoutException)):
            _app_log.debug("[classify_upstream_error] category=timeout exc_type=%s", type(item).__name__)
            return "timeout"
        if not isinstance(item, HTTPException):
            text = str(item).lower()
            exc_name = type(item).__name__.lower()
            if "timeout" in text or "timed out" in text or "timeout" in exc_name:
                _app_log.debug("[classify_upstream_error] category=timeout text_match exc_type=%s", type(item).__name__)
                return "timeout"

    for item in chain:
        if isinstance(item, (httpx.ConnectError, httpx.NetworkError, ConnectionError)):
            _app_log.debug("[classify_upstream_error] category=connection_error exc_type=%s", type(item).__name__)
            return "connection_error"

    for item in chain:
        if isinstance(item, HTTPException):
            if item.status_code == 429:
                _app_log.debug("[classify_upstream_error] category=http_429 exc_type=%s status=%d", type(item).__name__, item.status_code)
                return "http_429"
            if 500 <= item.status_code <= 599:
                _app_log.debug("[classify_upstream_error] category=http_5xx exc_type=%s status=%d", type(item).__name__, item.status_code)
                return "http_5xx"
            if 400 <= item.status_code <= 499:
                _app_log.debug("[classify_upstream_error] category=http_4xx exc_type=%s status=%d", type(item).__name__, item.status_code)
                return "http_4xx"
            _app_log.debug("[classify_upstream_error] category=unknown exc_type=%s status=%d", type(item).__name__, item.status_code)
            return "unknown"

        # httpx.HTTPStatusError keeps the real status on ``response`` rather than
        # on the exception itself.  Read it before falling back to text matching so
        # a 502 is not misleadingly reported as a connection failure.
        status_code = getattr(item, "status_code", None)
        if status_code is None:
            status_code = getattr(getattr(item, "response", None), "status_code", None)
        if status_code is not None and isinstance(status_code, int):
            if status_code == 429:
                _app_log.debug("[classify_upstream_error] category=http_429 exc_type=%s status=%d", type(item).__name__, status_code)
                return "http_429"
            if 500 <= status_code <= 599:
                _app_log.debug("[classify_upstream_error] category=http_5xx exc_type=%s status=%d", type(item).__name__, status_code)
                return "http_5xx"
            if 400 <= status_code <= 499:
                _app_log.debug("[classify_upstream_error] category=http_4xx exc_type=%s status=%d", type(item).__name__, status_code)
                return "http_4xx"

    text = str(exc).lower()
    # 429 必须作为独立状态码出现（非数字边界），否则 "14293 tokens"、端口
    # 8429、模型名 gpt-429 等子串都会误归限流并错触发回退（bug-2026-10-05 M-20①）。
    if re.search(r"(?<![\w.\-])429(?!\d)", text) or "rate limit" in text or "too many requests" in text:
        _app_log.debug("[classify_upstream_error] category=http_429 text_match exc_type=%s", type(exc).__name__)
        return "http_429"
    if any(code in text for code in (" 500", " 502", " 503", " 504", "http 500", "http 502", "http 503", "http 504")):
        _app_log.debug("[classify_upstream_error] category=http_5xx text_match exc_type=%s", type(exc).__name__)
        return "http_5xx"
    # 兜底分类（bug-2026-10-05 M-20②）：旧实现把一切未识别异常都归
    # connection_error，于是网关自身 bug（TypeError/KeyError/…）也会消耗整条
    # fallback 链。现在只有两类能拿到 connection_error：
    #   - 文本/类型带真实连接语义；
    #   - 其余非内部 bug 类型的异常（RuntimeError 等包装上游失败的惯例形状），
    #     保持可用性回退语义不变。
    # Python 内置错误类型在没有任何可分类信号时视为网关自身缺陷 → unknown，
    # 不命中任何触发器，不消耗回退预算。
    if isinstance(exc, (TypeError, ValueError, KeyError, IndexError, AttributeError, NameError, AssertionError, ZeroDivisionError, ImportError, UnboundLocalError)):
        _app_log.debug("[classify_upstream_error] category=unknown internal_bug_type exc_type=%s", type(exc).__name__)
        return "unknown"
    _app_log.debug("[classify_upstream_error] category=connection_error fallback exc_type=%s", type(exc).__name__)
    return "connection_error"


_SAME_TARGET_RETRY_TRIGGERS = frozenset({"timeout", "http_5xx", "http_429"})
_CONNECTION_RETRY_TOKENS = (
    "connection",
    "connect",
    "tls",
    "ssl",
    "eof",
    "reset",
    "refused",
    "unreachable",
    "无法连接",
)


def is_same_target_retryable(exc: Exception, trigger: str | None = None) -> bool:
    """Return whether an unused first-byte failure should retry the same target once.

    Unknown RuntimeError values currently classify as connection_error; those must
    not be retried, or every generic fallback test would double-hit the primary.
    Gateway-owned attempt_timeout is also excluded: that budget already decided
    to leave the current attempt, so retrying it would only delay fallback.
    """
    if "fallback attempt timeout" in str(exc).lower():
        return False
    category = trigger or classify_upstream_error(exc)
    if category in _SAME_TARGET_RETRY_TRIGGERS:
        return True
    if category != "connection_error":
        return False
    for item in _exception_chain(exc):
        if isinstance(item, (httpx.ConnectError, httpx.NetworkError, ConnectionError)):
            return True
        text = str(item).lower()
        if any(token in text for token in _CONNECTION_RETRY_TOKENS):
            return True
    return False
