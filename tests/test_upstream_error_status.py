"""上游异常 → 客户端 HTTP 状态码映射的回归测试。

背景：网关此前把原始上游异常（litellm/httpx）一律压成客户端 500
（"Upstream request failed. Please retry later"），客户端无法区分
"请求被上游拒绝（4xx，重试无意义）"与"网关/上游故障（5xx，可退避重试）"。
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from main import app
from app.config import load_config
from app.core.text import client_status_for_upstream_error
from app.database import init_db, add_provider, add_user, add_user_api_key

client = TestClient(app)
headers = {"Authorization": "Bearer user-key"}


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    config_path = str(tmp_path / "config.json")
    config = load_config(config_path, force_reload=True)
    config.config = {
        "host": "0.0.0.0",
        "port": 8000,
        "database": db_path,
        "logging": {"enabled": False, "level": "INFO", "log_dir": "logs", "retention_days": 30, "console": False},
    }
    config.save()
    init_db(db_path)
    add_provider({
        "id": "test-provider",
        "name": "Test Provider",
        "provider_type": "openai",
        "api_base": "https://api.test.com/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "allowed-model", "name": "Allowed", "enabled": True}],
    })
    add_user({"username": "alice", "display_name": "Alice", "enabled": True})
    add_user_api_key("alice", "default", ["allowed-model"])
    from app.database import get_db
    with get_db() as db:
        db.execute("UPDATE user_api_keys SET key = 'user-key' WHERE username = 'alice'")
    yield config


def _http_status_error(status: int, message: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://upstream.test/v1/responses")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(message or f"Error {status}", request=request, response=response)


class _LiteLLMStyleError(Exception):
    """模拟 litellm.BadRequestError 等直接挂 status_code 属性的异常。"""

    def __init__(self, status_code: int, message: str = "upstream said no"):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# 单元：映射规则
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc,expected", [
    (_http_status_error(400), 400),
    (_http_status_error(404), 404),
    (_http_status_error(422), 422),
    (_LiteLLMStyleError(400), 400),
    # 上游 401/403 是网关侧的上游凭据问题，不能伪装成客户端 key 失效
    (_http_status_error(401), 502),
    (_http_status_error(403), 502),
    (_http_status_error(408), 504),
    (_http_status_error(429), 429),
    (_http_status_error(500), 502),
    (_http_status_error(503), 502),
    (TimeoutError("upstream timed out"), 504),
    (httpx.ConnectError("connection refused"), 502),
    # 无法分类的异常保持 500（网关自身问题）
    (Exception("boom"), 500),
])
def test_client_status_mapping(exc, expected):
    assert client_status_for_upstream_error(exc) == expected


def test_wrapped_upstream_status_survives_exception_chain():
    inner = _http_status_error(400)
    outer = RuntimeError("call failed")
    outer.__cause__ = inner
    assert client_status_for_upstream_error(outer) == 400


# ---------------------------------------------------------------------------
# 集成：四个代理端点保留上游 4xx/5xx 语义
# ---------------------------------------------------------------------------

CHAT_BODY = {"model": "allowed-model", "messages": [{"role": "user", "content": "hi"}]}
COMPLETIONS_BODY = {"model": "allowed-model", "prompt": "hi", "max_tokens": 8}
MESSAGES_BODY = {"model": "allowed-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
RESPONSES_BODY = {"model": "allowed-model", "input": "hi"}


@pytest.mark.parametrize("endpoint,payload", [
    ("/v1/chat/completions", CHAT_BODY),
    ("/v1/completions", COMPLETIONS_BODY),
    ("/v1/messages", MESSAGES_BODY),
    ("/v1/responses", RESPONSES_BODY),
])
@pytest.mark.parametrize("exc_factory,expected", [
    (lambda: _http_status_error(400), 400),
    (lambda: _http_status_error(429), 429),
    (lambda: _http_status_error(503), 502),
    (lambda: TimeoutError("upstream timed out"), 504),
])
def test_proxy_endpoints_map_upstream_errors(monkeypatch, endpoint, payload, exc_factory, expected):
    async def failing_call(*args, **kwargs):
        raise exc_factory()

    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", failing_call)
    response = client.post(endpoint, headers=headers, json=payload)
    assert response.status_code == expected
    # 客户端消息保持安全（不含上游原文），且不再对 4xx 误报"稍后重试"
    text = response.text
    assert "api.test.com" not in text
    assert "upstream said no" not in text
    # 失败日志不能因状态码修正而丢失
    from app.database import get_db
    with get_db() as db:
        row = db.execute("SELECT COUNT(*) AS c FROM request_logs WHERE status != 'ok'").fetchone()
    assert row["c"] >= 1


def test_generic_exception_still_returns_500(monkeypatch):
    async def failing_call(*args, **kwargs):
        raise RuntimeError("gateway internal bug")

    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", failing_call)
    response = client.post("/v1/chat/completions", headers=headers, json=CHAT_BODY)
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# 审查报告 #2/#3：权威状态优先，文本启发式不得决定客户端状态码；
# 状态码与文案语义必须一致。
# ---------------------------------------------------------------------------

def test_text_only_status_hints_stay_conservative():
    # 无权威状态码，仅异常文本里出现 "500"/"429"：不能映射成可重试的 502/429。
    from app.core.text import client_status_for_upstream_error
    assert client_status_for_upstream_error(
        Exception('Bad Request: {"error":"max_tokens must be <= 500"}')
    ) == 500
    assert client_status_for_upstream_error(
        Exception("upstream rejected: 429 something")
    ) == 500


def test_timeout_text_cannot_veto_authoritative_status():
    # 审查报告二轮 #1：上游以 400 拒绝，只因文本带 "timeout" 字样，
    # 不得被文本型 timeout 通道改写成可重试的 504。
    from app.core.text import client_status_for_upstream_error, friendly_error_msg
    exc = _http_status_error(400, message="the timeout param in this request is invalid")
    assert client_status_for_upstream_error(exc) == 400
    assert "rejected" in friendly_error_msg(exc)
    # 真超时（isinstance 级）仍然 504 + 超时文案
    assert client_status_for_upstream_error(TimeoutError("upstream timed out")) == 504
    assert "timed out" in friendly_error_msg(TimeoutError("upstream timed out"))


def test_upstream_credential_failure_has_distinct_message():
    # 审查报告二轮 #4：上游 401/403 与真 5xx 文案必须可区分。
    from app.core.text import client_status_for_upstream_error, friendly_error_msg
    cred = friendly_error_msg(_http_status_error(401))
    assert "credentials" in cred
    assert client_status_for_upstream_error(_http_status_error(401)) == 502
    five_xx = friendly_error_msg(_http_status_error(503))
    assert "credentials" not in five_xx and "retry later" in five_xx
    # 408 复用超时文案
    assert "timed out" in friendly_error_msg(_http_status_error(408))


# ---------------------------------------------------------------------------
# 三轮：统一分类器——状态码与文案永远同源（classify_for_client）
# ---------------------------------------------------------------------------

def test_status_and_message_are_coherent_on_chained_exceptions():
    from app.core.text import classify_for_client

    # 外层连接失败、cause 是权威 400：状态码与文案必须同一结论（400 + 拒绝）
    inner = _http_status_error(400)
    outer = ConnectionError("connection reset")
    outer.__cause__ = inner
    status, message = classify_for_client(outer)
    assert status == 400 and "rejected" in message

    # 权威 429、cause 是连接错误：限流语义不得丢失
    inner2 = httpx.ConnectError("connection refused")
    outer2 = _http_status_error(429)
    outer2.__cause__ = inner2
    status2, message2 = classify_for_client(outer2)
    assert status2 == 429 and "rate limited" in message2


def test_text_only_429_gets_generic_message_not_rate_limit():
    # 审查报告三轮 #2：无权威状态码时，文案不得比状态码更具体。
    from app.core.text import classify_for_client
    status, message = classify_for_client(Exception("HTTP 429 slow down"))
    assert status == 500
    assert "rate limited" not in message
    assert "Upstream request failed" in message


def test_hard_timeout_wins_over_wrapped_5xx_status():
    # ConnectTimeout 包在带 502 的链里：硬超时证据优先 → 504
    import httpx as _httpx
    from app.core.text import classify_for_client
    inner = _httpx.ConnectTimeout("connect timed out")
    outer = _http_status_error(502)
    outer.__cause__ = inner
    status, message = classify_for_client(outer)
    assert status == 504 and "timed out" in message


def test_anthropic_upstream_exception_mapping_sanitizes_detail():
    # 审查报告三轮 #4：适配器把上游原文拼进 detail 的路径已收口。
    from app.adapters.anthropic import _http_exception_from_upstream
    exc400 = _http_exception_from_upstream(400, "internal model name leaked /quota details")
    assert exc400.status_code == 400
    assert "leaked" not in exc400.detail and "quota" not in exc400.detail
    assert "rejected" in exc400.detail
    exc401 = _http_exception_from_upstream(401, "invalid api key sk-secret-token")
    assert exc401.status_code == 502
    assert "sk-secret-token" not in exc401.detail and "credentials" in exc401.detail
    exc503 = _http_exception_from_upstream(503, "upstream blew up")
    assert exc503.status_code == 502 and "blew up" not in exc503.detail


def test_client_message_matches_status_semantics():
    from app.core.text import friendly_error_msg
    rejected = friendly_error_msg(_http_status_error(400))
    assert "rejected" in rejected and "retry later" not in rejected
    # 429 / 5xx 仍提示稍后重试（与可重试语义一致）
    assert "rate limited" in friendly_error_msg(_http_status_error(429))
    assert "retry later" in friendly_error_msg(_http_status_error(503))
