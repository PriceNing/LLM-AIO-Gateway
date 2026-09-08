"""入站请求体上限的行为测试（「当前问题.md」S4）。"""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import load_config
from app.core.body_limit import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    RequestBodyLimitMiddleware,
    max_request_body_bytes,
)
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def temp_config(tmp_path):
    config_path = str(tmp_path / "config.json")
    config = load_config(config_path, force_reload=True)
    config.config = {
        "host": "127.0.0.1",
        "port": 8000,
        "database": str(tmp_path / "test.db"),
        "logging": {"enabled": False, "level": "INFO", "log_dir": "logs", "retention_days": 30, "console": False},
        "defaults": {"max_request_body_bytes": 2048},
    }
    config.save()
    return config


# -- 配置解析 --

def test_limit_reads_configured_value():
    assert max_request_body_bytes() == 2048


def test_limit_falls_back_on_invalid_value(monkeypatch):
    import app.core.body_limit as module

    monkeypatch.setattr(module, "get_default", lambda key, fallback=None: "not-a-number")
    assert max_request_body_bytes() == DEFAULT_MAX_REQUEST_BODY_BYTES


def test_limit_is_clamped_to_a_minimum(monkeypatch):
    import app.core.body_limit as module

    monkeypatch.setattr(module, "get_default", lambda key, fallback=None: 1)
    assert max_request_body_bytes() == 1024


# -- 集成：声明 Content-Length 的请求在读取前就被拒绝 --

def test_declared_oversized_body_returns_413():
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "x" * 4096}]})
    response = client.post("/v1/chat/completions", content=body, headers={"Authorization": "Bearer missing"})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_body_too_large"


def test_small_body_is_not_rejected_by_the_limit():
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer missing"},
    )
    # 因缺少有效 Key 返回 401，说明请求已经穿过体积限制进入业务校验。
    assert response.status_code == 401
    assert "too large" not in response.text


# -- 单元：分块请求的累计计数 --

def _scope(method="POST", path="/v1/chat/completions", headers=None):
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("127.0.0.1", 1234),
        "headers": headers or [],
    }


class _Collector:
    def __init__(self):
        self.messages = []

    async def __call__(self, message):
        self.messages.append(message)

    @property
    def status(self):
        for message in self.messages:
            if message.get("type") == "http.response.start":
                return message["status"]
        return None


@pytest.mark.asyncio
async def test_chunked_body_over_limit_returns_413():
    async def app_stub(scope, receive, send):
        # 应用试图读完整个请求体。
        while True:
            message = await receive()
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    chunks = [b"a" * 1024, b"b" * 1024, b"c" * 1024]
    sent = list(chunks)

    async def receive():
        if sent:
            return {"type": "http.request", "body": sent.pop(0), "more_body": bool(sent)}
        return {"type": "http.request", "body": b"", "more_body": False}

    collector = _Collector()
    await RequestBodyLimitMiddleware(app_stub)(_scope(), receive, collector)
    assert collector.status == 413


@pytest.mark.asyncio
async def test_chunked_body_within_limit_passes_through():
    async def app_stub(scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    sent = [b"a" * 512, b"b" * 512]

    async def receive():
        if sent:
            return {"type": "http.request", "body": sent.pop(0), "more_body": bool(sent)}
        return {"type": "http.request", "body": b"", "more_body": False}

    collector = _Collector()
    await RequestBodyLimitMiddleware(app_stub)(_scope(), receive, collector)
    assert collector.status == 200


@pytest.mark.asyncio
async def test_get_requests_are_not_buffered_by_the_middleware():
    async def app_stub(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.disconnect"}

    collector = _Collector()
    await RequestBodyLimitMiddleware(app_stub)(_scope(method="GET", path="/v1/models"), receive, collector)
    assert collector.status == 200
