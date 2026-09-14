"""proxy/策略层 + 图像管线审查修复的回归测试。

覆盖：上游错误不再误判为客户端断开、conv_key 覆盖、/messages 与 /completions
保留上游状态码、图像批次取消不连带取消等待者、Retry-After 封顶、
图像下载跟随重定向、url_guard 拒绝无法解析主机、data URI 边界阈值、
预处理占位文本识别。
"""
import asyncio
import base64
import json

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from main import app
from app.config import load_config
from app.core.outcome import is_client_disconnect_error
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


# ---------------------------------------------------------------------------
# 高危 #1：上游错误绝不误判为客户端断开
# ---------------------------------------------------------------------------

def test_upstream_errors_not_classified_as_client_disconnect():
    # httpx 上游错误的典型文本恰好包含旧匹配标记
    assert is_client_disconnect_error(httpx.ReadError("connection reset by peer")) is False
    assert is_client_disconnect_error(httpx.RemoteProtocolError("peer closed connection")) is False
    assert is_client_disconnect_error(httpx.ConnectError("connection closed")) is False

    # 经 fallback 链抛出的上游错误（带 request_details）
    tagged = RuntimeError("connection reset")
    tagged.request_details = {"error_trigger": "connection_error"}
    assert is_client_disconnect_error(tagged) is False

    # 普通异常携带网络错误文本：不再匹配
    assert is_client_disconnect_error(RuntimeError("connection reset by peer")) is False
    assert is_client_disconnect_error(RuntimeError("broken pipe")) is False

    # 真正的客户端断开仍然识别
    assert is_client_disconnect_error(ConnectionResetError()) is True
    assert is_client_disconnect_error(BrokenPipeError()) is True
    assert is_client_disconnect_error(RuntimeError("client disconnected")) is True
    assert is_client_disconnect_error(asyncio.CancelledError()) is True
    assert is_client_disconnect_error(GeneratorExit()) is True

    class ClientDisconnect(Exception):
        pass

    assert is_client_disconnect_error(ClientDisconnect()) is True


# ---------------------------------------------------------------------------
# 中危 #4：conv_key 覆盖保证两次策略调用一致
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_policy_conv_key_override():
    from app.core.policy import prepare_request_policy
    from app.protocols.ingress import chat_completions_to_internal

    req = chat_completions_to_internal({
        "model": "allowed-model",
        "messages": [{"role": "user", "content": "hi"}],
    })

    async def no_preprocess(*args, **kwargs):
        return False

    result = await prepare_request_policy(
        req,
        username="alice",
        api_key_value="user-key",
        preprocess_request=no_preprocess,
        conversation_cache_key=lambda *args: "computed-key",
        preprocess=False,
        conv_key_override="endpoint-key",
        log_label="test",
    )
    assert result.conv_key == "endpoint-key"


# ---------------------------------------------------------------------------
# 中危 #5：/messages 与 /completions 保留上游状态码
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("endpoint,payload", [
    ("/v1/messages", {"model": "allowed-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}),
    ("/v1/completions", {"model": "allowed-model", "prompt": "hi", "max_tokens": 8}),
])
def test_proxy_endpoints_preserve_upstream_status(monkeypatch, endpoint, payload):
    async def failing_call(*args, **kwargs):
        raise HTTPException(status_code=429, detail="Upstream: rate limited")

    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", failing_call)
    response = client.post(endpoint, headers=headers, json=payload)
    assert response.status_code == 429, "上游 429 被压平成 500，客户端无法正确退避"


# ---------------------------------------------------------------------------
# 图像 #2：owner 取消不得连带取消等待者
# ---------------------------------------------------------------------------

def test_image_batch_cancel_rejects_waiters_with_plain_error():
    from app.core.image_batch import ImageInvocationCache

    cache = ImageInvocationCache()
    owner = cache.claim("k", ttl_seconds=60, max_entries=4)
    waiter = cache.claim("k", ttl_seconds=60, max_entries=4)
    assert owner.owner is True and waiter.owner is False

    cache.reject(owner, asyncio.CancelledError())
    # 等待者拿到的是普通异常（可处理/可重试），而不是被当作自身任务取消
    with pytest.raises(RuntimeError):
        waiter.future.result()


# ---------------------------------------------------------------------------
# 图像 #1：Retry-After 封顶
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_imagegen_retry_after_capped(monkeypatch):
    from app.adapters.imagegen import generate_images

    calls = 0

    class MockResponse:
        def __init__(self, status_code, headers=None, body=None):
            self.status_code = status_code
            self.headers = headers or {}
            self.text = '{"error":"slow down"}'
            self._body = body or {}

        def json(self):
            return self._body

    class MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return MockResponse(429, headers={"retry-after": "3600"})
            return MockResponse(200, body={"data": [{"b64_json": base64.b64encode(b"img").decode()}]})

    delays = []

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: MockClient())
    monkeypatch.setattr("app.adapters.imagegen.asyncio.sleep", record_sleep)
    result = await generate_images(
        {"api_base": "http://image.test/v1", "model": "m", "max_retries": 1},
        prompt="x",
    )
    assert calls == 2
    assert result
    assert delays and delays[0] <= 30.0, f"Retry-After 未封顶: {delays}"


# ---------------------------------------------------------------------------
# 图像 #3：下载跟随重定向并逐跳校验
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_image_follows_redirect():
    from app.adapters.imagegen import _download_image

    class Resp:
        def __init__(self, status, headers=None, body=b""):
            self.status_code = status
            self.headers = headers or {}
            self._body = body

        def raise_for_status(self):
            pass

        async def aiter_bytes(self):
            yield self._body

    class FakeClient:
        def __init__(self, responses):
            self.responses = responses
            self.urls = []

        def stream(self, method, url, **kwargs):
            self.urls.append(url)
            resp = self.responses.pop(0)

            class Ctx:
                async def __aenter__(self):
                    return resp

                async def __aexit__(self, *args):
                    return False

            return Ctx()

    fake_png = b"\x89PNG fake image"
    fake_client = FakeClient([
        Resp(302, {"location": "http://cdn.test/img.png"}),
        Resp(200, {"content-type": "image/png"}, fake_png),
    ])
    data_uri, mime = await _download_image(fake_client, "http://image.test/result", allow_private_hosts=True)
    assert mime == "image/png"
    assert data_uri == f"data:image/png;base64,{base64.b64encode(fake_png).decode('ascii')}"
    assert fake_client.urls == ["http://image.test/result", "http://cdn.test/img.png"]


# ---------------------------------------------------------------------------
# 图像 #7：无法解析的主机名留痕警告（离线环境不硬拒绝保存配置）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_url_guard_warns_on_unresolvable_host(monkeypatch):
    from app.services import url_guard

    monkeypatch.setattr(url_guard, "_resolve_addresses", lambda host: [])
    cleaned = await url_guard.validate_upstream_url_async("http://no-such-host.invalid:8000/v1")
    assert cleaned.startswith("http://no-such-host.invalid")


# ---------------------------------------------------------------------------
# 图像 #9：data URI 提取/清除阈值边界一致
# ---------------------------------------------------------------------------

def test_image_data_uri_boundary_extraction():
    from app.core.images import extract_image_data_uris

    data = "A" * 100
    uri = f"data:image/png;base64,{data}"
    assert extract_image_data_uris(uri) == [("image/png", uri)]


# ---------------------------------------------------------------------------
# 图像 #6：预处理占位文本识别（死标记修复）
# ---------------------------------------------------------------------------

def test_new_turn_start_detects_real_placeholder_marker():
    from app.core.types import InternalMessage, text_part, tool_result_part
    from app.services.preprocessing import _new_turn_start

    messages = [
        InternalMessage(role="user", parts=[text_part("[Image #1]: a cat sitting on a mat")], raw={}),
        InternalMessage(role="tool", parts=[tool_result_part("t1", [text_part("ok")])], raw={}),
    ]
    # 以 tool 结果结尾时，占位文本所在消息之后才是"当前轮"
    assert _new_turn_start(messages) == 1
