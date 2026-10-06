"""bug-2026-10-05 二次审查 R 系列回归测试（R-1 ~ R-6）。

R-1: 请求时钉 IP 已接入网关自有的 httpx 上游路径（含降级/缓存/错误形态）。
R-2: imagegen allow_private 开关不再绕过恒封地址检查。
R-3: 生图预算 continuation 轮按张扣减、退款与扣减配对、过期窗口不复活。
R-4: provider 健康检查并发有上限。
R-5: discovery 体积上限在流式读取过程中生效（不再先全量入内存）。
R-6: lifespan 启动段重置停机标志。
"""

import asyncio
import json

import pytest

import app.services.url_guard as url_guard
from app.services.url_guard import pinned_request


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fake_stream_client(body=b"\x89PNG ok", status=200, headers=None):
    class Resp:
        def __init__(self):
            self.status_code = status
            self.headers = headers if headers is not None else {"content-type": "image/png"}

        async def aiter_bytes(self):
            yield body

        def raise_for_status(self):
            return None

    class Ctx:
        def __init__(self, resp):
            self._resp = resp

        async def __aenter__(self):
            return self._resp

        async def __aexit__(self, *args):
            return False

    class Client:
        def __init__(self):
            self.calls = []

        def stream(self, method, url, **kwargs):
            self.calls.append((url, kwargs))
            return Ctx(Resp())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    return Client()


# ---------------------------------------------------------------------------
# R-1: pinned_request 语义
# ---------------------------------------------------------------------------

@pytest.fixture
def chat_image_db(tmp_path, monkeypatch):
    # 与 tests/test_chat_image_bridge.py 同名 fixture 同构：桥接路径的
    # 最小可用环境（provider/用户/生图器/会话 key 固定）。
    from app.database import (
        add_admin, add_provider, add_user, add_user_api_key, init_db,
        set_model_image_generation, upsert_image_generator,
    )
    from app.security import hash_password

    db_path = str(tmp_path / "gateway.db")
    previous_path = None
    previous_initialized = None
    import app.database as db_mod
    previous_path, previous_initialized = db_mod.DB_PATH, db_mod._initialized
    init_db(db_path)
    add_admin("admin", hash_password("secret"), "Admin")
    add_user({"username": "alice", "display_name": "Alice", "enabled": True})
    key = add_user_api_key("alice", "default", ["chat/chat-model"])["key"]
    add_provider({
        "id": "chat", "name": "Chat", "api_base": "http://chat.test/v1",
        "models": [{"id": "chat-model", "name": "Chat Model"}],
    })
    set_model_image_generation("chat/chat-model", True)
    upsert_image_generator(
        "default", {"api_base": "http://image.test/v1", "model": "image-model", "enabled": True}
    )
    image_dir = tmp_path / "generated-images"
    monkeypatch.setattr("app.core.image_results.image_result_directory", lambda: image_dir)
    monkeypatch.setattr(
        "app.router.proxy._conversation_cache_key",
        lambda *args, **kwargs: "test:conv:key",
    )
    try:
        yield {
            "headers": {"Authorization": f"Bearer {key}"},
            "image_dir": image_dir,
            "conv_key": "test:conv:key",
        }
    finally:
        db_mod.DB_PATH = previous_path
        db_mod._initialized = previous_initialized

@pytest.mark.asyncio
async def test_pinned_request_rewrites_hostname_to_resolved_ip(monkeypatch):
    monkeypatch.setattr(url_guard, "_resolve_addresses", lambda host: ["93.184.216.34"])
    target = await pinned_request("https://api.wired.test/v1/messages")
    assert target.rewritten is True
    assert target.url == "https://93.184.216.34/v1/messages"
    assert target.headers == {"Host": "api.wired.test"}
    assert target.extensions == {"sni_hostname": "api.wired.test"}


@pytest.mark.asyncio
async def test_pinned_request_passthrough_on_unresolvable(monkeypatch):
    # conftest 默认把所有主机名解析为 NXDOMAIN：降级为原样放行，
    # 请求参数与钉 IP 接线前完全一致。
    target = await pinned_request("https://nowhere.invalid/v1/messages")
    assert target.rewritten is False
    assert target.url == "https://nowhere.invalid/v1/messages"
    assert target.headers == {}
    assert target.extensions == {}


@pytest.mark.asyncio
async def test_pinned_request_blocked_address_raises_connection_error(monkeypatch):
    # rebinding 已发生（解析命中恒封地址）：必须以连接失败形态出现，
    # 让 fallback 的 connection_error 触发器继续生效，而不是降级放行。
    monkeypatch.setattr(url_guard, "_resolve_addresses", lambda host: ["169.254.169.254"])
    with pytest.raises(ConnectionError):
        await pinned_request("https://rebind.test/v1")


@pytest.mark.asyncio
async def test_pinned_request_ip_literal_untouched():
    target = await pinned_request("http://127.0.0.1:8080/v1/messages")
    assert target.rewritten is False
    assert target.url == "http://127.0.0.1:8080/v1/messages"
    assert target.headers == {}
    assert target.extensions == {}


@pytest.mark.asyncio
async def test_pinned_request_caches_resolution(monkeypatch):
    calls = []

    def fake_resolve(host):
        calls.append(host)
        return ["93.184.216.34"]

    monkeypatch.setattr(url_guard, "_resolve_addresses", fake_resolve)
    await pinned_request("https://cached.test/v1")
    await pinned_request("https://cached.test/v1")
    assert calls == ["cached.test"]


@pytest.mark.asyncio
async def test_pinned_request_skips_rewriting_behind_env_proxy(monkeypatch):
    # 环境配置了代理时不钉 IP：改写 URL 会破坏代理的路由/证书语义。
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
    monkeypatch.setattr(url_guard, "_resolve_addresses", lambda host: ["93.184.216.34"])
    target = await pinned_request("https://proxied.test/v1")
    assert target.rewritten is False
    assert target.url == "https://proxied.test/v1"


def test_pinned_target_rewrite_url_replaces_only_netloc():
    target = url_guard.PinnedTarget(
        url="https://1.2.3.4:8443/prompt", host_header="host.test:8443",
        sni_hostname="host.test", rewritten=True,
    )
    assert target.rewrite_url("https://host.test:8443/view?x=1") == "https://1.2.3.4:8443/view?x=1"
    plain = url_guard.PinnedTarget(url="https://host.test/prompt", host_header="", sni_hostname="")
    assert plain.rewrite_url("https://host.test/view") == "https://host.test/view"


# ---------------------------------------------------------------------------
# R-1: 接线到真实请求路径
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_adapter_posts_to_pinned_ip(monkeypatch):
    from app.adapters import anthropic as anthropic_adapter

    monkeypatch.setattr(url_guard, "_resolve_addresses", lambda host: ["93.184.216.34"])
    captured = {}

    class Resp:
        status_code = 500
        text = "boom"

        def json(self):
            return {"error": {"message": "boom"}}

    class Client:
        async def post(self, url, headers=None, json=None, **extra):
            captured["url"] = url
            captured["headers"] = headers or {}
            captured["extra"] = extra
            return Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(anthropic_adapter, "shared_client", lambda *args, **kwargs: Client())
    with pytest.raises(Exception):
        await anthropic_adapter.anthropic_messages_completion(
            {"id": "p", "api_base": "https://api.wired.test/v1", "api_key": "sk-k"},
            [{"role": "user", "content": "hi"}], {}, 100, None, "model",
        )
    assert captured["url"] == "https://93.184.216.34/v1/messages"
    assert captured["headers"]["Host"] == "api.wired.test"
    assert captured["headers"]["x-api-key"] == "sk-k"
    assert captured["extra"]["extensions"] == {"sni_hostname": "api.wired.test"}


@pytest.mark.asyncio
async def test_anthropic_adapter_unresolvable_host_keeps_original_request(monkeypatch):
    from app.adapters import anthropic as anthropic_adapter

    captured = {}

    class Resp:
        status_code = 500
        text = "boom"

        def json(self):
            return {"error": {"message": "boom"}}

    class Client:
        async def post(self, url, headers=None, json=None, **extra):
            captured["url"] = url
            captured["headers"] = headers or {}
            captured["extra"] = extra
            return Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(anthropic_adapter, "shared_client", lambda *args, **kwargs: Client())
    with pytest.raises(Exception):
        await anthropic_adapter.anthropic_messages_completion(
            {"id": "p", "api_base": "https://nowhere.invalid/v1", "api_key": "sk-k"},
            [{"role": "user", "content": "hi"}], {}, 100, None, "model",
        )
    # 降级放行：URL、headers、kwargs 与钉 IP 接线前一致。
    assert captured["url"] == "https://nowhere.invalid/v1/messages"
    assert "Host" not in captured["headers"]
    assert "extensions" not in captured["extra"]


@pytest.mark.asyncio
async def test_discover_models_requests_pinned_ip(monkeypatch):
    from app.services.discovery import discover_models

    provider = {
        "id": "p", "enabled": True, "api_base": "https://api.wired.test/v1",
        "api_key": "k", "provider_type": "openai",
    }
    monkeypatch.setattr("app.services.discovery.get_provider", lambda pid: provider)
    monkeypatch.setattr(url_guard, "_resolve_addresses", lambda host: ["93.184.216.34"])

    client = _fake_stream_client(body=b'{"data": [{"id": "m1"}]}')
    monkeypatch.setattr("app.services.discovery.httpx.AsyncClient", lambda *a, **k: client)

    models = await discover_models("p")
    assert [m["id"] for m in models] == ["m1"]
    url, kwargs = client.calls[0]
    assert url == "https://93.184.216.34/v1/models"
    assert kwargs["headers"]["Host"] == "api.wired.test"
    assert kwargs["extensions"] == {"sni_hostname": "api.wired.test"}


# ---------------------------------------------------------------------------
# R-2: imagegen allow_private 不绕过恒封检查
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_allow_private_download_still_blocks_metadata_ip_literal():
    from app.adapters.imagegen import _download_image

    client = _fake_stream_client()
    with pytest.raises(ValueError):
        await _download_image(client, "http://169.254.169.254/latest/meta-data", allow_private_hosts=True)
    assert client.calls == []


@pytest.mark.asyncio
async def test_allow_private_download_still_blocks_metadata_hostname():
    from app.adapters.imagegen import _download_image

    client = _fake_stream_client()
    with pytest.raises(ValueError):
        await _download_image(client, "http://metadata.google.internal/x", allow_private_hosts=True)
    assert client.calls == []


@pytest.mark.asyncio
async def test_allow_private_download_blocks_rebind_to_link_local(monkeypatch):
    from app.adapters.imagegen import _download_image

    monkeypatch.setattr(
        "app.adapters.imagegen.socket.getaddrinfo",
        lambda host, *args, **kwargs: [(2, 1, 6, "", ("169.254.169.254", 0))],
    )
    client = _fake_stream_client()
    with pytest.raises(ValueError):
        await _download_image(client, "http://evil.test/img.png", allow_private_hosts=True)
    assert client.calls == []


@pytest.mark.asyncio
async def test_allow_private_download_allows_private_and_pins_ip(monkeypatch):
    from app.adapters.imagegen import _download_image

    monkeypatch.setattr(
        "app.adapters.imagegen.socket.getaddrinfo",
        lambda host, *args, **kwargs: [(2, 1, 6, "", ("10.0.0.9", 0))],
    )
    client = _fake_stream_client()
    data_uri, mime = await _download_image(client, "http://internal.test/img.png", allow_private_hosts=True)
    assert mime == "image/png"
    url, kwargs = client.calls[0]
    assert url == "http://10.0.0.9/img.png"
    assert kwargs["headers"]["Host"] == "internal.test"


# ---------------------------------------------------------------------------
# R-3: 预算退款与扣减配对
# ---------------------------------------------------------------------------

def test_refund_skips_expired_budget_window():
    from app.core.state import image_generation_budget, charge_image_generation_budget, refund_image_generation_budget

    key = "r3:expired"
    image_generation_budget._data.pop(key, None)
    image_generation_budget._timestamps.pop(key, None)
    try:
        assert charge_image_generation_budget(key, 3) == 3
        # 强制条目过期（时间戳早于 TTL）
        image_generation_budget._timestamps[key] -= image_generation_budget.ttl + 1
        refund_image_generation_budget(key, 3)
        # 过期条目被丢弃而不是被旧请求写回复活
        assert key not in image_generation_budget._data
    finally:
        image_generation_budget._data.pop(key, None)
        image_generation_budget._timestamps.pop(key, None)


def test_refund_clamps_to_zero_within_window():
    from app.core.state import image_generation_budget, charge_image_generation_budget, refund_image_generation_budget

    key = "r3:clamp"
    image_generation_budget._data.pop(key, None)
    image_generation_budget._timestamps.pop(key, None)
    try:
        assert charge_image_generation_budget(key, 2) == 2
        refund_image_generation_budget(key, 5)
        assert int(image_generation_budget._data.get(key) or 0) == 0
    finally:
        image_generation_budget._data.pop(key, None)
        image_generation_budget._timestamps.pop(key, None)


@pytest.mark.asyncio
async def test_continuation_round_charges_budget_and_refunds_only_charged(chat_image_db, monkeypatch):
    """R-3 核心场景：初始批 1 张成功（扣 1），continuation 请求 2 张但预算只允许 1 张。

    修复前：continuation 轮不扣预算、失败却退款，used 被凭空抬高归还；
    修复后：continuation 按张扣减，失败只退本轮扣过的张数，窗口末值 == 1。
    """
    from main import app
    from fastapi.testclient import TestClient
    from app.core.output import InternalOutputMessage, InternalToolCallOutput
    from app.core.image_bridge import IMAGE_BRIDGE_TOOL_NAME
    from app.core.state import image_generation_budget
    from app.adapters.imagegen import ImageGenerationResult

    image_generation_budget._data.clear()
    image_generation_budget._timestamps.clear()
    monkeypatch.setattr(
        "app.core.state.get_default",
        lambda key, fallback=None: 2 if key == "image_generation_budget_limit" else fallback,
    )

    model_calls = []

    def _tool_call(call_id, prompt):
        return InternalToolCallOutput(
            id=call_id, call_id=call_id, name=IMAGE_BRIDGE_TOOL_NAME,
            arguments=json.dumps({"prompt": prompt}, ensure_ascii=False),
        )

    async def fake_model(*args, **kwargs):
        model_calls.append(kwargs.get("log_label", "initial"))
        if len(model_calls) == 1:
            return InternalOutputMessage(
                tool_calls=[_tool_call("call_a", "A")], finish_reason="tool_calls",
            ), {"id": "chat"}, "chat"
        if len(model_calls) == 2:
            # continuation 一次要 2 张：预算只剩 1，应被截断为 1 张执行。
            return InternalOutputMessage(
                tool_calls=[_tool_call("call_b", "B"), _tool_call("call_c", "C")],
                finish_reason="tool_calls",
            ), {"id": "chat"}, "chat"
        return InternalOutputMessage(text="done", finish_reason="stop"), {"id": "chat"}, "chat"

    generated = []

    async def fake_generate(config, **kwargs):
        prompt = kwargs["prompt"]
        generated.append(prompt)
        if prompt != "A":
            raise RuntimeError("backend exploded")
        return [ImageGenerationResult("data:image/png;base64,AAAA")]

    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_model)
    monkeypatch.setattr("app.router.proxy.generate_images", fake_generate)

    response = TestClient(app).post(
        "/v1/chat/completions", headers=chat_image_db["headers"],
        json={"model": "chat/chat-model", "messages": [{"role": "user", "content": "生成一个苹果"}]},
    )
    assert response.status_code == 200, response.text
    # C 被预算截断，从未执行；B 执行且失败。
    assert generated == ["A", "B"]
    # A 成功消耗 1 张；B 被扣 1 又因失败退回 1 → 窗口末值应为 1（只剩 A 的消耗）。
    # 修复前这里会是 0：B 从未被扣减，退款却凭空抬高了剩余额度。
    assert sum(int(v) for v in image_generation_budget._data.values()) == 1


# ---------------------------------------------------------------------------
# R-4: 健康检查并发上限
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_all_provider_health_caps_concurrency(monkeypatch):
    from app.services import discovery

    providers = [{"id": f"p{i}"} for i in range(20)]
    monkeypatch.setattr(discovery, "get_providers", lambda: providers)

    state = {"live": 0, "peak": 0}

    async def fake_check(provider_id, timeout=10.0):
        state["live"] += 1
        state["peak"] = max(state["peak"], state["live"])
        await asyncio.sleep(0.01)
        state["live"] -= 1
        return {"provider_id": provider_id, "ok": True}

    monkeypatch.setattr(discovery, "check_provider_health", fake_check)
    results = await discovery.check_all_provider_health()
    assert len(results) == 20
    assert state["peak"] <= discovery._HEALTH_CHECK_CONCURRENCY


# ---------------------------------------------------------------------------
# R-5: 体积上限在流式读取中生效
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_capped_aborts_stream_without_buffering_whole_body():
    from app.services.discovery import _get_capped, _MAX_DISCOVERY_BYTES

    seen = {"chunks": 0}

    class Resp:
        status_code = 200
        headers = {}

        async def aiter_bytes(self):
            while True:
                seen["chunks"] += 1
                yield b"x" * (1024 * 1024)

    class Ctx:
        async def __aenter__(self):
            return Resp()

        async def __aexit__(self, *args):
            return False

    class Client:
        def stream(self, method, url, **kwargs):
            return Ctx()

    with pytest.raises(RuntimeError):
        await _get_capped(Client(), "https://huge.test/models")
    # 只多读一个 chunk 就中断，而不是把无界 body 全部持有。
    assert seen["chunks"] <= _MAX_DISCOVERY_BYTES // (1024 * 1024) + 2


@pytest.mark.asyncio
async def test_get_capped_fast_fails_on_declared_content_length():
    from app.services.discovery import _get_capped, _MAX_DISCOVERY_BYTES

    class Resp:
        status_code = 200
        headers = {"content-length": str(_MAX_DISCOVERY_BYTES + 1)}

        async def aiter_bytes(self):
            raise AssertionError("Content-Length 已超限时不应再读 body")
            yield b""

    class Ctx:
        async def __aenter__(self):
            return Resp()

        async def __aexit__(self, *args):
            return False

    class Client:
        def stream(self, method, url, **kwargs):
            return Ctx()

    with pytest.raises(RuntimeError):
        await _get_capped(Client(), "https://huge.test/models")


# ---------------------------------------------------------------------------
# R-6: lifespan 启动段重置停机标志
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lifespan_resets_shutting_down_flag(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import main as main_module
    from app.core import outcome
    import app.database as db_mod

    previous_db_path = db_mod.DB_PATH
    previous_initialized = db_mod._initialized
    cfg = SimpleNamespace(config={"database": str(tmp_path / "r6.db"), "logging": None})
    monkeypatch.setattr(main_module, "load_config", lambda *args, **kwargs: cfg)
    try:
        outcome.set_shutting_down(True)
        async with main_module.lifespan(None):
            assert outcome.is_shutting_down() is False
        # 退出段仍然置位（L-18 归因语义不变）
        assert outcome.is_shutting_down() is True
    finally:
        outcome.set_shutting_down(False)
        db_mod.DB_PATH = previous_db_path
        db_mod._initialized = previous_initialized
