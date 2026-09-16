"""模型能力元数据功能测试：内置启发式、上游透传、管理员覆盖三层合并。"""
import json

import pytest
from fastapi.testclient import TestClient

from main import app
from app.config import load_config
from app.core.model_capabilities import (
    builtin_capabilities,
    merge_capabilities,
    normalize_capabilities,
    resolve_model_capabilities,
)
from app.database import (
    init_db,
    add_admin,
    add_provider,
    get_db,
    get_provider,
    merge_upstream_model_capabilities,
    set_model_capabilities,
)
from app.security import create_session, hash_password
from app.services.discovery import parse_models, upstream_capabilities

client = TestClient(app)


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
    add_admin("admin", hash_password("secret"), "Admin")
    token = create_session("admin")
    yield {"headers": {"Authorization": f"Bearer {token}"}, "db_path": db_path}


# ---------------------------------------------------------------------------
# 内置启发式
# ---------------------------------------------------------------------------

def test_builtin_capabilities_known_families():
    caps = builtin_capabilities("claude-sonnet-4-5")
    assert caps["context_window"] == 200000
    assert caps["supports_vision"] is True

    caps = builtin_capabilities("deepseek-chat")
    assert caps["supports_vision"] is False
    assert caps["context_window"] == 163840

    caps = builtin_capabilities("gpt-4o-mini")
    assert caps["context_window"] == 128000
    assert caps["supports_vision"] is True

    caps = builtin_capabilities("gemini-2.5-pro")
    assert caps["context_window"] == 1000000


def test_builtin_capabilities_unknown_and_non_chat():
    assert builtin_capabilities("my-custom-model") == {}
    assert builtin_capabilities("text-embedding-3-large") == {}
    assert builtin_capabilities("whisper-1") == {}


def test_builtin_short_markers_use_word_boundaries():
    # "sao10k" 含 "o1" 子串，不得误命中 o 系列（审计 #1）
    assert builtin_capabilities("sao10k/l3.3-euryale-70b") == {}
    assert builtin_capabilities("saO10k/l3-lunaris-8b", "SaO10K") == {}
    # 真正的 o 系列仍命中（连字符分段边界）
    assert builtin_capabilities("o1-mini")["context_window"] == 200000
    assert builtin_capabilities("o3")["supports_vision"] is True
    # 带连字符的长标记保持子串匹配
    assert builtin_capabilities("gpt-4o-2024-08-06")["context_window"] == 128000
    # 非对话短标记同样词边界：sora-2/flux-1 命中，"sorabet" 不误伤
    assert builtin_capabilities("sora-2") == {}
    assert builtin_capabilities("flux-1") == {}


def test_client_entry_omits_false_booleans():
    # 审计 #4：只输出正向声明，"缺失=未知"的既有契约不变
    from app.core.model_capabilities import capabilities_for_client_entry
    entry = capabilities_for_client_entry({"supports_vision": False, "supports_tools": False, "context_window": 100})
    assert "supports_vision" not in entry
    assert "supports_tools" not in entry
    assert entry["context_window"] == 100
    entry = capabilities_for_client_entry({"supports_vision": True})
    assert entry["supports_vision"] is True


def test_normalize_and_merge():
    assert normalize_capabilities({"context_window": "128000", "supports_vision": "yes"}) == {
        "context_window": 128000, "supports_vision": True,
    }
    assert normalize_capabilities({"context_window": -5, "junk": 1}) == {}
    merged = merge_capabilities(
        {"context_window": 100, "supports_vision": False},
        {"context_window": 200},
    )
    assert merged == {"context_window": 200, "supports_vision": False}


def test_resolve_prefers_stored_over_builtin():
    model = {
        "id": "claude-sonnet-4-5",
        "name": "claude",
        "capabilities": {"context_window": 500000, "admin_keys": ["context_window"]},
    }
    resolved = resolve_model_capabilities(model)
    assert resolved["context_window"] == 500000  # 存储值优先
    assert resolved["supports_vision"] is True   # 内置兜底其余键


# ---------------------------------------------------------------------------
# 上游元数据提取
# ---------------------------------------------------------------------------

def test_upstream_capabilities_openrouter_style():
    caps = upstream_capabilities({
        "id": "anthropic/claude-sonnet-4",
        "context_length": 200000,
        "top_provider": {"context_length": 200000, "max_output_tokens": 64000},
        "architecture": {"input_modalities": ["text", "image"]},
        "supported_parameters": ["tools", "response_format"],
        "pricing": {"prompt": "0.000003", "completion": "0.000015"},
    })
    assert caps["context_window"] == 200000
    assert caps["max_output_tokens"] == 64000
    assert caps["supports_vision"] is True
    assert caps["supports_tools"] is True
    assert caps["input_modalities"] == ["text", "image"]
    assert caps["pricing"]["prompt"] == "0.000003"


def test_parse_models_plain_openai_returns_no_caps():
    models = parse_models({"data": [{"id": "gpt-x", "created": 1, "owned_by": "org"}]})
    assert models == [{"id": "gpt-x", "name": "gpt-x"}]


# ---------------------------------------------------------------------------
# DB 读写与 admin_keys 保护
# ---------------------------------------------------------------------------

def _add_test_provider(model_id="m1"):
    add_provider({
        "id": "p1", "name": "P1", "provider_type": "openai",
        "api_base": "http://192.168.1.10:8000", "api_key": "k",
        "models": [{"id": model_id, "name": model_id, "enabled": True}],
    })


def test_set_model_capabilities_roundtrip(temp_db):
    _add_test_provider()
    assert set_model_capabilities("p1/m1", {"context_window": 32000, "supports_vision": True}) is True
    model = next(m for m in get_provider("p1")["models"] if m["id"] == "m1")
    assert model["capabilities"]["context_window"] == 32000
    assert model["capabilities"]["supports_vision"] is True
    assert "context_window" in model["capabilities"]["admin_keys"]
    assert set_model_capabilities("p1/ghost", {"context_window": 1}) is False


def test_upstream_merge_preserves_admin_keys(temp_db):
    _add_test_provider()
    set_model_capabilities("p1/m1", {"context_window": 32000})
    with get_db() as db:
        merge_upstream_model_capabilities(db, "p1", "m1", {
            "context_window": 999000,          # admin 覆盖过 → 不得冲掉
            "max_output_tokens": 4096,          # 未覆盖 → 上游生效
        })
    model = next(m for m in get_provider("p1")["models"] if m["id"] == "m1")
    assert model["capabilities"]["context_window"] == 32000
    assert model["capabilities"]["max_output_tokens"] == 4096


def test_admin_null_clears_override(temp_db):
    _add_test_provider()
    set_model_capabilities("p1/m1", {"context_window": 32000})
    set_model_capabilities("p1/m1", {"context_window": None})
    model = next(m for m in get_provider("p1")["models"] if m["id"] == "m1")
    assert "context_window" not in model["capabilities"]
    assert "context_window" not in (model["capabilities"].get("admin_keys") or [])


# ---------------------------------------------------------------------------
# /v1/models 输出与 admin 端点
# ---------------------------------------------------------------------------

def test_models_endpoint_advertises_capabilities(temp_db):
    from app.database import add_user, add_user_api_key
    add_provider({
        "id": "anth", "name": "Anth", "provider_type": "anthropic",
        "api_base": "https://ai.example", "api_key": "k",
        "models": [
            {"id": "claude-sonnet-4-5", "name": "claude-sonnet-4-5", "enabled": True},
            {"id": "deepseek-chat", "name": "deepseek-chat", "enabled": True},
        ],
    })
    add_user({"username": "alice"})
    add_user_api_key("alice", "default", ["*"])
    from app.database import get_db as _get_db
    with _get_db() as db:
        db.execute("UPDATE user_api_keys SET key = 'user-key' WHERE username = 'alice'")

    r = client.get("/v1/models", headers={"Authorization": "Bearer user-key"})
    assert r.status_code == 200
    data = {m["id"]: m for m in r.json()["data"]}

    claude = data["anth/claude-sonnet-4-5"]
    assert claude["context_window"] == 200000
    assert claude["context_length"] == 200000  # OpenRouter 风格别名
    assert claude["supports_vision"] is True
    assert claude["image_support"] is True
    assert claude["supports_tools"] is True

    deepseek = data["anth/deepseek-chat"]
    assert deepseek["context_window"] == 163840
    assert deepseek.get("supports_vision") is not True
    assert "image_support" not in deepseek


def test_admin_capabilities_endpoint(temp_db):
    _add_test_provider("claude-sonnet-4-5")
    r = client.put("/admin/models/capabilities", headers=temp_db["headers"], json={
        "model_id": "p1/claude-sonnet-4-5",
        "capabilities": {"context_window": 1000000, "supports_tools": False},
    })
    assert r.status_code == 200

    model = next(m for m in get_provider("p1")["models"] if m["id"] == "claude-sonnet-4-5")
    assert model["capabilities"]["context_window"] == 1000000
    assert model["capabilities"]["supports_tools"] is False

    # 非法输入
    r = client.put("/admin/models/capabilities", headers=temp_db["headers"], json={"model_id": "", "capabilities": {}})
    assert r.status_code == 400
    r = client.put("/admin/models/capabilities", headers=temp_db["headers"], json={"model_id": "p1/claude-sonnet-4-5", "capabilities": "x"})
    assert r.status_code == 400
    r = client.put("/admin/models/capabilities", headers=temp_db["headers"], json={"model_id": "p1/ghost", "capabilities": {}})
    assert r.status_code == 404


def test_admin_models_list_includes_capabilities(temp_db):
    _add_test_provider("gpt-4o")
    r = client.get("/admin/models", headers=temp_db["headers"])
    assert r.status_code == 200
    entry = next(m for m in r.json()["models"] if m["id"] == "p1/gpt-4o")
    assert entry["capabilities"]["context_window"] == 128000
    assert entry["capabilities_overridden"] == []


# ---------------------------------------------------------------------------
# 模型刷新时上游能力透传
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refresh_stores_upstream_capabilities(temp_db, monkeypatch):
    from app.services import discovery

    add_provider({
        "id": "or", "name": "OR", "provider_type": "openai",
        "api_base": "http://192.168.1.50:8000", "api_key": "k",
        "models": [],
    })

    async def fake_discover(provider_id):
        return [{
            "id": "rich-model",
            "name": "rich-model",
            "capabilities": {"context_window": 123456, "supports_vision": True},
        }]

    monkeypatch.setattr(discovery, "discover_models", fake_discover)
    await discovery.refresh_provider_models("or")

    model = next(m for m in get_provider("or")["models"] if m["id"] == "rich-model")
    assert model["capabilities"]["context_window"] == 123456
    assert model["capabilities"]["supports_vision"] is True


# ---------------------------------------------------------------------------
# 在线注册表（OpenRouter）
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_registry_cache():
    from app.services import model_registry
    def _clear():
        with model_registry._mem_lock:
            model_registry._mem_cache.update({"map": None, "count": 0, "fetched_at": "", "url": "", "loaded_at": 0.0, "last_attempt_at": 0.0, "last_error": ""})
    _clear()
    yield
    _clear()


def _fake_registry_client(payload):
    body_bytes = json.dumps(payload).encode("utf-8")

    class FakeStreamResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            pass

        async def aiter_bytes(self):
            yield body_bytes

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, **kwargs):
            class Ctx:
                async def __aenter__(self):
                    return FakeStreamResponse()

                async def __aexit__(self, *args):
                    return False
            return Ctx()

    return FakeClient


_REGISTRY_PAYLOAD = {"data": [
    {
        "id": "deepseek/deepseek-flash",
        "context_length": 1048576,
        "architecture": {"input_modalities": ["text", "image"]},
        "top_provider": {"context_length": 1048576, "max_completion_tokens": 943718},
        "supported_parameters": ["tools", "reasoning"],
    },
    {
        "id": "openai/gpt-6-astra",
        "context_length": 1050000,
        "architecture": {"input_modalities": ["file", "image", "text"]},
        "top_provider": {"max_completion_tokens": 128000},
        "supported_parameters": ["tools"],
    },
    {"id": "some/text-only-model", "context_length": 8192, "architecture": {"input_modalities": ["text"]}},
]}


@pytest.mark.asyncio
async def test_registry_fetch_lookup_and_persistence(temp_db, monkeypatch):
    import httpx
    from app.services import model_registry

    _stub_guard(monkeypatch)
    monkeypatch.setattr(httpx, "AsyncClient", _fake_registry_client(_REGISTRY_PAYLOAD))
    result = await model_registry.fetch_and_store_registry()
    assert result["status"] == "ok" and result["models"] == 3

    # 全 id 与裸 slug 都能命中；网关复合 id（provider/model）按裸名匹配
    caps = model_registry.registry_lookup("deepseek/deepseek-flash")
    assert caps["context_window"] == 1048576
    assert caps["supports_vision"] is True
    assert model_registry.registry_lookup("p1/deepseek-flash")["context_window"] == 1048576
    assert model_registry.registry_lookup("gpt-6-astra")["context_window"] == 1050000
    assert model_registry.registry_lookup("totally-unknown-model") == {}

    # 持久化：清掉内存缓存后仍能从 DB 命中（离线可用）
    with model_registry._mem_lock:
        model_registry._mem_cache.update({"map": None, "loaded_at": 0.0})
    assert model_registry.registry_lookup("deepseek-flash")["supports_vision"] is True


@pytest.mark.asyncio
async def test_registry_fetch_failure_keeps_cache(temp_db, monkeypatch):
    import httpx
    from app.services import model_registry

    _stub_guard(monkeypatch)

    class BoomClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("network down")

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", _fake_registry_client(_REGISTRY_PAYLOAD))
    await model_registry.fetch_and_store_registry()
    monkeypatch.setattr(httpx, "AsyncClient", BoomClient)
    result = await model_registry.fetch_and_store_registry()
    assert result["status"] == "error"
    # 失败不清空既有缓存
    assert model_registry.registry_lookup("deepseek-flash")["context_window"] == 1048576


@pytest.mark.asyncio
async def test_registry_failure_backoff(temp_db, monkeypatch):
    """审计 #2：拉取失败后在退避窗口内不得每个维护周期都重试。"""
    import httpx
    from app.services import model_registry

    class BoomClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("network down")

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", BoomClient)
    assert model_registry.registry_needs_refresh() is True  # 从未尝试过 → 需要拉取
    result = await model_registry.fetch_and_store_registry()
    assert result["status"] == "error"

    status = model_registry.registry_status()
    assert status["stale"] is True
    assert status["last_error"]
    # 失败后处于退避窗口 → 维护循环不再触发重试
    assert model_registry.registry_needs_refresh() is False

    # 退避窗口过后恢复重试
    with model_registry._mem_lock:
        model_registry._mem_cache["last_attempt_at"] -= 7200
    assert model_registry.registry_needs_refresh() is True


def test_registry_lookup_map_conflict_rules():
    """审计 #5：正式条目优先于 :variant；同级冲突的裸名键确定性废弃。"""
    from app.services.model_registry import _build_lookup_map

    lookup = _build_lookup_map([
        # 变体在后、能力不同 → 不得覆盖正式条目
        {"id": "inclusionai/ling-3.0-flash-vl", "capabilities": {"context_window": 131072}},
        {"id": "inclusionai/ling-3.0-flash-vl:free", "capabilities": {"context_window": 262144}},
        # 两家厂商同名裸名、能力不同 → 裸名键废弃，全 id 保留
        {"id": "vendor-a/m", "capabilities": {"context_window": 100}},
        {"id": "vendor-b/m", "capabilities": {"context_window": 200}},
    ])
    assert lookup["inclusionai/ling-3.0-flash-vl"]["context_window"] == 131072
    assert lookup["ling-3.0-flash-vl"]["context_window"] == 131072
    assert "m" not in lookup
    assert lookup["vendor-a/m"]["context_window"] == 100
    assert lookup["vendor-b/m"]["context_window"] == 200

    # 顺序颠倒结果不变（确定性）
    lookup2 = _build_lookup_map(list(reversed([
        {"id": "inclusionai/ling-3.0-flash-vl", "capabilities": {"context_window": 131072}},
        {"id": "inclusionai/ling-3.0-flash-vl:free", "capabilities": {"context_window": 262144}},
        {"id": "vendor-a/m", "capabilities": {"context_window": 100}},
        {"id": "vendor-b/m", "capabilities": {"context_window": 200}},
    ])))
    assert lookup2 == lookup


@pytest.mark.asyncio
async def test_registry_fetch_rejects_unsafe_url(temp_db):
    """审计 #7：注册表 URL 也要过 url_guard（元数据地址/非法 scheme 拒绝）。"""
    from app.services import model_registry

    result = await model_registry.fetch_and_store_registry("http://169.254.169.254/latest/meta-data")
    assert result["status"] == "error"
    result = await model_registry.fetch_and_store_registry("file:///etc/passwd")
    assert result["status"] == "error"


def test_registry_remote_beats_builtin_but_loses_to_admin(temp_db):
    from app.core.model_capabilities import resolve_model_capabilities

    model = {"id": "deepseek-chat", "name": "", "capabilities": {}}
    remote = {"context_window": 163840}
    resolved = resolve_model_capabilities(model, remote=remote)
    assert resolved["context_window"] == 163840  # 在线注册表 > 内置表

    model["capabilities"] = {"context_window": 32000, "admin_keys": ["context_window"]}
    resolved = resolve_model_capabilities(model, remote=remote)
    assert resolved["context_window"] == 32000  # 管理员覆盖 > 在线注册表


def test_models_endpoint_uses_online_registry(temp_db, monkeypatch):
    import httpx
    from app.services import model_registry
    from app.database import add_user, add_user_api_key

    monkeypatch.setattr(httpx, "AsyncClient", _fake_registry_client(_REGISTRY_PAYLOAD))
    from app.database import save_model_registry
    from app.services.discovery import upstream_capabilities
    entries = []
    for item in _REGISTRY_PAYLOAD["data"]:
        caps = upstream_capabilities(item)
        if caps:
            entries.append({"id": item["id"], "capabilities": caps})
    save_model_registry("fake", entries, "2026-09-14 00:00:00")

    add_provider({
        "id": "ds", "name": "DS", "provider_type": "openai",
        "api_base": "http://192.168.1.60:8000", "api_key": "k",
        "models": [{"id": "deepseek-flash", "name": "deepseek-flash", "enabled": True}],
    })
    add_user({"username": "bob"})
    add_user_api_key("bob", "default", ["*"])
    with get_db() as db:
        db.execute("UPDATE user_api_keys SET key = 'user-key' WHERE username = 'bob'")

    r = client.get("/v1/models", headers={"Authorization": "Bearer user-key"})
    assert r.status_code == 200
    entry = next(m for m in r.json()["data"] if m["id"] == "ds/deepseek-flash")
    assert entry["context_window"] == 1048576
    assert entry["supports_vision"] is True
    assert entry["image_support"] is True


def test_admin_registry_endpoints(temp_db, monkeypatch):
    import httpx
    _stub_guard(monkeypatch)
    monkeypatch.setattr(httpx, "AsyncClient", _fake_registry_client(_REGISTRY_PAYLOAD))

    r = client.post("/admin/models/registry/refresh", headers=temp_db["headers"])
    assert r.status_code == 200
    assert r.json()["models"] == 3

    r = client.get("/admin/models/registry/status", headers=temp_db["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["model_count"] == 3
    assert body["stale"] is False


# ---------------------------------------------------------------------------
# 第二轮审计：严格布尔解析 / 整数钳制 / 守卫细节
# ---------------------------------------------------------------------------

def _stub_guard(monkeypatch):
    """测试隔离：跳过 url_guard 的 DNS 解析（不联网）。"""
    from app.services import url_guard

    async def _passthrough(url, **kwargs):
        return url

    monkeypatch.setattr(url_guard, "validate_upstream_url_async", _passthrough)


def test_normalize_strict_bool_parsing():
    from app.core.model_capabilities import normalize_capabilities

    # 字符串 "false"/"0"/"no" 必须解析为 False，绝不能 bool() 强转成 True
    assert normalize_capabilities({"supports_vision": "false"}) == {"supports_vision": False}
    assert normalize_capabilities({"supports_vision": "0", "supports_tools": "no"}) == {
        "supports_vision": False, "supports_tools": False,
    }
    assert normalize_capabilities({"supports_vision": "true", "supports_tools": 1}) == {
        "supports_vision": True, "supports_tools": True,
    }
    # 无法识别的值丢弃该键（保持"未知"），而不是猜测
    assert normalize_capabilities({"supports_vision": "maybe"}) == {}
    assert normalize_capabilities({"supports_vision": 2}) == {}
    # bool 是 int 子类：True 不得变成 context_window=1
    assert normalize_capabilities({"context_window": True}) == {}


def test_normalize_int_limits():
    from app.core.model_capabilities import normalize_capabilities

    # 异常大值丢弃，不得直出客户端
    assert normalize_capabilities({"context_window": 10 ** 12}) == {}
    assert normalize_capabilities({"max_output_tokens": -1}) == {}
    assert normalize_capabilities({"context_window": "200000"}) == {"context_window": 200000}


def test_upstream_capabilities_string_false_not_amplified():
    caps = upstream_capabilities({"id": "m", "supports_vision": "false"})
    assert caps.get("supports_vision") is not True
    caps = upstream_capabilities({"id": "m", "capabilities": {"vision": "no"}})
    assert caps.get("supports_vision") is not True


def test_lookup_map_variant_only_conflict_dropped():
    """仅有变体条目（无正式条目）且能力不同时，同键冲突也必须确定性废弃。"""
    from app.services.model_registry import _build_lookup_map

    lookup = _build_lookup_map([
        {"id": "vendor/m:free", "capabilities": {"context_window": 100}},
        {"id": "vendor/m:extended", "capabilities": {"context_window": 200}},
    ])
    assert "m" not in lookup
    assert "vendor/m" not in lookup
    # 顺序颠倒结果一致
    lookup2 = _build_lookup_map([
        {"id": "vendor/m:extended", "capabilities": {"context_window": 200}},
        {"id": "vendor/m:free", "capabilities": {"context_window": 100}},
    ])
    assert lookup2 == lookup


@pytest.mark.asyncio
async def test_registry_fetch_enforces_size_limit(temp_db, monkeypatch):
    import httpx
    from app.services import model_registry

    _stub_guard(monkeypatch)
    monkeypatch.setattr(model_registry, "_MAX_REGISTRY_BYTES", 16)

    class BigResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            pass

        async def aiter_bytes(self):
            yield b"x" * 64

    class BigClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, **kwargs):
            class Ctx:
                async def __aenter__(self):
                    return BigResponse()

                async def __aexit__(self, *args):
                    return False
            return Ctx()

    monkeypatch.setattr(httpx, "AsyncClient", BigClient)
    result = await model_registry.fetch_and_store_registry("https://registry.test/models")
    assert result["status"] == "error"
    assert "exceeds" in result["error"]


@pytest.mark.asyncio
async def test_registry_fetch_enforces_redirect_limit(temp_db, monkeypatch):
    import httpx
    from app.services import model_registry

    _stub_guard(monkeypatch)

    class RedirectResponse:
        status_code = 302
        headers = {"location": "https://registry.test/next"}

        def raise_for_status(self):
            pass

        async def aiter_bytes(self):
            yield b""

    class RedirectClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, **kwargs):
            class Ctx:
                async def __aenter__(self):
                    return RedirectResponse()

                async def __aexit__(self, *args):
                    return False
            return Ctx()

    monkeypatch.setattr(httpx, "AsyncClient", RedirectClient)
    result = await model_registry.fetch_and_store_registry("https://registry.test/models")
    assert result["status"] == "error"
    assert "redirect" in result["error"]


def test_registry_status_shape_when_disabled(temp_db, monkeypatch):
    from app.services import model_registry

    monkeypatch.setattr(model_registry, "registry_enabled", lambda: False)
    status = model_registry.registry_status()
    # 禁用时也要保持完整字段契约
    for key in ("enabled", "url", "fetched_at", "model_count", "ttl_seconds", "stale", "last_error"):
        assert key in status
    assert status["enabled"] is False
    assert model_registry.registry_lookup("deepseek-flash") == {}
    assert model_registry.registry_needs_refresh() is False


# ---------------------------------------------------------------------------
# 第三轮审计：读取侧归一化 / bool 收口 / 限额表一致性 / 禁用态刷新提示
# ---------------------------------------------------------------------------

def test_resolve_renormalizes_persisted_bad_values():
    """审计 #1：DB 残留的旧版异常值（bool 强转/无上限时代写入）不得直出客户端。"""
    from app.core.model_capabilities import capabilities_for_client_entry, resolve_model_capabilities

    model = {
        "id": "m", "name": "",
        "capabilities": {"context_window": 10 ** 12, "supports_vision": "yes-ish"},
    }
    resolved = resolve_model_capabilities(model, remote={"max_output_tokens": 10 ** 15})
    entry = capabilities_for_client_entry(resolved)
    assert "context_window" not in entry
    assert "context_length" not in entry
    assert "max_output_tokens" not in entry
    assert "supports_vision" not in entry


def test_upstream_capabilities_rejects_bool_ints():
    """审计 #2：context_length: true 不得变成 1 token 上下文。"""
    assert upstream_capabilities({"id": "m", "context_length": True}) == {}
    assert upstream_capabilities({"id": "m", "top_provider": {"max_completion_tokens": False}}) == {}
    # 正常数值不受影响
    assert upstream_capabilities({"id": "m", "context_length": 8192}) == {"context_window": 8192}


def test_int_limits_table_covers_all_int_keys():
    """审计 #4：_INT_KEYS 与 _INT_LIMITS 键集合必须一致，防止将来加键漏配。"""
    from app.core.model_capabilities import _INT_KEYS, _INT_LIMITS
    assert set(_INT_KEYS) == set(_INT_LIMITS)


def test_normalize_pricing_rejects_bool():
    """审计 #6：{"pricing": {"prompt": true}} 不得变成 "True"。"""
    from app.core.model_capabilities import normalize_capabilities
    assert normalize_capabilities({"pricing": {"prompt": True, "completion": "1.5"}}) == {
        "pricing": {"completion": "1.5"},
    }


def test_refresh_endpoint_notes_when_registry_disabled(temp_db, monkeypatch):
    """审计 #5：禁用状态下手动刷新成功也要明确提示结果不会生效。"""
    import httpx
    from app.services import model_registry

    _stub_guard(monkeypatch)
    monkeypatch.setattr(httpx, "AsyncClient", _fake_registry_client(_REGISTRY_PAYLOAD))
    monkeypatch.setattr(model_registry, "registry_enabled", lambda: False)

    r = client.post("/admin/models/registry/refresh", headers=temp_db["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "NOT applied" in body.get("note", "")


# ---------------------------------------------------------------------------
# 第四轮审计：非法输入显式报错 / overridden 交集 / 禁用态失败提示
# ---------------------------------------------------------------------------

def test_set_model_capabilities_rejects_invalid_values(temp_db):
    """审计 #1：非法值不得静默 no-op 还返回 True。"""
    _add_test_provider("gpt-4o")
    with pytest.raises(ValueError):
        set_model_capabilities("p1/gpt-4o", {"context_window": 200000000})  # 超上限
    with pytest.raises(ValueError):
        set_model_capabilities("p1/gpt-4o", {"context_window": 0})           # 非正数
    with pytest.raises(ValueError):
        set_model_capabilities("p1/gpt-4o", {"supports_vision": "maybe"})    # 无法解析的布尔
    with pytest.raises(ValueError):
        set_model_capabilities("p1/gpt-4o", {"unknown_key": 1})              # 未知键
    # 合法输入与 null 清除不受影响
    assert set_model_capabilities("p1/gpt-4o", {"context_window": 128000}) is True
    assert set_model_capabilities("p1/gpt-4o", {"context_window": None}) is True


def test_admin_capabilities_endpoint_returns_400_for_invalid(temp_db):
    _add_test_provider("gpt-4o")
    r = client.put("/admin/models/capabilities", headers=temp_db["headers"], json={
        "model_id": "p1/gpt-4o", "capabilities": {"context_window": 0},
    })
    assert r.status_code == 400
    assert "invalid capability values" in r.json()["detail"]


def test_capabilities_overridden_intersects_resolved(temp_db):
    """审计 #2：历史脏行的 admin_keys 指向已丢弃的值时，不得虚报"已覆盖"。"""
    _add_test_provider("m1")
    with get_db() as db:
        db.execute(
            "UPDATE provider_models SET capabilities = ? WHERE provider_id = 'p1' AND model_id = 'm1'",
            (json.dumps({"context_window": "abc", "admin_keys": ["context_window"]}),),
        )
    r = client.get("/admin/models", headers=temp_db["headers"])
    assert r.status_code == 200
    entry = next(m for m in r.json()["models"] if m["id"] == "p1/m1")
    assert entry["capabilities_overridden"] == []
    assert "context_window" not in entry["capabilities"] or isinstance(entry["capabilities"].get("context_window"), int)


def test_refresh_failure_mentions_disabled_state(temp_db, monkeypatch):
    """审计 #3：禁用 + 拉取失败时，502 detail 必须带上禁用上下文。"""
    import httpx
    from app.services import model_registry

    _stub_guard(monkeypatch)

    class BoomClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("network down")

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", BoomClient)
    monkeypatch.setattr(model_registry, "registry_enabled", lambda: False)

    r = client.post("/admin/models/registry/refresh", headers=temp_db["headers"])
    assert r.status_code == 502
    assert "model_registry_enabled=false" in r.json()["detail"]


# ---------------------------------------------------------------------------
# supports_reasoning 能力字段
# ---------------------------------------------------------------------------

def test_builtin_reasoning_flags():
    assert builtin_capabilities("deepseek-flash")["supports_reasoning"] is True
    assert builtin_capabilities("gpt-5.6-luna")["supports_reasoning"] is True
    assert builtin_capabilities("o3-mini")["supports_reasoning"] is True
    assert builtin_capabilities("qwen3-coder")["supports_reasoning"] is True
    # 未确认的家族不猜测（键缺失 = 未知）
    assert "supports_reasoning" not in builtin_capabilities("gpt-4o")
    assert "supports_reasoning" not in builtin_capabilities("qwen2.5-7b-instruct")


def test_upstream_capabilities_reasoning_extraction():
    # OpenRouter 风格：supported_parameters 含 reasoning/include_reasoning
    assert upstream_capabilities({"id": "m", "supported_parameters": ["tools", "include_reasoning"]})["supports_reasoning"] is True
    # 明确不带推理参数 → 显式 False
    assert upstream_capabilities({"id": "m", "supported_parameters": ["tools", "temperature"]})["supports_reasoning"] is False
    # 无任何信号 → 不输出该键
    assert "supports_reasoning" not in upstream_capabilities({"id": "m", "context_length": 8192})
    # 字符串 "false" 不得被强转成 True
    assert upstream_capabilities({"id": "m", "supports_reasoning": "false"})["supports_reasoning"] is False


def test_client_entry_reasoning_positive_only():
    from app.core.model_capabilities import capabilities_for_client_entry
    assert capabilities_for_client_entry({"supports_reasoning": True})["supports_reasoning"] is True
    assert "supports_reasoning" not in capabilities_for_client_entry({"supports_reasoning": False})


def test_admin_reasoning_override_roundtrip(temp_db):
    """管理员显式选"不支持"推理 → 覆盖内置 True，客户端不再声明该字段。"""
    _add_test_provider("deepseek-flash")
    r = client.put("/admin/models/capabilities", headers=temp_db["headers"], json={
        "model_id": "p1/deepseek-flash", "capabilities": {"supports_reasoning": False},
    })
    assert r.status_code == 200
    model = next(m for m in get_provider("p1")["models"] if m["id"] == "deepseek-flash")
    assert model["capabilities"]["supports_reasoning"] is False
    assert "supports_reasoning" in model["capabilities"]["admin_keys"]
