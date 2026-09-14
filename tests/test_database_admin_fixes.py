"""数据库/管理端审查修复的回归测试。

覆盖：update_provider 不清空 preprocessor、迁移幂等、密码更新返回值、
日志保留清理、admin 输入校验（400 而非 500）、allowed_models 归一化、
请求日志 images_generations 过滤、请求体限制可禁用、模型刷新保留配置。
"""
import json
import sqlite3
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from main import app
from app.config import load_config
from app.database import (
    init_db,
    add_admin,
    add_provider,
    add_user,
    get_db,
    get_provider,
    update_provider,
    update_admin_password,
)
from app.security import create_session, hash_password

client = TestClient(app)


def _set_model_preprocessor(provider_id: str, model_id: str, value: str = "1") -> None:
    with get_db() as db:
        db.execute(
            "UPDATE provider_models SET preprocessor = ? WHERE provider_id = ? AND model_id = ?",
            (value, provider_id, model_id),
        )


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
# #1 update_provider 不得清空模型 preprocessor 标记
# ---------------------------------------------------------------------------

def test_update_provider_models_without_preprocessor_keeps_flag(temp_db):
    add_provider({
        "id": "p1", "name": "P1", "provider_type": "openai",
        "api_base": "http://192.168.1.10:8000", "api_key": "k",
        "models": [{"id": "m1", "name": "m1", "enabled": True}],
    })
    _set_model_preprocessor("p1", "m1", "1")

    # 模拟 Pydantic ModelInfo 剥离后的载荷：只有 id/name/enabled
    updated = update_provider("p1", {"models": [{"id": "m1", "name": "m1", "enabled": True}]})
    model = next(m for m in updated["models"] if m["id"] == "m1")
    assert model["preprocessor"] == "1", "preprocessor 标记被静默清空"

    # 显式携带 preprocessor 时仍可更新
    updated = update_provider("p1", {"models": [{"id": "m1", "name": "m1", "enabled": True, "preprocessor": "other"}]})
    model = next(m for m in updated["models"] if m["id"] == "m1")
    assert model["preprocessor"] == "other"


# ---------------------------------------------------------------------------
# #2 provider options/headers 迁移必须幂等
# ---------------------------------------------------------------------------

def test_provider_options_migration_idempotent(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    # 构造带旧 extra_headers 列的库
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE providers (
            id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '',
            provider_type TEXT NOT NULL DEFAULT 'openai',
            api_base TEXT NOT NULL DEFAULT '', api_key TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL DEFAULT '',
            extra_headers TEXT NOT NULL DEFAULT '{}'
        )
    """)
    conn.execute(
        "INSERT INTO providers (id, name, api_base, extra_headers) VALUES (?, ?, ?, ?)",
        ("deepseek1", "DS", "http://192.168.1.20:8000", json.dumps({"thinking": "enabled", "X-Custom": "1"})),
    )
    conn.commit()
    conn.close()

    config_path = str(tmp_path / "config.json")
    config = load_config(config_path, force_reload=True)
    config.config = {"database": db_path, "logging": {"enabled": False}}
    config.save()

    init_db(db_path)
    provider = get_provider("deepseek1")
    assert provider["provider_options"] == {"thinking": "enabled"}
    assert provider["upstream_headers"] == {"X-Custom": "1"}

    # 管理员有意清空 options（例如关闭 thinking）
    update_provider("deepseek1", {"provider_options": {}, "upstream_headers": {}})
    assert get_provider("deepseek1")["provider_options"] == {}

    # 重启（再次 init）绝不能把清空的配置恢复回来
    init_db(db_path)
    provider = get_provider("deepseek1")
    assert provider["provider_options"] == {}, "迁移非幂等：清空的配置被重新恢复"
    assert provider["upstream_headers"] == {}


def test_provider_options_migration_idempotent_when_drop_column_fails(tmp_path):
    """DROP COLUMN 失败（列被索引引用）时迁移仍必须幂等。

    旧实现以“extra_headers 列是否已删除”作为迁移完成判据，DROP 失败就会
    每次启动重跑，把管理员清空的 provider_options/upstream_headers 复活。
    """
    db_path = str(tmp_path / "legacy_idx.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE providers (
            id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '',
            provider_type TEXT NOT NULL DEFAULT 'openai',
            api_base TEXT NOT NULL DEFAULT '', api_key TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL DEFAULT '',
            extra_headers TEXT NOT NULL DEFAULT '{}'
        )
    """)
    # 索引引用 extra_headers，使 ALTER TABLE ... DROP COLUMN 报 OperationalError
    conn.execute("CREATE INDEX idx_legacy_extra_headers ON providers(extra_headers)")
    conn.execute(
        "INSERT INTO providers (id, name, api_base, extra_headers) VALUES (?, ?, ?, ?)",
        ("deepseek1", "DS", "http://192.168.1.20:8000", json.dumps({"X-Custom": "1"})),
    )
    conn.commit()
    conn.close()

    config_path = str(tmp_path / "config.json")
    config = load_config(config_path, force_reload=True)
    config.config = {"database": db_path, "logging": {"enabled": False}}
    config.save()

    init_db(db_path)
    assert get_provider("deepseek1")["upstream_headers"] == {"X-Custom": "1"}

    update_provider("deepseek1", {"provider_options": {}, "upstream_headers": {}})
    assert get_provider("deepseek1")["upstream_headers"] == {}

    # 旧列仍在，但版本号已推进，迁移不得重跑
    init_db(db_path)
    provider = get_provider("deepseek1")
    assert provider["upstream_headers"] == {}, "DROP COLUMN 失败时迁移非幂等：清空的配置被复活"
    assert provider["provider_options"] == {}

    with get_db() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] >= 1


# ---------------------------------------------------------------------------
# #4 update_admin_password 返回值语义
# ---------------------------------------------------------------------------

def test_update_admin_password_rowcount(temp_db):
    assert update_admin_password("admin", hash_password("newpass")) is True
    assert update_admin_password("ghost", hash_password("x")) is False


# ---------------------------------------------------------------------------
# #5 日志保留：非空过期目录必须被删除
# ---------------------------------------------------------------------------

def test_cleanup_old_logs_removes_nonempty_expired_dirs(tmp_path):
    from app.services.logger import LogManager

    log_root = tmp_path / "logs"
    old_day = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    old_dir = log_root / old_day
    old_dir.mkdir(parents=True)
    (old_dir / "app.log").write_text("line\n", encoding="utf-8")
    fresh_day = datetime.now().strftime("%Y-%m-%d")
    fresh_dir = log_root / fresh_day
    fresh_dir.mkdir(parents=True)
    (fresh_dir / "app.log").write_text("line\n", encoding="utf-8")

    mgr = LogManager()
    mgr._log_dir = str(log_root)
    mgr._retention_days = 30
    mgr._cleanup_old_logs()

    assert not old_dir.exists(), "过期且非空的日志目录未被删除"
    assert fresh_dir.exists()


# ---------------------------------------------------------------------------
# #6/#7 admin 输入校验：400 而不是 500
# ---------------------------------------------------------------------------

def test_update_routing_rule_invalid_scope_returns_400(temp_db):
    r = client.post("/admin/routing-rules", headers=temp_db["headers"], json={
        "id": "r1", "name": "R", "match_model": "*", "target_model": "x",
    })
    assert r.status_code == 200
    r = client.put("/admin/routing-rules/r1", headers=temp_db["headers"], json={"match_scope": "bogus"})
    assert r.status_code == 400


def test_stats_history_invalid_ts_returns_400(temp_db):
    r = client.get("/admin/stats/history", params={"from_ts": "abc"}, headers=temp_db["headers"])
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# #8 配置导入健壮性
# ---------------------------------------------------------------------------

def test_import_provider_without_name_does_not_500(temp_db):
    r = client.post("/admin/config/import", headers=temp_db["headers"], json={
        "mode": "skip",
        "providers": [{
            "id": "imp1", "provider_type": "openai",
            "api_base": "http://192.168.1.30:8000", "api_key": "k",
        }],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["providers"].get("created") == 1
    assert get_provider("imp1")["name"] == "imp1"


def test_import_bad_entry_reported_not_500(temp_db):
    r = client.post("/admin/config/import", headers=temp_db["headers"], json={
        "mode": "skip",
        "routing_rules": [{"id": "bad1", "match_scope": "bogus", "match_model": "*", "target_model": "x"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["routing_rules"].get("failed") == 1
    assert body["errors"]


# ---------------------------------------------------------------------------
# #9 认证端点非字符串输入：400 而不是 500
# ---------------------------------------------------------------------------

def test_login_non_string_fields_return_400(temp_db):
    r = client.post("/auth/login", json={"username": None, "password": "x"})
    assert r.status_code == 400
    r = client.post("/auth/login", json={"username": 123, "password": "x"})
    assert r.status_code == 400


def test_create_user_non_string_username_returns_400(temp_db):
    r = client.post("/admin/users", headers=temp_db["headers"], json={"username": 123})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# #10 allowed_models 归一化
# ---------------------------------------------------------------------------

def test_allowed_models_string_wrapped_and_invalid_rejected(temp_db):
    add_user({"username": "alice"})
    r = client.post("/admin/users/alice/api-keys", headers=temp_db["headers"], json={"allowed_models": "gpt-4"})
    assert r.status_code == 200
    assert r.json()["allowed_models"] == ["gpt-4"]

    r = client.post("/admin/users/alice/api-keys", headers=temp_db["headers"], json={"allowed_models": 42})
    assert r.status_code == 400

    # 空字符串/空列表不得默认为 ["*"]（全模型放行），必须明确拒绝。
    r = client.post("/admin/users/alice/api-keys", headers=temp_db["headers"], json={"allowed_models": "   "})
    assert r.status_code == 400
    r = client.post("/admin/users/alice/api-keys", headers=temp_db["headers"], json={"allowed_models": []})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# #11 请求日志可按 images_generations 过滤
# ---------------------------------------------------------------------------

def test_request_logs_filter_images_generations(temp_db):
    from app.database import add_request_log
    add_request_log(
        timestamp="2026-06-06 12:00:00", endpoint="images_generations",
        username="alice", api_key="sk-aio-***", requested_model="img", model="img",
        provider="img", status="ok", stream=False, tokens=0,
        request_body={}, response_body={}, details={},
    )
    r = client.get("/admin/request-logs", params={"endpoint": "images_generations"}, headers=temp_db["headers"])
    assert r.status_code == 200
    assert r.json()["total"] == 1


# ---------------------------------------------------------------------------
# #16 请求体限制可显式禁用
# ---------------------------------------------------------------------------

def test_body_limit_zero_disables(monkeypatch):
    from app.core import body_limit
    monkeypatch.setattr(body_limit, "get_default", lambda key, fallback: 0)
    assert body_limit.max_request_body_bytes() == 0
    monkeypatch.setattr(body_limit, "get_default", lambda key, fallback: -5)
    assert body_limit.max_request_body_bytes() == 0


# ---------------------------------------------------------------------------
# #14 模型刷新保留有管理员配置的过期模型
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refresh_keeps_configured_stale_models(temp_db, monkeypatch):
    from app.services import discovery

    add_provider({
        "id": "p2", "name": "P2", "provider_type": "openai",
        "api_base": "http://192.168.1.40:8000", "api_key": "k",
        "models": [
            {"id": "keep-me", "name": "keep-me", "enabled": True},
            {"id": "drop-me", "name": "drop-me", "enabled": True},
        ],
    })
    _set_model_preprocessor("p2", "keep-me", "1")

    async def fake_discover(provider_id):
        return [{"id": "new-model", "name": "new-model"}]

    monkeypatch.setattr(discovery, "discover_models", fake_discover)
    result = await discovery.refresh_provider_models("p2")

    provider = get_provider("p2")
    ids = {m["id"] for m in provider["models"]}
    assert "keep-me" in ids, "有 preprocessor 配置的过期模型被误删"
    assert "drop-me" not in ids
    assert "new-model" in ids
    assert result["removed"] == 1
