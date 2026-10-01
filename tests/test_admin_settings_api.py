"""设置 API 的行为测试：写入只动指定键、校验在服务端强制、写完立即生效。

这些用例同时是"文件方案四个坑"的回归锁：
1. 禁止整表回写（否则内置默认被固化，升级后新默认值跟不上）；
2. 重置 = 删键；
3. 写盘前重读磁盘，不覆盖管理员的 SSH 手改；
4. 写不进去要响亮失败，不能退成"内存生效、重启丢失"。
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_default, load_config
from app.core.state import reasoning_cache
from app.database import add_admin, init_db
from app.security import create_session, hash_password
from main import app

client = TestClient(app)


@pytest.fixture
def admin_api(tmp_path):
    db_path = str(tmp_path / "test.db")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "host": "127.0.0.1",
        "port": 8000,
        "database": db_path,
        "logging": {"enabled": False, "level": "INFO", "log_dir": str(tmp_path / "logs"), "retention_days": 30, "console": False},
    }), encoding="utf-8")
    cfg = load_config(str(config_path), force_reload=True)
    init_db(db_path)
    add_admin("admin", hash_password("secret"), "Admin")
    token = create_session("admin")
    yield {
        "headers": {"Authorization": f"Bearer {token}"},
        "config": cfg,
        "path": config_path,
    }
    load_config(str(tmp_path / "teardown.json"), force_reload=True)


def _disk_defaults(path):
    return json.loads(path.read_text(encoding="utf-8")).get("defaults", {})


# -- 读取 --

def test_get_settings_requires_admin_session():
    response = client.get("/admin/settings")
    assert response.status_code == 401


def test_get_settings_returns_full_schema(admin_api):
    body = client.get("/admin/settings", headers=admin_api["headers"]).json()
    keys = {item["key"] for item in body["items"]}
    from app.config import default_config
    assert keys == set(default_config()["defaults"])
    entry = next(item for item in body["items"] if item["key"] == "max_request_body_bytes")
    assert entry["unit"] == "bytes"
    assert entry["max"] == 512 * 1024 * 1024
    assert entry["source"] == "builtin"
    assert entry["written"] is False
    assert entry["danger"] is None
    assert body["file"]["path"] == str(admin_api["path"])
    assert body["file"]["writable"] is True
    assert {"host", "port", "database", "logging"} <= set(body["readOnly"])
    # 三个运行时钩子必须都已注册，否则固化值改了不生效
    assert set(body["runtimeHooks"]) >= {"state_caches", "request_log", "litellm_timeout"}


# -- 写入 --

def test_put_writes_only_the_touched_key(admin_api):
    response = client.put("/admin/settings", headers=admin_api["headers"],
                          json={"values": {"max_tokens": 2048}})
    assert response.status_code == 200
    assert _disk_defaults(admin_api["path"]) == {"max_tokens": 2048}
    assert get_default("max_tokens", 16384) == 2048
    item = next(i for i in response.json()["items"] if i["key"] == "max_tokens")
    assert item["written"] is True and item["source"] == "file"
    assert set(response.json()["applied"].values()) == {"applied"}


def test_put_takes_effect_without_restart(admin_api):
    client.put("/admin/settings", headers=admin_api["headers"],
               json={"values": {"reasoning_cache_ttl": 77}})
    # reasoning_cache 的 ttl 是构造时定死的，只有 runtime hook 推送后才会变
    assert reasoning_cache.ttl == 77
    reasoning_cache.configure(1800, reasoning_cache.max_size)


def test_put_rejects_invalid_values(admin_api):
    response = client.put("/admin/settings", headers=admin_api["headers"],
                          json={"values": {"temperature": 2.5}})
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)
    assert "temperature" in response.json()["detail"]
    assert _disk_defaults(admin_api["path"]) == {}


def test_put_rejects_unknown_keys(admin_api):
    response = client.put("/admin/settings", headers=admin_api["headers"],
                          json={"values": {"injected_key": 1}})
    assert response.status_code == 400
    assert _disk_defaults(admin_api["path"]) == {}


def test_put_requires_empty_values(admin_api):
    assert client.put("/admin/settings", headers=admin_api["headers"], json={"values": {}}).status_code == 400
    assert client.put("/admin/settings", headers=admin_api["headers"], json={}).status_code == 400


def test_put_dangerous_value_requires_confirmation(admin_api):
    response = client.put("/admin/settings", headers=admin_api["headers"],
                          json={"values": {"max_request_body_bytes": 0}})
    assert response.status_code == 409
    assert "max_request_body_bytes" in response.json()["detail"]
    assert _disk_defaults(admin_api["path"]) == {}

    confirmed = client.put("/admin/settings", headers=admin_api["headers"],
                           json={"values": {"max_request_body_bytes": 0}, "confirmDanger": True})
    assert confirmed.status_code == 200
    assert _disk_defaults(admin_api["path"]) == {"max_request_body_bytes": 0}
    item = next(i for i in confirmed.json()["items"] if i["key"] == "max_request_body_bytes")
    assert item["danger"] == "settings.danger.bodyLimitOff"


def test_put_fails_loudly_when_file_not_writable(admin_api, monkeypatch):
    from app.config import ConfigManager

    def refuse(self, section, values=None, remove=()):
        raise OSError("Read-only file system")

    monkeypatch.setattr(ConfigManager, "patch", refuse)
    response = client.put("/admin/settings", headers=admin_api["headers"],
                          json={"values": {"max_tokens": 4096}})
    assert response.status_code == 500
    assert "配置文件写入失败" in response.json()["detail"]
    # 内存也绝不能被"半提交"：写盘失败就不改内存
    assert get_default("max_tokens", 16384) == 16384


def test_put_preserves_external_hand_edits(admin_api):
    """管理员在 SSH 上改过的键，界面写另一个键时不得被覆盖。"""
    on_disk = json.loads(admin_api["path"].read_text(encoding="utf-8"))
    on_disk["defaults"] = {"tool_only_limit": 7}
    admin_api["path"].write_text(json.dumps(on_disk), encoding="utf-8")

    client.put("/admin/settings", headers=admin_api["headers"], json={"values": {"temperature": 0.1}})
    saved = _disk_defaults(admin_api["path"])
    assert saved == {"tool_only_limit": 7, "temperature": 0.1}


# -- 重置为默认 --

def test_reset_deletes_key_from_file(admin_api):
    client.put("/admin/settings", headers=admin_api["headers"], json={"values": {"max_tokens": 512}})
    response = client.post("/admin/settings/reset", headers=admin_api["headers"], json={"keys": ["max_tokens"]})
    assert response.status_code == 200
    assert "max_tokens" not in _disk_defaults(admin_api["path"])
    assert get_default("max_tokens", 16384) == 16384
    item = next(i for i in response.json()["items"] if i["key"] == "max_tokens")
    assert item["written"] is False and item["source"] == "builtin"


def test_reset_rejects_unknown_and_empty(admin_api):
    assert client.post("/admin/settings/reset", headers=admin_api["headers"],
                       json={"keys": ["nope"]}).status_code == 400
    assert client.post("/admin/settings/reset", headers=admin_api["headers"],
                       json={"keys": []}).status_code == 400


# -- 从磁盘重载 --

def test_reload_picks_up_external_edits(admin_api):
    on_disk = json.loads(admin_api["path"].read_text(encoding="utf-8"))
    on_disk["defaults"] = {"max_tokens": 3333}
    admin_api["path"].write_text(json.dumps(on_disk), encoding="utf-8")

    body = client.post("/admin/config/reload", headers=admin_api["headers"]).json()
    assert body["changedKeys"] == ["max_tokens"]
    assert body["restartRequiredChangedKeys"] == []
    assert get_default("max_tokens", 16384) == 3333


def test_reload_reports_restart_required_keys_separately(admin_api):
    on_disk = json.loads(admin_api["path"].read_text(encoding="utf-8"))
    on_disk["port"] = 9100
    admin_api["path"].write_text(json.dumps(on_disk), encoding="utf-8")

    body = client.post("/admin/config/reload", headers=admin_api["headers"]).json()
    assert "port" in body["restartRequiredChangedKeys"]


# -- 审计 --

def test_settings_write_is_audited(admin_api, monkeypatch):
    """config.json 不入库、生产在 volume 里，审计日志是唯一的历史。"""
    from app.router import admin

    recorded = []
    monkeypatch.setattr(admin._settings_log, "info", lambda *args, **kwargs: recorded.append(args))
    client.put("/admin/settings", headers=admin_api["headers"], json={"values": {"max_tokens": 123}})
    assert recorded
    fmt, *params = recorded[0]
    line = fmt % tuple(params)
    assert "admin=admin" in line and "key=max_tokens" in line and "new=123" in line
    assert "source=ui" in line


def test_settings_log_channel_exists():
    from app.services.logger import available_log_channels
    assert "settings" in available_log_channels()


# -- 与导入导出联动 --

def test_export_includes_only_written_settings(admin_api):
    client.put("/admin/settings", headers=admin_api["headers"], json={"values": {"max_tokens": 64}})
    body = client.get("/admin/config/export", headers=admin_api["headers"]).json()
    assert body["version"] == 2
    assert body["settings"] == {"max_tokens": 64}


def test_import_applies_settings_through_same_validation(admin_api):
    response = client.post("/admin/config/import", headers=admin_api["headers"], json={
        "mode": "replace",
        "providers": [],
        "settings": {"max_tokens": 4096, "temperature": 9, "bogus": 1},
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "partial"
    assert len(payload["errors"]) == 2
    assert _disk_defaults(admin_api["path"]) == {"max_tokens": 4096}
    assert get_default("max_tokens", 16384) == 4096


def test_import_skip_mode_leaves_settings_untouched(admin_api):
    client.put("/admin/settings", headers=admin_api["headers"], json={"values": {"max_tokens": 21}})
    response = client.post("/admin/config/import", headers=admin_api["headers"], json={
        "mode": "skip",
        "providers": [],
        "settings": {"max_tokens": 999999},
    })
    assert response.status_code == 200
    assert _disk_defaults(admin_api["path"]) == {"max_tokens": 21}


def test_import_accepts_version_one_payload_without_settings(admin_api):
    response = client.post("/admin/config/import", headers=admin_api["headers"], json={
        "mode": "replace",
        "providers": [],
    })
    assert response.status_code == 200
    assert _disk_defaults(admin_api["path"]) == {}


# -- 前端装配 --

def test_static_page_wires_settings_bundle():
    """设置页的 DOM/脚本装配：漏一个 script 标签或 section id 就是"页面看起来没变化"。"""
    from pathlib import Path

    import app as app_pkg

    static = Path(app_pkg.__file__).parent / "web" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    assert 'id="settings-section"' in html
    assert 'data-section="settings"' in html
    assert "js/settings.js" in html
    assert 'id="config-section"' not in html

    js = (static / "js" / "settings.js").read_text(encoding="utf-8")
    for name in ("loadSettings", "saveSettings", "resetSetting", "reloadConfigFile", "renderDiagnostics"):
        assert f"function {name}" in js, name
    # 项目规则：JS 源码不得出现字面 U+2028/U+2029 行分隔符
    assert "\u2028" not in js and "\u2029" not in js


# -- 风险确认不得有第二条绕过路径 --

def test_import_refuses_dangerous_settings_without_confirmation(admin_api):
    """导入路径必须和 PUT 一样挡下风险值，否则一份导出文件就能静默禁用保护。"""
    response = client.post("/admin/config/import", headers=admin_api["headers"], json={
        "mode": "replace",
        "providers": [],
        "settings": {"max_request_body_bytes": 0, "max_tokens": 512},
    })
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "partial"
    assert any("max_request_body_bytes" in error for error in payload["errors"])
    # 非风险键照常导入，风险键被挡在门外
    assert _disk_defaults(admin_api["path"]) == {"max_tokens": 512}


def test_import_accepts_dangerous_settings_with_confirmation(admin_api):
    response = client.post("/admin/config/import", headers=admin_api["headers"], json={
        "mode": "replace",
        "providers": [],
        "confirmDanger": True,
        "settings": {"max_request_body_bytes": 0},
    })
    assert response.json()["status"] == "ok"
    assert _disk_defaults(admin_api["path"]) == {"max_request_body_bytes": 0}


def test_put_empty_redaction_list_requires_confirmation(admin_api):
    """清空脱敏字段 = 凭据明文进请求日志，属于需要确认的取值。"""
    response = client.put("/admin/settings", headers=admin_api["headers"],
                          json={"values": {"request_log_redact_fields": []}})
    assert response.status_code == 409
    assert "request_log_redact_fields" in response.json()["detail"]
    assert _disk_defaults(admin_api["path"]) == {}

    confirmed = client.put("/admin/settings", headers=admin_api["headers"],
                           json={"values": {"request_log_redact_fields": []}, "confirmDanger": True})
    assert confirmed.status_code == 200
    assert _disk_defaults(admin_api["path"]) == {"request_log_redact_fields": []}
