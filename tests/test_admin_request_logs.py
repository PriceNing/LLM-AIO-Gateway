"""Tests for request log and config export/import admin endpoints."""
import json
import pytest
from fastapi.testclient import TestClient
from main import app
from app.config import load_config
from app.database import (
    init_db, add_admin, add_provider, add_routing_rule, add_fallback_policy,
    add_request_log, list_request_logs, clear_request_logs, add_user,
    add_user_api_key, get_user,
)
from app.security import create_session, hash_password

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
    yield {"headers": {"Authorization": f"Bearer {token}"}}


def _add_log(**over):
    base = dict(
        timestamp="2026-06-06 12:00:00",
        endpoint="chat_completions",
        username="alice",
        api_key="sk-aio-***",
        requested_model="gpt-4o",
        model="gpt-4o",
        provider="openai",
        status="ok",
        stream=False,
        tokens=12,
        request_body={"messages": [{"role": "user", "content": "hi"}]},
        response_body={"id": "r1", "choices": [{"message": {"role": "assistant", "content": "hello"}}]},
        details={"fallback_status": "unused"},
    )
    base.update(over)
    return add_request_log(**base)


# -- Request log endpoints --

def test_list_request_logs_empty(temp_db):
    r = client.get("/admin/request-logs", headers=temp_db["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert body["total"] == 0


def test_list_request_logs_returns_items(temp_db):
    _add_log()
    _add_log(endpoint="messages", status="partial")
    r = client.get("/admin/request-logs", headers=temp_db["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2
    # latest first
    assert body["items"][0]["endpoint"] == "messages"
    assert body["items"][1]["endpoint"] == "chat_completions"


def test_list_request_logs_filters(temp_db):
    _add_log(endpoint="chat_completions", status="ok")
    _add_log(endpoint="messages", status="partial")
    r = client.get("/admin/request-logs", params={"endpoint": "messages"}, headers=temp_db["headers"])
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["endpoint"] == "messages"
    assert body["items"][0]["status"] == "partial"

    r = client.get("/admin/request-logs", params={"status": "ok"}, headers=temp_db["headers"])
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["status"] == "ok"

    r = client.get("/admin/request-logs", params={"endpoint": "bogus"}, headers=temp_db["headers"])
    assert r.status_code == 400


def test_request_log_detail(temp_db):
    lid = _add_log()
    r = client.get(f"/admin/request-logs/{lid}", headers=temp_db["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == lid
    assert body["request_body"] == {"messages": [{"role": "user", "content": "hi"}]}
    assert body["details"] == {"fallback_status": "unused"}


def test_request_log_detail_404(temp_db):
    r = client.get("/admin/request-logs/9999", headers=temp_db["headers"])
    assert r.status_code == 404


def test_delete_request_log(temp_db):
    lid = _add_log()
    r = client.delete(f"/admin/request-logs/{lid}", headers=temp_db["headers"])
    assert r.status_code == 200
    assert r.json() == {"status": "deleted", "log_id": lid}
    assert all(r["id"] != lid for r in list_request_logs(limit=10))


def test_clear_request_logs(temp_db):
    _add_log()
    _add_log(endpoint="messages")
    r = client.post("/admin/request-logs/clear", headers=temp_db["headers"])
    assert r.status_code == 200
    assert r.json()["removed"] == 2
    r = client.get("/admin/request-logs", headers=temp_db["headers"])
    assert r.json()["total"] == 0


def test_request_logs_require_auth():
    r = client.get("/admin/request-logs")
    assert r.status_code == 401
    r = client.post("/admin/request-logs/clear")
    assert r.status_code == 401


# -- Config export --

def test_export_config_redacts_secrets(temp_db):
    add_provider({
        "id": "p1", "name": "P1", "provider_type": "openai",
        "api_base": "https://api.example.com/v1", "api_key": "secret-key",
        "enabled": True, "extra_headers": {"X-Foo": "bar"},
        "models": [{"id": "m1", "name": "M1", "enabled": True}],
    })
    r = client.get("/admin/config/export", headers=temp_db["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1
    assert body["include_secrets"] is False
    assert body["providers"][0]["api_key"] == ""
    assert body["providers"][0].get("extra_headers") in (None, {})


def test_export_config_includes_secrets(temp_db):
    add_provider({
        "id": "p1", "name": "P1", "provider_type": "openai",
        "api_base": "https://api.example.com/v1", "api_key": "secret-key",
        "enabled": True, "models": [],
    })
    r = client.get("/admin/config/export", params={"include_secrets": "true"}, headers=temp_db["headers"])
    body = r.json()
    assert body["include_secrets"] is True
    assert body["providers"][0]["api_key"] == "secret-key"


# -- Config import --

def test_import_config_skip_mode(temp_db):
    add_provider({
        "id": "p1", "name": "Original", "provider_type": "openai",
        "api_base": "https://a.example.com/v1", "api_key": "key-a",
        "enabled": True, "models": [],
    })
    payload = {
        "mode": "skip",
        "providers": [
            {"id": "p1", "name": "Updated", "provider_type": "openai",
             "api_base": "https://b.example.com/v1", "api_key": "key-b", "enabled": True, "models": []},
            {"id": "p2", "name": "New", "provider_type": "openai",
             "api_base": "https://c.example.com/v1", "api_key": "key-c", "enabled": True, "models": []},
        ],
    }
    r = client.post("/admin/config/import", json=payload, headers=temp_db["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "skip"
    assert body["summary"]["providers"] == {"skipped": 1, "created": 1}
    # p1 unchanged
    r = client.get("/admin/config/export", params={"include_secrets": "true"}, headers=temp_db["headers"])
    p1 = next(p for p in r.json()["providers"] if p["id"] == "p1")
    assert p1["name"] == "Original"
    assert p1["api_base"] == "https://a.example.com/v1"


def test_import_config_replace_mode(temp_db):
    add_provider({
        "id": "p1", "name": "Original", "provider_type": "openai",
        "api_base": "https://a.example.com/v1", "api_key": "key-a",
        "enabled": True, "models": [],
    })
    payload = {
        "mode": "replace",
        "providers": [
            {"id": "p1", "name": "Updated", "provider_type": "openai",
             "api_base": "https://b.example.com/v1", "api_key": "", "enabled": True, "models": []},
        ],
    }
    r = client.post("/admin/config/import", json=payload, headers=temp_db["headers"])
    body = r.json()
    assert body["summary"]["providers"] == {"updated": 1}
    r = client.get("/admin/config/export", params={"include_secrets": "true"}, headers=temp_db["headers"])
    p1 = next(p for p in r.json()["providers"] if p["id"] == "p1")
    assert p1["name"] == "Updated"
    # empty api_key was filtered out, so original key preserved
    assert p1["api_key"] == "key-a"


def test_import_config_merge_mode(temp_db):
    add_routing_rule({
        "id": "r1", "name": "old", "enabled": True, "username": "",
        "api_key_pattern": "", "match_model": "gpt-4o", "target_model": "gpt-4o-mini",
        "target_provider": "",
    })
    payload = {
        "mode": "merge",
        "routing_rules": [
            {"id": "r1", "name": "renamed"},
        ],
    }
    r = client.post("/admin/config/import", json=payload, headers=temp_db["headers"])
    body = r.json()
    assert body["summary"]["routing_rules"] == {"updated": 1}
    from app.database import get_routing_rule
    rule = get_routing_rule("r1")
    assert rule["name"] == "renamed"
    # merge should not clear fields omitted in payload
    assert rule["match_model"] == "gpt-4o"


def test_import_config_fallback_policy(temp_db):
    add_fallback_policy({
        "id": "f1", "name": "F1", "enabled": True,
        "match_provider": "openai", "match_model": "gpt-4o",
        "triggers": {"timeout": True}, "chain": [{"provider": "anthropic", "model": "haiku"}],
    })
    payload = {
        "mode": "replace",
        "fallback_policies": [
            {"id": "f1", "name": "Renamed", "enabled": True,
             "match_provider": "anthropic", "match_model": "*",
             "triggers": {"http_429": True}, "chain": []},
        ],
    }
    r = client.post("/admin/config/import", json=payload, headers=temp_db["headers"])
    body = r.json()
    assert body["summary"]["fallback_policies"] == {"updated": 1}
    from app.database import get_fallback_policy
    pol = get_fallback_policy("f1")
    assert pol["name"] == "Renamed"
    assert pol["match_provider"] == "anthropic"


def test_import_config_invalid_mode(temp_db):
    r = client.post("/admin/config/import", json={"mode": "destroy", "providers": []}, headers=temp_db["headers"])
    assert r.status_code == 400


def test_import_config_invalid_shape(temp_db):
    r = client.post("/admin/config/import", json={"providers": "nope"}, headers=temp_db["headers"])
    assert r.status_code == 400


def test_export_import_roundtrip(temp_db):
    add_provider({
        "id": "p1", "name": "P1", "provider_type": "openai",
        "api_base": "https://api.example.com/v1", "api_key": "key-a",
        "enabled": True, "models": [],
    })
    add_routing_rule({
        "id": "r1", "name": "rule", "enabled": True, "username": "",
        "api_key_pattern": "", "match_model": "*", "target_model": "",
        "target_provider": "",
    })
    r = client.get("/admin/config/export", params={"include_secrets": "true"}, headers=temp_db["headers"])
    body = r.json()
    r = client.post("/admin/config/import", json={**body, "mode": "skip"}, headers=temp_db["headers"])
    summary = r.json()["summary"]
    assert summary["providers"]["skipped"] == 1
    assert summary["routing_rules"]["skipped"] == 1


def test_export_users_includes_api_keys(temp_db):
    add_user({"username": "bob", "display_name": "Bob", "enabled": True})
    key = add_user_api_key("bob", "work", ["p1/m1"])
    r = client.get("/admin/users/export", headers=temp_db["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1
    user = body["users"][0]
    assert user["username"] == "bob"
    assert user["api_keys"][0]["key"] == key["key"]
    assert user["api_keys"][0]["allowed_models"] == ["p1/m1"]


def test_import_users_creates_user_and_preserves_api_key(temp_db):
    payload = {
        "mode": "replace",
        "users": [{
            "username": "alice",
            "display_name": "Alice",
            "enabled": True,
            "api_keys": [{
                "key": "sk-aio-imported",
                "name": "imported",
                "allowed_models": ["*"],
                "enabled": True,
                "stats": {"total_calls": 2, "failed_calls": 1, "total_tokens": 30},
                "created_at": "2026-06-01",
            }],
        }],
    }
    r = client.post("/admin/users/import", json=payload, headers=temp_db["headers"])
    assert r.status_code == 200
    assert r.json()["summary"] == {"users": {"created": 1}, "api_keys": {"created": 1}}
    user = get_user("alice")
    assert user["display_name"] == "Alice"
    assert user["api_keys"][0]["key"] == "sk-aio-imported"
    assert user["api_keys"][0]["stats"]["total_tokens"] == 30


def test_config_endpoints_require_auth():
    r = client.get("/admin/config/export")
    assert r.status_code == 401
    r = client.post("/admin/config/import", json={"providers": []})
    assert r.status_code == 401
    r = client.get("/admin/users/export")
    assert r.status_code == 401
    r = client.post("/admin/users/import", json={"users": []})
    assert r.status_code == 401
