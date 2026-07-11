"""PR2: rejected (401/403) and cancelled outcomes surface in logs/stats."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.config import load_config
from app.database import get_global_stats, init_db, add_user, add_user_api_key
from app.router.proxy import (
    ensure_model_allowed,
    get_request_log,
    verify_api_key,
)
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    import app.database as db_mod
    from app.router import proxy as proxy_mod

    db_path = str(tmp_path / "test.db")
    config_path = str(tmp_path / "config.json")
    config = load_config(config_path, force_reload=True)
    config.config = {
        "host": "0.0.0.0",
        "port": 8000,
        "database": db_path,
        "logging": {
            "enabled": False,
            "level": "INFO",
            "log_dir": "logs",
            "retention_days": 30,
            "console": False,
        },
    }
    config.save()
    db_mod._initialized = False
    init_db(db_path)
    proxy_mod.clear_request_log()
    add_user({"username": "bob", "display_name": "Bob", "enabled": True})
    add_user_api_key("bob", "default", ["gpt-4"])
    yield
    proxy_mod.clear_request_log()
    db_mod._initialized = False


def test_invalid_api_key_is_logged_as_rejected():
    with pytest.raises(HTTPException) as exc:
        verify_api_key("Bearer sk-bad-key", endpoint="chat_completions")
    assert exc.value.status_code == 401

    log = get_request_log()
    assert log
    assert log[0]["status"] == "rejected"
    assert log[0]["endpoint"] == "chat_completions"
    assert log[0]["success"] is False

    stats = get_global_stats()
    assert int(stats.get("rejected_calls", 0) or 0) >= 1
    assert int(stats.get("failed_calls", 0) or 0) >= 1


def test_model_allow_deny_is_logged_as_rejected():
    with pytest.raises(HTTPException) as exc:
        ensure_model_allowed(
            {"username": "bob"},
            {"key": "k", "allowed_models": ["gpt-4"]},
            "claude-3",
            endpoint="chat_completions",
        )
    assert exc.value.status_code == 403

    log = get_request_log()
    assert log
    assert log[0]["status"] == "rejected"
    assert log[0]["requested_model"] == "claude-3"
    assert "not allowed" in (log[0].get("error_message") or log[0]["details"].get("error_message", ""))

    stats = get_global_stats()
    assert int(stats.get("rejected_calls", 0) or 0) >= 1


def test_chat_completions_401_via_http():
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer totally-invalid"},
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401
    log = get_request_log()
    assert any(item.get("status") == "rejected" for item in log)
