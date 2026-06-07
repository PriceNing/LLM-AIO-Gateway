"""Integration test: proxy endpoints write to request_logs and admin can fetch them."""
import json
import pytest
from fastapi.testclient import TestClient
from main import app
from app.config import load_config
from app.database import (
    init_db, add_admin, add_provider, add_request_log, list_request_logs,
    count_request_logs, clear_request_logs,
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


def test_chat_completions_writes_request_log(temp_db):
    add_provider({
        "id": "mock-openai", "name": "Mock", "provider_type": "openai",
        "api_base": "http://127.0.0.1:1/v1", "api_key": "",
        "enabled": True, "models": [{"id": "m1", "name": "M1", "enabled": True}],
    })
    from app.database import add_user, add_user_api_key as add_key
    add_user({"username": "alice", "display_name": "Alice", "enabled": True})
    api_key_row = add_key("alice", "sk-aio-test")

    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key_row['key']}"},
        json={"model": "mock-openai/m1", "messages": [{"role": "user", "content": "hi"}]},
    )
    # Upstream is unreachable; we expect 500 but request log still written
    assert r.status_code in (200, 500)

    rows = list_request_logs(limit=10)
    assert len(rows) == 1
    entry = rows[0]
    assert entry["endpoint"] == "chat_completions"
    assert entry["username"] == "alice"
    assert entry["requested_model"] == "mock-openai/m1"
    assert entry["request_body"]["messages"][0]["content"] == "hi"
    # status should be 'fail' since upstream unreachable
    assert entry["status"] in ("fail", "partial")
    assert entry["response_body"]["error"]["message"]
    assert entry["response_body"]["status"] in ("fail", "partial")
    # details has error message
    assert entry["error"]


def test_trim_request_logs_keeps_recent(temp_db):
    from app.config import get_default
    from app.database import trim_request_logs
    for i in range(10):
        add_request_log(
            timestamp=f"2026-06-06 12:00:{i:02d}",
            endpoint="chat_completions", username="alice", api_key="sk",
            requested_model="m", model="m", provider="p",
            status="ok", stream=False, tokens=0,
        )
    # Default request_log_max is 200 — all 10 should remain
    trim_request_logs(get_default("request_log_max", 500))
    assert count_request_logs() == 10
    # Keep only 3 — oldest 7 dropped
    trim_request_logs(3)
    assert count_request_logs() == 3


def test_purge_old_request_logs_on_overflow(temp_db):
    """Insert >500 logs and confirm we never exceed request_log_max."""
    from app.config import get_default
    for i in range(550):
        add_request_log(
            timestamp=f"2026-06-06 12:00:00",  # same timestamp
            endpoint="chat_completions", username="alice", api_key="sk",
            requested_model="m", model="m", provider="p",
            status="ok", stream=False, tokens=0,
        )
    # trigger trim manually
    from app.database import trim_request_logs
    trim_request_logs(get_default("request_log_max", 500))
    # Should be capped at the default
    total = count_request_logs()
    assert total <= get_default("request_log_max", 500)


def test_record_request_log_stream_summary_response_body(temp_db):
    from app.router.proxy import _record_request_log

    _record_request_log(
        endpoint="chat_completions",
        username="alice",
        api_key_value="sk-test",
        requested_model="m",
        final_model="m",
        final_provider="p",
        request_body={"stream": True},
        response_body=None,
        streamed_text="hello",
        streamed_reasoning="thinking",
        streamed_tool_calls=[{"id": "call_1", "name": "tool", "arguments": "{}"}],
        usage={"total_tokens": 3},
        success=True,
        status="ok",
        tokens=3,
    )

    entry = list_request_logs(limit=1)[0]
    assert entry["stream"] is True
    assert entry["response_body"] == {
        "type": "stream_summary",
        "endpoint": "chat_completions",
        "status": "ok",
        "model": "m",
        "text": "hello",
        "reasoning": "thinking",
        "tool_calls": [{"id": "call_1", "name": "tool", "arguments": "{}"}],
        "usage": {"total_tokens": 3},
    }
