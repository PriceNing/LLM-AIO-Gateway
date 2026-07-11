"""Proactive fallback attempt_timeout should cut hung primaries short."""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app.config import load_config
from app.core.output import InternalOutputEvent
from app.core.policy import apply_fallback_policy
from app.database import add_fallback_policy, add_provider, add_user, add_user_api_key, init_db
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    import app.database as db_mod

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
    add_user({"username": "alice", "display_name": "Alice", "enabled": True})
    key = add_user_api_key("alice", "default", ["*"])
    yield {"headers": {"Authorization": f"Bearer {key['key']}"}}
    db_mod._initialized = False


def test_apply_fallback_policy_exposes_attempt_timeout():
    add_fallback_policy({
        "name": "t",
        "match_provider": "PixelAPI",
        "match_model": "gpt-5.5",
        "attempt_timeout": 30,
        "chain": [{"model": "gpt-5.5", "provider_id": "qianye"}],
    })
    decision = apply_fallback_policy("PixelAPI", "gpt-5.5", "")
    assert decision.matched is True
    assert decision.attempt_timeout == 30


def test_nonstream_primary_attempt_timeout_switches_to_fallback(monkeypatch, temp_db):
    import app.router.proxy as proxy
    from app.core.output import InternalOutputMessage

    add_provider({
        "id": "slow-primary",
        "name": "Slow Primary",
        "provider_type": "openai",
        "api_base": "https://slow.example/v1",
        "api_key": "k",
        "enabled": True,
        "models": [{"id": "slow-model", "name": "Slow", "enabled": True}],
    })
    add_provider({
        "id": "fast-fallback",
        "name": "Fast Fallback",
        "provider_type": "openai",
        "api_base": "https://fast.example/v1",
        "api_key": "k",
        "enabled": True,
        "models": [{"id": "fast-model", "name": "Fast", "enabled": True}],
    })
    add_fallback_policy({
        "name": "slow then fast",
        "enabled": True,
        "match_provider": "slow-primary",
        "match_model": "*slow*",
        "attempt_timeout": 5,
        "triggers": {"timeout": True, "connection_error": True, "http_5xx": True, "http_4xx": True, "http_429": True},
        "chain": [{"model": "fast-model", "provider_id": "fast-fallback"}],
    })

    calls = []

    async def fake_call_nonstream_target(target, internal, *, temperature, max_tokens, log_label, stage):
        calls.append((stage, target.model, target.provider_id))
        if target.provider_id == "slow-primary" or "slow" in str(target.model):
            await asyncio.sleep(30)
            raise RuntimeError("should have been cancelled by attempt_timeout")
        return (
            InternalOutputMessage(role="assistant", text="fallback-ok", usage={"total_tokens": 3}),
            {"id": "fast-fallback", "provider_type": "openai"},
            "fast-fallback",
        )

    monkeypatch.setattr(proxy, "_call_nonstream_target", fake_call_nonstream_target)

    started = time.monotonic()
    response = client.post(
        "/v1/chat/completions",
        headers=temp_db["headers"],
        json={
            "model": "slow-primary/slow-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert "fallback-ok" in response.text
    assert elapsed < 12.0
    assert calls[0][0] == "primary"
    assert any(stage == "fallback" for stage, _, _ in calls)


def test_stream_primary_attempt_timeout_switches_to_fallback(monkeypatch, temp_db):
    import app.router.proxy as proxy

    add_provider({
        "id": "slow-stream",
        "name": "Slow Stream",
        "provider_type": "openai",
        "api_base": "https://slow-stream.example/v1",
        "api_key": "k",
        "enabled": True,
        "models": [{"id": "slow-s", "name": "SlowS", "enabled": True}],
    })
    add_provider({
        "id": "fast-stream",
        "name": "Fast Stream",
        "provider_type": "openai",
        "api_base": "https://fast-stream.example/v1",
        "api_key": "k",
        "enabled": True,
        "models": [{"id": "fast-s", "name": "FastS", "enabled": True}],
    })
    add_fallback_policy({
        "name": "stream timeout fallback",
        "enabled": True,
        "match_provider": "slow-stream",
        "match_model": "*slow*",
        "attempt_timeout": 5,
        "triggers": {"timeout": True, "connection_error": True, "http_5xx": True, "http_4xx": True, "http_429": True},
        "chain": [{"model": "fast-s", "provider_id": "fast-stream"}],
    })

    calls = []

    async def fake_stream_events(**kwargs):
        calls.append(kwargs["provider_id"])
        if kwargs["provider_id"] == "slow-stream":
            await asyncio.sleep(30)
            yield InternalOutputEvent(kind="text_delta", text="too-late")
            return
        yield InternalOutputEvent(kind="message_start", role="assistant")
        yield InternalOutputEvent(kind="text_delta", text="stream-fallback-ok")
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    monkeypatch.setattr(proxy, "iter_openai_chat_output_events", fake_stream_events)

    started = time.monotonic()
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=temp_db["headers"],
        json={
            "model": "slow-stream/slow-s",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        body = response.read().decode("utf-8")
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert "stream-fallback-ok" in body
    assert elapsed < 12.0
    assert calls == ["slow-stream", "fast-stream"]
