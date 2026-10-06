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


# -- 流式块间空闲超时（stream_idle_timeout_seconds，默认 0=关闭）--
#
# 事故侧（2026-10-03）：provider.request_timeout=120 既当首字超时又当块间空闲超时，
# 本地长上下文引擎需要 300 秒容忍 prefill，于是真死流也要 5 分钟才发现。本层只补
# "产出开始之后" 这一段，并且**不得**在已产出后换目标（客户端会看到重复内容）。


def _add_stream_provider(provider_id: str, model_id: str) -> None:
    add_provider({
        "id": provider_id,
        "name": provider_id,
        "provider_type": "openai",
        "api_base": f"https://{provider_id}.example/v1",
        "api_key": "k",
        "enabled": True,
        "models": [{"id": model_id, "name": model_id, "enabled": True}],
    })


def test_stream_idle_timeout_cuts_dead_stream_without_retry(monkeypatch, temp_db):
    import app.router.proxy as proxy

    _add_stream_provider("dead-stream", "dead-s")

    calls = []

    async def fake_stream_events(**kwargs):
        calls.append(kwargs["provider_id"])
        yield InternalOutputEvent(kind="message_start", role="assistant")
        yield InternalOutputEvent(kind="text_delta", text="partial-answer")
        await asyncio.sleep(30)  # 产出开始后上游静默：连接已死
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    idle_logs = []
    monkeypatch.setattr(proxy, "iter_openai_chat_output_events", fake_stream_events)
    monkeypatch.setattr(proxy, "_stream_idle_timeout_seconds", lambda: 2.0)
    monkeypatch.setattr(
        proxy._app_log, "warning",
        lambda fmt, *args: idle_logs.append(str(fmt) % args if args else str(fmt)),
    )

    started = time.monotonic()
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=temp_db["headers"],
        json={"model": "dead-stream/dead-s", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        body = response.read().decode("utf-8")
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    # 已吐过字节的流：报错收尾，既不重试（只有一次尝试）也不复制已输出内容。
    assert calls == ["dead-stream"]
    assert body.count("partial-answer") == 1
    assert '"error"' in body
    assert elapsed < 12.0
    assert any("[stream.idle_timeout]" in line for line in idle_logs), idle_logs


def test_stream_idle_timeout_is_disabled_by_default(monkeypatch, temp_db):
    """默认 0 = 关闭：不改变现有部署行为（1 秒静默不会被砍）。"""
    import app.router.proxy as proxy

    _add_stream_provider("slow-but-alive", "alive-s")
    assert proxy._stream_idle_timeout_seconds() == 0.0

    async def fake_stream_events(**kwargs):
        yield InternalOutputEvent(kind="message_start", role="assistant")
        yield InternalOutputEvent(kind="text_delta", text="first-chunk")
        await asyncio.sleep(1.0)
        yield InternalOutputEvent(kind="text_delta", text="-second-chunk")
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    monkeypatch.setattr(proxy, "iter_openai_chat_output_events", fake_stream_events)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=temp_db["headers"],
        json={"model": "slow-but-alive/alive-s", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    # 两个 chunk 分别落在各自的 SSE data 帧里：断言都在、且没有 error 帧。
    assert "first-chunk" in body and "-second-chunk" in body
    assert '"error"' not in body
    assert body.rstrip().endswith("data: [DONE]")


def test_stream_idle_timeout_config_key_is_exposed_and_validated():
    """新键必须自动进设置页 schema（条目由 default_config 生成），且范围受限。"""
    from app.config import default_config
    from app.core.settings_schema import schema, validate

    assert default_config()["defaults"]["stream_idle_timeout_seconds"] == 0
    entry = schema()["stream_idle_timeout_seconds"]
    assert entry["group"] == "upstream"
    assert entry["unit"] == "seconds"
    assert entry["min"] == 0 and entry["max"] == 3600
    assert entry["declared"] is True
    assert entry["hot"] is True  # 每次请求现读，写入即生效
    assert validate("stream_idle_timeout_seconds", 90) == 90
    assert validate("stream_idle_timeout_seconds", 0) == 0
    assert validate("stream_idle_timeout_seconds", 90) == 90
    assert validate("stream_idle_timeout_seconds", 0) == 0


@pytest.mark.asyncio
async def test_native_responses_stream_idle_timeout_cuts_dead_stream(temp_db, monkeypatch):
    """native Responses 的原始 SSE 直通路径也必须被砍。

    这条路径不经过 IR 事件流（`_stream_events_with_fallbacks`），而是把上游 SSE
    帧原样转发，所以块间空闲超时要单独接进去——它同样是本地单槽引擎的常见入口。
    """
    import app.router.proxy as proxy

    monkeypatch.setattr(proxy, "_stream_idle_timeout_seconds", lambda: 0.2)
    frames = [b'data: {"type":"response.output_text.delta","delta":"hello"}' + b"\n\n"]

    async def dead_frames():
        for frame in frames:
            yield frame
        await asyncio.sleep(30)  # 首个含输出的帧之后彻底静默

    idle_logs = []
    monkeypatch.setattr(
        proxy._app_log, "warning",
        lambda fmt, *args: idle_logs.append(str(fmt) % args if args else str(fmt)),
    )

    collected = []
    with pytest.raises(TimeoutError):
        async for frame in proxy._native_responses_stream_with_accounting(
            dead_frames(),
            username="alice",
            api_key_value="k",
            model="native-m",
            provider_id="native-p",
            requested_model="native-m",
            policy={},
            request_body={},
        ):
            collected.append(frame)

    # 已转发的帧不会被撤回；L-17 起中途失败会追加 response.failed 终帧
    assert collected[:len(frames)] == frames
    assert b'"response.failed"' in collected[-1]
    assert collected[-1] != frames[-1]
    assert any("[stream.idle_timeout]" in line for line in idle_logs), idle_logs
