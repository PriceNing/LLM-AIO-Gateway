"""
Integration tests for proxy edge cases - routing, message conversion, circuit breaker.
"""
import json
import pytest
from fastapi.testclient import TestClient
from main import app
from app.config import load_config
from app.database import init_db, add_provider, add_user, add_user_api_key
from app.core.policy import apply_routing_rules as _apply_routing_rules
from app.core.policy import wildcard_match as _wildcard_match
from app.core.text import mask_key as _mask_key
from app.core.tool_args import fix_tool_args as _fix_tool_args
from app.core.tool_args import sanitize_args as _sanitize_args
from app.protocols.ingress import responses_tools_to_chat_tools
from app.protocols.ir import ir_to_anthropic_messages, ir_to_openai_messages, openai_messages_to_ir, responses_input_to_ir
from app.core.state import (
    TTLDict,
    conversation_cache_key as _conversation_cache_key,
    remember_response_chain_key as _remember_response_chain_key,
    response_chain_cache as _response_chain_cache,
)
from app.router.proxy import (
    ensure_model_allowed, ensure_routed_model_allowed, allowed_models_for,
)
from app.services.routing_targets import adapter_provider_id

client = TestClient(app)


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Use a temporary database and config for each test."""
    import app.database as db_mod
    db_path = str(tmp_path / "test.db")
    config_path = str(tmp_path / "config.json")
    config = load_config(config_path, force_reload=True)
    config.config = {
        "host": "0.0.0.0",
        "port": 8000,
        "database": db_path,
        "logging": {"enabled": False, "level": "INFO", "log_dir": "logs", "retention_days": 30, "console": False}
    }
    config.save()
    db_mod._initialized = False
    init_db(db_path)

    add_user({"username": "alice", "display_name": "Alice", "enabled": True})
    add_user_api_key("alice", "default", ["*"])
    from app.database import get_db
    with get_db() as db:
        db.execute("UPDATE user_api_keys SET key = 'user-key' WHERE username = 'alice'")

    yield {"headers": {"Authorization": "Bearer user-key"}}
    db_mod._initialized = False


# -- Conversation cache key --

def test_cache_key_text_messages():
    messages = [{"role": "user", "content": "Hello world"}]
    key1 = _conversation_cache_key("api-key-1", messages)
    key2 = _conversation_cache_key("api-key-1", messages)
    assert key1 == key2  # Same conversation


def test_cache_key_different_conversations():
    key1 = _conversation_cache_key("k1", [{"role": "user", "content": "Hello"}])
    key2 = _conversation_cache_key("k1", [{"role": "user", "content": "Goodbye"}])
    assert key1 != key2


def test_cache_key_pure_image_isolation():
    """Pure-image conversations should NOT share the same cache key."""
    img1 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}]}]
    img2 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/b.jpg"}}]}]
    key1 = _conversation_cache_key("k1", img1)
    key2 = _conversation_cache_key("k1", img2)
    assert key1 != key2  # Fix #2: different images -> different keys


def test_cache_key_different_api_keys():
    messages = [{"role": "user", "content": "Hi"}]
    key1 = _conversation_cache_key("key-a", messages)
    key2 = _conversation_cache_key("key-b", messages)
    assert key1 != key2  # Different API keys -> different cache keys


def test_cache_key_same_first_user_with_different_tool_history():
    base = [
        {"role": "user", "content": "same start"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_a", "content": "A"},
    ]
    other = [
        {"role": "user", "content": "same start"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_b", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_b", "content": "B"},
    ]
    assert _conversation_cache_key("k1", base) == _conversation_cache_key("k1", other)


# -- Wildcard matching --

def test_wildcard_match_exact():
    assert _wildcard_match("deepseek-v4-pro", "deepseek-v4-pro") is True


def test_wildcard_match_star():
    assert _wildcard_match("deepseek-*", "deepseek-v4-pro") is True
    assert _wildcard_match("deepseek-*", "deepseek-v4-flash") is True
    assert _wildcard_match("deepseek-*", "openai-gpt-4") is False


def test_wildcard_match_complex():
    assert _wildcard_match("*minimax*", "MiniMax-M2.7-highspeed") is True
    assert _wildcard_match("*-highspeed", "MiniMax-M2.7-highspeed") is True


def test_wildcard_match_case_insensitive():
    assert _wildcard_match("MINIMAX*", "minimax-M2.7") is True


# -- Sanitize args --

def test_sanitize_args():
    assert _sanitize_args('{"url": undefined}') == '{"url": ""}'
    assert _sanitize_args('{"a": undefined, "b": undefined}') == '{"a": "", "b": ""}'


# -- Fix tool args in-place --

def test_fix_tool_args_no_function():
    tc = {"index": 0}
    _fix_tool_args(tc)
    assert tc == {"index": 0}


def test_fix_tool_args_replaces_undefined():
    tc = {"function": {"name": "search", "arguments": '{"q": undefined}'}}
    _fix_tool_args(tc)
    assert tc["function"]["arguments"] == '{"q": ""}'


# -- Key masking --

def test_mask_key():
    assert _mask_key("sk-aio-abcdefghijklmnopqrstuvwxyz1234567890AB") == "sk-a...90AB"
    assert _mask_key("short") == "short"


# -- Responses input conversion --

def test_responses_input_ir_projects_simple_message_to_openai_shape():
    input_data = [
        {"type": "message", "role": "user", "content": "Hello"}
    ]
    messages = ir_to_openai_messages(responses_input_to_ir(input_data))
    assert messages == [{"role": "user", "content": "Hello"}]


def test_responses_input_ir_projects_developer_message_to_system_shape():
    input_data = [
        {"type": "message", "role": "developer", "content": "You are helpful."},
        {"type": "message", "role": "user", "content": "Hi"}
    ]
    messages = ir_to_openai_messages(responses_input_to_ir(input_data))
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_responses_input_ir_projects_function_calls_to_openai_tool_shape():
    input_data = [
        {"type": "message", "role": "user", "content": "Search for cats"},
        {"type": "function_call", "call_id": "call_1", "name": "search", "arguments": '{"q": "cats"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "Found 5 cats"}
    ]
    messages = ir_to_openai_messages(responses_input_to_ir(input_data))
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "search"
    assert messages[2]["role"] == "tool"


def test_responses_tools_project_to_openai_chat_tools():
    tools = [
        {"type": "function", "name": "search", "description": "Search", "parameters": {"type": "object"}},
        {"type": "web_search", "name": "web"},
    ]
    converted = responses_tools_to_chat_tools(tools)
    assert len(converted) == 1  # web_search filtered out
    assert converted[0]["function"]["name"] == "search"


# -- Routing rules --

def test_routing_rules_match_model(temp_db):
    from app.database import add_routing_rule
    add_routing_rule({
        "name": "reroute", "enabled": True,
        "username": "", "api_key_pattern": "",
        "match_model": "old-model", "target_model": "new-model", "target_provider": ""
    })
    decision = _apply_routing_rules("alice", "user-key", "old-model", "old-model")

    assert decision.matched is True
    assert decision.target_model == "new-model"
    assert decision.target_provider == ""
    assert decision.rule_name == "reroute"
    assert decision.source == "routing_rule"


def test_routing_rules_no_match(temp_db):
    decision = _apply_routing_rules("alice", "user-key", "unmatched", "unmatched")

    assert decision.matched is False
    assert decision.target_model == "unmatched"
    assert decision.target_provider == ""
    assert decision.source == "default"


def test_routing_rules_match_wildcard_and_provider(temp_db):
    from app.database import add_routing_rule
    add_routing_rule({
        "name": "wild-provider", "enabled": True,
        "username": "alice", "api_key_pattern": "user",
        "match_model": "old-*", "target_model": "new-model", "target_provider": "target-provider"
    })
    decision = _apply_routing_rules("alice", "user-key", "source/old-model", "old-model")

    assert decision.matched is True
    assert decision.target_model == "new-model"
    assert decision.target_provider == "target-provider"
    assert decision.rule_name == "wild-provider"


def test_routing_rules_can_set_provider_only(temp_db):
    from app.database import add_routing_rule
    add_routing_rule({
        "name": "provider-only", "enabled": True,
        "username": "", "api_key_pattern": "",
        "match_model": "same-model", "target_model": "", "target_provider": "target-provider"
    })
    decision = _apply_routing_rules("alice", "user-key", "same-model", "same-model")

    assert decision.matched is True
    assert decision.target_model == "same-model"
    assert decision.target_provider == "target-provider"


def test_fallback_policy_matches_current_target(temp_db):
    from app.core.policy import apply_fallback_policy
    from app.database import add_fallback_policy
    add_fallback_policy({
        "name": "fallback-chain", "enabled": True,
        "match_provider": "primary-provider", "match_model": "primary-model",
        "chain": [
            {"model": "fallback-a", "provider_id": "provider-a"},
            "fallback-b",
        ],
    })
    decision = apply_fallback_policy("primary-provider", "primary-model", "http_5xx")

    assert decision.matched is True
    assert decision.policy_name == "fallback-chain"
    assert decision.chain[0].model == "fallback-a"
    assert decision.chain[0].provider_id == "provider-a"
    assert decision.chain[1].model == "fallback-b"


def test_fallback_policy_prefers_specific_model_over_wildcard(temp_db):
    from app.core.policy import apply_fallback_policy
    from app.database import add_fallback_policy
    add_fallback_policy({
        "name": "wildcard-fallback", "enabled": True,
        "match_provider": "primary-provider", "match_model": "*",
        "chain": [{"model": "wildcard-backup", "provider_id": "backup-provider"}],
    })
    add_fallback_policy({
        "name": "specific-fallback", "enabled": True,
        "match_provider": "primary-provider", "match_model": "gpt-5.5",
        "chain": [{"model": "specific-backup", "provider_id": "backup-provider"}],
    })

    decision = apply_fallback_policy("primary-provider", "gpt-5.5", "http_5xx")

    assert decision.matched is True
    assert decision.policy_name == "specific-fallback"
    assert decision.chain[0].model == "specific-backup"


def test_routing_rules_do_not_carry_fallback_chain(temp_db):
    from app.database import add_routing_rule
    add_routing_rule({
        "name": "route-only", "enabled": True,
        "username": "", "api_key_pattern": "",
        "match_model": "old-model", "target_model": "primary-model", "target_provider": "primary-provider",
    })
    decision = _apply_routing_rules("alice", "user-key", "old-model", "old-model")

    assert decision.matched is True
    assert decision.target_model == "primary-model"
    assert not hasattr(decision, "fallbacks")


# -- TTLDict --

def test_ttldict_operations():
    d = TTLDict(ttl_seconds=60, max_size=100)
    d["a"] = 1
    d["b"] = 2
    assert d["a"] == 1
    assert len(d) == 2
    assert set(d.keys()) == {"a", "b"}


# -- API endpoint: model listing with routing --

def test_list_models_returns_all_when_wildcard(temp_db):
    add_provider({"id": "p1", "name": "P1", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True,
                  "models": [{"id": "m1", "name": "M1", "enabled": True}]})
    response = client.get("/v1/models", headers=temp_db["headers"])
    assert response.status_code == 200
    model_ids = [m["id"] for m in response.json()["data"]]
    assert "p1/m1" in model_ids


# -- API endpoint: auth status --

def test_auth_status():
    response = client.get("/auth/status")
    assert response.status_code == 200
    data = response.json()
    assert "has_admin" in data


# -- ensure_model_allowed with composite IDs --

def test_ensure_model_allowed_wildcard_passes():
    ensure_model_allowed({}, {"allowed_models": ["*"]}, "any-model")


def test_ensure_model_allowed_exact_simple():
    ensure_model_allowed({}, {"allowed_models": ["gpt-4"]}, "gpt-4")


def test_ensure_model_allowed_exact_composite():
    ensure_model_allowed({}, {"allowed_models": ["opencode-go/deepseek-v4-flash"]}, "opencode-go/deepseek-v4-flash")


def test_ensure_model_allowed_composite_extracts_simple():
    ensure_model_allowed({}, {"allowed_models": ["deepseek-v4-flash"]}, "opencode-go/deepseek-v4-flash")


def test_ensure_model_allowed_simple_rejected_when_only_composite_allowed():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        ensure_model_allowed({}, {"allowed_models": ["opencode-go/deepseek-v4-flash"]}, "deepseek-v4-flash")
    assert exc.value.status_code == 403
    assert "provider-qualified" in exc.value.detail


def test_ensure_model_allowed_simple_passes_when_simple_allowed_with_composites():
    ensure_model_allowed({}, {"allowed_models": ["deepseek-v4-flash", "opencode-go/deepseek-v4-flash"]}, "deepseek-v4-flash")


def test_ensure_model_allowed_blocked():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        ensure_model_allowed({}, {"allowed_models": ["gpt-4"]}, "claude-3")
    assert exc.value.status_code == 403


def test_ensure_routed_model_allowed_accepts_allowed_composite_target():
    ensure_routed_model_allowed(
        {},
        {"allowed_models": ["PixelAPI/gpt-5.5"]},
        "gpt-5.4",
        "PixelAPI/gpt-5.5",
    )


def test_ensure_routed_model_allowed_accepts_provider_target_pair():
    ensure_routed_model_allowed(
        {},
        {"allowed_models": ["PixelAPI/gpt-5.5"]},
        "gpt-5.4",
        "gpt-5.5",
        "PixelAPI",
    )


# -- Anthropic adapter model extraction --

@pytest.mark.asyncio
async def test_anthropic_adapter_model_extraction():
    from app.adapters.anthropic import anthropic_messages_completion
    from unittest.mock import patch, MagicMock

    provider = {"api_base": "https://api.deepseek.com/anthropic", "api_key": "sk-test"}
    body = {"system": "", "tools": []}

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": "msg_1",
        "content": [{"type": "text", "text": "Hello"}],
        "model": "deepseek-v4-pro",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }

    async def mock_post(*args, **kwargs):
        return mock_resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post) as mock_fn:
        await anthropic_messages_completion(
            provider, [], body, 100, 0.7, "deepseek/deepseek-v4-pro"
        )
        sent_body = mock_fn.call_args[1]["json"]
        assert sent_body["model"] == "deepseek-v4-pro"  #  deepseek/deepseek-v4-pro


@pytest.mark.asyncio
async def test_responses_anthropic_stream_uses_internal_events(monkeypatch):
    from app.adapters import anthropic_streaming
    import types

    provider = {
        "id": "pixel-api",
        "provider_type": "anthropic",
        "api_base": "https://ai-pixel.online",
        "api_key": "sk-test",
        "extra_headers": {},
    }

    class FakeStream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield "event: message_start"
            yield "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}"
            yield "event: content_block_start"
            yield "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}"
            yield "event: content_block_delta"
            yield "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}"
            yield "event: message_delta"
            yield "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":2}}"
            yield "event: message_stop"
            yield "data: {\"type\":\"message_stop\"}"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            self.called = {"args": args, "kwargs": kwargs}
            return FakeStream()

    monkeypatch.setattr(anthropic_streaming, "httpx", types.SimpleNamespace(AsyncClient=FakeClient))

    from app.adapters.anthropic_streaming import iter_anthropic_output_events
    from app.protocols.egress import render_responses_sse

    chunks = []
    events = iter_anthropic_output_events(
        provider_info=provider,
        messages=[{"role": "user", "content": "Hi"}],
        body={"system": "", "tools": []},
        max_tokens=16,
        temperature=0.7,
        model="gpt-5.5",
    )
    async for line in render_responses_sse(events, model="gpt-5.5"):
        chunks.append(line)

    joined = "".join(chunks)
    assert "response.created" in joined
    assert "response.completed" in joined
    assert "hello" in joined


@pytest.mark.asyncio
async def test_anthropic_stream_usage_accepts_openai_compatible_keys(monkeypatch):
    from app.adapters import anthropic_streaming
    from app.adapters.anthropic_streaming import iter_anthropic_output_events
    import types

    class FakeStream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield 'event: message_start'
            yield 'data: {"type":"message_start","usage":{"prompt_tokens":11}}'
            yield 'event: message_delta'
            yield 'data: {"type":"message_delta","usage":{"completion_tokens":5}}'
            yield 'event: message_stop'
            yield 'data: {"type":"message_stop"}'

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            return FakeStream()

    monkeypatch.setattr(anthropic_streaming, "httpx", types.SimpleNamespace(AsyncClient=FakeClient))

    usage_events = []
    async for event in iter_anthropic_output_events(
        provider_info={"id": "anth", "api_base": "https://anth.example", "api_key": "key"},
        messages=[{"role": "user", "content": "hi"}],
        body={},
        max_tokens=16,
        temperature=0.7,
        model="claude-test",
    ):
        if event.kind == "usage":
            usage_events.append(event.usage)

    assert usage_events[-1] == {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16}


def test_anthropic_message_url_avoids_duplicate_v1():
    from app.adapters.anthropic import _anthropic_message_url

    assert _anthropic_message_url("https://anth.example/v1") == "https://anth.example/v1/messages"
    assert _anthropic_message_url("https://anth.example") == "https://anth.example/v1/messages"


def test_adapter_provider_id_prefers_resolved_provider():
    assert adapter_provider_id({"id": "NewAPI"}, "") == "NewAPI"
    assert adapter_provider_id(None, "pixel-api") == "pixel-api"


@pytest.mark.asyncio
async def test_chat_sse_tool_call_done_includes_finish_reason():
    from app.core.output import InternalOutputEvent
    from app.protocols.egress import render_chat_completions_sse

    async def events():
        yield InternalOutputEvent(kind="message_start", role="assistant")
        yield InternalOutputEvent(kind="tool_call_start", tool_index=0, tool_call_id="call_1", name="run")
        yield InternalOutputEvent(kind="tool_call_arguments_delta", tool_index=0, tool_call_id="call_1", name="run", arguments_delta="{}", arguments="{}")
        yield InternalOutputEvent(kind="message_done", finish_reason="tool_calls")

    chunks = []
    async for line in render_chat_completions_sse(events(), model="gpt-test"):
        chunks.append(line)

    assert '"finish_reason": "tool_calls"' in "".join(chunks)


@pytest.mark.asyncio
async def test_responses_sse_adds_tool_item_when_arguments_arrive_first():
    from app.core.output import InternalOutputEvent
    from app.protocols.egress import render_responses_sse

    async def events():
        yield InternalOutputEvent(
            kind="tool_call_arguments_delta",
            tool_index=0,
            tool_call_id="call_1",
            call_id="call_1",
            name="run",
            arguments_delta="{}",
            arguments="{}",
        )
        yield InternalOutputEvent(kind="message_done", finish_reason="tool_calls")

    chunks = []
    async for line in render_responses_sse(events(), model="gpt-test"):
        chunks.append(line)

    joined = "".join(chunks)
    added_pos = joined.index('"type": "response.output_item.added"')
    delta_pos = joined.index('"type": "response.function_call_arguments.delta"')
    assert added_pos < delta_pos
    assert '"type": "function_call"' in joined


@pytest.mark.asyncio
async def test_anthropic_messages_sse_includes_input_tokens():
    from app.core.output import InternalOutputEvent
    from app.protocols.egress import render_anthropic_messages_sse

    async def events():
        yield InternalOutputEvent(kind="usage", usage={"input_tokens": 7, "output_tokens": 2, "total_tokens": 9})
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    chunks = []
    async for line in render_anthropic_messages_sse(events(), model="claude-test"):
        chunks.append(line)

    joined = "".join(chunks)
    assert '"input_tokens": 7' in joined
    assert '"output_tokens": 2' in joined


@pytest.mark.asyncio
async def test_anthropic_stream_accumulates_tool_delta_without_block_start(monkeypatch):
    from app.adapters import anthropic_streaming
    from app.adapters.anthropic_streaming import iter_anthropic_output_events
    import types

    class FakeStream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield 'event: content_block_delta'
            yield 'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"a\\":"}}'
            yield 'event: content_block_delta'
            yield 'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"1}"}}'
            yield 'event: content_block_stop'
            yield 'data: {"type":"content_block_stop","index":0}'
            yield 'event: message_delta'
            yield 'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":1}}'
            yield 'event: message_stop'
            yield 'data: {"type":"message_stop"}'

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            return FakeStream()

    monkeypatch.setattr(anthropic_streaming, "httpx", types.SimpleNamespace(AsyncClient=FakeClient))

    seen = []
    async for event in iter_anthropic_output_events(
        provider_info={"id": "anth", "api_base": "https://anth.example", "api_key": "key"},
        messages=[{"role": "user", "content": "hi"}],
        body={},
        max_tokens=16,
        temperature=0.7,
        model="claude-test",
    ):
        seen.append(event)

    assert any(event.kind == "tool_call_arguments_delta" and event.arguments == '{"a":1}' for event in seen)


@pytest.mark.asyncio
async def test_anthropic_stream_raises_on_upstream_error_event(monkeypatch):
    from app.adapters import anthropic_streaming
    from app.adapters.anthropic_streaming import iter_anthropic_output_events
    import types

    class FakeStream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield 'event: error'
            yield 'data: {"type":"error","error":{"message":"bad stream"}}'

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *args, **kwargs):
            return FakeStream()

    monkeypatch.setattr(anthropic_streaming, "httpx", types.SimpleNamespace(AsyncClient=FakeClient))

    with pytest.raises(Exception, match="bad stream"):
        async for _ in iter_anthropic_output_events(
            provider_info={"id": "anth", "api_base": "https://anth.example", "api_key": "key"},
            messages=[{"role": "user", "content": "hi"}],
            body={},
            max_tokens=16,
            temperature=0.7,
            model="claude-test",
        ):
            pass


def test_ir_projects_openai_system_to_anthropic_system_without_duplication():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    anthropic_msgs, system_text = ir_to_anthropic_messages(openai_messages_to_ir(messages))
    assert len(anthropic_msgs) == 1
    assert anthropic_msgs[0]["role"] == "user"
    assert system_text == "You are helpful."


def test_chat_completions_anthropic_provider_uses_direct_adapter(monkeypatch, temp_db):
    add_provider({
        "id": "anth-chat",
        "name": "Anth Chat",
        "provider_type": "anthropic",
        "api_base": "https://anth.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "anth-model", "name": "Anth Model", "enabled": True}],
    })

    called = {}

    async def fake_anthropic_completion(provider_info, internal):
        called["provider_type"] = provider_info["provider_type"]
        called["model"] = internal.target_model
        from app.core.output import InternalOutputMessage
        return InternalOutputMessage(
            role="assistant",
            text="ok",
            finish_reason="stop",
            usage={"total_tokens": 3},
        )

    def fail_litellm(*args, **kwargs):
        raise AssertionError("chat/completions should not route Anthropic providers through liteLLM")

    monkeypatch.setattr("app.router.proxy.anthropic_messages_completion_for_internal", fake_anthropic_completion)
    monkeypatch.setattr("app.router.proxy.create_chat_completion", fail_litellm)

    response = client.post("/v1/chat/completions", headers=temp_db["headers"], json={
        "model": "anth-chat/anth-model",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
    })

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert called == {"provider_type": "anthropic", "model": "anth-chat/anth-model"}


def test_completions_anthropic_provider_uses_direct_adapter(monkeypatch, temp_db):
    add_provider({
        "id": "anth-text",
        "name": "Anth Text",
        "provider_type": "anthropic",
        "api_base": "https://anth.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "text-model", "name": "Text Model", "enabled": True}],
    })

    called = {}

    async def fake_anthropic_completion(provider_info, internal):
        called["provider_type"] = provider_info["provider_type"]
        called["endpoint"] = internal.endpoint
        called["prompt"] = internal.messages[0].parts[0].text
        from app.core.output import InternalOutputMessage
        return InternalOutputMessage(
            role="assistant",
            text="completion ok",
            finish_reason="stop",
            usage={"total_tokens": 4},
        )

    def fail_litellm(*args, **kwargs):
        raise AssertionError("/completions should not route Anthropic providers through liteLLM")

    monkeypatch.setattr("app.router.proxy.anthropic_messages_completion_for_internal", fake_anthropic_completion)
    monkeypatch.setattr("app.router.proxy.create_chat_completion", fail_litellm)

    response = client.post("/v1/completions", headers=temp_db["headers"], json={
        "model": "anth-text/text-model",
        "prompt": "Complete me",
    })

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "completion ok"
    assert called == {"provider_type": "anthropic", "endpoint": "completions", "prompt": "Complete me"}


def test_root_proxy_aliases_are_registered_and_callable(monkeypatch, temp_db):
    from app.core.output import InternalOutputMessage

    add_provider({
        "id": "root-alias",
        "name": "Root Alias",
        "provider_type": "openai",
        "api_base": "https://root-alias.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "alias-model", "name": "Alias Model", "enabled": True}],
    })

    def fake_completion(**kwargs):
        class Usage:
            prompt_tokens = 1
            completion_tokens = 1
            total_tokens = 2

        class Message:
            content = "alias ok"
            tool_calls = None
            reasoning_content = None

        class Choice:
            message = Message()
            finish_reason = "stop"

        class Response:
            choices = [Choice()]
            usage = Usage()

        return Response()

    async def fake_internal_completion(provider_info, internal):
        return InternalOutputMessage(role="assistant", text="alias ok", finish_reason="stop", usage={"total_tokens": 2})

    monkeypatch.setattr("app.router.proxy.create_chat_completion", fake_completion)
    monkeypatch.setattr("app.router.proxy.anthropic_messages_completion_for_internal", fake_internal_completion)

    models = client.get("/models", headers=temp_db["headers"])
    assert models.status_code == 200
    assert "root-alias/alias-model" in [item["id"] for item in models.json()["data"]]

    chat = client.post("/chat/completions", headers=temp_db["headers"], json={
        "model": "root-alias/alias-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == "alias ok"

    completion = client.post("/completions", headers=temp_db["headers"], json={
        "model": "root-alias/alias-model",
        "prompt": "hi",
    })
    assert completion.status_code == 200
    assert completion.json()["choices"][0]["text"] == "alias ok"

    messages = client.post("/messages", headers=temp_db["headers"], json={
        "model": "root-alias/alias-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert messages.status_code == 200
    assert messages.json()["content"][0]["text"] == "alias ok"

    responses = client.post("/responses", headers=temp_db["headers"], json={
        "model": "root-alias/alias-model",
        "input": "hi",
    })
    assert responses.status_code == 200
    assert responses.json()["output"][0]["content"][0]["text"] == "alias ok"


def test_responses_allows_alias_when_route_target_is_allowed(monkeypatch, temp_db):
    from app.database import add_routing_rule, get_db

    add_provider({
        "id": "PixelAPI",
        "name": "PixelAPI",
        "provider_type": "openai",
        "api_base": "https://pixel.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "gpt-5.5", "name": "gpt-5.5", "enabled": True}],
    })
    with get_db() as db:
        db.execute("UPDATE user_api_keys SET allowed_models = ? WHERE key = 'user-key'", ('["PixelAPI/gpt-5.5"]',))
    add_routing_rule({
        "name": "Codex alias",
        "enabled": True,
        "match_model": "gpt-5.4",
        "target_model": "PixelAPI/gpt-5.5",
    })

    called = {}

    def fake_completion(**kwargs):
        called["model"] = kwargs["model"]
        called["provider_id"] = kwargs["provider_id"]

        class Usage:
            prompt_tokens = 1
            completion_tokens = 1
            total_tokens = 2

        class Message:
            content = "route ok"
            tool_calls = None
            reasoning_content = None

        class Choice:
            message = Message()
            finish_reason = "stop"

        class Response:
            choices = [Choice()]
            usage = Usage()

        return Response()

    async def fail_anthropic(provider_info, internal):
        raise AssertionError("OpenAI-compatible route should use liteLLM adapter")

    monkeypatch.setattr("app.router.proxy.create_chat_completion", fake_completion)
    monkeypatch.setattr("app.router.proxy.anthropic_messages_completion_for_internal", fail_anthropic)

    response = client.post("/responses", headers=temp_db["headers"], json={
        "model": "gpt-5.4",
        "input": "hi",
    })

    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "route ok"
    assert called == {"model": "PixelAPI/gpt-5.5", "provider_id": "PixelAPI"}


def test_completions_openai_provider_uses_ir_chat_adapter(monkeypatch, temp_db):
    add_provider({
        "id": "openai-text",
        "name": "OpenAI Text",
        "provider_type": "openai",
        "api_base": "https://openai.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "text-model", "name": "Text Model", "enabled": True}],
    })

    called = {}

    def fake_chat_completion(**kwargs):
        called["messages"] = kwargs["messages"]
        called["provider_id"] = kwargs["provider_id"]

        class Message:
            content = "openai completion"
            reasoning_content = None
            tool_calls = []

        class Choice:
            message = Message()
            finish_reason = "stop"

        class Response:
            choices = [Choice()]
            usage = {"total_tokens": 5}

        return Response()

    monkeypatch.setattr("app.router.proxy.create_chat_completion", fake_chat_completion)

    response = client.post("/v1/completions", headers=temp_db["headers"], json={
        "model": "openai-text/text-model",
        "prompt": "Complete me",
    })

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "openai completion"
    assert called["messages"] == [{"role": "user", "content": "Complete me"}]
    assert called["provider_id"] == "openai-text"


def test_chat_completions_nonstream_uses_fallback_on_upstream_error(monkeypatch, temp_db):
    from app.database import add_fallback_policy, add_routing_rule
    add_provider({
        "id": "primary-fail",
        "name": "Primary Fail",
        "provider_type": "openai",
        "api_base": "https://primary.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "primary-model", "name": "Primary", "enabled": True}],
    })
    add_provider({
        "id": "fallback-ok",
        "name": "Fallback OK",
        "provider_type": "openai",
        "api_base": "https://fallback.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "fallback-model", "name": "Fallback", "enabled": True}],
    })
    add_routing_rule({
        "name": "fallback-rule", "enabled": True,
        "match_model": "source-model", "target_model": "primary-model", "target_provider": "primary-fail",
    })
    add_fallback_policy({
        "name": "primary fallback", "enabled": True,
        "match_provider": "primary-fail", "match_model": "primary-model",
        "chain": [{"model": "fallback-model", "provider_id": "fallback-ok"}],
    })

    calls = []

    def fake_chat_completion(**kwargs):
        calls.append((kwargs["model"], kwargs["provider_id"]))
        if kwargs["provider_id"] == "primary-fail":
            raise RuntimeError("primary unavailable")

        class Message:
            content = "fallback response"
            reasoning_content = None
            tool_calls = []

        class Choice:
            message = Message()
            finish_reason = "stop"

        class Response:
            choices = [Choice()]
            usage = {"total_tokens": 7}

        return Response()

    monkeypatch.setattr("app.router.proxy.create_chat_completion", fake_chat_completion)

    response = client.post("/v1/chat/completions", headers=temp_db["headers"], json={
        "model": "source-model",
        "messages": [{"role": "user", "content": "hi"}],
    })

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "fallback response"
    assert calls == [("primary-model", "primary-fail"), ("fallback-model", "fallback-ok")]
    assert response.json()["model"] == "fallback-model"


def test_chat_completions_nonstream_fallback_matches_resolved_provider(monkeypatch, temp_db):
    from app.database import add_fallback_policy

    add_provider({
        "id": "resolved-primary",
        "name": "Resolved Primary",
        "provider_type": "openai",
        "api_base": "https://resolved-primary.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "resolved-primary-model", "name": "Primary", "enabled": True}],
    })
    add_provider({
        "id": "resolved-fallback",
        "name": "Resolved Fallback",
        "provider_type": "openai",
        "api_base": "https://resolved-fallback.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "resolved-fallback-model", "name": "Fallback", "enabled": True}],
    })
    add_fallback_policy({
        "name": "resolved provider fallback", "enabled": True,
        "match_provider": "resolved-primary", "match_model": "resolved-primary/resolved-primary-model",
        "chain": [{"model": "resolved-fallback-model", "provider_id": "resolved-fallback"}],
    })

    calls = []

    def fake_chat_completion(**kwargs):
        calls.append((kwargs["model"], kwargs["provider_id"]))
        if kwargs["provider_id"] == "resolved-primary":
            raise RuntimeError("resolved primary unavailable")

        class Message:
            content = "resolved fallback response"
            reasoning_content = None
            tool_calls = []

        class Choice:
            message = Message()
            finish_reason = "stop"

        class Response:
            choices = [Choice()]
            usage = {"total_tokens": 13}

        return Response()

    monkeypatch.setattr("app.router.proxy.create_chat_completion", fake_chat_completion)

    response = client.post("/v1/chat/completions", headers=temp_db["headers"], json={
        "model": "resolved-primary/resolved-primary-model",
        "messages": [{"role": "user", "content": "hi"}],
    })

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "resolved fallback response"
    assert calls == [("resolved-primary/resolved-primary-model", "resolved-primary"), ("resolved-fallback-model", "resolved-fallback")]


def test_chat_completions_stream_uses_fallback_before_output(monkeypatch, temp_db):
    from app.database import add_fallback_policy, add_routing_rule
    from app.core.output import InternalOutputEvent

    add_provider({
        "id": "primary-stream-fail",
        "name": "Primary Stream Fail",
        "provider_type": "openai",
        "api_base": "https://primary-stream.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "primary-stream-model", "name": "Primary", "enabled": True}],
    })
    add_provider({
        "id": "fallback-stream-ok",
        "name": "Fallback Stream OK",
        "provider_type": "openai",
        "api_base": "https://fallback-stream.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "fallback-stream-model", "name": "Fallback", "enabled": True}],
    })
    add_routing_rule({
        "name": "stream-fallback-rule", "enabled": True,
        "match_model": "stream-source", "target_model": "primary-stream-model", "target_provider": "primary-stream-fail",
    })
    add_fallback_policy({
        "name": "stream primary fallback", "enabled": True,
        "match_provider": "primary-stream-fail", "match_model": "primary-stream-model",
        "chain": [{"model": "fallback-stream-model", "provider_id": "fallback-stream-ok"}],
    })

    calls = []

    async def fake_stream_events(**kwargs):
        calls.append((kwargs["model"], kwargs["provider_id"]))
        if kwargs["provider_id"] == "primary-stream-fail":
            raise RuntimeError("primary stream unavailable")
        yield InternalOutputEvent(kind="message_start", role="assistant")
        yield InternalOutputEvent(kind="text_delta", text="fallback stream response")
        yield InternalOutputEvent(kind="usage", usage={"total_tokens": 11})
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    monkeypatch.setattr("app.router.proxy.iter_openai_chat_output_events", fake_stream_events)

    with client.stream("POST", "/v1/chat/completions", headers=temp_db["headers"], json={
        "model": "stream-source",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "fallback stream response" in body
    assert "primary stream unavailable" not in body
    assert calls == [("primary-stream-model", "primary-stream-fail"), ("fallback-stream-model", "fallback-stream-ok")]


def test_responses_stream_uses_fallback_on_empty_primary_stream(monkeypatch, temp_db):
    from app.database import add_fallback_policy, add_routing_rule
    from app.core.output import InternalOutputEvent

    add_provider({
        "id": "responses-empty-primary",
        "name": "Responses Empty Primary",
        "provider_type": "openai",
        "api_base": "https://responses-empty.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "responses-empty-model", "name": "Primary", "enabled": True}],
    })
    add_provider({
        "id": "responses-fallback-ok",
        "name": "Responses Fallback OK",
        "provider_type": "openai",
        "api_base": "https://responses-fallback.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "responses-fallback-model", "name": "Fallback", "enabled": True}],
    })
    add_routing_rule({
        "name": "responses-empty-rule", "enabled": True,
        "match_model": "responses-empty-source", "target_model": "responses-empty-model", "target_provider": "responses-empty-primary",
    })
    add_fallback_policy({
        "name": "responses empty stream fallback", "enabled": True,
        "match_provider": "responses-empty-primary", "match_model": "responses-empty-model",
        "chain": [{"model": "responses-fallback-model", "provider_id": "responses-fallback-ok"}],
    })

    calls = []

    async def fake_stream_events(**kwargs):
        calls.append((kwargs["model"], kwargs["provider_id"]))
        if kwargs["provider_id"] == "responses-empty-primary":
            yield InternalOutputEvent(kind="message_start", role="assistant")
            yield InternalOutputEvent(kind="message_done", finish_reason="stop")
            return
        yield InternalOutputEvent(kind="message_start", role="assistant")
        yield InternalOutputEvent(kind="text_delta", text="fallback response text")
        yield InternalOutputEvent(kind="usage", usage={"total_tokens": 23})
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    monkeypatch.setattr("app.router.proxy.iter_openai_chat_output_events", fake_stream_events)

    with client.stream("POST", "/v1/responses", headers=temp_db["headers"], json={
        "model": "responses-empty-source",
        "input": [{"type": "message", "role": "user", "content": "hi"}],
        "stream": True,
    }) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "fallback response text" in body
    assert '"total_tokens": 23' in body
    assert calls == [
        ("responses-empty-model", "responses-empty-primary"),
        ("responses-fallback-model", "responses-fallback-ok"),
    ]


def test_chat_completions_stream_preprocesses_images_for_fallback_target(monkeypatch, temp_db):
    from app.database import add_fallback_policy
    from app.core.output import InternalOutputEvent
    from app.core.types import text_part
    import app.router.proxy as proxy

    add_provider({
        "id": "native-vision-fail",
        "name": "Native Vision Fail",
        "provider_type": "openai",
        "api_base": "https://native-vision.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "native-vision-model", "name": "Native Vision", "enabled": True}],
    })
    add_provider({
        "id": "text-fallback-ok",
        "name": "Text Fallback OK",
        "provider_type": "openai",
        "api_base": "https://text-fallback.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "text-fallback-model", "name": "Text Fallback", "enabled": True, "preprocessor": "1"}],
    })
    add_fallback_policy({
        "name": "vision to text fallback", "enabled": True,
        "match_provider": "native-vision-fail", "match_model": "native-vision-fail/native-vision-model",
        "chain": [{"model": "text-fallback-model", "provider_id": "text-fallback-ok"}],
    })

    calls = []
    fallback_messages = []

    async def fake_policy_preprocess_request(internal, model, provider_id, requested_model):
        if provider_id != "text-fallback-ok":
            return False
        assert model == "text-fallback-model"
        for message in internal.messages:
            if any(part.kind == "image" for part in message.parts):
                message.parts = [text_part("[image described by fallback preprocessor]")]
        return True

    async def fake_stream_events(**kwargs):
        calls.append((kwargs["model"], kwargs["provider_id"]))
        if kwargs["provider_id"] == "native-vision-fail":
            raise RuntimeError("native stream unavailable")
        fallback_messages.extend(kwargs["messages"])
        yield InternalOutputEvent(kind="message_start", role="assistant")
        yield InternalOutputEvent(kind="text_delta", text="fallback saw text")
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    monkeypatch.setattr(proxy, "_policy_preprocess_request", fake_policy_preprocess_request)
    monkeypatch.setattr(proxy, "iter_openai_chat_output_events", fake_stream_events)

    with client.stream("POST", "/v1/chat/completions", headers=temp_db["headers"], json={
        "model": "native-vision-fail/native-vision-model",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
        "stream": True,
    }) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "fallback saw text" in body
    assert calls == [("native-vision-fail/native-vision-model", "native-vision-fail"), ("text-fallback-model", "text-fallback-ok")]
    assert fallback_messages[-1]["content"] == "[image described by fallback preprocessor]"


def test_chat_completions_stream_preserves_images_for_native_vision_fallback(monkeypatch, temp_db):
    from app.database import add_fallback_policy
    from app.core.output import InternalOutputEvent

    add_provider({
        "id": "text-primary-fail",
        "name": "Text Primary Fail",
        "provider_type": "openai",
        "api_base": "https://text-primary.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "text-model", "name": "Text Model", "enabled": True}],
    })
    add_provider({
        "id": "native-vision-ok",
        "name": "Native Vision OK",
        "provider_type": "openai",
        "api_base": "https://native-vision-ok.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "gpt-4o-vision", "name": "GPT 4o Vision", "enabled": True}],
    })
    add_fallback_policy({
        "name": "text to native vision fallback", "enabled": True,
        "match_provider": "text-primary-fail", "match_model": "text-primary-fail/text-model",
        "chain": [{"model": "gpt-4o-vision", "provider_id": "native-vision-ok"}],
    })

    calls = []
    fallback_messages = []

    async def fake_stream_events(**kwargs):
        calls.append((kwargs["model"], kwargs["provider_id"]))
        if kwargs["provider_id"] == "text-primary-fail":
            raise RuntimeError("primary cannot accept images")
        fallback_messages.extend(kwargs["messages"])
        yield InternalOutputEvent(kind="message_start", role="assistant")
        yield InternalOutputEvent(kind="text_delta", text="native vision saw image")
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    monkeypatch.setattr("app.router.proxy.iter_openai_chat_output_events", fake_stream_events)

    with client.stream("POST", "/v1/chat/completions", headers=temp_db["headers"], json={
        "model": "text-primary-fail/text-model",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}],
        "stream": True,
    }) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "native vision saw image" in body
    assert calls == [("text-primary-fail/text-model", "text-primary-fail"), ("gpt-4o-vision", "native-vision-ok")]
    content = fallback_messages[-1]["content"]
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)


def test_chat_completions_stream_fallback_matches_resolved_provider(monkeypatch, temp_db):
    from app.database import add_fallback_policy
    from app.core.output import InternalOutputEvent

    add_provider({
        "id": "resolved-stream-primary",
        "name": "Resolved Stream Primary",
        "provider_type": "openai",
        "api_base": "https://resolved-stream-primary.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "resolved-stream-primary-model", "name": "Primary", "enabled": True}],
    })
    add_provider({
        "id": "resolved-stream-fallback",
        "name": "Resolved Stream Fallback",
        "provider_type": "openai",
        "api_base": "https://resolved-stream-fallback.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "resolved-stream-fallback-model", "name": "Fallback", "enabled": True}],
    })
    add_fallback_policy({
        "name": "resolved stream fallback", "enabled": True,
        "match_provider": "resolved-stream-primary", "match_model": "resolved-stream-primary/resolved-stream-primary-model",
        "chain": [{"model": "resolved-stream-fallback-model", "provider_id": "resolved-stream-fallback"}],
    })

    calls = []

    async def fake_stream_events(**kwargs):
        calls.append((kwargs["model"], kwargs["provider_id"]))
        if kwargs["provider_id"] == "resolved-stream-primary":
            raise RuntimeError("resolved stream primary unavailable")
        yield InternalOutputEvent(kind="message_start", role="assistant")
        yield InternalOutputEvent(kind="text_delta", text="resolved stream fallback response")
        yield InternalOutputEvent(kind="usage", usage={"total_tokens": 17})
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    monkeypatch.setattr("app.router.proxy.iter_openai_chat_output_events", fake_stream_events)

    with client.stream("POST", "/v1/chat/completions", headers=temp_db["headers"], json={
        "model": "resolved-stream-primary/resolved-stream-primary-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "resolved stream fallback response" in body
    assert calls == [
        ("resolved-stream-primary/resolved-stream-primary-model", "resolved-stream-primary"),
        ("resolved-stream-fallback-model", "resolved-stream-fallback"),
    ]


def test_chat_completions_stream_logs_fallback_target_model(monkeypatch, temp_db):
    from app.database import add_fallback_policy, add_routing_rule
    from app.core.output import InternalOutputEvent
    import app.router.proxy as proxy

    add_provider({
        "id": "primary-stream-log-fail",
        "name": "Primary Stream Log Fail",
        "provider_type": "openai",
        "api_base": "https://primary-stream-log.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "primary-stream-log-model", "name": "Primary", "enabled": True}],
    })
    add_provider({
        "id": "fallback-stream-log-ok",
        "name": "Fallback Stream Log OK",
        "provider_type": "openai",
        "api_base": "https://fallback-stream-log.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "fallback-stream-log-model", "name": "Fallback", "enabled": True}],
    })
    add_routing_rule({
        "name": "stream-log-rule", "enabled": True,
        "match_model": "stream-log-source", "target_model": "primary-stream-log-model", "target_provider": "primary-stream-log-fail",
    })
    add_fallback_policy({
        "name": "stream log fallback", "enabled": True,
        "match_provider": "primary-stream-log-fail", "match_model": "primary-stream-log-model",
        "chain": [{"model": "fallback-stream-log-model", "provider_id": "fallback-stream-log-ok"}],
    })

    logged = []

    def fake_log_request(username, api_key, model, provider_id, endpoint, success, tokens, requested_model="", **kwargs):
        logged.append((model, provider_id, endpoint, success, requested_model))

    async def fake_stream_events(**kwargs):
        if kwargs["provider_id"] == "primary-stream-log-fail":
            raise RuntimeError("primary stream unavailable")
        yield InternalOutputEvent(kind="message_start", role="assistant")
        yield InternalOutputEvent(kind="text_delta", text="fallback stream response")
        yield InternalOutputEvent(kind="usage", usage={"total_tokens": 19})
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    monkeypatch.setattr(proxy, "_log_request", fake_log_request)
    monkeypatch.setattr("app.router.proxy.iter_openai_chat_output_events", fake_stream_events)

    with client.stream("POST", "/v1/chat/completions", headers=temp_db["headers"], json={
        "model": "stream-log-source",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "fallback stream response" in body
    assert logged[-1] == ("fallback-stream-log-ok/fallback-stream-log-model", "fallback-stream-log-ok", "chat_completions", True, "stream-log-source")


def test_chat_completions_stream_does_not_fallback_after_output(monkeypatch, temp_db):
    from app.database import add_fallback_policy, add_routing_rule
    from app.core.output import InternalOutputEvent

    add_provider({
        "id": "primary-stream-midfail",
        "name": "Primary Stream Midfail",
        "provider_type": "openai",
        "api_base": "https://primary-midfail.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "primary-midfail-model", "name": "Primary", "enabled": True}],
    })
    add_provider({
        "id": "fallback-stream-unused",
        "name": "Fallback Stream Unused",
        "provider_type": "openai",
        "api_base": "https://fallback-unused.example/v1",
        "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "fallback-unused-model", "name": "Fallback", "enabled": True}],
    })
    add_routing_rule({
        "name": "stream-midfail-rule", "enabled": True,
        "match_model": "stream-midfail-source", "target_model": "primary-midfail-model", "target_provider": "primary-stream-midfail",
    })
    add_fallback_policy({
        "name": "stream midfail fallback", "enabled": True,
        "match_provider": "primary-stream-midfail", "match_model": "primary-midfail-model",
        "chain": [{"model": "fallback-unused-model", "provider_id": "fallback-stream-unused"}],
    })

    calls = []

    async def fake_stream_events(**kwargs):
        calls.append((kwargs["model"], kwargs["provider_id"]))
        yield InternalOutputEvent(kind="message_start", role="assistant")
        yield InternalOutputEvent(kind="text_delta", text="partial primary")
        raise RuntimeError("primary failed after output")

    monkeypatch.setattr("app.router.proxy.iter_openai_chat_output_events", fake_stream_events)

    with client.stream("POST", "/v1/chat/completions", headers=temp_db["headers"], json={
        "model": "stream-midfail-source",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as response:
        body = response.read().decode("utf-8")

    assert response.status_code == 200
    assert "partial primary" in body
    assert "primary failed after output" in body
    assert calls == [("primary-midfail-model", "primary-stream-midfail")]


def test_anthropic_output_uses_reasoning_as_text_when_visible_text_empty():
    from app.adapters.anthropic import _anthropic_response_to_internal

    output = _anthropic_response_to_internal({
        "content": [{"type": "thinking", "thinking": "hidden but useful"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 3, "output_tokens": 5},
    })

    assert output.reasoning == "hidden but useful"
    assert output.text == "hidden but useful"
    assert output.finish_reason == "length"


def test_remember_response_chain_key_uses_final_conv_key():
    _response_chain_cache.drop("resp-test")
    _remember_response_chain_key("resp-test", "conv-123")
    assert _response_chain_cache.get("resp-test") == "conv-123"
