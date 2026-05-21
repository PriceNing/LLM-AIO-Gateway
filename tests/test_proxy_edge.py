"""
Integration tests for proxy edge cases — routing, message conversion, circuit breaker.
"""
import json
import pytest
from fastapi.testclient import TestClient
from main import app
from app.config import load_config
from app.database import init_db, add_provider, add_user, add_user_api_key
from app.router.proxy import (
    _conversation_cache_key, _sanitize_args, _wildcard_match,
    _fix_tool_args, _convert_responses_input, _convert_responses_tools,
    _mask_key, _apply_routing_rules,
    TTLDict, ensure_model_allowed, allowed_models_for,
)

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


# ── Conversation cache key ──

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
    assert key1 != key2  # Fix #2: different images → different keys


def test_cache_key_different_api_keys():
    messages = [{"role": "user", "content": "Hi"}]
    key1 = _conversation_cache_key("key-a", messages)
    key2 = _conversation_cache_key("key-b", messages)
    assert key1 != key2  # Different API keys → different cache keys


# ── Wildcard matching ──

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


# ── Sanitize args ──

def test_sanitize_args():
    assert _sanitize_args('{"url": undefined}') == '{"url": ""}'
    assert _sanitize_args('{"a": undefined, "b": undefined}') == '{"a": "", "b": ""}'


# ── Fix tool args in-place ──

def test_fix_tool_args_no_function():
    tc = {"index": 0}
    _fix_tool_args(tc)
    assert tc == {"index": 0}


def test_fix_tool_args_replaces_undefined():
    tc = {"function": {"name": "search", "arguments": '{"q": undefined}'}}
    _fix_tool_args(tc)
    assert tc["function"]["arguments"] == '{"q": ""}'


# ── Key masking ──

def test_mask_key():
    assert _mask_key("sk-aio-abcdefghijklmnopqrstuvwxyz1234567890AB") == "sk-a...90AB"
    assert _mask_key("short") == "short"


# ── Responses input conversion ──

def test_convert_responses_input_simple():
    input_data = [
        {"type": "message", "role": "user", "content": "Hello"}
    ]
    messages = _convert_responses_input(input_data)
    assert messages == [{"role": "user", "content": "Hello"}]


def test_convert_responses_input_with_system():
    input_data = [
        {"type": "message", "role": "developer", "content": "You are helpful."},
        {"type": "message", "role": "user", "content": "Hi"}
    ]
    messages = _convert_responses_input(input_data)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_convert_responses_input_with_function_calls():
    input_data = [
        {"type": "message", "role": "user", "content": "Search for cats"},
        {"type": "function_call", "call_id": "call_1", "name": "search", "arguments": '{"q": "cats"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "Found 5 cats"}
    ]
    messages = _convert_responses_input(input_data)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "search"
    assert messages[2]["role"] == "tool"


def test_convert_responses_tools():
    tools = [
        {"type": "function", "name": "search", "description": "Search", "parameters": {"type": "object"}},
        {"type": "web_search", "name": "web"},
    ]
    converted = _convert_responses_tools(tools)
    assert len(converted) == 1  # web_search filtered out
    assert converted[0]["function"]["name"] == "search"


# ── Routing rules ──

def test_routing_rules_match_model(temp_db):
    from app.database import add_routing_rule
    add_routing_rule({
        "name": "reroute", "enabled": True,
        "username": "", "api_key_pattern": "",
        "match_model": "old-model", "target_model": "new-model", "target_provider": ""
    })
    target_model, target_provider = _apply_routing_rules("alice", "user-key", "old-model", "old-model")
    assert target_model == "new-model"


def test_routing_rules_no_match(temp_db):
    target_model, target_provider = _apply_routing_rules("alice", "user-key", "unmatched", "unmatched")
    assert target_model == "unmatched"


# ── TTLDict ──

def test_ttldict_operations():
    d = TTLDict(ttl_seconds=60, max_size=100)
    d["a"] = 1
    d["b"] = 2
    assert d["a"] == 1
    assert len(d) == 2
    assert set(d.keys()) == {"a", "b"}


# ── API endpoint: model listing with routing ──

def test_list_models_returns_all_when_wildcard(temp_db):
    add_provider({"id": "p1", "name": "P1", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True,
                  "models": [{"id": "m1", "name": "M1", "enabled": True}]})
    response = client.get("/v1/models", headers=temp_db["headers"])
    assert response.status_code == 200
    model_ids = [m["id"] for m in response.json()["data"]]
    assert "p1/m1" in model_ids


# ── API endpoint: auth status ──

def test_auth_status():
    response = client.get("/auth/status")
    assert response.status_code == 200
    data = response.json()
    assert "has_admin" in data


# ── ensure_model_allowed with composite IDs ──

def test_ensure_model_allowed_wildcard_passes():
    """通配符 allowed_models → 任何模型名都放行。"""
    ensure_model_allowed({}, {"allowed_models": ["*"]}, "any-model")


def test_ensure_model_allowed_exact_simple():
    """简单模型名精确匹配。"""
    ensure_model_allowed({}, {"allowed_models": ["gpt-4"]}, "gpt-4")


def test_ensure_model_allowed_exact_composite():
    """复合 ID 精确匹配。"""
    ensure_model_allowed({}, {"allowed_models": ["opencode-go/deepseek-v4-flash"]}, "opencode-go/deepseek-v4-flash")


def test_ensure_model_allowed_composite_extracts_simple():
    """客户端传复合 ID，allowed 配的是简单 model_id → 放行。"""
    ensure_model_allowed({}, {"allowed_models": ["deepseek-v4-flash"]}, "opencode-go/deepseek-v4-flash")


def test_ensure_model_allowed_simple_in_composite_allowed():
    """客户端传简单名，allowed 配的是复合 ID → 放行。"""
    ensure_model_allowed({}, {"allowed_models": ["opencode-go/deepseek-v4-flash"]}, "deepseek-v4-flash")


def test_ensure_model_allowed_blocked():
    """不在 allowed 列表中 → 403。"""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        ensure_model_allowed({}, {"allowed_models": ["gpt-4"]}, "claude-3")
    assert exc.value.status_code == 403


# ── Anthropic passthrough model extraction ──

@pytest.mark.asyncio
async def test_anthropic_passthrough_model_extraction():
    """_anthropic_passthrough 应从复合 ID 提取纯模型名发给上游。"""
    from app.router.proxy import _anthropic_passthrough
    from unittest.mock import AsyncMock, patch, MagicMock

    provider = {"api_base": "https://api.deepseek.com/anthropic", "api_key": "sk-test"}
    body = {"system": "", "tools": []}

    # 构造一个真实的 response mock（非 coroutine）
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
        await _anthropic_passthrough(
            provider, [], body, 100, 0.7, "deepseek/deepseek-v4-pro"
        )
        sent_body = mock_fn.call_args[1]["json"]
        assert sent_body["model"] == "deepseek-v4-pro"  # 不是 deepseek/deepseek-v4-pro
