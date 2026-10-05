"""F1/F2：畸形 tools / tool_choice 必须是 4xx，不得裸 500。

背景（GATEWAY-FINDINGS-2026-10-05.md）：
- `tools: 123 / 1.5 / true / null` 经网关返回 HTTP 500 纯文本 `Internal Server Error`，
  而同样的输入直连上游是 400。崩点有 4 处，其中 3 处在日志表达式里
  （`len(body.get("tools", []))` 的默认值只在键缺失时生效，显式 null 拿到 None）。
- `tool_choice: 123 / []` 被 `_chat_tool_choice` 原样透传给下游客户端库，
  库侧校验抛异常后被归类为 upstream failure → 500「请联系管理员」，
  但请求根本没有发出去。
"""
import json

import pytest
from fastapi.testclient import TestClient

from main import app
from app.config import load_config
from app.database import init_db, add_user, add_user_api_key

client = TestClient(app, raise_server_exceptions=False)

# 注意：`[1, 2]` 不在列表里——list 内含非 dict 项属于既有的宽容行为
# （tools_from_chat 逐项跳过），文档也未要求收紧。
MALFORMED_TOOLS = [123, 1.5, True, None, "notalist", {"a": 1}]
MALFORMED_TOOL_CHOICE = [123, [], 0, 1.5, True, {"a": 1}, "bogus"]
# 适配器层只拦“不可能是任何协议合法形状”的值；dict 仍透传交上游 4xx。
NON_PROJECTABLE_TOOL_CHOICE = [123, [], 0, 1.5, True]
VALID_TOOL_CHOICE = [
    "auto", "none", "required",
    {"type": "function", "function": {"name": "x"}},
    {"type": "auto"}, {"type": "none"}, {"type": "required"},
    # 旧形：无 type 字段但能解析出工具名（以前透传且上游可用，不得收紧成 400）
    {"function": {"name": "x"}},
    # Anthropic 形状与并行开关
    {"type": "any"}, {"type": "tool", "name": "y"},
    {"type": "auto", "disable_parallel_tool_use": True},
]


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
        "logging": {"enabled": False, "level": "INFO", "log_dir": "logs",
                    "retention_days": 30, "console": False},
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


def _post_chat(headers, **overrides):
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    body.update(overrides)
    return client.post("/v1/chat/completions", headers=headers, json=body)


def _assert_client_error_shape(response):
    """4xx + 网关标准 JSON 错误形状，且不含堆栈。"""
    assert response.status_code == 400, f"期望 400，实际 {response.status_code}: {response.text[:200]}"
    assert response.headers.get("content-type", "").startswith("application/json"), response.text[:200]
    payload = response.json()
    assert "detail" in payload
    detail = payload["detail"] if isinstance(payload["detail"], str) else json.dumps(payload["detail"])
    assert "Traceback" not in detail
    assert "Internal Server Error" not in detail
    assert "Upstream request failed" not in detail


@pytest.mark.parametrize("tools", MALFORMED_TOOLS)
def test_malformed_tools_returns_400_not_500(temp_db, tools):
    _assert_client_error_shape(_post_chat(temp_db["headers"], tools=tools))


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/messages", "/v1/responses"])
def test_malformed_tools_400_on_every_ingress(endpoint, temp_db):
    if endpoint.endswith("/messages"):
        body = {"model": "m", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}], "tools": 123}
    elif endpoint.endswith("/responses"):
        body = {"model": "m", "input": "hi", "tools": 123}
    else:
        body = {"model": "m", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}], "tools": 123}
    response = client.post(endpoint, headers=temp_db["headers"], json=body)
    _assert_client_error_shape(response)


@pytest.mark.parametrize("tools", MALFORMED_TOOLS)
def test_tools_logging_helpers_never_raise(tools):
    """日志/诊断表达式不得有抛异常的能力。"""
    from app.router.proxy import _tools_count, _tools_preview

    body = {"tools": tools}
    assert isinstance(_tools_count(body), int)
    assert isinstance(_tools_preview(body), str)
    # 键缺失与显式 null 都必须安全
    assert _tools_count({}) == 0
    assert _tools_count({"tools": None}) == 0
    assert _tools_preview({"tools": None}) == "none"


@pytest.mark.parametrize("messages", [123, 1.5, True, None, "notalist", {"a": 1}])
def test_malformed_messages_returns_400_not_500(temp_db, messages):
    """与 F1 同一 bug 类：`len(body.get("messages", []))` 对显式 null 会抛 TypeError。"""
    response = client.post(
        "/v1/chat/completions", headers=temp_db["headers"],
        json={"model": "m", "messages": messages, "max_tokens": 16},
    )
    _assert_client_error_shape(response)


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/messages"])
def test_malformed_messages_400_on_both_message_endpoints(endpoint, temp_db):
    body = {"model": "m", "messages": 123, "max_tokens": 16}
    _assert_client_error_shape(client.post(endpoint, headers=temp_db["headers"], json=body))


def test_null_instructions_does_not_crash_responses_logging(temp_db):
    """`len(body.get("instructions", ""))` 对显式 null 会抛 TypeError。

    差分断言：本 fixture 未配 provider，合法请求本身也会失败，因此不能直接
    断言 != 500；改为要求 null 与合法请求**同形**。修复前 null 会在日志表达式
    里抛 TypeError，得到纯文本 `Internal Server Error`；修复后两者一致。
    """
    headers = temp_db["headers"]
    valid = client.post("/v1/responses", headers=headers, json={"model": "m", "input": "hi"})
    with_null = client.post(
        "/v1/responses", headers=headers,
        json={"model": "m", "input": "hi", "instructions": None},
    )
    assert with_null.status_code == valid.status_code
    assert "Internal Server Error" not in with_null.text
    assert with_null.headers.get("content-type", "").startswith("application/json")
    assert "detail" in with_null.json()


def test_ir_message_parsers_never_raise_on_scalars():
    """IR 层防御：`for msg in messages or []` 对 123 这类真值标量会抛 TypeError。"""
    from app.protocols.ir import anthropic_messages_to_ir, openai_messages_to_ir

    for value in (123, 1.5, True, "notalist", {"a": 1}):
        assert openai_messages_to_ir(value) == []
        assert anthropic_messages_to_ir(value) == []


@pytest.mark.parametrize("tool_choice", MALFORMED_TOOL_CHOICE)
def test_malformed_tool_choice_returns_400(temp_db, tool_choice):
    _assert_client_error_shape(_post_chat(temp_db["headers"], tool_choice=tool_choice))


@pytest.mark.parametrize("tool_choice", VALID_TOOL_CHOICE)
def test_valid_tool_choice_shapes_pass_through(temp_db, tool_choice, monkeypatch):
    """合法形状不得被新校验误伤：应走到上游调用（这里让它立刻失败即可）。"""
    import app.router.proxy as proxy_mod

    calls = {}

    async def _fake_call(*args, **kwargs):
        calls["reached"] = True
        raise RuntimeError("stub upstream")

    monkeypatch.setattr(proxy_mod, "_call_nonstream_with_fallbacks", _fake_call)
    response = _post_chat(temp_db["headers"], tool_choice=tool_choice)
    assert response.status_code != 400, f"合法 tool_choice 被误拒: {tool_choice} -> {response.text[:200]}"
    assert calls.get("reached") is True


def test_chat_tool_choice_projects_valid_shapes():
    from app.adapters.openai import _chat_tool_choice

    assert _chat_tool_choice(None) is None
    assert _chat_tool_choice("auto") == "auto"
    assert _chat_tool_choice({"type": "any"}) == "required"
    assert _chat_tool_choice({"type": "function", "function": {"name": "x"}}) == {
        "type": "function", "function": {"name": "x"},
    }
    assert _chat_tool_choice({"type": "tool", "name": "y"}) == {
        "type": "function", "function": {"name": "y"},
    }


@pytest.mark.parametrize("tool_choice", NON_PROJECTABLE_TOOL_CHOICE)
def test_chat_tool_choice_rejects_non_projectable_shapes(tool_choice):
    from app.adapters.openai import _chat_tool_choice

    with pytest.raises(ValueError):
        _chat_tool_choice(tool_choice)


def test_list_of_non_dict_tools_still_tolerated(temp_db, monkeypatch):
    """`tools: [1, 2]` 保持既有宽容行为（逐项跳过），不得被新校验收紧成 400。"""
    import app.router.proxy as proxy_mod

    reached = {}

    async def _fake_call(*args, **kwargs):
        reached["yes"] = True
        raise RuntimeError("stub upstream")

    monkeypatch.setattr(proxy_mod, "_call_nonstream_with_fallbacks", _fake_call)
    response = _post_chat(temp_db["headers"], tools=[1, 2])
    assert response.status_code != 400, response.text[:200]
    assert reached.get("yes") is True
