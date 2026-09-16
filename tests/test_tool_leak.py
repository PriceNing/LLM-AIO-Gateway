"""泄漏工具调用抢救层（core/tool_leak）测试。

标记字面量全部用拼接构造，避免测试文件自身被任何工具调用解析器误识别。
"""
import json

import pytest

from app.core.output import InternalOutputMessage
from app.core.tool_leak import parse_leaked_tool_calls, repair_output
from app.core.types import InternalTool

OPEN = "<" + "tool_call" + ">"
CLOSE = "</" + "tool_call" + ">"
FUNC_CLOSE = "</" + "function" + ">"
PARAM_CLOSE = "</" + "parameter" + ">"


def fopen(name):
    return "<" + "function=" + name + ">"


def popen(key):
    return "<" + "parameter=" + key + ">"


def block(name, pairs):
    inner = "".join(popen(k) + "\n" + v + "\n" + PARAM_CLOSE for k, v in pairs)
    return OPEN + "\n" + fopen(name) + inner + FUNC_CLOSE + "\n" + CLOSE


TOOLS = [InternalTool(name="exec_command", parameters={
    "type": "object",
    "properties": {"cmd": {"type": "string"}, "max_output_tokens": {"type": "number"}},
    "required": ["cmd"],
})]

# 事故真实泄漏形态（脱敏重构：与 2026-09-16 Codex×llama.cpp 事件同构）
INCIDENT_TEXT = block("exec_command", [
    ("cmd", '$idx = Get-ChildItem "$env:LOCALAPPDATA' + chr(92) + 'npm-cache' + chr(92) + '_cacache" -Recurse; '
            'Write-Output ("{0}  |  latest={1}" -f $d, $m)'),
    ("max_output_tokens", "2000"),
])


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------

def test_parse_happy_path_with_type_coercion():
    blocks = parse_leaked_tool_calls(INCIDENT_TEXT, TOOLS)
    assert blocks and len(blocks) == 1
    assert blocks[0]["name"] == "exec_command"
    assert blocks[0]["arguments"]["cmd"].startswith("$idx = Get-ChildItem")
    assert blocks[0]["arguments"]["max_output_tokens"] == 2000  # number 类型已转换


def test_parse_multiple_blocks():
    text = block("exec_command", [("cmd", "a")]) + "\n" + block("exec_command", [("cmd", "b")])
    blocks = parse_leaked_tool_calls(text, TOOLS)
    assert [b["arguments"]["cmd"] for b in blocks] == ["a", "b"]


def test_parse_rejects_unknown_tool():
    assert parse_leaked_tool_calls(block("delete_everything", [("cmd", "rm -rf")]), TOOLS) is None


def test_parse_rejects_unknown_parameter():
    assert parse_leaked_tool_calls(block("exec_command", [("path", "x")]), TOOLS) is None


def test_parse_rejects_prose_wrapped():
    assert parse_leaked_tool_calls("Sure! " + INCIDENT_TEXT, TOOLS) is None
    assert parse_leaked_tool_calls(INCIDENT_TEXT + "\nDone.", TOOLS) is None


def test_parse_rejects_code_fence():
    assert parse_leaked_tool_calls("```xml\n" + INCIDENT_TEXT + "\n```", TOOLS) is None


def test_parse_rejects_malformed():
    assert parse_leaked_tool_calls(INCIDENT_TEXT.replace(FUNC_CLOSE, ""), TOOLS) is None
    assert parse_leaked_tool_calls(INCIDENT_TEXT.replace(CLOSE, ""), TOOLS) is None
    # 参数值里出现闭合标签 → 剩余文本无法消费 → 放弃
    assert parse_leaked_tool_calls(block("exec_command", [("cmd", "x" + PARAM_CLOSE + "y")]), TOOLS) is None


def test_parse_requires_declared_tools():
    assert parse_leaked_tool_calls(INCIDENT_TEXT, []) is None
    assert parse_leaked_tool_calls(INCIDENT_TEXT, None) is None


def test_parse_boolean_and_object_types():
    tools = [InternalTool(name="t", parameters={"type": "object", "properties": {
        "flag": {"type": "boolean"}, "payload": {"type": "object"}}})]
    blocks = parse_leaked_tool_calls(block("t", [("flag", "true"), ("payload", '{"a":1}')]), tools)
    assert blocks[0]["arguments"] == {"flag": True, "payload": {"a": 1}}
    assert parse_leaked_tool_calls(block("t", [("flag", "maybe")]), tools) is None


def test_namespace_flattened_alias():
    tools = [InternalTool(name="node_repl.exec_command", parameters={
        "type": "object", "properties": {"cmd": {"type": "string"}}})]
    assert parse_leaked_tool_calls(block("exec_command", [("cmd", "ls")]), tools) is not None


# ---------------------------------------------------------------------------
# repair_output（IR 消息级）
# ---------------------------------------------------------------------------

def test_repair_output_mutates_message():
    output = InternalOutputMessage(text=INCIDENT_TEXT, finish_reason="stop")
    count = repair_output(output, TOOLS)
    assert count == 1
    assert output.text == ""
    assert output.finish_reason == "tool_calls"
    call = output.tool_calls[0]
    assert call.name == "exec_command"
    assert json.loads(call.arguments)["max_output_tokens"] == 2000
    assert call.id.startswith("call_")


def test_repair_output_noop_for_clean_text():
    output = InternalOutputMessage(text="normal answer", finish_reason="stop")
    assert repair_output(output, TOOLS) == 0
    assert output.text == "normal answer"


# ---------------------------------------------------------------------------
# 端点集成：非流式抢救 / 流式检测 / 开关
# ---------------------------------------------------------------------------
from fastapi import HTTPException
from fastapi.testclient import TestClient

from main import app
from app.config import load_config
from app.core.output import InternalOutputEvent
from app.database import init_db, add_provider, add_user, add_user_api_key, get_db
from app.protocols.ingress import chat_completions_to_internal

client = TestClient(app)
headers = {"Authorization": "Bearer user-key"}


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    config_path = str(tmp_path / "config.json")
    config = load_config(config_path, force_reload=True)
    config.config = {
        "host": "0.0.0.0", "port": 8000, "database": db_path,
        "logging": {"enabled": False},
    }
    config.save()
    init_db(db_path)
    add_provider({
        "id": "test-provider", "name": "TP", "provider_type": "openai",
        "api_base": "https://api.test.com/v1", "api_key": "upstream-key",
        "enabled": True,
        "models": [{"id": "leaky-model", "name": "leaky-model", "enabled": True}],
    })
    add_user({"username": "alice", "display_name": "Alice", "enabled": True})
    add_user_api_key("alice", "default", ["leaky-model"])
    with get_db() as db:
        db.execute("UPDATE user_api_keys SET key = 'user-key' WHERE username = 'alice'")
    yield


def _leaky_output():
    return InternalOutputMessage(
        text=INCIDENT_TEXT, finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
    )


def _request_tools():
    return [{"type": "function", "function": {
        "name": "exec_command",
        "description": "run",
        "parameters": {"type": "object", "properties": {
            "cmd": {"type": "string"}, "max_output_tokens": {"type": "number"}},
            "required": ["cmd"]},
    }}]


def test_chat_completions_nonstream_repair(monkeypatch):
    async def fake_call(policy, internal, **kw):
        return _leaky_output(), None, "test-provider"
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_call)
    r = client.post("/v1/chat/completions", headers=headers, json={
        "model": "leaky-model", "messages": [{"role": "user", "content": "hi"}],
        "tools": _request_tools(),
    })
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    assert msg.get("content") in (None, "")
    calls = msg.get("tool_calls") or []
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "exec_command"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["max_output_tokens"] == 2000
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_messages_nonstream_repair(monkeypatch):
    async def fake_call(policy, internal, **kw):
        return _leaky_output(), None, "test-provider"
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_call)
    r = client.post("/v1/messages", headers=headers, json={
        "model": "test-provider/leaky-model",
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64,
        "tools": [{"name": "exec_command", "description": "run", "input_schema": {
            "type": "object", "properties": {"cmd": {"type": "string"},
                                             "max_output_tokens": {"type": "number"}},
            "required": ["cmd"]}}],
    })
    assert r.status_code == 200
    blocks = r.json()["content"]
    tool_use = [b for b in blocks if b.get("type") == "tool_use"]
    assert len(tool_use) == 1
    assert tool_use[0]["name"] == "exec_command"
    assert tool_use[0]["input"]["max_output_tokens"] == 2000


def test_repair_disabled_by_config(monkeypatch):
    async def fake_call(policy, internal, **kw):
        return _leaky_output(), None, "test-provider"
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_call)
    monkeypatch.setattr("app.router.proxy.get_default", lambda key, fallback=None: False if key == "repair_tool_leaks" else fallback)
    r = client.post("/v1/chat/completions", headers=headers, json={
        "model": "leaky-model", "messages": [{"role": "user", "content": "hi"}],
        "tools": _request_tools(),
    })
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    # 关闭后保持原行为：XML 原样透传
    assert OPEN in (msg.get("content") or "")
    assert not msg.get("tool_calls")


def test_clean_text_untouched(monkeypatch):
    async def fake_call(policy, internal, **kw):
        out = InternalOutputMessage(text="all good", finish_reason="stop")
        return out, None, "test-provider"
    monkeypatch.setattr("app.router.proxy._call_nonstream_with_fallbacks", fake_call)
    r = client.post("/v1/chat/completions", headers=headers, json={
        "model": "leaky-model", "messages": [{"role": "user", "content": "hi"}],
        "tools": _request_tools(),
    })
    assert r.json()["choices"][0]["message"]["content"] == "all good"


@pytest.mark.asyncio
async def test_stream_detect_only(declared_capture):
    from app.core.streaming import stream_internal_output

    async def _events():
        yield InternalOutputEvent(kind="text_delta", text=INCIDENT_TEXT)
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    lines = []
    async for line in stream_internal_output(
        events=_events(), endpoint="chat_completions", model="m", username="legacy",
        api_key_value="k", provider_id="p", requested_model="m",
        log_request=declared_capture, declared_tools=TOOLS,
    ):
        lines.append(line)
    joined = "".join(lines)
    # 行为不变：文本原样流给客户端
    assert "Get-ChildItem" in joined
    # 但 details 里留下检测标记
    assert declared_capture.captured.get("tool_leak_detected") == 1


@pytest.fixture
def declared_capture():
    class Cap:
        captured = {}
        def __call__(self, user, key, model, provider, endpoint, success, tokens, requested, **kwargs):
            self.captured = kwargs.get("details") or {}
    return Cap()
