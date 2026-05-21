"""
消息转换与规范化测试 — Anthropic↔OpenAI 转换、消息合并、
think 标签处理、_sanitize_args 边界、Responses API 转换。
"""
import json
import pytest
from app.router.proxy import (
    _anthropic_to_openai_messages,
    _anthropic_content_to_openai,
    _openai_to_anthropic_content,
    _map_stop_reason,
    _normalize_messages,
    _extract_and_strip_think,
    _strip_think_tags,
    _sanitize_args,
    _fix_tool_args,
    _convert_responses_input,
    _convert_responses_tools,
    _wildcard_match,
    _mask_key,
    _friendly_error_msg,
)

# 足够长的 base64 图片数据（>100 字符）
_IMG1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" * 3
_IMG2 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKM//2Q==" * 3


# ── _anthropic_content_to_openai 测试 ──

def test_anthropic_content_to_openai_string():
    result = _anthropic_content_to_openai("Hello world")
    assert result == ["Hello world"]


def test_anthropic_content_to_openai_text_blocks():
    result = _anthropic_content_to_openai([
        {"type": "text", "text": "Part A"},
        {"type": "text", "text": "Part B"},
    ])
    assert result == ["Part A", "Part B"]


def test_anthropic_content_to_openai_image_block():
    result = _anthropic_content_to_openai([
        {"type": "text", "text": "Look:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}},
    ])
    assert len(result) == 2
    assert result[0] == "Look:"
    assert isinstance(result[1], dict)
    assert result[1]["type"] == "image_url"
    assert "data:image/png;base64," in result[1]["image_url"]["url"]


def test_anthropic_content_to_openai_tool_use_skipped():
    """tool_use 块在 _anthropic_content_to_openai 中应被跳过（由上层处理）。"""
    result = _anthropic_content_to_openai([
        {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
    ])
    assert result == []


def test_anthropic_content_to_openai_non_dict_items():
    result = _anthropic_content_to_openai(["plain", 123, None])
    assert result == ["plain", "123", "None"]


# ── _anthropic_to_openai_messages 测试 ──

def test_anthropic_to_openai_simple():
    msgs, has_tools = _anthropic_to_openai_messages([
        {"role": "user", "content": "Hello"}
    ])
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello"
    assert has_tools is False


def test_anthropic_to_openai_with_system():
    msgs, _ = _anthropic_to_openai_messages(
        [{"role": "user", "content": "Hi"}],
        system_prompt="You are helpful."
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful."


def test_anthropic_to_openai_assistant_with_tool_use():
    msgs, has_tools = _anthropic_to_openai_messages([
        {"role": "user", "content": "Search cats"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me search"},
            {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "cats"}}
        ]},
    ])
    assert has_tools is True
    assert len(msgs) == 2
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Let me search"
    assert len(msgs[1]["tool_calls"]) == 1
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "search"
    assert json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]) == {"q": "cats"}


def test_anthropic_to_openai_tool_result_in_user_message():
    """Anthropic 格式中 tool_result 嵌入在 user 消息内。"""
    msgs, has_tools = _anthropic_to_openai_messages([
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "Found 3 cats"},
            {"type": "text", "text": "What next?"},
        ]},
    ])
    # tool 消息必须先于 user 文本，满足 DeepSeek 等严格供应商的 tool_calls→tool 邻接要求
    assert len(msgs) == 2
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "call_1"
    assert msgs[0]["content"] == "Found 3 cats"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "What next?"


def test_anthropic_to_openai_tool_result_with_image():
    """tool_result 中嵌套图片 — 应转换为 image_url 格式。"""
    msgs, _ = _anthropic_to_openai_messages([
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": [
                {"type": "text", "text": "Screenshot:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}},
            ]},
        ]},
    ])
    assert len(msgs) == 1  # 只有 tool 消息（user 部分为空被跳过）
    assert msgs[0]["role"] == "tool"
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


def test_anthropic_to_openai_tool_result_role():
    """Claude Code 发送的 tool_result 角色消息。"""
    msgs, _ = _anthropic_to_openai_messages([
        {"role": "tool_result", "tool_use_id": "call_1", "content": "Result text"},
    ])
    assert len(msgs) == 1
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "call_1"
    assert msgs[0]["content"] == "Result text"


def test_anthropic_to_openai_tool_result_role_with_image():
    msgs, _ = _anthropic_to_openai_messages([
        {"role": "tool_result", "tool_use_id": "call_1", "content": [
            {"type": "text", "text": "File:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}},
        ]},
    ])
    assert msgs[0]["role"] == "tool"
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


def test_anthropic_to_openai_user_with_image():
    msgs, _ = _anthropic_to_openai_messages([
        {"role": "user", "content": [
            {"type": "text", "text": "Describe:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _IMG2}},
        ]},
    ])
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert "data:image/jpeg;base64," in content[1]["image_url"]["url"]


# ── _openai_to_anthropic_content 测试 ──

def test_openai_to_anthropic_text_only():
    result = _openai_to_anthropic_content({"role": "assistant", "content": "Hello"})
    assert len(result) == 1
    assert result[0] == {"type": "text", "text": "Hello"}


def test_openai_to_anthropic_with_tool_calls():
    result = _openai_to_anthropic_content({
        "role": "assistant",
        "content": "Let me search",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q":"cats"}'}}
        ]
    })
    assert len(result) == 2
    assert result[0] == {"type": "text", "text": "Let me search"}
    assert result[1]["type"] == "tool_use"
    assert result[1]["id"] == "call_1"
    assert result[1]["name"] == "search"
    assert result[1]["input"] == {"q": "cats"}


def test_openai_to_anthropic_empty_content():
    result = _openai_to_anthropic_content({"role": "assistant", "content": ""})
    assert result == [{"type": "text", "text": ""}]


def test_openai_to_anthropic_invalid_json_args():
    """工具调用参数为非法 JSON 时应回退到空 dict。"""
    result = _openai_to_anthropic_content({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "not json"}}
        ]
    })
    assert result[0]["type"] == "tool_use"
    assert result[0]["input"] == {}


# ── _map_stop_reason 测试 ──

def test_map_stop_reason_all():
    assert _map_stop_reason("stop") == "end_turn"
    assert _map_stop_reason("length") == "max_tokens"
    assert _map_stop_reason("tool_calls") == "tool_use"
    assert _map_stop_reason("unknown") == "end_turn"
    assert _map_stop_reason("") == "end_turn"


# ── _normalize_messages 测试 ──

def test_normalize_consecutive_system():
    msgs = [
        {"role": "system", "content": "Rule 1"},
        {"role": "system", "content": "Rule 2"},
    ]
    result = _normalize_messages(msgs)
    assert len(result) == 1
    assert result[0]["content"] == "Rule 1\n\nRule 2"


def test_normalize_consecutive_user():
    msgs = [
        {"role": "user", "content": "Question 1"},
        {"role": "user", "content": "Question 2"},
    ]
    result = _normalize_messages(msgs)
    assert len(result) == 1
    assert result[0]["content"] == "Question 1\n\nQuestion 2"


def test_normalize_no_merge_assistant():
    """assistant 消息不应合并（保留 tool_use/tool_result 连续性）。"""
    msgs = [
        {"role": "assistant", "content": "Reply 1"},
        {"role": "assistant", "content": "Reply 2"},
    ]
    result = _normalize_messages(msgs)
    assert len(result) == 2


def test_normalize_no_merge_tool():
    msgs = [
        {"role": "tool", "content": "Result 1", "tool_call_id": "c1"},
        {"role": "tool", "content": "Result 2", "tool_call_id": "c2"},
    ]
    result = _normalize_messages(msgs)
    assert len(result) == 2


def test_normalize_mixed_content_types():
    """合并时 prev 是字符串、当前是列表的情况。"""
    msgs = [
        {"role": "user", "content": "Text first"},
        {"role": "user", "content": [{"type": "text", "text": "Then list"}]},
    ]
    result = _normalize_messages(msgs)
    assert len(result) == 1
    content = result[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2


def test_normalize_list_then_string():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "List first"}]},
        {"role": "user", "content": "Then string"},
    ]
    result = _normalize_messages(msgs)
    assert len(result) == 1
    content = result[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2


def test_normalize_list_then_list():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "A"}]},
        {"role": "user", "content": [{"type": "text", "text": "B"}]},
    ]
    result = _normalize_messages(msgs)
    assert len(result) == 1
    content = result[0]["content"]
    assert len(content) == 2


def test_normalize_preserves_extra_fields():
    """合并时保留后一条消息的额外字段（如 reasoning_content）。"""
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "user", "content": "Q2", "reasoning_content": "think"},
    ]
    result = _normalize_messages(msgs)
    assert result[0].get("reasoning_content") == "think"


def test_normalize_empty():
    assert _normalize_messages([]) == []


def test_normalize_single():
    msgs = [{"role": "user", "content": "Hi"}]
    result = _normalize_messages(msgs)
    assert result == msgs


def test_normalize_interleaved():
    """交替出现的不同角色不应合并。"""
    msgs = [
        {"role": "system", "content": "S1"},
        {"role": "user", "content": "U1"},
        {"role": "system", "content": "S2"},
        {"role": "user", "content": "U2"},
    ]
    result = _normalize_messages(msgs)
    assert len(result) == 4


# ── _extract_and_strip_think / _strip_think_tags 测试 ──

def test_extract_and_strip_think():
    text = "Before <think>This is thinking</think> After"
    cleaned, thinking = _extract_and_strip_think(text)
    # 正则 r'<think>(.*?)</think>\s*' 会吞掉 </think> 后的空格
    assert cleaned == "Before After"
    assert thinking == "This is thinking"


def test_extract_and_strip_think_multiple():
    text = "<think>First</think> <think>Second</think> Done"
    cleaned, thinking = _extract_and_strip_think(text)
    assert "First" in thinking
    assert "Second" in thinking
    assert "First" not in cleaned
    assert "Second" not in cleaned


def test_extract_and_strip_think_no_tags():
    text = "Just plain text"
    cleaned, thinking = _extract_and_strip_think(text)
    assert cleaned == "Just plain text"
    assert thinking == ""


def test_extract_and_strip_think_empty():
    cleaned, thinking = _extract_and_strip_think("")
    assert cleaned == ""
    assert thinking == ""


def test_extract_and_strip_think_none():
    cleaned, thinking = _extract_and_strip_think(None)
    assert cleaned is None
    assert thinking == ""


def test_strip_think_tags():
    assert _strip_think_tags("Hello <think>ignore</think> World") == "Hello World"
    assert _strip_think_tags("No tags here") == "No tags here"


# ── _sanitize_args 边界测试 ──

def test_sanitize_args_simple():
    assert _sanitize_args('{"url": undefined}') == '{"url": ""}'


def test_sanitize_args_multiple():
    assert _sanitize_args('{"a": undefined, "b": undefined}') == '{"a": "", "b": ""}'


def test_sanitize_args_no_undefined():
    assert _sanitize_args('{"a": 1, "b": "hello"}') == '{"a": 1, "b": "hello"}'


def test_sanitize_args_undefined_in_string_value():
    """字符串值中的 'undefined' 不应被替换。"""
    args = '{"query": "find undefined values"}'
    assert _sanitize_args(args) == args


def test_sanitize_args_undefined_partial_word():
    """'undefined' 作为单词一部分时不应替换。"""
    args = '{"key": "undefined_value"}'
    assert _sanitize_args(args) == args


def test_sanitize_args_undefined_with_escaped_quotes():
    """正确处理转义引号。"""
    args = r'{"desc": "say \"undefined\" please"}'
    assert _sanitize_args(args) == args


def test_sanitize_args_empty():
    assert _sanitize_args("") == ""


# ── _fix_tool_args 测试 ──

def test_fix_tool_args_no_function():
    tc = {"index": 0}
    _fix_tool_args(tc)
    assert tc == {"index": 0}


def test_fix_tool_args_replaces_undefined():
    tc = {"function": {"name": "search", "arguments": '{"q": undefined}'}}
    _fix_tool_args(tc)
    assert tc["function"]["arguments"] == '{"q": ""}'


def test_fix_tool_args_no_undefined():
    tc = {"function": {"name": "search", "arguments": '{"q": "valid"}'}}
    original = dict(tc["function"])
    _fix_tool_args(tc)
    assert tc["function"] == original


# ── _wildcard_match 测试 ──

def test_wildcard_exact():
    assert _wildcard_match("gpt-4", "gpt-4") is True


def test_wildcard_star():
    assert _wildcard_match("gpt-*", "gpt-4-turbo") is True
    assert _wildcard_match("gpt-*", "claude-3") is False


def test_wildcard_multiple_stars():
    assert _wildcard_match("*mini*", "MiniMax-M2.7") is True


def test_wildcard_case_insensitive():
    assert _wildcard_match("GPT*", "gpt-4") is True


# ── _mask_key 测试 ──

def test_mask_key_normal():
    assert _mask_key("sk-aio-abcdefghijklmnopqrstuvwxyz1234567890AB") == "sk-a...90AB"


def test_mask_key_short():
    assert _mask_key("short") == "short"


def test_mask_key_exact_eight():
    assert _mask_key("12345678") == "12345678"


# ── _convert_responses_input 更多边界测试 ──

def test_convert_responses_input_developer_role():
    """developer 角色应转换为 system。"""
    result = _convert_responses_input([
        {"type": "message", "role": "developer", "content": "You are helpful."},
        {"type": "message", "role": "user", "content": "Hi"},
    ])
    assert result[0]["role"] == "system"


def test_convert_responses_input_input_image_attaches_to_user():
    """顶层 input_image 应附加到最近的 user 消息。"""
    result = _convert_responses_input([
        {"type": "message", "role": "user", "content": "Look at this:"},
        {"type": "input_image", "image_url": "https://example.com/img.jpg"},
    ])
    assert len(result) == 1
    content = result[0]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "image_url"


def test_convert_responses_input_input_image_no_user():
    """没有前置 user 消息时，input_image 应创建新 user 消息。"""
    result = _convert_responses_input([
        {"type": "message", "role": "system", "content": "System"},
        {"type": "input_image", "image_url": "https://example.com/img.jpg"},
    ])
    assert result[-1]["role"] == "user"
    assert isinstance(result[-1]["content"], list)


def test_convert_responses_input_reasoning_skipped():
    """reasoning 类型应被跳过。"""
    result = _convert_responses_input([
        {"type": "message", "role": "user", "content": "Hi"},
        {"type": "reasoning", "content": "Thinking..."},
    ])
    assert len(result) == 1


def test_convert_responses_input_string_content():
    """字符串类型的 content 应直接使用。"""
    result = _convert_responses_input([
        {"type": "message", "role": "user", "content": "Plain string"},
    ])
    assert result[0]["content"] == "Plain string"


def test_convert_responses_input_content_with_images():
    """包含 input_image 的 content 列表应保留为列表格式。"""
    result = _convert_responses_input([
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "Describe:"},
            {"type": "input_image", "image_url": "https://example.com/img.jpg"},
        ]},
    ])
    content = result[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2


def test_convert_responses_input_function_call_with_reasoning():
    """function_call 中的 reasoning_content 应被保留。"""
    result = _convert_responses_input([
        {"type": "message", "role": "user", "content": "Search"},
        {"type": "function_call", "call_id": "c1", "name": "search",
         "arguments": '{"q":"x"}', "reasoning_content": "Need to search"},
    ])
    assert result[1].get("reasoning_content") == "Need to search"


def test_convert_responses_input_non_dict_skipped():
    result = _convert_responses_input(["not a dict", 123])
    assert result == []


def test_convert_responses_input_fallback_role():
    """带有 role 字段的非标准消息类型应被处理。"""
    result = _convert_responses_input([
        {"role": "user", "content": "Simple fallback"},
    ])
    assert len(result) == 1
    assert result[0]["role"] == "user"


# ── _convert_responses_tools 更多边界测试 ──

def test_convert_responses_tools_already_formatted():
    """已经是 Chat Completions 格式的工具应直接通过。"""
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    result = _convert_responses_tools(tools)
    assert result == tools


def test_convert_responses_tools_strips_openai_fields():
    """应去除 OpenAI 专有字段（strict, additionalProperties）。"""
    tools = [{
        "type": "function",
        "name": "search",
        "description": "Search",
        "strict": True,
        "additionalProperties": False,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "q": {"type": "string", "additionalProperties": False}
            }
        }
    }]
    result = _convert_responses_tools(tools)
    func = result[0]["function"]
    assert "strict" not in func
    assert "additionalProperties" not in func
    assert "additionalProperties" not in func["parameters"]
    assert "additionalProperties" not in func["parameters"]["properties"]["q"]


def test_convert_responses_tools_filters_non_function():
    result = _convert_responses_tools([
        {"type": "web_search", "name": "web"},
        {"type": "custom", "name": "custom"},
        {"type": "function", "name": "my_func", "parameters": {}},
    ])
    assert len(result) == 1
    assert result[0]["function"]["name"] == "my_func"


def test_convert_responses_tools_non_dict_skipped():
    result = _convert_responses_tools(["not dict", 123, None])
    assert result == []


def test_convert_responses_tools_empty():
    assert _convert_responses_tools([]) == []


# ── _friendly_error_msg ──

def test_friendly_error_msg_content_moderation():
    e = Exception("litellm.APIConnectionError: OpenAIException - output new_sensitive (1027)")
    result = _friendly_error_msg(e)
    assert "内容被上游安全策略拦截（输出端）" in result
    assert "output new_sensitive" in result  # 原始错误保留在消息中


def test_friendly_error_msg_no_image_support():
    e = Exception("No endpoints found that support image input")
    result = _friendly_error_msg(e)
    assert "该模型不支持图像输入" in result


def test_friendly_error_msg_unmapped_passthrough():
    e = Exception("Some unknown error message")
    result = _friendly_error_msg(e)
    assert result == "Some unknown error message"


# ── system 数组格式 → 字符串转换 ──

def test_anthropic_to_openai_system_as_array():
    """Anthropic system 字段为 content block 数组时（含 cache_control），应转为纯字符串。"""
    msgs, has_tools = _anthropic_to_openai_messages(
        [{"role": "user", "content": "Hello"}],
        system_prompt=[
            {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Be concise.", "cache_control": {"type": "ephemeral"}},
        ]
    )
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful.\nBe concise."
    assert isinstance(msgs[0]["content"], str)  # 不是数组


def test_anthropic_to_openai_system_array_with_empty_blocks():
    """system 数组中含空文本块 → 应被过滤。"""
    msgs, _ = _anthropic_to_openai_messages(
        [{"role": "user", "content": "Hi"}],
        system_prompt=[
            {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Valid text.", "cache_control": {"type": "ephemeral"}},
        ]
    )
    assert msgs[0]["content"] == "Valid text."


# ── 空文本块过滤 ──

def test_anthropic_content_to_openai_filters_empty_text():
    """_anthropic_content_to_openai 应过滤 text: "" 的块。"""
    result = _anthropic_content_to_openai([
        {"type": "text", "text": ""},
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": ""},
    ])
    assert result == ["Hello"]


def test_anthropic_to_openai_assistant_empty_text_filtered():
    """assistant 消息含空文本块 → 不应产生 content="" 的消息。"""
    msgs, _ = _anthropic_to_openai_messages(
        [{"role": "assistant", "content": [
            {"type": "text", "text": ""},
            {"type": "text", "text": "I can help."},
        ]}],
        system_prompt=""
    )
    assert msgs[0]["content"] == "I can help."


def test_anthropic_to_openai_assistant_all_text_empty():
    """assistant 消息所有文本块为空 → content 应为 None。"""
    msgs, _ = _anthropic_to_openai_messages(
        [{"role": "assistant", "content": [{"type": "text", "text": ""}]}],
        system_prompt=""
    )
    assert msgs[0]["content"] is None


def test_anthropic_to_openai_image_only_user_no_empty_text():
    """纯图片用户消息（无文本）→ 不应产生空字符串。"""
    msgs, _ = _anthropic_to_openai_messages(
        [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _IMG1}}
        ]}],
        system_prompt=""
    )
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "image_url"
