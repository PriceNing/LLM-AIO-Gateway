"""Tests for proxy message conversion and response helpers."""
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
    _strip_billing_header,
    _conversation_cache_key,
    _inject_reasoning_content,
    _remember_reasoning_content,
    _reasoning_cache,
    _reasoning_tool_cache,
)

# Test section
_IMG1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" * 3
_IMG2 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKM//2Q==" * 3


# Test section

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
    """Test behavior."""
    result = _anthropic_content_to_openai([
        {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
    ])
    assert result == []


def test_anthropic_content_to_openai_non_dict_items():
    result = _anthropic_content_to_openai(["plain", 123, None])
    assert result == ["plain", "123", "None"]


# Test section

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
    """Test behavior."""
    msgs, has_tools = _anthropic_to_openai_messages([
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "Found 3 cats"},
            {"type": "text", "text": "What next?"},
        ]},
    ])
    # Test section
    assert len(msgs) == 2
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "call_1"
    assert msgs[0]["content"] == "Found 3 cats"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "What next?"


def test_anthropic_to_openai_tool_result_with_image():
    """Test behavior."""
    msgs, _ = _anthropic_to_openai_messages([
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": [
                {"type": "text", "text": "Screenshot:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}},
            ]},
        ]},
    ])
    assert len(msgs) == 1
    assert msgs[0]["role"] == "tool"
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


def test_anthropic_to_openai_tool_result_role():
    """Test behavior."""
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


# Test section

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
    """Test behavior."""
    result = _openai_to_anthropic_content({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "not json"}}
        ]
    })
    assert result[0]["type"] == "tool_use"
    assert result[0]["input"] == {}


# Test section

def test_map_stop_reason_all():
    assert _map_stop_reason("stop") == "end_turn"
    assert _map_stop_reason("length") == "max_tokens"
    assert _map_stop_reason("tool_calls") == "tool_use"
    assert _map_stop_reason("unknown") == "end_turn"
    assert _map_stop_reason("") == "end_turn"


# Test section

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
    """Test behavior."""
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
    """Test behavior."""
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
    """Test behavior."""
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
    """Test behavior."""
    msgs = [
        {"role": "system", "content": "S1"},
        {"role": "user", "content": "U1"},
        {"role": "system", "content": "S2"},
        {"role": "user", "content": "U2"},
    ]
    result = _normalize_messages(msgs)
    assert len(result) == 4


# Test section

def test_extract_and_strip_think():
    text = "Before <think>This is thinking</think> After"
    cleaned, thinking = _extract_and_strip_think(text)
    # Test section
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


# Test section

def test_sanitize_args_simple():
    assert _sanitize_args('{"url": undefined}') == '{"url": ""}'


def test_sanitize_args_multiple():
    assert _sanitize_args('{"a": undefined, "b": undefined}') == '{"a": "", "b": ""}'


def test_sanitize_args_no_undefined():
    assert _sanitize_args('{"a": 1, "b": "hello"}') == '{"a": 1, "b": "hello"}'


def test_sanitize_args_undefined_in_string_value():
    """Test behavior."""
    args = '{"query": "find undefined values"}'
    assert _sanitize_args(args) == args


def test_sanitize_args_undefined_partial_word():
    """Test behavior."""
    args = '{"key": "undefined_value"}'
    assert _sanitize_args(args) == args


def test_sanitize_args_undefined_with_escaped_quotes():
    """Test behavior."""
    args = r'{"desc": "say \"undefined\" please"}'
    assert _sanitize_args(args) == args


def test_sanitize_args_empty():
    assert _sanitize_args("") == ""


# Test section

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


# Test section

def test_wildcard_exact():
    assert _wildcard_match("gpt-4", "gpt-4") is True


def test_wildcard_star():
    assert _wildcard_match("gpt-*", "gpt-4-turbo") is True
    assert _wildcard_match("gpt-*", "claude-3") is False


def test_wildcard_multiple_stars():
    assert _wildcard_match("*mini*", "MiniMax-M2.7") is True


def test_wildcard_case_insensitive():
    assert _wildcard_match("GPT*", "gpt-4") is True


# Test section

def test_mask_key_normal():
    assert _mask_key("sk-aio-abcdefghijklmnopqrstuvwxyz1234567890AB") == "sk-a...90AB"


def test_mask_key_short():
    assert _mask_key("short") == "short"


def test_mask_key_exact_eight():
    assert _mask_key("12345678") == "12345678"


# Test section

def test_convert_responses_input_developer_role():
    """Test behavior."""
    result = _convert_responses_input([
        {"type": "message", "role": "developer", "content": "You are helpful."},
        {"type": "message", "role": "user", "content": "Hi"},
    ])
    assert result[0]["role"] == "system"


def test_convert_responses_input_input_image_attaches_to_user():
    """Test behavior."""
    result = _convert_responses_input([
        {"type": "message", "role": "user", "content": "Look at this:"},
        {"type": "input_image", "image_url": "https://example.com/img.jpg"},
    ])
    assert len(result) == 1
    content = result[0]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "image_url"


def test_convert_responses_input_input_image_no_user():
    """Test behavior."""
    result = _convert_responses_input([
        {"type": "message", "role": "system", "content": "System"},
        {"type": "input_image", "image_url": "https://example.com/img.jpg"},
    ])
    assert result[-1]["role"] == "user"
    assert isinstance(result[-1]["content"], list)


def test_convert_responses_input_reasoning_skipped():
    """Test behavior."""
    result = _convert_responses_input([
        {"type": "message", "role": "user", "content": "Hi"},
        {"type": "reasoning", "content": "Thinking..."},
    ])
    assert len(result) == 1


def test_convert_responses_input_reasoning_item_backfills_assistant():
    """Test behavior."""
    result = _convert_responses_input([
        {"type": "message", "role": "user", "content": "Search"},
        {"type": "function_call", "call_id": "call_1", "name": "search", "arguments": '{"q":"x"}'},
        {"type": "reasoning", "content": "Need to search"},
        {"type": "function_call_output", "call_id": "call_1", "output": "Found"},
    ])
    assert result[1].get("reasoning_content") == "Need to search"


def test_convert_responses_input_string_content():
    """Test behavior."""
    result = _convert_responses_input([
        {"type": "message", "role": "user", "content": "Plain string"},
    ])
    assert result[0]["content"] == "Plain string"


def test_convert_responses_input_content_with_images():
    """Test behavior."""
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
    """Test behavior."""
    result = _convert_responses_input([
        {"type": "message", "role": "user", "content": "Search"},
        {"type": "function_call", "call_id": "c1", "name": "search",
         "arguments": '{"q":"x"}', "reasoning_content": "Need to search"},
    ])
    assert result[1].get("reasoning_content") == "Need to search"


def test_convert_responses_input_merges_assistant_text_before_function_call():
    """Test behavior."""
    result = _convert_responses_input([
        {"type": "message", "role": "user", "content": "Search"},
        {"type": "message", "role": "assistant", "content": "Let me check."},
        {"type": "function_call", "call_id": "c1", "name": "search", "arguments": '{"q":"x"}'},
    ])
    assert len(result) == 2
    assert result[1]["role"] == "assistant"
    assert result[1]["tool_calls"][0]["id"] == "c1"
    assert result[1]["content"] is None
    assert result[1]["reasoning_content"] == "Let me check."


def test_convert_responses_input_function_call_output_with_reasoning_backfills_assistant():
    """Test behavior."""
    result = _convert_responses_input([
        {"type": "message", "role": "user", "content": "Search"},
        {"type": "function_call", "call_id": "c1", "name": "search", "arguments": '{"q":"x"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "Found", "reasoning_content": "Need to search"},
    ])
    assert result[1].get("reasoning_content") == "Need to search"
    assert result[2]["role"] == "tool"


def test_convert_responses_input_non_dict_skipped():
    result = _convert_responses_input(["not a dict", 123])
    assert result == []


def test_convert_responses_input_fallback_role():
    """Test behavior."""
    result = _convert_responses_input([
        {"role": "user", "content": "Simple fallback"},
    ])
    assert len(result) == 1
    assert result[0]["role"] == "user"


# Test section

def test_convert_responses_tools_already_formatted():
    """Test behavior."""
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    result = _convert_responses_tools(tools)
    assert result == tools


def test_convert_responses_tools_strips_openai_fields():
    """Test behavior."""
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


def test_anthropic_tool_block_indexes_unique():
    from app.router.proxy import _anthropic_tool_block_index

    tool_uses = {}
    assert _anthropic_tool_block_index(False, tool_uses) == 0
    tool_uses[0] = {"block_index": 0}
    assert _anthropic_tool_block_index(False, tool_uses) == 1
    assert _anthropic_tool_block_index(True, tool_uses) == 2


# -- _friendly_error_msg --

def test_friendly_error_msg_content_moderation():
    e = Exception("litellm.APIConnectionError: OpenAIException - output new_sensitive (1027)")
    result = _friendly_error_msg(e)
    assert "" in result
    assert "output new_sensitive" in result


def test_friendly_error_msg_no_image_support():
    e = Exception("No endpoints found that support image input")
    result = _friendly_error_msg(e)
    assert "" in result


def test_friendly_error_msg_unmapped_passthrough():
    e = Exception("Some unknown error message")
    result = _friendly_error_msg(e)
    assert result == "Some unknown error message"


# Test section

def test_anthropic_to_openai_system_as_array():
    """Test behavior."""
    msgs, has_tools = _anthropic_to_openai_messages(
        [{"role": "user", "content": "Hello"}],
        system_prompt=[
            {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Be concise.", "cache_control": {"type": "ephemeral"}},
        ]
    )
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful.\nBe concise."
    assert isinstance(msgs[0]["content"], str)


def test_anthropic_to_openai_system_array_with_empty_blocks():
    """Test behavior."""
    msgs, _ = _anthropic_to_openai_messages(
        [{"role": "user", "content": "Hi"}],
        system_prompt=[
            {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Valid text.", "cache_control": {"type": "ephemeral"}},
        ]
    )
    assert msgs[0]["content"] == "Valid text."


def test_anthropic_to_openai_system_array_strips_billing_header():
    msgs, _ = _anthropic_to_openai_messages(
        [{"role": "user", "content": "Hi"}],
        system_prompt=_strip_billing_header([
            {
                "type": "text",
                "text": "Prompt\nx-anthropic-billing-header: cc_version=2.1.37; cc_entrypoint=cli; cch=random;",
                "cache_control": {"type": "ephemeral"},
            },
        ])
    )
    assert msgs[0]["content"] == "Prompt"


def test_strip_billing_header_preserves_cache_control():
    """Test behavior."""
    result = _strip_billing_header([
        {
            "type": "text",
            "text": "Prompt\nx-anthropic-billing-header: cc_version=2.1.37; cc_entrypoint=cli; cch=random;",
            "cache_control": {"type": "ephemeral"},
        },
    ])
    assert result == [
        {
            "type": "text",
            "text": "Prompt",
            "cache_control": {"type": "ephemeral"},
        },
    ]


def test_strip_billing_header_removes_any_header_line_shape():
    result = _strip_billing_header(
        "Before\n"
        "x-anthropic-billing-header: cch=random; cc_entrypoint=cli; cc_version=2.2.0; extra=value;\n"
        "After"
    )
    assert result == "Before\nAfter"


def test_normalize_system_list_strips_billing_header_preserves_cache_control():
    result = _normalize_messages([
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Prompt\nx-anthropic-billing-header: cc_version=2.1.37; cc_entrypoint=cli; cch=random;",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
        {"role": "user", "content": "Hi"},
    ])
    assert result[0]["content"] == [
        {
            "type": "text",
            "text": "Prompt",
            "cache_control": {"type": "ephemeral"},
        },
    ]


# Test section

def test_reasoning_injection_uses_tool_id_and_active_segment_only():
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "old_call", "type": "function", "function": {"name": "old", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "old_call", "content": "old result"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "new_call", "type": "function", "function": {"name": "new", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "new_call", "content": "new result"},
    ]
    conv_key = _conversation_cache_key("key", messages)
    _reasoning_cache.drop(conv_key)
    _reasoning_tool_cache.drop(conv_key)

    _remember_reasoning_content(conv_key, "old reasoning", ["old_call"])
    _remember_reasoning_content(conv_key, "new reasoning", ["new_call"])

    assert _inject_reasoning_content(messages, conv_key, "test") == 1
    assert "reasoning_content" not in messages[1]
    assert messages[5]["reasoning_content"] == "new reasoning"


def test_reasoning_injection_fallback_limited_to_active_tool_segment():
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "old_call", "type": "function", "function": {"name": "old", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "old_call", "content": "old result"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "new_call_1", "type": "function", "function": {"name": "new1", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "new_call_1", "content": "new result 1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "new_call_2", "type": "function", "function": {"name": "new2", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "new_call_2", "content": "new result 2"},
    ]
    conv_key = _conversation_cache_key("key2", messages)
    _reasoning_cache.drop(conv_key)
    _reasoning_tool_cache.drop(conv_key)
    _reasoning_cache[conv_key] = "latest reasoning"

    assert _inject_reasoning_content(messages, conv_key, "test") == 2
    assert "reasoning_content" not in messages[1]
    assert messages[5]["reasoning_content"] == "latest reasoning"
    assert messages[7]["reasoning_content"] == "latest reasoning"


def test_reasoning_injection_does_not_add_empty_fields_by_default():
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]
    conv_key = _conversation_cache_key("key3", messages)
    _reasoning_cache.drop(conv_key)
    _reasoning_tool_cache.drop(conv_key)

    assert _inject_reasoning_content(messages, conv_key, "test") == 0
    assert "reasoning_content" not in messages[1]


def test_reasoning_cache_survives_image_preprocess_flow():
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": [
            {"type": "text", "text": "Describe:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        ]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "tool2", "arguments": "{}"}}]},
    ]
    conv_key = _conversation_cache_key("key4", messages)
    _reasoning_cache.drop(conv_key)
    _reasoning_tool_cache.drop(conv_key)
    _remember_reasoning_content(conv_key, "cached reasoning", ["call_2"])

    assert _inject_reasoning_content(messages, conv_key, "test") == 1
    assert messages[5]["reasoning_content"] == "cached reasoning"


def test_anthropic_content_to_openai_filters_empty_text():
    """Test behavior."""
    result = _anthropic_content_to_openai([
        {"type": "text", "text": ""},
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": ""},
    ])
    assert result == ["Hello"]


def test_anthropic_to_openai_assistant_empty_text_filtered():
    """Test behavior."""
    msgs, _ = _anthropic_to_openai_messages(
        [{"role": "assistant", "content": [
            {"type": "text", "text": ""},
            {"type": "text", "text": "I can help."},
        ]}],
        system_prompt=""
    )
    assert msgs[0]["content"] == "I can help."


def test_anthropic_to_openai_assistant_all_text_empty():
    """Test behavior."""
    msgs, _ = _anthropic_to_openai_messages(
        [{"role": "assistant", "content": [{"type": "text", "text": ""}]}],
        system_prompt=""
    )
    assert msgs[0]["content"] is None


def test_anthropic_to_openai_image_only_user_no_empty_text():
    """Test behavior."""
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
