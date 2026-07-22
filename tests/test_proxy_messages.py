"""Tests for proxy message conversion and response helpers."""
import json
import pytest
from app.core.policy import wildcard_match as _wildcard_match
from app.core.policy import normalize_messages
from app.core.policy import inject_reasoning_content
from app.core.text import friendly_error_msg as _friendly_error_msg
from app.core.text import mask_key as _mask_key
from app.core.text import strip_billing_header as _strip_billing_header
from app.core.text import message_text as _message_text
from app.core.text import attr as _attr
from app.core.think import extract_and_strip_think as _extract_and_strip_think
from app.core.think import strip_think_tags as _strip_think_tags
from app.core.tool_args import fix_tool_args as _fix_tool_args
from app.core.tool_args import sanitize_args as _sanitize_args
from app.core.types import InternalRequest
from app.core.state import (
    conversation_cache_key as _conversation_cache_key,
    reasoning_cache as _reasoning_cache,
    reasoning_context as _reasoning_context,
    reasoning_tool_cache as _reasoning_tool_cache,
    reasoning_tool_global_cache as _reasoning_tool_global_cache,
    remember_reasoning_content as _remember_reasoning_content,
    remember_response_chain_key as _remember_response_chain_key,
)
from app.protocols.ingress import responses_tools_to_chat_tools
from app.protocols.ir import anthropic_messages_to_ir, ir_to_anthropic_messages, ir_to_openai_messages, openai_messages_to_ir, responses_input_to_ir
from app.adapters.openai import chat_messages_from_internal


def _normalize_internal_messages_for_test(messages):
    if not messages:
        return []
    request = InternalRequest(
        endpoint="chat_completions",
        requested_model="test",
        target_model="test",
        messages=openai_messages_to_ir(messages),
    )
    normalize_messages(request)
    return chat_messages_from_internal(request)

# Test section
_IMG1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" * 3
_IMG2 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKM//2Q==" * 3


# Test section

def test_anthropic_content_to_ir_string():
    result = anthropic_messages_to_ir([{"role": "user", "content": "Hello world"}])
    assert result[0].parts[0].kind == "text"
    assert result[0].parts[0].text == "Hello world"


def test_anthropic_content_to_ir_text_blocks():
    result = anthropic_messages_to_ir([{"role": "user", "content": [
        {"type": "text", "text": "Part A"},
        {"type": "text", "text": "Part B"},
    ]}])
    assert [part.text for part in result[0].parts] == ["Part A", "Part B"]


def test_anthropic_content_to_ir_image_block():
    result = anthropic_messages_to_ir([{"role": "user", "content": [
        {"type": "text", "text": "Look:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}},
    ]}])
    assert len(result[0].parts) == 2
    assert result[0].parts[0].text == "Look:"
    assert result[0].parts[1].kind == "image"
    projected = ir_to_openai_messages(result)
    assert "data:image/png;base64," in projected[0]["content"][1]["image_url"]["url"]


def test_anthropic_content_to_ir_tool_use_preserved():
    result = anthropic_messages_to_ir([{"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
    ]}])
    assert result[0].parts[0].kind == "tool_call"
    assert result[0].parts[0].tool_call_id == "t1"


def test_anthropic_content_to_ir_non_dict_items():
    result = anthropic_messages_to_ir([{"role": "user", "content": ["plain", 123, None]}])
    assert [part.kind for part in result[0].parts] == ["unknown", "unknown", "unknown"]


# Test section

def test_ir_projects_anthropic_simple_message_to_openai_shape():
    ir = anthropic_messages_to_ir([
        {"role": "user", "content": "Hello"}
    ])
    msgs = ir_to_openai_messages(ir)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello"


def test_ir_projects_anthropic_system_to_openai_system_shape():
    msgs = ir_to_openai_messages(anthropic_messages_to_ir([{"role": "user", "content": "Hi"}], "You are helpful."))
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful."


def test_ir_projects_anthropic_tool_use_to_openai_tool_call_shape():
    ir = anthropic_messages_to_ir([
        {"role": "user", "content": "Search cats"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me search"},
            {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "cats"}}
        ]},
    ])
    msgs = ir_to_openai_messages(ir)
    assert any(part.kind == "tool_call" for message in ir for part in message.parts)
    assert len(msgs) == 2
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Let me search"
    assert len(msgs[1]["tool_calls"]) == 1
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "search"
    assert json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]) == {"q": "cats"}


def test_ir_projects_anthropic_tool_result_to_openai_tool_message_shape():
    msgs = ir_to_openai_messages(anthropic_messages_to_ir([
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "Found 3 cats"},
            {"type": "text", "text": "What next?"},
        ]},
    ]))
    assert len(msgs) == 2
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "call_1"
    assert msgs[0]["content"] == "Found 3 cats"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "What next?"


def test_ir_projects_anthropic_tool_result_image_to_openai_tool_message_shape():
    msgs = ir_to_openai_messages(anthropic_messages_to_ir([
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": [
                {"type": "text", "text": "Screenshot:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}},
            ]},
        ]},
    ]))
    assert len(msgs) == 1
    assert msgs[0]["role"] == "tool"
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


def test_ir_projects_anthropic_tool_result_role_to_openai_tool_role():
    msgs = ir_to_openai_messages(anthropic_messages_to_ir([{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "Result text"},
    ]}]))
    assert len(msgs) == 1
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "call_1"
    assert msgs[0]["content"] == "Result text"


def test_ir_projects_anthropic_tool_result_role_with_image_to_openai_shape():
    msgs = ir_to_openai_messages(anthropic_messages_to_ir([{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_1", "content": [
            {"type": "text", "text": "File:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}},
        ]},
    ]}]))
    assert msgs[0]["role"] == "tool"
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


def test_ir_projects_anthropic_user_image_to_openai_image_shape():
    msgs = ir_to_openai_messages(anthropic_messages_to_ir([
        {"role": "user", "content": [
            {"type": "text", "text": "Describe:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _IMG2}},
        ]},
    ]))
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert "data:image/jpeg;base64," in content[1]["image_url"]["url"]


def test_ir_projects_anthropic_tool_result_image_back_to_anthropic_shape():
    messages, system = ir_to_anthropic_messages(anthropic_messages_to_ir([{
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": [
                {"type": "text", "text": "Screenshot:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _IMG1}},
            ],
        }],
    }]))

    assert system == ""
    tool_result = messages[0]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_1"
    assert isinstance(tool_result["content"], list)
    assert tool_result["content"][0] == {"type": "text", "text": "Screenshot:"}
    assert tool_result["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _IMG1},
    }


def test_ir_projects_openai_tool_result_image_back_to_anthropic_shape():
    messages, system = ir_to_anthropic_messages(openai_messages_to_ir([{
        "role": "tool",
        "tool_call_id": "call_1",
        "content": [
            {"type": "text", "text": "Screenshot:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_IMG1}"}},
        ],
    }]))

    assert system == ""
    tool_result = messages[0]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_1"
    assert tool_result["content"][0] == {"type": "text", "text": "Screenshot:"}
    assert tool_result["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _IMG1},
    }


# Test section

def test_openai_to_anthropic_text_only():
    messages, system = ir_to_anthropic_messages(openai_messages_to_ir([{"role": "assistant", "content": "Hello"}]))
    result = messages[0]["content"]
    assert system == ""
    assert result == [{"type": "text", "text": "Hello"}]


def test_openai_to_anthropic_with_tool_calls():
    messages, system = ir_to_anthropic_messages(openai_messages_to_ir([{
        "role": "assistant",
        "content": "Let me search",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q":"cats"}'}}
        ]
    }]))
    result = messages[0]["content"]
    assert system == ""
    assert len(result) == 2
    assert result[0] == {"type": "text", "text": "Let me search"}
    assert result[1]["type"] == "tool_use"
    assert result[1]["id"] == "call_1"
    assert result[1]["name"] == "search"
    assert result[1]["input"] == {"q": "cats"}


def test_openai_to_anthropic_empty_content():
    messages, system = ir_to_anthropic_messages(openai_messages_to_ir([{"role": "assistant", "content": ""}]))
    result = messages[0]["content"]
    assert system == ""
    assert result == [{"type": "text", "text": ""}]


def test_openai_to_anthropic_invalid_json_args():
    """Test behavior."""
    messages, system = ir_to_anthropic_messages(openai_messages_to_ir([{
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "not json"}}
        ]
    }]))
    result = messages[0]["content"]
    assert system == ""
    assert result[0]["type"] == "tool_use"
    assert result[0]["input"] == {}


# Test section

def test_map_stop_reason_all():
    from app.protocols.egress import _anthropic_stop_reason

    assert _anthropic_stop_reason("stop") == "end_turn"
    assert _anthropic_stop_reason("length") == "max_tokens"
    assert _anthropic_stop_reason("tool_calls") == "tool_use"
    assert _anthropic_stop_reason("unknown") == "end_turn"
    assert _anthropic_stop_reason("") == "end_turn"


# Test section

def test_normalize_consecutive_system():
    msgs = [
        {"role": "system", "content": "Rule 1"},
        {"role": "system", "content": "Rule 2"},
    ]
    result = _normalize_internal_messages_for_test(msgs)
    assert len(result) == 1
    assert result[0]["content"] == "Rule 1\n\nRule 2"


def test_normalize_consecutive_user():
    msgs = [
        {"role": "user", "content": "Question 1"},
        {"role": "user", "content": "Question 2"},
    ]
    result = _normalize_internal_messages_for_test(msgs)
    assert len(result) == 1
    assert result[0]["content"] == "Question 1\n\nQuestion 2"


def test_normalize_no_merge_assistant():
    """Test behavior."""
    msgs = [
        {"role": "assistant", "content": "Reply 1"},
        {"role": "assistant", "content": "Reply 2"},
    ]
    result = _normalize_internal_messages_for_test(msgs)
    assert len(result) == 2


def test_normalize_no_merge_tool():
    msgs = [
        {"role": "tool", "content": "Result 1", "tool_call_id": "c1"},
        {"role": "tool", "content": "Result 2", "tool_call_id": "c2"},
    ]
    result = _normalize_internal_messages_for_test(msgs)
    assert len(result) == 2


def test_normalize_mixed_content_types():
    """Test behavior."""
    msgs = [
        {"role": "user", "content": "Text first"},
        {"role": "user", "content": [{"type": "text", "text": "Then list"}]},
    ]
    result = _normalize_internal_messages_for_test(msgs)
    assert len(result) == 1
    assert result[0]["content"] == "Text first\n\nThen list"


def test_normalize_list_then_string():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "List first"}]},
        {"role": "user", "content": "Then string"},
    ]
    result = _normalize_internal_messages_for_test(msgs)
    assert len(result) == 1
    assert result[0]["content"] == "List first\n\nThen string"


def test_normalize_list_then_list():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "A"}]},
        {"role": "user", "content": [{"type": "text", "text": "B"}]},
    ]
    result = _normalize_internal_messages_for_test(msgs)
    assert len(result) == 1
    assert result[0]["content"] == "A\n\nB"


def test_normalize_preserves_extra_fields():
    """Test behavior."""
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "user", "content": "Q2", "reasoning_content": "think"},
    ]
    result = _normalize_internal_messages_for_test(msgs)
    assert result[0].get("reasoning_content") == "think"


def test_normalize_empty():
    assert _normalize_internal_messages_for_test([]) == []


def test_normalize_single():
    msgs = [{"role": "user", "content": "Hi"}]
    result = _normalize_internal_messages_for_test(msgs)
    assert result == msgs


def test_normalize_interleaved():
    """Test behavior."""
    msgs = [
        {"role": "system", "content": "S1"},
        {"role": "user", "content": "U1"},
        {"role": "system", "content": "S2"},
        {"role": "user", "content": "U2"},
    ]
    result = _normalize_internal_messages_for_test(msgs)
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

def test_responses_input_ir_projects_developer_role_to_openai_system():
    """Test behavior."""
    result = ir_to_openai_messages(responses_input_to_ir([
        {"type": "message", "role": "developer", "content": "You are helpful."},
        {"type": "message", "role": "user", "content": "Hi"},
    ]))
    assert result[0]["role"] == "system"


def test_responses_input_ir_attaches_input_image_to_user_message():
    """Test behavior."""
    result = ir_to_openai_messages(responses_input_to_ir([
        {"type": "message", "role": "user", "content": "Look at this:"},
        {"type": "input_image", "image_url": "https://example.com/img.jpg"},
    ]))
    assert len(result) == 1
    content = result[0]["content"]
    assert isinstance(content, list)
    assert content[1]["type"] == "image_url"


def test_responses_input_ir_creates_user_message_for_orphan_input_image():
    """Test behavior."""
    result = ir_to_openai_messages(responses_input_to_ir([
        {"type": "message", "role": "system", "content": "System"},
        {"type": "input_image", "image_url": "https://example.com/img.jpg"},
    ]))
    assert result[-1]["role"] == "user"
    assert isinstance(result[-1]["content"], list)


def test_responses_input_ir_keeps_standalone_reasoning_out_of_openai_projection():
    """Test behavior."""
    result = ir_to_openai_messages(responses_input_to_ir([
        {"type": "message", "role": "user", "content": "Hi"},
        {"type": "reasoning", "content": "Thinking..."},
    ]))
    assert len(result) == 1


def test_responses_input_ir_backfills_reasoning_item_onto_assistant_tool_call():
    """Test behavior."""
    result = ir_to_openai_messages(responses_input_to_ir([
        {"type": "message", "role": "user", "content": "Search"},
        {"type": "function_call", "call_id": "call_1", "name": "search", "arguments": '{"q":"x"}'},
        {"type": "reasoning", "content": "Need to search"},
        {"type": "function_call_output", "call_id": "call_1", "output": "Found"},
    ]))
    assert result[1].get("reasoning_content") == "Need to search"


def test_responses_input_ir_projects_string_content_to_openai_shape():
    """Test behavior."""
    result = ir_to_openai_messages(responses_input_to_ir([
        {"type": "message", "role": "user", "content": "Plain string"},
    ]))
    assert result[0]["content"] == "Plain string"


def test_responses_input_ir_projects_image_content_to_openai_shape():
    """Test behavior."""
    result = ir_to_openai_messages(responses_input_to_ir([
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "Describe:"},
            {"type": "input_image", "image_url": "https://example.com/img.jpg"},
        ]},
    ]))
    content = result[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2


def test_responses_input_ir_preserves_function_call_reasoning():
    """Test behavior."""
    result = ir_to_openai_messages(responses_input_to_ir([
        {"type": "message", "role": "user", "content": "Search"},
        {"type": "function_call", "call_id": "c1", "name": "search",
         "arguments": '{"q":"x"}', "reasoning_content": "Need to search"},
    ]))
    assert result[1].get("reasoning_content") == "Need to search"


def test_responses_input_ir_treats_assistant_text_before_function_call_as_reasoning():
    """Test behavior."""
    result = ir_to_openai_messages(responses_input_to_ir([
        {"type": "message", "role": "user", "content": "Search"},
        {"type": "message", "role": "assistant", "content": "Let me check."},
        {"type": "function_call", "call_id": "c1", "name": "search", "arguments": '{"q":"x"}'},
    ]))
    assert len(result) == 2
    assert result[1]["role"] == "assistant"
    assert result[1]["tool_calls"][0]["id"] == "c1"
    assert result[1]["content"] is None
    assert result[1]["reasoning_content"] == "Let me check."


def test_responses_input_ir_backfills_function_call_output_reasoning_to_assistant():
    """Test behavior."""
    result = ir_to_openai_messages(responses_input_to_ir([
        {"type": "message", "role": "user", "content": "Search"},
        {"type": "function_call", "call_id": "c1", "name": "search", "arguments": '{"q":"x"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "Found", "reasoning_content": "Need to search"},
    ]))
    assert result[1].get("reasoning_content") == "Need to search"
    assert result[2]["role"] == "tool"


def test_responses_input_ir_skips_non_dict_items():
    result = ir_to_openai_messages(responses_input_to_ir(["not a dict", 123]))
    assert result == []


def test_responses_input_ir_accepts_fallback_message_shape():
    """Test behavior."""
    result = ir_to_openai_messages(responses_input_to_ir([
        {"role": "user", "content": "Simple fallback"},
    ]))
    assert len(result) == 1
    assert result[0]["role"] == "user"


# Test section

def test_responses_tools_preserve_already_openai_formatted_tools():
    """Test behavior."""
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    result = responses_tools_to_chat_tools(tools)
    assert result == tools


def test_responses_tools_strip_openai_specific_schema_fields():
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
    result = responses_tools_to_chat_tools(tools)
    func = result[0]["function"]
    assert "strict" not in func
    assert "additionalProperties" not in func
    assert "additionalProperties" not in func["parameters"]
    assert "additionalProperties" not in func["parameters"]["properties"]["q"]


def test_responses_tools_filter_non_function_tools():
    result = responses_tools_to_chat_tools([
        {"type": "web_search", "name": "web"},
        {"type": "custom", "name": "custom"},
        {"type": "function", "name": "my_func", "parameters": {}},
    ])
    assert len(result) == 2
    assert result[0]["function"]["name"] == "custom"
    assert result[0]["function"]["parameters"]["required"] == ["input"]
    assert result[1]["function"]["name"] == "my_func"


def test_responses_tools_skip_non_dict_items():
    result = responses_tools_to_chat_tools(["not dict", 123, None])
    assert result == []


def test_responses_tools_empty_list():
    assert responses_tools_to_chat_tools([]) == []


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


def test_friendly_error_msg_unmapped_is_preserved():
    e = Exception("Some unknown error message")
    result = _friendly_error_msg(e)
    assert result == "Some unknown error message"


# Test section

def test_ir_projects_anthropic_system_array_to_openai_system_text():
    """Test behavior."""
    msgs = ir_to_openai_messages(anthropic_messages_to_ir(
        [{"role": "user", "content": "Hello"}],
        [
            {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Be concise.", "cache_control": {"type": "ephemeral"}},
        ]
    ))
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful.\nBe concise."
    assert isinstance(msgs[0]["content"], str)


def test_ir_projects_anthropic_system_array_filters_empty_blocks():
    """Test behavior."""
    msgs = ir_to_openai_messages(anthropic_messages_to_ir(
        [{"role": "user", "content": "Hi"}],
        [
            {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Valid text.", "cache_control": {"type": "ephemeral"}},
        ]
    ))
    assert msgs[0]["content"] == "Valid text."


def test_ir_projects_anthropic_system_array_after_billing_header_strip():
    msgs = ir_to_openai_messages(anthropic_messages_to_ir(
        [{"role": "user", "content": "Hi"}],
        _strip_billing_header([
            {
                "type": "text",
                "text": "Prompt\nx-anthropic-billing-header: cc_version=2.1.37; cc_entrypoint=cli; cch=random;",
                "cache_control": {"type": "ephemeral"},
            },
        ])
    ))
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
    request = InternalRequest(
        endpoint="messages",
        requested_model="test",
        target_model="test",
        messages=anthropic_messages_to_ir(
            [{"role": "user", "content": "Hi"}],
            [
                {
                    "type": "text",
                    "text": "Prompt\nx-anthropic-billing-header: cc_version=2.1.37; cc_entrypoint=cli; cch=random;",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        ),
    )
    normalize_messages(request)
    system_part = request.messages[0].parts[0]
    assert system_part.text == "Prompt"
    assert system_part.extensions["cache_control"] == {"type": "ephemeral"}
    assert chat_messages_from_internal(request)[0]["content"] == "Prompt"


def test_normalize_openai_system_list_strips_billing_header():
    result = _normalize_internal_messages_for_test([
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
    assert result[0]["content"] == "Prompt"


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

    messages = openai_messages_to_ir(messages)
    assert inject_reasoning_content(messages, _reasoning_cache.get(conv_key), _reasoning_tool_cache.get(conv_key, {})) == 1
    projected = ir_to_openai_messages(messages)
    assert "reasoning_content" not in projected[1]
    assert projected[5]["reasoning_content"] == "new reasoning"


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

    messages = openai_messages_to_ir(messages)
    assert inject_reasoning_content(messages, _reasoning_cache.get(conv_key), _reasoning_tool_cache.get(conv_key, {})) == 2
    projected = ir_to_openai_messages(messages)
    assert "reasoning_content" not in projected[1]
    assert projected[5]["reasoning_content"] == "latest reasoning"
    assert projected[7]["reasoning_content"] == "latest reasoning"


def test_reasoning_injection_does_not_add_empty_fields_by_default():
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]
    conv_key = _conversation_cache_key("key3", messages)
    _reasoning_cache.drop(conv_key)
    _reasoning_tool_cache.drop(conv_key)

    messages = openai_messages_to_ir(messages)
    assert inject_reasoning_content(messages, _reasoning_cache.get(conv_key), _reasoning_tool_cache.get(conv_key, {})) == 0
    projected = ir_to_openai_messages(messages)
    assert "reasoning_content" not in projected[1]


def test_reasoning_injection_stops_when_client_already_provides_reasoning():
    messages = openai_messages_to_ir([
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "tool", "arguments": "{}"}}], "reasoning_content": "client reasoning"},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": "result"},
    ])

    assert inject_reasoning_content(messages, "cached reasoning", {"call_2": "cached reasoning"}) == 1
    projected = ir_to_openai_messages(messages)
    assert projected[1]["reasoning_content"] == "client reasoning"
    assert projected[3]["reasoning_content"] == "cached reasoning"


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

    messages = openai_messages_to_ir(messages)
    assert inject_reasoning_content(messages, _reasoning_cache.get(conv_key), _reasoning_tool_cache.get(conv_key, {})) == 1
    projected = ir_to_openai_messages(messages)
    assert projected[5]["reasoning_content"] == "cached reasoning"


def test_conversation_cache_key_prefers_response_chain_id():
    messages_a = [{"role": "user", "content": "Same first message"}]
    messages_b = [{"role": "user", "content": "Same first message"}]
    key_a = _conversation_cache_key("api", messages_a)
    key_b = _conversation_cache_key("api", messages_b)
    assert key_a == key_b

    _remember_response_chain_key("resp_chain_1", "stable-conv-key")
    key_c = _conversation_cache_key("api", [{"role": "user", "content": "Different"}], "resp_chain_1")
    assert key_c == "stable-conv-key"


def test_conversation_cache_key_uses_followup_user_messages():
    first_turn = [
        {"role": "user", "content": "Do the task"},
    ]
    followup_a = [
        {"role": "user", "content": "Do the task"},
        {"role": "assistant", "content": "Done"},
        {"role": "user", "content": "Other docs?"},
    ]
    followup_b = [
        {"role": "user", "content": "Do the task"},
        {"role": "assistant", "content": "Done"},
        {"role": "user", "content": "Run tests"},
    ]

    assert _conversation_cache_key("api", first_turn) != _conversation_cache_key("api", followup_a)
    assert _conversation_cache_key("api", followup_a) != _conversation_cache_key("api", followup_b)


def test_conversation_cache_key_ignores_internal_tool_results_for_reasoning_replay():
    before_tool_result = openai_messages_to_ir([
        {"role": "user", "content": "Diagnose my computer"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
    ])
    after_tool_result = openai_messages_to_ir([
        {"role": "user", "content": "Diagnose my computer"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "CPU and memory results"},
    ])

    assert _conversation_cache_key("api", before_tool_result) == _conversation_cache_key("api", after_tool_result)


def test_reasoning_context_falls_back_to_tool_id_when_conversation_key_drifts():
    producing_messages = openai_messages_to_ir([
        {"role": "user", "content": "Diagnose my computer"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
    ])
    replay_messages = openai_messages_to_ir([
        {"role": "user", "content": "Diagnose my computer"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "partial result"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": "more result"},
    ])
    old_key = _conversation_cache_key("api-drift", producing_messages)
    new_key = _conversation_cache_key("api-drift", replay_messages)
    _reasoning_cache.drop(old_key)
    _reasoning_cache.drop(new_key)
    _reasoning_tool_cache.drop(old_key)
    _reasoning_tool_cache.drop(new_key)
    _reasoning_tool_global_cache.drop("call_1")
    _reasoning_tool_global_cache.drop("call_2")

    _remember_reasoning_content(old_key, "reasoning for call 1", ["call_1"])
    cached_rc, tool_map = _reasoning_context(new_key, replay_messages)

    assert cached_rc is None
    assert tool_map == {"call_1": "reasoning for call 1"}
    assert inject_reasoning_content(replay_messages, cached_rc, tool_map) == 1
    projected = ir_to_openai_messages(replay_messages)
    assert projected[1]["reasoning_content"] == "reasoning for call 1"
    assert "reasoning_content" not in projected[5]


def test_anthropic_content_to_ir_filters_empty_text():
    result = anthropic_messages_to_ir([{"role": "user", "content": [
        {"type": "text", "text": ""},
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": ""},
    ]}])
    assert [part.text for part in result[0].parts] == ["Hello"]


def test_ir_projects_anthropic_assistant_filters_empty_text():
    msgs = ir_to_openai_messages(anthropic_messages_to_ir([{"role": "assistant", "content": [
            {"type": "text", "text": ""},
            {"type": "text", "text": "I can help."},
        ]}]))
    assert msgs[0]["content"] == "I can help."


def test_ir_projects_anthropic_assistant_all_empty_text_as_none():
    msgs = ir_to_openai_messages(anthropic_messages_to_ir([{"role": "assistant", "content": [{"type": "text", "text": ""}]}]))
    assert msgs[0]["content"] is None


def test_ir_projects_anthropic_image_only_user_without_empty_text():
    msgs = ir_to_openai_messages(anthropic_messages_to_ir([{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _IMG1}}
        ]}]))
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "image_url"

# --- message_text ---

def test_message_text_string():
    assert _message_text("hello") == "hello"

def test_message_text_empty_string():
    assert _message_text("") == ""

def test_message_text_list_of_text_parts():
    content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert _message_text(content) == "a\nb"

def test_message_text_list_with_string_parts():
    content = ["hello", "world"]
    assert _message_text(content) == "hello\nworld"

def test_message_text_list_mixed():
    content = ["plain", {"type": "text", "text": "structured"}]
    assert _message_text(content) == "plain\nstructured"

def test_message_text_list_skips_non_text():
    content = [{"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "ok"}]
    assert _message_text(content) == "ok"

def test_message_text_none():
    assert _message_text(None) == ""

def test_message_text_empty_list():
    assert _message_text([]) == ""

def test_message_text_empty_text_filtered():
    content = [{"type": "text", "text": ""}, {"type": "text", "text": "ok"}]
    assert _message_text(content) == "ok"


# --- attr ---

def test_attr_dict():
    assert _attr({"a": 1}, "a") == 1
    assert _attr({"a": 1}, "b") is None
    assert _attr({"a": 1}, "b", 99) == 99

def test_attr_object():
    class Obj:
        x = 42
    assert _attr(Obj(), "x") == 42
    assert _attr(Obj(), "y") is None
    assert _attr(Obj(), "y", 0) == 0

def test_attr_dict_none_obj():
    assert _attr(None, "x") is None
    assert _attr(None, "x", "default") == "default"
