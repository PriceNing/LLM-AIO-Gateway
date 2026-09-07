import pytest

from app.protocols.ingress import (
    anthropic_messages_to_internal,
    chat_completions_to_internal,
    completions_to_internal,
    responses_to_internal,
)
from app.core.policy import prepare_request_policy
from app.core.policy import RouteTarget, RoutingDecision
from app.core.policy import has_missing_reasoning_content_for_tool_calls, fix_tool_args, inject_reasoning_content, request_has_tools, strip_tools
from app.protocols.ir import ir_to_anthropic_messages, ir_to_openai_messages, openai_messages_to_ir, responses_input_to_ir
from app.core.types import InternalMessage, text_part
from app.adapters.anthropic import anthropic_body_from_internal
from app.adapters.openai import chat_kwargs_from_internal, chat_messages_from_internal
from app.services.preprocessing import preprocess_messages


def test_ir_to_openai_messages_moves_late_system_turns_to_front():
    messages = [InternalMessage(role="user", parts=[text_part("hi")]), InternalMessage(role="system", parts=[text_part("rules")])]
    assert [m["role"] for m in ir_to_openai_messages(messages)] == ["system", "user"]
    assert ir_to_openai_messages(messages)[0]["content"] == "rules"


def test_ir_to_openai_messages_collapses_multiple_system_turns():
    messages = [
        InternalMessage(role="system", parts=[text_part("base")]),
        InternalMessage(role="user", parts=[text_part("hi")]),
        InternalMessage(role="system", parts=[text_part("extra")]),
    ]
    projected = ir_to_openai_messages(messages)
    assert [m["role"] for m in projected] == ["system", "user"]
    assert projected[0]["content"] == "base\n\nextra"


def test_responses_input_merges_instructions_and_developer_into_one_system():
    messages = responses_input_to_ir(
        [
            {"type": "message", "role": "developer", "content": "You are helpful."},
            {"type": "message", "role": "user", "content": "Hi"},
            {"type": "message", "role": "system", "content": "Stay concise."},
        ],
        instructions="Base rules",
    )
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].parts[0].text == "Base rules\n\nYou are helpful.\n\nStay concise."


def test_ingress_uses_config_temperature_default(monkeypatch):
    monkeypatch.setattr("app.protocols.ingress.get_default", lambda key, fallback=None: 0.25 if key == "temperature" else fallback)

    assert chat_completions_to_internal({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).temperature == 0.25
    assert completions_to_internal({"model": "m", "prompt": "hi"}).temperature == 0.25
    assert anthropic_messages_to_internal({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).temperature == 0.25
    assert responses_to_internal({"model": "m", "input": "hi"}).temperature == 0.25


def test_ingress_preserves_explicit_null_temperature():
    req = chat_completions_to_internal({"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": None})

    assert req.temperature is None


def test_chat_completions_to_internal_normalizes_tools():
    req = chat_completions_to_internal({
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "run", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
        "stream": True,
    })

    assert req.endpoint == "chat_completions"
    assert req.requested_model == "gpt-test"
    assert req.messages[0].role == "user"
    assert req.messages[0].parts[0].kind == "text"
    assert req.messages[0].parts[0].text == "hi"
    assert req.tools[0].name == "run"
    assert req.chat_tools()[0]["function"]["name"] == "run"
    assert req.extra["tool_choice"] == "auto"
    assert req.stream is True


def test_chat_completions_preserves_reasoning_parameters_for_openai_adapter():
    body = {
        "model": "qwen-test",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "xhigh",
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
    }

    req = chat_completions_to_internal(body)
    kwargs = chat_kwargs_from_internal(req)

    assert req.extra["reasoning_effort"] == "xhigh"
    assert req.extra["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }
    assert kwargs["reasoning_effort"] == "xhigh"
    assert kwargs["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }
    assert {"reasoning_effort", "chat_template_kwargs"}.issubset(kwargs["allowed_openai_params"])


def test_responses_nested_reasoning_effort_projects_to_openai_adapter():
    req = responses_to_internal({
        "model": "qwen-test",
        "input": "hi",
        "reasoning": {"effort": "low"},
    })
    kwargs = chat_kwargs_from_internal(req)

    assert req.extra["reasoning_effort"] == "low"
    assert req.extra["enable_thinking"] is True
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["enable_thinking"] is True
    assert "reasoning_effort" in kwargs["allowed_openai_params"]
    assert "reasoning" not in kwargs


def test_completions_to_internal_wraps_prompt_as_user_message():
    req = completions_to_internal({
        "model": "text-test",
        "prompt": ["line one", "line two"],
        "stream": True,
        "max_completion_tokens": 42,
    })

    assert req.endpoint == "completions"
    assert req.requested_model == "text-test"
    assert req.messages[0].role == "user"
    assert req.messages[0].parts[0].text == "line one\nline two"
    assert req.stream is True
    assert req.max_tokens == 42


def test_anthropic_messages_to_internal_always_converts_to_internal_shape():
    body = {
        "model": "claude-test",
        "system": "sys",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "tools": [{"name": "run", "input_schema": {"type": "object"}}],
    }
    req = anthropic_messages_to_internal(body)

    assert req.endpoint == "messages"
    assert req.system == ""
    assert req.messages[0].role == "system"
    assert req.messages[0].parts[0].text == "sys"
    assert req.messages[1].role == "user"
    assert req.messages[1].parts[0].text == "hi"
    assert req.anthropic_tools()[0]["name"] == "run"


def test_anthropic_messages_to_internal_converts_for_openai_provider():
    req = anthropic_messages_to_internal({
        "model": "openai-test",
        "system": "sys",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "tool_choice": {"type": "auto"},
    })

    projected = chat_messages_from_internal(req)
    assert projected[0] == {"role": "system", "content": "sys"}
    assert projected[1] == {"role": "user", "content": "hi"}
    assert req.tool_choice is None


def test_anthropic_unknown_blocks_project_to_openai_text_placeholders():
    req = anthropic_messages_to_internal({
        "model": "openai-test",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Use this"},
            {"type": "tool_reference", "id": "toolu_1", "name": "artifact"},
        ]}],
    })

    projected = chat_messages_from_internal(req)

    assert projected[0] == {"role": "user", "content": [
        {"type": "text", "text": "Use this"},
        {"type": "text", "text": "[tool_reference: artifact]"},
    ]}


def test_anthropic_messages_to_internal_preserves_reasoning_parameters_for_openai_adapter():
    req = anthropic_messages_to_internal({
        "model": "openai-test",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "medium",
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
        "enable_thinking": False,
    })
    kwargs = chat_kwargs_from_internal(req)

    assert req.extra["reasoning_effort"] == "medium"
    assert req.extra["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    assert req.extra["enable_thinking"] is False
    assert kwargs["reasoning_effort"] == "medium"
    assert kwargs["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    assert kwargs["enable_thinking"] is False
    assert {
        "reasoning_effort",
        "chat_template_kwargs",
        "enable_thinking",
    }.issubset(kwargs["allowed_openai_params"])


def test_anthropic_messages_to_internal_preserves_any_tool_choice():
    req = anthropic_messages_to_internal({
        "model": "claude-test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "run", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any"},
    })
    messages, body = anthropic_body_from_internal(req)

    assert messages[0]["content"][0]["text"] == "hi"
    assert req.tool_choice == {"type": "any"}
    assert body["tool_choice"] == {"type": "any"}


def test_anthropic_messages_tool_choice_projects_to_openai_chat_shape():
    req = anthropic_messages_to_internal({
        "model": "openai-test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "run", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "run"},
    })

    kwargs = chat_kwargs_from_internal(req)
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "run"}}


def test_ir_preserves_anthropic_cache_control_tool_use_tool_result_and_images():
    req = anthropic_messages_to_internal({
        "model": "claude-test",
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reason", "signature": "sig_abc"},
                {"type": "text", "text": "I will call"},
                {"type": "tool_use", "id": "toolu_1", "name": "run", "input": {"x": 1}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "ok"}]},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
            ]},
        ],
    })

    system = req.messages[0]
    assistant = req.messages[1]
    user = req.messages[2]
    assert system.parts[0].extensions["cache_control"] == {"type": "ephemeral"}
    assert assistant.parts[0].kind == "reasoning"
    assert assistant.parts[0].extensions["signature"] == "sig_abc"
    assert assistant.parts[2].kind == "tool_call"
    assert assistant.parts[2].tool_call_id == "toolu_1"
    assert assistant.parts[2].arguments == {"x": 1}
    assert user.parts[0].kind == "tool_result"
    assert user.parts[0].tool_call_id == "toolu_1"
    assert user.parts[1].kind == "image"

    anthropic_messages, system_out = ir_to_anthropic_messages(req.messages)
    assert system_out[0]["cache_control"] == {"type": "ephemeral"}
    assert anthropic_messages[0]["content"][0] == {"type": "thinking", "thinking": "reason", "signature": "sig_abc"}
    assert anthropic_messages[0]["content"][2]["id"] == "toolu_1"


def test_anthropic_adapter_uses_ir():
    req = anthropic_messages_to_internal({
        "model": "claude-test",
        "system": "sys-v2",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "from-v2"}]}],
        "tools": [{"name": "run", "description": "Run", "input_schema": {"type": "object"}}],
    })
    messages, body = anthropic_body_from_internal(req)

    assert body["system"] == "sys-v2"
    assert body["tools"][0]["name"] == "run"
    assert messages[0]["content"][0]["text"] == "from-v2"


def test_openai_adapter_requires_and_uses_ir():
    req = chat_completions_to_internal({
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "from-v2"}],
    })
    assert chat_messages_from_internal(req)[0]["content"] == "from-v2"

    req.messages = []
    with pytest.raises(ValueError):
        chat_messages_from_internal(req)


def test_responses_to_internal_converts_function_items_tools_and_additional_tools():
    req = responses_to_internal({
        "model": "gpt-test",
        "instructions": "sys",
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": [
                {"type": "custom", "name": "exec", "description": "Run JS"}
            ]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "call_1", "name": "run", "arguments": "{}"},
        ],
        "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}, "strict": True}],
        "tool_choice": "auto",
    })

    assert req.endpoint == "responses"
    projected = chat_messages_from_internal(req)
    assert projected[0] == {"role": "system", "content": "sys"}
    assert projected[1] == {"role": "user", "content": "hi"}
    assert projected[2]["tool_calls"][0]["id"] == "call_1"
    assert req.messages[1].parts[0].kind == "text"
    assert req.messages[2].parts[0].kind == "tool_call"
    assert req.messages[2].parts[0].tool_call_id == "call_1"
    assert req.tools[0].name == "run"
    assert "strict" not in req.extra["tools"][0]["function"]
    assert req.tool_choice is None
    assert req.extra["responses_custom_tools"]["exec"]["argument_field"] == "input"
    assert req.tools[1].name == "exec"
    assert req.chat_tools()[1]["function"]["parameters"]["required"] == ["input"]


def test_responses_namespace_tools_keep_original_chat_names():
    req = responses_to_internal({
        "model": "gpt-test",
        "input": "hi",
        "tools": [{
            "type": "namespace",
            "name": "collaboration",
            "tools": [{"type": "function", "name": "spawn_agent", "parameters": {"type": "object"}}],
        }, {
            "type": "custom",
            "name": "exec_command",
            "description": "Run a command",
        }],
    })

    names = [tool.name for tool in req.tools]
    assert "spawn_agent" in names
    assert "exec_command" in names
    assert "collaboration-spawn_agent" not in names
    assert req.extra["responses_namespace_tools"]["spawn_agent"] == {"namespace": "collaboration", "name": "spawn_agent"}
    kwargs = chat_kwargs_from_internal(req)
    chat_names = [tool["function"]["name"] for tool in kwargs["tools"]]
    assert "spawn_agent" in chat_names
    assert "exec_command" in chat_names


def test_responses_namespace_custom_tools_survive_chat_rewrite():
    req = responses_to_internal({
        "model": "gpt-test",
        "input": "hi",
        "tools": [{
            "type": "namespace",
            "name": "functions",
            "tools": [
                {"type": "custom", "name": "exec", "description": "Run JavaScript"},
                {"type": "custom", "name": "apply_patch", "description": "Apply a patch"},
                {"type": "function", "name": "wait", "parameters": {"type": "object"}},
            ],
        }],
    })

    names = [tool.name for tool in req.tools]
    assert "exec" in names
    assert "apply_patch" in names
    assert "wait" in names

    kwargs = chat_kwargs_from_internal(req)
    chat_names = [tool["function"]["name"] for tool in kwargs["tools"]]
    assert "exec" in chat_names
    assert "apply_patch" in chat_names
    assert "wait" in chat_names

    custom = req.extra["responses_custom_tools"]
    assert custom["exec"] == {"name": "exec", "argument_field": "input"}
    assert custom["apply_patch"] == {"name": "apply_patch", "argument_field": "patch"}
    assert custom["functions.exec"] == {"name": "exec", "argument_field": "input"}


def test_responses_tool_choice_projects_to_openai_chat_shape():
    req = responses_to_internal({
        "model": "gpt-test",
        "input": "hi",
        "tools": [{"type": "function", "name": "run", "parameters": {"type": "object"}}],
        "tool_choice": {"type": "function", "name": "run"},
    })

    kwargs = chat_kwargs_from_internal(req)
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "run"}}


def test_responses_to_internal_ignores_non_additional_tools_blocks():
    req = responses_to_internal({
        "model": "gpt-test",
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"},
        ],
    })

    assert "responses_custom_tools" not in req.extra


def test_responses_to_internal_replays_custom_tool_call_as_chat_tool_message():
    req = responses_to_internal({
        "model": "gpt-test",
        "input": [
            {"type": "custom_tool_call", "call_id": "call_patch", "name": "apply_patch", "input": "*** Begin Patch"},
            {"type": "custom_tool_call_output", "call_id": "call_patch", "output": "Done"},
        ],
        "tools": [{"type": "custom", "name": "apply_patch", "description": "Apply patch"}],
    })

    projected = chat_messages_from_internal(req)
    assert projected[0]["role"] == "assistant"
    assert projected[0]["tool_calls"][0]["function"]["name"] == "apply_patch"
    assert projected[0]["tool_calls"][0]["function"]["arguments"] == '{"patch": "*** Begin Patch"}'
    assert projected[1] == {"role": "tool", "tool_call_id": "call_patch", "content": "Done"}


def test_ir_preserves_responses_images_and_reasoning_content():
    req = responses_to_internal({
        "model": "gpt-test",
        "instructions": "sys",
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "see"},
                {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": "high"},
            ]},
            {"type": "message", "role": "assistant", "content": "done", "reasoning_content": "hidden"},
        ],
    })

    user = req.messages[1]
    assistant = req.messages[2]
    assert user.parts[0].kind == "text"
    assert user.parts[1].kind == "image"
    assert user.parts[1].source["url"] == "data:image/png;base64,abc"
    assert assistant.parts[0].kind == "reasoning"
    assert assistant.parts[0].text == "hidden"

    openai_messages = ir_to_openai_messages(req.messages)
    assert openai_messages[1]["content"][1]["image_url"]["url"] == "data:image/png;base64,abc"
    assert openai_messages[2]["reasoning_content"] == "hidden"


def test_ir_preserves_external_image_url_for_anthropic_as_text_placeholder():
    req = chat_completions_to_internal({
        "model": "claude-test",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "see"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ]}],
    })

    anthropic_messages, _ = ir_to_anthropic_messages(req.messages)
    assert anthropic_messages[0]["content"][1] == {"type": "text", "text": "[image URL: https://example.com/a.png]"}


def test_ir_projects_openai_image_content_in_original_order():
    req = chat_completions_to_internal({
        "model": "gpt-test",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "before"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            {"type": "text", "text": "after"},
        ]}],
    })

    content = ir_to_openai_messages(req.messages)[0]["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "text"]
    assert content[0]["text"] == "before"
    assert content[2]["text"] == "after"


@pytest.mark.parametrize(
    ("image_item", "expected_source"),
    [
        ({"type": "image", "image_url": {"url": "https://example.com/opencode.png", "detail": "high"}}, {"kind": "url", "url": "https://example.com/opencode.png", "detail": "high"}),
        ({"type": "image", "url": "https://example.com/top-level.png"}, {"kind": "url", "url": "https://example.com/top-level.png"}),
        ({"type": "image", "data": "abc123", "mimeType": "image/jpeg"}, {"kind": "base64", "media_type": "image/jpeg", "data": "abc123"}),
        ({"type": "file", "file_data": "data:image/png;base64,abc123"}, {"kind": "url", "url": "data:image/png;base64,abc123"}),
    ],
)
def test_chat_completions_to_internal_accepts_opencode_image_variants(image_item, expected_source):
    req = chat_completions_to_internal({
        "model": "gpt-test",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "see"}, image_item]}],
    })

    image = req.messages[0].parts[1]
    assert image.kind == "image"
    assert image.source == expected_source


def test_thinking_reasoning_requires_real_reasoning_for_every_tool_call():
    req = responses_to_internal({
        "model": "gpt-test",
        "input": [
            {"type": "function_call", "call_id": "old", "name": "run", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "old", "output": "ok"},
            {"type": "function_call", "call_id": "new", "name": "run", "arguments": "{}", "reasoning_content": "new reasoning"},
            {"type": "function_call_output", "call_id": "new", "output": "ok"},
        ],
    })
    assert has_missing_reasoning_content_for_tool_calls(req.messages) is True

    for message in req.messages:
        if message.role == "assistant" and any(part.tool_call_id == "old" for part in message.parts):
            message.parts.insert(0, text_part("old reasoning"))
            message.parts[0].kind = "reasoning"
    assert has_missing_reasoning_content_for_tool_calls(req.messages) is False


def test_policy_inject_reasoning_content_active_tool_segment_only():
    req = chat_completions_to_internal({
        "model": "gpt-test",
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "old", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "old", "content": "old result"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "new", "type": "function", "function": {"name": "run", "arguments": "{}"}}]},
        ],
    })

    injected = inject_reasoning_content(req.messages, "fallback", {"new": "new reasoning"})

    assert injected == 1
    assert req.messages[1].parts[0].kind == "tool_call"
    assert req.messages[5].parts[0].kind == "reasoning"
    assert req.messages[5].parts[0].text == "new reasoning"


def test_policy_tool_only_limit_strips_internal_tools_and_extra():
    req = chat_completions_to_internal({
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "run", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
    })

    assert request_has_tools(req) is True
    strip_tools(req)

    assert request_has_tools(req) is False
    assert req.tools == []
    assert req.tool_choice is None
    assert "tools" not in req.extra
    assert "tool_choice" not in req.extra


def test_policy_fix_tool_args_repairs_raw_arguments():
    req = responses_to_internal({
        "model": "gpt-test",
        "input": [
            {"type": "message", "role": "user", "content": "search"},
            {"type": "function_call", "call_id": "call_1", "name": "run", "arguments": '{"url": undefined}'},
        ],
    })

    assert fix_tool_args(req) == 1

    tc_part = req.messages[1].parts[0]
    assert tc_part.raw_arguments == '{"url": ""}'
    assert tc_part.arguments == {"url": ""}
    assert chat_messages_from_internal(req)[1]["tool_calls"][0]["function"]["arguments"] == '{"url": ""}'


def test_coerce_tool_arguments_json_wraps_invalid_history_for_llamacpp():
    import json
    from app.core.tool_args import coerce_tool_arguments_json

    raw = '{"cmd":"node -e \\n      return "";\\n"}'
    try:
        json.loads(raw)
        raise AssertionError("fixture must be invalid JSON")
    except json.JSONDecodeError:
        pass

    coerced = coerce_tool_arguments_json(raw)
    assert json.loads(coerced) == {"input": raw}

    valid = '{"cmd":"echo hi"}'
    assert coerce_tool_arguments_json(valid) == valid
    assert coerce_tool_arguments_json({"cmd": "echo hi"}) == '{"cmd": "echo hi"}'
    assert json.loads(coerce_tool_arguments_json('{"value": NaN}')) == {"input": '{"value": NaN}'}


def test_policy_and_openai_projection_repair_invalid_tool_arguments():
    import json
    raw = '{"cmd":"node -e \\n      return "";\\n"}'
    req = responses_to_internal({
        "model": "llamacpp-linux/Qwen3.8-27B",
        "input": [
            {"type": "message", "role": "user", "content": "run tests"},
            {"type": "function_call", "call_id": "call_broken", "name": "exec_command", "arguments": raw},
            {"type": "function_call_output", "call_id": "call_broken", "output": "ok"},
        ],
    })

    tc_part = next(part for msg in req.messages for part in msg.parts if part.kind == "tool_call")
    assert tc_part.raw_arguments == raw
    assert fix_tool_args(req) == 1
    assert json.loads(tc_part.raw_arguments) == {"input": raw}

    projected = chat_messages_from_internal(req)
    args = projected[1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args) == {"input": raw}


async def _fake_preprocess_request(request, model, provider_id, requested_model):
    request.messages.append(openai_messages_to_ir([{"role": "user", "content": f"request-target={model};requested={requested_model}"}])[0])
    return True


def _fake_conv_key(api_key, messages, previous_response_id):
    assert messages and hasattr(messages[0], "parts")
    return f"{api_key}:{previous_response_id}:{len(messages)}"


@pytest.mark.asyncio
async def test_prepare_request_policy_routes_normalizes_preprocesses_and_injects(monkeypatch):
    from app.core import policy

    monkeypatch.setattr(policy, "apply_routing_rules", lambda *_: RoutingDecision(
        requested_model="source-model",
        resolved_model="source-model",
        target=RouteTarget(model="target-model", provider_id="target-provider"),
        matched=True,
        rule_id=7,
        rule_name="route-test",
        source="routing_rule",
        reason="test route",
    ))
    req = chat_completions_to_internal({
        "model": "source-model",
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ],
        "tools": [{"type": "function", "function": {"name": "run"}}],
    })

    result = await prepare_request_policy(
        req,
        username="alice",
        api_key_value="key",
        preprocess_request=_fake_preprocess_request,
        conversation_cache_key=_fake_conv_key,
        log_label="chat",
    )

    assert req.target_model == "target-model"
    assert req.provider_id == "target-provider"
    projected = chat_messages_from_internal(req)
    assert projected[0] == {"role": "user", "content": "a\n\nb"}
    assert projected[-1]["content"] == "request-target=source-model;requested=source-model"
    assert result.routing.matched is True
    assert result.routing.rule_id == 7
    assert result.routing.rule_name == "route-test"
    assert result.routing.target_model == "target-model"
    assert result.routing.target_provider == "target-provider"
    assert result.modified_by_preprocessor is True
    assert result.reasoning_injected == 0


@pytest.mark.asyncio
async def test_prepare_request_policy_can_preprocess_request_and_update_internal(monkeypatch):
    from app.core import policy

    monkeypatch.setattr(policy, "apply_routing_rules", lambda *_: RoutingDecision(
        requested_model="source-model",
        resolved_model="source-model",
        target=RouteTarget(model="target-model", provider_id="target-provider"),
        matched=True,
    ))
    req = chat_completions_to_internal({
        "model": "source-model",
        "messages": [{"role": "user", "content": "a"}],
    })

    result = await prepare_request_policy(
        req,
        username="alice",
        api_key_value="key",
        preprocess_request=_fake_preprocess_request,
        conversation_cache_key=_fake_conv_key,
        log_label="chat",
    )

    assert result.modified_by_preprocessor is True
    assert req.messages[-1].parts[0].text == "request-target=source-model;requested=source-model"


@pytest.mark.asyncio
async def test_preprocess_messages_replaces_current_images(monkeypatch):
    req = chat_completions_to_internal({
        "model": "gpt-test",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe"},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
        ]}],
    })

    async def fake_describe(**kwargs):
        return "A test image"

    monkeypatch.setattr("app.services.preprocessing.describe_image", fake_describe)
    await preprocess_messages(req.messages, {"id": "vision", "enabled": True, "max_images": 5})

    projected = ir_to_openai_messages(req.messages)
    assert "A test image" in projected[0]["content"]
    assert "image_url" not in str(projected[0]["content"])


@pytest.mark.asyncio
async def test_preprocess_messages_maps_duplicate_images_to_same_description(monkeypatch):
    req = chat_completions_to_internal({
        "model": "gpt-test",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/b.jpg"}},
        ]}],
    })
    calls = []

    async def fake_describe(image_url="", image_data="", preprocessor_config=None):
        calls.append(image_url or image_data)
        return "A image" if image_url.endswith("a.jpg") else "B image"

    monkeypatch.setattr("app.services.preprocessing.describe_image", fake_describe)
    await preprocess_messages(req.messages, {"id": "vision", "enabled": True, "max_images": 5})

    text = ir_to_openai_messages(req.messages)[0]["content"]
    assert calls == ["https://example.com/a.jpg", "https://example.com/b.jpg"]
    assert text.count("A image") == 2
    assert text.count("B image") == 1


def test_chat_completions_to_internal_accepts_native_image_blocks():
    req = chat_completions_to_internal({
        "model": "gpt-test",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
        ]}],
    })

    image = req.messages[0].parts[1]
    assert image.kind == "image"
    assert image.source == {"kind": "base64", "media_type": "image/png", "data": "abc"}
