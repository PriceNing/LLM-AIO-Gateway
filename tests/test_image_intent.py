from app.core.image_intent import latest_user_text
from app.protocols.responses_features import is_system_turn as _responses_is_system_turn


def test_thread_title_template_unwraps_real_user_prompt():
    hello = (
        "Generate a concise UI title (up to 36 characters) for this task.\n"
        "Fill the structured title field with plain text.\n"
        "User prompt:\n你好"
    )
    draw = (
        "Generate a concise UI title (up to 36 characters) for this task.\n"
        "User prompt:\n帮我画一张猫的图片"
    )
    assert latest_user_text(hello) == "你好"
    assert latest_user_text(draw) == "帮我画一张猫的图片"


def test_wrapped_prompt_markers_are_stripped():
    samples = [
        "User message:\n你好",
        "User input:\n你好",
        "## Prompt\n你好",
        "## User input\n你好",
        "用户提示：\n你好",
        "用户输入：\n你好",
    ]
    for text in samples:
        assert latest_user_text(text) == "你好"


def test_environment_context_user_item_is_skipped():
    input_data = [
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "<environment_context>\n  <cwd>/tmp</cwd>\n</environment_context>"},
        ]},
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "你好"},
        ]},
    ]
    assert latest_user_text(input_data) == "你好"


def test_thread_title_xml_wrapper_is_not_user_text():
    greeting = "<thread_title>打招呼</thread_title>\n\n你好"
    drawing = "<thread_title>画猫</thread_title>\n\n帮我画一张猫的图片"
    assert latest_user_text(greeting) == "你好"
    assert latest_user_text(drawing) == "帮我画一张猫的图片"


def test_thread_title_source_is_a_system_turn():
    assert _responses_is_system_turn({
        "client_metadata": {"x-codex-turn-metadata": {
            "thread_source": "thread_title",
            "turn_trigger": "thread_title",
        }},
    }) is True
