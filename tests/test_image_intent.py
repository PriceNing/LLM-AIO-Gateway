from app.core.image_intent import is_image_generation_intent, latest_user_text
from app.router.proxy import _responses_is_system_turn


def _user_message(text: str) -> list[dict]:
    return [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}]


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
    assert is_image_generation_intent(hello) is False
    assert is_image_generation_intent(_user_message(hello)) is False
    assert latest_user_text(draw) == "帮我画一张猫的图片"
    assert is_image_generation_intent(draw) is True


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
        assert is_image_generation_intent(text) is False


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
    assert is_image_generation_intent([input_data[0]]) is False
    assert is_image_generation_intent(input_data) is False


def test_thread_title_xml_wrapper_is_not_user_intent():
    greeting = "<thread_title>打招呼</thread_title>\n\n你好"
    drawing = "<thread_title>画猫</thread_title>\n\n帮我画一张猫的图片"
    assert latest_user_text(greeting) == "你好"
    assert is_image_generation_intent(greeting) is False
    assert latest_user_text(drawing) == "帮我画一张猫的图片"
    assert is_image_generation_intent(drawing) is True


def test_explicit_image_requests_still_match():
    assert is_image_generation_intent("生成一个苹果的图像") is True
    assert is_image_generation_intent("请画一张赛博朋克城市图片") is True
    assert is_image_generation_intent("generate an image of an apple") is True
    assert is_image_generation_intent("draw a poster of a red cat") is True
    assert is_image_generation_intent("帮我画一张猫的图片") is True


def test_cn_paint_concrete_object_matches():
    # "画一个X" (paint verb + quantity + concrete subject, no explicit 图/图片)
    # is an image-generation request; abstract/diagram nouns are not.
    assert is_image_generation_intent("画一个红苹果") is True
    assert is_image_generation_intent("画一个流程图") is False
    assert is_image_generation_intent("画一下代码") is False
    assert is_image_generation_intent("制作一个架构图") is False


def test_cn_paint_measure_words_and_abstract_exclusions():
    # Broader measure words (m6): 片/束/只/对/座/栋/棵/条 all generate.
    assert is_image_generation_intent("画一片海") is True
    assert is_image_generation_intent("画一束花") is True
    assert is_image_generation_intent("画一只猫") is True
    assert is_image_generation_intent("画一对蝴蝶") is True
    assert is_image_generation_intent("画一座山") is True
    # Abstract/creative subjects that are not bitmap requests (m7).
    assert is_image_generation_intent("画一首歌") is False
    assert is_image_generation_intent("画一个故事") is False
    assert is_image_generation_intent("画一个脚本") is False
    assert is_image_generation_intent("画一个小说") is False


def test_vector_format_requests_are_not_routed_to_bridge():
    # Vector-format requests (svg / 矢量) ask for a vector file the raster
    # backend cannot produce; the model should answer with markup, not a bitmap.
    assert is_image_generation_intent("使用svg画一个骑自行车的鹈鹕") is False
    assert is_image_generation_intent("画一张SVG海报") is False
    assert is_image_generation_intent("画一个矢量logo") is False
    assert is_image_generation_intent("draw a vector logo of a bird") is False
    assert is_image_generation_intent("generate an svg of a cat") is False
    # "矢量风格" is a look, not a file type -- still generates a bitmap.
    assert is_image_generation_intent("画一张矢量风格的插画") is True
    assert is_image_generation_intent("画一个红苹果") is True


def test_thread_title_source_is_a_system_turn():
    assert _responses_is_system_turn({
        "client_metadata": {"x-codex-turn-metadata": {
            "thread_source": "thread_title",
            "turn_trigger": "thread_title",
        }},
    }) is True
