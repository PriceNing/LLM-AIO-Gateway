"""InternalOutputMessage.request_details 专用字段的回归测试。

背景：liteLLM 非流式适配器的 output.raw 是 ModelResponse 对象（非 dict），
旧版 _attach_output_request_details 往 raw["request_details"] 写元数据会静默
丢失，导致 openai 非流式成功请求的日志里没有 upstream_endpoint/fallback 字段，
冒烟测试无法断言实际上游协议。
"""
from app.core.output import InternalOutputMessage
from app.router.proxy import _attach_output_request_details, _finalize_success_details, _output_request_details


class _NonDictRaw:
    """模拟 liteLLM ModelResponse：非 dict 的 raw。"""


def test_attach_survives_non_dict_raw():
    output = InternalOutputMessage(text="hi", raw=_NonDictRaw())
    _attach_output_request_details(output, upstream_endpoint="chat_completions", fallback_status="unused")
    details = _output_request_details(output)
    assert details["upstream_endpoint"] == "chat_completions"
    assert details["fallback_status"] == "unused"


def test_attach_merges_multiple_calls():
    output = InternalOutputMessage(text="hi", raw=_NonDictRaw())
    _attach_output_request_details(output, upstream_endpoint="chat_completions")
    _attach_output_request_details(output, attempt_index=0)
    details = _output_request_details(output)
    assert details == {"upstream_endpoint": "chat_completions", "attempt_index": 0}


def test_dict_raw_path_still_supported():
    output = InternalOutputMessage(text="hi", raw={})
    _attach_output_request_details(output, upstream_endpoint="messages")
    assert output.raw["request_details"]["upstream_endpoint"] == "messages"
    assert _output_request_details(output)["upstream_endpoint"] == "messages"


def test_legacy_dict_raw_without_attach_is_readable():
    output = InternalOutputMessage(text="hi", raw={"request_details": {"upstream_endpoint": "messages"}})
    assert _output_request_details(output) == {"upstream_endpoint": "messages"}


def test_finalize_success_details_includes_upstream():
    output = InternalOutputMessage(text="hi", raw=_NonDictRaw())
    _attach_output_request_details(output, upstream_endpoint="chat_completions")
    details = _finalize_success_details(output, policy=None, extra=None)
    assert details.get("upstream_endpoint") == "chat_completions"
