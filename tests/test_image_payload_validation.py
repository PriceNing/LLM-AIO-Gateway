"""网关不得把"看起来像图片、实际非法"的载荷提升为 image_url 附件。

背景（生产事故 2026-09-22，request_id 35d11d8925e8）：
agent 的 evaluate_browser 工具输出被长度上限从中间切断，tool 消息文本里留下一段
23977 字符（%4==1）的 base64。旧实现只看长度与正则形状，把它当作图片附件发给上游，
导致上游返回 400 `invalid base64 format` —— 非法请求是网关自己组装出来的。
"""
import base64
import json

import pytest

from app.core.images import extract_image_data_uris, has_image_content, normalize_image_content
from tests.image_fixtures import (
    JPEG_B64,
    SVG_DATA_URI,
    bmp_b64,
    bmp_bytes,
    jpeg_data_uri,
    png_b64,
    png_data_uri,
    tiff_b64,
    truncated,
    webp_b64,
    webp_bytes,
)


# ---------------------------------------------------------------------------
# 完整性校验：网关不主动制造非法载荷
# ---------------------------------------------------------------------------

def test_truncated_base64_length_not_multiple_of_four_is_not_extracted():
    """生产原案：base64 被从中间切断，长度 %4 == 1。"""
    payload = JPEG_B64[:len(JPEG_B64) - 3]      # 213 字符，%4 == 1
    assert len(payload) % 4 == 1
    assert extract_image_data_uris(f"data:image/jpeg;base64,{payload}") == []


def test_length_aligned_but_missing_container_end_marker_is_rejected():
    """长度合法（%4==0）却没有 JPEG FFD9 / PNG IEND 结束标记，同样必须拒绝。"""
    payload = truncated(png_b64(64, 64), keep=120, align=True)
    assert len(payload) >= 100 and len(payload) % 4 == 0
    raw = base64.b64decode(payload)
    assert not raw.endswith(b"IEND\xaeB`\x82")
    assert extract_image_data_uris(f"data:image/png;base64,{payload}") == []


def test_misaligned_truncation_is_rejected():
    payload = truncated(png_b64(64, 64), keep=130, align=False)
    assert len(payload) % 4 != 0
    assert extract_image_data_uris(f"data:image/png;base64,{payload}") == []


def test_valid_base64_with_wrong_magic_bytes_is_rejected():
    """能解码但不是位图（例如被截断的 JSON 文本恰好是合法 base64）。"""
    payload = base64.b64encode(b'{"state": "ok", "partial": ').decode("ascii") * 3
    assert len(payload) >= 100 and len(payload) % 4 == 0
    assert extract_image_data_uris(f"data:image/png;base64,{payload}") == []


def test_padding_shape_variants_are_rejected():
    """padding 位置/数量非法时不得被当作图片（显式锁定对严格解码的依赖）。"""
    good = png_b64(16, 16)
    assert extract_image_data_uris(f"data:image/png;base64,{good}") != []

    variants = {
        "padding 出现在中间": good[:40] + "=" + good[40:],
        "尾部三个 padding": good[:-2] + "===",
    }
    for label, payload in variants.items():
        assert extract_image_data_uris(f"data:image/png;base64,{payload}") == [], label


def test_complete_bitmaps_are_still_extracted():
    """回归保护：真实完整图片仍要正常提升，否则视觉能力被误伤。"""
    uri = png_data_uri(16, 16)
    assert extract_image_data_uris(uri) == [("image/png", uri)]
    assert extract_image_data_uris(jpeg_data_uri()) == [("image/jpeg", jpeg_data_uri())]


def test_bmp_and_webp_complete_by_length_field():
    """无结束标记的格式靠容器自带长度字段证明完整（bfSize / RIFF size）。"""
    for mime, payload in (("bmp", bmp_b64()), ("webp", webp_b64())):
        assert extract_image_data_uris(f"data:image/{mime};base64,{payload}") == [
            (f"image/{mime}", f"data:image/{mime};base64,{payload}")], mime


def test_bmp_truncation_is_rejected_by_bfsize():
    """截断的 BMP：魔数仍在、base64 仍合法，但 bfSize 与实际长度不符 → 必须拒绝。"""
    raw = bmp_bytes(1200)
    cut = raw[:int(len(raw) * 0.4)]
    cut = cut[:len(cut) - (len(cut) % 4)]
    # 前置条件：确实是“魔数在、长度不一致”的截断形态，而不是被长度门槛顺带拦下
    assert cut[:2] == b"BM"
    assert int.from_bytes(cut[2:6], "little") != len(cut)
    payload = base64.b64encode(cut).decode("ascii")
    assert len(payload) >= 100
    assert extract_image_data_uris(f"data:image/bmp;base64,{payload}") == []


def test_webp_truncation_is_rejected_by_riff_size():
    raw = webp_bytes()
    cut = raw[:int(len(raw) * 0.5)]
    cut = cut[:len(cut) - (len(cut) % 4)]
    payload = base64.b64encode(cut).decode("ascii")
    assert len(payload) >= 100
    assert extract_image_data_uris(f"data:image/webp;base64,{payload}") == []


def test_webp_with_inconsistent_riff_size_is_rejected():
    """头部声称的长度与实际不符（典型的“拼了一半”）→ 不提升。"""
    raw = bytearray(webp_bytes())
    raw[4:8] = (len(raw) + 999).to_bytes(4, "little")
    payload = base64.b64encode(bytes(raw)).decode("ascii")
    assert extract_image_data_uris(f"data:image/webp;base64,{payload}") == []


def test_tiff_is_not_promoted_because_completeness_is_unprovable():
    """既无结束标记也无总长度字段的格式：无法证明完整就不动（默认拒绝）。"""
    assert extract_image_data_uris(f"data:image/tiff;base64,{tiff_b64()}") == []


def test_svg_and_unknown_formats_are_left_untouched():
    """网关无法证明它是位图时一律不动，交给上游自己判断（不写厂商白名单）。"""
    assert extract_image_data_uris(SVG_DATA_URI) == []
    assert extract_image_data_uris("data:image/png;base64," + JPEG_B64) == []  # 声明 png 实为 jpeg


# ---------------------------------------------------------------------------
# 删除位置精确性：短 URI 不得挖掉长载荷的前缀
# ---------------------------------------------------------------------------

def test_verified_short_uri_does_not_mangle_longer_payload():
    """base64 字符类包含 ``=``，已验证的短 URI 可能恰为未验证长载荷的前缀。

    旧实现用 str.replace 删除，会把长载荷的前缀挖掉留下碎片；
    现在按位置从后往前删，未验证的那段必须逐字节原样保留。
    """
    short = png_b64(16, 16)
    assert short.endswith("=")
    # 长载荷 = 短 URI + 额外 base64 字符（正则会把 == 后的字符一并吃进来）
    long_payload = short + "AAAAAAAA"
    assert extract_image_data_uris(f"data:image/png;base64,{long_payload}") == []

    messages = [{"role": "user", "content": f"a data:image/png;base64,{short} b data:image/png;base64,{long_payload} c"}]
    normalize_image_content(messages)
    parts = messages[0]["content"]
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")

    # 长载荷完整保留（旧实现用 str.replace 会把它的前缀一并挖掉，只剩碎片）
    assert f"data:image/png;base64,{long_payload}" in text, "长载荷必须逐字节原样保留"
    # 被提升的那个 URI 已从文本中按位置移除：只剩长载荷那一个 URI 前缀
    assert text.count("data:image/png;base64,") == 1, text
    assert "AAAA" in text
    urls = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]
    assert urls == [f"data:image/png;base64,{short}"]


def test_duplicate_uri_is_promoted_only_once():
    """同一份载荷重复出现只提升一次（与预处理层“同轮重复图片去重”口径一致）。"""
    uri = png_data_uri(16, 16)
    messages = [{"role": "user", "content": f"{uri} 再看一次 {uri}"}]
    normalize_image_content(messages)
    parts = messages[0]["content"]
    urls = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]
    assert urls == [uri]


# ---------------------------------------------------------------------------
# 边界收窄：只改写 user 消息
# ---------------------------------------------------------------------------

def test_tool_message_data_uri_is_not_promoted():
    """工具返回值里的 data URI 是日志/载荷文本，不是要给模型看的图片。"""
    uri = png_data_uri(16, 16)
    messages = [{"role": "tool", "tool_call_id": "call_1",
                 "content": f"Evaluation value: \"{uri}\""}]
    normalize_image_content(messages)
    assert isinstance(messages[0]["content"], str), "tool 消息必须保持纯文本"
    assert uri in messages[0]["content"], "原文本不得被删改"


def test_assistant_message_data_uri_is_not_promoted():
    uri = png_data_uri(16, 16)
    messages = [{"role": "assistant", "content": f"结果如下 {uri}"}]
    normalize_image_content(messages)
    assert messages[0]["content"] == f"结果如下 {uri}"


def test_user_message_data_uri_is_promoted():
    uri = png_data_uri(16, 16)
    messages = [{"role": "user", "content": f"看这张图 {uri}"}]
    normalize_image_content(messages)
    parts = messages[0]["content"]
    assert isinstance(parts, list)
    assert parts[0] == {"type": "text", "text": "看这张图"}
    assert parts[1]["image_url"]["url"] == uri


def test_user_list_content_data_uri_is_promoted():
    uri = png_data_uri(16, 16)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"内联图片 {uri}"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]}]
    normalize_image_content(messages)
    parts = messages[0]["content"]
    urls = [p.get("image_url", {}).get("url") for p in parts if p.get("type") == "image_url"]
    assert uri in urls and "https://example.com/a.png" in urls


# ---------------------------------------------------------------------------
# 只删除"确实被提取"的那段 URI
# ---------------------------------------------------------------------------

def test_unverified_uri_survives_when_sibling_is_promoted():
    """同一条消息里既有完整图片又有残缺载荷时，残缺的那段必须原样留在文本里。"""
    good = png_data_uri(16, 16)
    bad = "data:image/jpeg;base64," + JPEG_B64[:len(JPEG_B64) - 3]
    messages = [{"role": "user", "content": f"好图 {good} 坏图 {bad}"}]
    normalize_image_content(messages)
    parts = messages[0]["content"]
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    urls = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]

    assert urls == [good]
    assert good not in text, "被提取的 URI 不应再留在文本里（避免重复计费）"
    assert bad.split(",", 1)[1] in text, "未通过校验的载荷不得被删除"


def test_normalize_never_rewrites_roles_other_than_user():
    uri = png_data_uri(16, 16)
    messages = [
        {"role": "system", "content": f"sys {uri}"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": uri},
        {"role": "user", "content": "无图片"},
    ]
    snapshot = json.dumps(messages, sort_keys=True)
    normalize_image_content(messages)
    # 只有 user 会被改写；这里 user 无图，整份消息必须逐字节不变
    assert json.dumps(messages, sort_keys=True) == snapshot


# ---------------------------------------------------------------------------
# 图片检测与提升保持一致（避免"检测到图片但没附件"）
# ---------------------------------------------------------------------------

def test_has_image_content_false_for_truncated_payload():
    messages = [{"role": "tool", "content": "data:image/png;base64," + truncated(png_b64(64, 64), 130, align=False)}]
    assert has_image_content(messages) is False


def test_has_image_content_true_for_complete_payload():
    assert has_image_content([{"role": "user", "content": png_data_uri(16, 16)}]) is True


def test_has_image_content_still_true_for_explicit_image_part():
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]}]
    assert has_image_content(messages) is True


# ---------------------------------------------------------------------------
# 可观测性：上游 4xx 响应体必须能从异常链里落到日志
# ---------------------------------------------------------------------------

def test_error_detail_for_log_recovers_body_from_exception_chain():
    """liteLLM 会把原始异常包在链上；日志必须能拿到 param/code 这类真正诊断字段。"""
    import httpx
    from openai import BadRequestError

    from app.core.text import error_detail_for_log

    request = httpx.Request("POST", "https://upstream.test/v1/chat/completions")
    payload = {"error": {"code": "400", "message": "Param Incorrect",
                         "param": "invalid base64 format"}}
    response = httpx.Response(400, request=request, content=json.dumps(payload).encode())
    raw = BadRequestError("Error code: 400", response=response, body=payload)
    wrapped = RuntimeError("litellm.BadRequestError: OpenAIException - Param Incorrect")
    wrapped.__cause__ = raw

    detail = error_detail_for_log(wrapped)
    assert "Param Incorrect" in detail
    assert "invalid base64 format" in detail, "上游给出的具体参数错误不得被丢弃"


def test_error_detail_for_log_does_not_duplicate_body():
    """str(exc) 已含响应体时不重复拼接（避免日志膨胀）。"""
    import httpx
    from openai import BadRequestError

    from app.core.text import error_detail_for_log

    body = {"error": {"message": "boom", "param": "bad"}}
    request = httpx.Request("POST", "https://upstream.test/v1/chat/completions")
    response = httpx.Response(400, request=request, content=json.dumps(body).encode())
    exc = BadRequestError("Error code: 400 - " + response.text, response=response, body=body)

    detail = error_detail_for_log(exc)
    assert detail.count('"bad"') == 1
