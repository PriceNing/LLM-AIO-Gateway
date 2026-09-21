"""live_eval 纯函数判定的单元测试。

tools/live_eval/live_eval.py 不在 testpaths 覆盖范围内，其 verdict/score/matrix
判定属"静默失效"类型（假绿比报错更危险），这里补纯函数级回归：
- unsupported 只接受上游 4xx 拒绝，5xx/429/401 必须是 fail（审查报告 #1）；
- 无计分用例的模型 no_signal=True，不得被当作满分健康；
- 覆盖矩阵聚合与缺格判定。
"""
import base64
import importlib.util
import struct
import sys
from pathlib import Path

import pytest

# 按文件加载，不改全局 sys.path（避免 tools/live_eval 目录遮蔽标准库/依赖，
# 审查报告 #5）。先注册进 sys.modules 再 exec：dataclasses 处理带字符串注解的
# 字段时会通过 sys.modules[cls.__module__] 解析命名空间（本模块有
# `from __future__ import annotations`），未注册会在收集期抛
# AttributeError: 'NoneType' object has no attribute '__dict__'。
_MODULE_NAME = "live_eval_under_test"
_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "live_eval" / "live_eval.py"
_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
le = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = le
_spec.loader.exec_module(le)


def _case(name="c", endpoint="/v1/responses", verdict="pass", score=1.0, observed=""):
    return le.CaseResult(
        name=name, endpoint=endpoint, ok=verdict == "pass", score=score,
        verdict=verdict, observed_upstream=observed,
    )


# ---------------------------------------------------------------------------
# unsupported 判定（审查报告 #1）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,expect,probe,expected", [
    (400, "chat_tool", False, True),          # 上游拒绝工具探针 -> unsupported
    (400, "chat_text", True, True),           # 多模态探针（capability_probe）
    (422, "responses_tool", False, True),
    (500, "chat_tool", False, False),         # 网关故障必须保持 fail，不得洗白
    (502, "chat_tool", False, False),
    (504, "responses_tool", False, False),
    (429, "chat_tool", False, False),         # 限流是瞬时状态，不是能力缺失
    (401, "chat_tool", False, False),         # 冒烟 key 配置错误，是 fail
    (403, "chat_tool", False, False),
    (404, "chat_tool", False, False),         # 上游"模型不存在"是配置错误，不是缺能力
    (405, "chat_tool", False, False),
    (410, "chat_tool", False, False),
    (408, "chat_tool", False, False),
    (None, "chat_tool", False, False),
    (200, "chat_text", True, False),
    (400, "chat_text", False, False),         # 非能力探针的 4xx 仍是 fail
])
def test_is_unsupported_rejection(status, expect, probe, expected):
    assert le.is_unsupported_rejection(status, expect, probe) is expected


# ---------------------------------------------------------------------------
# 得分与 no_signal（审查报告 #4）
# ---------------------------------------------------------------------------

def test_score_excludes_skip_and_unsupported():
    cases = [
        _case("a", verdict="pass"),
        _case("b", verdict="fail", score=0.0),
        _case("c", verdict="skip"),
        _case("d", verdict="unsupported"),
    ]
    result = le.ModelResult(model="m", cases=cases)
    assert result.score == 0.5
    assert not result.no_signal


def test_no_signal_when_nothing_scored():
    cases = [_case("a", verdict="skip"), _case("b", verdict="unsupported")]
    result = le.ModelResult(model="m", cases=cases)
    assert result.no_signal
    assert result.passed == 0


# ---------------------------------------------------------------------------
# 上游协议断言与覆盖矩阵
# ---------------------------------------------------------------------------

def test_expected_upstream_rules():
    assert le.expected_upstream_for("responses", "openai") == "responses|chat_completions"
    assert le.expected_upstream_for("chat_completions", "openai") == "chat_completions"
    assert le.expected_upstream_for("responses", "anthropic") == "messages"
    assert le.expected_upstream_for("messages", "") == ""


def test_observed_upstream_picks_latest_match():
    entries = [
        {"endpoint": "responses", "full_time": "2026-09-15T12:00:01", "responses_mode": "native", "upstream_endpoint": "responses"},
        {"endpoint": "responses", "full_time": "2026-09-15T12:00:00", "upstream_endpoint": "chat_completions"},
    ]
    assert le.observed_upstream_from_logs(entries, "responses") == "responses"
    assert le.observed_upstream_from_logs(entries[1:], "responses") == "chat_completions"
    assert le.observed_upstream_from_logs(entries, "messages") == ""
    # details 嵌套形态（/admin/stats 两种形状都要能读）
    nested = [{"endpoint": "messages", "full_time": "x", "details": {"upstream_endpoint": "messages"}}]
    assert le.observed_upstream_from_logs(nested, "messages") == "messages"


def test_coverage_matrix_and_missing_cells():
    r1 = le.ModelResult(model="a", cases=[
        _case("x", endpoint="/v1/responses", observed="responses"),
        _case("y", endpoint="/v1/chat/completions", observed="chat_completions"),
    ])
    matrix = le.build_coverage_matrix([r1])
    assert matrix["responses"]["responses"] == 1
    assert matrix["chat_completions"]["chat_completions"] == 1
    missing = le.missing_reachable_cells(matrix)
    assert ("chat_completions", "messages") in missing
    assert ("responses", "responses") not in missing
    assert len(missing) == len(le.REACHABLE_CELLS) - 2


def test_full_matrix_has_no_missing_cells():
    log_name_to_path = {v: k for k, v in le.ENDPOINT_LOG_NAME.items()}
    cases = [
        _case(f"{ep}_{up}", endpoint=log_name_to_path[ep], observed=up)
        for ep, up in le.REACHABLE_CELLS
    ]
    results = [le.ModelResult(model="full", cases=cases)]
    matrix = le.build_coverage_matrix(results)
    assert le.missing_reachable_cells(matrix) == []


# ---------------------------------------------------------------------------
# 探针图片（1x1 会被真实上游判为无效图，审查报告确认 64x64 修复正确）
# ---------------------------------------------------------------------------

def test_probe_image_is_valid_png_64():
    data = base64.b64decode(le.PROBE_IMAGE_PNG_B64)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (64, 64)


# ---------------------------------------------------------------------------
# 生图探针（chat / responses）
# ---------------------------------------------------------------------------

def test_chat_image_generation_probe_detects_data_uri():
    payload = {"choices": [{"message": {"content": "![Generated image](data:image/jpeg;base64,AAAA)"}}]}
    ok, score, _ = le.evaluate_payload(200, payload, "chat_image_generation")
    assert ok and score == 1.0


def test_chat_image_generation_probe_fails_without_data_uri():
    payload = {"choices": [{"message": {"content": "I cannot generate images."}}]}
    ok, score, _ = le.evaluate_payload(200, payload, "chat_image_generation")
    assert not ok and score < 1.0


def test_responses_image_generation_probe_detects_call():
    payload = {"id": "resp_x", "output": [
        {"type": "image_generation_call", "status": "completed", "result": "AAAA"},
    ]}
    ok, score, _ = le.evaluate_payload(200, payload, "responses_image_generation")
    assert ok and score == 1.0


def test_responses_image_generation_probe_detects_inline_data_uri():
    # image bridge 路径：结果作为 continuation assistant message，output_text 含 data URI。
    payload = {"id": "resp_x", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "![Generated image](data:image/png;base64,AAAA)"}]},
    ]}
    ok, score, _ = le.evaluate_payload(200, payload, "responses_image_generation")
    assert ok and score == 1.0


def test_responses_image_generation_probe_fails_when_absent():
    payload = {"id": "resp_x", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
    ]}
    ok, score, _ = le.evaluate_payload(200, payload, "responses_image_generation")
    assert not ok and score < 1.0


def test_responses_image_generation_probe_ignores_incomplete_call():
    payload = {"id": "resp_x", "output": [
        {"type": "image_generation_call", "status": "in_progress", "result": ""},
    ]}
    ok, _, _ = le.evaluate_payload(200, payload, "responses_image_generation")
    assert not ok


def _fake_client():
    class _C:
        def request(self, *a, **k):
            return 200, {}, 0

        def get_recent_logs(self, *a, **k):
            return []

    return _C()


def test_build_cases_gates_image_generation_probe():
    # 支持生图的模型：加两个生图探针 case。
    supported = le.build_cases(client=_fake_client(), model="m", include_multimodal=False,
                               include_stream=False, caps={"supports_image_generation": True})
    names = {c.name for c in supported}
    assert "chat_image_generation" in names and "responses_image_generation" in names
    # 不支持生图的模型：两个探针都 skip（不真实请求）。
    unsupported = le.build_cases(client=_fake_client(), model="m", include_multimodal=False,
                                 include_stream=False, caps={})
    skipped = {c.name: c.verdict for c in unsupported
               if c.name in ("chat_image_generation", "responses_image_generation")}
    assert skipped == {"chat_image_generation": "skip", "responses_image_generation": "skip"}
