#!/usr/bin/env python
"""客户端错误状态码映射的收口校验脚本（审查基线固化，可重复执行）。

检查项（对应审查报告"复核基线"四条；第 10 轮修复四处"跳过即绿"）：
1. 差分语料：client_status_for_upstream_error 与 friendly_error_msg 的状态码/文案
   类别必须同源一致，且上游原文绝不泄漏进客户端文案；语料导入失败按 FAIL 计，
   不再静默缩减覆盖面；
2. 关键错误路径存在状态码断言：不仅匹配用例名，还断言用例内的关键 assert 子串
   （防"删名字保不住断言、留名字删掉断言"两种弱化）；
3. app/ 下写死 status_code=500/502 仅剩白名单：按 (文件, 规范化行文本) 计数比对，
   同文件内替换成不同文本的写死点会作为新签名暴露；
4. README/README_en/AGENTS/CLAUDE 的测试计数与 pytest collect 实际数一致：
   收集失败按 FAIL 计（离线/受限环境可显式 --skip-counts 降级为 SKIP 并打印）。

执行入口：pytest 常规套件中的 tests/test_error_mapping_gate.py 会运行检查 1–3；
本脚本独立运行时执行全部四项（检查 4 需要子进程收集，不适合放进 pytest 内）。

用法：
    python tools/scripts/check_error_mapping.py [--skip-counts]
退出码 0 = 全部通过；1 = 存在失败项。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import httpx  # noqa: E402

from app.core.text import (  # noqa: E402
    _CONNECTION_FAILURE,
    _GENERIC_UPSTREAM_FAILURE,
    _RATE_LIMIT_FAILURE,
    _TIMEOUT_FAILURE,
    _UPSTREAM_CREDENTIAL_FAILURE,
    _UPSTREAM_REJECTION,
    client_status_for_upstream_error,
    friendly_error_msg,
)

LEAK_TOKEN = "LEAK-TOKEN-9f3c"

# 文案类别 → 允许的状态码集合（模式表/余额覆盖文案豁免，见 check 1 的 else 分支）
CATEGORY_RULES: list[tuple[str, set[int]]] = [
    (_UPSTREAM_REJECTION, set(range(400, 500)) - {401, 403, 408, 429}),
    (_RATE_LIMIT_FAILURE, {429}),
    (_TIMEOUT_FAILURE, {504}),
    (_UPSTREAM_CREDENTIAL_FAILURE, {502}),
    (_CONNECTION_FAILURE, {502}),
    (_GENERIC_UPSTREAM_FAILURE, {500, 502}),
]

# 写死状态码白名单：(相对路径, 规范化行文本) → 允许出现次数。
# 只覆盖"网关自撰语义错误"与"管理员端点"，不在客户端错误映射面内。
HARDCODED_WHITELIST: Counter = Counter({
    ("app/adapters/anthropic_streaming.py",
     'raise HTTPException(status_code=502, detail="Upstream: anthropic stream closed before message completion")'): 1,
    ("app/core/image_orchestration.py", "status_code=502,"): 1,  # 模型未调用生图工具（网关自撰文案）
    ("app/router/admin.py",
     'raise HTTPException(status_code=502, detail="Failed to fetch models from server")'): 1,
    ("app/router/admin.py", "raise HTTPException(status_code=502, detail=detail)"): 1,
})

# 关键路径的测试断言必须存在：文件 → 必须同时出现的子串（用例名 + 断言文本）。
REQUIRED_TEST_MARKERS: dict[str, list[str]] = {
    "tests/test_upstream_error_status.py": [
        "def test_client_status_mapping",
        "assert client_status_for_upstream_error(exc) == expected",
        "def test_proxy_endpoints_map_upstream_errors",
        "assert response.status_code == expected",
        "def test_text_only_status_hints_stay_conservative",
        "def test_status_and_message_are_coherent_on_chained_exceptions",
    ],
    "tests/test_image_generation.py": [
        "def test_images_generation_maps_client_status",
        "assert response.status_code == expected_client",
    ],
    "tests/test_proxy_edge.py": [
        "def test_anthropic_stream_raises_on_upstream_error_event",
        'assert excinfo.value.status_code == 502',
        '"bad stream" not in',
    ],
    "tests/test_live_eval_probes.py": [
        "def test_is_unsupported_rejection",
        "assert le.is_unsupported_rejection(status, expect, probe) is expected",
        "def test_coverage_matrix_and_missing_cells",
    ],
}

COUNT_DOCS = ["README.md", "README_en.md", "AGENTS.md", "CLAUDE.md"]


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", f"https://upstream.test/{LEAK_TOKEN}/v1")
    response = httpx.Response(status, request=request, text=f"raw body {LEAK_TOKEN}")
    return httpx.HTTPStatusError(f"Error {status} {LEAK_TOKEN}", request=request, response=response)


class _LiteLLMStyle(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _corpus() -> tuple[list[tuple[str, Exception]], list[str]]:
    """返回 (语料, 导入失败清单)。导入失败不再静默跳过（第 10 轮 #2a）。"""
    cases: list[tuple[str, Exception]] = []
    import_failures: list[str] = []
    for code in (100, 200, 301, 400, 401, 403, 404, 408, 413, 422, 429, 500, 502, 503, 599):
        cases.append((f"httpx HTTPStatusError({code})", _status_error(code)))
    for code in (400, 401, 429, 503):
        cases.append((f"litellm-style({code})", _LiteLLMStyle(code, f"bad {LEAK_TOKEN}")))
    for text in (
        f"HTTP 429 slow down {LEAK_TOKEN}",
        f"max_tokens must be <= 500 {LEAK_TOKEN}",
        f"upstream timed out {LEAK_TOKEN}",
        f"connection refused {LEAK_TOKEN}",
        f"boom {LEAK_TOKEN}",
    ):
        cases.append((f"text-only: {text[:24]}", Exception(text)))

    chained1 = ConnectionError("connection reset")
    chained1.__cause__ = _status_error(400)
    cases.append(("ConnectionError <- HTTPStatusError(400)", chained1))

    chained2 = _status_error(429)
    chained2.__cause__ = httpx.ConnectError("connection refused")
    cases.append(("HTTPStatusError(429) <- ConnectError", chained2))

    chained3 = _status_error(502)
    chained3.__cause__ = httpx.ConnectTimeout("connect timed out")
    cases.append(("HTTPStatusError(502) <- ConnectTimeout", chained3))

    try:
        from app.adapters.anthropic import _http_exception_from_upstream
        for code in (302, 400, 401, 429, 503):
            cases.append((f"_http_exception_from_upstream({code})", _http_exception_from_upstream(code, f"raw {LEAK_TOKEN}")))
    except Exception as exc:
        import_failures.append(f"app.adapters.anthropic 语料导入失败：{exc}")

    try:
        from app.adapters.imagegen import ImageBackendHTTPError
        for code in (400, 429, 503):
            cases.append((f"ImageBackendHTTPError({code})", ImageBackendHTTPError(code, f"backend {LEAK_TOKEN}")))
    except Exception as exc:
        import_failures.append(f"app.adapters.imagegen 语料导入失败：{exc}")
    return cases, import_failures


def check_diff_corpus() -> list[str]:
    failures: list[str] = []
    cases, import_failures = _corpus()
    for item in import_failures:
        failures.append(f"[corpus-import] {item}")
    for label, exc in cases:
        status = client_status_for_upstream_error(exc)
        message = friendly_error_msg(exc)
        if LEAK_TOKEN in message:
            failures.append(f"[leak] {label}: 上游原文泄漏进客户端文案")
            continue
        for constant, allowed in CATEGORY_RULES:
            if constant in message:
                if status not in allowed:
                    failures.append(
                        f"[coherence] {label}: 状态码 {status} 与文案类别矛盾（{constant[:32]}... 允许 {sorted(allowed)}）"
                    )
                break
        else:
            # 未命中统一文案 = 模式表/余额覆盖，仅允许文案差异（状态码仍须在 4xx/5xx 语义面）
            if not (400 <= status <= 599):
                failures.append(f"[pattern-override] {label}: 状态码 {status} 不在 4xx/5xx 面")
    return failures


def check_required_test_markers() -> list[str]:
    failures: list[str] = []
    for rel, needles in REQUIRED_TEST_MARKERS.items():
        path = REPO / rel
        if not path.exists():
            failures.append(f"[tests] 缺少文件 {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"[tests] {rel} 缺少必需片段：{needle[:60]}")
    return failures


def check_hardcoded_status_codes() -> list[str]:
    failures: list[str] = []
    pattern = re.compile(r"status_code\s*=\s*(?:500|502)\b")
    found: Counter = Counter()
    for path in (REPO / "app").rglob("*.py"):
        rel = str(path.relative_to(REPO)).replace("\\", "/")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if pattern.search(line):
                found[(rel, line.strip())] += 1
    for (rel, line), count in found.items():
        allowed = HARDCODED_WHITELIST.get((rel, line), 0)
        if count > allowed:
            failures.append(f"[hardcode] {rel}: “{line[:72]}” 出现 {count} 次（白名单上限 {allowed}）")
    stale = [key for key in HARDCODED_WHITELIST if key not in found]
    for rel, line in stale:
        print(f"  note: 白名单条目已消失，建议清理：{rel}: {line[:60]}")
    return failures


def collect_test_count() -> int:
    """返回收集数；失败抛 RuntimeError（不再返回 None 静默跳过，第 10 轮 #2b）。"""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    match = re.search(r"(\d+) tests? collected", (result.stdout or "") + (result.stderr or ""))
    if not match:
        raise RuntimeError(f"pytest collect 输出无法解析（rc={result.returncode}）")
    return int(match.group(1))


def check_doc_counts(expected: int) -> list[str]:
    failures: list[str] = []
    for rel in COUNT_DOCS:
        path = REPO / rel
        if not path.exists():
            failures.append(f"[counts] 缺少文档 {rel}")
            continue
        found = re.findall(r"(\d+) passed", path.read_text(encoding="utf-8", errors="replace"))
        if not found:
            failures.append(f"[counts] {rel} 未声明测试计数")
        for value in found:
            if int(value) != expected:
                failures.append(f"[counts] {rel} 计数 {value} != 实际 {expected}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-counts", action="store_true",
                        help="显式跳过文档计数检查（离线/受限环境）；默认收集失败按 FAIL 计。")
    args = parser.parse_args(argv)

    results: list[tuple[str, list[str]]] = [
        ("差分语料状态码/文案同源一致", check_diff_corpus()),
        ("关键错误路径存在状态码断言", check_required_test_markers()),
        ("写死状态码仅剩白名单", check_hardcoded_status_codes()),
    ]
    if args.skip_counts:
        print("SKIP  文档测试计数一致（--skip-counts）")
    else:
        try:
            results.append(("文档测试计数一致", check_doc_counts(collect_test_count())))
        except Exception as exc:
            results.append((f"文档测试计数一致（收集失败：{exc}；可用 --skip-counts 显式降级）", ["[counts] pytest 收集失败"]))

    failed = False
    for title, failures in results:
        if failures:
            failed = True
            print(f"FAIL  {title}")
            for item in failures:
                print(f"      - {item}")
        else:
            print(f"PASS  {title}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
