"""收口基线的 pytest 执行入口（审查报告第 10 轮 #3）。

tools/scripts/check_error_mapping.py 的检查 1–3 以常规用例形式运行，保证
"是否收口"在日常 `pytest tests/` 中自动执行，不再只是手动命令。
检查 4（文档计数 vs pytest collect）需要子进程收集，独立脚本才执行；
本文件同时防止语料导入静默失败（检查函数已把导入失败计为 FAIL）。
"""
import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_MODULE_NAME = "check_error_mapping_under_test"
_PATH = _REPO / "tools" / "scripts" / "check_error_mapping.py"

_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _PATH)
check = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = check
_spec.loader.exec_module(check)


def test_gate_diff_corpus():
    failures = check.check_diff_corpus()
    assert not failures, "差分语料出现矛盾/泄漏：\n" + "\n".join(failures)


def test_gate_required_test_markers():
    failures = check.check_required_test_markers()
    assert not failures, "关键错误路径断言缺失：\n" + "\n".join(failures)


def test_gate_hardcoded_status_codes():
    failures = check.check_hardcoded_status_codes()
    assert not failures, "写死 status_code=500/502 超出白名单：\n" + "\n".join(failures)
