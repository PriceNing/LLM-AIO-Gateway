"""read_log_entries 两遍扫描实现的语义回归（bug-2026-10-05 L-22）。

关键契约：新→旧排序、total=全部匹配数、offset/limit 只应用一次、
level/q 过滤口径与逐行解析结果一致。
"""
import json

import pytest

from app.services import logger as logger_mod


@pytest.fixture()
def log_file(tmp_path, monkeypatch):
    def _write(lines, channel="app"):
        path = tmp_path / f"{channel}.log"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setattr(logger_mod, "_log_file_path", lambda date, channel=channel: path)
        return path
    return _write


def _entry(line_no, level="INFO", msg="hello"):
    return json.dumps({"ts": f"2026-10-05T00:00:{line_no:02d}Z", "request_id": f"r{line_no}",
                       "level": level, "logger": "app", "msg": msg})


def test_default_page_is_newest_first(log_file):
    log_file([_entry(i) for i in range(1, 6)])
    out = logger_mod.read_log_entries("2026-10-05", "app", limit=3)
    assert out["total"] == 5
    assert [e["line"] for e in out["items"]] == [5, 4, 3]


def test_offset_applies_exactly_once(log_file):
    log_file([_entry(i) for i in range(1, 11)])
    out = logger_mod.read_log_entries("2026-10-05", "app", limit=3, offset=2)
    assert [e["line"] for e in out["items"]] == [8, 7, 6]
    assert out["total"] == 10


def test_level_filter_counts_all_matches_for_total(log_file):
    lines = [_entry(i, level="INFO" if i % 2 else "ERROR") for i in range(1, 9)]
    log_file(lines)
    out = logger_mod.read_log_entries("2026-10-05", "app", limit=2, level="error")
    assert out["total"] == 4
    assert [e["line"] for e in out["items"]] == [8, 6]
    assert all(e["level"] == "ERROR" for e in out["items"])


def test_text_query_and_non_json_lines(log_file):
    lines = [_entry(1, msg="keepme one"), "raw not json", _entry(3, msg="drop two"), _entry(4, msg="keepme three")]
    log_file(lines)
    out = logger_mod.read_log_entries("2026-10-05", "app", limit=10, q="keepme")
    assert out["total"] == 2
    assert [e["line"] for e in out["items"]] == [4, 1]
    # 无 q 时非 JSON 行也要按原文入页
    out_all = logger_mod.read_log_entries("2026-10-05", "app", limit=10)
    assert out_all["total"] == 4
    assert any("raw not json" == e["msg"] for e in out_all["items"])


def test_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_mod, "_log_file_path", lambda date, channel="app": tmp_path / "nope.log")
    out = logger_mod.read_log_entries("2026-10-05", "app")
    assert out == {"items": [], "total": 0, "limit": 200, "offset": 0, "path": str(tmp_path / "nope.log")}
