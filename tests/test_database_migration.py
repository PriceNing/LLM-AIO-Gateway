"""启动路径回归：旧库 + 新代码必须能起。

背景（2026-10-01 生产事故）：v0.14.1 把 `CREATE INDEX ... ON request_logs(request_id)`
写进 `_SCHEMA`，而 `init_db()` 先 `executescript(_SCHEMA)` 后跑 `migrate()`。对已存在的
库，`CREATE TABLE IF NOT EXISTS` 是空操作（旧表没有该列），索引语句直接抛
`no such column: request_id`，lifespan 退出 → 容器崩溃重启循环。全新库不触发，所以
「每个测试都建临时新库」的套件全绿也盖不住。

本文件用发布版本 schema 快照造旧库，跑真实的 `init_db()`，把这条路径变成永久闸门：
以后任何列/索引顺序错误都会在这里先炸，而不是在生产第一次启动时炸。
"""
from pathlib import Path
import glob
import sqlite3

import pytest

import app.database as db_mod

FIXTURES = sorted(Path(__file__).parent.glob("fixtures/schema_v*.sql"))


def test_schema_snapshots_exist():
    """快照缺失时本文件会生成 0 个用例，pytest 依旧全绿——闸门静默消失。

    CI 的镜像发布只靠 `pytest tests/ -q` 卡住，所以这条必须响：
    忘记 `git add tests/fixtures/`、或把快照误删时，要的是红掉，不是静默通过。
    """
    assert FIXTURES, (
        "tests/fixtures/schema_v*.sql 不存在：启动路径回归闸门已失效，"
        "必须提交至少一个发布版本 schema 快照"
    )
    for snapshot in FIXTURES:
        text = snapshot.read_text(encoding="utf-8")
        assert "CREATE TABLE" in text, f"{snapshot.name} 不是有效 schema 快照"
        # 快照必须是「升级前」的库：不得包含后来新增的列，否则它盖不到迁移路径。
        assert "request_id TEXT" not in text, (
            f"{snapshot.name} 已含 request_id 列，用它造不出旧库，预检会假绿"
        )

# 快照里存在的表，升级后必须仍然存在（防止改名/误删）。
_SNAPSHOT_TABLES = {
    "providers", "provider_models", "preprocessors", "users", "user_api_keys",
    "admins", "routing_rules", "fallback_policies", "global_stats",
    "request_records", "request_logs", "model_registry", "image_generators",
}


@pytest.fixture
def restore_db_state():
    """init_db() 会改模块级 DB_PATH/_initialized，测试前后必须还原。"""
    original_path = db_mod.DB_PATH
    original_flag = db_mod._initialized
    yield
    db_mod.DB_PATH = original_path
    db_mod._initialized = original_flag


def _build_legacy_db(tmp_path, snapshot: Path) -> str:
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.executescript(snapshot.read_text(encoding="utf-8"))
    # 哨兵行：升级不得丢历史数据，且旧行没有新列值也必须能用默认值补上。
    conn.execute(
        "INSERT INTO request_logs (timestamp, endpoint, requested_model, status)"
        " VALUES ('2026-09-30 10:00:00', 'chat_completions', 'legacy-model', 'ok')"
    )
    conn.commit()
    conn.close()
    return path


@pytest.mark.parametrize("snapshot", FIXTURES, ids=lambda p: p.name)
def test_init_db_boots_on_released_schema(snapshot, tmp_path, restore_db_state):
    db_mod._initialized = False
    path = _build_legacy_db(tmp_path, snapshot)

    init_ok = True
    try:
        db_mod.init_db(path)
    except sqlite3.OperationalError as exc:  # pragma: no cover - 失败路径
        init_ok = False
        raise AssertionError(
            f"旧库（{snapshot.name}）+ 新代码启动失败，生产会进崩溃重启循环：{exc}"
        ) from exc
    assert init_ok

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    sentinel = conn.execute(
        "SELECT requested_model, request_id FROM request_logs WHERE endpoint = 'chat_completions'"
    ).fetchall()
    conn.close()

    assert _SNAPSHOT_TABLES <= tables, f"升级后表缺失：{_SNAPSHOT_TABLES - tables}"
    assert [row["requested_model"] for row in sentinel] == ["legacy-model"]
    assert sentinel[0]["request_id"] == ""


@pytest.mark.parametrize("snapshot", FIXTURES, ids=lambda p: p.name)
def test_repeated_restarts_stay_ok(snapshot, tmp_path, restore_db_state):
    """每次容器重启都会再跑一遍 init_db；第二次、第三次同样必须成功。"""
    db_mod._initialized = False
    path = _build_legacy_db(tmp_path, snapshot)

    db_mod.init_db(path)
    for _ in range(2):
        db_mod._initialized = False
        db_mod.init_db(path)

    conn = sqlite3.connect(path)
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(request_logs)")}
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.close()
    assert "idx_reqlog_request_id" in indexes
    assert integrity == "ok"


@pytest.mark.parametrize("snapshot", FIXTURES, ids=lambda p: p.name)
def test_writes_work_after_upgrade(snapshot, tmp_path, restore_db_state):
    """升级后的库要能立刻写入并按新列检索，不需要人工迁移。"""
    db_mod._initialized = False
    path = _build_legacy_db(tmp_path, snapshot)
    db_mod.init_db(path)

    log_id = db_mod.add_request_log(
        timestamp="2026-10-01 12:00:00",
        endpoint="chat_completions",
        username="alice",
        api_key="sk-aio-***",
        requested_model="m",
        model="m",
        provider="p",
        status="fail",
        stream=False,
        tokens=0,
        error="upstream 400",
        request_id="cafe1234beef",
    )
    assert log_id > 0

    rows = db_mod.list_request_logs(request_id="cafe12")
    assert [row["id"] for row in rows] == [log_id]
    assert db_mod.count_request_logs(request_id="cafe12") == 1
    assert db_mod.list_request_logs(request_id="deadbeef") == []
