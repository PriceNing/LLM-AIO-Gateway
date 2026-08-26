"""Tests for custom model source mechanism (auto vs manual).

Confirmed semantics (2026-08-26, after subagent review):
1. provider_models gets a `source` column (TEXT NOT NULL DEFAULT 'auto').
2. add_provider models array = manual seeds -> source='manual'.
   DB column DEFAULT 'auto' applies only to rows inserted by refresh discovery.
3. update_provider does NOT flip an existing model's source (merge/insert only;
   existing auto models stay auto even if passed again).
4. refresh_provider_models only deletes stale AUTO models; MANUAL models are
   kept even when absent from upstream. Manual model_name is NOT overwritten
   by refresh (UPDATE branch limited to source='auto').
5. Deleting a provider cascades to ALL its models (auto + manual).
6. A delete-model entry point exists (delete_provider_model).
"""
import sqlite3

import pytest

import app.database as db_mod
from app.database import (
    add_provider, get_provider, delete_provider, init_db,
    update_provider,
)
from app.services.discovery import refresh_provider_models


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Isolate each test in a throwaway SQLite file."""
    previous_path = db_mod.DB_PATH
    previous_initialized = db_mod._initialized
    db_mod._initialized = False
    init_db(str(tmp_path / "custom_source.db"))
    try:
        yield
    finally:
        db_mod.DB_PATH = previous_path
        db_mod._initialized = previous_initialized


def _provider(pid="prov-1", models=None):
    return {
        "id": pid,
        "name": pid,
        "provider_type": "openai",
        "api_base": "https://api.example/v1",
        "api_key": "sk-test",
        "enabled": True,
        "models": models or [],
    }


def _model_ids(provider):
    return {m["id"] for m in provider["models"]}


def _set_source(pid, model_id, source):
    with db_mod.get_db() as db:
        db.execute("UPDATE provider_models SET source = ? WHERE provider_id = ? AND model_id = ?",
                   (source, pid, model_id))


# --- Requirement 1: source column exists, defaults to 'auto' ------------------

def test_provider_models_have_source_column():
    """DB column exists with DEFAULT 'auto' (for refresh-inserted rows)."""
    with db_mod.get_db() as db:
        cols = {r[1] for r in db.execute("PRAGMA table_info(provider_models)").fetchall()}
    assert "source" in cols


def test_source_column_default_is_auto_in_schema():
    """A raw INSERT (like refresh does) without source gets 'auto'."""
    add_provider(_provider("p"))
    with db_mod.get_db() as db:
        db.execute(
            "INSERT INTO provider_models (provider_id, model_id, model_name, enabled) VALUES ('p','raw','Raw',1)"
        )
    p = get_provider("p")
    row = next(m for m in p["models"] if m["id"] == "raw")
    assert row["source"] == "auto"


# --- Requirement 2: add_provider models are manual seeds ----------------------

def test_add_provider_marks_models_manual():
    add_provider(_provider("p", [
        {"id": "custom-1", "name": "Custom 1", "enabled": True},
    ]))
    p = get_provider("p")
    assert p["models"][0]["id"] == "custom-1"
    assert p["models"][0]["source"] == "manual"


# --- Requirement 3: update_provider merge/insert, does NOT flip source --------

def test_update_provider_adds_new_model_as_manual():
    add_provider(_provider("p"))
    update_provider("p", {"models": [{"id": "new-manual", "name": "N", "enabled": True}]})
    p = get_provider("p")
    newm = next(m for m in p["models"] if m["id"] == "new-manual")
    assert newm["source"] == "manual"


def test_update_provider_does_not_flip_existing_auto_model():
    """Passing an existing auto model in update must NOT turn it manual."""
    add_provider(_provider("p"))
    with db_mod.get_db() as db:
        db.execute(
            "INSERT INTO provider_models (provider_id, model_id, model_name, enabled) VALUES ('p','auto-1','A',1)"
        )
    # re-pass the same auto model via update
    update_provider("p", {"models": [{"id": "auto-1", "name": "A", "enabled": True}]})
    p = get_provider("p")
    row = next(m for m in p["models"] if m["id"] == "auto-1")
    assert row["source"] == "auto"


def test_update_provider_merge_keeps_models_not_in_payload():
    """update_provider is merge/insert, never replaces: models not in the
    payload must survive (manual and auto alike)."""
    add_provider(_provider("p", [
        {"id": "manual-1", "name": "M", "enabled": True},
    ]))
    with db_mod.get_db() as db:
        db.execute(
            "INSERT INTO provider_models (provider_id, model_id, model_name, enabled) VALUES ('p','auto-1','A',1)"
        )
    # update only mentions auto-1; manual-1 omitted -> must stay
    update_provider("p", {"models": [{"id": "auto-1", "name": "A", "enabled": True}]})
    p = get_provider("p")
    assert _model_ids(p) == {"manual-1", "auto-1"}


# --- Requirement 3.5: explicit "make manual" opt-in ---------------------------
# Editing an auto model does NOT promote it. The caller must send
# source="manual" explicitly. Omitting source keeps the stored value, so the
# rule-3 "never flip" semantics stay intact for plain edits.

def test_update_provider_without_source_keeps_auto():
    """Renaming an auto model without source must leave it auto."""
    add_provider(_provider("p"))
    with db_mod.get_db() as db:
        db.execute(
            "INSERT INTO provider_models (provider_id, model_id, model_name, enabled) VALUES ('p','auto-1','Before',1)"
        )
    update_provider("p", {"models": [{"id": "auto-1", "name": "After", "enabled": True}]})
    p = get_provider("p")
    row = next(m for m in p["models"] if m["id"] == "auto-1")
    assert row["source"] == "auto"
    assert row["name"] == "After"          # the edit still applies


def test_update_provider_explicit_manual_promotes_auto():
    """Passing source='manual' promotes an auto model."""
    add_provider(_provider("p"))
    with db_mod.get_db() as db:
        db.execute(
            "INSERT INTO provider_models (provider_id, model_id, model_name, enabled) VALUES ('p','auto-1','A',1)"
        )
    update_provider("p", {"models": [
        {"id": "auto-1", "name": "Pinned", "enabled": True, "source": "manual"},
    ]})
    p = get_provider("p")
    row = next(m for m in p["models"] if m["id"] == "auto-1")
    assert row["source"] == "manual"
    assert row["name"] == "Pinned"


def test_update_provider_explicit_auto_does_not_demote_manual():
    """source='auto' must never demote a manual model (only 'manual' acts)."""
    add_provider(_provider("p", [{"id": "m1", "name": "M", "enabled": True}]))
    update_provider("p", {"models": [
        {"id": "m1", "name": "M", "enabled": True, "source": "auto"},
    ]})
    p = get_provider("p")
    row = next(m for m in p["models"] if m["id"] == "m1")
    assert row["source"] == "manual"


@pytest.mark.asyncio
async def test_promoted_model_survives_refresh(monkeypatch):
    """An auto model promoted to manual is no longer deleted by refresh."""
    add_provider(_provider("p"))
    with db_mod.get_db() as db:
        db.execute(
            "INSERT INTO provider_models (provider_id, model_id, model_name, enabled) VALUES ('p','preview-x','Preview',1)"
        )
    # promote it the way the UI checkbox does
    update_provider("p", {"models": [
        {"id": "preview-x", "name": "Preview", "enabled": True, "source": "manual"},
    ]})

    async def discover(_pid):
        return [{"id": "other", "name": "Other"}]

    monkeypatch.setattr("app.services.discovery.discover_models", discover)

    result = await refresh_provider_models("p")
    p = get_provider("p")
    ids = _model_ids(p)

    assert "preview-x" in ids              # survived: upstream no longer lists it
    assert "other" in ids
    assert result["removed"] == 0


@pytest.mark.asyncio
async def test_unpromoted_model_still_deleted_by_refresh(monkeypatch):
    """Counterpart: editing without the checkbox leaves it deletable."""
    add_provider(_provider("p"))
    with db_mod.get_db() as db:
        db.execute(
            "INSERT INTO provider_models (provider_id, model_id, model_name, enabled) VALUES ('p','preview-x','Preview',1)"
        )
    # plain rename, no source -> stays auto
    update_provider("p", {"models": [
        {"id": "preview-x", "name": "Renamed", "enabled": True},
    ]})

    async def discover(_pid):
        return [{"id": "other", "name": "Other"}]

    monkeypatch.setattr("app.services.discovery.discover_models", discover)

    result = await refresh_provider_models("p")
    ids = _model_ids(get_provider("p"))

    assert "preview-x" not in ids
    assert result["removed"] == 1


@pytest.mark.asyncio
async def test_promoted_model_name_not_overwritten_by_refresh(monkeypatch):
    """After promotion the upstream name must not clobber the local one."""
    add_provider(_provider("p"))
    with db_mod.get_db() as db:
        db.execute(
            "INSERT INTO provider_models (provider_id, model_id, model_name, enabled) VALUES ('p','shared','Upstream Name',1)"
        )
    update_provider("p", {"models": [
        {"id": "shared", "name": "My Label", "enabled": True, "source": "manual"},
    ]})

    async def discover(_pid):
        # upstream still lists it, with a different name
        return [{"id": "shared", "name": "Upstream Renamed"}]

    monkeypatch.setattr("app.services.discovery.discover_models", discover)

    await refresh_provider_models("p")
    row = next(m for m in get_provider("p")["models"] if m["id"] == "shared")
    assert row["source"] == "manual"
    assert row["name"] == "My Label"


# --- Requirement 4: refresh only deletes stale AUTO models --------------------

@pytest.mark.asyncio
async def test_refresh_keeps_manual_model_not_in_upstream(monkeypatch):
    add_provider(_provider("p", [
        {"id": "auto-model", "name": "Auto", "enabled": True},
        {"id": "hidden-manual", "name": "Hidden", "enabled": True},
    ]))
    _set_source("p", "hidden-manual", "manual")

    async def discover(_pid):
        return [
            {"id": "auto-model", "name": "Auto"},
            {"id": "brand-new", "name": "New"},
        ]

    monkeypatch.setattr("app.services.discovery.discover_models", discover)

    result = await refresh_provider_models("p")
    p = get_provider("p")
    ids = _model_ids(p)

    assert "auto-model" in ids
    assert "hidden-manual" in ids          # kept despite absent from upstream
    assert "brand-new" in ids
    hidden = next(m for m in p["models"] if m["id"] == "hidden-manual")
    assert hidden["source"] == "manual"
    assert result["removed"] == 0


@pytest.mark.asyncio
async def test_refresh_still_deletes_stale_auto_model(monkeypatch):
    add_provider(_provider("p", [
        {"id": "gone-auto", "name": "Gone", "enabled": True},
    ]))
    # add_provider seeds are manual by design (rule 1), so flip this one to
    # auto to model a row that a previous refresh had discovered.
    _set_source("p", "gone-auto", "auto")

    async def discover(_pid):
        return [{"id": "still-here", "name": "Here"}]

    monkeypatch.setattr("app.services.discovery.discover_models", discover)

    result = await refresh_provider_models("p")
    p = get_provider("p")
    assert "gone-auto" not in _model_ids(p)
    assert "still-here" in _model_ids(p)
    assert result["removed"] == 1


@pytest.mark.asyncio
async def test_refresh_mixed_deletes_auto_keeps_manual_counts(monkeypatch):
    """Mixed: one stale auto removed + one manual kept -> removed counts only auto."""
    add_provider(_provider("p", [
        {"id": "gone-auto", "name": "Gone", "enabled": True},
        {"id": "kept-manual", "name": "Kept", "enabled": True},
    ]))
    # This model was seeded by add_provider (manual by rule 1), so flip to
    # auto to simulate a row that a previous refresh discovered.
    _set_source("p", "gone-auto", "auto")
    _set_source("p", "kept-manual", "manual")

    async def discover(_pid):
        return [{"id": "upstream-only", "name": "U"}]

    monkeypatch.setattr("app.services.discovery.discover_models", discover)

    result = await refresh_provider_models("p")
    p = get_provider("p")
    assert _model_ids(p) == {"kept-manual", "upstream-only"}
    assert result["removed"] == 1


@pytest.mark.asyncio
async def test_refresh_does_not_overwrite_manual_model_name(monkeypatch):
    """If a manual model id also appears upstream, refresh must NOT rename it."""
    add_provider(_provider("p", [
        {"id": "shared-id", "name": "My Custom Name", "enabled": True},
    ]))
    _set_source("p", "shared-id", "manual")

    async def discover(_pid):
        return [{"id": "shared-id", "name": "Upstream Name"}]

    monkeypatch.setattr("app.services.discovery.discover_models", discover)

    await refresh_provider_models("p")
    p = get_provider("p")
    row = next(m for m in p["models"] if m["id"] == "shared-id")
    assert row["name"] == "My Custom Name"
    assert row["source"] == "manual"


@pytest.mark.asyncio
async def test_refresh_empty_discovered_keeps_everything(monkeypatch):
    """Empty upstream list -> nothing deleted (safe guard), auto kept too."""
    add_provider(_provider("p", [
        {"id": "auto-1", "name": "A", "enabled": True},
        {"id": "manual-1", "name": "M", "enabled": True},
    ]))
    _set_source("p", "manual-1", "manual")

    async def discover(_pid):
        return []

    monkeypatch.setattr("app.services.discovery.discover_models", discover)

    result = await refresh_provider_models("p")
    p = get_provider("p")
    assert _model_ids(p) == {"auto-1", "manual-1"}
    assert result["removed"] == 0


@pytest.mark.asyncio
async def test_refresh_error_path_keeps_existing(monkeypatch):
    """discover raises -> refresh reports error, removes nothing."""
    add_provider(_provider("p", [
        {"id": "auto-1", "name": "A", "enabled": True},
    ]))

    async def discover(_pid):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.services.discovery.discover_models", discover)

    result = await refresh_provider_models("p")
    p = get_provider("p")
    assert _model_ids(p) == {"auto-1"}
    assert result["removed"] == 0
    assert "boom" in result.get("error", "")


@pytest.mark.asyncio
async def test_refresh_provider_isolation(monkeypatch):
    """Refreshing provider A must not touch provider B's manual model."""
    add_provider(_provider("a", [{"id": "m-a", "name": "MA", "enabled": True}]))
    add_provider(_provider("b", [{"id": "m-b", "name": "MB", "enabled": True}]))
    _set_source("b", "m-b", "manual")

    async def discover(pid):
        return [] if pid == "a" else [{"id": "m-b", "name": "MB"}]

    monkeypatch.setattr("app.services.discovery.discover_models", discover)

    await refresh_provider_models("a")
    b = get_provider("b")
    assert _model_ids(b) == {"m-b"}
    assert next(m for m in b["models"] if m["id"] == "m-b")["source"] == "manual"


# --- Requirement 5: deleting provider cascades to all models ------------------

def test_delete_provider_cascades_auto_and_manual_models():
    add_provider(_provider("p", [
        {"id": "auto-1", "name": "A", "enabled": True},
        {"id": "manual-1", "name": "M", "enabled": True},
    ]))
    _set_source("p", "manual-1", "manual")

    assert delete_provider("p") is True
    with db_mod.get_db() as db:
        remaining = db.execute(
            "SELECT COUNT(*) AS c FROM provider_models WHERE provider_id = 'p'"
        ).fetchone()["c"]
    assert remaining == 0


# --- Requirement 6: delete a single manual/auto model -------------------------

def test_delete_provider_model_removes_single_model():
    from app.database import delete_provider_model
    add_provider(_provider("p", [
        {"id": "auto-1", "name": "A", "enabled": True},
        {"id": "manual-1", "name": "M", "enabled": True},
    ]))
    _set_source("p", "manual-1", "manual")

    assert delete_provider_model("p", "manual-1") is True
    p = get_provider("p")
    assert _model_ids(p) == {"auto-1"}


def test_delete_provider_model_missing_returns_false():
    from app.database import delete_provider_model
    add_provider(_provider("p"))
    assert delete_provider_model("p", "nope") is False


# --- Requirement 1e: legacy DB migration --------------------------------------

def test_legacy_db_migration_adds_source_column():
    """A DB created before source existed gets the column; existing rows = auto."""
    import tempfile, os
    tmp = tempfile.mktemp(suffix=".db")
    try:
        # Build a legacy schema WITHOUT source, insert a row, then run init_db
        legacy_conn = sqlite3.connect(tmp)
        legacy_conn.execute("""
            CREATE TABLE providers (
                id TEXT PRIMARY KEY, name TEXT, provider_type TEXT, api_base TEXT,
                api_key TEXT, enabled INTEGER, extra_headers TEXT, request_timeout INTEGER,
                retry_count INTEGER, retry_backoff REAL, force_chat_completions INTEGER
            )
        """)
        legacy_conn.execute("""
            CREATE TABLE provider_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_name TEXT NOT NULL DEFAULT '',
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT '',
                preprocessor TEXT NOT NULL DEFAULT '',
                image_generation TEXT NOT NULL DEFAULT '',
                responses_status TEXT NOT NULL DEFAULT 'unknown',
                responses_checked_at TEXT NOT NULL DEFAULT '',
                responses_expires_at TEXT NOT NULL DEFAULT '',
                responses_streaming INTEGER NOT NULL DEFAULT 0,
                responses_streaming_status TEXT NOT NULL DEFAULT 'unknown',
                responses_tool_types TEXT NOT NULL DEFAULT '[]',
                responses_error TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE,
                UNIQUE(provider_id, model_id)
            )
        """)
        legacy_conn.execute(
            "INSERT INTO providers (id,name,provider_type,api_base,api_key,enabled,extra_headers,request_timeout,retry_count,retry_backoff,force_chat_completions) VALUES ('legacy','L','openai','', '',1,'{}',120,0,0.5,0)"
        )
        legacy_conn.execute(
            "INSERT INTO provider_models (provider_id,model_id,model_name,enabled) VALUES ('legacy','m1','M1',1)"
        )
        legacy_conn.commit()
        legacy_conn.close()

        # init_db on this legacy file runs migrations
        init_db(tmp)
        # Use a fresh standalone connection (not the module pool) so the file
        # handle is fully closed before os.remove on Windows.
        mig_conn = sqlite3.connect(tmp)
        try:
            cols = {r[1] for r in mig_conn.execute("PRAGMA table_info(provider_models)").fetchall()}
            assert "source" in cols
            src = mig_conn.execute("SELECT source FROM provider_models WHERE model_id='m1'").fetchone()[0]
            assert src == "auto"
        finally:
            mig_conn.close()
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except PermissionError:
                pass
