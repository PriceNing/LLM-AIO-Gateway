import sqlite3
import json
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from app.db import fallback as fallback_db
from app.db import request_logs as request_logs_db
from app.db import routing as routing_db

_lock = threading.Lock()
_initialized = False

DB_PATH: str = "data.db"

# -- Connection management --

def _db_path() -> str:
    return DB_PATH


def init_db(path: Optional[str] = None) -> None:
    """Initialize database: create tables if not exist, enable WAL."""
    global DB_PATH, _initialized
    if path:
        DB_PATH = path
    db_file = Path(_db_path())
    if str(db_file) != ":memory:" and db_file.parent != Path("."):
        db_file.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with sqlite3.connect(_db_path()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA)
            # Migration: add created_at to provider_models if missing
            _migrate_provider_models_created_at(conn)
            # Migration: add extra_headers to providers if missing
            _migrate_providers_extra_headers(conn)
            _migrate_provider_request_options(conn)
            _migrate_provider_force_chat_completions(conn)
            _remove_legacy_provider_responses_capability(conn)
            _migrate_model_responses_capability(conn)
            _migrate_preprocessors(conn)
            _migrate_image_generation(conn)
            _migrate_provider_models_source(conn)
            _migrate_request_records_image_generation(conn)
            _migrate_image_generation_stats(conn)
            # Routing and fallback policy migrations.
            routing_db.migrate(conn)
            fallback_db.migrate(conn)
            request_logs_db.migrate(conn)
        _initialized = True


def _migrate_provider_models_created_at(conn: sqlite3.Connection) -> None:
    """Add created_at and preprocessor columns to provider_models if missing."""
    for col, default in (("created_at", "''"), ("preprocessor", "''")):
        try:
            conn.execute(f"ALTER TABLE provider_models ADD COLUMN {col} TEXT NOT NULL DEFAULT {default}")
        except sqlite3.OperationalError:
            pass  # Column already exists


def _migrate_providers_extra_headers(conn: sqlite3.Connection) -> None:
    """Add extra_headers column to providers if missing, and initialize DeepSeek defaults."""
    try:
        conn.execute("ALTER TABLE providers ADD COLUMN extra_headers TEXT NOT NULL DEFAULT '{}'")
    except sqlite3.OperationalError:
        pass  # Column already exists
    # Initialize extra_headers for existing DeepSeek providers that have empty value
    rows = conn.execute(
        "SELECT id, extra_headers FROM providers WHERE extra_headers IS NULL OR extra_headers = '{}'"
    ).fetchall()
    for row in rows:
        pid = row[0]
        name_row = conn.execute("SELECT name FROM providers WHERE id = ?", (pid,)).fetchone()
        provider_name = (name_row[0] if name_row else "").lower()
        if "deepseek" in pid.lower() or "deepseek" in provider_name:
            conn.execute(
                "UPDATE providers SET extra_headers = ? WHERE id = ?",
                ('{"thinking": "enabled"}', pid)
            )


def _migrate_provider_request_options(conn: sqlite3.Connection) -> None:
    """Add per-provider upstream timeout/retry options for existing databases."""
    columns = {
        "request_timeout": "INTEGER NOT NULL DEFAULT 120",
        "retry_count": "INTEGER NOT NULL DEFAULT 0",
        "retry_backoff": "REAL NOT NULL DEFAULT 0.5",
    }
    existing = {row[1] for row in conn.execute("PRAGMA table_info(providers)").fetchall()}
    for col, ddl in columns.items():
        if col not in existing:
            try:
                conn.execute(f"ALTER TABLE providers ADD COLUMN {col} {ddl}")
            except sqlite3.OperationalError:
                pass


def _remove_legacy_provider_responses_capability(conn: sqlite3.Connection) -> None:
    """Drop obsolete provider-level Responses cache columns from existing databases."""
    legacy_columns = (
        "responses_status",
        "responses_checked_at",
        "responses_streaming",
        "responses_streaming_status",
        "responses_tool_types",
        "responses_error",
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(providers)").fetchall()}
    for column in legacy_columns:
        if column in existing:
            conn.execute(f"ALTER TABLE providers DROP COLUMN {column}")


def _migrate_model_responses_capability(conn: sqlite3.Connection) -> None:
    """Store native Responses capability per provider/model, never per provider."""
    columns = {
        "responses_status": "TEXT NOT NULL DEFAULT 'unknown'",
        "responses_checked_at": "TEXT NOT NULL DEFAULT ''",
        "responses_expires_at": "TEXT NOT NULL DEFAULT ''",
        "responses_streaming": "INTEGER NOT NULL DEFAULT 0",
        "responses_streaming_status": "TEXT NOT NULL DEFAULT 'unknown'",
        "responses_tool_types": "TEXT NOT NULL DEFAULT '[]'",
        "responses_error": "TEXT NOT NULL DEFAULT ''",
    }
    existing = {row[1] for row in conn.execute("PRAGMA table_info(provider_models)").fetchall()}
    for col, ddl in columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE provider_models ADD COLUMN {col} {ddl}")


def _migrate_preprocessors(conn: sqlite3.Connection) -> None:
    """Add preprocessor columns if an older database created the table partially."""
    columns = {
        "api_base": "TEXT NOT NULL DEFAULT ''",
        "model": "TEXT NOT NULL DEFAULT ''",
        "api_key": "TEXT NOT NULL DEFAULT ''",
        "timeout": "INTEGER NOT NULL DEFAULT 120",
        "max_images": "INTEGER NOT NULL DEFAULT 10",
        "prompt": "TEXT NOT NULL DEFAULT ''",
        "enabled": "INTEGER NOT NULL DEFAULT 1",
        "max_tokens": "INTEGER NOT NULL DEFAULT 2048",
        "created_at": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    }
    existing = {row[1] for row in conn.execute("PRAGMA table_info(preprocessors)").fetchall()}
    for col, ddl in columns.items():
        if col not in existing:
            try:
                conn.execute(f"ALTER TABLE preprocessors ADD COLUMN {col} {ddl}")
            except sqlite3.OperationalError:
                pass


def _migrate_provider_models_source(conn: sqlite3.Connection) -> None:
    """Add the model source column (auto vs manual) for existing databases.

    Existing rows default to 'auto' so refresh keeps its historical behaviour
    for everything that was discovered from an upstream /models endpoint.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(provider_models)").fetchall()}
    if "source" not in existing:
        try:
            conn.execute("ALTER TABLE provider_models ADD COLUMN source TEXT NOT NULL DEFAULT 'auto'")
        except sqlite3.OperationalError:
            pass


def _migrate_provider_force_chat_completions(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(providers)").fetchall()}
    if "force_chat_completions" not in existing:
        conn.execute("ALTER TABLE providers ADD COLUMN force_chat_completions INTEGER NOT NULL DEFAULT 0")


def _migrate_image_generation(conn: sqlite3.Connection) -> None:
    """Add independent image-generation settings and per-model capability flags."""
    existing_models = {row[1] for row in conn.execute("PRAGMA table_info(provider_models)").fetchall()}
    if "image_generation" not in existing_models:
        try:
            conn.execute("ALTER TABLE provider_models ADD COLUMN image_generation TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_generators (
            id TEXT PRIMARY KEY,
            backend_type TEXT NOT NULL DEFAULT 'existing_model',
            provider_model TEXT NOT NULL DEFAULT '',
            api_base TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            api_key TEXT NOT NULL DEFAULT '',
            timeout INTEGER NOT NULL DEFAULT 180,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT ''
        )
    """)
    generator_columns = {row[1] for row in conn.execute("PRAGMA table_info(image_generators)").fetchall()}
    for column, ddl in {
        "workflow": "TEXT NOT NULL DEFAULT '{}'",
        "workflow_mapping": "TEXT NOT NULL DEFAULT '{}'",
        "poll_interval": "REAL NOT NULL DEFAULT 1.0",
    }.items():
        if column not in generator_columns:
            conn.execute(f"ALTER TABLE image_generators ADD COLUMN {column} {ddl}")


def _migrate_request_records_image_generation(conn: sqlite3.Connection) -> None:
    """Add image-generation dimensions to historical request statistics."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(request_records)").fetchall()}
    columns = {
        "request_kind": "TEXT NOT NULL DEFAULT ''",
        "image_model": "TEXT NOT NULL DEFAULT ''",
        "image_count": "INTEGER NOT NULL DEFAULT 0",
        "image_bytes": "INTEGER NOT NULL DEFAULT 0",
    }
    for column, ddl in columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE request_records ADD COLUMN {column} {ddl}")


def _migrate_image_generation_stats(conn: sqlite3.Connection) -> None:
    """Initialize durable image counters from retained request logs once."""
    keys = {
        "image_generation_calls",
        "image_generation_failed_calls",
        "image_generation_images",
        "image_generation_bytes",
    }
    existing = {
        row[0] for row in conn.execute(
            "SELECT key FROM global_stats WHERE key IN (?, ?, ?, ?)", tuple(sorted(keys))
        ).fetchall()
    }
    if not keys.issubset(existing):
        calls = failures = images = image_bytes = 0
        for row in conn.execute("SELECT status, details FROM request_logs").fetchall():
            try:
                details = json.loads(row[1] or "{}")
            except (TypeError, json.JSONDecodeError):
                details = {}
            mode = str(details.get("responses_mode") or "")
            is_image = (
                details.get("request_kind") == "image_generation"
                or details.get("upstream_endpoint") == "images/generations"
                or "image_generation" in mode
            )
            if not is_image:
                continue
            calls += 1
            failures += 0 if row[0] in {"ok", "degraded"} else 1
            try:
                images += max(0, int(details.get("image_count") or 0))
                image_bytes += max(0, int(details.get("image_bytes") or 0))
            except (TypeError, ValueError):
                pass
        initial = {
            "image_generation_calls": calls,
            "image_generation_failed_calls": failures,
            "image_generation_images": images,
            "image_generation_bytes": image_bytes,
        }
    else:
        initial = {}
    for key in sorted(keys):
        if key in initial:
            conn.execute(
                "INSERT OR IGNORE INTO global_stats (key, value) VALUES (?, ?)",
                (key, str(initial[key])),
            )


def _ensure_init() -> None:
    if not _initialized:
        init_db()


@contextmanager
def get_db():
    """Get a database connection with WAL mode enabled."""
    _ensure_init()
    conn = sqlite3.connect(_db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# -- Schema --

_SCHEMA = """
CREATE TABLE IF NOT EXISTS admins (
    username TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    total_calls INTEGER NOT NULL DEFAULT 0,
    failed_calls INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS user_api_keys (
    key TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT 'default',
    allowed_models TEXT NOT NULL DEFAULT '["*"]',
    enabled INTEGER NOT NULL DEFAULT 1,
    total_calls INTEGER NOT NULL DEFAULT 0,
    failed_calls INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider_type TEXT NOT NULL DEFAULT 'openai',
    api_base TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    extra_headers TEXT NOT NULL DEFAULT '{}',
    request_timeout INTEGER NOT NULL DEFAULT 120,
    retry_count INTEGER NOT NULL DEFAULT 0,
    retry_backoff REAL NOT NULL DEFAULT 0.5,
    force_chat_completions INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS provider_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT '',
    preprocessor TEXT NOT NULL DEFAULT '',
    image_generation TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'auto',
    responses_status TEXT NOT NULL DEFAULT 'unknown',
    responses_checked_at TEXT NOT NULL DEFAULT '',
    responses_expires_at TEXT NOT NULL DEFAULT '',
    responses_streaming INTEGER NOT NULL DEFAULT 0,
    responses_streaming_status TEXT NOT NULL DEFAULT 'unknown',
    responses_tool_types TEXT NOT NULL DEFAULT '[]',
    responses_error TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE,
    UNIQUE(provider_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_provider_models_model_id ON provider_models(model_id);

CREATE TABLE IF NOT EXISTS preprocessors (
    id TEXT PRIMARY KEY,
    api_base TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    timeout INTEGER NOT NULL DEFAULT 120,
    max_images INTEGER NOT NULL DEFAULT 10,
    prompt TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    max_tokens INTEGER NOT NULL DEFAULT 2048,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS image_generators (
    id TEXT PRIMARY KEY,
    backend_type TEXT NOT NULL DEFAULT 'existing_model',
    provider_model TEXT NOT NULL DEFAULT '',
    api_base TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    timeout INTEGER NOT NULL DEFAULT 180,
    workflow TEXT NOT NULL DEFAULT '{}',
    workflow_mapping TEXT NOT NULL DEFAULT '{}',
    poll_interval REAL NOT NULL DEFAULT 1.0,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS routing_rules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'New Rule',
    enabled INTEGER NOT NULL DEFAULT 1,
    username TEXT NOT NULL DEFAULT '',
    api_key_pattern TEXT NOT NULL DEFAULT '',
    match_model TEXT NOT NULL DEFAULT '',
    match_scope TEXT NOT NULL DEFAULT 'any',
    target_model TEXT NOT NULL DEFAULT '',
    target_provider TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS fallback_policies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'New Fallback Policy',
    enabled INTEGER NOT NULL DEFAULT 1,
    match_provider TEXT NOT NULL DEFAULT '',
    match_model TEXT NOT NULL DEFAULT '*',
    triggers TEXT NOT NULL DEFAULT '{}',
    chain TEXT NOT NULL DEFAULT '[]',
    attempt_timeout INTEGER NOT NULL DEFAULT 60,
    created_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS global_stats (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT '0'
);

CREATE TABLE IF NOT EXISTS request_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    username TEXT NOT NULL DEFAULT '',
    success INTEGER NOT NULL DEFAULT 1,
    tokens INTEGER NOT NULL DEFAULT 0,
    request_kind TEXT NOT NULL DEFAULT '',
    image_model TEXT NOT NULL DEFAULT '',
    image_count INTEGER NOT NULL DEFAULT 0,
    image_bytes INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_req_ts ON request_records(timestamp);
CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    username TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    requested_model TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    stream INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0,
    request_body TEXT,
    response_body TEXT,
    details TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_reqlog_ts ON request_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_reqlog_endpoint ON request_logs(endpoint);
"""

# -- Helpers --

def _row_to_dict(row: sqlite3.Row) -> dict:
    if row is None:
        return None
    return dict(row)


def _json_loads(s: str):
    if not s:
        return None  # empty string is not valid JSON - treated as "not configured" by callers
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes")
    return False


def _clamp_int(value, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def _clamp_float(value, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


# -- Global stats --

def get_global_stats() -> dict:
    with get_db() as db:
        rows = db.execute("SELECT key, value FROM global_stats").fetchall()
        result = {}
        for r in rows:
            v = r["value"]
            result[r["key"]] = int(v) if v.lstrip("-").isdigit() else v
        return result


def increment_global_stats(
    success: bool,
    *,
    degraded: bool = False,
    rejected: bool = False,
    cancelled: bool = False,
    stateful_fallback_blocked: bool = False,
) -> None:
    """Increment global counters.

    - total_calls always +1
    - failed_calls +1 when success is False
    - degraded_calls +1 when success is True and degraded is True
    - rejected_calls +1 when auth/allow-list rejected
    - cancelled_calls +1 when client disconnected
    """
    with get_db() as db:
        for key in (
            "total_calls",
            "failed_calls",
            "degraded_calls",
            "rejected_calls",
            "cancelled_calls",
            "stateful_fallback_blocked_calls",
            "last_reset",
        ):
            default = "" if key == "last_reset" else "0"
            db.execute(
                "INSERT OR IGNORE INTO global_stats (key, value) VALUES (?, ?)",
                (key, default),
            )
        db.execute("UPDATE global_stats SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key = 'total_calls'")
        if not success:
            db.execute("UPDATE global_stats SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key = 'failed_calls'")
            if rejected:
                db.execute(
                    "UPDATE global_stats SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key = 'rejected_calls'"
                )
            if cancelled:
                db.execute(
                    "UPDATE global_stats SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key = 'cancelled_calls'"
                )
        elif degraded:
            db.execute("UPDATE global_stats SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key = 'degraded_calls'")
        if stateful_fallback_blocked:
            db.execute("UPDATE global_stats SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key = 'stateful_fallback_blocked_calls'")


def increment_image_generation_stats(success: bool, image_count: int = 0, image_bytes: int = 0) -> None:
    """Increment image-generation counters once per completed gateway request."""
    increments = {
        "image_generation_calls": 1,
        "image_generation_failed_calls": 0 if success else 1,
        "image_generation_images": max(0, int(image_count or 0)),
        "image_generation_bytes": max(0, int(image_bytes or 0)),
    }
    with get_db() as db:
        for key, amount in increments.items():
            db.execute(
                "INSERT OR IGNORE INTO global_stats (key, value) VALUES (?, '0')",
                (key,),
            )
            if amount:
                db.execute(
                    "UPDATE global_stats SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT) WHERE key = ?",
                    (amount, key),
                )


def reset_global_stats() -> None:
    today = date.today().isoformat()
    with get_db() as db:
        for key in (
            "total_calls",
            "failed_calls",
            "degraded_calls",
            "rejected_calls",
            "cancelled_calls",
            "stateful_fallback_blocked_calls",
            "image_generation_calls",
            "image_generation_failed_calls",
            "image_generation_images",
            "image_generation_bytes",
        ):
            db.execute(
                "INSERT OR REPLACE INTO global_stats (key, value) VALUES (?, '0')",
                (key,),
            )
        db.execute("INSERT OR REPLACE INTO global_stats (key, value) VALUES ('last_reset', ?)", (today,))


# -- Admins --

def get_admins() -> list:
    with get_db() as db:
        rows = db.execute("SELECT * FROM admins ORDER BY created_at").fetchall()
        return [_row_to_dict(r) for r in rows]


def get_admin(username: str) -> Optional[dict]:
    with get_db() as db:
        row = db.execute("SELECT * FROM admins WHERE username = ?", (username,)).fetchone()
        return _row_to_dict(row)


def add_admin(username: str, password_hash: str, display_name: str = "") -> dict:
    today = date.today().isoformat()
    with get_db() as db:
        try:
            db.execute(
                "INSERT INTO admins (username, display_name, password_hash, enabled, created_at) VALUES (?, ?, ?, 1, ?)",
                (username, display_name or username, password_hash, today)
            )
        except sqlite3.IntegrityError:
            raise ValueError("Admin already exists")
    return {"username": username, "display_name": display_name or username, "password_hash": password_hash, "enabled": True, "created_at": today}


def update_admin_password(username: str, password_hash: str) -> bool:
    with get_db() as db:
        db.execute(
            "UPDATE admins SET password_hash = ? WHERE username = ?",
            (password_hash, username)
        )
        return db.total_changes > 0


# -- Users --

def _api_key_from_row(k: sqlite3.Row) -> dict:
    kd = _row_to_dict(k)
    kd["allowed_models"] = _json_loads(kd.get("allowed_models", '["*"]'))
    kd["stats"] = {"total_calls": kd.pop("total_calls", 0), "failed_calls": kd.pop("failed_calls", 0), "total_tokens": kd.pop("total_tokens", 0)}
    kd["enabled"] = _to_bool(kd.get("enabled"))
    return kd


def _user_from_row(r: sqlite3.Row) -> dict:
    user = _row_to_dict(r)
    user["stats"] = {"total_calls": user.pop("total_calls", 0), "failed_calls": user.pop("failed_calls", 0), "total_tokens": user.pop("total_tokens", 0)}
    user["enabled"] = _to_bool(user.get("enabled"))
    user["api_keys"] = []
    return user


def get_users() -> list:
    with get_db() as db:
        users_rows = db.execute("SELECT username, display_name, enabled, total_calls, failed_calls, total_tokens, created_at FROM users ORDER BY created_at").fetchall()
        keys_rows = db.execute("SELECT * FROM user_api_keys ORDER BY username, created_at").fetchall()
        keys_by_user: dict[str, list] = {}
        for k in keys_rows:
            uname = k["username"]
            keys_by_user.setdefault(uname, []).append(_api_key_from_row(k))
        result = []
        for r in users_rows:
            user = _user_from_row(r)
            user["api_keys"] = keys_by_user.get(user["username"], [])
            result.append(user)
        return result


def get_user(username: str) -> Optional[dict]:
    with get_db() as db:
        row = db.execute("SELECT username, display_name, enabled, total_calls, failed_calls, total_tokens, created_at FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return None
        user = _user_from_row(row)
        keys = db.execute("SELECT * FROM user_api_keys WHERE username = ? ORDER BY created_at", (username,)).fetchall()
        user["api_keys"] = [_api_key_from_row(k) for k in keys]
        return user


def add_user(user_info: dict) -> dict:
    username = user_info.get("username", "").strip()
    if not username:
        raise ValueError("username is required")
    today = date.today().isoformat()
    with get_db() as db:
        existing = db.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            raise ValueError("User already exists")
        db.execute(
            "INSERT INTO users (username, display_name, enabled, created_at) VALUES (?, ?, ?, ?)",
            (username, user_info.get("display_name") or username, 1 if user_info.get("enabled", True) else 0, today)
        )
    return get_user(username)


def update_user(username: str, updates: dict) -> Optional[dict]:
    with get_db() as db:
        existing = db.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if not existing:
            return None
        if "display_name" in updates:
            db.execute("UPDATE users SET display_name = ? WHERE username = ?", (updates["display_name"], username))
        if "enabled" in updates:
            db.execute("UPDATE users SET enabled = ? WHERE username = ?", (1 if updates["enabled"] else 0, username))
    return get_user(username)


def delete_user(username: str) -> bool:
    with get_db() as db:
        cursor = db.execute("DELETE FROM users WHERE username = ?", (username,))
        return cursor.rowcount > 0


# -- API Keys --

def add_user_api_key(username: str, name: str, allowed_models: Optional[list] = None) -> dict:
    from app.security import new_api_key
    today = date.today().isoformat()
    allowed = allowed_models if allowed_models is not None else ["*"]
    key = new_api_key()
    with get_db() as db:
        existing = db.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if not existing:
            raise ValueError("User not found")
        db.execute(
            "INSERT INTO user_api_keys (key, username, name, allowed_models, enabled, created_at) VALUES (?, ?, ?, ?, 1, ?)",
            (key, username, name or "default", json.dumps(allowed), today)
        )
    return {"key": key, "name": name or "default", "allowed_models": allowed, "created_at": today, "enabled": True, "stats": {"total_calls": 0, "failed_calls": 0, "total_tokens": 0}}


def update_user_api_key(username: str, key: str, updates: dict) -> Optional[dict]:
    with get_db() as db:
        row = db.execute("SELECT * FROM user_api_keys WHERE key = ? AND username = ?", (key, username)).fetchone()
        if not row:
            return None
        if "name" in updates:
            db.execute("UPDATE user_api_keys SET name = ? WHERE key = ?", (updates["name"], key))
        if "allowed_models" in updates:
            db.execute("UPDATE user_api_keys SET allowed_models = ? WHERE key = ?", (json.dumps(updates["allowed_models"]), key))
        if "enabled" in updates:
            db.execute("UPDATE user_api_keys SET enabled = ? WHERE key = ?", (1 if updates["enabled"] else 0, key))
        row2 = db.execute("SELECT * FROM user_api_keys WHERE key = ?", (key,)).fetchone()
        kd = _row_to_dict(row2)
        kd["allowed_models"] = _json_loads(kd.get("allowed_models", '["*"]'))
        kd["stats"] = {"total_calls": kd.pop("total_calls", 0), "failed_calls": kd.pop("failed_calls", 0), "total_tokens": kd.pop("total_tokens", 0)}
        kd["enabled"] = _to_bool(kd.get("enabled"))
        return kd


def delete_user_api_key(username: str, key: str) -> bool:
    with get_db() as db:
        cursor = db.execute("DELETE FROM user_api_keys WHERE key = ? AND username = ?", (key, username))
        return cursor.rowcount > 0


# -- Find user by API key --

def find_user_by_api_key(key: str) -> Optional[tuple[dict, dict]]:
    with get_db() as db:
        row = db.execute("""
            SELECT u.username, u.display_name, u.enabled as user_enabled,
                   k.key, k.name, k.allowed_models, k.enabled as key_enabled, k.total_calls, k.failed_calls, k.total_tokens, k.created_at
            FROM users u
            JOIN user_api_keys k ON k.username = u.username
            WHERE k.key = ?
        """, (key,)).fetchone()
        if not row:
            return None
        r = dict(row)
        if not r.get("user_enabled") or not r.get("key_enabled"):
            return None
        user = {"username": r["username"], "display_name": r["display_name"], "enabled": bool(r["user_enabled"])}
        api_key = {
            "key": r["key"], "name": r["name"],
            "allowed_models": _json_loads(r["allowed_models"]),
            "enabled": bool(r["key_enabled"]),
            "stats": {"total_calls": r["total_calls"], "failed_calls": r["failed_calls"], "total_tokens": r["total_tokens"]},
            "created_at": r["created_at"]
        }
        return user, api_key


# -- Increment usage stats --

# -- Request history records --

def add_request_record(
    model: str,
    username: str,
    success: bool,
    tokens: int = 0,
    *,
    request_kind: str = "",
    image_model: str = "",
    image_count: int = 0,
    image_bytes: int = 0,
) -> None:
    """Insert a request record for historical stats. Called from _log_request."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as db:
        db.execute(
            """INSERT INTO request_records (
                   timestamp, model, username, success, tokens,
                   request_kind, image_model, image_count, image_bytes
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now, model, username, 1 if success else 0, tokens,
                request_kind or "", image_model or "",
                max(0, int(image_count or 0)), max(0, int(image_bytes or 0)),
            ),
        )


_HISTORY_GRANULARITY = {
    "hour":  "%Y-%m-%d %H:00",
    "day":   "%Y-%m-%d",
    "week":  "%Y-%W",
    "month": "%Y-%m",
}

_HISTORY_DELTA = {"hour": timedelta(hours=1), "day": timedelta(days=1),
                   "week": timedelta(weeks=1), "month": timedelta(days=31)}

_HISTORY_STEP_FMT = {"hour": "%Y-%m-%d %H:00", "day": "%Y-%m-%d",
                      "week": "%Y-%U", "month": "%Y-%m"}


def _local_tz():
    return datetime.now().astimezone().tzinfo


def _parse_history_boundary(ts: str, end_of_day: bool = False) -> datetime:
    value = str(ts or "").strip().replace("T", " ")
    if len(value) == 10:
        value += " 23:59:59" if end_of_day else " 00:00:00"
    fmt = "%Y-%m-%d %H:%M:%S" if len(value) >= 19 else "%Y-%m-%d %H:%M"
    parsed = datetime.strptime(value[:19] if fmt.endswith("%S") else value[:16], fmt)
    return parsed.replace(tzinfo=_local_tz())


def _history_query_bounds(from_ts: str, to_ts: str) -> tuple[str, str, datetime, datetime]:
    start_local = _parse_history_boundary(from_ts, end_of_day=False)
    end_local = _parse_history_boundary(to_ts, end_of_day=True)
    start_utc = start_local.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    end_utc = end_local.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return start_utc, end_utc, start_local, end_local


def _bucket_for_timestamp(timestamp: str, granularity: str) -> str:
    dt_utc = datetime.strptime(timestamp[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    local_dt = dt_utc.astimezone(_local_tz())
    return local_dt.strftime(_HISTORY_STEP_FMT[granularity])


def _zero_pad_timeline(rows, from_ts, to_ts, granularity, model_bucket_rows):
    """Fill in missing buckets so the timeline has no gaps."""
    step_fmt = _HISTORY_STEP_FMT[granularity]
    delta = _HISTORY_DELTA[granularity]

    start = _parse_history_boundary(from_ts, end_of_day=False)
    end = _parse_history_boundary(to_ts, end_of_day=True)

    # Build a dict from bucket -> row data
    row_map = {r["bucket"] or "": r for r in rows}
    model_map = {}
    for r in model_bucket_rows:
        model_map.setdefault(r["bucket"] or "", []).append(r)

    all_buckets = []
    cur = start
    while cur <= end:
        b = cur.strftime(step_fmt)
        all_buckets.append(b)
        if granularity == "month":
            # Advance to the first day of the next month; avoid day overflow
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1, day=1)
            else:
                cur = cur.replace(month=cur.month + 1, day=1)
        else:
            cur += delta

    padded_rows = []
    padded_model_rows = []
    for b in all_buckets:
        existing = row_map.get(b)
        if existing:
            padded_rows.append(existing)
        else:
            padded_rows.append({
                "bucket": b, "total": 0, "failed": 0, "tokens": 0,
                "image_calls": 0, "image_failed": 0, "images": 0, "image_bytes": 0,
            })
        for mr in model_map.get(b, []):
            padded_model_rows.append(mr)
        # Missing bucket -> no model rows needed (all zeros)

    return padded_rows, all_buckets, padded_model_rows


def get_history_stats(from_ts: str, to_ts: str, granularity: str = "day") -> dict:
    """Aggregate historical stats by granularity. Returns timeline + model breakdown."""
    from_query, to_query, _, _ = _history_query_bounds(from_ts, to_ts)
    with get_db() as db:
        # Timeline: total calls, failures, tokens per bucket
        rows = db.execute("""
            SELECT timestamp,
                   COUNT(*) AS total,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failed,
                   SUM(tokens) AS tokens,
                   SUM(CASE WHEN request_kind = 'image_generation' THEN 1 ELSE 0 END) AS image_calls,
                   SUM(CASE WHEN request_kind = 'image_generation' AND success = 0 THEN 1 ELSE 0 END) AS image_failed,
                   SUM(image_count) AS images,
                   SUM(image_bytes) AS image_bytes
            FROM request_records
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY timestamp
            ORDER BY timestamp
        """, (from_query, to_query)).fetchall()

        # Model breakdown for the period
        model_rows = db.execute("""
            SELECT model, COUNT(*) AS total,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failed,
                   SUM(tokens) AS tokens
            FROM request_records
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY model
            ORDER BY total DESC
        """, (from_query, to_query)).fetchall()

        # User breakdown for the period
        user_rows = db.execute("""
            SELECT username, COUNT(*) AS total,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failed,
                   SUM(tokens) AS tokens
            FROM request_records
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY username
            ORDER BY total DESC
        """, (from_query, to_query)).fetchall()

        # Per-model per-bucket breakdown for trend chart
        model_bucket_rows = db.execute("""
            SELECT timestamp, model,
                   COUNT(*) AS total,
                   SUM(tokens) AS tokens
            FROM request_records
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY timestamp, model
            ORDER BY timestamp, model
        """, (from_query, to_query)).fetchall()

    bucket_rows = {}
    for row in rows:
        bucket = _bucket_for_timestamp(row["timestamp"], granularity)
        current = bucket_rows.setdefault(bucket, {
            "bucket": bucket, "total": 0, "failed": 0, "tokens": 0,
            "image_calls": 0, "image_failed": 0, "images": 0, "image_bytes": 0,
        })
        current["total"] += row["total"] or 0
        current["failed"] += row["failed"] or 0
        current["tokens"] += row["tokens"] or 0
        current["image_calls"] += row["image_calls"] or 0
        current["image_failed"] += row["image_failed"] or 0
        current["images"] += row["images"] or 0
        current["image_bytes"] += row["image_bytes"] or 0

    bucket_model_rows = {}
    for row in model_bucket_rows:
        bucket = _bucket_for_timestamp(row["timestamp"], granularity)
        key = (bucket, row["model"])
        current = bucket_model_rows.setdefault(key, {"bucket": bucket, "model": row["model"], "total": 0, "tokens": 0})
        current["total"] += row["total"] or 0
        current["tokens"] += row["tokens"] or 0

    rows = [bucket_rows[key] for key in sorted(bucket_rows)]
    model_bucket_rows = [bucket_model_rows[key] for key in sorted(bucket_model_rows)]

    # Zero-pad the timeline so every bucket is present
    rows, bucket_labels, model_bucket_rows = _zero_pad_timeline(rows, from_ts, to_ts, granularity, model_bucket_rows)

    timeline = {
        "labels": [r["bucket"] or "" for r in rows],
        "total":  [r["total"] for r in rows],
        "failed": [r["failed"] for r in rows],
        "tokens": [r["tokens"] for r in rows],
        "image_calls": [r["image_calls"] for r in rows],
        "image_failed": [r["image_failed"] for r in rows],
        "images": [r["images"] for r in rows],
        "image_bytes": [r["image_bytes"] for r in rows],
    }

    # Build per-model timeline matrix for stacked bar chart
    all_models = sorted({r["model"] for r in model_bucket_rows})
    model_bucket_map = {}
    for r in model_bucket_rows:
        model_bucket_map[(r["bucket"] or "", r["model"])] = {"total": r["total"], "tokens": r["tokens"]}
    timeline_models = {
        "labels": bucket_labels,
        "models": all_models,
        "calls": [[model_bucket_map.get((b, m), {}).get("total", 0) for b in bucket_labels] for m in all_models],
        "tokens": [[model_bucket_map.get((b, m), {}).get("tokens", 0) for b in bucket_labels] for m in all_models],
    }

    models = [
        {"model": r["model"], "total": r["total"], "failed": r["failed"], "tokens": r["tokens"]}
        for r in model_rows
    ]
    users = [
        {"username": r["username"], "total": r["total"], "failed": r["failed"], "tokens": r["tokens"]}
        for r in user_rows
    ]
    overall = {
        "total_calls": sum(r["total"] for r in rows),
        "failed_calls": sum(r["failed"] for r in rows),
        "total_tokens": sum(r["tokens"] for r in rows),
        "image_generation_calls": sum(r["image_calls"] for r in rows),
        "image_generation_failed_calls": sum(r["image_failed"] for r in rows),
        "image_generation_images": sum(r["images"] for r in rows),
        "image_generation_bytes": sum(r["image_bytes"] for r in rows),
    }
    return {"timeline": timeline, "timeline_models": timeline_models, "models": models, "users": users, "overall": overall}


def delete_request_records_before(ts: str) -> int:
    """Delete request records older than ts. Returns number of deleted rows."""
    with get_db() as db:
        cursor = db.execute("DELETE FROM request_records WHERE timestamp < ?", (ts,))
        return cursor.rowcount


def increment_user_usage(username: str, api_key_value: str, success: bool, tokens: int = 0) -> None:
    with get_db() as db:
        db.execute("UPDATE users SET total_calls = total_calls + 1, total_tokens = total_tokens + ? WHERE username = ?", (tokens, username))
        if not success:
            db.execute("UPDATE users SET failed_calls = failed_calls + 1 WHERE username = ?", (username,))
        db.execute("UPDATE user_api_keys SET total_calls = total_calls + 1, total_tokens = total_tokens + ? WHERE key = ?", (tokens, api_key_value))
        if not success:
            db.execute("UPDATE user_api_keys SET failed_calls = failed_calls + 1 WHERE key = ?", (api_key_value,))


def reset_user_stats() -> None:
    with get_db() as db:
        db.execute("UPDATE users SET total_calls = 0, failed_calls = 0, total_tokens = 0")
        db.execute("UPDATE user_api_keys SET total_calls = 0, failed_calls = 0, total_tokens = 0")


# -- Preprocessors --

PREPROCESSOR_DEFAULTS = {
    "api_base": "",
    "model": "",
    "api_key": "",
    "timeout": 120,
    "max_images": 10,
    "prompt": "",
    "enabled": True,
    "max_tokens": 2048,
}


def _preprocessor_from_row(row: sqlite3.Row) -> dict:
    data = _row_to_dict(row)
    data["enabled"] = _to_bool(data.get("enabled"))
    for key in ("timeout", "max_images", "max_tokens"):
        try:
            data[key] = int(data.get(key) or PREPROCESSOR_DEFAULTS[key])
        except (TypeError, ValueError):
            data[key] = PREPROCESSOR_DEFAULTS[key]
    return data


def _normalize_preprocessor_config(config: dict) -> dict:
    normalized = dict(PREPROCESSOR_DEFAULTS)
    normalized.update({k: v for k, v in (config or {}).items() if v is not None})
    for key in ("api_base", "model", "api_key", "prompt"):
        normalized[key] = str(normalized.get(key) or "")
    for key in ("timeout", "max_images", "max_tokens"):
        try:
            normalized[key] = int(normalized.get(key) or PREPROCESSOR_DEFAULTS[key])
        except (TypeError, ValueError):
            normalized[key] = PREPROCESSOR_DEFAULTS[key]
    normalized["enabled"] = _to_bool(normalized.get("enabled", True))
    return normalized


def get_preprocessors() -> dict:
    with get_db() as db:
        rows = db.execute("SELECT * FROM preprocessors ORDER BY id").fetchall()
        return {row["id"]: {k: v for k, v in _preprocessor_from_row(row).items() if k != "id"} for row in rows}


def get_enabled_preprocessor() -> Optional[dict]:
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM preprocessors WHERE enabled = 1 ORDER BY updated_at DESC, id LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return _preprocessor_from_row(row)


def upsert_preprocessor(preprocessor_id: str, config: dict) -> dict:
    preprocessor_id = str(preprocessor_id or "").strip()
    if not preprocessor_id:
        raise ValueError("preprocessor id is required")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as db:
        current = db.execute("SELECT * FROM preprocessors WHERE id = ?", (preprocessor_id,)).fetchone()
        merged = _preprocessor_from_row(current) if current else {}
        merged.update(config or {})
        normalized = _normalize_preprocessor_config(merged)
        if normalized["enabled"]:
            db.execute("UPDATE preprocessors SET enabled = 0 WHERE id <> ?", (preprocessor_id,))
        if current:
            db.execute(
                """
                UPDATE preprocessors
                SET api_base = ?, model = ?, api_key = ?, timeout = ?, max_images = ?,
                    prompt = ?, enabled = ?, max_tokens = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    normalized["api_base"], normalized["model"], normalized["api_key"],
                    normalized["timeout"], normalized["max_images"], normalized["prompt"],
                    1 if normalized["enabled"] else 0, normalized["max_tokens"], now,
                    preprocessor_id,
                ),
            )
        else:
            db.execute(
                """
                INSERT INTO preprocessors
                    (id, api_base, model, api_key, timeout, max_images, prompt, enabled, max_tokens, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    preprocessor_id, normalized["api_base"], normalized["model"], normalized["api_key"],
                    normalized["timeout"], normalized["max_images"], normalized["prompt"],
                    1 if normalized["enabled"] else 0, normalized["max_tokens"], now, now,
                ),
            )
        row = db.execute("SELECT * FROM preprocessors WHERE id = ?", (preprocessor_id,)).fetchone()
        return _preprocessor_from_row(row)


def delete_preprocessor(preprocessor_id: str) -> bool:
    with get_db() as db:
        cursor = db.execute("DELETE FROM preprocessors WHERE id = ?", (preprocessor_id,))
        return cursor.rowcount > 0


# -- Providers --

PROVIDER_REQUEST_DEFAULTS = {
    "request_timeout": 120,
    "retry_count": 0,
    "retry_backoff": 0.5,
}


def _normalize_provider_request_options(provider: dict) -> dict:
    return {
        "request_timeout": _clamp_int(provider.get("request_timeout"), 120, 1, 3600),
        "retry_count": _clamp_int(provider.get("retry_count"), 0, 0, 10),
        "retry_backoff": _clamp_float(provider.get("retry_backoff"), 0.5, 0, 60),
    }


def _provider_from_row(row: sqlite3.Row) -> dict:
    p = _row_to_dict(row)
    p["enabled"] = _to_bool(p["enabled"])
    p["extra_headers"] = _json_loads(p.get("extra_headers", "{}")) or {}
    p.update(_normalize_provider_request_options(p))
    p["force_chat_completions"] = _to_bool(p.get("force_chat_completions", 0))
    return p


def _model_from_row(row: sqlite3.Row) -> dict:
    image_generation = row["image_generation"] if "image_generation" in row.keys() else ""
    source = (row["source"] if "source" in row.keys() else "") or "auto"
    model = {
        "id": row["model_id"], "name": row["model_name"],
        "enabled": _to_bool(row["enabled"]), "preprocessor": row["preprocessor"] or "",
        "image_generation": bool(image_generation),
        "source": source,
        "responses_status": row["responses_status"] or "unknown",
        "responses_checked_at": row["responses_checked_at"] or "",
        "responses_expires_at": row["responses_expires_at"] or "",
        "responses_streaming": _to_bool(row["responses_streaming"]),
        "responses_streaming_status": row["responses_streaming_status"] or "unknown",
        "responses_tool_types": _json_loads(row["responses_tool_types"] or "[]") or [],
        "responses_error": row["responses_error"] or "",
    }
    return model


IMAGE_GENERATOR_DEFAULTS = {
    "backend_type": "existing_model", "provider_model": "", "api_base": "",
    "model": "", "api_key": "", "timeout": 180, "enabled": True,
    "workflow": {}, "workflow_mapping": {}, "poll_interval": 1.0,
}


def _image_generator_from_row(row: sqlite3.Row) -> dict:
    data = _row_to_dict(row)
    data["enabled"] = _to_bool(data.get("enabled"))
    try:
        data["timeout"] = int(data.get("timeout") or 180)
    except (TypeError, ValueError):
        data["timeout"] = 180
    data["workflow"] = _json_loads(data.get("workflow", "{}")) or {}
    data["workflow_mapping"] = _json_loads(data.get("workflow_mapping", "{}")) or {}
    try:
        data["poll_interval"] = float(data.get("poll_interval") or 1.0)
    except (TypeError, ValueError):
        data["poll_interval"] = 1.0
    return data


def get_image_generators() -> dict:
    with get_db() as db:
        rows = db.execute("SELECT * FROM image_generators ORDER BY id").fetchall()
        return {row["id"]: {k: v for k, v in _image_generator_from_row(row).items() if k != "id"} for row in rows}


def get_enabled_image_generator() -> Optional[dict]:
    with get_db() as db:
        row = db.execute("SELECT * FROM image_generators WHERE enabled = 1 ORDER BY updated_at DESC, id LIMIT 1").fetchone()
        return _image_generator_from_row(row) if row else None


def upsert_image_generator(generator_id: str, config: dict) -> dict:
    generator_id = str(generator_id or "").strip()
    if not generator_id:
        raise ValueError("image generator id is required")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as db:
        row = db.execute("SELECT * FROM image_generators WHERE id = ?", (generator_id,)).fetchone()
        merged = dict(IMAGE_GENERATOR_DEFAULTS)
        if row:
            merged.update(_image_generator_from_row(row))
        merged.update(config or {})
        merged["backend_type"] = str(merged.get("backend_type") or "existing_model")
        for key in ("provider_model", "api_base", "model", "api_key"):
            merged[key] = str(merged.get(key) or "")
        for key in ("workflow", "workflow_mapping"):
            if not isinstance(merged.get(key), dict):
                raise ValueError(f"{key} must be an object")
        try:
            merged["timeout"] = max(1, min(3600, int(merged.get("timeout") or 180)))
        except (TypeError, ValueError):
            merged["timeout"] = 180
        try:
            merged["poll_interval"] = max(0.2, min(10.0, float(merged.get("poll_interval") or 1.0)))
        except (TypeError, ValueError):
            merged["poll_interval"] = 1.0
        merged["enabled"] = _to_bool(merged.get("enabled", True))
        if merged["enabled"]:
            db.execute("UPDATE image_generators SET enabled = 0 WHERE id <> ?", (generator_id,))
        values = (merged["backend_type"], merged["provider_model"], merged["api_base"], merged["model"], merged["api_key"], merged["timeout"], json.dumps(merged["workflow"], ensure_ascii=False), json.dumps(merged["workflow_mapping"], ensure_ascii=False), merged["poll_interval"], 1 if merged["enabled"] else 0, now)
        if row:
            db.execute("UPDATE image_generators SET backend_type=?, provider_model=?, api_base=?, model=?, api_key=?, timeout=?, workflow=?, workflow_mapping=?, poll_interval=?, enabled=?, updated_at=? WHERE id=?", values + (generator_id,))
        else:
            db.execute("INSERT INTO image_generators (id, backend_type, provider_model, api_base, model, api_key, timeout, workflow, workflow_mapping, poll_interval, enabled, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (generator_id,) + values[:-1] + (now, now))
        return _image_generator_from_row(db.execute("SELECT * FROM image_generators WHERE id = ?", (generator_id,)).fetchone())


def delete_image_generator(generator_id: str) -> bool:
    with get_db() as db:
        return db.execute("DELETE FROM image_generators WHERE id = ?", (generator_id,)).rowcount > 0


def set_model_image_generation(model_id: str, enabled: bool) -> bool:
    mid = parse_model_id(model_id)
    with get_db() as db:
        if mid.is_composite:
            cur = db.execute("UPDATE provider_models SET image_generation = ? WHERE provider_id = ? AND model_id = ?", ("1" if enabled else "", mid.provider_id, mid.model_name))
            # Older admin pages could prepend the provider to an already
            # composite model value. Accept that stale form once so a cached
            # browser can still turn the switch off after this fix ships.
            repeated_prefix = f"{mid.provider_id}/"
            if cur.rowcount == 0 and mid.model_name.startswith(repeated_prefix):
                cur = db.execute(
                    "UPDATE provider_models SET image_generation = ? WHERE provider_id = ? AND model_id = ?",
                    ("1" if enabled else "", mid.provider_id, mid.model_name[len(repeated_prefix):]),
                )
            if cur.rowcount == 0:
                # Some legacy provider refreshes stored the provider prefix in
                # model_id itself. Keep accepting the canonical admin value
                # until those rows are naturally refreshed.
                cur = db.execute(
                    "UPDATE provider_models SET image_generation = ? WHERE provider_id = ? AND model_id = ?",
                    ("1" if enabled else "", mid.provider_id, f"{mid.provider_id}/{mid.model_name}"),
                )
        else:
            cur = db.execute("UPDATE provider_models SET image_generation = ? WHERE model_id = ?", ("1" if enabled else "", mid.model_name))
        return cur.rowcount > 0


def get_model_image_generation(provider_id: str, model: str) -> bool:
    model_name = parse_model_id(model).model_name
    with get_db() as db:
        row = db.execute("SELECT image_generation FROM provider_models WHERE provider_id = ? AND (model_id IN (?, ?) OR model_name = ?) LIMIT 1", (provider_id, model, model_name, model_name)).fetchone()
        return bool(row and row["image_generation"])

def get_providers() -> list:
    with get_db() as db:
        rows = db.execute("SELECT * FROM providers ORDER BY id").fetchall()
        result = []
        for r in rows:
            p = _provider_from_row(r)
            models_rows = db.execute("SELECT * FROM provider_models WHERE provider_id = ? ORDER BY model_id", (p["id"],)).fetchall()
            p["models"] = [_model_from_row(m) for m in models_rows]
            result.append(p)
        return result


def get_provider(provider_id: str) -> Optional[dict]:
    with get_db() as db:
        row = db.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
        if not row:
            return None
        p = _provider_from_row(row)
        models_rows = db.execute("SELECT * FROM provider_models WHERE provider_id = ? ORDER BY model_id", (provider_id,)).fetchall()
        p["models"] = [_model_from_row(m) for m in models_rows]
        return p


def get_model_responses_capability(provider_id: str, model: str) -> dict | None:
    model_name = parse_model_id(model).model_name
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM provider_models WHERE provider_id = ? AND (model_id IN (?, ?) OR model_name = ?) LIMIT 1",
            (provider_id, model, model_name, model_name),
        ).fetchone()
        return _model_from_row(row) if row else None


def set_model_responses_capability(provider_id: str, model: str, *, status: str, streaming: bool = False,
                                   streaming_status: str = "unknown", tool_types: list | None = None,
                                   error: str = "", expires_at: str = "") -> None:
    if status not in {"unknown", "supported", "unsupported", "degraded"}:
        raise ValueError("invalid Responses capability status")
    with get_db() as db:
        db.execute(
            "UPDATE provider_models SET responses_status = ?, responses_checked_at = ?, responses_expires_at = ?, responses_streaming = ?, responses_streaming_status = ?, responses_tool_types = ?, responses_error = ? WHERE provider_id = ? AND (model_id IN (?, ?) OR model_name = ?)",
            (status, datetime.now(timezone.utc).isoformat(), expires_at, 1 if streaming else 0, streaming_status, json.dumps(tool_types or [], ensure_ascii=False), str(error)[:500], provider_id, model, parse_model_id(model).model_name, parse_model_id(model).model_name),
        )


def update_model_responses_capability(provider_id: str, model: str, **updates) -> None:
    """Update only observed model-level capability fields without erasing evidence."""
    allowed = {
        "responses_status": "status",
        "responses_expires_at": "expires_at",
        "responses_streaming": "streaming",
        "responses_streaming_status": "streaming_status",
        "responses_tool_types": "tool_types",
        "responses_error": "error",
    }
    fields = []
    values = []
    for column, key in allowed.items():
        if key not in updates:
            continue
        value = updates[key]
        if key == "streaming":
            value = 1 if value else 0
        elif key == "tool_types":
            value = json.dumps(value or [], ensure_ascii=False)
        elif key == "error":
            value = str(value)[:500]
        fields.append(f"{column} = ?")
        values.append(value)
    if not fields:
        return
    fields.append("responses_checked_at = ?")
    values.append(datetime.now(timezone.utc).isoformat())
    model_name = parse_model_id(model).model_name
    with get_db() as db:
        db.execute(
            f"UPDATE provider_models SET {', '.join(fields)} WHERE provider_id = ? AND (model_id IN (?, ?) OR model_name = ?)",
            (*values, provider_id, model, model_name, model_name),
        )


def update_model_responses_tool_types(provider_id: str, model: str, tool_types: list[str]) -> None:
    normalized = sorted({str(item) for item in tool_types if item})
    with get_db() as db:
        db.execute("UPDATE provider_models SET responses_tool_types = ? WHERE provider_id = ? AND (model_id IN (?, ?) OR model_name = ?)", (json.dumps(normalized, ensure_ascii=False), provider_id, model, parse_model_id(model).model_name, parse_model_id(model).model_name))


def add_provider(provider: dict) -> dict:
    with get_db() as db:
        try:
            extra_headers_json = json.dumps(provider.get("extra_headers", {}), ensure_ascii=False)
            options = _normalize_provider_request_options(provider)
            db.execute(
                """
                INSERT INTO providers
                    (id, name, provider_type, api_base, api_key, enabled, extra_headers, request_timeout, retry_count, retry_backoff, force_chat_completions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (provider["id"], provider["name"], provider.get("provider_type", "openai"),
                 provider.get("api_base", ""), provider.get("api_key", ""),
                 1 if provider.get("enabled", True) else 0,
                 extra_headers_json, options["request_timeout"], options["retry_count"], options["retry_backoff"], 1 if provider.get("force_chat_completions", False) else 0)
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Provider '{provider['id']}' already exists")
        for m in provider.get("models", []):
            db.execute(
                "INSERT OR IGNORE INTO provider_models (provider_id, model_id, model_name, enabled, preprocessor, image_generation, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    provider["id"],
                    m["id"],
                    m.get("name", m["id"]),
                    1 if m.get("enabled", True) else 0,
                    m.get("preprocessor", ""),
                    "1" if m.get("image_generation") else "",
                    "manual",
                )
            )
    options = _normalize_provider_request_options(provider)
    return {
        "id": provider["id"],
        "name": provider["name"],
        "provider_type": provider.get("provider_type", "openai"),
        "api_base": provider.get("api_base", ""),
        "api_key": provider.get("api_key", ""),
        "enabled": provider.get("enabled", True),
        "extra_headers": provider.get("extra_headers", {}),
        **options,
        "force_chat_completions": bool(provider.get("force_chat_completions", False)),
        "models": [
            {
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "enabled": m.get("enabled", True),
                "preprocessor": m.get("preprocessor", ""),
                "image_generation": bool(m.get("image_generation")),
                "source": "manual",
                "responses_status": "unknown",
                "responses_checked_at": "",
                "responses_expires_at": "",
                "responses_streaming": False,
                "responses_streaming_status": "unknown",
                "responses_tool_types": [],
                "responses_error": "",
            }
            for m in provider.get("models", [])
        ]
    }


def update_provider(provider_id: str, updates: dict) -> Optional[dict]:
    with get_db() as db:
        existing = db.execute("SELECT 1 FROM providers WHERE id = ?", (provider_id,)).fetchone()
        if not existing:
            return None
        _updatable = {"name", "provider_type", "api_base", "api_key"}
        for key in _updatable:
            if key in updates:
                db.execute(f"UPDATE providers SET {key} = ? WHERE id = ?", (updates[key], provider_id))
        if "force_chat_completions" in updates:
            db.execute(
                "UPDATE providers SET force_chat_completions = ? WHERE id = ?",
                (1 if updates["force_chat_completions"] else 0, provider_id),
            )
        if any(key in updates for key in PROVIDER_REQUEST_DEFAULTS):
            options = _normalize_provider_request_options({**PROVIDER_REQUEST_DEFAULTS, **updates})
            for key, value in options.items():
                if key in updates:
                    db.execute(f"UPDATE providers SET {key} = ? WHERE id = ?", (value, provider_id))
        if "extra_headers" in updates:
            db.execute("UPDATE providers SET extra_headers = ? WHERE id = ?",
                       (json.dumps(updates["extra_headers"], ensure_ascii=False), provider_id))
        if "enabled" in updates:
            db.execute("UPDATE providers SET enabled = ? WHERE id = ?", (1 if updates["enabled"] else 0, provider_id))
        if "models" in updates:
            existing_ids = {m["model_id"] for m in db.execute("SELECT model_id FROM provider_models WHERE provider_id = ?", (provider_id,)).fetchall()}
            for m in updates["models"]:
                if m["id"] in existing_ids:
                    if "image_generation" in m:
                        db.execute(
                            "UPDATE provider_models SET model_name = ?, enabled = ?, preprocessor = ?, image_generation = ? WHERE provider_id = ? AND model_id = ?",
                            (m.get("name", m["id"]), 1 if m.get("enabled", True) else 0, m.get("preprocessor", ""), "1" if m.get("image_generation") else "", provider_id, m["id"])
                        )
                    else:
                        db.execute(
                            "UPDATE provider_models SET model_name = ?, enabled = ?, preprocessor = ? WHERE provider_id = ? AND model_id = ?",
                            (m.get("name", m["id"]), 1 if m.get("enabled", True) else 0, m.get("preprocessor", ""), provider_id, m["id"])
                        )
                    # Explicit opt-in only: the caller must send source="manual"
                    # to promote an auto-discovered model. Omitting source keeps
                    # the stored value, so plain edits never flip auto -> manual.
                    if m.get("source") == "manual":
                        db.execute(
                            "UPDATE provider_models SET source = 'manual' WHERE provider_id = ? AND model_id = ?",
                            (provider_id, m["id"])
                        )
                else:
                    db.execute(
                        "INSERT OR IGNORE INTO provider_models (provider_id, model_id, model_name, enabled, preprocessor, image_generation, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (provider_id, m["id"], m.get("name", m["id"]), 1 if m.get("enabled", True) else 0, m.get("preprocessor", ""), "1" if m.get("image_generation") else "", "manual")
                    )
        if any(key in updates for key in {"api_base", "api_key", "provider_type", "models"}):
            db.execute("UPDATE provider_models SET responses_status = 'unknown', responses_checked_at = '', responses_expires_at = '', responses_streaming = 0, responses_streaming_status = 'unknown', responses_tool_types = '[]', responses_error = '' WHERE provider_id = ?", (provider_id,))
        # Fetch updated state within same transaction
        row = db.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
        if not row:
            return None
        p = _provider_from_row(row)
        models_rows = db.execute("SELECT * FROM provider_models WHERE provider_id = ? ORDER BY model_id", (provider_id,)).fetchall()
        p["models"] = [_model_from_row(m) for m in models_rows]
        return p


def delete_provider(provider_id: str) -> bool:
    with get_db() as db:
        cursor = db.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
        return cursor.rowcount > 0


def delete_provider_model(provider_id: str, model_id: str) -> bool:
    """Delete a single model row from a provider.

    Works for both auto-discovered and manually added models: a manual model
    is never removed by refresh, so an explicit delete entry point is the only
    way to get rid of one.
    """
    with get_db() as db:
        cursor = db.execute(
            "DELETE FROM provider_models WHERE provider_id = ? AND model_id = ?",
            (provider_id, model_id),
        )
        return cursor.rowcount > 0


class ModelId:
    """Unified model identifier that encapsulates provider/model composite parsing.

    Supports simple "model" format and composite "provider/model" format.
    Can be compared with strings such as model_id in ["allowed-model", "provider/model"].
    """

    __slots__ = ("provider_id", "model_name")

    def __init__(self, provider_id: str = "", model_name: str = ""):
        self.provider_id = provider_id
        self.model_name = model_name

    @classmethod
    def parse(cls, raw: str) -> "ModelId":
        if not raw:
            return cls("", "")
        if "/" in raw:
            parts = raw.split("/", 1)
            return cls(parts[0], parts[1])
        return cls("", raw)

    @property
    def composite(self) -> str:
        return f"{self.provider_id}/{self.model_name}" if self.provider_id else self.model_name

    @property
    def is_composite(self) -> bool:
        return bool(self.provider_id)

    def __str__(self) -> str:
        return self.composite

    def __repr__(self) -> str:
        return f"ModelId(provider={self.provider_id!r}, model={self.model_name!r})"

    def __eq__(self, other):
        if isinstance(other, ModelId):
            return (self.provider_id, self.model_name) == (other.provider_id, other.model_name)
        if isinstance(other, str):
            if self.composite == other:
                return True
            # Composite ID equals simple name: compare the model_name part
            if self.model_name == other:
                return True
            # Simple name equals composite ID string: compare the model suffix
            if not self.is_composite and "/" in other:
                return self.model_name == other.rsplit("/", 1)[-1]
            return False
        return NotImplemented

    def __hash__(self):
        return hash((self.provider_id, self.model_name))

    def __bool__(self):
        return bool(self.model_name)


def parse_model_id(model_id: str) -> ModelId:
    """Parse a model identifier into a ModelId object."""
    return ModelId.parse(model_id)


def find_provider_by_model(model_id: str) -> Optional[dict]:
    """Find the first enabled provider that serves the given model.

    Supports exact provider/model composite matching; without a prefix, returns the first matching provider.
    """
    mid = parse_model_id(model_id)
    with get_db() as db:
        if mid.provider_id:
            row = db.execute("""
                SELECT p.* FROM providers p
                JOIN provider_models m ON m.provider_id = p.id
                WHERE p.id = ? AND m.model_id = ? AND m.enabled = 1 AND p.enabled = 1
            """, (mid.provider_id, mid.model_name)).fetchone()
        else:
            row = db.execute("""
                SELECT p.* FROM providers p
                JOIN provider_models m ON m.provider_id = p.id
                WHERE m.model_id = ? AND m.enabled = 1 AND p.enabled = 1
                ORDER BY p.id
            """, (mid.model_name,)).fetchone()
        if not row:
            return None
        p = _row_to_dict(row)
        p["enabled"] = _to_bool(p["enabled"])
        p["extra_headers"] = _json_loads(p.get("extra_headers", "{}")) or {}
        return p


# -- Routing rules --

def get_routing_rules() -> list:
    return routing_db.get_routing_rules(get_db)


def get_routing_rule(rule_id: str) -> Optional[dict]:
    return routing_db.get_routing_rule(get_db, rule_id)


def add_routing_rule(rule: dict) -> dict:
    return routing_db.add_routing_rule(get_db, get_routing_rule, rule)


def update_routing_rule(rule_id: str, updates: dict) -> Optional[dict]:
    return routing_db.update_routing_rule(get_db, get_routing_rule, rule_id, updates)


def delete_routing_rule(rule_id: str) -> bool:
    return routing_db.delete_routing_rule(get_db, rule_id)


def get_fallback_policies() -> list:
    return fallback_db.get_fallback_policies(get_db)


def get_fallback_policy(policy_id: str) -> Optional[dict]:
    return fallback_db.get_fallback_policy(get_db, policy_id)


def add_fallback_policy(policy: dict) -> dict:
    return fallback_db.add_fallback_policy(get_db, get_fallback_policy, policy)


def update_fallback_policy(policy_id: str, updates: dict) -> Optional[dict]:
    return fallback_db.update_fallback_policy(get_db, get_fallback_policy, policy_id, updates)


def delete_fallback_policy(policy_id: str) -> bool:
    return fallback_db.delete_fallback_policy(get_db, policy_id)


# -- Request logs (full request/response viewer) --

def add_request_log(
    timestamp: str,
    endpoint: str,
    username: str,
    api_key: str,
    requested_model: str,
    model: str,
    provider: str,
    status: str,
    stream: bool,
    tokens: int,
    request_body=None,
    response_body=None,
    details: Optional[dict] = None,
    error: Optional[str] = None,
) -> int:
    return request_logs_db.add_request_log(
        get_db,
        timestamp=timestamp,
        endpoint=endpoint,
        username=username,
        api_key=api_key,
        requested_model=requested_model,
        model=model,
        provider=provider,
        status=status,
        stream=stream,
        tokens=tokens,
        request_body=request_body,
        response_body=response_body,
        details=details,
        error=error,
    )


def update_request_log(
    log_id: int,
    timestamp: str,
    endpoint: str,
    username: str,
    api_key: str,
    requested_model: str,
    model: str,
    provider: str,
    status: str,
    stream: bool,
    tokens: int,
    request_body=None,
    response_body=None,
    details: Optional[dict] = None,
    error: Optional[str] = None,
) -> bool:
    return request_logs_db.update_request_log(
        get_db,
        log_id,
        timestamp=timestamp,
        endpoint=endpoint,
        username=username,
        api_key=api_key,
        requested_model=requested_model,
        model=model,
        provider=provider,
        status=status,
        stream=stream,
        tokens=tokens,
        request_body=request_body,
        response_body=response_body,
        details=details,
        error=error,
    )


def list_request_logs(
    limit: int = 100,
    offset: int = 0,
    endpoint: Optional[str] = None,
    username: Optional[str] = None,
    status: Optional[str] = None,
) -> list:
    return request_logs_db.list_request_logs(
        get_db,
        limit=limit,
        offset=offset,
        endpoint=endpoint,
        username=username,
        status=status,
    )


def count_request_logs(
    endpoint: Optional[str] = None,
    username: Optional[str] = None,
    status: Optional[str] = None,
) -> int:
    return request_logs_db.count_request_logs(
        get_db,
        endpoint=endpoint,
        username=username,
        status=status,
    )


def get_request_log(log_id: int) -> Optional[dict]:
    return request_logs_db.get_request_log(get_db, log_id)


def delete_request_log(log_id: int) -> bool:
    return request_logs_db.delete_request_log(get_db, log_id)


def clear_request_logs() -> int:
    return request_logs_db.clear_request_logs(get_db)


def trim_request_logs(keep: int) -> int:
    return request_logs_db.trim_request_logs(get_db, keep)
