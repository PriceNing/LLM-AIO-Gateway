import sqlite3
import json
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from app.db import fallback as fallback_db
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
            # Routing and fallback policy migrations.
            routing_db.migrate(conn)
            fallback_db.migrate(conn)
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
    extra_headers TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS provider_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT '',
    preprocessor TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE,
    UNIQUE(provider_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_provider_models_model_id ON provider_models(model_id);

CREATE TABLE IF NOT EXISTS routing_rules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'New Rule',
    enabled INTEGER NOT NULL DEFAULT 1,
    username TEXT NOT NULL DEFAULT '',
    api_key_pattern TEXT NOT NULL DEFAULT '',
    match_model TEXT NOT NULL DEFAULT '',
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
    tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_req_ts ON request_records(timestamp);
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


# -- Global stats --

def get_global_stats() -> dict:
    with get_db() as db:
        rows = db.execute("SELECT key, value FROM global_stats").fetchall()
        result = {}
        for r in rows:
            v = r["value"]
            result[r["key"]] = int(v) if v.lstrip("-").isdigit() else v
        return result


def increment_global_stats(success: bool) -> None:
    with get_db() as db:
        db.execute("INSERT OR IGNORE INTO global_stats (key, value) VALUES ('total_calls', '0')")
        db.execute("INSERT OR IGNORE INTO global_stats (key, value) VALUES ('failed_calls', '0')")
        db.execute("INSERT OR IGNORE INTO global_stats (key, value) VALUES ('last_reset', '')")
        db.execute("UPDATE global_stats SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key = 'total_calls'")
        if not success:
            db.execute("UPDATE global_stats SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key = 'failed_calls'")


def reset_global_stats() -> None:
    today = date.today().isoformat()
    with get_db() as db:
        db.execute("UPDATE global_stats SET value = '0' WHERE key IN ('total_calls', 'failed_calls')")
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

def add_request_record(model: str, username: str, success: bool, tokens: int = 0) -> None:
    """Insert a request record for historical stats. Called from _log_request."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as db:
        db.execute(
            "INSERT INTO request_records (timestamp, model, username, success, tokens) VALUES (?, ?, ?, ?, ?)",
            (now, model, username, 1 if success else 0, tokens)
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


def _zero_pad_timeline(rows, from_ts, to_ts, granularity, model_bucket_rows):
    """Fill in missing buckets so the timeline has no gaps."""
    fmt = _HISTORY_GRANULARITY[granularity]
    step_fmt = _HISTORY_STEP_FMT[granularity]
    delta = _HISTORY_DELTA[granularity]

    # Parse from/to boundaries in the bucket format
    start = datetime.strptime(from_ts[:19] if len(from_ts) > 10 else from_ts[:10], from_ts[:19].count(":") == 0 and "%Y-%m-%d" or "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(to_ts[:19] if len(to_ts) > 10 else to_ts[:10], to_ts[:19].count(":") == 0 and "%Y-%m-%d" or "%Y-%m-%d %H:%M:%S")

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

    zero_row = {"total": 0, "failed": 0, "tokens": 0}
    padded_rows = []
    padded_model_rows = []
    for b in all_buckets:
        existing = row_map.get(b)
        if existing:
            padded_rows.append(existing)
        else:
            padded_rows.append({"bucket": b, "total": 0, "failed": 0, "tokens": 0})
        for mr in model_map.get(b, []):
            padded_model_rows.append(mr)
        # Missing bucket -> no model rows needed (all zeros)

    return padded_rows, all_buckets, padded_model_rows


def get_history_stats(from_ts: str, to_ts: str, granularity: str = "day") -> dict:
    """Aggregate historical stats by granularity. Returns timeline + model breakdown."""
    fmt = _HISTORY_GRANULARITY.get(granularity, "%Y-%m-%d")
    with get_db() as db:
        # Timeline: total calls, failures, tokens per bucket
        rows = db.execute("""
            SELECT strftime(?, timestamp) AS bucket,
                   COUNT(*) AS total,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failed,
                   SUM(tokens) AS tokens
            FROM request_records
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY bucket
            ORDER BY bucket
        """, (fmt, from_ts, to_ts)).fetchall()

        # Model breakdown for the period
        model_rows = db.execute("""
            SELECT model, COUNT(*) AS total,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failed,
                   SUM(tokens) AS tokens
            FROM request_records
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY model
            ORDER BY total DESC
        """, (from_ts, to_ts)).fetchall()

        # User breakdown for the period
        user_rows = db.execute("""
            SELECT username, COUNT(*) AS total,
                   SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failed,
                   SUM(tokens) AS tokens
            FROM request_records
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY username
            ORDER BY total DESC
        """, (from_ts, to_ts)).fetchall()

        # Per-model per-bucket breakdown for trend chart
        model_bucket_rows = db.execute("""
            SELECT strftime(?, timestamp) AS bucket, model,
                   COUNT(*) AS total,
                   SUM(tokens) AS tokens
            FROM request_records
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY bucket, model
            ORDER BY bucket, model
        """, (fmt, from_ts, to_ts)).fetchall()

    # Zero-pad the timeline so every bucket is present
    rows, bucket_labels, model_bucket_rows = _zero_pad_timeline(rows, from_ts, to_ts, granularity, model_bucket_rows)

    timeline = {
        "labels": [r["bucket"] or "" for r in rows],
        "total":  [r["total"] for r in rows],
        "failed": [r["failed"] for r in rows],
        "tokens": [r["tokens"] for r in rows],
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


# -- Providers --

def get_providers() -> list:
    with get_db() as db:
        rows = db.execute("SELECT * FROM providers ORDER BY id").fetchall()
        result = []
        for r in rows:
            p = _row_to_dict(r)
            p["enabled"] = _to_bool(p["enabled"])
            models_rows = db.execute("SELECT * FROM provider_models WHERE provider_id = ? ORDER BY model_id", (p["id"],)).fetchall()
            p["models"] = [{"id": m["model_id"], "name": m["model_name"], "enabled": _to_bool(m["enabled"]), "preprocessor": m["preprocessor"] or ""} for m in models_rows]
            p["extra_headers"] = _json_loads(p.get("extra_headers", "{}")) or {}
            result.append(p)
        return result


def get_provider(provider_id: str) -> Optional[dict]:
    with get_db() as db:
        row = db.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
        if not row:
            return None
        p = _row_to_dict(row)
        p["enabled"] = _to_bool(p["enabled"])
        models_rows = db.execute("SELECT * FROM provider_models WHERE provider_id = ? ORDER BY model_id", (provider_id,)).fetchall()
        p["models"] = [{"id": m["model_id"], "name": m["model_name"], "enabled": _to_bool(m["enabled"]), "preprocessor": m["preprocessor"] or ""} for m in models_rows]
        p["extra_headers"] = _json_loads(p.get("extra_headers", "{}")) or {}
        return p


def add_provider(provider: dict) -> dict:
    with get_db() as db:
        try:
            extra_headers_json = json.dumps(provider.get("extra_headers", {}), ensure_ascii=False)
            db.execute(
                "INSERT INTO providers (id, name, provider_type, api_base, api_key, enabled, extra_headers) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (provider["id"], provider["name"], provider.get("provider_type", "openai"),
                 provider.get("api_base", ""), provider.get("api_key", ""),
                 1 if provider.get("enabled", True) else 0,
                 extra_headers_json)
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Provider '{provider['id']}' already exists")
        for m in provider.get("models", []):
            db.execute(
                "INSERT OR IGNORE INTO provider_models (provider_id, model_id, model_name, enabled) VALUES (?, ?, ?, ?)",
                (provider["id"], m["id"], m.get("name", m["id"]), 1 if m.get("enabled", True) else 0)
            )
    return {
        "id": provider["id"],
        "name": provider["name"],
        "provider_type": provider.get("provider_type", "openai"),
        "api_base": provider.get("api_base", ""),
        "api_key": provider.get("api_key", ""),
        "enabled": provider.get("enabled", True),
        "models": [{"id": m["id"], "name": m.get("name", m["id"]), "enabled": m.get("enabled", True)} for m in provider.get("models", [])]
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
        if "extra_headers" in updates:
            db.execute("UPDATE providers SET extra_headers = ? WHERE id = ?",
                       (json.dumps(updates["extra_headers"], ensure_ascii=False), provider_id))
        if "enabled" in updates:
            db.execute("UPDATE providers SET enabled = ? WHERE id = ?", (1 if updates["enabled"] else 0, provider_id))
        if "models" in updates:
            existing_ids = {m["model_id"] for m in db.execute("SELECT model_id FROM provider_models WHERE provider_id = ?", (provider_id,)).fetchall()}
            for m in updates["models"]:
                if m["id"] in existing_ids:
                    db.execute(
                        "UPDATE provider_models SET model_name = ?, enabled = ?, preprocessor = ? WHERE provider_id = ? AND model_id = ?",
                        (m.get("name", m["id"]), 1 if m.get("enabled", True) else 0, m.get("preprocessor", ""), provider_id, m["id"])
                    )
                else:
                    db.execute(
                        "INSERT OR IGNORE INTO provider_models (provider_id, model_id, model_name, enabled, preprocessor) VALUES (?, ?, ?, ?, ?)",
                        (provider_id, m["id"], m.get("name", m["id"]), 1 if m.get("enabled", True) else 0, m.get("preprocessor", ""))
                    )
        # Fetch updated state within same transaction
        row = db.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
        if not row:
            return None
        p = _row_to_dict(row)
        p["enabled"] = _to_bool(p["enabled"])
        models_rows = db.execute("SELECT * FROM provider_models WHERE provider_id = ? ORDER BY model_id", (provider_id,)).fetchall()
        p["models"] = [{"id": m["model_id"], "name": m["model_name"], "enabled": _to_bool(m["enabled"]), "preprocessor": m["preprocessor"] or ""} for m in models_rows]
        return p


def delete_provider(provider_id: str) -> bool:
    with get_db() as db:
        cursor = db.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
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
