import json
import sqlite3
import uuid
from datetime import date
from typing import Optional


DEFAULT_TRIGGERS = {
    "timeout": True,
    "connection_error": True,
    "http_429": True,
    "http_5xx": True,
    "http_4xx": False,
}


def json_loads(value: str, fallback):
    if not value:
        return fallback
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return fallback
    return parsed if parsed is not None else fallback


def json_dumps(value, fallback) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        value = fallback
    return json.dumps(value, ensure_ascii=False)


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes")
    return False


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fallback_policies (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL DEFAULT 'New Fallback Policy',
            enabled INTEGER NOT NULL DEFAULT 1,
            match_provider TEXT NOT NULL DEFAULT '',
            match_model TEXT NOT NULL DEFAULT '*',
            triggers TEXT NOT NULL DEFAULT '{}',
            chain TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT ''
        )
        """
    )


def normalize_policy(row: sqlite3.Row | None) -> Optional[dict]:
    if row is None:
        return None
    data = dict(row)
    data["enabled"] = to_bool(data.get("enabled"))
    triggers = dict(DEFAULT_TRIGGERS)
    raw_triggers = json_loads(data.get("triggers", "{}"), {})
    if isinstance(raw_triggers, dict):
        triggers.update({key: bool(value) for key, value in raw_triggers.items()})
    data["triggers"] = triggers
    chain = json_loads(data.get("chain", "[]"), [])
    data["chain"] = chain if isinstance(chain, list) else []
    return data


def get_fallback_policies(get_db) -> list:
    with get_db() as db:
        rows = db.execute("SELECT * FROM fallback_policies ORDER BY rowid").fetchall()
        return [normalize_policy(row) for row in rows]


def get_fallback_policy(get_db, policy_id: str) -> Optional[dict]:
    with get_db() as db:
        row = db.execute("SELECT * FROM fallback_policies WHERE id = ?", (policy_id,)).fetchone()
        return normalize_policy(row)


def add_fallback_policy(get_db, get_policy, policy: dict) -> dict:
    entry_id = policy.get("id") or uuid.uuid4().hex[:8]
    triggers = dict(DEFAULT_TRIGGERS)
    if isinstance(policy.get("triggers"), dict):
        triggers.update({key: bool(value) for key, value in policy["triggers"].items()})
    with get_db() as db:
        db.execute(
            """
            INSERT INTO fallback_policies
                (id, name, enabled, match_provider, match_model, triggers, chain, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                policy.get("name", "New Fallback Policy"),
                1 if policy.get("enabled", True) else 0,
                policy.get("match_provider", ""),
                policy.get("match_model", "*"),
                json_dumps(triggers, DEFAULT_TRIGGERS),
                json_dumps(policy.get("chain", []), []),
                policy.get("created_at") or date.today().isoformat(),
            ),
        )
    return get_policy(entry_id)


def update_fallback_policy(get_db, get_policy, policy_id: str, updates: dict) -> Optional[dict]:
    with get_db() as db:
        existing = db.execute("SELECT * FROM fallback_policies WHERE id = ?", (policy_id,)).fetchone()
        if not existing:
            return None
        current = normalize_policy(existing) or {}
        for key in ("name", "enabled", "match_provider", "match_model", "triggers", "chain"):
            if key not in updates:
                continue
            value = updates[key]
            if key == "enabled":
                value = 1 if value else 0
            elif key == "triggers":
                triggers = dict(DEFAULT_TRIGGERS)
                if isinstance(value, dict):
                    triggers.update({trigger: bool(enabled) for trigger, enabled in value.items()})
                else:
                    triggers.update(current.get("triggers", {}))
                value = json_dumps(triggers, DEFAULT_TRIGGERS)
            elif key == "chain":
                value = json_dumps(value, [])
            db.execute(f"UPDATE fallback_policies SET {key} = ? WHERE id = ?", (value, policy_id))
    return get_policy(policy_id)


def delete_fallback_policy(get_db, policy_id: str) -> bool:
    with get_db() as db:
        cursor = db.execute("DELETE FROM fallback_policies WHERE id = ?", (policy_id,))
        return cursor.rowcount > 0
