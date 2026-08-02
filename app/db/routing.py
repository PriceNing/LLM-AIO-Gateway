import json
import sqlite3
import uuid
from typing import Optional


def json_loads(value: str):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def json_dumps_list(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return "[]"


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes")
    return False


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def migrate(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(routing_rules)").fetchall()}
    if "match_scope" not in columns:
        conn.execute(
            "ALTER TABLE routing_rules ADD COLUMN match_scope TEXT NOT NULL DEFAULT 'any'"
        )


def normalize_rule(row: sqlite3.Row | None) -> Optional[dict]:
    data = row_to_dict(row)
    if not data:
        return None
    data["enabled"] = to_bool(data["enabled"])
    scope = str(data.get("match_scope") or "any").lower()
    data["match_scope"] = scope if scope in {"any", "unqualified", "qualified"} else "any"
    return data


def get_routing_rules(get_db) -> list:
    with get_db() as db:
        rows = db.execute("SELECT * FROM routing_rules ORDER BY rowid").fetchall()
        return [normalize_rule(row) for row in rows]


def get_routing_rule(get_db, rule_id: str) -> Optional[dict]:
    with get_db() as db:
        row = db.execute("SELECT * FROM routing_rules WHERE id = ?", (rule_id,)).fetchone()
        return normalize_rule(row)


def add_routing_rule(get_db, get_rule, rule: dict) -> dict:
    entry_id = rule.get("id") or uuid.uuid4().hex[:8]
    match_scope = str(rule.get("match_scope") or "any").lower()
    if match_scope not in {"any", "unqualified", "qualified"}:
        raise ValueError("match_scope must be one of: any, unqualified, qualified")
    with get_db() as db:
        db.execute(
            "INSERT INTO routing_rules (id, name, enabled, username, api_key_pattern, match_model, match_scope, target_model, target_provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry_id,
                rule.get("name", "New Rule"),
                1 if rule.get("enabled", True) else 0,
                rule.get("username", ""),
                rule.get("api_key_pattern", ""),
                rule.get("match_model", ""),
                match_scope,
                rule.get("target_model", ""),
                rule.get("target_provider", ""),
            ),
        )
    return get_rule(entry_id)


def update_routing_rule(get_db, get_rule, rule_id: str, updates: dict) -> Optional[dict]:
    with get_db() as db:
        existing = db.execute("SELECT 1 FROM routing_rules WHERE id = ?", (rule_id,)).fetchone()
        if not existing:
            return None
        for key in ("name", "enabled", "username", "api_key_pattern", "match_model", "match_scope", "target_model", "target_provider"):
            if key in updates:
                value = updates[key]
                if key == "enabled":
                    value = 1 if value else 0
                elif key == "match_scope":
                    value = str(value or "any").lower()
                    if value not in {"any", "unqualified", "qualified"}:
                        raise ValueError("match_scope must be one of: any, unqualified, qualified")
                db.execute(f"UPDATE routing_rules SET {key} = ? WHERE id = ?", (value, rule_id))
    return get_rule(rule_id)


def delete_routing_rule(get_db, rule_id: str) -> bool:
    with get_db() as db:
        cursor = db.execute("DELETE FROM routing_rules WHERE id = ?", (rule_id,))
        return cursor.rowcount > 0
