import json
import sqlite3
from typing import Any, Iterable, Optional





def migrate(conn):
    return None


def row_to_dict(row):
    return dict(row) if row is not None else None


def _loads_json(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def add_request_log(
    get_db,
    *,
    timestamp,
    endpoint,
    username,
    api_key,
    requested_model,
    model,
    provider,
    status,
    stream,
    tokens,
    request_body=None,
    response_body=None,
    details=None,
    error=None,
):
    with get_db() as db:
        cursor = db.execute(
            """
            INSERT INTO request_logs (
                timestamp, endpoint, username, api_key, requested_model, model, provider,
                status, stream, tokens, request_body, response_body, details, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                endpoint,
                username or "",
                api_key or "",
                requested_model or "",
                model or "",
                provider or "",
                status or "",
                1 if stream else 0,
                int(tokens or 0),
                json.dumps(request_body, ensure_ascii=False, default=str)
                if request_body is not None
                else None,
                json.dumps(response_body, ensure_ascii=False, default=str)
                if response_body is not None
                else None,
                json.dumps(details or {}, ensure_ascii=False, default=str),
                error or "",
            ),
        )
        return int(cursor.lastrowid or 0)


def update_request_log(
    get_db,
    log_id,
    *,
    timestamp,
    endpoint,
    username,
    api_key,
    requested_model,
    model,
    provider,
    status,
    stream,
    tokens,
    request_body=None,
    response_body=None,
    details=None,
    error=None,
):
    """Replace a running request-log row with its latest lifecycle snapshot."""
    with get_db() as db:
        cursor = db.execute(
            """
            UPDATE request_logs SET
                timestamp = ?, endpoint = ?, username = ?, api_key = ?,
                requested_model = ?, model = ?, provider = ?, status = ?,
                stream = ?, tokens = ?, request_body = ?, response_body = ?,
                details = ?, error = ?
            WHERE id = ?
            """,
            (
                timestamp,
                endpoint,
                username or "",
                api_key or "",
                requested_model or "",
                model or "",
                provider or "",
                status or "",
                1 if stream else 0,
                int(tokens or 0),
                json.dumps(request_body, ensure_ascii=False, default=str)
                if request_body is not None else None,
                json.dumps(response_body, ensure_ascii=False, default=str)
                if response_body is not None else None,
                json.dumps(details or {}, ensure_ascii=False, default=str),
                error or "",
                int(log_id),
            ),
        )
        return cursor.rowcount > 0


def list_request_logs(
    get_db,
    *,
    limit=100,
    offset=0,
    endpoint=None,
    username=None,
    status=None,
):
    where = []
    params = []
    if endpoint:
        where.append("endpoint = ?")
        params.append(endpoint)
    if username:
        where.append("username = ?")
        params.append(username)
    if status:
        where.append("status = ?")
        params.append(status)
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = (
        "SELECT * FROM request_logs"
        + where_clause
        + " ORDER BY id DESC LIMIT ? OFFSET ?"
    )
    params.extend([int(limit), int(offset)])
    with get_db() as db:
        rows = db.execute(sql, params).fetchall()
    return [_decode_request_log(row_to_dict(r)) for r in rows]


def count_request_logs(
    get_db,
    *,
    endpoint=None,
    username=None,
    status=None,
):
    where = []
    params = []
    if endpoint:
        where.append("endpoint = ?")
        params.append(endpoint)
    if username:
        where.append("username = ?")
        params.append(username)
    if status:
        where.append("status = ?")
        params.append(status)
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    with get_db() as db:
        row = db.execute(
            "SELECT COUNT(*) AS n FROM request_logs" + where_clause, params
        ).fetchone()
    return int(row["n"] if row else 0)


def get_request_log(get_db, log_id):
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM request_logs WHERE id = ?", (int(log_id),)
        ).fetchone()
    if not row:
        return None
    return _decode_request_log(row_to_dict(row))


def delete_request_log(get_db, log_id):
    with get_db() as db:
        cursor = db.execute("DELETE FROM request_logs WHERE id = ?", (int(log_id),))
        return cursor.rowcount > 0


def clear_request_logs(get_db):
    with get_db() as db:
        cursor = db.execute("DELETE FROM request_logs")
        return cursor.rowcount


def trim_request_logs(get_db, keep):
    if not keep or keep <= 0:
        return 0
    with get_db() as db:
        cursor = db.execute(
            """
            DELETE FROM request_logs
            WHERE id NOT IN (
                SELECT id FROM request_logs ORDER BY id DESC LIMIT ?
            )
            """,
            (int(keep),),
        )
        return cursor.rowcount


def _decode_request_log(entry):
    if not entry:
        return entry
    entry = dict(entry)
    entry["stream"] = bool(entry.get("stream"))
    entry["request_body"] = _loads_json(entry.get("request_body"))
    entry["response_body"] = _loads_json(entry.get("response_body"))
    entry["details"] = _loads_json(entry.get("details")) or {}
    details = entry["details"]
    mode = str(details.get("responses_mode") or "")
    is_image = (
        details.get("request_kind") == "image_generation"
        or entry.get("endpoint") == "images_generations"
        or details.get("upstream_endpoint") == "images/generations"
        or "image_generation" in mode
    )
    entry["request_kind"] = "image_generation" if is_image else str(details.get("request_kind") or "text_generation")
    for key in (
        "image_model", "image_count", "image_bytes", "image_artifact_count",
        "image_requested_count", "image_succeeded_count", "image_failed_count",
        "image_retried_count", "image_reused_count", "image_completed_count",
        "image_batch_id",
        "responses_mode", "upstream_endpoint",
        "reasoning_effort", "chat_template_kwargs", "enable_thinking",
        "tps", "completion_tokens", "duration_ms", "generation_ms",
    ):
        if key in details:
            entry[key] = details[key]
    return entry
