"""
Unit tests for app.database - stats, provider lookup, routing rules, TTLDict.
"""
import time
from datetime import timedelta, timezone
import pytest
import sqlite3
from app.database import (
    init_db, get_db,
    increment_global_stats, get_global_stats, reset_global_stats,
    get_history_stats,
    find_provider_by_model, parse_model_id,
    add_provider, get_provider, get_providers, update_provider, delete_provider,
    add_routing_rule, get_routing_rules, update_routing_rule, delete_routing_rule,
    add_fallback_policy, get_fallback_policies, update_fallback_policy, delete_fallback_policy,
    delete_preprocessor, get_enabled_preprocessor, get_preprocessors,
    upsert_preprocessor,
    get_model_responses_capability, set_model_responses_capability,
)


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Each test gets a fresh temporary database."""
    import app.database as db_mod
    db_path = str(tmp_path / "test.db")
    # Reset module state
    db_mod._initialized = False
    init_db(db_path)
    yield
    db_mod._initialized = False


# -- Global stats --

def test_increment_and_get_stats():
    increment_global_stats(success=True)
    increment_global_stats(success=True)
    increment_global_stats(success=False)
    increment_global_stats(success=True, degraded=True)
    increment_global_stats(success=False, rejected=True)
    increment_global_stats(success=False, cancelled=True)
    stats = get_global_stats()
    assert stats["total_calls"] == 6
    assert stats["failed_calls"] == 3
    assert stats["degraded_calls"] == 1
    assert stats["rejected_calls"] == 1
    assert stats["cancelled_calls"] == 1


def test_stats_initial_values():
    # Seed defaults by making one call (inserts total_calls/failed_calls if missing)
    increment_global_stats(success=True)
    stats = get_global_stats()
    assert stats["total_calls"] == 1  # one call seeded
    assert stats["failed_calls"] == 0
    assert int(stats.get("degraded_calls", 0) or 0) == 0


def test_reset_global_stats():
    increment_global_stats(success=True)
    increment_global_stats(success=False)
    increment_global_stats(success=True, degraded=True)
    increment_global_stats(success=False, rejected=True)
    increment_global_stats(success=False, cancelled=True)
    reset_global_stats()
    stats = get_global_stats()
    assert stats["total_calls"] == 0
    assert stats["failed_calls"] == 0
    assert stats["degraded_calls"] == 0
    assert stats["rejected_calls"] == 0
    assert stats["cancelled_calls"] == 0
    assert stats["last_reset"] != ""


def test_history_stats_queries_local_day_against_utc_records(monkeypatch):
    import app.database as db_mod

    monkeypatch.setattr(db_mod, "_local_tz", lambda: timezone(timedelta(hours=8)))
    with get_db() as db:
        db.execute(
            "INSERT INTO request_records (timestamp, model, username, success, tokens) VALUES (?, ?, ?, ?, ?)",
            ("2026-05-29 08:30:00", "model-a", "alice", 1, 123),
        )

    stats = get_history_stats("2026-05-29", "2026-05-29", "hour")

    assert stats["overall"] == {"total_calls": 1, "failed_calls": 0, "total_tokens": 123}
    assert stats["timeline"]["labels"][16] == "2026-05-29 16:00"
    assert stats["timeline"]["total"][16] == 1
    assert stats["timeline"]["tokens"][16] == 123


# -- Provider CRUD --


def test_legacy_provider_responses_columns_are_physically_removed(tmp_path):
    """Existing provider-level capability fields migrate to model-level fields only."""
    import app.database as db_mod

    legacy_path = tmp_path / "legacy-responses.db"
    with sqlite3.connect(legacy_path) as conn:
        conn.executescript(
            """
            CREATE TABLE providers (
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
                responses_status TEXT NOT NULL DEFAULT 'unknown',
                responses_checked_at TEXT NOT NULL DEFAULT '',
                responses_streaming INTEGER NOT NULL DEFAULT 0,
                responses_streaming_status TEXT NOT NULL DEFAULT 'unknown',
                responses_tool_types TEXT NOT NULL DEFAULT '[]',
                responses_error TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE provider_models (
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
            """
        )
        conn.execute(
            "INSERT INTO providers (id, name, responses_status) VALUES (?, ?, ?)",
            ("legacy", "Legacy provider", "supported"),
        )
        conn.execute(
            "INSERT INTO provider_models (provider_id, model_id, model_name) VALUES (?, ?, ?)",
            ("legacy", "legacy-model", "Legacy model"),
        )

    db_mod._initialized = False
    init_db(str(legacy_path))

    with sqlite3.connect(legacy_path) as conn:
        provider_columns = {row[1] for row in conn.execute("PRAGMA table_info(providers)")}
        model_columns = {row[1] for row in conn.execute("PRAGMA table_info(provider_models)")}
        provider_row = conn.execute("SELECT id, name FROM providers WHERE id = ?", ("legacy",)).fetchone()

    legacy_columns = {
        "responses_status", "responses_checked_at", "responses_streaming",
        "responses_streaming_status", "responses_tool_types", "responses_error",
    }
    assert provider_columns.isdisjoint(legacy_columns)
    assert provider_row == ("legacy", "Legacy provider")
    assert {
        "responses_status", "responses_checked_at", "responses_expires_at",
        "responses_streaming", "responses_streaming_status",
        "responses_tool_types", "responses_error",
    }.issubset(model_columns)

    set_model_responses_capability("legacy", "legacy-model", status="supported")
    capability = get_model_responses_capability("legacy", "legacy-model")
    assert capability["responses_status"] == "supported"
    assert "responses_status" not in get_provider("legacy")


def test_add_and_get_provider():
    add_provider({
        "id": "test-p", "name": "Test P",
        "provider_type": "openai", "api_base": "https://api.t.com/v1",
        "api_key": "k", "enabled": True,
        "request_timeout": 45, "retry_count": 2, "retry_backoff": 1.5,
        "models": [{"id": "m1", "name": "Model 1", "enabled": True, "preprocessor": "1"}]
    })
    p = get_provider("test-p")
    assert p is not None
    assert p["name"] == "Test P"
    assert p["provider_type"] == "openai"
    assert len(p["models"]) == 1
    assert p["models"][0]["id"] == "m1"
    assert p["models"][0]["preprocessor"] == "1"
    assert p["request_timeout"] == 45
    assert p["retry_count"] == 2
    assert p["retry_backoff"] == 1.5


def test_add_duplicate_provider_raises():
    add_provider({"id": "dup", "name": "Dup", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True, "models": []})
    with pytest.raises(ValueError, match="already exists"):
        add_provider({"id": "dup", "name": "Dup2", "provider_type": "openai",
                      "api_base": "", "api_key": "", "enabled": True, "models": []})


def test_update_provider():
    add_provider({"id": "up", "name": "Old", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True, "models": []})
    result = update_provider("up", {"name": "New Name", "request_timeout": 60, "retry_count": 3, "retry_backoff": 2})
    assert result["name"] == "New Name"
    assert result["request_timeout"] == 60
    assert result["retry_count"] == 3
    assert result["retry_backoff"] == 2


def test_update_nonexistent_provider():
    assert update_provider("nope", {"name": "X"}) is None


def test_delete_provider_cascade():
    add_provider({"id": "del-p", "name": "Del", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True,
                  "models": [{"id": "mm", "name": "M", "enabled": True}]})
    assert get_provider("del-p") is not None
    result = delete_provider("del-p")
    assert result is True
    assert get_provider("del-p") is None
    # Models should be cascade-deleted
    from app.database import find_provider_by_model
    assert find_provider_by_model("mm") is None


def test_delete_nonexistent_provider():
    assert delete_provider("ghost") is False


# -- find_provider_by_model (ORDER BY fix verification) --

def test_find_provider_by_model_first_match_order():
    """When two providers have the same model, the one with lower id wins."""
    add_provider({"id": "aaa", "name": "AAA", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True,
                  "models": [{"id": "shared", "name": "Shared", "enabled": True}]})
    add_provider({"id": "zzz", "name": "ZZZ", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True,
                  "models": [{"id": "shared", "name": "Shared", "enabled": True}]})
    provider = find_provider_by_model("shared")
    assert provider is not None
    assert provider["id"] == "aaa"  # ORDER BY p.id -> aaa before zzz


def test_find_provider_by_model_not_found():
    assert find_provider_by_model("nonexistent-model") is None


def test_find_provider_by_model_disabled_provider_ignored():
    add_provider({"id": "disabled-p", "name": "Disabled", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": False,
                  "models": [{"id": "hidden", "name": "Hidden", "enabled": True}]})
    assert find_provider_by_model("hidden") is None


# -- parse_model_id --

def test_parse_model_id_simple():
    mid = parse_model_id("gpt-4")
    assert mid.provider_id == ""
    assert mid.model_name == "gpt-4"
    assert mid.composite == "gpt-4"
    assert not mid.is_composite


def test_parse_model_id_composite():
    mid = parse_model_id("openai/gpt-4")
    assert mid.provider_id == "openai"
    assert mid.model_name == "gpt-4"
    assert mid.composite == "openai/gpt-4"
    assert mid.is_composite


def test_parse_model_id_nested_path():
    """Test behavior."""
    mid = parse_model_id("deepseek/deepseek-v4/pro")
    assert mid.provider_id == "deepseek"
    assert mid.model_name == "deepseek-v4/pro"


def test_parse_model_id_empty():
    mid = parse_model_id("")
    assert mid.provider_id == ""
    assert mid.model_name == ""
    assert mid.composite == ""
    assert not mid.is_composite


# -- find_provider_by_model with composite ID --

def test_find_provider_by_model_composite_id():
    """Test behavior."""
    add_provider({"id": "p-a", "name": "A", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True,
                  "models": [{"id": "shared", "name": "Shared A", "enabled": True}]})
    add_provider({"id": "p-b", "name": "B", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True,
                  "models": [{"id": "shared", "name": "Shared B", "enabled": True}]})
    # Test section
    provider = find_provider_by_model("p-b/shared")
    assert provider is not None
    assert provider["id"] == "p-b"
    assert provider["name"] == "B"


def test_find_provider_by_model_composite_nonexistent_provider():
    """Test behavior."""
    add_provider({"id": "p-a", "name": "A", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True,
                  "models": [{"id": "m1", "name": "M1", "enabled": True}]})
    assert find_provider_by_model("nonexistent/m1") is None


def test_find_provider_by_model_composite_nonexistent_model():
    """Test behavior."""
    add_provider({"id": "p-a", "name": "A", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True,
                  "models": [{"id": "m1", "name": "M1", "enabled": True}]})
    assert find_provider_by_model("p-a/nonexistent") is None


# -- Routing rules CRUD --

def test_add_and_list_routing_rules():
    rule = add_routing_rule({
        "name": "Test Rule", "enabled": True,
        "username": "", "api_key_pattern": "",
        "match_model": "test-*", "target_model": "target-model",
        "target_provider": "",
    })
    assert rule["name"] == "Test Rule"
    assert "fallback_models" not in rule
    rules = get_routing_rules()
    assert len(rules) == 1
    assert rules[0]["match_model"] == "test-*"


def test_update_routing_rule():
    rule = add_routing_rule({
        "name": "Old Rule", "enabled": True,
        "username": "", "api_key_pattern": "",
        "match_model": "*", "target_model": "t1", "target_provider": ""
    })
    updated = update_routing_rule(rule["id"], {"name": "New Rule", "enabled": False, "fallback_models": ["t2"]})
    assert updated["name"] == "New Rule"
    assert updated["enabled"] is False
    assert "fallback_models" not in updated


def test_add_update_delete_fallback_policy():
    policy = add_fallback_policy({
        "name": "Pixel fallback",
        "match_provider": "PixelAPI",
        "match_model": "gpt-5.5",
        "triggers": {"http_5xx": True, "http_4xx": False},
        "chain": [{"model": "gpt-5.5", "provider_id": "NewAPI"}],
        "attempt_timeout": 45,
    })
    assert policy["name"] == "Pixel fallback"
    assert policy["enabled"] is True
    assert policy["triggers"]["http_5xx"] is True
    assert policy["triggers"]["http_4xx"] is False
    assert policy["chain"][0]["provider_id"] == "NewAPI"
    assert policy["attempt_timeout"] == 45

    policies = get_fallback_policies()
    assert len(policies) == 1

    updated = update_fallback_policy(policy["id"], {
        "enabled": False,
        "chain": [{"model": "deepseek-v4-flash", "provider_id": "deepseek"}],
        "attempt_timeout": 90,
    })
    assert updated["enabled"] is False
    assert updated["chain"][0]["model"] == "deepseek-v4-flash"
    assert updated["attempt_timeout"] == 90
    assert delete_fallback_policy(policy["id"]) is True
    assert delete_fallback_policy(policy["id"]) is False


def test_fallback_attempt_timeout_defaults_and_clamps():
    policy = add_fallback_policy({
        "name": "default timeout",
        "match_model": "*",
        "chain": [{"model": "m", "provider_id": "p"}],
    })
    assert policy["attempt_timeout"] == 60

    too_low = update_fallback_policy(policy["id"], {"attempt_timeout": 1})
    assert too_low["attempt_timeout"] == 5

    too_high = update_fallback_policy(policy["id"], {"attempt_timeout": 99999})
    assert too_high["attempt_timeout"] == 3600


def test_delete_routing_rule():
    rule = add_routing_rule({
        "name": "To Delete", "enabled": True,
        "username": "", "api_key_pattern": "",
        "match_model": "*", "target_model": "t", "target_provider": ""
    })
    assert delete_routing_rule(rule["id"]) is True
    assert delete_routing_rule(rule["id"]) is False  # already gone


# -- TTLDict --

def test_ttldict_basic():
    from app.core.state import TTLDict
    d = TTLDict(ttl_seconds=10, max_size=10)
    d["key1"] = "value1"
    assert d["key1"] == "value1"
    assert d.get("key1") == "value1"
    assert len(d) == 1


def test_ttldict_expiry():
    from app.core.state import TTLDict
    d = TTLDict(ttl_seconds=0, max_size=10)  # instant expiry
    d["ephemeral"] = "gone"
    time.sleep(0.01)
    assert d.get("ephemeral") is None
    with pytest.raises(KeyError):
        _ = d["ephemeral"]


def test_ttldict_max_size_eviction():
    from app.core.state import TTLDict
    d = TTLDict(ttl_seconds=3600, max_size=3)
    for i in range(5):
        d[f"key{i}"] = f"value{i}"
    assert len(d) == 3  # only 3 survive


def test_ttldict_get_default():
    from app.core.state import TTLDict
    d = TTLDict(ttl_seconds=3600, max_size=10)
    assert d.get("missing", "default") == "default"


def test_zero_pad_timeline_month_does_not_overflow():
    from app.database import _zero_pad_timeline
    rows = [{"bucket": "2026-01", "total": 1, "failed": 0, "tokens": 5}]
    model_rows = [{"bucket": "2026-01", "model": "m1", "total": 1, "failed": 0, "tokens": 5}]
    padded_rows, labels, padded_model_rows = _zero_pad_timeline(
        rows, "2026-01-31 00:00:00", "2026-03-31 23:59:59", "month", model_rows
    )
    assert labels == ["2026-01", "2026-02", "2026-03"]
    assert len(padded_rows) == 3
    assert len(padded_model_rows) == 1

# --- db/routing.py helpers ---

def test_routing_json_loads_valid():
    from app.db.routing import json_loads
    assert json_loads('[1, 2]') == [1, 2]
    assert json_loads('{"a": 1}') == {"a": 1}

def test_routing_json_loads_invalid():
    from app.db.routing import json_loads
    assert json_loads('{bad}') is None
    assert json_loads('') is None
    assert json_loads(None) is None

def test_routing_json_dumps_list():
    from app.db.routing import json_dumps_list
    assert json_dumps_list([1, 2]) == '[1, 2]'
    assert json_dumps_list('already string') == 'already string'
    assert json_dumps_list(None) == '[]'
    assert json_dumps_list(42) == '[]'

def test_routing_to_bool():
    from app.db.routing import to_bool
    assert to_bool(True) is True
    assert to_bool(False) is False
    assert to_bool(1) is True
    assert to_bool(0) is False
    assert to_bool('true') is True
    assert to_bool('True') is True
    assert to_bool('1') is True
    assert to_bool('yes') is True
    assert to_bool('false') is False
    assert to_bool('no') is False
    assert to_bool(None) is False

def test_routing_row_to_dict():
    from app.db.routing import row_to_dict
    assert row_to_dict(None) is None

def test_routing_normalize_rule_none():
    from app.db.routing import normalize_rule
    assert normalize_rule(None) is None


# --- db/fallback.py helpers ---

def test_fallback_json_loads_with_fallback():
    from app.db.fallback import json_loads
    assert json_loads('[1]', []) == [1]
    assert json_loads('', []) == []
    assert json_loads(None, [42]) == [42]
    assert json_loads('{bad}', {'x': 1}) == {'x': 1}

def test_fallback_json_dumps():
    from app.db.fallback import json_dumps
    assert json_dumps([1, 2], []) == '[1, 2]'
    assert json_dumps('already', []) == 'already'
    assert json_dumps(None, [42]) == '[42]'

def test_fallback_to_bool():
    from app.db.fallback import to_bool
    assert to_bool(True) is True
    assert to_bool(False) is False
    assert to_bool(1) is True
    assert to_bool(0) is False
    assert to_bool('true') is True
    assert to_bool('1') is True
    assert to_bool(None) is False

def test_fallback_normalize_policy_none():
    from app.db.fallback import normalize_policy
    assert normalize_policy(None) is None

def test_fallback_default_triggers():
    from app.db.fallback import DEFAULT_TRIGGERS
    assert DEFAULT_TRIGGERS['timeout'] is True
    assert DEFAULT_TRIGGERS['connection_error'] is True
    assert DEFAULT_TRIGGERS['http_429'] is True
    assert DEFAULT_TRIGGERS['http_5xx'] is True
    assert DEFAULT_TRIGGERS['http_4xx'] is False


# --- preprocessors ---

def test_preprocessor_defaults_and_single_enabled():
    first = upsert_preprocessor("vision-a", {"api_base": "http://a", "model": "va"})
    assert first["timeout"] == 120
    assert first["max_images"] == 10
    assert first["max_tokens"] == 2048
    assert first["enabled"] is True

    second = upsert_preprocessor("vision-b", {"api_base": "http://b", "model": "vb", "enabled": True})
    assert second["enabled"] is True
    preprocessors = get_preprocessors()
    assert preprocessors["vision-a"]["enabled"] is False
    assert preprocessors["vision-b"]["enabled"] is True
    assert get_enabled_preprocessor()["id"] == "vision-b"


def test_delete_preprocessor_config():
    upsert_preprocessor("vision-a", {"api_base": "http://a", "model": "va"})
    assert delete_preprocessor("vision-a") is True
    assert delete_preprocessor("vision-a") is False
