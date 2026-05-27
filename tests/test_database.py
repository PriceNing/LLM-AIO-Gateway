"""
Unit tests for app.database - stats, provider lookup, routing rules, TTLDict.
"""
import time
import pytest
from app.database import (
    init_db, get_db,
    increment_global_stats, get_global_stats, reset_global_stats,
    find_provider_by_model, parse_model_id,
    add_provider, get_provider, get_providers, update_provider, delete_provider,
    add_routing_rule, get_routing_rules, update_routing_rule, delete_routing_rule,
    add_fallback_policy, get_fallback_policies, update_fallback_policy, delete_fallback_policy,
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
    stats = get_global_stats()
    assert stats["total_calls"] == 3
    assert stats["failed_calls"] == 1


def test_stats_initial_values():
    # Seed defaults by making one call (inserts total_calls/failed_calls if missing)
    increment_global_stats(success=True)
    stats = get_global_stats()
    assert stats["total_calls"] == 1  # one call seeded
    assert stats["failed_calls"] == 0


def test_reset_global_stats():
    increment_global_stats(success=True)
    increment_global_stats(success=False)
    reset_global_stats()
    stats = get_global_stats()
    assert stats["total_calls"] == 0
    assert stats["failed_calls"] == 0
    assert stats["last_reset"] != ""


# -- Provider CRUD --

def test_add_and_get_provider():
    add_provider({
        "id": "test-p", "name": "Test P",
        "provider_type": "openai", "api_base": "https://api.t.com/v1",
        "api_key": "k", "enabled": True,
        "models": [{"id": "m1", "name": "Model 1", "enabled": True}]
    })
    p = get_provider("test-p")
    assert p is not None
    assert p["name"] == "Test P"
    assert p["provider_type"] == "openai"
    assert len(p["models"]) == 1
    assert p["models"][0]["id"] == "m1"


def test_add_duplicate_provider_raises():
    add_provider({"id": "dup", "name": "Dup", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True, "models": []})
    with pytest.raises(ValueError, match="already exists"):
        add_provider({"id": "dup", "name": "Dup2", "provider_type": "openai",
                      "api_base": "", "api_key": "", "enabled": True, "models": []})


def test_update_provider():
    add_provider({"id": "up", "name": "Old", "provider_type": "openai",
                  "api_base": "", "api_key": "", "enabled": True, "models": []})
    result = update_provider("up", {"name": "New Name"})
    assert result["name"] == "New Name"


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
    })
    assert policy["name"] == "Pixel fallback"
    assert policy["enabled"] is True
    assert policy["triggers"]["http_5xx"] is True
    assert policy["triggers"]["http_4xx"] is False
    assert policy["chain"][0]["provider_id"] == "NewAPI"

    policies = get_fallback_policies()
    assert len(policies) == 1

    updated = update_fallback_policy(policy["id"], {
        "enabled": False,
        "chain": [{"model": "deepseek-v4-flash", "provider_id": "deepseek"}],
    })
    assert updated["enabled"] is False
    assert updated["chain"][0]["model"] == "deepseek-v4-flash"
    assert delete_fallback_policy(policy["id"]) is True
    assert delete_fallback_policy(policy["id"]) is False


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
