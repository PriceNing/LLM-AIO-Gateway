import pytest
from fastapi.testclient import TestClient
from main import app
from app.config import load_config
from app.database import init_db, add_admin, upsert_preprocessor
from app.security import create_session, hash_password

client = TestClient(app)


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Use a temporary database and config for each test."""
    db_path = str(tmp_path / "test.db")
    config_path = str(tmp_path / "config.json")
    config = load_config(config_path, force_reload=True)
    config.config = {
        "host": "0.0.0.0",
        "port": 8000,
        "database": db_path,
        "logging": {"enabled": False, "level": "INFO", "log_dir": "logs", "retention_days": 30, "console": False},
    }
    config.save()
    init_db(db_path)
    upsert_preprocessor("vision-model", {
        "api_base": "http://127.0.0.1:8080/v1",
        "model": "test-vision",
        "api_key": "test-key",
        "timeout": 30,
        "max_images": 20,
        "prompt": "Please describe the image content.",
        "enabled": True,
    })
    upsert_preprocessor("disabled-vision", {
        "api_base": "http://127.0.0.1:8081/v1",
        "model": "disabled-vision",
        "api_key": "",
        "timeout": 30,
        "max_images": 20,
        "prompt": "Describe",
        "enabled": False,
    })
    add_admin("admin", hash_password("secret"), "Admin")
    token = create_session("admin")
    yield {"headers": {"Authorization": f"Bearer {token}"}}


def test_login():
    response = client.post("/auth/login", json={"username": "admin", "password": "secret"})
    assert response.status_code == 200
    assert response.json()["token"]


def test_list_providers(temp_db):
    response = client.get("/admin/providers", headers=temp_db["headers"])
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_admin_requires_login():
    response = client.get("/admin/providers")
    assert response.status_code == 401


def test_create_provider(temp_db):
    response = client.post("/admin/providers", json={
        "id": "test-admin-provider",
        "name": "Test Admin Provider",
        "provider_type": "openai",
        "api_base": "https://api.test.com/v1",
        "api_key": "test-key",
        "enabled": True,
        "models": []
    }, headers=temp_db["headers"])
    assert response.status_code == 200
    assert response.json()["id"] == "test-admin-provider"


def test_user_lifecycle(temp_db):
    headers = temp_db["headers"]
    response = client.post("/admin/users", json={
        "username": "alice",
        "display_name": "Alice",
        "enabled": True
    }, headers=headers)
    assert response.status_code == 200

    response = client.post("/admin/users/alice/api-keys", json={"name": "default"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["key"].startswith("sk-aio-")

    response = client.get("/admin/users", headers=headers)
    assert response.status_code == 200
    assert response.json()["users"][0]["username"] == "alice"


def test_setup_first_admin(temp_db):
    # Delete existing admin to simulate clean state
    from app.database import get_db
    with get_db() as db:
        db.execute("DELETE FROM admins")

    response = client.post("/auth/setup", json={"username": "root", "password": "secret"})
    assert response.status_code == 200
    assert response.json()["token"]

    # Restore admin
    from app.database import add_admin
    add_admin("admin", hash_password("secret"), "Admin")


def test_get_stats(temp_db):
    response = client.get("/admin/stats", headers=temp_db["headers"])
    assert response.status_code == 200
    data = response.json()
    assert "total_calls" in data
    assert "success_rate" in data


def test_stats_request_detail_exposes_actual_upstream_endpoint(temp_db):
    import app.router.proxy as proxy

    proxy.clear_request_log()
    try:
        proxy._log_request(
            "alice", "sk-test", "minimax/MiniMax-M3", "minimax",
            "responses", True, 12,
            details={"upstream_endpoint": "messages"},
        )
        response = client.get("/admin/stats", headers=temp_db["headers"])
        assert response.status_code == 200
        entry = response.json()["request_log"][0]
        assert entry["endpoint"] == "responses"
        assert entry["upstream_endpoint"] == "messages"
    finally:
        proxy.clear_request_log()


def test_stats_distinguishes_image_generation(temp_db):
    import app.router.proxy as proxy

    proxy.clear_request_log()
    try:
        proxy._log_request(
            "alice", "sk-test", "chat-model", "chat",
            "responses", True, 7, "chat/chat-model",
            details={
                "responses_mode": "model_driven_image_generation_markdown",
                "upstream_endpoint": "images/generations",
                "image_model": "grok-imagine-image",
                "image_count": 2,
                "image_bytes": 4096,
                "image_artifact_count": 2,
            },
        )
        response = client.get("/admin/stats", headers=temp_db["headers"])
        assert response.status_code == 200
        data = response.json()
        assert data["image_generation_calls"] == 1
        assert data["image_generation_images"] == 2
        assert data["image_generation_bytes"] == 4096
        entry = data["request_log"][0]
        assert entry["request_kind"] == "image_generation"
        assert entry["image_model"] == "grok-imagine-image"
        assert entry["image_count"] == 2
        assert entry["image_artifact_count"] == 2
    finally:
        proxy.clear_request_log()


def test_realtime_stats_timeline_skips_large_idle_gaps(temp_db):
    import app.router.proxy as proxy

    proxy.clear_request_log()
    try:
        proxy._log_request("alice", "sk-test", "model-a", "provider-a", "chat", True, 10)
        with proxy._request_log_lock:
            proxy._request_log[0]["full_time"] = "2026-05-29 10:00:00"
            proxy._request_log[0]["time"] = "10:00:00"

        proxy._log_request("alice", "sk-test", "model-a", "provider-a", "chat", True, 10)
        with proxy._request_log_lock:
            proxy._request_log[0]["full_time"] = "2026-05-29 16:00:00"
            proxy._request_log[0]["time"] = "16:00:00"

        data = proxy.get_timeline_model_data()

        assert data["labels"] == ["10:00", "16:00"]
        assert data["calls"] == [[1, 1]]
    finally:
        proxy.clear_request_log()


def test_realtime_stats_timeline_preserves_small_idle_gaps(temp_db):
    import app.router.proxy as proxy

    proxy.clear_request_log()
    try:
        proxy._log_request("alice", "sk-test", "model-a", "provider-a", "chat", True, 10)
        with proxy._request_log_lock:
            proxy._request_log[0]["full_time"] = "2026-05-29 10:00:00"
            proxy._request_log[0]["time"] = "10:00:00"

        proxy._log_request("alice", "sk-test", "model-a", "provider-a", "chat", False, 0)
        with proxy._request_log_lock:
            proxy._request_log[0]["full_time"] = "2026-05-29 10:02:00"
            proxy._request_log[0]["time"] = "10:02:00"

        timeline = proxy.get_timeline_data()

        assert timeline["labels"] == ["10:00", "10:01", "10:02"]
        assert timeline["success"] == [1, 0, 0]
        assert timeline["failed"] == [0, 0, 1]
    finally:
        proxy.clear_request_log()


# -- Auth edge cases --

def test_auth_status_returns_has_admin():
    response = client.get("/auth/status")
    assert response.status_code == 200
    assert response.json()["has_admin"] is True


def test_login_wrong_password():
    response = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert response.status_code == 401


def test_login_nonexistent_user():
    response = client.post("/auth/login", json={"username": "ghost", "password": "x"})
    assert response.status_code == 401


def test_setup_duplicate_admin():
    """Creating a second admin when one already exists should fail."""
    response = client.post("/auth/setup", json={"username": "root2", "password": "secret"})
    assert response.status_code == 409


def test_logout_invalidates_session(temp_db):
    response = client.post("/auth/logout", headers=temp_db["headers"])
    assert response.status_code == 200
    # Same token should now be rejected
    response2 = client.get("/admin/providers", headers=temp_db["headers"])
    assert response2.status_code == 401


def test_change_password(temp_db):
    response = client.put("/auth/password", headers=temp_db["headers"], json={
        "current_password": "secret",
        "new_password": "new-secret",
    })
    assert response.status_code == 200

    login = client.post("/auth/login", json={"username": "admin", "password": "new-secret"})
    assert login.status_code == 200
    assert login.json()["token"]


def test_me_endpoint(temp_db):
    response = client.get("/auth/me", headers=temp_db["headers"])
    assert response.status_code == 200
    assert response.json()["username"] == "admin"


# -- Provider CRUD edge cases --

def test_update_provider(temp_db):
    client.post("/admin/providers", json={
        "id": "updatable", "name": "Old", "provider_type": "openai",
        "api_base": "", "api_key": "", "enabled": True, "models": []
    }, headers=temp_db["headers"])
    response = client.put("/admin/providers/updatable", json={
        "name": "New Name"
    }, headers=temp_db["headers"])
    assert response.status_code == 200
    assert response.json()["name"] == "New Name"


def test_update_nonexistent_provider(temp_db):
    response = client.put("/admin/providers/nope", json={"name": "X"}, headers=temp_db["headers"])
    assert response.status_code == 404


def test_delete_provider(temp_db):
    client.post("/admin/providers", json={
        "id": "to-delete", "name": "Del", "provider_type": "openai",
        "api_base": "", "api_key": "", "enabled": True, "models": []
    }, headers=temp_db["headers"])
    response = client.delete("/admin/providers/to-delete", headers=temp_db["headers"])
    assert response.status_code == 200


def test_delete_nonexistent_provider(temp_db):
    response = client.delete("/admin/providers/nope", headers=temp_db["headers"])
    assert response.status_code == 404


def test_create_duplicate_provider(temp_db):
    client.post("/admin/providers", json={
        "id": "dup", "name": "Dup", "provider_type": "openai",
        "api_base": "", "api_key": "", "enabled": True, "models": []
    }, headers=temp_db["headers"])
    response = client.post("/admin/providers", json={
        "id": "dup", "name": "Dup2", "provider_type": "openai",
        "api_base": "", "api_key": "", "enabled": True, "models": []
    }, headers=temp_db["headers"])
    assert response.status_code == 400


# -- User CRUD edge cases --

def test_update_user(temp_db):
    client.post("/admin/users", json={"username": "bob", "display_name": "Bob", "enabled": True}, headers=temp_db["headers"])
    response = client.put("/admin/users/bob", json={"display_name": "Bobby"}, headers=temp_db["headers"])
    assert response.status_code == 200
    assert response.json()["display_name"] == "Bobby"


def test_delete_user(temp_db):
    client.post("/admin/users", json={"username": "eve", "display_name": "Eve", "enabled": True}, headers=temp_db["headers"])
    response = client.delete("/admin/users/eve", headers=temp_db["headers"])
    assert response.status_code == 200


def test_delete_nonexistent_user(temp_db):
    response = client.delete("/admin/users/ghost", headers=temp_db["headers"])
    assert response.status_code == 404


def test_update_user_api_key(temp_db):
    client.post("/admin/users", json={"username": "carol", "display_name": "Carol", "enabled": True}, headers=temp_db["headers"])
    key_resp = client.post("/admin/users/carol/api-keys", json={"name": "mykey"}, headers=temp_db["headers"])
    key_value = key_resp.json()["key"]
    response = client.put(f"/admin/users/carol/api-keys/{key_value}", json={"name": "renamed"}, headers=temp_db["headers"])
    assert response.status_code == 200
    assert response.json()["name"] == "renamed"


def test_delete_user_api_key(temp_db):
    client.post("/admin/users", json={"username": "dave", "display_name": "Dave", "enabled": True}, headers=temp_db["headers"])
    key_resp = client.post("/admin/users/dave/api-keys", json={"name": "temp"}, headers=temp_db["headers"])
    key_value = key_resp.json()["key"]
    response = client.delete(f"/admin/users/dave/api-keys/{key_value}", headers=temp_db["headers"])
    assert response.status_code == 200


# -- Routing rules --

def test_create_and_list_routing_rules(temp_db):
    response = client.post("/admin/routing-rules", json={
        "name": "Test Rule", "enabled": True,
        "match_model": "test-*", "target_model": "target",
        "target_provider": ""
    }, headers=temp_db["headers"])
    assert response.status_code == 200
    assert response.json()["match_model"] == "test-*"

    list_resp = client.get("/admin/routing-rules", headers=temp_db["headers"])
    assert list_resp.status_code == 200
    assert len(list_resp.json()["rules"]) == 1


def test_update_routing_rule(temp_db):
    resp = client.post("/admin/routing-rules", json={
        "name": "Old", "enabled": True,
        "match_model": "*", "target_model": "t1", "target_provider": ""
    }, headers=temp_db["headers"])
    rule_id = resp.json()["id"]
    response = client.put(f"/admin/routing-rules/{rule_id}", json={
        "name": "New", "enabled": False
    }, headers=temp_db["headers"])
    assert response.status_code == 200
    assert response.json()["name"] == "New"
    assert response.json()["enabled"] is False


def test_delete_routing_rule(temp_db):
    resp = client.post("/admin/routing-rules", json={
        "name": "Del", "enabled": True,
        "match_model": "*", "target_model": "t", "target_provider": ""
    }, headers=temp_db["headers"])
    rule_id = resp.json()["id"]
    response = client.delete(f"/admin/routing-rules/{rule_id}", headers=temp_db["headers"])
    assert response.status_code == 200


def test_delete_nonexistent_routing_rule(temp_db):
    response = client.delete("/admin/routing-rules/nonexistent", headers=temp_db["headers"])
    assert response.status_code == 404


def test_routing_dry_run_reports_matching_rule_and_provider(temp_db):
    client.post("/admin/providers", json={
        "id": "target-prov", "name": "Target Provider", "provider_type": "openai",
        "api_base": "https://api.test/v1", "api_key": "upstream", "enabled": True,
        "models": [{"id": "target-model", "name": "Target Model", "enabled": True}],
    }, headers=temp_db["headers"])
    rule_resp = client.post("/admin/routing-rules", json={
        "name": "Dry Run Rule", "enabled": True, "username": "alice",
        "api_key_pattern": "secret", "match_model": "source-*",
        "target_model": "target-model", "target_provider": "target-prov",
    }, headers=temp_db["headers"])

    response = client.post("/admin/routing-rules/dry-run", json={
        "username": "alice",
        "api_key": "sk-secret-value",
        "model": "source-model",
    }, headers=temp_db["headers"])

    assert response.status_code == 200
    data = response.json()
    assert data["routing"]["matched"] is True
    assert data["routing"]["rule_id"] == rule_resp.json()["id"]
    assert data["routing"]["target_model"] == "target-model"
    assert data["routing"]["target_provider"] == "target-prov"
    assert data["provider"]["found"] is True
    assert data["provider"]["id"] == "target-prov"
    assert data["effective"]["provider_type"] == "openai"
    assert "secret" not in data["input"]["api_key"]


def test_routing_dry_run_reports_no_match(temp_db):
    response = client.post("/admin/routing-rules/dry-run", json={
        "username": "alice",
        "api_key": "sk-test",
        "model": "plain-model",
    }, headers=temp_db["headers"])

    assert response.status_code == 200
    data = response.json()
    assert data["routing"]["matched"] is False
    assert data["routing"]["target_model"] == "plain-model"
    assert data["provider"]["found"] is False


def test_routing_dry_run_requires_model(temp_db):
    response = client.post("/admin/routing-rules/dry-run", json={"username": "alice"}, headers=temp_db["headers"])
    assert response.status_code == 400


def test_fallback_policy_crud_and_dry_run(temp_db):
    created = client.post("/admin/fallback-policies", json={
        "name": "Primary fallback",
        "enabled": True,
        "match_provider": "primary-provider",
        "match_model": "primary-*",
        "triggers": {"http_5xx": True, "http_4xx": False},
        "chain": [{"provider_id": "backup-provider", "model": "backup-model"}],
    }, headers=temp_db["headers"])
    assert created.status_code == 200
    policy_id = created.json()["id"]

    fetched = client.get(f"/admin/fallback-policies/{policy_id}", headers=temp_db["headers"])
    assert fetched.status_code == 200
    assert fetched.json()["id"] == policy_id

    dry = client.post("/admin/fallback-policies/dry-run", json={
        "provider_id": "primary-provider",
        "model": "primary-model",
        "trigger": "http_5xx",
    }, headers=temp_db["headers"])
    assert dry.status_code == 200
    assert dry.json()["fallback"]["matched"] is True
    assert dry.json()["fallback"]["policy_id"] == policy_id
    assert dry.json()["fallback"]["chain"][0] == {"model": "backup-model", "provider_id": "backup-provider"}

    blocked = client.post("/admin/fallback-policies/dry-run", json={
        "provider_id": "primary-provider",
        "model": "primary-model",
        "trigger": "http_4xx",
    }, headers=temp_db["headers"])
    assert blocked.status_code == 200
    assert blocked.json()["fallback"]["matched"] is False

    updated = client.put(f"/admin/fallback-policies/{policy_id}", json={"enabled": False}, headers=temp_db["headers"])
    assert updated.status_code == 200
    assert updated.json()["enabled"] is False

    deleted = client.delete(f"/admin/fallback-policies/{policy_id}", headers=temp_db["headers"])
    assert deleted.status_code == 200


# -- Stats reset --

def test_reset_stats(temp_db):
    response = client.post("/admin/stats/reset", headers=temp_db["headers"])
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_stats_history(temp_db):
    response = client.get("/admin/stats/history", headers=temp_db["headers"])
    assert response.status_code == 200
    data = response.json()
    assert "overall" in data
    assert "timeline" in data


# -- Model listing --

def test_admin_list_models(temp_db):
    client.post("/admin/providers", json={
        "id": "mp", "name": "MP", "provider_type": "openai",
        "api_base": "", "api_key": "", "enabled": True,
        "models": [{"id": "m-a", "name": "MA", "enabled": True}]
    }, headers=temp_db["headers"])
    response = client.get("/admin/models", headers=temp_db["headers"])
    assert response.status_code == 200
    model_ids = [m["id"] for m in response.json()["models"]]
    assert "mp/m-a" in model_ids


def test_provider_health_endpoint(temp_db, monkeypatch):
    client.post("/admin/providers", json={
        "id": "health-admin", "name": "Health Admin", "provider_type": "openai",
        "api_base": "https://api.test/v1", "api_key": "test", "enabled": True,
        "models": []
    }, headers=temp_db["headers"])

    async def fake_health(provider_id):
        return {"provider_id": provider_id, "ok": True, "status": "ok", "model_count": 1}

    monkeypatch.setattr("app.router.admin.check_provider_health", fake_health)

    response = client.get("/admin/providers/health-admin/health", headers=temp_db["headers"])
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_provider_health_all_endpoint(temp_db, monkeypatch):
    async def fake_health_all():
        return [{"provider_id": "p1", "ok": True, "status": "ok"}]

    monkeypatch.setattr("app.router.admin.check_all_provider_health", fake_health_all)

    response = client.get("/admin/providers/health-all", headers=temp_db["headers"])
    assert response.status_code == 200
    assert response.json()["results"][0]["provider_id"] == "p1"


def test_provider_health_endpoint_missing_provider(temp_db):
    response = client.get("/admin/providers/missing/health", headers=temp_db["headers"])
    assert response.status_code == 404


def test_provider_refresh_routes(monkeypatch, temp_db):
    import app.router.admin as admin_router

    client.post("/admin/providers", json={
        "id": "refreshable", "name": "Refreshable", "provider_type": "openai",
        "api_base": "https://refresh.example/v1", "api_key": "key", "enabled": True,
        "models": [],
    }, headers=temp_db["headers"])

    async def fake_refresh_provider_models(provider_id):
        return {"provider_id": provider_id, "status": "ok"}

    async def fake_refresh_all_providers():
        return [{"provider_id": "refreshable", "status": "ok"}]

    monkeypatch.setattr(admin_router, "refresh_provider_models", fake_refresh_provider_models)
    monkeypatch.setattr(admin_router, "refresh_all_providers", fake_refresh_all_providers)

    one = client.post("/admin/providers/refreshable/refresh", headers=temp_db["headers"])
    assert one.status_code == 200
    assert one.json() == {"provider_id": "refreshable", "status": "ok"}

    all_resp = client.post("/admin/providers/refresh-all", headers=temp_db["headers"])
    assert all_resp.status_code == 200
    assert all_resp.json()["results"] == [{"provider_id": "refreshable", "status": "ok"}]


# -- Preprocessor endpoints --

def test_list_preprocessors(temp_db):
    response = client.get("/admin/preprocessors", headers=temp_db["headers"])
    assert response.status_code == 200
    data = response.json()
    assert "preprocessors" in data
    assert "models" in data
    # Test section
    assert "vision-model" in data["preprocessors"]


def test_list_preprocessors_requires_auth():
    response = client.get("/admin/preprocessors")
    assert response.status_code == 401


def test_update_preprocessor_enable_disables_others(temp_db):
    """Test behavior."""
    headers = temp_db["headers"]
    # Test section
    resp1 = client.put("/admin/preprocessors/disabled-vision", json={
        "enabled": True,
    }, headers=headers)
    assert resp1.status_code == 200
    assert resp1.json()["config"]["enabled"] is True

    # Test section
    resp_list = client.get("/admin/preprocessors", headers=headers)
    preprocessors = resp_list.json()["preprocessors"]
    assert preprocessors["vision-model"]["enabled"] is False
    assert preprocessors["disabled-vision"]["enabled"] is True


def test_update_preprocessor_partial(temp_db):
    """Test behavior."""
    headers = temp_db["headers"]
    resp = client.put("/admin/preprocessors/vision-model", json={
        "max_images": 10,
        "prompt": "New prompt",
    }, headers=headers)
    assert resp.status_code == 200
    config = resp.json()["config"]
    assert config["max_images"] == 10
    assert config["prompt"] == "New prompt"
    # Test section
    assert config["model"] == "test-vision"


def test_delete_preprocessor(temp_db):
    headers = temp_db["headers"]
    resp = client.delete("/admin/preprocessors/disabled-vision", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # Test section
    resp_list = client.get("/admin/preprocessors", headers=headers)
    assert "disabled-vision" not in resp_list.json()["preprocessors"]


def test_delete_nonexistent_preprocessor(temp_db):
    resp = client.delete("/admin/preprocessors/nonexistent", headers=temp_db["headers"])
    assert resp.status_code == 404


def test_toggle_model_preprocessor(temp_db):
    """Test behavior."""
    headers = temp_db["headers"]
    # Test section
    from app.database import add_provider
    add_provider({
        "id": "pp-test", "name": "PP Test", "provider_type": "openai",
        "api_base": "", "api_key": "", "enabled": True,
        "models": [{"id": "pp-model", "name": "PP Model", "enabled": True}]
    })
    # Test section
    resp = client.put("/admin/models/preprocessor", json={
        "model_id": "pp-test/pp-model",
        "enabled": True
    }, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["preprocessor"] is True

    # Test section
    resp_list = client.get("/admin/preprocessors", headers=headers)
    models = resp_list.json()["models"]
    pp_model = next((m for m in models if m["model_id"] == "pp-test/pp-model"), None)
    assert pp_model is not None
    assert pp_model["preprocessor"] is True

    # Test section
    resp2 = client.put("/admin/models/preprocessor", json={
        "model_id": "pp-test/pp-model",
        "enabled": False
    }, headers=headers)
    assert resp2.status_code == 200
    assert resp2.json()["preprocessor"] is False


def test_toggle_model_preprocessor_rejects_non_boolean_enabled(temp_db):
    headers = temp_db["headers"]
    from app.database import add_provider
    add_provider({
        "id": "pp-bool-test", "name": "PP Bool Test", "provider_type": "openai",
        "api_base": "", "api_key": "", "enabled": True,
        "models": [{"id": "pp-model", "name": "PP Model", "enabled": True}]
    })

    resp = client.put("/admin/models/preprocessor", json={
        "model_id": "pp-bool-test/pp-model",
        "enabled": "false"
    }, headers=headers)

    assert resp.status_code == 400
    assert resp.json()["detail"] == "enabled must be a boolean"


def test_fetch_preprocessor_models_requires_auth():
    response = client.get("/admin/preprocessors/fetch-models", params={"api_base": "http://x"})
    assert response.status_code == 401


def test_model_test_requires_auth():
    response = client.post("/admin/models/test", json={"model_id": "p/m"})
    assert response.status_code == 401


def test_model_test_openai_success(temp_db, monkeypatch):
    from types import SimpleNamespace
    from app.database import add_provider

    add_provider({
        "id": "test-openai", "name": "Test OpenAI", "provider_type": "openai",
        "api_base": "http://example.test/v1", "api_key": "test-key", "enabled": True,
        "models": [{"id": "chat-model", "name": "Chat Model", "enabled": True}]
    })

    def fake_create_chat_completion(**kwargs):
        assert kwargs["model"] == "chat-model"
        assert kwargs["provider_id"] == "test-openai"
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="OK", reasoning_content=""),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr("app.router.admin.create_chat_completion", fake_create_chat_completion)

    response = client.post(
        "/admin/models/test",
        json={"model_id": "test-openai/chat-model"},
        headers=temp_db["headers"],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["provider_id"] == "test-openai"
    assert data["model"] == "chat-model"
    assert data["preview"] == "OK"
    assert isinstance(data["latency_ms"], int)


def test_model_test_missing_model_id(temp_db):
    response = client.post("/admin/models/test", json={}, headers=temp_db["headers"])
    assert response.status_code == 400


def test_preprocessor_test_requires_auth():
    response = client.post("/admin/preprocessors/test", json={"preprocessor_id": "vision-model"})
    assert response.status_code == 401


def test_preprocessor_test_success(temp_db, monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "OK vision"}}],
                "usage": {"total_tokens": 3},
            }

    class FakeAsyncClient:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("app.router.admin.httpx.AsyncClient", FakeAsyncClient)

    response = client.post(
        "/admin/preprocessors/test",
        json={"preprocessor_id": "vision-model"},
        headers=temp_db["headers"],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["preprocessor_id"] == "vision-model"
    assert data["model"] == "test-vision"
    assert data["preview"] == "OK vision"
    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["messages"][0]["content"][1]["type"] == "image_url"


def test_preprocessor_test_missing_id(temp_db):
    response = client.post("/admin/preprocessors/test", json={}, headers=temp_db["headers"])
    assert response.status_code == 400
