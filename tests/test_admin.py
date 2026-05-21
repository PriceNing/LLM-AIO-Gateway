import pytest
from fastapi.testclient import TestClient
from main import app
from app.config import load_config
from app.database import init_db, add_admin
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
        "preprocessors": {
            "vision-model": {
                "api_base": "http://127.0.0.1:8080/v1",
                "model": "test-vision",
                "api_key": "test-key",
                "timeout": 30,
                "max_images": 20,
                "prompt": "请描述图片中的内容。",
                "enabled": True,
            },
            "disabled-vision": {
                "api_base": "http://127.0.0.1:8081/v1",
                "model": "disabled-vision",
                "api_key": "",
                "timeout": 30,
                "max_images": 20,
                "prompt": "Describe",
                "enabled": False,
            },
        }
    }
    config.save()
    init_db(db_path)
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


# ── Auth edge cases ──

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


def test_me_endpoint(temp_db):
    response = client.get("/auth/me", headers=temp_db["headers"])
    assert response.status_code == 200
    assert response.json()["username"] == "admin"


# ── Provider CRUD edge cases ──

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


# ── User CRUD edge cases ──

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


# ── Routing rules ──

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


# ── Stats reset ──

def test_reset_stats(temp_db):
    response = client.post("/admin/stats/reset", headers=temp_db["headers"])
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── Model listing ──

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


# ── Preprocessor endpoints ──

def test_list_preprocessors(temp_db):
    response = client.get("/admin/preprocessors", headers=temp_db["headers"])
    assert response.status_code == 200
    data = response.json()
    assert "preprocessors" in data
    assert "models" in data
    # 预处理器配置应包含测试用的 vision-model
    assert "vision-model" in data["preprocessors"]


def test_list_preprocessors_requires_auth():
    response = client.get("/admin/preprocessors")
    assert response.status_code == 401


def test_update_preprocessor_enable_disables_others(temp_db):
    """启用一个预处理器时应自动禁用其余所有。"""
    headers = temp_db["headers"]
    # 先确保有两个预处理器，且 vision-model 启用、disabled-vision 禁用
    resp1 = client.put("/admin/preprocessors/disabled-vision", json={
        "enabled": True,
    }, headers=headers)
    assert resp1.status_code == 200
    assert resp1.json()["config"]["enabled"] is True

    # 验证 vision-model 已被自动禁用
    resp_list = client.get("/admin/preprocessors", headers=headers)
    preprocessors = resp_list.json()["preprocessors"]
    assert preprocessors["vision-model"]["enabled"] is False
    assert preprocessors["disabled-vision"]["enabled"] is True


def test_update_preprocessor_partial(temp_db):
    """部分更新预处理器配置（只更新某些字段）。"""
    headers = temp_db["headers"]
    resp = client.put("/admin/preprocessors/vision-model", json={
        "max_images": 10,
        "prompt": "New prompt",
    }, headers=headers)
    assert resp.status_code == 200
    config = resp.json()["config"]
    assert config["max_images"] == 10
    assert config["prompt"] == "New prompt"
    # 未指定的字段应保持不变
    assert config["model"] == "test-vision"


def test_delete_preprocessor(temp_db):
    headers = temp_db["headers"]
    resp = client.delete("/admin/preprocessors/disabled-vision", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # 确认已删除
    resp_list = client.get("/admin/preprocessors", headers=headers)
    assert "disabled-vision" not in resp_list.json()["preprocessors"]


def test_delete_nonexistent_preprocessor(temp_db):
    resp = client.delete("/admin/preprocessors/nonexistent", headers=temp_db["headers"])
    assert resp.status_code == 404


def test_toggle_model_preprocessor(temp_db):
    """切换模型级别的预处理器开关（使用复合 ID）。"""
    headers = temp_db["headers"]
    # 先添加一个 provider 和 model
    from app.database import add_provider
    add_provider({
        "id": "pp-test", "name": "PP Test", "provider_type": "openai",
        "api_base": "", "api_key": "", "enabled": True,
        "models": [{"id": "pp-model", "name": "PP Model", "enabled": True}]
    })
    # 启用预处理器（使用复合 ID）
    resp = client.put("/admin/models/preprocessor", json={
        "model_id": "pp-test/pp-model",
        "enabled": True
    }, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["preprocessor"] is True

    # 验证 models 列表中反映了变更（复合 ID）
    resp_list = client.get("/admin/preprocessors", headers=headers)
    models = resp_list.json()["models"]
    pp_model = next((m for m in models if m["model_id"] == "pp-test/pp-model"), None)
    assert pp_model is not None
    assert pp_model["preprocessor"] is True

    # 禁用预处理器
    resp2 = client.put("/admin/models/preprocessor", json={
        "model_id": "pp-test/pp-model",
        "enabled": False
    }, headers=headers)
    assert resp2.status_code == 200
    assert resp2.json()["preprocessor"] is False


def test_fetch_preprocessor_models_requires_auth():
    response = client.get("/admin/preprocessors/fetch-models", params={"api_base": "http://x"})
    assert response.status_code == 401
