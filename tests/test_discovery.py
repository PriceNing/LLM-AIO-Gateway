import pytest

from app.services.discovery import (
    model_list_urls,
    parse_models,
    auth_headers,
    check_provider_health,
    refresh_provider_models,
)
from app.database import add_provider, get_provider, get_db, init_db


def _set_source(pid, model_id, source):
    with get_db() as db:
        db.execute("UPDATE provider_models SET source = ? WHERE provider_id = ? AND model_id = ?",
                   (source, pid, model_id))


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Keep provider discovery tests isolated from the developer database."""
    import app.database as db_mod

    previous_path = db_mod.DB_PATH
    previous_initialized = db_mod._initialized
    db_mod._initialized = False
    init_db(str(tmp_path / "discovery.db"))
    try:
        yield
    finally:
        db_mod.DB_PATH = previous_path
        db_mod._initialized = previous_initialized


def test_anthropic_model_urls_add_v1_when_missing():
    urls = model_list_urls("https://api.minimaxi.com/anthropic", "anthropic")
    assert urls[0] == "https://api.minimaxi.com/anthropic/v1/models"
    assert urls[1] == "https://api.minimaxi.com/anthropic/models"
    # Parent-path fallback for Anthropic-compatible endpoints
    assert urls[2] == "https://api.minimaxi.com/v1/models"
    assert urls[3] == "https://api.minimaxi.com/models"


def test_model_list_urls_openai_type():
    urls = model_list_urls("https://api.openai.com/v1", "openai")
    assert urls == ["https://api.openai.com/v1/models"]


def test_model_list_urls_anthropic_with_v1_suffix():
    """Anthropic type with api_base ending in /v1 should not append another /v1,
    but should still try parent path for Anthropic-compatible endpoints."""
    urls = model_list_urls("https://api.minimaxi.com/v1", "anthropic")
    assert urls == ["https://api.minimaxi.com/v1/models", "https://api.minimaxi.com/models"]


def test_model_list_urls_trailing_slash_handled():
    urls = model_list_urls("https://api.example.com/v1/", "openai")
    assert urls == ["https://api.example.com/v1/models"]


def test_parse_models_handles_null_data():
    assert parse_models({"data": None}) == []


def test_parse_models_handles_missing_data():
    assert parse_models({}) == []


def test_parse_models_handles_empty_data_list():
    assert parse_models({"data": []}) == []


def test_parse_models_extracts_correctly():
    data = {
        "data": [
            {"id": "model-1", "display_name": "Model One"},
            {"id": "model-2"},
        ]
    }
    models = parse_models(data)
    assert len(models) == 2
    assert models[0]["id"] == "model-1"
    assert models[0]["name"] == "Model One"
    assert models[1]["name"] == "model-2"  # fallback to id


def test_parse_models_handles_models_field():
    """Some APIs use 'models' instead of 'data'."""
    data = {"models": [{"id": "alt-model", "name": "Alt"}]}
    models = parse_models(data)
    assert len(models) == 1
    assert models[0]["id"] == "alt-model"


def test_parse_models_skips_non_dict_items():
    data = {"data": ["string-item", {"id": "valid"}, 123]}
    models = parse_models(data)
    assert len(models) == 1
    assert models[0]["id"] == "valid"


# -- Auth headers --

def test_auth_headers_with_key():
    headers = auth_headers("sk-test", "openai")
    assert len(headers) == 1
    assert headers[0] == {"Authorization": "Bearer sk-test"}


def test_auth_headers_anthropic_extra_header():
    headers = auth_headers("sk-test", "anthropic")
    assert len(headers) == 2
    assert headers[0] == {"Authorization": "Bearer sk-test"}
    assert headers[1] == {"x-api-key": "sk-test", "anthropic-version": "2023-06-01"}


def test_auth_headers_empty_key():
    """Empty api_key should return [{}] - no auth header, not 'Bearer '."""
    headers = auth_headers("", "openai")
    assert headers == [{}]


@pytest.mark.asyncio
async def test_check_provider_health_ok(monkeypatch):
    add_provider({
        "id": "health-ok",
        "name": "Health OK",
        "provider_type": "openai",
        "api_base": "https://api.example/v1",
        "api_key": "sk-test",
        "enabled": True,
        "models": [],
    })

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "m1"}]}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers=None):
            return Response()

    monkeypatch.setattr("app.services.discovery.httpx.AsyncClient", Client)

    result = await check_provider_health("health-ok")
    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["status_code"] == 200
    assert result["model_count"] == 1
    assert result["checked_url"] == "https://api.example/v1/models"


@pytest.mark.asyncio
async def test_check_provider_health_error(monkeypatch):
    add_provider({
        "id": "health-bad",
        "name": "Health Bad",
        "provider_type": "openai",
        "api_base": "https://api.example/v1",
        "api_key": "sk-test",
        "enabled": True,
        "models": [],
    })

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers=None):
            raise RuntimeError("boom")

    monkeypatch.setattr("app.services.discovery.httpx.AsyncClient", Client)

    result = await check_provider_health("health-bad")
    assert result["ok"] is False
    assert result["status"] == "error"
    assert "boom" in result["error"]
    assert result["attempts"][0]["url"] == "https://api.example/v1/models"


@pytest.mark.asyncio
async def test_refresh_provider_models_replaces_stale_models(monkeypatch):
    add_provider({
        "id": "refresh-sync",
        "name": "Refresh Sync",
        "provider_type": "openai",
        "api_base": "https://api.example/v1",
        "api_key": "sk-test",
        "enabled": True,
        "models": [
            {"id": "old-model", "name": "Old Model", "enabled": True},
            {"id": "renamed-model", "name": "Before", "enabled": False},
        ],
    })
    # Models seeded by add_provider are manual by default (rule 1).
    # Flip them to auto so they behave like a previous refresh discovered them.
    _set_source("refresh-sync", "old-model", "auto")
    _set_source("refresh-sync", "renamed-model", "auto")

    async def discover(_provider_id):
        return [
            {"id": "renamed-model", "name": "After"},
            {"id": "new-model", "name": "New Model"},
        ]

    monkeypatch.setattr("app.services.discovery.discover_models", discover)

    result = await refresh_provider_models("refresh-sync")
    provider = get_provider("refresh-sync")
    models = {m["id"]: m for m in provider["models"]}

    assert result["count"] == 2
    assert result["added"] == 1
    assert result["updated"] == 1
    assert result["removed"] == 1
    assert set(models) == {"renamed-model", "new-model"}
    assert models["renamed-model"]["name"] == "After"
    assert models["renamed-model"]["enabled"] is False
    assert models["new-model"]["enabled"] is True


@pytest.mark.asyncio
async def test_discover_models_follows_anthropic_pagination(monkeypatch):
    from app.services.discovery import discover_models

    add_provider({
        "id": "paged-anthropic",
        "name": "Paged Anthropic",
        "provider_type": "anthropic",
        "api_base": "https://api.example/v1",
        "api_key": "sk-test",
        "enabled": True,
        "models": [],
    })
    calls = []

    class Response:
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, **kwargs):
            calls.append(kwargs.get("params"))
            if kwargs.get("params"):
                return Response({"data": [{"id": "model-2"}], "has_more": False})
            return Response({"data": [{"id": "model-1"}], "has_more": True, "last_id": "model-1"})

    monkeypatch.setattr("app.services.discovery.httpx.AsyncClient", Client)

    models = await discover_models("paged-anthropic")

    assert [model["id"] for model in models] == ["model-1", "model-2"]
    assert calls == [None, {"after_id": "model-1"}]
