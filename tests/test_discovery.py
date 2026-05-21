from app.services.discovery import (
    model_list_urls,
    parse_models,
    auth_headers,
)


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


# ── Auth headers ──

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
    """Empty api_key should return [{}] — no auth header, not 'Bearer '."""
    headers = auth_headers("", "openai")
    assert headers == [{}]
