from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _registered_routes() -> set[tuple[str, str]]:
    routes = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        for method in getattr(route, "methods", set()) or set():
            if method != "HEAD":
                routes.add((method, path))
    return routes


def test_expected_http_routes_are_registered():
    expected = {
        ("GET", "/openapi.json"),
        ("GET", "/docs"),
        ("GET", "/docs/oauth2-redirect"),
        ("GET", "/redoc"),
        ("GET", "/auth/status"),
        ("POST", "/auth/setup"),
        ("POST", "/auth/login"),
        ("GET", "/auth/me"),
        ("POST", "/auth/logout"),
        ("PUT", "/auth/password"),
        ("GET", "/admin/providers"),
        ("POST", "/admin/providers"),
        ("PUT", "/admin/providers/{provider_id}"),
        ("DELETE", "/admin/providers/{provider_id}"),
        ("POST", "/admin/providers/{provider_id}/refresh"),
        ("POST", "/admin/providers/refresh-all"),
        ("GET", "/admin/providers/health-all"),
        ("GET", "/admin/providers/{provider_id}/health"),
        ("GET", "/admin/models"),
        ("GET", "/admin/users"),
        ("POST", "/admin/users"),
        ("PUT", "/admin/users/{username}"),
        ("DELETE", "/admin/users/{username}"),
        ("POST", "/admin/users/{username}/api-keys"),
        ("PUT", "/admin/users/{username}/api-keys/{key}"),
        ("DELETE", "/admin/users/{username}/api-keys/{key}"),
        ("GET", "/admin/stats"),
        ("GET", "/admin/stats/history"),
        ("POST", "/admin/stats/reset"),
        ("GET", "/admin/routing-rules"),
        ("POST", "/admin/routing-rules"),
        ("PUT", "/admin/routing-rules/{rule_id}"),
        ("DELETE", "/admin/routing-rules/{rule_id}"),
        ("POST", "/admin/routing-rules/dry-run"),
        ("GET", "/admin/fallback-policies"),
        ("POST", "/admin/fallback-policies"),
        ("GET", "/admin/fallback-policies/{policy_id}"),
        ("PUT", "/admin/fallback-policies/{policy_id}"),
        ("DELETE", "/admin/fallback-policies/{policy_id}"),
        ("POST", "/admin/fallback-policies/dry-run"),
        ("GET", "/admin/preprocessors"),
        ("PUT", "/admin/preprocessors/{preprocessor_id}"),
        ("DELETE", "/admin/preprocessors/{preprocessor_id}"),
        ("GET", "/admin/preprocessors/fetch-models"),
        ("PUT", "/admin/models/preprocessor"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/responses"),
        ("GET", "/models"),
        ("POST", "/chat/completions"),
        ("POST", "/completions"),
        ("POST", "/messages"),
        ("POST", "/responses"),
    }

    assert expected <= _registered_routes()


def test_documentation_routes_serve_http_responses():
    for path in ("/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"):
        response = client.get(path)
        assert response.status_code == 200
