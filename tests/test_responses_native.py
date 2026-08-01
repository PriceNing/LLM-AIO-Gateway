import pytest
import httpx

from app.adapters.responses import iter_sse_frames, native_responses_body, responses_url, sse_payload
from app.core.types import InternalRequest
from app.protocols.ingress import responses_to_internal
from app.router.proxy import (
    _native_response_with_fallbacks,
    _observed_response_tool_types,
    _responses_requires_native,
    _responses_required_tool_types,
    _native_response_target_supported,
)
from app.core.policy import RouteTarget
from app.config import load_config
from app.database import (
    add_fallback_policy,
    add_provider,
    init_db,
    set_provider_responses_capability,
)
from main import app


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    """Keep native Responses tests independent from the workspace data.db."""
    import app.database as db_mod

    db_path = str(tmp_path / "test.db")
    config = load_config(str(tmp_path / "config.json"), force_reload=True)
    config.config = {
        "host": "0.0.0.0", "port": 8000, "database": db_path,
        "logging": {"enabled": False, "level": "INFO", "log_dir": "logs", "retention_days": 30, "console": False},
    }
    config.save()
    db_mod._initialized = False
    init_db(db_path)
    yield
    db_mod._initialized = False


def test_native_responses_body_keeps_codex_tool_and_rewrites_routed_model():
    internal = responses_to_internal({
        "model": "client-model",
        "input": "inspect the workspace",
        "tools": [{"type": "computer", "name": "computer", "display_width": 1024}],
        "parallel_tool_calls": True,
        "include": ["computer_call_output"],
    })
    internal.target_model = "provider/gpt-5"

    body = native_responses_body(internal)

    assert body["model"] == "gpt-5"
    assert body["tools"][0]["type"] == "computer"
    assert body["tools"][0]["display_width"] == 1024
    assert body["include"] == ["computer_call_output"]
    assert body["parallel_tool_calls"] is True


def test_native_responses_url_handles_standard_v1_base():
    assert responses_url("https://example.test/v1") == "https://example.test/v1/responses"


def test_non_native_features_are_rejected_from_chat_downgrade():
    assert _responses_requires_native({"tools": [{"type": "computer"}]}) == ["tool:computer"]
    assert _responses_requires_native({"tools": [{"type": "function", "name": "x"}]}) == []


def test_responses_previous_id_is_not_sent_as_chat_extra():
    internal = responses_to_internal({"model": "m", "input": "hello", "previous_response_id": "resp_previous"})
    assert internal.previous_response_id == "resp_previous"
    assert "previous_response_id" not in internal.extra


def test_required_tool_types_are_extracted_for_capability_matching():
    assert _responses_required_tool_types({"tools": [{"type": "computer"}, {"type": "custom"}]}) == {"computer", "custom"}


def test_observed_tool_types_require_an_actual_output_item():
    assert _observed_response_tool_types({"output": []}) == set()
    assert _observed_response_tool_types({"output": [{"type": "custom_tool_call"}]}) == {"custom"}
    assert _observed_response_tool_types({"output": [{"type": "function_call", "namespace": "mcp"}]}) == {"namespace"}


@pytest.mark.asyncio
async def test_sse_frame_parser_ignores_keepalive_and_preserves_frame_bytes():
    async def chunks():
        yield b": keepalive\n\n"
        yield b"data: {\"type\": \"response.created\"}\n\n"

    frames = [frame async for frame in iter_sse_frames(chunks())]
    assert frames[0] == b": keepalive\n\n"
    assert sse_payload(frames[0]) is None
    assert sse_payload(frames[1])["type"] == "response.created"


@pytest.mark.asyncio
async def test_sse_frame_parser_preserves_crlf_framing():
    async def chunks():
        yield b"data: {\"type\": \"response.created\"}\r\n\r\n"

    frames = [frame async for frame in iter_sse_frames(chunks())]
    assert frames == [b"data: {\"type\": \"response.created\"}\r\n\r\n"]
    assert sse_payload(frames[0])["type"] == "response.created"


@pytest.mark.asyncio
async def test_native_fallback_is_used_when_primary_lacks_responses_capability(monkeypatch):
    add_provider({
        "id": "native-primary-off", "name": "Primary", "provider_type": "openai",
        "api_base": "https://primary.invalid/v1", "api_key": "key",
        "models": [{"id": "primary-model"}],
    })
    add_provider({
        "id": "native-fallback-on", "name": "Fallback", "provider_type": "openai",
        "api_base": "https://fallback.invalid/v1", "api_key": "key",
        "models": [{"id": "fallback-model"}],
    })
    set_provider_responses_capability("native-primary-off", status="unsupported")
    set_provider_responses_capability("native-fallback-on", status="supported")
    add_fallback_policy({
        "name": "native capability fallback",
        "match_provider": "native-primary-off", "match_model": "primary-model",
        "chain": [{"model": "fallback-model", "provider_id": "native-fallback-on"}],
    })

    calls = []

    async def fake_post(provider, internal):
        calls.append((provider["id"], internal.target_model))
        return {"object": "response", "id": "resp_fallback", "output": []}

    monkeypatch.setattr("app.router.proxy.post_native_response", fake_post)
    internal = responses_to_internal({
        "model": "primary-model", "input": "use computer",
        "tools": [{"type": "computer"}],
    })
    internal.target_model = "primary-model"
    internal.provider_id = "native-primary-off"

    response, target, provider_id, attempts = await _native_response_with_fallbacks(
        internal, stream=False, required_tool_types={"computer"},
    )

    assert response["id"] == "resp_fallback"
    assert (target.model, provider_id) == ("fallback-model", "native-fallback-on")
    assert calls == [("native-fallback-on", "fallback-model")]
    assert attempts[0]["status"] == "skipped"
    assert attempts[-1]["status"] == "success"


@pytest.mark.asyncio
async def test_native_only_request_reports_no_compatible_fallback():
    add_provider({
        "id": "native-none", "name": "None", "provider_type": "openai",
        "api_base": "https://none.invalid/v1", "api_key": "key",
        "models": [{"id": "none-model"}],
    })
    set_provider_responses_capability("native-none", status="unsupported")
    internal = responses_to_internal({"model": "none-model", "input": "x", "tools": [{"type": "computer"}]})
    internal.provider_id = "native-none"

    with pytest.raises(RuntimeError, match="No native Responses fallback target available") as raised:
        await _native_response_with_fallbacks(internal, stream=False, required_tool_types={"computer"})

    assert raised.value.native_capability_unavailable is True


@pytest.mark.asyncio
async def test_unknown_same_model_native_fallback_is_probed_then_used(monkeypatch):
    add_provider({
        "id": "native-primary-fail", "name": "Primary", "provider_type": "openai",
        "api_base": "https://primary.invalid/v1", "api_key": "key",
        "models": [{"id": "shared-model"}],
    })
    add_provider({
        "id": "native-fallback-unknown", "name": "Fallback", "provider_type": "openai",
        "api_base": "https://fallback.invalid/v1", "api_key": "key",
        "models": [{"id": "shared-model"}],
    })
    set_provider_responses_capability("native-primary-fail", status="supported")
    add_fallback_policy({
        "name": "probe unknown fallback", "match_provider": "native-primary-fail",
        "match_model": "shared-model", "chain": [{"model": "shared-model", "provider_id": "native-fallback-unknown"}],
    })

    async def fake_post(provider, internal):
        if provider["id"] == "native-primary-fail":
            raise httpx.HTTPStatusError("bad gateway", request=httpx.Request("POST", "https://primary.invalid"), response=httpx.Response(502, request=httpx.Request("POST", "https://primary.invalid")))
        return {"object": "response", "id": "resp_probed", "output": []}

    async def fake_probe(provider_id):
        assert provider_id == "native-fallback-unknown"
        set_provider_responses_capability(provider_id, status="supported")
        return {"status": "supported", "streaming": False}

    monkeypatch.setattr("app.router.proxy.post_native_response", fake_post)
    monkeypatch.setattr("app.router.proxy.probe_responses_capability", fake_probe)
    internal = responses_to_internal({"model": "shared-model", "input": "hello"})
    internal.provider_id = "native-primary-fail"

    response, target, provider_id, attempts = await _native_response_with_fallbacks(internal, stream=False, required_tool_types=set())

    assert response["id"] == "resp_probed"
    assert (target.model, provider_id) == ("shared-model", "native-fallback-unknown")
    assert [item["status"] for item in attempts] == ["failed", "success"]


@pytest.mark.asyncio
async def test_native_fallback_keeps_administrator_chain_order(monkeypatch):
    add_provider({
        "id": "order-primary", "name": "Primary", "provider_type": "openai",
        "api_base": "https://primary.invalid/v1", "api_key": "key",
        "models": [{"id": "primary-model"}],
    })
    add_provider({
        "id": "order-first", "name": "First", "provider_type": "openai",
        "api_base": "https://first.invalid/v1", "api_key": "key",
        "models": [{"id": "other-model"}],
    })
    add_provider({
        "id": "order-second", "name": "Second", "provider_type": "openai",
        "api_base": "https://second.invalid/v1", "api_key": "key",
        "models": [{"id": "primary-model"}],
    })
    for provider_id in ("order-primary", "order-first", "order-second"):
        set_provider_responses_capability(provider_id, status="supported")
    add_fallback_policy({
        "name": "preserve native order", "match_provider": "order-primary",
        "match_model": "primary-model",
        "chain": [
            {"model": "other-model", "provider_id": "order-first"},
            {"model": "primary-model", "provider_id": "order-second"},
        ],
    })

    calls = []

    async def fake_post(provider, internal):
        calls.append(provider["id"])
        if provider["id"] in {"order-primary", "order-first"}:
            request = httpx.Request("POST", "https://example.invalid/responses")
            raise httpx.HTTPStatusError("bad gateway", request=request, response=httpx.Response(502, request=request))
        return {"object": "response", "id": "resp_order", "output": []}

    monkeypatch.setattr("app.router.proxy.post_native_response", fake_post)
    internal = responses_to_internal({"model": "primary-model", "input": "hello"})
    internal.provider_id = "order-primary"
    response, _target, provider_id, attempts = await _native_response_with_fallbacks(internal, stream=False, required_tool_types=set())

    assert response["id"] == "resp_order"
    assert provider_id == "order-second"
    assert calls == ["order-primary", "order-first", "order-second"]
    assert [item["provider_id"] for item in attempts] == calls


@pytest.mark.asyncio
async def test_stateful_request_blocks_fallback_when_primary_capability_unknown():
    add_provider({
        "id": "stateful-unknown-primary", "name": "Primary", "provider_type": "openai",
        "api_base": "https://primary.invalid/v1", "api_key": "key",
        "models": [{"id": "stateful-model"}],
    })
    add_provider({
        "id": "stateful-fallback", "name": "Fallback", "provider_type": "openai",
        "api_base": "https://fallback.invalid/v1", "api_key": "key",
        "models": [{"id": "fallback-model"}],
    })
    set_provider_responses_capability("stateful-fallback", status="supported")
    add_fallback_policy({
        "name": "stateful block", "match_provider": "stateful-unknown-primary",
        "match_model": "stateful-model",
        "chain": [{"model": "fallback-model", "provider_id": "stateful-fallback"}],
    })
    internal = responses_to_internal({
        "model": "stateful-model", "previous_response_id": "resp_old",
        "input": [{"type": "function_call_output", "call_id": "call_old", "output": "ok"}],
    })
    internal.provider_id = "stateful-unknown-primary"
    with pytest.raises(RuntimeError, match="fallback blocked") as raised:
        await _native_response_with_fallbacks(internal, stream=False, required_tool_types=set(), stateful_markers=["previous_response_id", "function_call_output"])
    assert raised.value.request_details["fallback_reason"] == "stateful_codex_tools"
    assert raised.value.request_details["stateful_fallback_blocked"] is True
