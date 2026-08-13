import pytest
import httpx
from datetime import datetime, timedelta, timezone

from app.adapters.responses import iter_sse_frames, native_responses_body, responses_url, sse_payload
from app.core.types import InternalRequest
from app.protocols.ingress import responses_to_internal
from app.router.proxy import (
    _native_responses_stream_with_accounting,
    _native_response_with_fallbacks,
    _observed_response_tool_types,
    _responses_requires_native,
    _responses_required_tool_types,
    _native_response_target_supported,
    _native_downgrade_details,
    _wait_for_native_response_output,
    _native_capability_for_request,
)
from app.core.policy import RouteTarget
from app.config import load_config
from app.database import (
    add_fallback_policy,
    add_provider,
    init_db,
    set_model_responses_capability, get_model_responses_capability,
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


def test_capability_is_scoped_to_provider_and_model_and_expired_negative_is_retryable():
    add_provider({
        "id": "deepseek", "name": "DeepSeek", "provider_type": "openai",
        "api_base": "https://deepseek.invalid/v1", "api_key": "key",
        "models": [{"id": "deepseek-v4-flash"}, {"id": "deepseek-v4-pro"}],
    })
    set_model_responses_capability("deepseek", "deepseek-v4-flash", status="supported")
    set_model_responses_capability("deepseek", "deepseek-v4-pro", status="unsupported", expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat())

    flash = get_model_responses_capability("deepseek", "deepseek-v4-flash")
    pro = get_model_responses_capability("deepseek", "deepseek-v4-pro")
    assert flash["responses_status"] == "supported"
    assert pro["responses_status"] == "unsupported"
    assert _native_response_target_supported(RouteTarget(model="deepseek-v4-flash", provider_id="deepseek"), stream=True, required_tool_types=set(), is_primary=True)[0]
    # The next real request, rather than a timer or refresh operation, probes
    # an expired negative entry.
    assert _native_response_target_supported(RouteTarget(model="deepseek-v4-pro", provider_id="deepseek"), stream=True, required_tool_types=set(), is_primary=True)[0]


def test_anthropic_provider_never_uses_openai_native_responses_probe_path():
    add_provider({
        "id": "minimax", "name": "MiniMax", "provider_type": "anthropic",
        "api_base": "https://minimax.invalid/anthropic", "api_key": "key",
        "models": [{"id": "MiniMax-M3"}],
    })
    assert _native_response_target_supported(
        RouteTarget(model="MiniMax-M3", provider_id="minimax"),
        stream=True, required_tool_types=set(), is_primary=True,
    ) == (None, "")


@pytest.mark.asyncio
async def test_unknown_capability_probe_caches_unsupported_on_llamacpp_400(monkeypatch):
    add_provider({
        "id": "llamacpp", "name": "llama.cpp", "provider_type": "openai",
        "api_base": "http://llamacpp.invalid/v1", "models": [{"id": "qwen"}],
    })
    request = httpx.Request("POST", "http://llamacpp.invalid/v1/responses")
    async def fake_post(provider, internal):
        raise httpx.HTTPStatusError(
            "bad request", request=request,
            response=httpx.Response(400, text="unsupported", request=request),
        )
    monkeypatch.setattr("app.router.proxy.post_native_response", fake_post)
    provider = {"id": "llamacpp", "provider_type": "openai"}
    assert await _native_capability_for_request(provider, "qwen") is False
    assert get_model_responses_capability("llamacpp", "qwen")["responses_status"] == "unsupported"


@pytest.mark.asyncio
async def test_real_native_request_promotes_unknown_model_without_background_probe(monkeypatch):
    add_provider({
        "id": "pixel", "name": "Pixel", "provider_type": "openai",
        "api_base": "https://pixel.invalid/v1", "api_key": "key",
        "models": [{"id": "gpt-5.6-luna"}],
    })
    async def fake_post(provider, internal):
        return {"object": "response", "id": "resp_pixel", "output": [{"type": "message"}]}
    monkeypatch.setattr("app.router.proxy.post_native_response", fake_post)
    internal = responses_to_internal({"model": "gpt-5.6-luna", "input": "hello"})
    internal.provider_id = "pixel"
    await _native_response_with_fallbacks(internal, stream=False, required_tool_types=set())
    capability = get_model_responses_capability("pixel", "gpt-5.6-luna")
    assert capability["responses_status"] == "supported"
    assert capability["responses_expires_at"]



def test_actual_upstream_endpoint_labels_native_openai_and_anthropic_paths():
    from app.router.proxy import _upstream_endpoint_for_provider

    assert _upstream_endpoint_for_provider({"provider_type": "openai"}) == "chat_completions"
    assert _upstream_endpoint_for_provider({"provider_type": "anthropic"}) == "messages"
    assert _upstream_endpoint_for_provider({"provider_type": "openai"}, native_responses=True) == "responses"


def test_native_downgrade_details_preserve_the_failed_responses_attempt():
    request = httpx.Request("POST", "https://pixel.invalid/v1/responses")
    error = httpx.HTTPStatusError(
        "bad gateway", request=request, response=httpx.Response(502, request=request),
    )
    details = _native_downgrade_details(error, [{"index": 0, "status": "failed"}])

    assert details == {
        "responses_mode": "compatibility_downgrade",
        "native_attempted": True,
        "native_failure_endpoint": "responses",
        "native_failure_status": 502,
        "native_failure_reason": "http_5xx",
        "native_failure_message": "bad gateway",
        "native_attempts": [{"index": 0, "status": "failed"}],
    }


@pytest.mark.asyncio
async def test_explicit_protocol_rejection_is_short_lived_model_negative_cache(monkeypatch):
    add_provider({
        "id": "deepseek", "name": "DeepSeek", "provider_type": "openai",
        "api_base": "https://deepseek.invalid/v1", "api_key": "key",
        "models": [{"id": "deepseek-v4-pro"}],
    })
    request = httpx.Request("POST", "https://deepseek.invalid/v1/responses")
    async def fake_post(provider, internal):
        raise httpx.HTTPStatusError(
            "Responses endpoint not supported", request=request,
            response=httpx.Response(404, text='{"error":{"message":"Responses endpoint not supported"}}', request=request),
        )
    monkeypatch.setattr("app.router.proxy.post_native_response", fake_post)
    internal = responses_to_internal({"model": "deepseek-v4-pro", "input": "hello"})
    internal.provider_id = "deepseek"
    with pytest.raises(httpx.HTTPStatusError):
        await _native_response_with_fallbacks(internal, stream=False, required_tool_types=set())
    capability = get_model_responses_capability("deepseek", "deepseek-v4-pro")
    assert capability["responses_status"] == "unsupported"
    assert capability["responses_expires_at"]


@pytest.mark.asyncio
async def test_transient_native_failure_invalidates_supported_capability(monkeypatch):
    add_provider({
        "id": "pixel-transient", "name": "Pixel", "provider_type": "openai",
        "api_base": "https://pixel.invalid/v1", "models": [{"id": "model"}],
    })
    set_model_responses_capability(
        "pixel-transient", "model", status="supported",
        expires_at="2999-01-01T00:00:00+00:00",
    )
    request = httpx.Request("POST", "https://pixel.invalid/v1/responses")
    async def fake_post(provider, internal):
        raise httpx.HTTPStatusError(
            "upstream unavailable", request=request,
            response=httpx.Response(503, request=request),
        )
    monkeypatch.setattr("app.router.proxy.post_native_response", fake_post)
    internal = responses_to_internal({"model": "model", "input": "hello"})
    internal.provider_id = "pixel-transient"
    with pytest.raises(httpx.HTTPStatusError):
        await _native_response_with_fallbacks(internal, stream=False, required_tool_types=set())
    capability = get_model_responses_capability("pixel-transient", "model")
    assert capability["responses_status"] == "unknown"


@pytest.mark.asyncio
async def test_model_not_found_404_does_not_create_negative_capability_cache(monkeypatch):
    add_provider({
        "id": "provider", "name": "Provider", "provider_type": "openai",
        "api_base": "https://provider.invalid/v1", "api_key": "key",
        "models": [{"id": "model"}],
    })
    request = httpx.Request("POST", "https://provider.invalid/v1/responses")

    async def fake_post(provider, internal):
        raise httpx.HTTPStatusError(
            "model not found", request=request,
            response=httpx.Response(404, text='{"error":{"message":"model does not exist"}}', request=request),
        )

    monkeypatch.setattr("app.router.proxy.post_native_response", fake_post)
    internal = responses_to_internal({"model": "model", "input": "hello"})
    internal.provider_id = "provider"
    with pytest.raises(httpx.HTTPStatusError):
        await _native_response_with_fallbacks(internal, stream=False, required_tool_types=set())
    assert get_model_responses_capability("provider", "model")["responses_status"] == "unknown"


@pytest.mark.asyncio
async def test_arbitrary_validation_error_does_not_create_negative_capability_cache(monkeypatch):
    add_provider({
        "id": "provider", "name": "Provider", "provider_type": "openai",
        "api_base": "https://provider.invalid/v1", "api_key": "key",
        "models": [{"id": "model"}],
    })
    request = httpx.Request("POST", "https://provider.invalid/v1/responses")

    async def fake_post(provider, internal):
        raise httpx.HTTPStatusError(
            "invalid input item", request=request,
            response=httpx.Response(422, text='{"detail":"invalid input"}', request=request),
        )

    monkeypatch.setattr("app.router.proxy.post_native_response", fake_post)
    internal = responses_to_internal({"model": "model", "input": "hello"})
    internal.provider_id = "provider"
    with pytest.raises(httpx.HTTPStatusError):
        await _native_response_with_fallbacks(internal, stream=False, required_tool_types=set())
    assert get_model_responses_capability("provider", "model")["responses_status"] == "unknown"


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
async def test_stream_created_then_failed_does_not_mark_model_responses_supported(monkeypatch):
    add_provider({
        "id": "pixel", "name": "Pixel", "provider_type": "openai",
        "api_base": "https://pixel.invalid/v1", "api_key": "key",
        "models": [{"id": "gpt-5.6-luna"}],
    })
    internal = responses_to_internal({"model": "gpt-5.6-luna", "input": "hello", "stream": True})

    async def events():
        yield b'data: {"type":"response.created","response":{"id":"resp_1"}}\n\n'
        yield b'data: {"type":"response.failed","response":{"id":"resp_1"}}\n\n'

    async def no_op(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.router.proxy._log_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.router.proxy._record_request_log", lambda **_kwargs: None)
    monkeypatch.setattr("app.router.proxy._record_success_metrics", lambda *_args, **_kwargs: None)
    frames = [frame async for frame in _native_responses_stream_with_accounting(
        events(), username="u", api_key_value="k", model="pixel/gpt-5.6-luna",
        provider_id="pixel", requested_model="pixel/gpt-5.6-luna", policy=None,
        request_body=internal.raw_body,
    )]
    assert len(frames) == 2
    assert get_model_responses_capability("pixel", "gpt-5.6-luna")["responses_status"] == "unknown"


@pytest.mark.asyncio
async def test_completed_native_stream_marks_model_responses_supported(monkeypatch):
    add_provider({
        "id": "pixel", "name": "Pixel", "provider_type": "openai",
        "api_base": "https://pixel.invalid/v1", "api_key": "key",
        "models": [{"id": "gpt-5.6-luna"}],
    })

    async def events():
        yield b'data: {"type":"response.output_item.done","item":{"type":"message"}}\n\n'
        yield b'data: {"type":"response.completed","response":{"id":"resp_1","output":[{"type":"message"}],"usage":{"total_tokens":1}}}\n\n'

    monkeypatch.setattr("app.router.proxy._log_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.router.proxy._record_request_log", lambda **_kwargs: None)
    monkeypatch.setattr("app.router.proxy._record_success_metrics", lambda *_args, **_kwargs: None)
    async for _frame in _native_responses_stream_with_accounting(
        events(), username="u", api_key_value="k", model="pixel/gpt-5.6-luna",
        provider_id="pixel", requested_model="pixel/gpt-5.6-luna", policy=None,
        request_body={},
    ):
        pass
    capability = get_model_responses_capability("pixel", "gpt-5.6-luna")
    assert capability["responses_status"] == "supported"
    assert capability["responses_streaming"] is True


@pytest.mark.asyncio
async def test_empty_completed_native_stream_is_rejected(monkeypatch):
    add_provider({
        "id": "pixel-empty", "name": "Pixel", "provider_type": "openai",
        "api_base": "https://pixel.invalid/v1", "api_key": "key",
        "models": [{"id": "empty-model"}],
    })

    async def events():
        yield b'data: {"type":"response.created","response":{"id":"resp_empty"}}\n\n'
        yield b'data: {"type":"response.completed","response":{"id":"resp_empty","output":[],"usage":{"total_tokens":1}}}\n\n'

    with pytest.raises(RuntimeError, match="client-visible output"):
        await _wait_for_native_response_output(events())
    assert get_model_responses_capability("pixel-empty", "empty-model")["responses_status"] == "unknown"


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
    set_model_responses_capability("native-primary-off", "primary-model", status="unsupported", expires_at="2999-01-01T00:00:00+00:00")
    set_model_responses_capability("native-fallback-on", "fallback-model", status="supported")
    add_fallback_policy({
        "name": "native capability fallback",
        "match_provider": "native-primary-off", "match_model": "primary-model",
        "chain": [{"model": "fallback-model", "provider_id": "native-fallback-on"}],
    })

    calls = []

    async def fake_post(provider, internal):
        calls.append((provider["id"], internal.target_model))
        return {"object": "response", "id": "resp_fallback", "output": [{"type": "computer_call"}]}

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
    set_model_responses_capability("native-none", "none-model", status="unsupported", expires_at="2999-01-01T00:00:00+00:00")
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
    set_model_responses_capability("native-primary-fail", "shared-model", status="supported")
    add_fallback_policy({
        "name": "probe unknown fallback", "match_provider": "native-primary-fail",
        "match_model": "shared-model", "chain": [{"model": "shared-model", "provider_id": "native-fallback-unknown"}],
    })

    async def fake_post(provider, internal):
        if provider["id"] == "native-primary-fail":
            raise httpx.HTTPStatusError("bad gateway", request=httpx.Request("POST", "https://primary.invalid"), response=httpx.Response(502, request=httpx.Request("POST", "https://primary.invalid")))
        return {"object": "response", "id": "resp_probed", "output": [{"type": "message"}]}

    monkeypatch.setattr("app.router.proxy.post_native_response", fake_post)
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
    for provider_id, model_id in (("order-primary", "primary-model"), ("order-first", "other-model"), ("order-second", "primary-model")):
        set_model_responses_capability(provider_id, model_id, status="supported")
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
        return {"object": "response", "id": "resp_order", "output": [{"type": "message"}]}

    monkeypatch.setattr("app.router.proxy.post_native_response", fake_post)
    internal = responses_to_internal({"model": "primary-model", "input": "hello"})
    internal.provider_id = "order-primary"
    response, _target, provider_id, attempts = await _native_response_with_fallbacks(internal, stream=False, required_tool_types=set())

    assert response["id"] == "resp_order"
    assert provider_id == "order-second"
    assert calls == ["order-primary", "order-first", "order-second"]
    assert [item["provider_id"] for item in attempts] == calls


@pytest.mark.asyncio
async def test_stateful_request_attempts_unknown_primary_before_any_cross_provider_fallback():
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
    set_model_responses_capability("stateful-fallback", "fallback-model", status="supported")
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
    async def fake_post(provider, internal):
        return {"object": "response", "id": "resp_stateful_compat", "output": [{"type": "message"}]}
    # Unknown is intentionally request-driven: the real user request first
    # verifies the selected target rather than preemptively moving a stateful
    # turn to another provider.
    from unittest.mock import patch
    with patch("app.router.proxy.post_native_response", side_effect=fake_post):
        response, _target, provider_id, attempts = await _native_response_with_fallbacks(
            internal, stream=False, required_tool_types=set(), stateful_markers=["previous_response_id", "function_call_output"]
        )
    assert response["id"] == "resp_stateful_compat"
    assert provider_id == "stateful-unknown-primary"
    assert attempts == [{"index": 0, "stage": "primary", "target": "stateful-model", "provider_id": "stateful-unknown-primary", "status": "success"}]


@pytest.mark.asyncio
async def test_stateful_request_blocks_only_after_primary_native_request_fails(monkeypatch):
    add_provider({"id": "stateful-primary-fails", "name": "Primary", "provider_type": "openai", "api_base": "https://primary.invalid/v1", "api_key": "key", "models": [{"id": "stateful-model"}]})
    add_provider({"id": "stateful-other", "name": "Other", "provider_type": "openai", "api_base": "https://other.invalid/v1", "api_key": "key", "models": [{"id": "other-model"}]})
    set_model_responses_capability("stateful-primary-fails", "stateful-model", status="supported")
    set_model_responses_capability("stateful-other", "other-model", status="supported")
    add_fallback_policy({"name": "stateful failure block", "match_provider": "stateful-primary-fails", "match_model": "stateful-model", "chain": [{"model": "other-model", "provider_id": "stateful-other"}]})
    async def fake_post(provider, internal):
        raise httpx.HTTPStatusError("bad gateway", request=httpx.Request("POST", "https://primary.invalid"), response=httpx.Response(502))
    monkeypatch.setattr("app.router.proxy.post_native_response", fake_post)
    internal = responses_to_internal({"model": "stateful-model", "previous_response_id": "resp_old", "input": [{"type": "function_call_output", "call_id": "call_old", "output": "ok"}]})
    internal.provider_id = "stateful-primary-fails"
    with pytest.raises(httpx.HTTPStatusError) as raised:
        await _native_response_with_fallbacks(internal, stream=False, required_tool_types=set(), stateful_markers=["previous_response_id", "function_call_output"])
    assert raised.value.request_details["fallback_reason"] == "stateful_codex_tools"
    assert raised.value.request_details["stateful_fallback_blocked"] is True


def test_stateful_markers_include_previous_response_id_and_not_plain_tools():
    from app.router.proxy import _responses_stateful_tool_markers
    assert _responses_stateful_tool_markers({"input": "hello", "tools": [{"type": "function"}]}) == []
    assert _responses_stateful_tool_markers({"previous_response_id": "resp_1", "input": "hello"}) == ["previous_response_id"]


def test_responses_native_required_fields_do_not_include_common_codex_options():
    from app.router.proxy import _responses_requires_native
    assert _responses_requires_native({"model": "x", "input": "hello", "max_output_tokens": 32, "tools": [{"type": "function", "name": "f"}]}) == []


def test_responses_native_required_fields_do_not_include_codex_custom_tools():
    from app.router.proxy import _responses_requires_native
    assert _responses_requires_native({"model": "x", "input": "hello", "tools": [{"type": "custom", "name": "exec_command"}]}) == []


def test_responses_compatibility_accepts_client_metadata_and_hosted_tools():
    from app.router.proxy import _responses_requires_native
    assert _responses_requires_native({
        "model": "x", "input": "hello", "client_metadata": {"client": "codex"},
        "tools": [{"type": "web_search"}],
    }) == []
