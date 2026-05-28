"""Tests for app/services/routing_targets.py"""
import pytest
from fastapi import HTTPException
import httpx

from app.core.policy import RouteTarget
from app.services.routing_targets import (
    adapter_provider_id,
    candidate_targets,
    classify_upstream_error,
    provider_for_log,
)


# --- adapter_provider_id / provider_for_log ---

def test_adapter_provider_id_prefers_info():
    assert adapter_provider_id({"id": "p1"}, "p2") == "p1"


def test_adapter_provider_id_falls_back_to_arg():
    assert adapter_provider_id(None, "p2") == "p2"
    assert adapter_provider_id({}, "p2") == "p2"


def test_adapter_provider_id_neither():
    assert adapter_provider_id(None, "") == ""


def test_provider_for_log_matches_adapter():
    assert provider_for_log({"id": "x"}, "y") == "x"
    assert provider_for_log(None, "y") == "y"


# --- candidate_targets ---

def test_candidate_targets_no_fallback():
    primary = RouteTarget(model="m1", provider_id="p1")
    targets = candidate_targets(primary, None)
    assert len(targets) == 1
    assert targets[0].model == "m1"


def test_candidate_targets_deduplicates():
    primary = RouteTarget(model="m1", provider_id="p1")
    dup = RouteTarget(model="m1", provider_id="p1")
    targets = candidate_targets(primary, [dup])
    assert len(targets) == 1


def test_candidate_targets_removes_empty_model():
    primary = RouteTarget(model="m1", provider_id="p1")
    empty = RouteTarget(model="", provider_id="p2")
    targets = candidate_targets(primary, [empty])
    assert len(targets) == 1


def test_candidate_targets_preserves_chain_order():
    primary = RouteTarget(model="m1", provider_id="p1")
    fb1 = RouteTarget(model="m2", provider_id="p2")
    fb2 = RouteTarget(model="m3", provider_id="p3")
    targets = candidate_targets(primary, [fb1, fb2])
    assert [(t.model, t.provider_id) for t in targets] == [
        ("m1", "p1"), ("m2", "p2"), ("m3", "p3")
    ]


def test_candidate_targets_skips_duplicates_in_chain():
    primary = RouteTarget(model="m1", provider_id="p1")
    fb1 = RouteTarget(model="m1", provider_id="p1")
    fb2 = RouteTarget(model="m2", provider_id="p2")
    fb3 = RouteTarget(model="m2", provider_id="p2")
    targets = candidate_targets(primary, [fb1, fb2, fb3])
    assert [(t.model, t.provider_id) for t in targets] == [
        ("m1", "p1"), ("m2", "p2")
    ]


# --- classify_upstream_error ---

def test_classify_http_429():
    assert classify_upstream_error(HTTPException(status_code=429)) == "http_429"


def test_classify_http_5xx():
    for code in (500, 502, 503, 504):
        assert classify_upstream_error(HTTPException(status_code=code)) == "http_5xx"


def test_classify_http_4xx():
    assert classify_upstream_error(HTTPException(status_code=400)) == "http_4xx"
    assert classify_upstream_error(HTTPException(status_code=404)) == "http_4xx"
    assert classify_upstream_error(HTTPException(status_code=408)) == "http_4xx"


def test_classify_http_unknown_status():
    assert classify_upstream_error(HTTPException(status_code=100)) == "unknown"
    assert classify_upstream_error(HTTPException(status_code=301)) == "unknown"


def test_classify_timeout_error():
    assert classify_upstream_error(TimeoutError("timed out")) == "timeout"


def test_classify_httpx_timeout():
    assert classify_upstream_error(httpx.TimeoutException("read timeout")) == "timeout"


def test_classify_httpx_connect_error():
    assert classify_upstream_error(httpx.ConnectError("refused")) == "connection_error"


def test_classify_httpx_network_error():
    assert classify_upstream_error(httpx.NetworkError("unreachable")) == "connection_error"


def test_classify_builtin_connection_error():
    assert classify_upstream_error(ConnectionError("reset")) == "connection_error"


def test_classify_text_timeout():
    assert classify_upstream_error(RuntimeError("Request timed out")) == "timeout"


def test_classify_text_timed_out():
    assert classify_upstream_error(RuntimeError("connection timed out")) == "timeout"


def test_classify_text_429():
    assert classify_upstream_error(RuntimeError("Got 429 from upstream")) == "http_429"


def test_classify_text_rate_limit():
    assert classify_upstream_error(RuntimeError("rate limit exceeded")) == "http_429"


def test_classify_text_5xx():
    assert classify_upstream_error(RuntimeError("server error 502")) == "http_5xx"


def test_classify_text_http_504():
    # Note: "timeout" appears before 5xx check in classify, so this is classified as timeout
    assert classify_upstream_error(RuntimeError("http 504 gateway timeout")) == "timeout"


def test_classify_text_pure_503():
    assert classify_upstream_error(RuntimeError("upstream returned 503")) == "http_5xx"


def test_classify_unknown_string_falls_back_to_connection_error():
    assert classify_upstream_error(RuntimeError("something weird")) == "connection_error"


def test_classify_empty_message():
    assert classify_upstream_error(RuntimeError("")) == "connection_error"
