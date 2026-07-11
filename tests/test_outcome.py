"""Tests for request outcome status helpers."""

from app.core.outcome import (
    apply_outcome_to_details,
    is_fallback_used,
    resolve_request_status,
    stats_counters_for_status,
)


def test_primary_success_is_ok():
    details = apply_outcome_to_details(
        {"fallback_status": "unused", "fallback_attempts": [{"index": 0, "stage": "primary", "status": "success"}]},
        success=True,
    )
    assert details["status"] == "ok"
    assert stats_counters_for_status("ok") == (True, False, False, False)


def test_fallback_used_is_degraded():
    details = {
        "fallback_status": "used",
        "attempt_index": 1,
        "fallback_attempts": [
            {"index": 0, "stage": "primary", "status": "failed", "provider": "PixelAPI"},
            {"index": 1, "stage": "fallback", "status": "success", "provider": "qianye"},
        ],
    }
    assert is_fallback_used(details) is True
    assert resolve_request_status(success=True, details=details) == "degraded"
    finalized = apply_outcome_to_details(details, success=True)
    assert finalized["status"] == "degraded"
    assert stats_counters_for_status("degraded") == (True, True, False, False)


def test_fallback_inferred_from_attempts_without_status_flag():
    details = {
        "fallback_attempts": [
            {"index": 0, "stage": "primary", "status": "failed"},
            {"index": 1, "stage": "fallback", "status": "success"},
        ]
    }
    assert resolve_request_status(success=True, details=details) == "degraded"


def test_partial_and_fail_paths():
    assert resolve_request_status(success=False, details={}, partial_output=True) == "partial"
    assert resolve_request_status(success=False, details={"status": "fail"}) == "fail"
    assert stats_counters_for_status("fail") == (False, False, False, False)
    assert stats_counters_for_status("partial") == (False, False, False, False)


def test_rejected_and_cancelled_outcomes():
    assert resolve_request_status(success=False, details={"status": "rejected"}) == "rejected"
    assert resolve_request_status(
        success=False,
        details={"status": "cancelled", "client_disconnected": True},
        partial_output=True,
    ) == "cancelled"
    assert stats_counters_for_status("rejected") == (False, False, True, False)
    assert stats_counters_for_status("cancelled") == (False, False, False, True)
    assert stats_counters_for_status("ok") == (True, False, False, False)
    assert stats_counters_for_status("degraded") == (True, True, False, False)
