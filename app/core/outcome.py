"""Request outcome status for stats / request logs.

Statuses:
- ok: primary target succeeded without fallback
- degraded: request ultimately succeeded after fallback (or equivalent)
- partial: stream failed after client-visible output started
- fail: hard failure with no usable completion
- rejected: auth / allow-list rejection (401/403) before upstream
- cancelled: client disconnected before a complete successful response
"""

from __future__ import annotations

from typing import Any, NamedTuple


def is_fallback_used(details: dict | None) -> bool:
    """True when the request recovered via a non-primary upstream attempt."""
    d = details or {}
    if str(d.get("fallback_status") or "") == "used":
        return True
    try:
        if int(d.get("attempt_index") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass

    attempts = d.get("fallback_attempts")
    if not isinstance(attempts, list) or not attempts:
        return False

    has_failed = False
    has_fallback_success = False
    has_any_success_after_fail = False
    for item in attempts:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "")
        stage = str(item.get("stage") or "")
        if status == "failed":
            has_failed = True
        if stage == "fallback" and status == "success":
            has_fallback_success = True
        if has_failed and status == "success":
            has_any_success_after_fail = True
    return has_fallback_success or has_any_success_after_fail


def resolve_request_status(
    *,
    success: bool,
    details: dict | None = None,
    partial_output: bool = False,
) -> str:
    """Derive the public request status from success flag and detail payload."""
    d = details or {}
    existing = str(d.get("status") or "").strip().lower()

    if not success:
        if existing in ("rejected", "cancelled"):
            return existing
        # Client disconnect with partial output still counts as cancelled.
        if existing == "partial" and bool(d.get("client_disconnected")):
            return "cancelled"
        if partial_output or existing == "partial" or bool(d.get("partial_output")):
            if bool(d.get("client_disconnected")):
                return "cancelled"
            return "partial"
        if existing == "fail":
            return "fail"
        return "fail"

    if existing == "degraded" or is_fallback_used(d):
        return "degraded"
    if existing == "ok":
        return "ok"
    return "ok"


def apply_outcome_to_details(
    details: dict | None,
    *,
    success: bool,
    partial_output: bool = False,
) -> dict[str, Any]:
    """Return a copy of details with a normalized status field."""
    d = dict(details or {})
    status = resolve_request_status(success=success, details=d, partial_output=partial_output)
    d["status"] = status
    if status == "degraded" and not d.get("fallback_status"):
        d["fallback_status"] = "used"
    if partial_output:
        d["partial_output"] = True
    return d


class OutcomeCounters(NamedTuple):
    hard_success: bool
    degraded: bool
    rejected: bool
    cancelled: bool


def stats_counters_for_status(status: str) -> OutcomeCounters:
    """Map status -> counter flags for global stats."""
    normalized = (status or "").strip().lower()
    if normalized == "ok":
        return OutcomeCounters(True, False, False, False)
    if normalized == "degraded":
        return OutcomeCounters(True, True, False, False)
    if normalized == "rejected":
        return OutcomeCounters(False, False, True, False)
    if normalized == "cancelled":
        return OutcomeCounters(False, False, False, True)
    return OutcomeCounters(False, False, False, False)


def is_client_disconnect_error(exc: BaseException) -> bool:
    """Detect client-gone errors from Starlette/anyio/asyncio transport layers."""
    if isinstance(exc, GeneratorExit):
        return True
    try:
        import asyncio

        if isinstance(exc, asyncio.CancelledError):
            return True
    except Exception:
        pass

    name = type(exc).__name__
    if name in {"ClientDisconnect", "CancelledError", "BrokenResourceError", "ClosedResourceError"}:
        return True

    module = getattr(type(exc), "__module__", "") or ""
    if "starlette" in module and "disconnect" in name.lower():
        return True

    text = str(exc).lower()
    markers = (
        "client disconnected",
        "connection reset",
        "broken pipe",
        "connection closed",
        "client has disconnected",
        "remote protocol error",
    )
    return any(marker in text for marker in markers)


def routing_details_from_policy(policy) -> dict[str, Any]:
    """Extract stable routing fields from RequestPolicyResult for request logs."""
    routing = getattr(policy, "routing", None)
    if routing is None:
        return {}
    matched = bool(getattr(routing, "matched", False))
    target_model = str(getattr(routing, "target_model", "") or "")
    target_provider = str(getattr(routing, "target_provider", "") or "")
    details: dict[str, Any] = {
        "routing_matched": matched,
        "routing_rule_id": getattr(routing, "rule_id", None) or "",
        "routing_rule_name": getattr(routing, "rule_name", "") or "",
        "routing_reason": getattr(routing, "reason", "") or "",
        "routed_model": target_model if matched else "",
        "routed_provider": target_provider if matched else "",
    }
    return details
