import json
import copy
import threading
import time
import uuid
from collections import deque
from typing import Optional

import anyio
from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from app.database import (
    get_providers, find_user_by_api_key,
    increment_global_stats, increment_user_usage, get_db,
    parse_model_id, add_request_record, add_request_log, trim_request_logs, get_enabled_preprocessor,
)
from app.core.text import friendly_error_msg, mask_key
from app.core.output import InternalOutputEvent

from app.core.state import (
    TOOL_ONLY_LIMIT,
    conversation_cache_key as _conversation_cache_key,
    ir_reasoning_message_count as _ir_reasoning_message_count,
    ir_tool_message_count as _ir_tool_message_count,
    reasoning_context as _reasoning_context,
    remember_reasoning_content as _remember_reasoning_content,
    remember_response_chain_key as _remember_response_chain_key,
    tool_only_turns as _tool_only_turns,
)
from app.core.streaming import stream_internal_output as _stream_internal_output
from app.protocols.ingress import (
    anthropic_messages_to_internal,
    chat_completions_to_internal,
    completions_to_internal,
    responses_to_internal,
)
from app.core.policy import RouteTarget, apply_fallback_policy, prepare_request_policy
from app.adapters.anthropic import (
    anthropic_body_from_internal,
    anthropic_messages_completion_for_internal,
)
from app.adapters.openai import chat_kwargs_from_internal, chat_messages_from_internal
from app.adapters.output import response_to_internal_output
from app.adapters.anthropic_streaming import iter_anthropic_output_events
from app.adapters.openai_streaming import iter_openai_chat_output_events
from app.protocols.egress import (
    render_anthropic_message,
    render_chat_completion,
    render_completion,
    render_response,
)
from app.services.lite_llm import create_chat_completion
from app.services.preprocessing import has_image_content, preprocess_messages
from app.services.routing_targets import candidate_targets, classify_upstream_error, provider_for_log, resolve_provider
from app.services.logger import get_logger
from app.config import get_default

_access_log = get_logger("access")
_error_log = get_logger("error")
_tool_log = get_logger("tool_calls")
_req_log = get_logger("request")
_app_log = get_logger("app")

router = APIRouter()

# Rolling log of recent requests for the admin stats dashboard
_request_log = deque(maxlen=get_default("request_log_max", 200))
_request_log_lock = threading.Lock()


def _attach_request_details(exc: Exception, **details) -> Exception:
    existing = getattr(exc, "request_details", None)
    merged = dict(existing) if isinstance(existing, dict) else {}
    for key, value in details.items():
        if value is not None:
            merged[key] = value
            try:
                setattr(exc, key, value)
            except Exception:
                pass
    try:
        setattr(exc, "request_details", merged)
    except Exception:
        pass
    return exc


def _request_details_from_exception(exc: Exception, **defaults) -> dict:
    existing = getattr(exc, "request_details", None)
    details = dict(existing) if isinstance(existing, dict) else {}
    for key, value in defaults.items():
        if key not in details and value is not None:
            details[key] = value
    details.setdefault("status", "fail")
    details.setdefault("error_message", friendly_error_msg(exc))
    return details


def _fallback_attempt_record(*, index: int, stage: str, target: RouteTarget, provider_id: str, status: str,
                             trigger: str = "", error: Exception | None = None) -> dict:
    display_model = _target_model_for_log(target, provider_id)
    record = {
        "index": index,
        "stage": stage,
        "model": display_model,
        "provider": provider_id or target.provider_id or "",
        "status": status,
    }
    if trigger:
        record["trigger"] = trigger
    if error is not None:
        record["error_message"] = friendly_error_msg(error)
    return record


def _append_fallback_attempt(details: dict, attempt: dict) -> None:
    attempts = details.setdefault("fallback_attempts", [])
    if isinstance(attempts, list):
        attempts.append(attempt)


def _output_request_details(output) -> dict:
    raw = getattr(output, "raw", None)
    if isinstance(raw, dict) and isinstance(raw.get("request_details"), dict):
        return dict(raw["request_details"])
    return {}


def _target_model_for_log(target: RouteTarget, provider_id: str = "") -> str:
    model_id = parse_model_id(target.model)
    resolved_provider = provider_id or target.provider_id or model_id.provider_id
    if resolved_provider and not model_id.is_composite:
        return f"{resolved_provider}/{model_id.model_name}"
    return model_id.composite


def _provider_model_for_target(target: RouteTarget) -> tuple[dict | None, dict | None]:
    provider = resolve_provider(target.model, target.provider_id)
    if not provider:
        return None, None
    model_name = parse_model_id(target.model).model_name
    for model in provider.get("models", []) or []:
        if model.get("id") == model_name:
            return provider, model
    return provider, None


def _target_supports_native_vision(target: RouteTarget) -> bool:
    provider, model = _provider_model_for_target(target)
    return bool(provider and model and _model_supports_native_vision(provider, model))


def _target_uses_preprocessor(target: RouteTarget) -> bool:
    _provider, model = _provider_model_for_target(target)
    return bool(model and model.get("preprocessor"))


async def _call_nonstream_target(target: RouteTarget, internal, *, temperature, max_tokens, log_label: str, stage: str):
    provider_info = resolve_provider(target.model, target.provider_id)
    adapter_provider_id = provider_for_log(provider_info, target.provider_id)
    _app_log.info(
        "[%s upstream.%s.start] target=%s provider=%s provider_type=%s",
        log_label,
        stage,
        target.model,
        adapter_provider_id or "-",
        provider_info.get("provider_type") if provider_info else "unknown",
    )
    if provider_info and provider_info.get("provider_type") == "anthropic":
        output = await anthropic_messages_completion_for_internal(provider_info, internal)
    else:
        response = await anyio.to_thread.run_sync(
            lambda: create_chat_completion(
                model=target.model,
                messages=chat_messages_from_internal(internal),
                provider_id=adapter_provider_id,
                temperature=temperature,
                max_tokens=max_tokens,
                **chat_kwargs_from_internal(internal),
            )
        )
        output = response_to_internal_output(response)
    _app_log.info(
        "[%s upstream.%s.success] target=%s provider=%s tokens=%s text_len=%d tool_calls=%d",
        log_label,
        stage,
        target.model,
        adapter_provider_id or "-",
        output.usage.get("total_tokens", 0),
        len(output.text or ""),
        len(output.tool_calls or []),
    )
    return output, provider_info, adapter_provider_id


async def _internal_for_target_attempt(internal, target: RouteTarget, *, is_fallback: bool):
    if not is_fallback or not has_image_content(internal.messages):
        return internal

    if _target_supports_native_vision(target):
        return internal

    if not _target_uses_preprocessor(target):
        return internal

    attempt = copy.deepcopy(internal)
    await _policy_preprocess_request(attempt, target.model, target.provider_id, target.model)
    return attempt


def _fallback_provider_id_for_target(target: RouteTarget) -> str:
    provider_info = resolve_provider(target.model, target.provider_id)
    return provider_for_log(provider_info, target.provider_id)


async def _call_nonstream_with_fallbacks(policy, internal, *, temperature, max_tokens, log_label: str):
    original_model = internal.target_model
    original_provider = internal.provider_id
    last_exc = None
    primary = RouteTarget(model=original_model, provider_id=original_provider)
    primary = RouteTarget(model=primary.model, provider_id=_fallback_provider_id_for_target(primary))
    targets = [primary]
    fallback_attempts = []
    _app_log.debug(
        "[%s pipeline] primary_call target=%s provider=%s",
        log_label,
        primary.model,
        primary.provider_id or "-",
    )
    for index, target in enumerate(targets):
        internal.target_model = target.model
        internal.provider_id = target.provider_id
        fallback_provider_id = _fallback_provider_id_for_target(target)
        try:
            output, provider_info, adapter_provider_id = await _call_nonstream_target(target, internal, temperature=temperature, max_tokens=max_tokens, log_label=log_label, stage="primary")
            fallback_attempts.append(_fallback_attempt_record(
                index=index,
                stage="primary",
                target=target,
                provider_id=fallback_provider_id,
                status="success",
            ))
            return output, provider_info, adapter_provider_id
        except Exception as exc:
            last_exc = exc
            trigger = classify_upstream_error(exc)
            fallback_attempts.append(_fallback_attempt_record(
                index=index,
                stage="primary",
                target=target,
                provider_id=fallback_provider_id,
                status="failed",
                trigger=trigger,
                error=exc,
            ))
            _app_log.warning(
                "[%s upstream.primary.failed] target=%s provider=%s trigger=%s error=%s",
                log_label,
                target.model,
                _fallback_provider_id_for_target(target) or "-",
                trigger,
                friendly_error_msg(exc),
            )
            decision = apply_fallback_policy(fallback_provider_id, target.model, trigger)
            if not decision.matched:
                _attach_request_details(
                    exc,
                    stream=False,
                    status="fail",
                    attempted_model=target.model,
                    attempted_provider=fallback_provider_id or "",
                    error_trigger=trigger,
                    error_stage="primary",
                    fallback_status="no_policy",
                    fallback_reason=decision.reason,
                    fallback_attempts=fallback_attempts,
                    error_message=friendly_error_msg(exc),
                )
                _app_log.info(
                    "[%s fallback.decision] matched=False source=%s provider=%s trigger=%s reason=%s",
                    log_label,
                    target.model,
                    fallback_provider_id or "-",
                    trigger,
                    decision.reason,
                )
                raise
            _app_log.info(
                "[%s fallback.decision] matched=True policy_id=%s policy='%s' source=%s provider=%s trigger=%s chain=%d",
                log_label,
                decision.policy_id,
                decision.policy_name,
                target.model,
                fallback_provider_id or "-",
                trigger,
                len(decision.chain),
            )
            targets = candidate_targets(primary, decision.chain)
            break

    for index, target in enumerate(targets[1:], 1):
        attempt_internal = await _internal_for_target_attempt(internal, target, is_fallback=True)
        attempt_internal.target_model = target.model
        attempt_internal.provider_id = target.provider_id
        try:
            _app_log.info(
                "[%s fallback.attempt.start] index=%d target=%s provider=%s after_error=%s",
                log_label,
                index,
                target.model,
                target.provider_id or "-",
                friendly_error_msg(last_exc) if last_exc else "",
            )
            output, provider_info, adapter_provider_id = await _call_nonstream_target(target, attempt_internal, temperature=temperature, max_tokens=max_tokens, log_label=log_label, stage="fallback")
            fallback_attempts.append(_fallback_attempt_record(
                index=index,
                stage="fallback",
                target=target,
                provider_id=target.provider_id or "",
                status="success",
            ))
            if getattr(output, "raw", None) is None:
                output.raw = {}
            if isinstance(output.raw, dict):
                output.raw.setdefault("request_details", {})["fallback_attempts"] = fallback_attempts
            internal.target_model = target.model
            internal.provider_id = target.provider_id
            return output, provider_info, adapter_provider_id
        except Exception as exc:
            last_exc = exc
            trigger = classify_upstream_error(exc)
            fallback_attempts.append(_fallback_attempt_record(
                index=index,
                stage="fallback",
                target=target,
                provider_id=target.provider_id or "",
                status="failed",
                trigger=trigger,
                error=exc,
            ))
            _attach_request_details(
                exc,
                stream=False,
                status="fail",
                attempted_model=target.model,
                attempted_provider=target.provider_id or "",
                error_trigger=trigger,
                error_stage="fallback",
                fallback_status="attempt_failed",
                fallback_attempts=fallback_attempts,
                error_message=friendly_error_msg(exc),
            )
            _app_log.warning(
                "[%s fallback.attempt.failed] index=%d target=%s provider=%s trigger=%s error=%s",
                log_label,
                index,
                target.model,
                target.provider_id or "-",
                trigger,
                friendly_error_msg(exc),
            )
    _app_log.error(
        "[%s fallback.exhausted] primary=%s provider=%s candidates=%d error=%s",
        log_label,
        primary.model,
        primary.provider_id or "-",
        max(len(targets) - 1, 0),
        friendly_error_msg(last_exc) if last_exc else "no target available",
    )
    if last_exc is not None:
        _attach_request_details(last_exc, fallback_status="exhausted", fallback_reason="all fallback targets failed", fallback_attempts=fallback_attempts)
    raise last_exc or RuntimeError("No routing target available")


def _stream_events_for_target(target: RouteTarget, internal, *, temperature, max_tokens, log_label: str, strip_thinking=True):
    provider_info = resolve_provider(target.model, target.provider_id)
    adapter_provider_id = provider_for_log(provider_info, target.provider_id)
    _app_log.info(
        "[%s upstream.stream.start] target=%s provider=%s provider_type=%s",
        log_label,
        target.model,
        adapter_provider_id or "-",
        provider_info.get("provider_type") if provider_info else "unknown",
    )
    if provider_info and provider_info.get("provider_type") == "anthropic":
        anthropic_msgs, anthropic_body = anthropic_body_from_internal(internal)
        events = iter_anthropic_output_events(
            provider_info=provider_info,
            messages=anthropic_msgs,
            body=anthropic_body,
            max_tokens=max_tokens,
            temperature=temperature,
            model=target.model,
        )
    else:
        events = iter_openai_chat_output_events(
            model=target.model,
            messages=chat_messages_from_internal(internal),
            provider_id=adapter_provider_id,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=chat_kwargs_from_internal(internal),
            strip_thinking=strip_thinking,
        )
    return events, provider_info, adapter_provider_id


def _is_client_visible_stream_event(event) -> bool:
    if event.kind in ("text_delta", "reasoning_delta"):
        return bool(event.text or event.reasoning)
    if event.kind in ("tool_call_start", "tool_call_arguments_delta", "tool_call_done"):
        return True
    return False


def _stream_event_has_payload(event: InternalOutputEvent) -> bool:
    return bool(
        (event.kind == "text_delta" and event.text)
        or (event.kind == "reasoning_delta" and event.reasoning)
        or (event.kind == "tool_call_start" and (event.tool_call_id or event.name))
        or (event.kind == "tool_call_arguments_delta" and (event.arguments_delta or event.arguments or event.tool_call_id or event.name))
        or (event.kind == "tool_call_done" and (event.tool_call_id or event.name))
    )


def _empty_stream_error(target: RouteTarget, provider_id: str) -> RuntimeError:
    exc = RuntimeError("upstream stream ended without response output")
    exc.attempted_model = target.model
    exc.attempted_provider = provider_id or target.provider_id or ""
    return exc


async def _stream_events_with_fallbacks(internal, *, temperature, max_tokens, log_label: str, strip_thinking=True):
    primary = RouteTarget(model=internal.target_model, provider_id=internal.provider_id)
    primary = RouteTarget(model=primary.model, provider_id=_fallback_provider_id_for_target(primary))
    targets = [primary]
    last_exc = None
    index = 0
    fallback_attempts = []

    while index < len(targets):
        target = targets[index]
        stage = "primary" if index == 0 else "fallback"
        attempt_internal = await _internal_for_target_attempt(internal, target, is_fallback=index > 0)
        attempt_internal.target_model = target.model
        attempt_internal.provider_id = target.provider_id
        fallback_provider_id = _fallback_provider_id_for_target(target)
        emitted = False
        pending_events = []
        try:
            events, provider_info, adapter_provider_id = _stream_events_for_target(
                target,
                attempt_internal,
                temperature=temperature,
                max_tokens=max_tokens,
                log_label=log_label,
                strip_thinking=strip_thinking,
            )
            yield InternalOutputEvent(kind="metadata", metadata={
                "model": _target_model_for_log(target, adapter_provider_id or ""),
                "provider_id": adapter_provider_id or "",
                "stream": True,
                "attempt_index": index,
                "fallback_status": "used" if index > 0 else "unused",
                "fallback_attempts": fallback_attempts + [_fallback_attempt_record(
                    index=index,
                    stage=stage,
                    target=target,
                    provider_id=adapter_provider_id or "",
                    status="started",
                )],
            })
            async for event in events:
                if _is_client_visible_stream_event(event):
                    if not emitted:
                        for pending in pending_events:
                            yield pending
                        pending_events = []
                    emitted = True
                    yield event
                elif emitted:
                    yield event
                else:
                    pending_events.append(event)
            if not emitted and not any(_stream_event_has_payload(event) for event in pending_events):
                raise _empty_stream_error(target, adapter_provider_id or fallback_provider_id or "")
            fallback_attempts.append(_fallback_attempt_record(
                index=index,
                stage=stage,
                target=target,
                provider_id=adapter_provider_id or "",
                status="success",
            ))
            yield InternalOutputEvent(kind="metadata", metadata={
                "model": _target_model_for_log(target, adapter_provider_id or ""),
                "provider_id": adapter_provider_id or "",
                "stream": True,
                "attempt_index": index,
                "fallback_status": "used" if index > 0 else "unused",
                "fallback_attempts": fallback_attempts,
            })
            _app_log.info(
                "[%s upstream.stream.success] stage=%s target=%s provider=%s",
                log_label,
                stage,
                target.model,
                adapter_provider_id or "-",
            )
            return
        except Exception as exc:
            last_exc = exc
            trigger = classify_upstream_error(exc)
            fallback_attempts.append(_fallback_attempt_record(
                index=index,
                stage=stage,
                target=target,
                provider_id=fallback_provider_id or "",
                status="failed",
                trigger=trigger,
                error=exc,
            ))
            _app_log.warning(
                "[%s upstream.stream.failed] stage=%s target=%s provider=%s trigger=%s emitted=%s error=%s",
                log_label,
                stage,
                target.model,
                fallback_provider_id or "-",
                trigger,
                emitted,
                friendly_error_msg(exc),
            )
            if emitted:
                _attach_request_details(
                    exc,
                    stream=True,
                    status="partial",
                    partial_output=True,
                    attempted_model=target.model,
                    attempted_provider=fallback_provider_id or "",
                    error_trigger=trigger,
                    error_stage=stage,
                    fallback_status="skipped",
                    fallback_reason="client_output_started",
                    fallback_attempts=fallback_attempts,
                    error_message=friendly_error_msg(exc),
                )
                _app_log.info(
                    "[%s fallback.stream.skipped] target=%s provider=%s trigger=%s reason=client_output_started",
                    log_label,
                    target.model,
                    fallback_provider_id or "-",
                    trigger,
                )
                raise
            if index == 0:
                decision = apply_fallback_policy(fallback_provider_id, target.model, trigger)
                if not decision.matched:
                    _attach_request_details(
                        exc,
                        stream=True,
                        status="fail",
                        partial_output=False,
                        attempted_model=target.model,
                        attempted_provider=fallback_provider_id or "",
                        error_trigger=trigger,
                        error_stage=stage,
                        fallback_status="no_policy",
                        fallback_reason=decision.reason,
                        fallback_attempts=fallback_attempts,
                        error_message=friendly_error_msg(exc),
                    )
                    _app_log.info(
                        "[%s fallback.stream.decision] matched=False source=%s provider=%s trigger=%s reason=%s",
                        log_label,
                        target.model,
                        fallback_provider_id or "-",
                        trigger,
                        decision.reason,
                    )
                    raise
                targets = candidate_targets(primary, decision.chain)
                _app_log.info(
                    "[%s fallback.stream.decision] matched=True policy_id=%s policy='%s' source=%s provider=%s trigger=%s chain=%d",
                    log_label,
                    decision.policy_id,
                    decision.policy_name,
                    target.model,
                    fallback_provider_id or "-",
                    trigger,
                    len(targets) - 1,
                )
            index += 1
            if index < len(targets):
                next_target = targets[index]
                _app_log.info(
                    "[%s fallback.stream.attempt.start] index=%d target=%s provider=%s after_error=%s",
                    log_label,
                    index,
                    next_target.model,
                    next_target.provider_id or "-",
                    friendly_error_msg(last_exc),
                )

    _app_log.error(
        "[%s fallback.stream.exhausted] primary=%s provider=%s candidates=%d error=%s",
        log_label,
        primary.model,
        primary.provider_id or "-",
        max(len(targets) - 1, 0),
        friendly_error_msg(last_exc) if last_exc else "no target available",
    )
    if last_exc is not None:
        _attach_request_details(last_exc, fallback_status="exhausted", fallback_reason="all fallback targets failed", fallback_attempts=fallback_attempts)
    raise last_exc or RuntimeError("No routing target available")


def _log_request(username: str, api_key: str, model: str, provider_id: str,
                 endpoint: str, success: bool, tokens: int,
                 requested_model: str = "", *, details: dict | None = None) -> None:
    detail = dict(details or {})
    status = str(detail.get("status") or ("ok" if success else "fail"))
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "full_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "username": username,
        "api_key": mask_key(api_key),
        "model": model,
        "requested_model": requested_model or model,
        "provider": provider_id or "",
        "endpoint": endpoint,
        "success": success,
        "tokens": tokens,
        "status": status,
        "details": detail,
    }
    for key in (
        "stream",
        "partial_output",
        "attempted_model",
        "attempted_provider",
        "fallback_status",
        "fallback_reason",
        "error_trigger",
        "error_stage",
        "error_message",
        "attempt_index",
    ):
        if key in detail:
            entry[key] = detail[key]
    with _request_log_lock:
        _request_log.appendleft(entry)
    # Also write to structured access log
    if success:
        _access_log.info("[OK] %s user=%s model=%s provider=%s tokens=%d",
                         endpoint, username, model, provider_id or "-", tokens)
    else:
        _access_log.warning("[FAIL] %s user=%s model=%s provider=%s",
                           endpoint, username, model, provider_id or "-")
    # Write to persistent history for stats
    try:
        add_request_record(model=requested_model or model, username=username, success=success, tokens=tokens)
    except Exception as e:
        _app_log.warning("Failed to log request: %s", e)


def _log_request_body(username: str, model: str, endpoint: str, body: dict) -> None:
    """Log request metadata for debugging (truncated body, DEBUG level by default)."""
    _req_log.debug(
        "[%s] user=%s model=%s stream=%s tools=%d msgs=%d body_len=%d",
        endpoint, username, model,
        body.get("stream", False),
        len(body.get("tools", [])),
        len(body.get("messages", [])),
        len(json.dumps(body, ensure_ascii=False, default=str)),
    )


async def _policy_preprocess_request(internal, model: str, provider_id: str, requested_model: str):
    check_model = requested_model or model
    has_img = has_image_content(internal.messages)
    _app_log.info(
        "[preprocess.decision] requested=%s target=%s provider=%s has_image=%s messages=%d",
        check_model,
        model,
        provider_id or "-",
        has_img,
        len(internal.messages),
    )

    mid = parse_model_id(check_model)
    with get_db() as db:
        if mid.provider_id:
            row = db.execute(
                "SELECT preprocessor FROM provider_models WHERE provider_id = ? AND model_id = ? AND enabled = 1 LIMIT 1",
                (mid.provider_id, mid.model_name)
            ).fetchone()
        else:
            row = db.execute(
                "SELECT preprocessor FROM provider_models WHERE model_id = ? AND enabled = 1 ORDER BY provider_id LIMIT 1",
                (mid.model_name,)
            ).fetchone()
    _app_log.debug("[preprocess.lookup] requested=%s row=%s", check_model, dict(row) if row else None)
    if not row or not row["preprocessor"]:
        if has_img:
            _app_log.warning("[preprocess.decision] enabled=False requested=%s reason=model_preprocessor_disabled", check_model)
        else:
            _app_log.info("[preprocess.decision] enabled=False requested=%s reason=no_images", check_model)
        return False

    preprocessor_config = get_enabled_preprocessor()
    if not preprocessor_config:
        _app_log.warning("[preprocess.decision] enabled=False requested=%s reason=no_enabled_preprocessor_config", check_model)
        return False
    preprocessor_id = preprocessor_config.get("id", "")
    preprocessor_config["id"] = preprocessor_id
    await preprocess_messages(internal.messages, preprocessor_config)
    _app_log.info(
        "[preprocess.vision.completed] requested=%s preprocessor=%s modified=%s messages=%d",
        check_model,
        preprocessor_id,
        has_img,
        len(internal.messages),
    )
    return has_img


def get_request_log() -> list:
    with _request_log_lock:
        return list(_request_log)


def clear_request_log() -> None:
    with _request_log_lock:
        _request_log.clear()


def _minute_key_from_log_entry(entry: dict) -> str:
    return entry.get("full_time", entry["time"])[:16].replace("T", " ")


def _parse_minute_key(value: str):
    from datetime import datetime

    return datetime.strptime(value[:16].replace("T", " "), "%Y-%m-%d %H:%M")


def _realtime_minute_keys_with_small_gaps(keys: list[str], max_gap_minutes: int = 30) -> list[str]:
    """Fill short gaps, but keep realtime charts focused when activity is sparse."""
    sorted_keys = sorted(set(keys))
    if len(sorted_keys) < 2:
        return sorted_keys

    from datetime import timedelta

    expanded = [sorted_keys[0]]
    previous = _parse_minute_key(sorted_keys[0])
    for key in sorted_keys[1:]:
        current = _parse_minute_key(key)
        gap_minutes = int((current - previous).total_seconds() // 60)
        if 1 < gap_minutes <= max_gap_minutes:
            cursor = previous + timedelta(minutes=1)
            while cursor < current:
                expanded.append(cursor.strftime("%Y-%m-%d %H:%M"))
                cursor += timedelta(minutes=1)
        expanded.append(key)
        previous = current
    return expanded


def get_timeline_data() -> dict:
    """Aggregate requests by minute for the realtime chart."""
    with _request_log_lock:
        snapshot = list(_request_log)
    if not snapshot:
        return {"labels": [], "success": [], "failed": []}
    buckets: dict[str, dict] = {}
    for entry in snapshot:
        minute = _minute_key_from_log_entry(entry)
        if minute not in buckets:
            buckets[minute] = {"label": minute[-5:], "success": 0, "failed": 0}
        if entry["success"]:
            buckets[minute]["success"] += 1
        else:
            buckets[minute]["failed"] += 1
    sorted_keys = _realtime_minute_keys_with_small_gaps(list(buckets.keys()))
    sorted_buckets = [(key, buckets.get(key, {"label": key[-5:], "success": 0, "failed": 0})) for key in sorted_keys]
    return {
        "labels": [b["label"] for _, b in sorted_buckets],
        "success": [b["success"] for _, b in sorted_buckets],
        "failed": [b["failed"] for _, b in sorted_buckets],
    }


def get_model_distribution() -> dict:
    """Model usage distribution for pie chart."""
    with _request_log_lock:
        snapshot = list(_request_log)
    counts: dict[str, int] = {}
    for entry in snapshot:
        model = entry["model"]
        counts[model] = counts.get(model, 0) + 1
    sorted_models = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return {
        "labels": [m for m, _ in sorted_models],
        "counts": [c for _, c in sorted_models],
    }


def get_model_stats() -> dict:
    """Aggregate per-model stats from recent request log."""
    with _request_log_lock:
        snapshot = list(_request_log)
    models = {}
    for entry in snapshot:
        mid = entry["model"]
        if mid not in models:
            models[mid] = {"total": 0, "failed": 0, "tokens": 0}
        models[mid]["total"] += 1
        if not entry["success"]:
            models[mid]["failed"] += 1
        models[mid]["tokens"] += entry["tokens"]
    return models


def get_timeline_model_data() -> dict:
    """Per-model per-minute breakdown from request log for the realtime chart."""
    with _request_log_lock:
        snapshot = list(_request_log)
    if not snapshot:
        return {"labels": [], "models": [], "calls": [], "tokens": []}
    buckets: dict[str, dict] = {}
    for entry in snapshot:
        minute = _minute_key_from_log_entry(entry)
        if minute not in buckets:
            buckets[minute] = {}
        model = entry["model"]
        if model not in buckets[minute]:
            buckets[minute][model] = {"total": 0, "tokens": 0}
        buckets[minute][model]["total"] += 1
        buckets[minute][model]["tokens"] += entry["tokens"]
    sorted_keys = _realtime_minute_keys_with_small_gaps(list(buckets.keys()))
    all_models = sorted({m for b in buckets.values() for m in b})
    return {
        "labels": [k[-5:] for k in sorted_keys],
        "models": all_models,
        "calls": [[buckets.get(k, {}).get(m, {}).get("total", 0) for k in sorted_keys] for m in all_models],
        "tokens": [[buckets.get(k, {}).get(m, {}).get("tokens", 0) for k in sorted_keys] for m in all_models],
    }


def verify_api_key(authorization: Optional[str] = Header(None)) -> tuple[dict, dict]:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")

    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")

    user_match = find_user_by_api_key(token)
    if user_match:
        return user_match

    raise HTTPException(status_code=401, detail="Invalid API key")

def allowed_models_for(user: dict, api_key: dict) -> list:
    # Only key-level allowed_models matters. User is just enable/disable.
    key_models = api_key.get("allowed_models")
    if key_models is None:
        return ["*"]  # not configured -> unrestricted
    if "*" in key_models:
        return ["*"]
    return key_models  # explicit list, empty = deny all


def _model_allowed_by_list(allowed: list, model: str) -> bool:
    requested = parse_model_id(model)
    for allowed_model in allowed:
        allowed_mid = parse_model_id(str(allowed_model))
        if allowed_mid.is_composite:
            if requested.is_composite and requested == allowed_mid:
                return True
        elif requested.model_name == allowed_mid.model_name:
            return True
    return False


def ensure_model_allowed(user: dict, api_key: dict, model: str) -> None:
    allowed = allowed_models_for(user, api_key)
    if "*" in allowed:
        return
    requested = parse_model_id(model)
    if _model_allowed_by_list(allowed, model):
        return
    if any("/" in str(allowed_model) for allowed_model in allowed) and not requested.is_composite:
        raise HTTPException(
            status_code=403,
            detail=f"Model '{model}' is not allowed for this API key; use a provider-qualified model id",
        )
    raise HTTPException(status_code=403, detail=f"Model '{model}' is not allowed for this API key")


def ensure_routed_model_allowed(user: dict, api_key: dict, requested_model: str, target_model: str, target_provider: str = "") -> None:
    if requested_model == target_model and not target_provider:
        ensure_model_allowed(user, api_key, requested_model)
        return
    allowed = allowed_models_for(user, api_key)
    if "*" in allowed:
        return

    target = parse_model_id(target_model)
    effective_target = f"{target_provider}/{target.model_name}" if target_provider and not target.is_composite else target.composite
    if _model_allowed_by_list(allowed, effective_target):
        return

    if _model_allowed_by_list(allowed, requested_model):
        return

    requested = parse_model_id(requested_model)
    if any("/" in str(allowed_model) for allowed_model in allowed) and not requested.is_composite:
        raise HTTPException(
            status_code=403,
            detail=f"Model '{requested_model}' is not allowed for this API key; use a provider-qualified model id",
        )
    raise HTTPException(status_code=403, detail=f"Model '{requested_model}' is not allowed for this API key")

@router.get("/models")
def list_models(authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)
    allowed = allowed_models_for(user, api_key)
    models = []

    for provider in get_providers():
        if provider.get("enabled"):
            for model in provider.get("models", []):
                if model.get("enabled"):
                    composite_id = f"{provider['id']}/{model['id']}"
                    # Check allow-list support for composite IDs, simple model IDs, and wildcard
                    if "*" not in allowed and model["id"] not in allowed and composite_id not in allowed:
                        continue
                    entry = {
                        "id": composite_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": provider["name"],
                        "provider": provider["id"]
                    }
                    # Models with native vision support or a vision preprocessor should advertise image support
                    # so clients such as Codex/OpenCode send image blocks instead of text placeholders.
                    if _model_should_advertise_vision(provider, model):
                        entry["supports_vision"] = True
                        entry["image_support"] = True
                        entry["multimodal"] = True
                    models.append(entry)

    return {"object": "list", "data": models}


def _model_supports_native_vision(provider: dict, model: dict) -> bool:
    """Best-effort client capability hint for models that accept images natively."""
    model_id = str(model.get("id") or "").lower()
    model_name = str(model.get("name") or "").lower()
    text = f"{model_id} {model_name}"

    if any(marker in text for marker in ("embedding", "rerank", "audio", "tts", "whisper", "image-")):
        return False

    vision_markers = (
        "gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "claude-3",
        "claude-opus-4",
        "claude-sonnet-4",
        "gemini",
        "qwen-vl",
        "qwen2-vl",
        "qwen2.5-vl",
        "qwen3-vl",
        "minicpm-v",
        "llava",
        "vision",
        "vl-",
        "-vl",
    )
    return any(marker in text for marker in vision_markers)


def _model_should_advertise_vision(provider: dict, model: dict) -> bool:
    return bool(model.get("preprocessor")) or _model_supports_native_vision(provider, model)

@router.post("/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)

    body = await request.json()
    internal = chat_completions_to_internal(body)
    model = internal.target_model
    temperature = internal.temperature
    max_tokens = internal.max_tokens
    provider_id = internal.provider_id
    stream = internal.stream

    username = user.get("username", "legacy")
    api_key_value = api_key.get("key", "")
    _log_request_body(username, model, "chat", body)

    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if not internal.messages:
        raise HTTPException(status_code=400, detail="messages is required")

    requested_model = model
    policy = await prepare_request_policy(
        internal,
        username=username,
        api_key_value=api_key_value,
        preprocess_request=_policy_preprocess_request,
        conversation_cache_key=_conversation_cache_key,
        reasoning_context=_reasoning_context,
        tool_only_turns=_tool_only_turns,
        tool_only_limit=TOOL_ONLY_LIMIT,
        log_label="chat",
    )
    model = internal.target_model
    provider_id = internal.provider_id
    ensure_routed_model_allowed(user, api_key, requested_model, model, provider_id)
    conv_key = policy.conv_key
    provider_info = None
    adapter_provider_id = provider_id or ""

    try:
        if stream:
            events = _stream_events_with_fallbacks(
                internal,
                temperature=temperature,
                max_tokens=max_tokens,
                log_label="chat",
            )
            return StreamingResponse(
                _stream_internal_output(
                    events=events,
                    endpoint="chat_completions",
                    model=model,
                    username=username,
                    api_key_value=api_key_value,
                    provider_id=adapter_provider_id,
                    requested_model=requested_model,
                    log_request=_log_request,
                    record_request_log=_build_stream_recorder("chat_completions", username, api_key_value, requested_model, body),
                    conv_key=conv_key,
                    remember_reasoning_content=_remember_reasoning_content,
                    tool_only_turns=_tool_only_turns,
                ),
                media_type="text/event-stream"
            )

        output, provider_info, adapter_provider_id = await _call_nonstream_with_fallbacks(
            policy,
            internal,
            temperature=temperature,
            max_tokens=max_tokens,
            log_label="chat",
        )
        model = internal.target_model
        provider_id = internal.provider_id
        logged_model = _target_model_for_log(RouteTarget(model=model, provider_id=adapter_provider_id or provider_id or ""), adapter_provider_id or provider_id or "")
        if output.reasoning:
            _remember_reasoning_content(conv_key, output.reasoning, [tool.id for tool in output.tool_calls])
            _app_log.debug("[chat_nonstream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d",
                          conv_key[:40], len(output.reasoning),
                          output.usage.get("prompt_cache_hit_tokens", 0), output.usage.get("prompt_cache_miss_tokens", 0))

        if output.tool_calls and not output.text:
            _tool_only_turns.increment(conv_key)
        else:
            _tool_only_turns.reset(conv_key)

        rendered = render_chat_completion(output, model=model)
        _record_request_log(
            endpoint="chat_completions",
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            final_model=logged_model, final_provider=adapter_provider_id or "",
            request_body=body, response_body=rendered,
            success=True, status="ok", tokens=output.usage.get("total_tokens", 0),
            usage=output.usage, details=_output_request_details(output),
        )
        _log_request(username, api_key_value, logged_model, adapter_provider_id or "", "chat_completions", True, output.usage.get("total_tokens", 0), requested_model, details=_output_request_details(output))
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, output.usage.get("total_tokens", 0))
        return rendered
    except Exception as e:
        _error_log.error("[chat] %s", str(e))
        details = _request_details_from_exception(
            e,
            stream=False,
            attempted_model=getattr(e, "attempted_model", None) or model or requested_model,
            attempted_provider=getattr(e, "attempted_provider", None) or provider_id or "",
        )
        _log_request(username, api_key_value, details.get("attempted_model") or requested_model, details.get("attempted_provider") or provider_id or "", "chat_completions", False, 0, requested_model, details=details)
        _record_request_log(
            endpoint="chat_completions",
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            final_model=details.get("attempted_model") or requested_model,
            final_provider=details.get("attempted_provider") or provider_id or "",
            request_body=body, response_body=None,
            success=False, status=details.get("status", "fail"),
            tokens=0, details=details, error_message=friendly_error_msg(e),
        )
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        raise HTTPException(status_code=500, detail=friendly_error_msg(e))

@router.post("/completions")
async def completions(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)

    body = await request.json()
    internal = completions_to_internal(body)
    model = internal.target_model
    provider_id = internal.provider_id
    stream = internal.stream
    temperature = internal.temperature
    max_tokens = internal.max_tokens

    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    username = user.get("username", "legacy")
    _log_request_body(username, model, "completions", body)
    api_key_value = api_key.get("key", "")
    requested_model = model

    policy = await prepare_request_policy(
        internal,
        username=username,
        api_key_value=api_key_value,
        preprocess_request=_policy_preprocess_request,
        conversation_cache_key=_conversation_cache_key,
        reasoning_context=None,
        normalize=True,
        log_label="completions",
    )
    model = internal.target_model
    provider_id = internal.provider_id
    ensure_routed_model_allowed(user, api_key, requested_model, model, provider_id)
    conv_key = policy.conv_key
    provider_info = None
    adapter_provider_id = provider_id or ""

    try:
        if stream:
            events = _stream_events_with_fallbacks(
                internal,
                temperature=temperature,
                max_tokens=max_tokens,
                log_label="completions",
            )
            return StreamingResponse(
                _stream_internal_output(
                    events=events,
                    endpoint="completions",
                    model=model,
                    username=username,
                    api_key_value=api_key_value,
                    provider_id=adapter_provider_id,
                    requested_model=requested_model,
                    log_request=_log_request,
                    record_request_log=_build_stream_recorder("completions", username, api_key_value, requested_model, body),
                    conv_key=conv_key,
                ),
                media_type="text/event-stream"
            )

        output, provider_info, adapter_provider_id = await _call_nonstream_with_fallbacks(
            policy,
            internal,
            temperature=temperature,
            max_tokens=max_tokens,
            log_label="completions",
        )
        model = internal.target_model
        provider_id = internal.provider_id
        logged_model = _target_model_for_log(RouteTarget(model=model, provider_id=adapter_provider_id or provider_id or ""), adapter_provider_id or provider_id or "")
        rendered = render_completion(output, model=model)
        _log_request(username, api_key_value, logged_model, adapter_provider_id or "", "completions", True, output.usage.get("total_tokens", 0), requested_model)
        _record_request_log(
            endpoint="completions",
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            final_model=logged_model, final_provider=adapter_provider_id or "",
            request_body=body, response_body=rendered,
            success=True, status="ok", tokens=output.usage.get("total_tokens", 0),
            usage=output.usage, details=_output_request_details(output),
        )
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, output.usage.get("total_tokens", 0))
        return rendered
    except Exception as e:
        details = _request_details_from_exception(
            e,
            stream=False,
            attempted_model=getattr(e, "attempted_model", None) or model or requested_model,
            attempted_provider=getattr(e, "attempted_provider", None) or provider_id or "",
        )
        _log_request(username, api_key_value, details.get("attempted_model") or model or requested_model, details.get("attempted_provider") or provider_id or "", "completions", False, 0, requested_model, details=details)
        _record_request_log(
            endpoint="completions",
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            final_model=details.get("attempted_model") or model or requested_model,
            final_provider=details.get("attempted_provider") or provider_id or "",
            request_body=body, response_body=None,
            success=False, status=details.get("status", "fail"),
            tokens=0, details=details, error_message=friendly_error_msg(e),
        )
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", str(e))
        raise HTTPException(status_code=500, detail=friendly_error_msg(e))

@router.post("/messages")
async def anthropic_messages(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)

    body = await request.json()
    model = body.get("model")
    anthropic_msgs = body.get("messages", [])
    provider_id = body.get("provider_id")
    stream = body.get("stream", False)
    previous_response_id = body.get("previous_response_id") or ""
    internal = anthropic_messages_to_internal({**body, "provider_id": provider_id})
    system_prompt = internal.system
    _app_log.debug("[ANTHRO_ENTRY] model=%s msgs=%d system=%s tools=%s",
                  model, len(anthropic_msgs),
                  "yes" if system_prompt else "no",
                  "yes" if body.get("tools") else "no")
    temperature = body.get("temperature")

    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    username = user.get("username", "legacy")
    api_key_value = api_key.get("key", "")
    requested_model = model

    policy = await prepare_request_policy(
        internal,
        username=username,
        api_key_value=api_key_value,
        preprocess_request=_policy_preprocess_request,
        conversation_cache_key=_conversation_cache_key,
        reasoning_context=_reasoning_context,
        normalize=False,
        log_label="messages",
    )
    model = internal.target_model
    provider_id = internal.provider_id
    ensure_routed_model_allowed(user, api_key, requested_model, model, provider_id)
    provider_info = resolve_provider(model, provider_id)
    adapter_provider_id = provider_for_log(provider_info, provider_id)
    previous_response_id = internal.previous_response_id
    max_tokens = internal.max_tokens
    temperature = internal.temperature
    system_prompt = internal.system
    _app_log.debug(
        "[messages] NORMALIZED anthropic(%d msgs) -> internal(%d msgs) system_prompt_len=%d tools=%s stream=%s max_tokens=%s model=%s provider_type=%s",
        len(anthropic_msgs), len(internal.messages), len(system_prompt) if system_prompt else 0,
        str(body.get("tools", [])[:10]) if body.get("tools") else "none",
        str(body.get("stream")), str(max_tokens), model,
        provider_info.get("provider_type") if provider_info else "unknown",
    )

    conv_key = policy.conv_key

    try:
        if stream:
            events = _stream_events_with_fallbacks(
                internal,
                temperature=temperature,
                max_tokens=max_tokens,
                log_label="messages",
                strip_thinking=False,
            )
            return StreamingResponse(
                _stream_internal_output(
                    events=events,
                    endpoint="messages",
                    model=model,
                    username=username,
                    api_key_value=api_key_value,
                    provider_id=adapter_provider_id,
                    requested_model=requested_model,
                    log_request=_log_request,
                    record_request_log=_build_stream_recorder("messages", username, api_key_value, requested_model, body),
                    conv_key=conv_key,
                    remember_reasoning_content=_remember_reasoning_content,
                ),
                media_type="text/event-stream"
            )

        output, provider_info, adapter_provider_id = await _call_nonstream_with_fallbacks(
            policy,
            internal,
            temperature=temperature,
            max_tokens=max_tokens,
            log_label="messages",
        )
        model = internal.target_model
        provider_id = internal.provider_id
        logged_model = _target_model_for_log(RouteTarget(model=model, provider_id=adapter_provider_id or provider_id or ""), adapter_provider_id or provider_id or "")
        if output.reasoning:
            _remember_reasoning_content(conv_key, output.reasoning, [tool.id for tool in output.tool_calls])
            _app_log.debug("[messages_nonstream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d",
                          conv_key[:60], len(output.reasoning),
                          output.usage.get("prompt_cache_hit_tokens", 0), output.usage.get("prompt_cache_miss_tokens", 0))
        rendered = render_anthropic_message(output, model=model)
        _log_request(username, api_key_value, logged_model, adapter_provider_id, "messages", True, output.usage.get("total_tokens", 0), requested_model, details=_output_request_details(output))
        _record_request_log(
            endpoint="messages",
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            final_model=logged_model, final_provider=adapter_provider_id,
            request_body=body, response_body=rendered,
            success=True, status="ok", tokens=output.usage.get("total_tokens", 0),
            usage=output.usage, details=_output_request_details(output),
        )
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, output.usage.get("total_tokens", 0))
        return rendered
    except Exception as e:
        details = _request_details_from_exception(
            e,
            stream=False,
            attempted_model=getattr(e, "attempted_model", None) or model or requested_model,
            attempted_provider=getattr(e, "attempted_provider", None) or adapter_provider_id or provider_id or "",
        )
        _log_request(username, api_key_value, details.get("attempted_model") or model or requested_model, details.get("attempted_provider") or adapter_provider_id or "", "messages", False, 0, requested_model, details=details)
        _record_request_log(
            endpoint="messages",
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            final_model=details.get("attempted_model") or model or requested_model,
            final_provider=details.get("attempted_provider") or adapter_provider_id or "",
            request_body=body, response_body=None,
            success=False, status=details.get("status", "fail"),
            tokens=0, details=details, error_message=friendly_error_msg(e),
        )
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", str(e))
        raise HTTPException(status_code=500, detail=friendly_error_msg(e))


@router.post("/responses")
async def responses_endpoint(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization)

    body = await request.json()
    internal = responses_to_internal(body)
    model = internal.target_model
    input_data = body.get("input", "")
    instructions = internal.metadata.get("instructions", "")
    temperature = internal.temperature
    max_tokens = internal.max_tokens
    provider_id = internal.provider_id
    stream = internal.stream
    previous_response_id = internal.previous_response_id

    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if not input_data:
        raise HTTPException(status_code=400, detail="input is required")

    # Log Codex request details for debugging
    tools_count = len(body.get("tools", []))
    input_len = len(json.dumps(body.get("input", ""), ensure_ascii=False))
    instructions_len = len(body.get("instructions", ""))
    # Log input item types for debugging tool loop
    if isinstance(body.get("input"), list):
        item_types = {}
        for item in body["input"]:
            t = item.get("type", "unknown") if isinstance(item, dict) else "non-dict"
            item_types[t] = item_types.get(t, 0) + 1
        _app_log.debug("[responses] model=%s stream=%s tools=%d input_len=%d instructions_len=%d input_types=%s", model, stream, tools_count, input_len, instructions_len, str(item_types))
    else:
        _app_log.debug("[responses] model=%s stream=%s tools=%d input_len=%d instructions_len=%d", model, stream, tools_count, input_len, instructions_len)

    requested_model = model
    username = user.get("username", "legacy")
    api_key_value = api_key.get("key", "")

    if isinstance(input_data, str):
        pass
    elif isinstance(input_data, list):
        _app_log.debug(
            "[responses CONVERT] input_items=%d ir_messages=%d roles=%s tool_msgs=%d rc_msgs=%d",
            len(input_data),
            len(internal.messages),
            [m.role for m in internal.messages],
            _ir_tool_message_count(internal.messages),
            _ir_reasoning_message_count(internal.messages),
        )
    else:
        raise HTTPException(status_code=400, detail="input must be a string or list of messages")

    policy = await prepare_request_policy(
        internal,
        username=username,
        api_key_value=api_key_value,
        preprocess_request=_policy_preprocess_request,
        conversation_cache_key=_conversation_cache_key,
        reasoning_context=_reasoning_context if isinstance(input_data, list) else None,
        log_label="responses",
    )
    model = internal.target_model
    provider_id = internal.provider_id
    ensure_routed_model_allowed(user, api_key, requested_model, model, provider_id)
    conv_key = policy.conv_key
    provider_info = None
    adapter_provider_id = provider_id or ""

    if isinstance(input_data, list):
        _app_log.debug(
            "[responses REASONING] injected=%d ir_messages=%d tool_msgs=%d rc_msgs=%d conv_key=%s",
            policy.reasoning_injected,
            len(internal.messages),
            _ir_tool_message_count(internal.messages),
            _ir_reasoning_message_count(internal.messages),
            conv_key[:60],
        )

    try:
        if stream:
            events = _stream_events_with_fallbacks(
                internal,
                temperature=temperature,
                max_tokens=max_tokens,
                log_label="responses",
            )
            return StreamingResponse(
                _stream_internal_output(
                    events=events,
                    endpoint="responses",
                    model=model,
                    username=username,
                    api_key_value=api_key_value,
                    provider_id=adapter_provider_id,
                    requested_model=requested_model,
                    log_request=_log_request,
                    record_request_log=_build_stream_recorder("responses", username, api_key_value, requested_model, body),
                    previous_response_id=previous_response_id,
                    conv_key=conv_key,
                    remember_response_chain_key=_remember_response_chain_key,
                    remember_reasoning_content=_remember_reasoning_content,
                    tool_only_turns=_tool_only_turns,
                ),
                media_type="text/event-stream"
            )

        output, provider_info, adapter_provider_id = await _call_nonstream_with_fallbacks(
            policy,
            internal,
            temperature=temperature,
            max_tokens=max_tokens,
            log_label="responses",
        )
        model = internal.target_model
        provider_id = internal.provider_id
        logged_model = _target_model_for_log(RouteTarget(model=model, provider_id=adapter_provider_id or provider_id or ""), adapter_provider_id or provider_id or "")
        if output.reasoning:
            _remember_reasoning_content(conv_key, output.reasoning, [tool.id for tool in output.tool_calls])
            _app_log.debug("[responses_nonstream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d",
                          conv_key, len(output.reasoning),
                          output.usage.get("prompt_cache_hit_tokens", 0), output.usage.get("prompt_cache_miss_tokens", 0))

        resp_id = f"resp_{uuid.uuid4().hex}"
        _remember_response_chain_key(resp_id, conv_key)
        rendered = render_response(output, model=model, previous_response_id=previous_response_id, response_id=resp_id)
        success_details = _output_request_details(output)
        success_details["response_id"] = resp_id
        _log_request(username, api_key_value, logged_model, adapter_provider_id, "responses", True, output.usage.get("total_tokens", 0), requested_model, details=success_details)
        _record_request_log(
            endpoint="responses",
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            final_model=logged_model, final_provider=adapter_provider_id,
            request_body=body, response_body=rendered,
            success=True, status="ok", tokens=output.usage.get("total_tokens", 0),
            usage=output.usage, details=success_details,
        )
        increment_global_stats(success=True)
        if username != "legacy":
            increment_user_usage(username, api_key_value, True, output.usage.get("total_tokens", 0))
        return rendered
    except Exception as e:
        details = _request_details_from_exception(
            e,
            stream=False,
            attempted_model=getattr(e, "attempted_model", None) or model or requested_model,
            attempted_provider=getattr(e, "attempted_provider", None) or provider_for_log(provider_info, provider_id),
        )
        _log_request(username, api_key_value, details.get("attempted_model") or model or requested_model, details.get("attempted_provider") or provider_for_log(provider_info, provider_id), "responses", False, 0, requested_model, details=details)
        _record_request_log(
            endpoint="responses",
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            final_model=details.get("attempted_model") or model or requested_model,
            final_provider=details.get("attempted_provider") or provider_for_log(provider_info, provider_id),
            request_body=body, response_body=None,
            success=False, status=details.get("status", "fail"),
            tokens=0, details=details, error_message=friendly_error_msg(e),
        )
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", str(e))
        raise HTTPException(status_code=500, detail=friendly_error_msg(e))


# -- Request/Response detail log recorder --
_PAYLOAD_MAX_BYTES = 64 * 1024
_STREAMED_TEXT_MAX = 16 * 1024
_STREAMED_REASONING_MAX = 16 * 1024
_STREAMED_TOOL_MAX = 8

def _truncate_payload(value, max_bytes=_PAYLOAD_MAX_BYTES):
    if value is None:
        return None
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        encoded = repr(value)
    if len(encoded.encode('utf-8')) <= max_bytes:
        return value
    truncated = encoded.encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')
    return {'_truncated': True, 'original_bytes': len(encoded.encode('utf-8')), 'data': truncated + '...'}

def _compact_text(value, max_chars):
    if value is None:
        return None
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + '...'


def _request_log_response_body(
    *,
    endpoint,
    final_model,
    response_body=None,
    streamed_text=None,
    streamed_reasoning=None,
    streamed_tool_calls=None,
    usage=None,
    success,
    status,
    error_message=None,
):
    if response_body is not None:
        return response_body
    if streamed_text is not None or streamed_reasoning is not None or streamed_tool_calls is not None:
        return {
            'type': 'stream_summary',
            'endpoint': endpoint,
            'status': status,
            'model': final_model or '',
            'text': _compact_text(streamed_text or '', _STREAMED_TEXT_MAX),
            'reasoning': _compact_text(streamed_reasoning or '', _STREAMED_REASONING_MAX),
            'tool_calls': (streamed_tool_calls or [])[:_STREAMED_TOOL_MAX],
            'usage': usage or {},
        }
    if not success or error_message:
        return {
            'error': {
                'message': error_message or 'request failed',
                'type': 'server_error',
            },
            'status': status,
            'model': final_model or '',
        }
    return None


def _record_request_log(
    *,
    endpoint,
    username,
    api_key_value,
    requested_model,
    final_model,
    final_provider,
    request_body,
    response_body=None,
    streamed_text=None,
    streamed_reasoning=None,
    streamed_tool_calls=None,
    usage=None,
    success,
    status,
    tokens,
    details=None,
    partial_output=False,
    error_message=None,
):
    payload_details = dict(details or {})
    if usage and 'usage' not in payload_details:
        payload_details['usage'] = usage
    if streamed_text is not None:
        compact = _compact_text(streamed_text, _STREAMED_TEXT_MAX)
        payload_details['streamed_text'] = compact
        if compact != streamed_text:
            payload_details['streamed_text_truncated'] = True
    if streamed_reasoning is not None:
        compact = _compact_text(streamed_reasoning, _STREAMED_REASONING_MAX)
        payload_details['streamed_reasoning'] = compact
        if compact != streamed_reasoning:
            payload_details['streamed_reasoning_truncated'] = True
    if streamed_tool_calls is not None:
        if len(streamed_tool_calls) > _STREAMED_TOOL_MAX:
            payload_details['streamed_tool_calls'] = streamed_tool_calls[:_STREAMED_TOOL_MAX]
            payload_details['streamed_tool_calls_truncated'] = True
        else:
            payload_details['streamed_tool_calls'] = streamed_tool_calls
    if partial_output and 'partial_output' not in payload_details:
        payload_details['partial_output'] = True
    if error_message and 'error_message' not in payload_details:
        payload_details['error_message'] = error_message
    final_response_body = _request_log_response_body(
        endpoint=endpoint,
        final_model=final_model,
        response_body=response_body,
        streamed_text=streamed_text,
        streamed_reasoning=streamed_reasoning,
        streamed_tool_calls=streamed_tool_calls,
        usage=usage,
        success=success,
        status=status,
        error_message=error_message,
    )
    try:
        add_request_log(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            endpoint=endpoint,
            username=username or '',
            api_key=mask_key(api_key_value or ''),
            requested_model=requested_model or '',
            model=final_model or '',
            provider=final_provider or '',
            status=status,
            stream=response_body is None and (streamed_text is not None or streamed_reasoning is not None or streamed_tool_calls is not None),
            tokens=int(tokens or 0),
            request_body=_truncate_payload(request_body),
            response_body=_truncate_payload(final_response_body),
            details=payload_details,
            error=(error_message or ''),
        )
    except Exception as exc:
        _app_log.warning('add_request_log failed: %s', exc)
    try:
        trim_request_logs(get_default('request_log_max', 500))
    except Exception as exc:
        _app_log.warning('trim_request_logs failed: %s', exc)

def _build_stream_recorder(
    endpoint,
    username,
    api_key_value,
    requested_model,
    request_body,
):
    def _record(**payload):
        _record_request_log(
            endpoint=endpoint,
            username=username,
            api_key_value=api_key_value,
            requested_model=requested_model,
            final_model=payload.get('final_model') or '',
            final_provider=payload.get('final_provider_id') or '',
            request_body=request_body,
            response_body=None,
            streamed_text=payload.get('streamed_text'),
            streamed_reasoning=payload.get('streamed_reasoning'),
            streamed_tool_calls=payload.get('streamed_tool_calls') or [],
            usage=payload.get('usage') or {},
            success=payload.get('success', True),
            status=payload.get('status', 'ok'),
            tokens=payload.get('tokens', 0),
            details=payload.get('details') or {},
            partial_output=payload.get('partial_output', False),
            error_message=payload.get('error_message'),
        )
    return _record
