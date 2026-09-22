import json
import copy
import hashlib
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Optional

import anyio
from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import FileResponse, StreamingResponse
from app.database import (
    get_providers, find_user_by_api_key,
    increment_global_stats, increment_image_generation_stats, increment_user_usage, get_db,
    parse_model_id, add_request_record, add_request_log, update_request_log, get_enabled_preprocessor,
    get_enabled_image_generator, get_model_image_generation,
    get_model_responses_capability, set_model_responses_capability, update_model_responses_capability, update_model_responses_tool_types, set_model_responses_tools_capability,
)
from app.core.text import client_status_for_upstream_error, friendly_error_msg, error_detail_for_log, mask_key
from app.core.image_intent import latest_user_text
from app.protocols.responses_features import cross_provider_incompatible_reasons
from app.core.image_bridge import (
    GATEWAY_IMAGE_ASSET_MARKER,
    GATEWAY_IMAGE_RESULT_MARKER,
    IMAGE_BRIDGE_TOOL_NAME,
    configure_internal_image_bridge,
    should_inject_image_bridge,
    gateway_generated_image_asset_context,
    has_gateway_generated_image_history,
    image_call_arguments,
    image_call_arguments_from_exec,
    image_call_arguments_list_from_exec,
    inject_hosted_image_capability,
    is_gateway_image_display_followup,
    sanitize_gateway_image_display_followup,
    sanitize_gateway_generated_image_history,
)
from app.core.image_results import (
    StoredImageResult,
    find_image_result,
    generation_results_from_stored,
    image_result_directory,
    store_image_results,
)
from app.core.image_batch import image_invocation_cache
from app.core.image_orchestration import (
    run_image_bridge,
    events_to_message as _events_to_message,
    message_to_events as _message_to_events,
    image_generator_identity as _image_generator_identity,
    image_prompt_key as _image_prompt_key,
)
from app.core.model_capabilities import capabilities_for_client_entry, resolve_model_capabilities
from app.services.model_registry import registry_lookup
from app.core.tool_leak import repair_output as _repair_output_tool_leaks
from app.core.outcome import (
    apply_outcome_to_details,
    routing_details_from_policy,
    stats_counters_for_status,
    is_client_disconnect_error,
)
from app.core.output import (
    InternalOutputEvent,
    InternalOutputMessage,
    InternalToolCallOutput,
    aclose_async_iterator,
)
from app.core.types import InternalMessage, append_system_text, prepend_system_text, text_part, tool_call_part, tool_result_part

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
from app.core.streaming import stream_internal_output as _stream_internal_output, _attach_stream_performance
from app.protocols.ingress import (
    anthropic_messages_to_internal,
    chat_completions_to_internal,
    completions_to_internal,
    responses_to_internal,
    thinking_fields_from_body,
)
from app.core.policy import RouteTarget, apply_fallback_policy, has_missing_reasoning_content_for_tool_calls, prepare_request_policy
from app.adapters.anthropic import (
    anthropic_body_from_internal,
    anthropic_messages_completion_for_internal,
)
from app.adapters.openai import chat_kwargs_from_internal, chat_messages_from_internal
from app.adapters.output import response_to_internal_output
from app.adapters.anthropic_streaming import iter_anthropic_output_events
from app.adapters.openai_streaming import iter_openai_chat_output_events
from app.adapters.responses import (
    EmptyNativeResponsesError,
    iter_sse_frames,
    native_completed_output_item,
    native_response_has_output,
    native_sse_error_message,
    native_sse_payload_has_output,
    observed_response_tool_types,
    post_native_response,
    split_sse_frame,
    sse_payload,
    stream_native_response,
)
from app.adapters.imagegen import generate_images, image_results_bytes
from app.protocols.egress import (
    render_anthropic_message,
    render_chat_completion,
    render_completion,
    render_response, render_responses_image_generation, render_responses_image_generation_sse,
    render_responses_sse,
    chat_image_url_only_output,
    generated_image_client_output,
)
from app.services.lite_llm import create_chat_completion
from app.services.preprocessing import has_image_content, preprocess_messages
from app.services.routing_targets import candidate_targets, classify_upstream_error, is_same_target_retryable, provider_for_log, resolve_provider, upstream_status_code
from app.services.responses_capability import (
    RESPONSES_CAPABILITY_PROBE_MARKER,
    mark_model_responses_unknown,
    native_capability_for_request,
    native_error_is_explicitly_unsupported,
    native_response_target_supported,
    responses_capability_expiry,
)
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


def _chat_latest_user_text(internal) -> str:
    """Extract the latest user prompt from an internal chat request."""
    for message in reversed(internal.messages):
        if message.role != "user":
            continue
        parts = [part.text for part in message.parts if part.kind == "text" and part.text]
        text = "\n".join(parts).strip()
        if text:
            return text
    return ""


async def _async_event_stream(items):
    """Yield a pre-materialized list of output events as an async iterator.

    The streaming image bridge buffers the upstream stream and renders an
    out-of-band result, so it hands renderers a plain list; this adapts that
    list to the async-iterator interface the SSE renderers expect.
    """
    for item in items:
        yield item


def _resolved_image_generator(config: dict) -> dict:
    """Resolve a configured provider/model reference without changing chat routing."""
    resolved = dict(config or {})
    provider_model = str(resolved.get("provider_model") or "").strip()
    if provider_model:
        image_mid = parse_model_id(provider_model)
        image_provider = resolve_provider(image_mid.model_name, image_mid.provider_id)
        if image_provider:
            if not resolved.get("api_base"):
                resolved["api_base"] = image_provider.get("api_base") or ""
            if not resolved.get("api_key"):
                resolved["api_key"] = image_provider.get("api_key") or ""
            if not resolved.get("upstream_headers"):
                resolved["upstream_headers"] = image_provider.get("upstream_headers") or {}
            resolved["model"] = image_mid.model_name
            resolved["provider_id"] = image_provider.get("id") or image_mid.provider_id
    return resolved


async def _generate_with_configured_backend(
    prompt: str,
    options: dict | None = None,
    *,
    generator: dict | None = None,
):
    if generator is None:
        configured = get_enabled_image_generator()
        if not configured:
            raise HTTPException(status_code=503, detail="No image-generation backend is enabled")
        generator = _resolved_image_generator(configured)
    else:
        generator = dict(generator)
    generator.setdefault("max_retries", get_default("image_generation_max_retries", 2))
    generator.setdefault("retry_base_seconds", get_default("image_generation_retry_base_seconds", 1.0))
    generator.setdefault("max_retry_delay_seconds", get_default("image_generation_max_retry_delay_seconds", 30.0))
    generator.setdefault("result_max_bytes", get_default("image_generation_result_max_bytes", 25 * 1024 * 1024))
    generator.setdefault("allow_private_download_hosts", get_default("image_download_allow_private_hosts", False))
    opts = options or {}
    prompt = str(prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="image generation prompt is required")
    if len(prompt) > 8000:
        _app_log.warning("[responses image_generation] truncating image prompt from %d to 8000 chars", len(prompt))
        prompt = prompt[:8000]
    backend_provider, backend_model = _image_generator_identity(generator)
    _app_log.info(
        "[responses image_generation] backend_type=%s api_base=%s provider_model=%s provider_id=%s model=%s effective_provider=%s effective_model=%s params=%s",
        generator.get("backend_type") or "-", generator.get("api_base") or "-",
        generator.get("provider_model") or "-", generator.get("provider_id") or "-",
        generator.get("model") or "-", backend_provider or "-", backend_model or "-",
        sorted(key for key in ("n", "size", "quality", "background", "output_format") if opts.get(key) not in (None, "")),
    )
    try:
        results = await generate_images(
            generator,
            prompt=prompt,
            model=generator.get("model") or None,
            n=int(opts.get("n") or 1),
            size=opts.get("size"),
            quality=opts.get("quality"),
            background=opts.get("background"),
            output_format=opts.get("output_format"),
        )
    except HTTPException as exc:
        _attach_request_details(
            exc,
            request_kind="image_generation",
            responses_mode="image_generation",
            upstream_endpoint="images/generations",
            image_model=backend_model,
            image_count=0,
            image_bytes=0,
            attempted_provider=backend_provider,
        )
        raise
    except Exception as exc:
        _error_log.exception("[responses image_generation] failed: %s", exc)
        error = HTTPException(status_code=client_status_for_upstream_error(exc), detail=friendly_error_msg(exc))
        _attach_request_details(
            error,
            request_kind="image_generation",
            responses_mode="image_generation",
            upstream_endpoint="images/generations",
            image_model=backend_model,
            image_count=0,
            image_bytes=0,
            attempted_provider=backend_provider,
        )
        raise error from exc
    return results, generator


@dataclass
class _CachedImageInvocation:
    stored: list[StoredImageResult]
    generator: dict[str, Any]
    backend_attempts: int


@dataclass
class _ImageInvocationOutcome:
    call: InternalToolCallOutput
    arguments: dict[str, Any]
    stored: list[StoredImageResult]
    generator: dict[str, Any]
    backend_attempts: int
    duration_ms: int
    reused: bool = False
    error: Exception | None = None


def _image_batch_key(
    body: dict,
    *,
    username: str,
    api_key_value: str,
    generator: dict,
    invocations: list[tuple[InternalToolCallOutput, dict[str, Any]]],
) -> str:
    """Scope idempotency to one user, current task prompt, backend and ordered batch."""
    payload = {
        "principal": hashlib.sha256(f"{username}\0{api_key_value}".encode()).hexdigest(),
        "task": latest_user_text(body.get("input") or ""),
        "backend": _image_generator_identity(generator)[0],
        "model": _image_generator_identity(generator)[1],
        "artifact_dir": str(image_result_directory()),
        "invocations": [
            {
                "prompt_key": _image_prompt_key(arguments),
                "filename": str(arguments.get("filename") or ""),
                "n": int(arguments.get("n") or 1),
            }
            for _, arguments in invocations
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _image_invocation_cache_key(
    batch_key: str, arguments: dict[str, Any], occurrence: int,
) -> str:
    payload = {
        "batch": batch_key,
        "prompt_key": _image_prompt_key(arguments),
        "filename": str(arguments.get("filename") or ""),
        "n": int(arguments.get("n") or 1),
        "occurrence": int(occurrence),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


async def _generate_and_store_cached(
    *,
    prompt: str,
    arguments: dict[str, Any],
    generator: dict,
    cache_key: str,
) -> tuple[list, _CachedImageInvocation, bool]:
    ttl = get_default("image_generation_idempotency_ttl_seconds", 300)
    max_entries = get_default("image_generation_idempotency_max_entries", 64)
    claim = image_invocation_cache.claim(cache_key, ttl_seconds=ttl, max_entries=max_entries)
    if claim.owner:
        try:
            generated, resolved_generator = await _generate_with_configured_backend(
                prompt, arguments, generator=generator,
            )
            stored = await anyio.to_thread.run_sync(store_image_results, generated)
            cached = _CachedImageInvocation(
                stored=stored,
                generator=resolved_generator,
                backend_attempts=max((int(item.backend_attempts or 1) for item in generated), default=1),
            )
            image_invocation_cache.resolve(claim, cached)
            return generated, cached, False
        except BaseException as exc:
            image_invocation_cache.reject(claim, exc)
            raise

    # 等待者不能无限阻塞在 future.result() 上：图像批次上限可达 2400s，
    # 多个等待者会长期占用 anyio 线程池。超时后失效幂等键并报错，
    # 客户端重试可重新发起。
    wait_timeout = float(get_default("image_generation_batch_timeout_seconds", 2400)) + 60
    try:
        cached = await anyio.to_thread.run_sync(partial(claim.future.result, timeout=wait_timeout))
    except TimeoutError:
        image_invocation_cache.invalidate(cache_key)
        raise RuntimeError(f"image generation wait timed out after {int(wait_timeout)}s")
    if not all(item.path.is_file() for item in cached.stored):
        image_invocation_cache.invalidate(cache_key)
        return await _generate_and_store_cached(
            prompt=prompt,
            arguments=arguments,
            generator=generator,
            cache_key=cache_key,
        )
    generated = await anyio.to_thread.run_sync(
        partial(
            generation_results_from_stored,
            cached.stored,
            size=arguments.get("size"),
            quality=arguments.get("quality"),
            output_format=arguments.get("output_format"),
            background=arguments.get("background"),
        )
    )
    return generated, cached, True


async def _execute_image_invocations(
    body: dict,
    *,
    username: str,
    api_key_value: str,
    invocations: list[tuple[InternalToolCallOutput, dict[str, Any]]],
    progress=None,
) -> list[_ImageInvocationOutcome]:
    configured = get_enabled_image_generator()
    if not configured:
        raise HTTPException(status_code=503, detail="No image-generation backend is enabled")
    generator = _resolved_image_generator(configured)
    batch_key = _image_batch_key(
        body,
        username=username,
        api_key_value=api_key_value,
        generator=generator,
        invocations=invocations,
    )
    batch_id = batch_key[:12]
    concurrency = max(1, min(8, int(get_default("image_generation_batch_concurrency", 1))))
    timeout_seconds = max(1, int(get_default("image_generation_batch_timeout_seconds", 2400)))
    outcomes: list[_ImageInvocationOutcome | None] = [None] * len(invocations)
    semaphore = anyio.Semaphore(concurrency)

    async def run_one(index: int, call: InternalToolCallOutput, arguments: dict[str, Any]):
        started = time.monotonic()
        prompt = str(arguments.get("prompt") or latest_user_text(body.get("input") or "") or "")
        occurrence = sum(
            1
            for prior_index in range(index)
            if _image_prompt_key(invocations[prior_index][1]) == _image_prompt_key(arguments)
            and str(invocations[prior_index][1].get("filename") or "")
            == str(arguments.get("filename") or "")
        )
        key = _image_invocation_cache_key(batch_key, arguments, occurrence)
        _app_log.info(
            "[responses image_generation.item_start] batch=%s index=%d total=%d filename=%s prompt_chars=%d",
            batch_id, index + 1, len(invocations), arguments.get("filename") or "-", len(prompt),
        )
        try:
            async with semaphore:
                _generated, cached, reused = await _generate_and_store_cached(
                    prompt=prompt,
                    arguments=arguments,
                    generator=generator,
                    cache_key=key,
                )
            outcome = _ImageInvocationOutcome(
                call=call,
                arguments=arguments,
                stored=list(cached.stored),
                generator=dict(cached.generator),
                backend_attempts=cached.backend_attempts,
                duration_ms=round((time.monotonic() - started) * 1000),
                reused=reused,
            )
            _app_log.info(
                "[responses image_generation.item_done] batch=%s index=%d total=%d status=success attempts=%d reused=%s duration_ms=%d bytes=%d",
                batch_id, index + 1, len(invocations), outcome.backend_attempts,
                str(reused).lower(), outcome.duration_ms,
                _stored_image_bytes(outcome.stored),
            )
        except Exception as exc:
            outcome = _ImageInvocationOutcome(
                call=call,
                arguments=arguments,
                stored=[],
                generator=dict(generator),
                backend_attempts=0,
                duration_ms=round((time.monotonic() - started) * 1000),
                error=exc,
            )
            _app_log.warning(
                "[responses image_generation.item_done] batch=%s index=%d total=%d status=failed duration_ms=%d error=%s",
                batch_id, index + 1, len(invocations), outcome.duration_ms, error_detail_for_log(exc),
            )
        outcomes[index] = outcome
        if progress is not None:
            progress(batch_id, [item for item in outcomes if item is not None], len(invocations))

    with anyio.move_on_after(timeout_seconds) as cancel_scope:
        async with anyio.create_task_group() as task_group:
            for index, (call, arguments) in enumerate(invocations):
                task_group.start_soon(run_one, index, call, arguments)
    if cancel_scope.cancel_called:
        for index, item in enumerate(outcomes):
            if item is None:
                call, arguments = invocations[index]
                outcomes[index] = _ImageInvocationOutcome(
                    call=call,
                    arguments=arguments,
                    stored=[],
                    generator=dict(generator),
                    backend_attempts=0,
                    duration_ms=timeout_seconds * 1000,
                    error=TimeoutError(f"image batch deadline exceeded after {timeout_seconds}s"),
                )
        _app_log.warning(
            "[responses image_generation.batch_timeout] batch=%s timeout_s=%d",
            batch_id, timeout_seconds,
        )
    return [item for item in outcomes if item is not None]


async def _nonstream_output_events(output: InternalOutputMessage):
    """Replay a buffered planner response through the normal Responses renderer."""
    yield InternalOutputEvent(kind="message_start", role=output.role)
    if output.reasoning:
        yield InternalOutputEvent(kind="reasoning_delta", reasoning=output.reasoning)
    if output.text:
        yield InternalOutputEvent(kind="text_delta", text=output.text)
    for index, tool in enumerate(output.tool_calls):
        yield InternalOutputEvent(
            kind="tool_call_start", tool_index=index, tool_call_id=tool.id,
            call_id=tool.call_id, name=tool.name,
        )
        yield InternalOutputEvent(
            kind="tool_call_arguments_delta", tool_index=index,
            tool_call_id=tool.id, call_id=tool.call_id, name=tool.name,
            arguments_delta=tool.arguments, arguments=tool.arguments,
        )
        yield InternalOutputEvent(
            kind="tool_call_done", tool_index=index, tool_call_id=tool.id,
            call_id=tool.call_id, name=tool.name, arguments=tool.arguments,
        )
    if output.usage:
        yield InternalOutputEvent(kind="usage", usage=output.usage)
    yield InternalOutputEvent(kind="message_done", finish_reason=output.finish_reason)


def _image_result_url(request: Request, token: str) -> str:
    prefix = "v1/" if request.url.path.startswith("/v1/") else ""
    return f"{str(request.base_url).rstrip('/')}/{prefix}image-results/{token}"


def _safe_asset_filename(value: Any, index: int, stored: StoredImageResult) -> str:
    """Return a portable suggested filename with the stored image's real suffix."""
    raw = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem = raw.rsplit(".", 1)[0] if "." in raw else raw
    stem = "".join(char if char.isalnum() or char in "-_" else "-" for char in stem)
    stem = "-".join(part for part in stem.split("-") if part).strip("-_")[:80]
    return f"{stem or f'generated-asset-{index}'}{stored.path.suffix.lower()}"


def _unique_asset_filename(filename: str, used_filenames: set[str]) -> str:
    key = filename.casefold()
    if key not in used_filenames:
        used_filenames.add(key)
        return filename
    stem, separator, extension = filename.rpartition(".")
    if not separator:
        stem, extension = filename, ""
    suffix = 2
    while True:
        candidate = f"{stem}-{suffix}{separator}{extension}"
        if candidate.casefold() not in used_filenames:
            used_filenames.add(candidate.casefold())
            return candidate
        suffix += 1


def _stored_image_artifacts(
    request: Request,
    stored: list[StoredImageResult],
    *,
    arguments: dict[str, Any],
    start_index: int,
    used_filenames: set[str] | None = None,
) -> list[dict[str, str]]:
    used_filenames = used_filenames if used_filenames is not None else set()
    artifacts = []
    for offset, item in enumerate(stored):
        index = start_index + offset
        filename_value = arguments.get("filename") if len(stored) == 1 else ""
        filename = _unique_asset_filename(
            _safe_asset_filename(filename_value, index, item), used_filenames
        )
        artifacts.append({
            "filename": filename,
            "url": _image_result_url(request, item.token),
            "mime_type": item.mime_type,
            "prompt": str(arguments.get("prompt") or "")[:500],
        })
    return artifacts


def _rollback_image_bridge_artifacts(
    exc: Exception,
    *,
    stored: list[StoredImageResult],
    image_results: list,
    image_model: str,
) -> None:
    """Preserve completed artifacts when a later bridge/continuation step fails."""
    if not stored:
        return
    details = _request_details_from_exception(exc)
    _attach_request_details(
        exc,
        request_kind="image_generation",
        responses_mode="model_driven_image_generation_failed",
        upstream_endpoint="images/generations",
        image_model=image_model,
        image_count=len(image_results),
        image_bytes=image_results_bytes(image_results),
        image_artifact_count=len(stored),
        image_preserved_count=len(stored),
    )


def _stored_image_bytes(items) -> int:
    """日志统计用的字节数；文件被后台清理任务删除不能影响已成功的结果判定。"""
    total = 0
    for item in items or []:
        try:
            total += item.path.stat().st_size
        except OSError:
            pass
    return total


def _incomplete_tool_history_http_error(exc: Exception | None = None) -> HTTPException:
    error = HTTPException(
        status_code=400,
        detail="Responses request has tool calls without matching outputs; Chat compatibility cannot repair this.",
    )
    if exc is not None:
        error.__cause__ = exc
        existing = getattr(exc, "request_details", None)
        if isinstance(existing, dict):
            error.request_details = existing
    return error


def _native_empty_output_error(response: dict | None = None) -> EmptyNativeResponsesError:
    error = EmptyNativeResponsesError("native Responses completed without client-visible output")
    _attach_request_details(
        error,
        native_empty_output=True,
        native_failure_reason="empty_completed_response",
        error_trigger="connection_error",
    )
    return error


async def _native_responses_stream_with_accounting(events, *, username, api_key_value, model, provider_id, requested_model, policy, request_body, fallback_attempts=None, required_tool_types=None, remember_response_chain_key=None, conv_key=""):
    """Forward raw Responses SSE while recording the terminal response lifecycle."""
    buffer = b""
    response_body = None
    failed = False
    saw_output = False
    completed_output_item = False
    upstream_endpoint = "responses"
    terminal_error = None
    stream_started_at = time.monotonic()
    first_output_at = None
    client_disconnected = False
    try:
        async for frame in iter_sse_frames(events):
            payload = sse_payload(frame)
            has_output = native_sse_payload_has_output(payload)
            if has_output and first_output_at is None:
                first_output_at = time.monotonic()
            saw_output = saw_output or has_output
            sse_error = native_sse_error_message(payload)
            if sse_error:
                failed = True
                terminal_error = sse_error
            if payload and payload.get("type") in {"response.completed", "response.failed", "response.incomplete"}:
                response_body = payload.get("response")
                terminal_error = payload.get("error") or (response_body or {}).get("error")
                failed = payload.get("type") != "response.completed"
                if not failed and not saw_output and not native_response_has_output(response_body):
                    failed = True
                    terminal_error = "native Responses completed without client-visible output"
                if not failed:
                    capability = get_model_responses_capability(provider_id, model) or {}
                    update_model_responses_capability(
                        provider_id,
                        model,
                        status="supported",
                        streaming=True,
                        streaming_status="supported",
                        tool_types=capability.get("responses_tool_types") or [],
                        expires_at=responses_capability_expiry("supported"),
                    )
            if payload and payload.get("type") == "response.output_item.done":
                item = payload.get("item") or {}
                if native_completed_output_item(item):
                    completed_output_item = True
                observed = observed_response_tool_types({"output": [item]})
                if observed:
                    capability = get_model_responses_capability(provider_id, model) or {}
                    update_model_responses_tool_types(provider_id, model, list(set(capability.get("responses_tool_types") or []) | observed))
            yield frame
    except BaseException as exc:
        failed = True
        client_disconnected = is_client_disconnect_error(exc)
        raise
    finally:
        usage = (response_body or {}).get("usage") or {}
        tokens = usage.get("total_tokens") or 0
        # Codex closes the SSE after a completed tool/message item and then
        # continues with a follow-up request. That is a finished turn, not a
        # user cancel.
        closed_after_output = bool(client_disconnected and completed_output_item and not terminal_error)
        if closed_after_output:
            failed = False
        success = (bool(response_body) and not failed and saw_output) or closed_after_output
        response_id = (response_body or {}).get("id")
        if success and response_id and remember_response_chain_key and conv_key:
            remember_response_chain_key(response_id, conv_key)
        details = {
            **routing_details_from_policy(policy),
            **_thinking_fields_from_payload(request_body if isinstance(request_body, dict) else {}),
            "responses_mode": "native",
            "upstream_endpoint": upstream_endpoint,
            "stream": True,
            "fallback_attempts": fallback_attempts or [],
        }
        if len(fallback_attempts or []) > 1:
            details.update({"fallback_status": "used", "attempt_index": len(fallback_attempts) - 1})
        if closed_after_output:
            details.update({"client_disconnected": True, "stream_closed_after_output": True})
        elif client_disconnected:
            details.update({"status": "cancelled", "client_disconnected": True, "error_message": "client disconnected"})
        elif not saw_output and response_body and not client_disconnected:
            details.update({"native_empty_output": True, "native_failure_reason": "empty_completed_response"})
        elif failed and response_body:
            details.update({"status": "partial", "partial_output": True})
        elif saw_output and not response_body and not client_disconnected:
            details.update({"status": "partial", "partial_output": True, "native_failure_reason": "missing_completed_event"})
        details = apply_outcome_to_details(
            details,
            success=success,
            partial_output=bool(not closed_after_output and saw_output and (failed or not response_body)),
        )
        _attach_stream_performance(details, usage, first_output_at, stream_started_at)
        status = details.get("status", "ok" if success else "fail")
        if success:
            error_text = ""
        elif client_disconnected:
            error_text = "client disconnected"
        elif terminal_error:
            error_text = str(terminal_error)
        elif saw_output:
            error_text = "native Responses stream ended after client-visible output without a completed event"
        else:
            error_text = "native Responses stream did not complete"

        def _accounting():
            # 记账是每请求最重的同步写入（全量 request_body 的 json.dumps +
            # SQLite 写），必须离开事件循环，与 streaming.py 的 P1 优化同口径。
            _log_request(username, api_key_value, model, provider_id, "responses", success, tokens, requested_model, details=details)
            _record_request_log(
                endpoint="responses",
                username=username,
                api_key_value=api_key_value,
                requested_model=requested_model,
                final_model=model,
                final_provider=provider_id,
                request_body=request_body,
                response_body=response_body,
                success=success,
                status=status,
                tokens=tokens,
                usage=usage,
                details=details,
                error_message=error_text,
                stream=True,
                request_started_at=stream_started_at,
                generation_started_at=first_output_at or stream_started_at,
            )
            _record_success_metrics(username, api_key_value, tokens, status)

        try:
            await anyio.to_thread.run_sync(_accounting)
        except Exception as accounting_exc:
            _app_log.warning("[responses native accounting] failed: %s", accounting_exc)


def _native_downgrade_details(exc: Exception, attempts: list[dict] | None = None) -> dict:
    """Describe a failed native attempt that was completed through compatibility."""
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    details = {
        "responses_mode": "compatibility_downgrade",
        "native_attempted": True,
        "native_failure_endpoint": "responses",
        "native_failure_reason": classify_upstream_error(exc),
        "native_failure_message": error_detail_for_log(exc),
    }
    request_details = getattr(exc, "request_details", None)
    if isinstance(request_details, dict):
        for key in ("native_empty_output", "native_failure_reason", "error_trigger"):
            if key in request_details:
                details[key] = request_details[key]
    if status is not None:
        details["native_failure_status"] = status
    if attempts:
        details["native_attempts"] = attempts
    return details


async def _wait_for_native_response_output(events) -> bytes:
    """Buffer native SSE until usable output, rejecting empty completion before fallback."""
    buffered = b""
    saw_output = False
    while True:
        chunk = await events.__anext__()
        buffered += chunk
        while (split := split_sse_frame(buffered)) is not None:
            frame, rest = split
            payload = sse_payload(frame)
            saw_output = saw_output or native_sse_payload_has_output(payload)
            if saw_output:
                return buffered
            if payload:
                sse_error = native_sse_error_message(payload)
                if sse_error:
                    error = RuntimeError(sse_error)
                    _attach_request_details(error, native_failure_reason="sse_error")
                    raise error
                event_type = str(payload.get("type") or "")
                if event_type in {"response.failed", "response.incomplete"}:
                    error = RuntimeError("native Responses stream ended unsuccessfully")
                    _attach_request_details(error, native_failure_reason=event_type)
                    raise error
                if event_type == "response.completed":
                    response = payload.get("response")
                    if not saw_output and not native_response_has_output(response):
                        raise _native_empty_output_error(response)
                    return buffered
            buffered = rest


async def _native_response_with_fallbacks(internal, *, stream: bool, required_tool_types: set[str], stateful_markers: list[str] | None = None):
    """Retry only native-capable fallback targets before any client output."""
    primary = RouteTarget(model=internal.target_model, provider_id=internal.provider_id)
    primary = RouteTarget(model=primary.model, provider_id=_fallback_provider_id_for_target(primary))
    targets = [primary]
    has_tools = bool(internal.tools)
    stateful_markers = list(stateful_markers or [])
    # Capability mismatch is local routing information, rather than an upstream
    # failure.  Still consult the configured fallback chain: otherwise an
    # advanced Responses request is rejected before it can reach a compatible
    # fallback provider.  An empty trigger intentionally ignores error-trigger
    # gates because no upstream request has been made yet.
    primary_provider, _primary_provider_id = native_response_target_supported(
        primary, stream=stream, required_tool_types=required_tool_types, is_primary=True, has_tools=has_tools,
    )
    if primary_provider is None:
        capability_fallback = apply_fallback_policy(
            _fallback_provider_id_for_target(primary), primary.model, trigger=""
        )
        if capability_fallback.matched:
            targets = candidate_targets(primary, capability_fallback.chain)
    last_exc = None
    index = 0
    attempts = []
    while index < len(targets):
        target = targets[index]
        provider, provider_id = native_response_target_supported(target, stream=stream, required_tool_types=required_tool_types, is_primary=index == 0, has_tools=has_tools)
        if provider is None:
            attempts.append({"index": index, "stage": "primary" if index == 0 else "fallback", "target": target.model, "provider_id": target.provider_id, "status": "skipped", "reason": "capability_mismatch"})
            index += 1
            continue
        if index > 0:
            native_body = (internal.metadata.get("responses_native") or {}).get("request_body") or internal.raw_body
            incompatible = cross_provider_incompatible_reasons(native_body)
            if incompatible:
                attempts.append({
                    "index": index,
                    "stage": "fallback",
                    "target": target.model,
                    "provider_id": target.provider_id,
                    "status": "skipped",
                    "reason": "native_dialect_incompatible",
                    "incompatible": incompatible,
                })
                index += 1
                continue
        attempt = copy.deepcopy(internal)
        attempt.target_model, attempt.provider_id = target.model, provider_id
        try:
            if stream:
                # Do not yield until the first chunk: this preserves the existing
                # stream fallback invariant.
                events = stream_native_response(provider, attempt)
                try:
                    first = await _wait_for_native_response_output(events)
                except BaseException:
                    # 首字节判定失败时生成器挂起在 yield 处，内部 async with
                    # 持有上游连接；不显式关闭就只能等 GC（S5 不变量）。
                    await aclose_async_iterator(events)
                    raise
                async def prefixed():
                    yield first
                    async for chunk in events:
                        yield chunk
                attempts.append({"index": index, "stage": "primary" if index == 0 else "fallback", "target": target.model, "provider_id": provider_id, "status": "success"})
                return prefixed(), target, provider_id, attempts
            response = await post_native_response(provider, attempt)
            if not native_response_has_output(response):
                raise _native_empty_output_error(response)
            attempts.append({"index": index, "stage": "primary" if index == 0 else "fallback", "target": target.model, "provider_id": provider_id, "status": "success"})
            set_model_responses_capability(
                provider_id, target.model, status="supported",
                streaming=False, streaming_status="unknown",
                error=RESPONSES_CAPABILITY_PROBE_MARKER,
                expires_at=responses_capability_expiry("supported"),
            )
            if has_tools:
                # 带 tools 的原生成功是工具形态的正向证据，解除既有负向记录。
                set_model_responses_tools_capability(
                    provider_id, target.model, status="supported",
                    expires_at=responses_capability_expiry("supported"),
                )
            return response, target, provider_id, attempts
        except Exception as exc:
            last_exc = exc
            is_empty_native = bool(getattr(exc, "native_empty_output", False))
            is_protocol_unsupported = native_error_is_explicitly_unsupported(exc)
            error_status = upstream_status_code(exc)
            tool_shape_rejection = bool(internal.tools) and error_status is not None and 400 <= error_status <= 499
            if is_protocol_unsupported:
                set_model_responses_capability(provider_id, target.model, status="unsupported", expires_at=responses_capability_expiry("unsupported"), error=error_detail_for_log(exc))
            elif tool_shape_rejection:
                # 含 tools 请求收到权威 4xx（如 thinking 模式拒绝强制 tool_choice、上游不支持
                # custom 工具）只证明“工具形态”不被原生支持：记工具形态级负向能力，
                # 后续带工具请求直接走 Chat（兼容路径），文本/流式继续原生。
                # 不能回到旧行为“整体降 unknown”：会把文本拖离原生（5 分钟摆动）；
                # 也不能简单“保持能力不动”：Codex 类 client-owned 工具的降级被策略阻断，
                # 保持能力会让每个请求都撞原生硬失败（审查 15 轮回归教训）。
                set_model_responses_tools_capability(
                    provider_id, target.model, status="unsupported",
                    expires_at=responses_capability_expiry("unsupported"),
                    error=error_detail_for_log(exc),
                )
                _app_log.info(
                    "[responses capability] tool-shape 4xx (%s) on provider=%s model=%s; recording tool-shape negative, keeping model text capability as-is",
                    error_status, provider_id, target.model,
                )
            else:
                # 真空白响应（零 output item，含 EmptyNativeResponsesError）无“请求形态”归因，
                # 是上游原生实现不可用的直接证据，仍记 transient 300s 自保护
                # （实现决定，缘由审查 14 轮 #3 提出）。
                mark_model_responses_unknown(provider_id, target.model, exc)
            attempts.append({"index": index, "stage": "primary" if index == 0 else "fallback", "target": target.model, "provider_id": provider_id, "status": "failed", "trigger": classify_upstream_error(exc), "error": error_detail_for_log(exc)})
            if index == 0:
                decision = apply_fallback_policy(provider_id, target.model, classify_upstream_error(exc))
                if is_empty_native and not decision.matched:
                    decision = apply_fallback_policy(provider_id, target.model, "")
                if decision.matched:
                    # A failed primary native request may have created provider-
                    # side response/tool state.  Do not migrate a stateful
                    # Responses turn to another provider.  This guard is
                    # intentionally inside the exception path: capability
                    # mismatch (where no upstream request was sent) remains
                    # eligible for the configured fallback chain.
                    provider_bound_markers = [
                        marker for marker in stateful_markers
                        if marker == "previous_response_id"
                    ]
                    if provider_bound_markers:
                        blocked_targets = candidate_targets(primary, decision.chain)[1:]
                        attempts.extend({
                            "index": blocked_index,
                            "stage": "fallback",
                            "target": blocked.model,
                            "provider_id": blocked.provider_id,
                            "status": "skipped",
                            "reason": "stateful_codex_tools",
                        } for blocked_index, blocked in enumerate(blocked_targets, start=1))
                        _attach_request_details(
                            exc,
                            fallback_attempts=attempts,
                            fallback_status="skipped",
                            fallback_reason="stateful_codex_tools",
                            fallback_safety_decision="blocked_cross_provider",
                            responses_stateful=True,
                            responses_state_markers=stateful_markers,
                            responses_provider_bound_markers=provider_bound_markers,
                            stateful_fallback_blocked=True,
                            error_trigger=classify_upstream_error(exc),
                            error_stage="primary",
                        )
                        raise
                    targets = candidate_targets(primary, decision.chain)
            index += 1
    error = last_exc or RuntimeError("No native Responses fallback target available")
    if last_exc is None:
        error.native_capability_unavailable = True
        error.required_tool_types = sorted(required_tool_types)
    _attach_request_details(error, fallback_attempts=attempts, fallback_status="exhausted", responses_stateful=bool(stateful_markers), responses_state_markers=stateful_markers)
    raise error


def _attach_request_details(exc: Exception, **details) -> Exception:
    """Attach policy/fallback metadata to an exception.

    契约：任何从 fallback 链或上游适配器向上抛出的异常，在到达
    ``core.streaming.stream_internal_output`` / ``core.outcome.is_client_disconnect_error``
    之前都必须经过本函数（至少在 exhausted 路径上打标）。后者用
    ``getattr(exc, "request_details", None) is dict`` 作为“这是上游错误、
    不是客户端断开”的判据；新增不经过本函数的上游异常路径会破坏该判据，
    使 httpx/httpcore 的 reset/remote-protocol 文本被误判为客户端取消。
    """
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
    details.setdefault("error_message", error_detail_for_log(exc))
    return details


def _log_upstream_http_exception_failure(
    endpoint: str,
    exc: HTTPException,
    *,
    username: str,
    api_key_value: str,
    requested_model: str,
    model: str,
    provider_id: str,
    body: dict,
) -> None:
    """HTTPException 透传路径也要留下失败日志与统计。

    状态码语义保留给客户端（429/502 等），但管理端的失败率/请求日志
    不能因为透传而丢失记录。
    """
    status = "rejected" if 400 <= int(exc.status_code or 500) < 500 else "fail"
    details = _request_details_from_exception(
        exc,
        stream=False,
        attempted_model=model or requested_model,
        attempted_provider=provider_id or "",
    )
    details["status"] = status
    counters = stats_counters_for_status(status)
    _log_request(username, api_key_value, model or requested_model, provider_id or "", endpoint, False, 0, requested_model, details=details)
    _record_request_log(
        endpoint=endpoint,
        username=username, api_key_value=api_key_value, requested_model=requested_model,
        final_model=model or requested_model, final_provider=provider_id or "",
        request_body=body, response_body=None,
        success=False, status=status, tokens=0, details=details,
        error_message=error_detail_for_log(exc),
    )
    increment_global_stats(False, degraded=counters.degraded, rejected=counters.rejected, cancelled=counters.cancelled)
    if username != "legacy":
        increment_user_usage(username, api_key_value, False, 0)


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
        record["error_message"] = error_detail_for_log(error)
    return record


def _append_fallback_attempt(details: dict, attempt: dict) -> None:
    attempts = details.setdefault("fallback_attempts", [])
    if isinstance(attempts, list):
        attempts.append(attempt)


def _maybe_repair_tool_leak(output, internal, *, endpoint: str, provider_id: str) -> None:
    """非流式回程的泄漏工具调用抢救（协议无关，见 core/tool_leak 模块注释）。

    上游推理框架（llama.cpp b10884 实测 ~5%）会概率性地把模板原生的
    XML 工具调用当普通文本下发；harness 收到后误以为回合结束。这里在
    IR 层做保守校验后还原为结构化工具调用。任何异常保持原样透传，
    绝不影响主链路。流式回程只检测不修改（见 stream_internal_output）。
    """
    if not bool(get_default("repair_tool_leaks", True)):
        return
    if getattr(output, "tool_calls", None):
        return  # 上游已给出结构化调用，无泄漏可救
    try:
        count = _repair_output_tool_leaks(output, internal.tools)
    except Exception as exc:
        _app_log.warning("[tool_leak] repair error endpoint=%s: %s", endpoint, str(exc)[:160])
        return
    if count:
        details = getattr(output, "request_details", None)
        if isinstance(details, dict):
            details["tool_leak_repaired"] = count
        _tool_log.warning(
            "[tool_leak.repaired] endpoint=%s provider=%s model=%s blocks=%d",
            endpoint, provider_id or "-", internal.target_model, count,
        )


def _output_request_details(output) -> dict:
    details: dict = {}
    dedicated = getattr(output, "request_details", None)
    if isinstance(dedicated, dict):
        details.update(dedicated)
    raw = getattr(output, "raw", None)
    if isinstance(raw, dict) and isinstance(raw.get("request_details"), dict):
        details.update(raw["request_details"])
    return details


def _merge_request_details(*parts: dict | None) -> dict:
    merged: dict = {}
    for part in parts:
        if isinstance(part, dict) and part:
            merged.update(part)
    return merged


def _merge_bridge_request_details(existing: dict | None, latest: dict | None) -> dict:
    """Merge metadata across planner and continuation upstream calls.

    A continuation is a second upstream request, so its attempt list must not
    erase the planner's primary/fallback history.  Once any stage used a
    fallback, the overall image-bridge request remains degraded.
    """
    merged = _merge_request_details(existing, latest)
    old_attempts = (existing or {}).get("fallback_attempts") if isinstance(existing, dict) else None
    new_attempts = (latest or {}).get("fallback_attempts") if isinstance(latest, dict) else None
    if isinstance(old_attempts, list) or isinstance(new_attempts, list):
        merged["fallback_attempts"] = [
            *(old_attempts if isinstance(old_attempts, list) else []),
            *(new_attempts if isinstance(new_attempts, list) else []),
        ]
    if str((existing or {}).get("fallback_status") or "") == "used" or str((latest or {}).get("fallback_status") or "") == "used":
        merged["fallback_status"] = "used"
    return merged


def _attach_output_request_details(output, **fields) -> None:
    # 主存储：专用字段（LiteLLM 路径的 raw 是非 dict 对象，不能依赖）。
    dedicated = getattr(output, "request_details", None)
    if isinstance(dedicated, dict):
        dedicated.update(fields)
    # 兼容存储：raw 为 dict 的适配器路径继续同步，旧读取方不受影响。
    if getattr(output, "raw", None) is None:
        output.raw = {}
    if isinstance(output.raw, dict):
        details = output.raw.setdefault("request_details", {})
        if not isinstance(details, dict):
            details = {}
            output.raw["request_details"] = details
        details.update(fields)


def _finalize_success_details(output=None, *, policy=None, extra: dict | None = None) -> dict:
    details = _merge_request_details(
        _output_request_details(output) if output is not None else {},
        routing_details_from_policy(policy) if policy is not None else {},
        extra,
    )
    details.update(_thinking_fields_from_payload(details))
    return apply_outcome_to_details(details, success=True, partial_output=False)


def _record_success_metrics(username: str, api_key_value: str, tokens: int, status: str) -> None:
    counters = stats_counters_for_status(status)
    increment_global_stats(
        counters.hard_success,
        degraded=counters.degraded,
        rejected=counters.rejected,
        cancelled=counters.cancelled,
    )
    if username != "legacy":
        increment_user_usage(username, api_key_value, counters.hard_success, tokens)


def _log_rejected_request(
    *,
    status_code: int,
    detail: str,
    endpoint: str = "",
    username: str = "",
    api_key_value: str = "",
    requested_model: str = "",
    model: str = "",
    provider: str = "",
) -> None:
    """Persist auth/allow-list rejections so they appear in stats and request logs."""
    req_model = requested_model or model or ""
    final_model = model or requested_model or "-"
    details = apply_outcome_to_details(
        {
            "status": "rejected",
            "http_status": status_code,
            "error_message": detail,
            "reject_reason": detail,
        },
        success=False,
    )
    details["status"] = "rejected"
    user_label = username or "anonymous"
    try:
        _log_request(
            user_label,
            api_key_value,
            final_model,
            provider,
            endpoint or "unknown",
            False,
            0,
            req_model,
            details=details,
        )
        _record_request_log(
            endpoint=endpoint or "unknown",
            username=user_label,
            api_key_value=api_key_value,
            requested_model=req_model,
            final_model=final_model,
            final_provider=provider,
            request_body=None,
            response_body={"error": {"message": detail, "type": "rejected", "code": status_code}},
            success=False,
            status="rejected",
            tokens=0,
            details=details,
            error_message=detail,
        )
        increment_global_stats(False, rejected=True)
        if username and username != "legacy" and api_key_value:
            increment_user_usage(username, api_key_value, False, 0)
    except Exception as exc:
        _app_log.warning("Failed to record rejected request: %s", exc)


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


def _upstream_endpoint_for_provider(provider_info: dict | None, *, native_responses: bool = False) -> str:
    """Name the actual protocol endpoint used for one upstream attempt."""
    if native_responses:
        return "responses"
    if provider_info and provider_info.get("provider_type") == "anthropic":
        return "messages"
    return "chat_completions"


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
    started_at = time.monotonic()
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
            ),
            # attempt_timeout 到期取消时必须立即返回，不能等线程跑完；否则
            # 上游挂死时"主动超时→fallback"要等到 litellm 自身超时才触发。
            # 被 abandon 的线程由 litellm 自己的 request_timeout 兜底回收；
            # 该值受 provider 配置限幅（_clamp_int 与 Pydantic le=3600，上限 1h），
            # 因此被弃线程的占用时长有硬上界，不会无限期占住 anyio 线程池。
            abandon_on_cancel=True,
        )
        output = response_to_internal_output(response)
    _attach_output_request_details(
        output,
        upstream_endpoint=_upstream_endpoint_for_provider(provider_info),
        duration_ms=max(0, round((time.monotonic() - started_at) * 1000)),
    )
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


def _target_requires_thinking_reasoning(target: RouteTarget) -> bool:
    provider_info = resolve_provider(target.model, target.provider_id)
    provider_options = (provider_info or {}).get("provider_options") or {}
    if isinstance(provider_options, str):
        try:
            provider_options = json.loads(provider_options)
        except (TypeError, ValueError):
            provider_options = {}
    if not isinstance(provider_options, dict):
        return False
    return provider_options.get("thinking") == "enabled"


async def _internal_for_target_attempt(internal, target: RouteTarget, *, is_fallback: bool):
    attempt = internal
    copied = False

    def _copy_if_needed():
        nonlocal attempt, copied
        if not copied:
            attempt = copy.deepcopy(internal)
            copied = True
        return attempt

    if is_fallback and has_image_content(internal.messages):
        if not _target_supports_native_vision(target) and _target_uses_preprocessor(target):
            attempt = _copy_if_needed()
            await _policy_preprocess_request(attempt, target.model, target.provider_id, target.model)

    if _target_requires_thinking_reasoning(target) and has_missing_reasoning_content_for_tool_calls(internal.messages):
        attempt = _copy_if_needed()
        attempt.extra["disable_thinking_for_missing_reasoning"] = True
        _app_log.warning(
            "[fallback] disabling thinking for target=%s provider=%s because historical tool calls lack reasoning_content",
            target.model,
            target.provider_id or "-",
        )

    return attempt


def _fallback_provider_id_for_target(target: RouteTarget) -> str:
    provider_info = resolve_provider(target.model, target.provider_id)
    return provider_for_log(provider_info, target.provider_id)


def _same_target_retry_limit() -> int:
    """Extra in-place attempts for a transient first-byte failure on one target."""
    try:
        return max(0, min(3, int(get_default("same_target_retry_limit", 1))))
    except (TypeError, ValueError):
        return 1


def _lookup_fallback_budget(provider_id: str, model: str):
    """Return matched fallback decision for proactive attempt timeout (ignore trigger filter)."""
    return apply_fallback_policy(provider_id, model, trigger="")


def _attempt_timeout_error(seconds: int, target: RouteTarget, provider_id: str) -> TimeoutError:
    exc = TimeoutError(f"fallback attempt timeout after {int(seconds)}s")
    exc.attempted_model = target.model
    exc.attempted_provider = provider_id or target.provider_id or ""
    return exc


async def _await_with_attempt_timeout(awaitable, *, timeout_s: int | None, target: RouteTarget, provider_id: str):
    if not timeout_s or timeout_s <= 0:
        return await awaitable
    try:
        with anyio.fail_after(timeout_s):
            return await awaitable
    except TimeoutError as exc:
        _app_log.warning(
            "[fallback.attempt_timeout] target=%s provider=%s timeout_s=%d",
            target.model,
            provider_id or "-",
            timeout_s,
        )
        raise _attempt_timeout_error(timeout_s, target, provider_id) from exc


async def _iter_events_with_first_output_timeout(events, *, timeout_s: int | None, target: RouteTarget, provider_id: str):
    """Yield stream events; enforce timeout only until first client-visible output."""
    agen = events if hasattr(events, "__anext__") else events.__aiter__()
    try:
        if not timeout_s or timeout_s <= 0:
            async for event in agen:
                yield event
            return

        emitted = False
        deadline = time.monotonic() + float(timeout_s)
        while True:
            try:
                if not emitted:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _attempt_timeout_error(timeout_s, target, provider_id)
                    with anyio.fail_after(remaining):
                        event = await agen.__anext__()
                else:
                    event = await agen.__anext__()
            except StopAsyncIteration:
                break
            except TimeoutError as exc:
                if emitted:
                    raise
                _app_log.warning(
                    "[fallback.attempt_timeout] stage=stream_first_output target=%s provider=%s timeout_s=%d",
                    target.model,
                    provider_id or "-",
                    timeout_s,
                )
                if isinstance(exc, TimeoutError) and "fallback attempt timeout" in str(exc):
                    raise
                raise _attempt_timeout_error(timeout_s, target, provider_id) from exc
            if _is_client_visible_stream_event(event):
                emitted = True
            yield event
    finally:
        # 超时、异常与上层提前丢弃都必须关闭上游，否则 HTTP 连接只能等 GC。
        await aclose_async_iterator(agen)


async def _call_nonstream_with_fallbacks(policy, internal, *, temperature, max_tokens, log_label: str):
    original_model = internal.target_model
    original_provider = internal.provider_id
    last_exc = None
    primary = RouteTarget(model=original_model, provider_id=original_provider)
    primary = RouteTarget(model=primary.model, provider_id=_fallback_provider_id_for_target(primary))
    targets = [primary]
    fallback_attempts = []
    same_target_retries: dict[tuple[str, str], int] = {}
    budget = _lookup_fallback_budget(primary.provider_id, primary.model)
    attempt_timeout = budget.attempt_timeout if budget.matched else None
    if attempt_timeout:
        _app_log.info(
            "[%s fallback.budget] policy_id=%s policy='%s' attempt_timeout=%ds primary=%s provider=%s",
            log_label,
            budget.policy_id,
            budget.policy_name,
            attempt_timeout,
            primary.model,
            primary.provider_id or "-",
        )
    _app_log.debug(
        "[%s pipeline] primary_call target=%s provider=%s",
        log_label,
        primary.model,
        primary.provider_id or "-",
    )
    # The primary target may be retried in place; fallback targets keep the
    # dedicated loop below so stage/index accounting stays unchanged.
    retry_primary = True
    while retry_primary:
        retry_primary = False
        target = primary
        attempt_internal = await _internal_for_target_attempt(
            internal, target, is_fallback=False,
        )
        attempt_internal.target_model = target.model
        attempt_internal.provider_id = target.provider_id
        fallback_provider_id = _fallback_provider_id_for_target(target)
        try:
            output, provider_info, adapter_provider_id = await _await_with_attempt_timeout(
                _call_nonstream_target(
                    target,
                    attempt_internal,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    log_label=log_label,
                    stage="primary",
                ),
                timeout_s=attempt_timeout,
                target=target,
                provider_id=fallback_provider_id,
            )
            fallback_attempts.append(_fallback_attempt_record(
                index=0,
                stage="primary",
                target=target,
                provider_id=fallback_provider_id,
                status="success",
            ))
            _attach_output_request_details(
                output,
                fallback_status="unused",
                attempt_index=0,
                fallback_attempts=fallback_attempts,
                upstream_endpoint=_upstream_endpoint_for_provider(provider_info),
            )
            return output, provider_info, adapter_provider_id
        except Exception as exc:
            last_exc = exc
            trigger = classify_upstream_error(exc)
            fallback_attempts.append(_fallback_attempt_record(
                index=0,
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
                error_detail_for_log(exc),
            )
            retry_key = (target.provider_id or "", target.model)
            retry_limit = _same_target_retry_limit()
            if is_same_target_retryable(exc, trigger) and same_target_retries.get(retry_key, 0) < retry_limit:
                same_target_retries[retry_key] = same_target_retries.get(retry_key, 0) + 1
                _app_log.warning(
                    "[%s upstream.retry] target=%s provider=%s trigger=%s retry=%d/%d",
                    log_label,
                    target.model,
                    fallback_provider_id or "-",
                    trigger,
                    same_target_retries[retry_key],
                    retry_limit,
                )
                retry_primary = True
                continue
            decision = apply_fallback_policy(fallback_provider_id, target.model, trigger)
            # Proactive attempt_timeout should still use the matched policy chain even if
            # the "timeout" trigger checkbox is off (the budget itself implies timeout switching).
            if not decision.matched and budget.matched and trigger == "timeout":
                decision = budget
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
                    error_message=error_detail_for_log(exc),
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
                "[%s fallback.decision] matched=True policy_id=%s policy='%s' source=%s provider=%s trigger=%s chain=%d attempt_timeout=%ds",
                log_label,
                decision.policy_id,
                decision.policy_name,
                target.model,
                fallback_provider_id or "-",
                trigger,
                len(decision.chain),
                decision.attempt_timeout,
            )
            targets = candidate_targets(primary, decision.chain)
            if decision.matched:
                attempt_timeout = decision.attempt_timeout
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
                error_detail_for_log(last_exc) if last_exc else "",
            )
            output, provider_info, adapter_provider_id = await _await_with_attempt_timeout(
                _call_nonstream_target(
                    target,
                    attempt_internal,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    log_label=log_label,
                    stage="fallback",
                ),
                timeout_s=attempt_timeout,
                target=target,
                provider_id=target.provider_id or "",
            )
            fallback_attempts.append(_fallback_attempt_record(
                index=index,
                stage="fallback",
                target=target,
                provider_id=target.provider_id or "",
                status="success",
            ))
            _attach_output_request_details(
                output,
                fallback_status="used",
                attempt_index=index,
                fallback_attempts=fallback_attempts,
                upstream_endpoint=_upstream_endpoint_for_provider(provider_info),
            )
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
                error_message=error_detail_for_log(exc),
            )
            _app_log.warning(
                "[%s fallback.attempt.failed] index=%d target=%s provider=%s trigger=%s error=%s",
                log_label,
                index,
                target.model,
                target.provider_id or "-",
                trigger,
                error_detail_for_log(exc),
            )
    _app_log.error(
        "[%s fallback.exhausted] primary=%s provider=%s candidates=%d error=%s",
        log_label,
        primary.model,
        primary.provider_id or "-",
        max(len(targets) - 1, 0),
        error_detail_for_log(last_exc) if last_exc else "no target available",
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


def _placeholder_only_stream_text(text: str) -> bool:
    """Return whether text is only an upstream placeholder, not useful output."""
    compact = "".join(str(text or "").split())
    return bool(compact) and all(char in ".…·。!?！？,，;；:：-_—~～" for char in compact)


def _pending_stream_has_usable_payload(events: list[InternalOutputEvent]) -> bool:
    text = "".join(event.text or "" for event in events if event.kind == "text_delta")
    if text and not _placeholder_only_stream_text(text):
        return True
    return any(
        _stream_event_has_payload(event) and event.kind != "text_delta"
        for event in events
    )


def _empty_stream_error(target: RouteTarget, provider_id: str, *, placeholder_only: bool = False) -> RuntimeError:
    message = (
        "upstream stream ended with placeholder-only response"
        if placeholder_only
        else "upstream stream ended without response output"
    )
    exc = RuntimeError(message)
    exc.attempted_model = target.model
    exc.attempted_provider = provider_id or target.provider_id or ""
    exc.placeholder_only_response = placeholder_only
    exc.empty_stream_response = not placeholder_only
    exc.confirmed_upstream = True
    return exc


async def _stream_events_with_fallbacks(internal, *, temperature, max_tokens, log_label: str, strip_thinking=True):
    primary = RouteTarget(model=internal.target_model, provider_id=internal.provider_id)
    primary = RouteTarget(model=primary.model, provider_id=_fallback_provider_id_for_target(primary))
    targets = [primary]
    last_exc = None
    index = 0
    fallback_attempts = []
    degenerate_retries: dict[tuple[str, str], int] = {}
    same_target_retries: dict[tuple[str, str], int] = {}
    budget = _lookup_fallback_budget(primary.provider_id, primary.model)
    attempt_timeout = budget.attempt_timeout if budget.matched else None
    if attempt_timeout:
        _app_log.info(
            "[%s fallback.budget] policy_id=%s policy='%s' attempt_timeout=%ds primary=%s provider=%s",
            log_label,
            budget.policy_id,
            budget.policy_name,
            attempt_timeout,
            primary.model,
            primary.provider_id or "-",
        )

    while index < len(targets):
        target = targets[index]
        stage = "primary" if index == 0 else "fallback"
        attempt_internal = await _internal_for_target_attempt(internal, target, is_fallback=index > 0)
        attempt_internal.target_model = target.model
        attempt_internal.provider_id = target.provider_id
        fallback_provider_id = _fallback_provider_id_for_target(target)
        emitted = False
        pending_events = []
        terminal_event = None
        events = None
        timed_events = None
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
                "attempt_timeout": attempt_timeout,
                "fallback_attempts": fallback_attempts + [_fallback_attempt_record(
                    index=index,
                    stage=stage,
                    target=target,
                    provider_id=adapter_provider_id or "",
                    status="started",
                )],
                "upstream_endpoint": _upstream_endpoint_for_provider(provider_info),
            })
            timed_events = _iter_events_with_first_output_timeout(
                events,
                timeout_s=attempt_timeout,
                target=target,
                provider_id=adapter_provider_id or fallback_provider_id or "",
            )
            async for event in timed_events:
                if event.kind == "message_done":
                    # Protocol renderers stop consuming as soon as they see
                    # message_done. Hold it until the attempt is recorded as
                    # successful so final fallback metadata reaches request
                    # accounting before the client stream terminates.
                    terminal_event = event
                    continue
                if (
                    not emitted
                    and event.kind == "text_delta"
                    and _placeholder_only_stream_text(
                        "".join(
                            pending.text or ""
                            for pending in pending_events
                            if pending.kind == "text_delta"
                        ) + (event.text or "")
                    )
                ):
                    # Some compatible upstreams occasionally finish a request
                    # with only "...".  Hold punctuation-only prefixes until
                    # substantive output arrives so a degenerate completion
                    # cannot prematurely terminate a Responses tool turn.
                    pending_events.append(event)
                elif _is_client_visible_stream_event(event):
                    if not emitted:
                        for pending in pending_events:
                            if not (
                                pending.kind == "text_delta"
                                and _placeholder_only_stream_text(pending.text or "")
                            ):
                                yield pending
                        pending_events = []
                    emitted = True
                    yield event
                elif emitted:
                    yield event
                else:
                    pending_events.append(event)
            if not emitted and not _pending_stream_has_usable_payload(pending_events):
                placeholder_only = any(
                    event.kind == "text_delta" and bool(event.text)
                    for event in pending_events
                )
                raise _empty_stream_error(
                    target,
                    adapter_provider_id or fallback_provider_id or "",
                    placeholder_only=placeholder_only,
                )
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
                "upstream_endpoint": _upstream_endpoint_for_provider(provider_info),
            })
            if terminal_event is not None:
                yield terminal_event
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
                error_detail_for_log(exc),
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
                    error_message=error_detail_for_log(exc),
                )
                _app_log.info(
                    "[%s fallback.stream.skipped] target=%s provider=%s trigger=%s reason=client_output_started",
                    log_label,
                    target.model,
                    fallback_provider_id or "-",
                    trigger,
                )
                raise
            retry_key = (target.provider_id or "", target.model)
            empty_or_placeholder = (
                getattr(exc, "placeholder_only_response", False)
                or getattr(exc, "empty_stream_response", False)
            )
            if index > 0 and empty_or_placeholder and degenerate_retries.get(retry_key, 0) < 1:
                degenerate_retries[retry_key] = degenerate_retries.get(retry_key, 0) + 1
                _app_log.warning(
                    "[%s fallback.stream.retry] target=%s provider=%s reason=%s retry=%d",
                    log_label,
                    target.model,
                    fallback_provider_id or "-",
                    "placeholder_only_response" if getattr(exc, "placeholder_only_response", False) else "empty_stream",
                    degenerate_retries[retry_key],
                )
                continue
            retry_limit = _same_target_retry_limit()
            if (
                not emitted
                and is_same_target_retryable(exc, trigger)
                and same_target_retries.get(retry_key, 0) < retry_limit
            ):
                same_target_retries[retry_key] = same_target_retries.get(retry_key, 0) + 1
                _app_log.warning(
                    "[%s stream.retry] target=%s provider=%s trigger=%s retry=%d/%d",
                    log_label,
                    target.model,
                    fallback_provider_id or "-",
                    trigger,
                    same_target_retries[retry_key],
                    retry_limit,
                )
                continue
            if index == 0:
                decision = apply_fallback_policy(fallback_provider_id, target.model, trigger)
                if not decision.matched and budget.matched and trigger == "timeout":
                    decision = budget
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
                        error_message=error_detail_for_log(exc),
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
                if decision.matched:
                    attempt_timeout = decision.attempt_timeout
                _app_log.info(
                    "[%s fallback.stream.decision] matched=True policy_id=%s policy='%s' source=%s provider=%s trigger=%s chain=%d attempt_timeout=%s",
                    log_label,
                    decision.policy_id,
                    decision.policy_name,
                    target.model,
                    fallback_provider_id or "-",
                    trigger,
                    len(targets) - 1,
                    attempt_timeout if attempt_timeout is not None else "-",
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
                    error_detail_for_log(last_exc),
                )
        finally:
            # 本轮无论成功、失败还是被上层提前丢弃，都必须释放上游流；
            # 否则被下一轮覆盖的旧生成器只能等 GC 才关闭（S5）。
            # timed_events 的 finally 会级联关闭它包裹的 events，无需重复关闭。
            await aclose_async_iterator(timed_events)

    _app_log.error(
        "[%s fallback.stream.exhausted] primary=%s provider=%s candidates=%d error=%s",
        log_label,
        primary.model,
        primary.provider_id or "-",
        max(len(targets) - 1, 0),
        error_detail_for_log(last_exc) if last_exc else "no target available",
    )
    if last_exc is not None:
        _attach_request_details(last_exc, fallback_status="exhausted", fallback_reason="all fallback targets failed", fallback_attempts=fallback_attempts)
    raise last_exc or RuntimeError("No routing target available")


def _thinking_fields_from_payload(payload: dict | None) -> dict:
    """Extract thinking controls from a request body or extra dict.

    Pi/OpenAI-compatible clients usually send only reasoning_effort. Persist
    that as enable_thinking so stats details do not render as undefined.
    """
    return thinking_fields_from_body(payload)


def _normalized_request_details(endpoint: str, details: dict | None) -> dict:
    """Promote image-generation requests to a stable logging dimension."""
    normalized = dict(details or {})
    for key, value in _thinking_fields_from_payload(normalized).items():
        normalized.setdefault(key, value)
    mode = str(normalized.get("responses_mode") or "")
    if (
        normalized.get("request_kind") == "image_generation"
        or endpoint == "images_generations"
        or normalized.get("upstream_endpoint") == "images/generations"
        or "image_generation" in mode
    ):
        normalized["request_kind"] = "image_generation"
        for key in (
            "image_count", "image_bytes", "image_artifact_count",
            "image_requested_count", "image_succeeded_count", "image_failed_count",
            "image_retried_count", "image_reused_count",
        ):
            try:
                normalized[key] = max(0, int(normalized.get(key) or 0))
            except (TypeError, ValueError):
                normalized[key] = 0
        normalized["image_model"] = str(normalized.get("image_model") or "")
    else:
        normalized.setdefault("request_kind", "text_generation")
    return normalized


def _record_image_generation_failure(
    *, username: str, api_key_value: str, requested_model: str, model: str,
    provider_id: str, endpoint: str, request_body: dict, exc: Exception,
    request_log_id: int | None = None,
) -> None:
    details = _normalized_request_details(
        endpoint,
        {
            **_request_details_from_exception(exc),
            "request_kind": "image_generation",
            "responses_mode": _request_details_from_exception(exc).get(
                "responses_mode", "image_generation"
            ),
            "upstream_endpoint": "images/generations",
            "error_message": error_detail_for_log(exc),
        },
    )
    final_model = str(details.get("attempted_model") or model or requested_model)
    final_provider = str(details.get("attempted_provider") or provider_id or "")
    _log_request(
        username, api_key_value, final_model, final_provider,
        endpoint, False, 0, requested_model, details=details,
    )
    _record_request_log(
        endpoint=endpoint, username=username, api_key_value=api_key_value,
        requested_model=requested_model, final_model=final_model,
        final_provider=final_provider, request_body=request_body,
        success=False, status="fail", tokens=0, details=details,
        error_message=error_detail_for_log(exc),
        log_id=request_log_id,
    )
    _record_success_metrics(username, api_key_value, 0, "fail")


def _log_request(username: str, api_key: str, model: str, provider_id: str,
                 endpoint: str, success: bool, tokens: int,
                 requested_model: str = "", *, details: dict | None = None) -> None:
    detail = _normalized_request_details(endpoint, details)
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
        "fallback_attempts",
        "routing_matched",
        "routing_rule_id",
        "routing_rule_name",
        "routing_reason",
        "routed_model",
        "routed_provider",
        "http_status",
        "reject_reason",
        "client_disconnected",
        "responses_stateful",
        "responses_state_markers",
        "fallback_safety_decision",
        "stateful_fallback_blocked",
        "upstream_endpoint",
        "responses_mode",
        "request_kind",
        "image_model",
        "image_backend_type",
        "image_backend_provider",
        "image_backend_model",
        "image_fallback_status",
        "planner_fallback_status",
        "planner_fallback_attempts",
        "image_count",
        "image_bytes",
        "image_artifact_count",
        "native_attempted",
        "native_failure_endpoint",
        "native_failure_status",
        "native_failure_reason",
        "native_failure_message",
        "native_attempts",
        "reasoning_effort",
        "chat_template_kwargs",
        "enable_thinking",
        "tps",
        "completion_tokens",
        "duration_ms",
        "generation_ms",
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
        if detail.get("request_kind") == "image_generation":
            increment_image_generation_stats(
                success,
                image_count=detail.get("image_count", 0),
                image_bytes=detail.get("image_bytes", 0),
            )
        add_request_record(
            model=requested_model or model,
            username=username,
            success=success,
            tokens=tokens,
            request_kind=detail.get("request_kind", ""),
            image_model=detail.get("image_model", ""),
            image_count=detail.get("image_count", 0),
            image_bytes=detail.get("image_bytes", 0),
        )
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


def verify_api_key(
    authorization: Optional[str] = None,
    *,
    endpoint: str = "",
    requested_model: str = "",
) -> tuple[dict, dict]:
    def _reject(detail: str, *, api_key_value: str = "") -> None:
        _log_rejected_request(
            status_code=401,
            detail=detail,
            endpoint=endpoint,
            api_key_value=api_key_value,
            requested_model=requested_model,
        )
        raise HTTPException(status_code=401, detail=detail)

    if not authorization:
        _reject("Missing Authorization header")

    if not authorization.startswith("Bearer "):
        _reject("Invalid Authorization format")

    token = authorization[7:].strip()
    if not token:
        _reject("Missing API key")

    user_match = find_user_by_api_key(token)
    if user_match:
        return user_match

    _reject("Invalid API key", api_key_value=token)
    raise HTTPException(status_code=401, detail="Invalid API key")  # unreachable, for type checkers

async def verify_api_key_async(
    authorization: Optional[str],
    *,
    endpoint: str = "",
    requested_model: str = "",
) -> tuple[dict, dict]:
    """Run the credential lookup off the event loop.

    认证是每请求必经的同步 SQLite 查询，留在事件循环里会把所有开流请求
    排在它后面（见「当前问题.md」P1）。
    """
    return await anyio.to_thread.run_sync(
        partial(verify_api_key, authorization, endpoint=endpoint, requested_model=requested_model)
    )


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


def ensure_model_allowed(user: dict, api_key: dict, model: str, *, endpoint: str = "") -> None:
    allowed = allowed_models_for(user, api_key)
    if "*" in allowed:
        return
    requested = parse_model_id(model)
    if _model_allowed_by_list(allowed, model):
        return

    def _deny(detail: str) -> None:
        _app_log.warning(
            "[model.allow.denied] model=%s user=%s key=%s reason=not_in_allow_list allow_list=%s",
            model,
            user.get("username", "?"),
            mask_key(api_key.get("key", "")),
            allowed,
        )
        _log_rejected_request(
            status_code=403,
            detail=detail,
            endpoint=endpoint,
            username=user.get("username", ""),
            api_key_value=api_key.get("key", ""),
            requested_model=model,
            model=model,
        )
        raise HTTPException(status_code=403, detail=detail)

    if any("/" in str(allowed_model) for allowed_model in allowed) and not requested.is_composite:
        _deny(f"Model '{model}' is not allowed for this API key; use a provider-qualified model id")
    _deny(f"Model '{model}' is not allowed for this API key")


def ensure_routed_model_allowed(
    user: dict,
    api_key: dict,
    requested_model: str,
    target_model: str,
    target_provider: str = "",
    *,
    endpoint: str = "",
) -> None:
    if requested_model == target_model and not target_provider:
        ensure_model_allowed(user, api_key, requested_model, endpoint=endpoint)
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

    def _deny(detail: str) -> None:
        _app_log.warning(
            "[model.allow.denied] requested=%s routed_to=%s user=%s key=%s reason=not_in_allow_list allow_list=%s",
            requested_model,
            effective_target,
            user.get("username", "?"),
            mask_key(api_key.get("key", "")),
            allowed,
        )
        _log_rejected_request(
            status_code=403,
            detail=detail,
            endpoint=endpoint,
            username=user.get("username", ""),
            api_key_value=api_key.get("key", ""),
            requested_model=requested_model,
            model=effective_target or target_model,
            provider=target_provider or "",
        )
        raise HTTPException(status_code=403, detail=detail)

    if any("/" in str(allowed_model) for allowed_model in allowed) and not requested.is_composite:
        _deny(f"Model '{requested_model}' is not allowed for this API key; use a provider-qualified model id")
    _deny(f"Model '{requested_model}' is not allowed for this API key")

@router.get("/models")
def list_models(authorization: Optional[str] = Header(None)):
    user, api_key = verify_api_key(authorization, endpoint="models")
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
                    # 能力元数据：内置家族表 < 在线注册表 < 上游透传 < 管理员覆盖，
                    # 让下游 harness 获取模型列表时顺带拿到上下文/视觉/工具能力。
                    caps = resolve_model_capabilities(model, remote=registry_lookup(model["id"], model.get("name", "")))
                    # Models with native vision support or a vision preprocessor should advertise image support
                    # so clients such as Codex/OpenCode send image blocks instead of text placeholders.
                    # 注意：`or caps.get("supports_vision")` 不是冗余——名称启发式
                    # 未覆盖但在线注册表/内置表知道支持视觉的模型（如新款
                    # deepseek-flash）靠这一支广告 image_support/multimodal。
                    if _model_should_advertise_vision(provider, model) or caps.get("supports_vision"):
                        entry["supports_vision"] = True
                        entry["image_support"] = True
                        entry["multimodal"] = True
                    # 生图能力：provider_models.image_generation 显式标记的模型才广告（正声明，
                    # 缺失 = 不支持），让 live_eval 等下游按能力 gate 生图探针。
                    if get_model_image_generation(provider["id"], model["id"]):
                        entry["supports_image_generation"] = True
                    # 客户端契约：能力字段只做正向声明，缺失 = 未知/不支持；
                    # capabilities_for_client_entry 不会输出显式 false。
                    entry.update(capabilities_for_client_entry(caps))
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
    user, api_key = await verify_api_key_async(authorization, endpoint="chat_completions")

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
    ensure_routed_model_allowed(
        user, api_key, requested_model, model, provider_id, endpoint="chat_completions"
    )
    conv_key = policy.conv_key
    provider_info = resolve_provider(model, provider_id)
    adapter_provider_id = provider_for_log(provider_info, provider_id)

    # Image-generation bridge for chat clients. 现代 harness 约定：工具可用性
    # 只由模型能力 + 后端配置决定（should_inject_image_bridge），调用时机由模型
    # 自主决定，成本由每会话生图预算后置控制（state.charge_image_generation_budget）。
    # 纯模型驱动：工具可用性只由模型能力 + 后端配置决定，调用时机完全由模型
    # 自主决定。注入了 bridge 工具后流式一律走缓冲路径（模型可能在任何一轮
    # 调用工具，网关必须在转发前看到完整响应才能执行生图）；模型不调用就
    # 原样 passthrough，不做任何意图判断或强制纠正。
    image_enabled = bool(provider_info and get_model_image_generation(adapter_provider_id, model))
    chat_user_text = _chat_latest_user_text(internal)
    image_bridge = should_inject_image_bridge(image_enabled=image_enabled)
    if image_bridge:
        configure_internal_image_bridge(internal, body)
        _app_log.info(
            "[chat image_generation.bridge_injected] model=%s provider=%s stream=%s",
            model, adapter_provider_id or "-", stream,
        )

    try:
        if stream:
            events = _stream_events_with_fallbacks(
                internal,
                temperature=temperature,
                max_tokens=max_tokens,
                log_label="chat" if not image_bridge else "chat.image_bridge",
            )
            if image_bridge:
                # 真流式 + 流末续接：文本实时转发给客户端；上游流结束时若模型
                # 调用了 bridge 工具，在这里执行生图并把结果作为同一 SSE 流的
                # 续接事件（客户端看到：实时文本 -> 暂停生成 -> 图片结果/续轮）。
                bridge_call_events: list = []
                forwarded_tool_indexes: set = set()
                image_log_id_box = [0]
                source_events = events  # 闭包晚绑定：先固定上游事件流，避免下方 events 重绑定后迭代到自身

                async def _run_bridge_continuation():
                    planner_output = _events_to_message(bridge_call_events)
                    _maybe_repair_tool_leak(planner_output, internal, endpoint="chat.image_bridge", provider_id=adapter_provider_id)
                    bridge_model = internal.target_model

                    configured_generator = _resolved_image_generator(get_enabled_image_generator() or {})
                    image_provider, image_model = _image_generator_identity(configured_generator)
                    image_provider = image_provider or adapter_provider_id
                    image_model = image_model or bridge_model
                    running_details = {
                        **routing_details_from_policy(policy),
                        **_thinking_fields_from_payload(body),
                        "request_kind": "image_generation",
                        "chat_mode": "model_driven_image_generation_running",
                        "upstream_endpoint": "images/generations",
                        "image_model": image_model,
                        "image_backend_provider": image_provider,
                        "image_backend_model": image_model,
                        "image_backend_type": str(configured_generator.get("backend_type") or ""),
                        "image_fallback_status": "unused",
                        "image_requested_count": 0,
                        "image_succeeded_count": 0,
                        "image_failed_count": 0,
                        "image_count": 0,
                        "image_bytes": 0,
                        "image_artifact_count": 0,
                        "stream": True,
                        "status": "running",
                    }
                    def _record_running():
                        image_log_id_box[0] = _record_request_log(
                            endpoint="chat_completions", username=username, api_key_value=api_key_value,
                            requested_model=requested_model, final_model=bridge_model,
                            final_provider=image_provider, request_body=body,
                            response_body=None, success=True, status="running", tokens=0,
                            details=running_details, stream=True,
                        )
                        return image_log_id_box[0]

                    def _on_progress(progress_details):
                        _record_request_log(
                            endpoint="chat_completions", username=username, api_key_value=api_key_value,
                            requested_model=requested_model, final_model=bridge_model,
                            final_provider=image_provider, request_body=body,
                            response_body=None, success=True, status="running", tokens=0,
                            details=progress_details, stream=True, log_id=image_log_id_box[0],
                        )

                    def _describe_upstream(out, provider_id):
                        details = _output_request_details(out)
                        final_model = _target_model_for_log(
                            RouteTarget(model=internal.target_model, provider_id=provider_id),
                            provider_id,
                        )
                        return details, final_model, provider_id

                    outcome = await run_image_bridge(
                        internal, policy=policy, model=bridge_model, temperature=temperature, max_tokens=max_tokens,
                        base_details={**routing_details_from_policy(policy), **_thinking_fields_from_payload(body)},
                        running_details=running_details,
                        configured_generator=configured_generator,
                        planner_output=planner_output,
                        planner_provider_info=provider_info,
                        planner_provider_id=adapter_provider_id,
                        has_client_image_exec_tool=False,
                        call_model=_call_nonstream_with_fallbacks,
                        execute_invocations=lambda invocations, progress=None: _execute_image_invocations(
                            body, username=username, api_key_value=api_key_value, invocations=invocations, progress=progress,
                        ),
                        build_artifacts=lambda stored, args, start_index, used_filenames: _stored_image_artifacts(
                            request, stored, arguments=args, start_index=start_index,
                            used_filenames=used_filenames,
                        ),
                        render_client_output=lambda results, stored, artifacts, usage: (
                            chat_image_url_only_output(results, stored, artifacts, usage), "markdown"
                        ),
                        latest_user_text=lambda: chat_user_text,
                        record_running=_record_running,
                        on_progress=_on_progress,
                        describe_upstream=_describe_upstream,
                        merge_upstream=_merge_bridge_request_details,
                        log_label="chat.image_bridge",
                    )
                    if outcome is None:
                        return
                    if outcome.display_mode == "passthrough":
                        _app_log.info(
                            "[chat.image_bridge passthrough] model=%s provider=%s tool_calls=%d",
                            outcome.bridge_final_model, outcome.bridge_final_provider or "-",
                            len(outcome.image_output.tool_calls),
                        )
                    else:
                        # 生图是流末续接执行，客户端已收到实时文本，先发一条
                        # 提示避免看起来卡住。
                        yield InternalOutputEvent(
                            kind="text_delta", text="\n\n（正在生成图片，请稍候…）\n\n",
                        )
                    for ev in _message_to_events(outcome.image_output):
                        if (
                            ev.kind == "message_done"
                            and forwarded_tool_indexes
                            and (ev.finish_reason or "stop") != "tool_calls"
                        ):
                            # 同轮已实时转发过客户端工具调用：最终 finish_reason
                            # 必须是 tool_calls，否则客户端会把工具轮当文本轮收尾。
                            ev = replace(ev, finish_reason="tool_calls")
                        yield ev

                async def _live_bridge_events():
                    async for ev in source_events:
                        if (
                            ev.kind in ("tool_call_start", "tool_call_arguments_delta", "tool_call_done")
                            and ev.name == IMAGE_BRIDGE_TOOL_NAME
                        ):
                            bridge_call_events.append(ev)
                            continue
                        if ev.kind == "tool_call_start":
                            forwarded_tool_indexes.add(ev.tool_index)
                        if ev.kind == "message_done" and bridge_call_events:
                            # renderer 在 message_done 处终止迭代（源流不会被耗尽），
                            # 所以 bridge 必须在转发 message_done 之前执行；续接事件
                            # 自带最终 message_done。
                            async for cont in _run_bridge_continuation():
                                yield cont
                            return
                        yield ev
                    if bridge_call_events:
                        # 上游流在 message_done 前就结束（异常截断等）：兜底续接。
                        async for cont in _run_bridge_continuation():
                            yield cont

                events = _live_bridge_events()
                # 生图 bridge 会先记一行 "running" 日志；流编排器的最终日志复用
                # 同一 log_id 覆盖它，避免请求列表里留下中间态记录。
                _base_stream_recorder = _build_stream_recorder("chat_completions", username, api_key_value, requested_model, body)

                def _bridge_aware_stream_recorder(**payload):
                    if image_log_id_box[0]:
                        payload["log_id"] = image_log_id_box[0]
                    _base_stream_recorder(**payload)

                record_request_log = _bridge_aware_stream_recorder
            else:
                record_request_log = _build_stream_recorder("chat_completions", username, api_key_value, requested_model, body)
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
                    record_request_log=record_request_log,
                    conv_key=conv_key,
                    remember_reasoning_content=_remember_reasoning_content,
                    tool_only_turns=_tool_only_turns,
                    base_details={**routing_details_from_policy(policy), **_thinking_fields_from_payload(body)},
                    render_extra={"include_usage": bool((body.get("stream_options") or {}).get("include_usage"))},
                    declared_tools=internal.tools,
                ),
                media_type="text/event-stream"
            )

        if image_bridge:
            # The entry prepare_request_policy already ran the full strategy
            # (routing/preprocess/reasoning/transforms) and every step is
            # idempotent, so the bridge reuses that policy instead of running a
            # second full pass. The bridge tool was injected above, after the
            # first pass, but policy decisions are insensitive to it.
            output, provider_info, adapter_provider_id = await _call_nonstream_with_fallbacks(
                policy,
                internal,
                temperature=temperature,
                max_tokens=max_tokens,
                log_label="chat.image_bridge",
            )
            _maybe_repair_tool_leak(output, internal, endpoint="chat.image_bridge", provider_id=adapter_provider_id)
            model = internal.target_model
            provider_id = internal.provider_id

            configured_generator = _resolved_image_generator(get_enabled_image_generator() or {})
            image_provider, image_model = _image_generator_identity(configured_generator)
            image_provider = image_provider or adapter_provider_id
            image_model = image_model or model
            running_details = {
                **routing_details_from_policy(policy),
                **_thinking_fields_from_payload(body),
                "request_kind": "image_generation",
                "chat_mode": "model_driven_image_generation_running",
                "upstream_endpoint": "images/generations",
                "image_model": image_model,
                "image_backend_provider": image_provider,
                "image_backend_model": image_model,
                "image_backend_type": str(configured_generator.get("backend_type") or ""),
                "image_fallback_status": "unused",
                "image_requested_count": 0,
                "image_succeeded_count": 0,
                "image_failed_count": 0,
                "image_count": 0,
                "image_bytes": 0,
                "image_artifact_count": 0,
                "status": "running",
            }
            _image_log_id_box = [0]

            def _record_running():
                _image_log_id_box[0] = _record_request_log(
                    endpoint="chat_completions", username=username, api_key_value=api_key_value,
                    requested_model=requested_model, final_model=model,
                    final_provider=image_provider, request_body=body,
                    response_body=None, success=True, status="running", tokens=0,
                    details=running_details, stream=False,
                )
                return _image_log_id_box[0]

            def _on_progress(progress_details):
                _record_request_log(
                    endpoint="chat_completions", username=username, api_key_value=api_key_value,
                    requested_model=requested_model, final_model=model,
                    final_provider=image_provider, request_body=body,
                    response_body=None, success=True, status="running", tokens=0,
                    details=progress_details, stream=False, log_id=_image_log_id_box[0],
                )

            def _describe_upstream(out, provider_id):
                details = _output_request_details(out)
                final_model = _target_model_for_log(
                    RouteTarget(model=internal.target_model, provider_id=provider_id),
                    provider_id,
                )
                return details, final_model, provider_id

            outcome = await run_image_bridge(
                internal,
                policy=policy,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_details={**routing_details_from_policy(policy), **_thinking_fields_from_payload(body)},
                running_details=running_details,
                configured_generator=configured_generator,
                planner_output=output,
                planner_provider_info=provider_info,
                planner_provider_id=adapter_provider_id,
                has_client_image_exec_tool=False,
                call_model=_call_nonstream_with_fallbacks,
                execute_invocations=lambda invocations, progress=None: _execute_image_invocations(
                    body, username=username, api_key_value=api_key_value,
                    invocations=invocations, progress=progress,
                ),
                build_artifacts=lambda stored, args, start_index, used_filenames: _stored_image_artifacts(
                    request, stored, arguments=args, start_index=start_index,
                    used_filenames=used_filenames,
                ),
                render_client_output=lambda results, stored, artifacts, usage: (
                    chat_image_url_only_output(results, stored, artifacts, usage), "markdown"
                ),
                latest_user_text=lambda: chat_user_text,
                record_running=_record_running,
                on_progress=_on_progress,
                describe_upstream=_describe_upstream,
                merge_upstream=_merge_bridge_request_details,
                log_label="chat.image_bridge",
            )
            if outcome is not None:
                image_output = outcome.image_output
                details = outcome.details
                request_status = outcome.request_status
                tokens = outcome.tokens
                bridge_final_model = outcome.bridge_final_model
                bridge_final_provider = outcome.bridge_final_provider
                image_request_log_id = _image_log_id_box[0]
                rendered = render_chat_completion(image_output, model=model)
                details = apply_outcome_to_details(details, success=True)
                _log_request(username, api_key_value, bridge_final_model, bridge_final_provider, "chat_completions", True, tokens, requested_model, details=details)
                _record_request_log(
                    endpoint="chat_completions", username=username, api_key_value=api_key_value,
                    requested_model=requested_model, final_model=bridge_final_model,
                    final_provider=bridge_final_provider, request_body=body,
                    response_body=rendered, success=True, status=request_status, tokens=tokens,
                    usage=outcome.usage, details=details, log_id=image_request_log_id,
                )
                _record_success_metrics(username, api_key_value, tokens, request_status)
                return rendered

            # The model chose not to generate an image. Never expose the
            # gateway's private proxy function in the client-visible response.
            output.tool_calls = [call for call in output.tool_calls if call.name != IMAGE_BRIDGE_TOOL_NAME]
            logged_model = _target_model_for_log(RouteTarget(model=internal.target_model, provider_id=adapter_provider_id), adapter_provider_id)
            rendered = render_chat_completion(output, model=model)
            tokens = output.usage.get("total_tokens", 0)
            details = {**routing_details_from_policy(policy), **_output_request_details(output), "chat_mode": "image_bridge_model_passthrough"}
            details = apply_outcome_to_details(details, success=True)
            _log_request(username, api_key_value, logged_model, adapter_provider_id or "", "chat_completions", True, tokens, requested_model, details=details)
            _record_request_log(
                endpoint="chat_completions", username=username, api_key_value=api_key_value,
                requested_model=requested_model, final_model=logged_model,
                final_provider=adapter_provider_id or "", request_body=body,
                response_body=rendered, success=True, status=details.get("status", "ok"), tokens=tokens,
                usage=output.usage, details=details,
            )
            _record_success_metrics(username, api_key_value, tokens, details.get("status", "ok"))
            return rendered

        output, provider_info, adapter_provider_id = await _call_nonstream_with_fallbacks(
            policy,
            internal,
            temperature=temperature,
            max_tokens=max_tokens,
            log_label="chat",
        )
        _maybe_repair_tool_leak(output, internal, endpoint="chat_completions", provider_id=adapter_provider_id)
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
        success_details = _finalize_success_details(output, policy=policy, extra=_thinking_fields_from_payload(body))
        status = success_details.get("status", "ok")
        tokens = output.usage.get("total_tokens", 0)
        _record_request_log(
            endpoint="chat_completions",
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            final_model=logged_model, final_provider=adapter_provider_id or "",
            request_body=body, response_body=rendered,
            success=True, status=status, tokens=tokens,
            usage=output.usage, details=success_details,
        )
        _log_request(username, api_key_value, logged_model, adapter_provider_id or "", "chat_completions", True, tokens, requested_model, details=success_details)
        _record_success_metrics(username, api_key_value, tokens, status)
        return rendered
    except HTTPException as http_exc:
        # image bridge correction 失败（502）等透传路径也要留下失败日志/统计，
        # 与 completions/messages/responses 端点对齐，避免可观测性盲区。
        _log_upstream_http_exception_failure(
            "chat_completions", http_exc,
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            model=model, provider_id=adapter_provider_id or provider_id or "", body=body,
        )
        raise
    except Exception as e:
        _error_log.error("[chat] %s", error_detail_for_log(e))
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
            tokens=0, details=details, error_message=error_detail_for_log(e),
        )
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        # 客户端只拿到安全消息，上游原文必须先落日志，否则无法回溯。
        _error_log.error("[chat_completions] FAILED: %s", error_detail_for_log(e))
        raise HTTPException(status_code=client_status_for_upstream_error(e), detail=friendly_error_msg(e))

@router.post("/completions")
async def completions(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = await verify_api_key_async(authorization, endpoint="completions")

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
    ensure_routed_model_allowed(
        user, api_key, requested_model, model, provider_id, endpoint="completions"
    )
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
                    declared_tools=internal.tools,
                    base_details={**routing_details_from_policy(policy), **_thinking_fields_from_payload(body)},
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
        _maybe_repair_tool_leak(output, internal, endpoint="completions", provider_id=adapter_provider_id)
        model = internal.target_model
        provider_id = internal.provider_id
        logged_model = _target_model_for_log(RouteTarget(model=model, provider_id=adapter_provider_id or provider_id or ""), adapter_provider_id or provider_id or "")
        rendered = render_completion(output, model=model)
        success_details = _finalize_success_details(output, policy=policy, extra=_thinking_fields_from_payload(body))
        status = success_details.get("status", "ok")
        tokens = output.usage.get("total_tokens", 0)
        _log_request(username, api_key_value, logged_model, adapter_provider_id or "", "completions", True, tokens, requested_model, details=success_details)
        _record_request_log(
            endpoint="completions",
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            final_model=logged_model, final_provider=adapter_provider_id or "",
            request_body=body, response_body=rendered,
            success=True, status=status, tokens=tokens,
            usage=output.usage, details=success_details,
        )
        _record_success_metrics(username, api_key_value, tokens, status)
        return rendered
    except HTTPException as http_exc:
        # 适配器把上游状态映射为 HTTPException（429/502 等），必须保留状态码
        # 语义透传，不能压成 500，否则客户端无法正确退避；同时补记失败
        # 日志/统计，避免透传造成可观测性回归。
        _log_upstream_http_exception_failure(
            "completions", http_exc,
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            model=model, provider_id=provider_id or "", body=body,
        )
        raise
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
            tokens=0, details=details, error_message=error_detail_for_log(e),
        )
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", error_detail_for_log(e))
        raise HTTPException(status_code=client_status_for_upstream_error(e), detail=friendly_error_msg(e))

@router.post("/messages")
async def anthropic_messages(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = await verify_api_key_async(authorization, endpoint="messages")

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
    ensure_routed_model_allowed(
        user, api_key, requested_model, model, provider_id, endpoint="messages"
    )
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
                    declared_tools=internal.tools,
                    remember_reasoning_content=_remember_reasoning_content,
                    base_details={**routing_details_from_policy(policy), **_thinking_fields_from_payload(body)},
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
        _maybe_repair_tool_leak(output, internal, endpoint="messages", provider_id=adapter_provider_id)
        model = internal.target_model
        provider_id = internal.provider_id
        logged_model = _target_model_for_log(RouteTarget(model=model, provider_id=adapter_provider_id or provider_id or ""), adapter_provider_id or provider_id or "")
        if output.reasoning:
            _remember_reasoning_content(conv_key, output.reasoning, [tool.id for tool in output.tool_calls])
            _app_log.debug("[messages_nonstream] STORED rc key=%s len=%d cache_hit=%d cache_miss=%d",
                          conv_key[:60], len(output.reasoning),
                          output.usage.get("prompt_cache_hit_tokens", 0), output.usage.get("prompt_cache_miss_tokens", 0))
        rendered = render_anthropic_message(output, model=model)
        success_details = _finalize_success_details(output, policy=policy, extra=_thinking_fields_from_payload(body))
        status = success_details.get("status", "ok")
        tokens = output.usage.get("total_tokens", 0)
        _log_request(username, api_key_value, logged_model, adapter_provider_id, "messages", True, tokens, requested_model, details=success_details)
        _record_request_log(
            endpoint="messages",
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            final_model=logged_model, final_provider=adapter_provider_id,
            request_body=body, response_body=rendered,
            success=True, status=status, tokens=tokens,
            usage=output.usage, details=success_details,
        )
        _record_success_metrics(username, api_key_value, tokens, status)
        return rendered
    except HTTPException as http_exc:
        # 保留 anthropic 适配器映射的上游状态码（429/502 等），同时补记失败日志。
        _log_upstream_http_exception_failure(
            "messages", http_exc,
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            model=model, provider_id=adapter_provider_id or provider_id or "", body=body,
        )
        raise
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
            tokens=0, details=details, error_message=error_detail_for_log(e),
        )
        increment_global_stats(success=False)
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", error_detail_for_log(e))
        raise HTTPException(status_code=client_status_for_upstream_error(e), detail=friendly_error_msg(e))


@router.post("/responses")
async def responses_endpoint(request: Request, authorization: Optional[str] = Header(None)):
    user, api_key = await verify_api_key_async(authorization, endpoint="responses")

    body = await request.json()
    # Read the hidden manifest before display follow-up sanitization replaces
    # the large generatedImage script with its compact placeholder.
    image_asset_context = gateway_generated_image_asset_context(body.get("input"))
    image_display_followup = is_gateway_image_display_followup(body.get("input"))
    image_already_generated = has_gateway_generated_image_history(body.get("input"))
    if image_display_followup:
        sanitize_gateway_image_display_followup(body.get("input"))
    if sanitize_gateway_generated_image_history(body.get("input")):
        _app_log.info("[responses image_generation.history_compacted] removed=base64_previews")
    internal = responses_to_internal(body)
    if image_asset_context:
        prepend_system_text(
            internal.messages,
            "Gateway-generated project assets from the current user task follow. "
            "These URLs are available even if the original display message is removed during "
            "conversation normalization. Download and use the originals before continuing; "
            "do not regenerate or replace them merely because they are not yet in the local "
            f"workspace.\n\n{image_asset_context}",
        )
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
        reasoning_context=None,
        normalize=False,
        preprocess=False,
        apply_ir_transforms=False,
        log_label="responses",
    )
    model = internal.target_model
    provider_id = internal.provider_id
    ensure_routed_model_allowed(
        user, api_key, requested_model, model, provider_id, endpoint="responses"
    )
    provider_info = resolve_provider(model, provider_id)
    adapter_provider_id = provider_for_log(provider_info, provider_id)
    # Responses wire-level request flags are computed once at ingress and
    # carried in IR metadata; endpoint code must not re-parse the raw body.
    meta = internal.metadata
    image_tool = meta.get("image_generation_tool")
    explicit_image_choice = isinstance(body.get("tool_choice"), dict) and body["tool_choice"].get("type") == "image_generation"
    image_enabled = bool(provider_info and get_model_image_generation(adapter_provider_id, model))
    system_turn = bool(meta.get("is_system_turn"))
    # Sub2API leaves Codex's client-owned image_gen namespace intact.  The
    # first Responses turn must therefore return a namespaced function_call;
    # Codex will then call /images/generations itself.  Do not replace this
    # protocol with the gateway's synthetic image_generation_call response.
    codex_image_tool = bool(meta.get("has_codex_image_function_tool"))
    image_bridge = False

    # A forced hosted-tool choice is an explicit invocation and can execute
    # directly. A declaration with tool_choice=auto is only a capability: the
    # model must choose it through the bridge below.
    if image_tool and explicit_image_choice:
        if isinstance(input_data, list) and any(isinstance(item, dict) and item.get("type") == "input_image" for item in input_data):
            raise HTTPException(status_code=400, detail="Image editing is not supported by the configured image-generation backend")
        if body.get("previous_response_id"):
            raise HTTPException(status_code=400, detail="Image generation cannot use previous_response_id")
        if not image_enabled:
            raise HTTPException(status_code=403, detail="Image generation is not enabled for the requested model")
        prompt = str(meta.get("latest_user_prompt") or "")
        try:
            image_results, generator = await _generate_with_configured_backend(prompt, {
                "n": image_tool.get("n") or body.get("n"),
                "size": image_tool.get("size") or body.get("size"),
                "quality": image_tool.get("quality") or body.get("quality"),
                "background": image_tool.get("background") or body.get("background"),
                "output_format": image_tool.get("output_format") or body.get("output_format"),
            })
        except Exception as exc:
            # 显式图像生成分支在主 try 之外，失败必须自己补记统计/请求日志，
            # 否则管理端只看到成功记录，图像成功率虚高。
            fail_details = _request_details_from_exception(
                exc, stream=False,
                attempted_model=model, attempted_provider=adapter_provider_id or "",
            )
            fail_details = {
                **fail_details,
                "request_kind": "image_generation",
                "responses_mode": "image_generation",
                "upstream_endpoint": "images/generations",
            }
            _log_request(username, api_key_value, model, adapter_provider_id or "", "responses", False, 0, requested_model, details=fail_details)
            _record_request_log(
                endpoint="responses", username=username, api_key_value=api_key_value,
                requested_model=requested_model, final_model=model,
                final_provider=adapter_provider_id or "", request_body=body,
                response_body=None, success=False, status=fail_details.get("status", "fail"),
                tokens=0, details=fail_details, error_message=error_detail_for_log(exc),
            )
            fail_counters = stats_counters_for_status(fail_details.get("status", "fail"))
            increment_global_stats(
                False,
                degraded=fail_counters.degraded,
                rejected=fail_counters.rejected,
                cancelled=fail_counters.cancelled,
            )
            if username != "legacy":
                increment_user_usage(username, api_key_value, False, 0)
            raise
        image_provider, image_model = _image_generator_identity(generator)
        details = {**routing_details_from_policy(policy), "request_kind": "image_generation", "responses_mode": "image_generation", "upstream_endpoint": "images/generations", "image_model": image_model, "image_backend_provider": image_provider, "image_backend_model": image_model, "image_backend_type": str(generator.get("backend_type") or ""), "image_fallback_status": "unused", "image_count": len(image_results), "image_bytes": image_results_bytes(image_results)}
        if stream:
            # Streaming requests return before the body iterator runs. Record
            # the completed backend operation here so the admin statistics do
            # not lose successful image requests.
            _log_request(username, api_key_value, image_model, image_provider, "responses", True, 0, requested_model, details=details)
            _record_request_log(endpoint="responses", username=username, api_key_value=api_key_value, requested_model=requested_model, final_model=image_model, final_provider=image_provider, request_body=body, response_body={"status": "completed", "output_count": len(image_results)}, success=True, status="ok", tokens=0, usage={}, details={**details, "stream": True}, stream=True)
            _record_success_metrics(username, api_key_value, 0, "ok")
            _app_log.info(
                "[responses image_generation.bridge_wire] stream=true output_items=%d image_bytes=%d "
                "partial=false done=true completed_output=true",
                len(image_results),
                image_results_bytes(image_results),
            )
            return StreamingResponse(
                render_responses_image_generation_sse(image_results, model=model, previous_response_id=previous_response_id,
                                                      tool={"type": "image_generation", "output_format": "png"}),
                media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        rendered = render_responses_image_generation(image_results, model=model, previous_response_id=previous_response_id,
                                                     tool={"type": "image_generation", "output_format": "png"})
        _log_request(username, api_key_value, image_model, image_provider, "responses", True, 0, requested_model, details=details)
        _record_request_log(
            endpoint="responses", username=username, api_key_value=api_key_value,
            requested_model=requested_model, final_model=image_model,
            final_provider=image_provider, request_body=body,
            response_body={"status": "completed", "output_count": len(image_results)},
            success=True, status="ok", tokens=0, usage={}, details=details,
        )
        _record_success_metrics(username, api_key_value, 0, "ok")
        return rendered

    # 现代 harness 约定：桥接工具可用性只由模型能力 + 后端配置决定
    # （should_inject_image_bridge，与 chat 端点同一入口，/messages 未来扩展
    # 直接复用），调用时机由模型自主决定，成本由每会话生图预算后置控制。
    # system turn 与客户端自有的 Codex image_gen 命名空间除外（前者没有用户
    # 请求上下文，后者走客户端自持的 /images/generations 回路）。
    should_bridge_image = should_inject_image_bridge(
        image_enabled=image_enabled,
        system_turn=system_turn,
        has_codex_image_function_tool=codex_image_tool,
    )
    if should_bridge_image:
        inject_hosted_image_capability(body)
        configure_internal_image_bridge(internal, body)
        image_bridge = True
        _app_log.info(
            "[responses image_generation.bridge_injected] model=%s provider=%s tool_choice=%s",
            model, adapter_provider_id or "-", body.get("tool_choice"),
        )
    elif image_enabled and system_turn:
        _app_log.info(
            "[responses image_generation.bridge_suppressed] model=%s provider=%s reason=system_turn",
            model, adapter_provider_id or "-",
        )
    conv_key = policy.conv_key

    if isinstance(input_data, list):
        _app_log.debug(
            "[responses REASONING] injected=%d ir_messages=%d tool_msgs=%d rc_msgs=%d conv_key=%s",
            policy.reasoning_injected,
            len(internal.messages),
            _ir_tool_message_count(internal.messages),
            _ir_reasoning_message_count(internal.messages),
            conv_key[:60],
        )

    bridge_stored_images: list[StoredImageResult] = []
    bridge_image_results = []
    bridge_image_model = ""
    image_request_log_id = 0
    try:
        provider_info = resolve_provider(model, provider_id)
        adapter_provider_id = provider_for_log(provider_info, provider_id)

        if image_bridge:
            # Run the model first. The proxy tool call is consumed by the
            # gateway; ordinary text and client-owned tools are replayed through
            # the normal Responses renderer.
            policy = await prepare_request_policy(
                internal, username=username, api_key_value=api_key_value,
                preprocess_request=_policy_preprocess_request,
                conversation_cache_key=_conversation_cache_key,
                reasoning_context=_reasoning_context if isinstance(input_data, list) else None,
                tool_only_turns=_tool_only_turns,
                tool_only_limit=TOOL_ONLY_LIMIT,
                log_label="responses.image_bridge",
                conv_key_override=conv_key,
            )
            configure_internal_image_bridge(internal, body)
            output, provider_info, adapter_provider_id = await _call_nonstream_with_fallbacks(
                policy, internal, temperature=temperature, max_tokens=max_tokens,
                log_label="responses.image_bridge",
            )
            _maybe_repair_tool_leak(output, internal, endpoint="responses.image_bridge", provider_id=adapter_provider_id)
            # The fallback runner attaches the authoritative attempt/final
            # target metadata to the output. Preserve it through the image
            # bridge so billing and admin stats use the model that actually
            # served the request, not the client-requested primary.
            bridge_upstream_details = _output_request_details(output)
            bridge_final_model = _target_model_for_log(
                RouteTarget(model=internal.target_model, provider_id=adapter_provider_id),
                adapter_provider_id,
            )
            bridge_final_provider = adapter_provider_id
            configured_generator = _resolved_image_generator(get_enabled_image_generator() or {})
            image_provider, image_model = _image_generator_identity(configured_generator)
            image_provider = image_provider or adapter_provider_id
            image_model = image_model or model
            running_details = {
                **routing_details_from_policy(policy),
                "request_kind": "image_generation",
                "responses_mode": "model_driven_image_generation_running",
                "upstream_endpoint": "images/generations",
                "image_model": image_model,
                "image_backend_provider": image_provider,
                "image_backend_model": image_model,
                "image_backend_type": str(configured_generator.get("backend_type") or ""),
                "image_fallback_status": "unused",
                "image_requested_count": 0,
                "image_succeeded_count": 0,
                "image_failed_count": 0,
                "image_count": 0,
                "image_bytes": 0,
                "image_artifact_count": 0,
                "status": "running",
            }
            _image_log_id_box = [0]

            def _record_running():
                _image_log_id_box[0] = _record_request_log(
                    endpoint="responses", username=username, api_key_value=api_key_value,
                    requested_model=requested_model, final_model=model,
                    final_provider=image_provider, request_body=body,
                    response_body=None, success=True, status="running", tokens=0,
                    details=running_details, stream=stream,
                )
                return _image_log_id_box[0]

            def _on_progress(progress_details):
                _record_request_log(
                    endpoint="responses", username=username, api_key_value=api_key_value,
                    requested_model=requested_model, final_model=model,
                    final_provider=image_provider, request_body=body,
                    response_body=None, success=True, status="running", tokens=0,
                    details=progress_details, stream=stream, log_id=_image_log_id_box[0],
                )

            def _describe_upstream(out, provider_id):
                details = _output_request_details(out)
                final_model = _target_model_for_log(
                    RouteTarget(model=internal.target_model, provider_id=provider_id),
                    provider_id,
                )
                return details, final_model, provider_id

            outcome = await run_image_bridge(
                internal,
                policy=policy,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_details=routing_details_from_policy(policy),
                running_details=running_details,
                configured_generator=configured_generator,
                planner_output=output,
                planner_provider_info=provider_info,
                planner_provider_id=adapter_provider_id,
                has_client_image_exec_tool=bool(meta.get("has_codex_generated_image_exec_tool")),
                call_model=_call_nonstream_with_fallbacks,
                execute_invocations=lambda invocations, progress=None: _execute_image_invocations(
                    body, username=username, api_key_value=api_key_value,
                    invocations=invocations, progress=progress,
                ),
                build_artifacts=lambda stored, args, start_index, used_filenames: _stored_image_artifacts(
                    request, stored, arguments=args, start_index=start_index,
                    used_filenames=used_filenames,
                ),
                render_client_output=lambda results, stored, artifacts, usage: generated_image_client_output(
                    bool(meta.get("has_codex_generated_image_exec_tool")),
                    results, stored, artifacts, usage,
                ),
                latest_user_text=lambda: latest_user_text(input_data),
                record_running=_record_running,
                on_progress=_on_progress,
                describe_upstream=_describe_upstream,
                merge_upstream=_merge_bridge_request_details,
                log_label="responses.image_bridge",
            )
            if outcome is not None:
                image_output = outcome.image_output
                details = outcome.details
                request_status = outcome.request_status
                tokens = outcome.tokens
                image_results = outcome.image_results
                bridge_final_model = outcome.bridge_final_model
                bridge_final_provider = outcome.bridge_final_provider
                image_request_log_id = _image_log_id_box[0]
                if stream:
                    details = apply_outcome_to_details(details, success=True)
                    _log_request(username, api_key_value, bridge_final_model, bridge_final_provider, "responses", True, tokens, requested_model, details=details)
                    _record_request_log(
                        endpoint="responses", username=username, api_key_value=api_key_value,
                        requested_model=requested_model, final_model=bridge_final_model,
                        final_provider=bridge_final_provider, request_body=body,
                        response_body={"status": "completed", "output_count": len(image_results)},
                        success=True, status=request_status, tokens=tokens, usage=image_output.usage,
                        details=details, stream=True, log_id=image_request_log_id,
                    )
                    _record_success_metrics(username, api_key_value, tokens, request_status)
                    return StreamingResponse(
                        render_responses_sse(
                            _nonstream_output_events(image_output), model=model,
                            previous_response_id=previous_response_id,
                            extra=internal.extra,
                        ),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                rendered = render_response(
                    image_output, model=model, previous_response_id=previous_response_id,
                    extra=internal.extra,
                )
                details = apply_outcome_to_details(details, success=True)
                _log_request(username, api_key_value, bridge_final_model, bridge_final_provider, "responses", True, tokens, requested_model, details=details)
                _record_request_log(
                    endpoint="responses", username=username, api_key_value=api_key_value,
                        requested_model=requested_model, final_model=bridge_final_model,
                        final_provider=bridge_final_provider, request_body=body,
                        response_body=rendered, success=True, status=request_status, tokens=tokens,
                        usage=image_output.usage, details=details, log_id=image_request_log_id,
                )
                _record_success_metrics(username, api_key_value, tokens, request_status)
                return rendered


            # The model chose not to generate an image. Never expose the
            # gateway's private proxy function in the client-visible response.
            output.tool_calls = [call for call in output.tool_calls if call.name != IMAGE_BRIDGE_TOOL_NAME]
            if stream:
                return StreamingResponse(
                    _stream_internal_output(
                        events=_nonstream_output_events(output), endpoint="responses",
                        model=model, username=username, api_key_value=api_key_value,
                        provider_id=adapter_provider_id, requested_model=requested_model,
                        log_request=_log_request,
                        record_request_log=_build_stream_recorder("responses", username, api_key_value, requested_model, body),
                        base_details={**routing_details_from_policy(policy), **_output_request_details(output), **_thinking_fields_from_payload(body), "responses_mode": "image_bridge_model_passthrough"},
                        previous_response_id=previous_response_id, conv_key=conv_key,
                        remember_response_chain_key=_remember_response_chain_key,
                        remember_reasoning_content=_remember_reasoning_content,
                        tool_only_turns=_tool_only_turns, render_extra=internal.extra,
                        declared_tools=internal.tools,
                    ),
                    media_type="text/event-stream",
                )
            resp_id = f"resp_{uuid.uuid4().hex}"
            _remember_response_chain_key(resp_id, conv_key)
            rendered = render_response(output, model=model, previous_response_id=previous_response_id, response_id=resp_id, extra=internal.extra)
            tokens = output.usage.get("total_tokens", 0)
            details = {**routing_details_from_policy(policy), **_output_request_details(output), "responses_mode": "image_bridge_model_passthrough", "response_id": resp_id}
            final_model = _target_model_for_log(RouteTarget(model=internal.target_model, provider_id=adapter_provider_id), adapter_provider_id)
            details = apply_outcome_to_details(details, success=True)
            _log_request(username, api_key_value, final_model, adapter_provider_id, "responses", True, tokens, requested_model, details=details)
            _record_request_log(
                endpoint="responses", username=username, api_key_value=api_key_value,
                requested_model=requested_model, final_model=final_model,
                final_provider=adapter_provider_id, request_body=body,
                response_body=rendered, success=True, status=details.get("status", "ok"), tokens=tokens,
                usage=output.usage, details=details,
            )
            _record_success_metrics(username, api_key_value, tokens, details.get("status", "ok"))
            return rendered

        native_required = list(meta.get("requires_native_responses") or [])
        required_tool_types = set(meta.get("required_tool_types") or [])
        stateful_markers = list(meta.get("stateful_tool_markers") or [])
        capability = get_model_responses_capability(adapter_provider_id, model) if provider_info else None
        native_downgrade_details = {}
        native_supported = await native_capability_for_request(provider_info, model, has_tools=bool(body.get("tools")))
        _app_log.info(
            "[responses capability] provider=%s model=%s native=%s",
            adapter_provider_id or "-", model, native_supported,
        )
        # A native-only feature may still be served by a configured native
        # fallback even when the routed primary lacks Responses support.  Basic
        # requests, on the other hand, remain eligible for the Chat/Anthropic
        # compatibility path when no native stream is available.
        # Native Responses forwarding is not suitable for client-owned
        # Codex image namespaces on ordinary provider API endpoints.  The
        # compatibility adapter keeps the original function name and egress
        # restores namespace=image_gen, matching Sub2API's client-owned loop.
        if (
            not image_display_followup
            and not image_already_generated
            and (native_supported and not codex_image_tool)
        ):
            _app_log.info("[responses native] provider=%s model=%s stream=%s", adapter_provider_id or "-", model, stream)
            try:
                if stream:
                    native_events, used_target, used_provider_id, native_attempts = await _native_response_with_fallbacks(internal, stream=True, required_tool_types=required_tool_types, stateful_markers=stateful_markers)
                    final_model = _target_model_for_log(used_target, used_provider_id)
                    return StreamingResponse(
                        _native_responses_stream_with_accounting(
                        native_events, username=username, api_key_value=api_key_value,
                        model=final_model, provider_id=used_provider_id, requested_model=requested_model, policy=policy, request_body=body,
                        fallback_attempts=native_attempts,
                        required_tool_types=required_tool_types,
                            remember_response_chain_key=_remember_response_chain_key, conv_key=conv_key,
                        ),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                rendered, used_target, adapter_provider_id, native_attempts = await _native_response_with_fallbacks(internal, stream=False, required_tool_types=required_tool_types, stateful_markers=stateful_markers)
                model = _target_model_for_log(used_target, adapter_provider_id)
                usage = rendered.get("usage") or {}
                tokens = usage.get("total_tokens") or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
                if rendered.get("id"):
                    _remember_response_chain_key(rendered["id"], conv_key)
                observed = observed_response_tool_types(rendered)
                if observed:
                    capability = get_model_responses_capability(adapter_provider_id, model) or {}
                    update_model_responses_tool_types(adapter_provider_id, model, list(set(capability.get("responses_tool_types") or []) | observed))
                native_details = apply_outcome_to_details({**routing_details_from_policy(policy), **_thinking_fields_from_payload(body), "responses_mode": "native", "upstream_endpoint": "responses", "fallback_attempts": native_attempts}, success=True)
                status = native_details.get("status", "ok")
                _log_request(username, api_key_value, model, adapter_provider_id, "responses", True, tokens, requested_model, details=native_details)
                _record_request_log(endpoint="responses", username=username, api_key_value=api_key_value, requested_model=requested_model, final_model=model, final_provider=adapter_provider_id, request_body=body, response_body=rendered, success=True, status=status, tokens=tokens, usage=usage, details=native_details)
                _record_success_metrics(username, api_key_value, tokens, status)
                return rendered
            except Exception as native_error:
                native_attempts = list(getattr(native_error, "request_details", {}).get("fallback_attempts", []) or [])
                client_owned_tools = list(meta.get("client_owned_tool_markers") or [])
                if meta.get("incomplete_tool_history"):
                    # Chat Completions cannot invent missing tool outputs.
                    # Downgrading this shape only repeats the same 400 across
                    # every fallback provider.
                    _attach_request_details(
                        native_error,
                        fallback_status="skipped",
                        fallback_reason="incomplete_tool_history",
                        responses_mode="native",
                    )
                    _app_log.warning(
                        "[responses native fallback] refusing Chat downgrade for incomplete tool history: %s",
                        native_error,
                    )
                    raise _incomplete_tool_history_http_error(native_error)
                if native_required or client_owned_tools:
                    # Codex custom/namespace tools and other native-only
                    # features must not silently fall back to Chat.  The Chat
                    # adapter can still serve ordinary text/function requests.
                    if getattr(native_error, "native_capability_unavailable", False) or native_error_is_explicitly_unsupported(native_error):
                        raise HTTPException(status_code=422, detail=(
                            "No configured provider supports native Responses required by this request: "
                            + ", ".join(native_required or client_owned_tools)
                        )) from native_error
                    raise
                _app_log.warning("[responses native fallback] no native target succeeded; downgrading basic request: %s", error_detail_for_log(native_error))
                native_downgrade_details = _native_downgrade_details(native_error, native_attempts)

        if meta.get("incomplete_tool_history"):
            raise _incomplete_tool_history_http_error()

        # The initial minimal policy deliberately leaves a native payload untouched.
        # Once native dispatch is ruled out, run the full IR policy required by the
        # Chat/Anthropic compatibility adapters.
        policy = await prepare_request_policy(
            internal, username=username, api_key_value=api_key_value,
            preprocess_request=_policy_preprocess_request, conversation_cache_key=_conversation_cache_key,
            reasoning_context=_reasoning_context if isinstance(input_data, list) else None,
            tool_only_turns=_tool_only_turns,
            tool_only_limit=TOOL_ONLY_LIMIT,
            log_label="responses",
            conv_key_override=conv_key,
        )
        model = internal.target_model
        provider_id = internal.provider_id
        provider_info = resolve_provider(model, provider_id)
        adapter_provider_id = provider_for_log(provider_info, provider_id)
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
                    base_details={**routing_details_from_policy(policy), **native_downgrade_details, **_thinking_fields_from_payload(body)},
                    previous_response_id=previous_response_id,
                    conv_key=conv_key,
                    remember_response_chain_key=_remember_response_chain_key,
                    remember_reasoning_content=_remember_reasoning_content,
                    tool_only_turns=_tool_only_turns,
                    render_extra=internal.extra,
                    declared_tools=internal.tools,
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
        _maybe_repair_tool_leak(output, internal, endpoint="responses", provider_id=adapter_provider_id)
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
        rendered = render_response(output, model=model, previous_response_id=previous_response_id, response_id=resp_id, extra=internal.extra)
        success_details = _finalize_success_details(
            output, policy=policy,
            extra={"response_id": resp_id, **native_downgrade_details, **_thinking_fields_from_payload(body)},
        )
        status = success_details.get("status", "ok")
        tokens = output.usage.get("total_tokens", 0)
        _log_request(username, api_key_value, logged_model, adapter_provider_id, "responses", True, tokens, requested_model, details=success_details)
        _record_request_log(
            endpoint="responses",
            username=username, api_key_value=api_key_value, requested_model=requested_model,
            final_model=logged_model, final_provider=adapter_provider_id,
            request_body=body, response_body=rendered,
            success=True, status=status, tokens=tokens,
            usage=output.usage, details=success_details,
        )
        _record_success_metrics(username, api_key_value, tokens, status)
        return rendered
    except HTTPException as e:
        _rollback_image_bridge_artifacts(
            e,
            stored=bridge_stored_images,
            image_results=bridge_image_results,
            image_model=bridge_image_model,
        )
        if _request_details_from_exception(e).get("request_kind") == "image_generation":
            _record_image_generation_failure(
                username=username, api_key_value=api_key_value,
                requested_model=requested_model, model=model,
                provider_id=adapter_provider_id, endpoint="responses",
                request_body=body, exc=e,
                request_log_id=image_request_log_id or None,
            )
        raise
    except Exception as e:
        _rollback_image_bridge_artifacts(
            e,
            stored=bridge_stored_images,
            image_results=bridge_image_results,
            image_model=bridge_image_model,
        )
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
            tokens=0, details=details, error_message=error_detail_for_log(e),
            log_id=image_request_log_id or None,
        )
        increment_global_stats(success=False, stateful_fallback_blocked=bool(details.get("stateful_fallback_blocked")))
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", error_detail_for_log(e))
        raise HTTPException(status_code=client_status_for_upstream_error(e), detail=friendly_error_msg(e))


async def _images_generation_request(request: Request, authorization: Optional[str]):
    user, api_key = await verify_api_key_async(authorization, endpoint="images_generations")
    body = await request.json()
    requested_model = str(body.get("model") or "")
    prompt = str(body.get("prompt") or "").strip()
    if not requested_model or not prompt:
        raise HTTPException(status_code=400, detail="model and prompt are required")
    internal_model = parse_model_id(requested_model)
    provider_info = resolve_provider(internal_model.model_name, internal_model.provider_id)
    provider_id = provider_for_log(provider_info, internal_model.provider_id)
    # The model in an Images request is an image-backend hint, not the chat
    # model selected by the user.  In particular, Codex's image_gen extension
    # always sends "gpt-image-2" even when this gateway routes the operation to
    # a configured Grok or external backend.  Endpoint authorization has
    # already been enforced above; applying the chat-model allow-list here
    # would incorrectly reject the client-owned image generation step.
    configured_generator = get_enabled_image_generator()
    if not configured_generator:
        raise HTTPException(status_code=503, detail="No image-generation backend is enabled")
    generator = _resolved_image_generator(configured_generator)
    _app_log.info(
        "[images endpoint] requested_model=%s backend_type=%s api_base=%s provider_model=%s provider_id=%s model=%s effective_provider=%s effective_model=%s",
        requested_model,
        generator.get("backend_type") or "-", generator.get("api_base") or "-",
        generator.get("provider_model") or "-", generator.get("provider_id") or "-",
        generator.get("model") or "-",
        _image_generator_identity(generator)[0] or "-", _image_generator_identity(generator)[1] or "-",
    )
    generator.setdefault("max_retries", get_default("image_generation_max_retries", 2))
    generator.setdefault("retry_base_seconds", get_default("image_generation_retry_base_seconds", 1.0))
    generator.setdefault("max_retry_delay_seconds", get_default("image_generation_max_retry_delay_seconds", 30.0))
    generator.setdefault("result_max_bytes", get_default("image_generation_result_max_bytes", 25 * 1024 * 1024))
    generator.setdefault("allow_private_download_hosts", get_default("image_download_allow_private_hosts", False))
    image_provider_id, image_model = _image_generator_identity(generator)
    image_provider_id = image_provider_id or provider_id
    image_model = image_model or requested_model
    try:
        results = await generate_images(generator, prompt=prompt, model=generator.get("model") or None, n=body.get("n", 1), size=body.get("size"), quality=body.get("quality"), background=body.get("background"), output_format=body.get("output_format"), extra={k: v for k, v in body.items() if k not in {"model", "prompt", "n", "size", "quality", "background", "output_format"}})
    except Exception as exc:
        username = user.get("username", "legacy")
        api_key_value = api_key.get("key", "")
        details = {
            "request_kind": "image_generation", "responses_mode": "image_generation",
            "upstream_endpoint": "images/generations", "image_model": image_model,
            "image_backend_provider": image_provider_id, "image_backend_model": image_model,
            "image_backend_type": str(generator.get("backend_type") or ""), "image_fallback_status": "unused",
            "image_count": 0, "image_bytes": 0, "error_message": error_detail_for_log(exc),
        }
        _log_request(username, api_key_value, image_model, image_provider_id, "images_generations", False, 0, requested_model, details=details)
        _record_request_log(
            endpoint="images_generations", username=username, api_key_value=api_key_value,
            requested_model=requested_model, final_model=image_model,
            final_provider=image_provider_id, request_body=body, success=False,
            status="fail", tokens=0, details=details, error_message=error_detail_for_log(exc),
        )
        _record_success_metrics(username, api_key_value, 0, "fail")
        _error_log.error("[images_generations] FAILED: %s", error_detail_for_log(exc))
        raise HTTPException(status_code=client_status_for_upstream_error(exc), detail=friendly_error_msg(exc)) from exc
    data = [{"b64_json": item.data_uri.split(",", 1)[1], "mime_type": item.mime_type} for item in results]
    details = {"request_kind": "image_generation", "responses_mode": "image_generation", "upstream_endpoint": "images/generations", "image_model": image_model, "image_backend_provider": image_provider_id, "image_backend_model": image_model, "image_backend_type": str(generator.get("backend_type") or ""), "image_fallback_status": "unused", "image_count": len(results), "image_bytes": image_results_bytes(results)}
    username = user.get("username", "legacy")
    api_key_value = api_key.get("key", "")
    _log_request(username, api_key_value, image_model, image_provider_id, "images_generations", True, 0, requested_model, details=details)
    _record_request_log(
        endpoint="images_generations", username=username, api_key_value=api_key_value,
        requested_model=requested_model, final_model=image_model,
        final_provider=image_provider_id, request_body=body,
        response_body={"created": True, "image_count": len(results)},
        success=True, status="ok", tokens=0, details=details,
    )
    _record_success_metrics(username, api_key_value, 0, "ok")
    return {"created": int(time.time()), "data": data}


@router.post("/images/generations")
async def images_generations_endpoint(request: Request, authorization: Optional[str] = Header(None)):
    return await _images_generation_request(request, authorization)


@router.get("/image-results/{token}")
async def image_result_endpoint(token: str):
    """Serve a generated image through its unguessable capability token."""
    result = find_image_result(token)
    if result is None:
        raise HTTPException(status_code=404, detail="Generated image not found or expired")
    return FileResponse(
        result.path,
        media_type=result.mime_type,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )


# -- Request/Response detail log recorder --
_PAYLOAD_MAX_BYTES = 64 * 1024
_STREAMED_TEXT_MAX = 16 * 1024
_STREAMED_REASONING_MAX = 16 * 1024
_STREAMED_TOOL_MAX = 8


def _redact_log_payload(value):
    fields = get_default(
        "request_log_redact_fields",
        ["api_key", "authorization", "cookie", "password", "secret", "token"],
    )
    blocked = {str(field).casefold() for field in fields} if isinstance(fields, list) else set()
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if str(key).casefold() in blocked else _redact_log_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_log_payload(item) for item in value]
    return value


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
    stream=None,
    usage=None,
    success,
    status,
    tokens,
    details=None,
    partial_output=False,
    error_message=None,
    log_id=None,
    generation_started_at=None,
    request_started_at=None,
):
    capture_payloads = bool(get_default("request_log_capture_payloads", True))
    payload_details = _normalized_request_details(endpoint, details)
    if isinstance(request_body, dict):
        payload_details.update(_thinking_fields_from_payload(request_body))
    if request_started_at is not None:
        payload_details["duration_ms"] = max(0, round((time.monotonic() - request_started_at) * 1000))
    if generation_started_at is not None:
        generation_elapsed = max(0.0, time.monotonic() - generation_started_at)
        payload_details["generation_ms"] = round(generation_elapsed * 1000)
        completion_tokens = (
            usage.get("completion_tokens") or usage.get("output_tokens", 0)
            if isinstance(usage, dict) else 0
        )
        try:
            completion_tokens = max(0, int(completion_tokens))
        except (TypeError, ValueError):
            completion_tokens = 0
        payload_details["completion_tokens"] = completion_tokens
        payload_details["tps"] = round(completion_tokens / generation_elapsed, 2) if generation_elapsed > 0 else None
    if usage and 'usage' not in payload_details:
        payload_details['usage'] = usage
    # Some OpenAI-compatible providers expose output_tokens while the
    # adapter's normalized completion_tokens is still zero. Prefer the
    # provider usage value when it is available, including over a stale 0.
    if isinstance(usage, dict):
        duration_ms = payload_details.get("generation_ms") or payload_details.get("duration_ms")
        try:
            duration_s = max(0.0, int(duration_ms or 0) / 1000)
            usage_completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            completion_tokens = max(0, int(usage_completion_tokens or payload_details.get("completion_tokens") or 0))
        except (TypeError, ValueError):
            duration_s = 0.0
            completion_tokens = 0
        if duration_s > 0 and (usage_completion_tokens or payload_details.get("tps") in (None, 0, 0.0)):
            payload_details["completion_tokens"] = completion_tokens
            payload_details["tps"] = round(completion_tokens / duration_s, 2)
    if capture_payloads and streamed_text is not None:
        compact = _compact_text(streamed_text, _STREAMED_TEXT_MAX)
        payload_details['streamed_text'] = compact
        if compact != streamed_text:
            payload_details['streamed_text_truncated'] = True
    if capture_payloads and streamed_reasoning is not None:
        compact = _compact_text(streamed_reasoning, _STREAMED_REASONING_MAX)
        payload_details['streamed_reasoning'] = compact
        if compact != streamed_reasoning:
            payload_details['streamed_reasoning_truncated'] = True
    if capture_payloads and streamed_tool_calls is not None:
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
    if capture_payloads:
        request_body = _redact_log_payload(request_body)
        final_response_body = _redact_log_payload(final_response_body)
    else:
        request_body = {"_omitted": True, "reason": "request_log_capture_payloads=false"}
        final_response_body = {"_omitted": True, "reason": "request_log_capture_payloads=false"}
    try:
        writer = update_request_log if log_id else add_request_log
        writer_kwargs = dict(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            endpoint=endpoint,
            username=username or '',
            api_key=mask_key(api_key_value or ''),
            requested_model=requested_model or '',
            model=final_model or '',
            provider=final_provider or '',
            status=status,
            stream=(stream if stream is not None else response_body is None and (streamed_text is not None or streamed_reasoning is not None or streamed_tool_calls is not None)),
            tokens=int(tokens or 0),
            request_body=_truncate_payload(request_body),
            response_body=_truncate_payload(final_response_body),
            details=payload_details,
            error=(error_message or ''),
        )
        if log_id:
            writer(int(log_id), **writer_kwargs)
            written_id = int(log_id)
        else:
            written_id = writer(**writer_kwargs)
    except Exception as exc:
        _app_log.warning('add_request_log failed: %s', exc)
        written_id = int(log_id or 0)
    # 裁剪与过期清理已移到后台周期任务（run_storage_maintenance），
    # 不再把全表扫描式 DELETE 放在每个请求的写路径上（P7）。
    return written_id

def _build_stream_recorder(
    endpoint,
    username,
    api_key_value,
    requested_model,
    request_body,
    request_started_at=None,
):
    def _record(**payload):
        _record_request_log(
            endpoint=endpoint,
            username=username,
            api_key_value=api_key_value,
            requested_model=requested_model,
            final_model=payload.get('final_model') or '',
            final_provider=payload.get('final_provider_id') or '',
            generation_started_at=payload.get('generation_started_at'),
            request_started_at=payload.get('request_started_at') or request_started_at,
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
            log_id=payload.get('log_id'),
        )
    return _record
