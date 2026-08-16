import json
import copy
import hashlib
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any, Optional

import anyio
from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import FileResponse, StreamingResponse
from app.database import (
    get_providers, find_user_by_api_key,
    increment_global_stats, increment_image_generation_stats, increment_user_usage, get_db,
    parse_model_id, add_request_record, add_request_log, update_request_log, trim_request_logs, get_enabled_preprocessor,
    get_enabled_image_generator, get_model_image_generation,
    get_model_responses_capability, set_model_responses_capability, update_model_responses_capability, update_model_responses_tool_types,
)
from app.core.text import friendly_error_msg, mask_key
from app.core.image_intent import is_image_generation_intent, latest_user_text
from app.core.image_bridge import (
    GATEWAY_IMAGE_ASSET_MARKER,
    GATEWAY_IMAGE_DISPLAY_CALL_PREFIX,
    GATEWAY_IMAGE_RESULT_MARKER,
    IMAGE_BRIDGE_TOOL_NAME,
    IMAGE_BRIDGE_CORRECTION_INSTRUCTIONS,
    configure_internal_image_bridge,
    gateway_generated_image_asset_context,
    has_codex_generated_image_exec_tool,
    has_codex_image_function_tool,
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
    image_preview_data_uri,
    store_image_results,
)
from app.core.image_batch import image_invocation_cache
from app.core.outcome import (
    apply_outcome_to_details,
    routing_details_from_policy,
    stats_counters_for_status,
    is_client_disconnect_error,
)
from app.core.output import InternalOutputEvent, InternalOutputMessage, InternalToolCallOutput
from app.core.types import InternalMessage, text_part, tool_call_part, tool_result_part

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
from app.adapters.responses import iter_sse_frames, post_native_response, split_sse_frame, sse_payload, stream_native_response
from app.adapters.imagegen import generate_images, image_results_bytes
from app.protocols.egress import (
    render_anthropic_message,
    render_chat_completion,
    render_completion,
    render_response, render_responses_image_generation, render_responses_image_generation_sse,
    render_responses_sse,
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
_RESPONSES_CAPABILITY_PROBE_MARKER = "auto_probe_v2"

router = APIRouter()

# Rolling log of recent requests for the admin stats dashboard
_request_log = deque(maxlen=get_default("request_log_max", 200))
_request_log_lock = threading.Lock()


def _responses_client_owned_tool_markers(body: dict) -> list[str]:
    """Identify Codex-owned Responses tools that Chat rewrite cannot preserve faithfully."""
    found: list[str] = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "")
        name = str(tool.get("name") or "")
        if tool_type == "custom" and name and f"custom:{name}" not in found:
            found.append(f"custom:{name}")
        elif tool_type == "namespace" and name and f"namespace:{name}" not in found:
            found.append(f"namespace:{name}")
    input_data = body.get("input")
    if isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict) or item.get("type") != "additional_tools":
                continue
            for tool in item.get("tools") or []:
                if not isinstance(tool, dict):
                    continue
                tool_type = str(tool.get("type") or "")
                name = str(tool.get("name") or "")
                if tool_type == "custom" and name and f"custom:{name}" not in found:
                    found.append(f"custom:{name}")
                elif tool_type == "namespace" and name and f"namespace:{name}" not in found:
                    found.append(f"namespace:{name}")
    return found


def _responses_requires_native(body: dict) -> list[str]:
    """Return request features that cannot be faithfully represented by Chat."""
    required = []
    chat_safe_fields = {
        "model", "input", "instructions", "tools", "tool_choice", "stream",
        "temperature", "top_p", "presence_penalty", "frequency_penalty", "stop",
        "user", "previous_response_id", "provider_id", "parallel_tool_calls",
        "max_output_tokens", "max_completion_tokens",
        # Responses metadata/options that have a reasonable IR/Chat
        # compatibility equivalent (or can safely be ignored by the adapter).
        # These must not force an unsupported provider down the native-only
        # path; Codex commonly sends them on every request.
        "reasoning", "text", "store", "metadata", "truncation", "include",
        "background", "service_tier", "safety_identifier", "prompt_cache_key",
        "prompt_cache_retention", "max_tool_calls", "top_logprobs", "logprobs",
        "client_metadata",
    }
    for field, value in body.items():
        if field not in chat_safe_fields and value not in (None, False, "", [], {}):
            required.append(field)
    # Hosted Responses tools are intentionally filtered by ingress when a
    # provider uses the compatibility path.  They must not turn an otherwise
    # compatible Codex request into a native-only request.
    chat_tool_types = {"function", "custom", "namespace", "web_search"}
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type") not in chat_tool_types:
            required.append(f"tool:{tool.get('type') or 'unknown'}")
    return required


def _responses_image_generation_tool(body: dict) -> dict | None:
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type") == "image_generation":
            return tool
    for item in body.get("input") or []:
        if not isinstance(item, dict) or item.get("type") != "additional_tools":
            continue
        for tool in item.get("tools") or []:
            if isinstance(tool, dict) and tool.get("type") == "image_generation":
                return tool
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "image_generation":
        return choice
    return None


def _responses_is_system_turn(body: dict) -> bool:
    """Return whether Codex identified this as an app-owned background turn.

    Codex creates auxiliary Responses requests for task titles and other UI
    metadata.  Their wrapped user prompt can mention image generation even
    though the request itself must only produce structured metadata.  The
    client marks these requests with ``thread_source=system``; use that
    explicit signal instead of guessing from prompt wording or schemas.
    """
    metadata = body.get("client_metadata")
    if not isinstance(metadata, dict):
        return False
    turn_metadata = metadata.get("x-codex-turn-metadata")
    if isinstance(turn_metadata, str):
        try:
            turn_metadata = json.loads(turn_metadata)
        except (TypeError, ValueError):
            return False
    return bool(
        isinstance(turn_metadata, dict)
        and str(turn_metadata.get("thread_source") or "").lower() == "system"
    )


def _responses_image_prompt(input_data: Any, instructions: Any = "") -> str:
    """Extract the current user request without forwarding conversation history."""
    prompt = latest_user_text(input_data).strip()
    if prompt:
        return prompt
    return str(instructions or "").strip()


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
            if not resolved.get("extra_headers"):
                resolved["extra_headers"] = image_provider.get("extra_headers") or {}
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
    _app_log.info(
        "[responses image_generation] backend=%s model=%s params=%s",
        generator.get("provider_id") or "-",
        generator.get("model") or "-",
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
            image_model=generator.get("model") or generator.get("provider_model") or "",
            image_count=0,
            image_bytes=0,
            attempted_provider=generator.get("provider_id") or "",
        )
        raise
    except Exception as exc:
        _error_log.exception("[responses image_generation] failed: %s", exc)
        error = HTTPException(status_code=502, detail=friendly_error_msg(exc))
        _attach_request_details(
            error,
            request_kind="image_generation",
            responses_mode="image_generation",
            upstream_endpoint="images/generations",
            image_model=generator.get("model") or generator.get("provider_model") or "",
            image_count=0,
            image_bytes=0,
            attempted_provider=generator.get("provider_id") or "",
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
        "backend": generator.get("provider_id") or "",
        "model": generator.get("model") or generator.get("provider_model") or "",
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

    cached = await anyio.to_thread.run_sync(claim.future.result)
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
                sum(item.path.stat().st_size for item in outcome.stored),
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
                batch_id, index + 1, len(invocations), outcome.duration_ms, friendly_error_msg(exc),
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


def _generated_image_asset_manifest(artifacts: list[dict[str, str]]) -> str:
    """Build a compact, model-readable handoff that survives conversation history."""
    if not artifacts:
        return ""
    lines = [
        GATEWAY_IMAGE_ASSET_MARKER,
        "Generated image originals are available as project assets:",
    ]
    for index, artifact in enumerate(artifacts, start=1):
        lines.append(
            f"{index}. `{artifact['filename']}` ({artifact['mime_type']}): "
            f"[download original]({artifact['url']})"
        )
    lines.extend([
        "For coding or design tasks, download these URLs into the project workspace with a "
        "terminal command before continuing, verify the files exist, and reference those files "
        "from the project. The images are stored by the gateway, not in the agent workspace.",
        "Do not claim image generation is unavailable and do not recreate these same assets "
        "with PIL, SVG, Canvas, or CSS unless the user explicitly requests a replacement.",
    ])
    return "\n".join(lines)


def _generated_image_markdown_output(
    image_results,
    stored: list[StoredImageResult],
    artifacts: list[dict[str, str]],
    usage: dict | None = None,
) -> InternalOutputMessage:
    """Publish inline data images with HTTP download links as a fallback.

    Older Codex-compatible clients do not advertise the generatedImage helper.
    Their Markdown renderer recognizes image syntax but may refuse remote HTTP
    image loads, leaving a blank thumbnail. A data URI keeps rendering local to
    the client while the stored artifact URL remains available for opening or
    downloading the original image.
    """
    # This is public assistant text. Private bridge markers and agent-only
    # instructions must never be rendered in the user's conversation.
    asset_links = [
        f"[`{artifact['filename']}` — download original]({artifact['url']})"
        for artifact in artifacts
    ]
    blocks = ["Original: " + " · ".join(asset_links)] if asset_links else []
    inline_limit = max(1, int(get_default("image_preview_inline_limit", 4)))
    inline_results = image_results[:inline_limit]
    for index, result in enumerate(inline_results, start=1):
        label = "Generated image" if len(stored) == 1 else f"Generated image {index}"
        blocks.append(f"![{label}]({image_preview_data_uri(result)})")
    omitted = len(image_results) - len(inline_results)
    if omitted > 0:
        blocks.append(
            f"{omitted} additional generated image preview(s) were omitted to keep the "
            "response small; all originals are listed above."
        )
    return InternalOutputMessage(
        text="\n\n".join(block for block in blocks if block),
        finish_reason="stop",
        usage=dict(usage or {}),
    )


def _generated_image_exec_output(
    image_results,
    artifacts: list[dict[str, str]],
    usage: dict | None = None,
) -> InternalOutputMessage:
    """Ask Codex's client-owned exec runtime to publish native image results."""
    display_source = "\n".join(
        "generatedImage({ image_url: %s, output_hint: %s });" % (
            json.dumps(str(result.data_uri or ""), ensure_ascii=False),
            json.dumps("The generated image has already been displayed to the user.", ensure_ascii=False),
        )
        for result in image_results
    )
    manifest = _generated_image_asset_manifest(artifacts)
    source = display_source
    if manifest:
        # Keep the project handoff in the tool history rather than ordinary
        # assistant text. Codex executes only generatedImage(); the trailing
        # block comment is recovered by the gateway on the next tool round
        # and injected into private system context for the model.
        source = f"{display_source}\n/*\n{manifest}\n*/"
    suffix = uuid.uuid4().hex
    return InternalOutputMessage(
        text="",
        tool_calls=[InternalToolCallOutput(
            id=f"ctc_gateway_image_display_{suffix}",
            call_id=f"{GATEWAY_IMAGE_DISPLAY_CALL_PREFIX}{suffix}",
            name="exec",
            arguments=json.dumps({"input": source}, ensure_ascii=False),
        )],
        finish_reason="tool_calls",
        usage=dict(usage or {}),
    )


def _generated_image_client_output(
    body: dict,
    image_results,
    stored: list[StoredImageResult],
    artifacts: list[dict[str, str]],
    usage: dict | None = None,
) -> tuple[InternalOutputMessage, str]:
    """Select Codex-native display when advertised, with Markdown as fallback."""
    if has_codex_generated_image_exec_tool(body):
        return _generated_image_exec_output(image_results, artifacts, usage), "codex_exec_generated_image"
    output = _generated_image_markdown_output(image_results, stored, artifacts, usage)
    return output, "assistant_message"


def _merge_image_bridge_output(
    planner_output: InternalOutputMessage,
    image_output: InternalOutputMessage,
) -> InternalOutputMessage:
    """Replace private image bridge calls without dropping client-owned work."""
    replacement_calls = list(image_output.tool_calls)
    merged_calls: list[InternalToolCallOutput] = []
    replacement_inserted = False
    for call in planner_output.tool_calls:
        is_private_image_call = call.name == IMAGE_BRIDGE_TOOL_NAME
        is_wrapped_image_call = call.name == "exec" and bool(
            image_call_arguments_from_exec(call.arguments)
        )
        if is_private_image_call or is_wrapped_image_call:
            if not replacement_inserted:
                merged_calls.extend(replacement_calls)
                replacement_inserted = True
            continue
        merged_calls.append(call)

    if replacement_calls and not replacement_inserted:
        merged_calls.extend(replacement_calls)

    text_parts = [part for part in (planner_output.text, image_output.text) if part]
    return InternalOutputMessage(
        role=planner_output.role,
        text="\n\n".join(text_parts),
        reasoning=planner_output.reasoning,
        tool_calls=merged_calls,
        finish_reason="tool_calls" if merged_calls else image_output.finish_reason,
        # image_output carries the aggregate usage from the initial planner
        # and every continuation round. planner_output may contain only the
        # final round and must not replace that total.
        usage=dict(image_output.usage or planner_output.usage),
    )


def _append_image_bridge_results(
    internal,
    invocations: list[tuple[InternalToolCallOutput, dict[str, Any], list[dict[str, str]]]],
    *,
    failed: list[tuple[InternalToolCallOutput, dict[str, Any], str]] | None = None,
) -> None:
    """Add gateway-executed image calls and compact results to the model history."""
    call_parts = []
    failed = failed or []
    for call, arguments, _ in invocations:
        call_parts.append(tool_call_part(
            call.call_id or call.id,
            call.name,
            arguments,
            raw_arguments=call.arguments,
        ))
    for call, arguments, _ in failed:
        call_parts.append(tool_call_part(
            call.call_id or call.id,
            call.name,
            arguments,
            raw_arguments=call.arguments,
        ))
    internal.messages.append(InternalMessage(role="assistant", parts=call_parts))
    for call, _, artifacts in invocations:
        if artifacts:
            summary = _generated_image_asset_manifest(artifacts)
        else:
            summary = (
                "This image was already generated and displayed earlier in the current task. "
                "Continue without regenerating it."
            )
        internal.messages.append(InternalMessage(
            role="tool",
            parts=[tool_result_part(call.call_id or call.id, [text_part(summary)])],
        ))
    for call, _, error_message in failed:
        summary = (
            "Image generation failed for this invocation after gateway retries. "
            f"Error: {error_message}. Continue the task using successful assets and retry only "
            "this failed prompt if it is still required; do not regenerate successful prompts."
        )
        internal.messages.append(InternalMessage(
            role="tool",
            parts=[tool_result_part(call.call_id or call.id, [text_part(summary)])],
        ))


def _image_bridge_invocations(
    output: InternalOutputMessage,
) -> list[tuple[InternalToolCallOutput, dict[str, Any]]]:
    invocations = []
    for call in output.tool_calls:
        if call.name == IMAGE_BRIDGE_TOOL_NAME:
            invocations.append((call, image_call_arguments(call.arguments)))
            continue
        if call.name == "exec":
            wrapped_arguments = image_call_arguments_list_from_exec(call.arguments)
            if len(wrapped_arguments) == 1:
                invocations.append((call, wrapped_arguments[0]))
                continue
            for index, wrapped_args in enumerate(wrapped_arguments, start=1):
                base_id = call.call_id or call.id or "exec_image"
                synthetic_id = f"{base_id}_image_{index}"
                invocations.append((InternalToolCallOutput(
                    id=synthetic_id,
                    call_id=synthetic_id,
                    name=IMAGE_BRIDGE_TOOL_NAME,
                    arguments=json.dumps(wrapped_args, ensure_ascii=False),
                    raw=call.raw,
                ), wrapped_args))
    return invocations


def _image_prompt_key(arguments: dict[str, Any]) -> str:
    prompt = " ".join(str(arguments.get("prompt") or "").lower().split())
    return json.dumps({
        "prompt": prompt,
        "size": arguments.get("size"),
        "quality": arguments.get("quality"),
        "background": arguments.get("background"),
        "output_format": arguments.get("output_format"),
    }, sort_keys=True, ensure_ascii=False)


def _responses_required_tool_types(body: dict) -> set[str]:
    return {str(tool.get("type") or "") for tool in body.get("tools") or [] if isinstance(tool, dict) and tool.get("type")}


def _responses_stateful_tool_markers(body: dict) -> list[str]:
    """Identify prior Responses tool/agent state that cannot cross providers safely."""
    input_data = body.get("input")
    found = []
    if body.get("previous_response_id"):
        found.append("previous_response_id")
    if not isinstance(input_data, list):
        return found
    marker_types = {
        "custom_tool_call_output",
        "function_call_output",
        "computer_call_output",
    }
    for item in input_data:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type in marker_types and item_type not in found:
            found.append(item_type)
    return found


def _observed_response_tool_types(response: dict) -> set[str]:
    observed = set()
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type == "custom_tool_call":
            observed.add("custom")
        elif item_type == "function_call":
            observed.add("namespace" if item.get("namespace") else "function")
        elif item_type.endswith("_call"):
            observed.add(item_type[:-5])
    return observed


class _EmptyNativeResponsesError(RuntimeError):
    """The upstream completed a Responses request without client-usable output."""

    native_empty_output = True


def _native_response_has_output(response: dict | None) -> bool:
    """Return whether a completed Responses payload contains usable output items."""
    if not isinstance(response, dict):
        return False
    output = response.get("output")
    if not isinstance(output, list):
        return bool(str(response.get("output_text") or "").strip())
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type in {"message", "function_call", "custom_tool_call", "computer_call", "image_generation_call"}:
            return True
        if item_type.endswith("_call") or item_type in {"reasoning", "output_text"}:
            return True
    return False


def _native_sse_payload_has_output(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    event_type = str(payload.get("type") or "")
    if event_type in {
        "response.output_item.added", "response.output_item.done",
        "response.content_part.added", "response.content_part.done",
        "response.output_text.delta", "response.output_text.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.computer_call.delta",
    }:
        item = payload.get("item") or payload.get("output_item") or {}
        if event_type.endswith(".delta") or event_type.endswith(".done"):
            return True
        return isinstance(item, dict) and bool(str(item.get("type") or ""))
    return False


def _native_empty_output_error(response: dict | None = None) -> _EmptyNativeResponsesError:
    error = _EmptyNativeResponsesError("native Responses completed without client-visible output")
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
    upstream_endpoint = "responses"
    try:
        async for frame in iter_sse_frames(events):
            payload = sse_payload(frame)
            saw_output = saw_output or _native_sse_payload_has_output(payload)
            if payload and payload.get("type") in {"response.completed", "response.failed", "response.incomplete"}:
                response_body = payload.get("response")
                terminal_error = payload.get("error") or (response_body or {}).get("error")
                failed = payload.get("type") != "response.completed"
                if not failed and not saw_output and not _native_response_has_output(response_body):
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
                        expires_at=_responses_capability_expiry("supported"),
                    )
            if payload and payload.get("type") == "response.output_item.done":
                observed = _observed_response_tool_types({"output": [payload.get("item") or {}]})
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
        success = bool(response_body) and not failed and saw_output
        response_id = (response_body or {}).get("id")
        if success and response_id and remember_response_chain_key and conv_key:
            remember_response_chain_key(response_id, conv_key)
        client_disconnected = locals().get("client_disconnected", False)
        details = {**routing_details_from_policy(policy), "responses_mode": "native", "upstream_endpoint": upstream_endpoint, "stream": True, "fallback_attempts": fallback_attempts or []}
        if len(fallback_attempts or []) > 1:
            details.update({"fallback_status": "used", "attempt_index": len(fallback_attempts) - 1})
        if client_disconnected:
            details.update({"status": "cancelled", "client_disconnected": True})
        elif not saw_output and response_body and not client_disconnected:
            details.update({"native_empty_output": True, "native_failure_reason": "empty_completed_response"})
        elif failed and response_body:
            details.update({"status": "partial", "partial_output": True})
        details = apply_outcome_to_details(details, success=success, partial_output=bool(response_body) and failed)
        _log_request(username, api_key_value, model, provider_id, "responses", success, tokens, requested_model, details=details)
        status = details.get("status", "ok" if success else "fail")
        error_text = str(locals().get("terminal_error") or "native Responses stream did not complete") if not success else ""
        _record_request_log(endpoint="responses", username=username, api_key_value=api_key_value, requested_model=requested_model, final_model=model, final_provider=provider_id, request_body=request_body, response_body=response_body, success=success, status=status, tokens=tokens, usage=usage, details=details, error_message=error_text)
        _record_success_metrics(username, api_key_value, tokens, status)


def _responses_capability_is_fresh(capability: dict | None) -> bool:
    if not capability or not capability.get("responses_expires_at"):
        return False
    try:
        return datetime.fromisoformat(capability["responses_expires_at"]) > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return False


def _responses_capability_expiry(status: str) -> str:
    ttl_key = {
        "supported": "responses_capability_supported_ttl",
        "unsupported": "responses_capability_unsupported_ttl",
    }.get(status, "responses_capability_transient_ttl")
    fallback = 604800 if status == "supported" else 21600 if status == "unsupported" else 300
    return (datetime.now(timezone.utc) + timedelta(seconds=max(0, int(get_default(ttl_key, fallback))))).isoformat()


def _mark_model_responses_unknown(provider_id: str, model: str, error: Exception | str = "") -> None:
    """Invalidate native capability after a transient upstream failure."""
    set_model_responses_capability(
        provider_id, model, status="unknown",
        expires_at=_responses_capability_expiry("transient"),
        error=str(error)[:500],
    )


def _native_error_is_explicitly_unsupported(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    if status in {404, 405, 501}:
        return True
    if status not in {400, 422}:
        return False
    try:
        detail = response.text.lower()
    except Exception:
        detail = str(exc).lower()
    mentions_responses = any(marker in detail for marker in ("/responses", "responses api", "response api", "responses endpoint", "native responses"))
    rejects_protocol = any(marker in detail for marker in (
        "not supported", "unsupported", "not implemented", "unknown endpoint",
        "method not allowed", "unprocessable", "invalid request",
    ))
    if status == 422:
        return mentions_responses or rejects_protocol or not detail.strip()
    return mentions_responses and rejects_protocol


def _native_downgrade_details(exc: Exception, attempts: list[dict] | None = None) -> dict:
    """Describe a failed native attempt that was completed through compatibility."""
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    details = {
        "responses_mode": "compatibility_downgrade",
        "native_attempted": True,
        "native_failure_endpoint": "responses",
        "native_failure_reason": classify_upstream_error(exc),
        "native_failure_message": friendly_error_msg(exc),
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


def _native_response_target_supported(target: RouteTarget, *, stream: bool, required_tool_types: set[str], is_primary: bool) -> tuple[dict | None, str]:
    provider = resolve_provider(target.model, target.provider_id)
    if not provider or provider.get("provider_type") != "openai":
        return None, ""
    capability = get_model_responses_capability(provider.get("id") or target.provider_id, target.model)
    # Only a fresh explicit negative result prevents a real user request from
    # attempting native Responses. Unknown, expired, and transient results are
    # deliberately request-driven rechecks.
    if _responses_capability_is_fresh(capability) and capability.get("responses_status") == "unsupported":
        return None, ""
    # ``responses_tool_types`` is learned from successful response output.  It is
    # therefore positive evidence, not an exhaustive declaration of what an
    # upstream can do.  Treating an absent entry as unsupported prevented a newly
    # discovered native fallback from ever handling Codex tools (the generic
    # capability probe deliberately does not execute tools).
    #
    # Keep accepting ``required_tool_types`` here so callers document why they
    # selected native dispatch; explicit negative capability data can be added
    # later without changing this boundary.
    del stream, required_tool_types, is_primary
    return provider, provider_for_log(provider, target.provider_id)


async def _probe_model_responses_capability(provider: dict, model: str) -> bool:
    """Probe native Responses support and cache the result per provider/model.

    A normal OpenAI-compatible provider is not proof of Responses support.  The
    probe intentionally requires a valid Responses object; HTTP 4xx responses
    (including llama.cpp's unsupported endpoint response) are cached as a
    negative capability, while transient 5xx/network failures remain unknown.
    """
    provider_id = str(provider.get("id") or "")
    probe = responses_to_internal({
        "model": model,
        "input": "capability probe",
        "stream": False,
    })
    probe.target_model = model
    probe.provider_id = provider_id
    try:
        payload = await post_native_response(provider, probe)
        supported = isinstance(payload, dict) and payload.get("object") == "response"
        if supported:
            set_model_responses_capability(
                provider_id, model, status="supported",
                streaming=False, streaming_status="unknown",
                error=_RESPONSES_CAPABILITY_PROBE_MARKER,
                expires_at=_responses_capability_expiry("supported"),
            )
        return supported
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
        # A 400/422 from the probe means this upstream accepted the endpoint
        # but rejected the minimal Responses protocol request.  For Chat-only
        # or incomplete Responses proxies this is the practical unsupported
        # signal.  Other 4xx responses are only negative when the body
        # explicitly identifies the Responses endpoint/protocol as unsupported.
        explicitly_unsupported = _native_error_is_explicitly_unsupported(exc)
        if status in {400, 422} or explicitly_unsupported:
            set_model_responses_capability(
                provider_id, model, status="unsupported",
                expires_at=_responses_capability_expiry("unsupported"),
                error=friendly_error_msg(exc),
            )
        return False


async def _native_capability_for_request(provider: dict | None, model: str) -> bool:
    if not provider or provider.get("provider_type") != "openai":
        return False
    provider_id = str(provider.get("id") or "")
    capability = get_model_responses_capability(provider_id, model)
    if _responses_capability_is_fresh(capability):
        status = capability.get("responses_status")
        if status == "supported":
            # Legacy optimistic supported rows are revalidated once.
            return capability.get("responses_error") == _RESPONSES_CAPABILITY_PROBE_MARKER
        if status == "unsupported":
            return False
    # Unknown/degraded rows deliberately probe immediately; their TTL is a
    # backoff marker, not permission to send native Responses.
    return await _probe_model_responses_capability(provider, model)


async def _wait_for_native_response_event(events) -> bytes:
    """Buffer initial SSE keepalives until an actual Responses lifecycle event."""
    buffered = b""
    while True:
        chunk = await events.__anext__()
        buffered += chunk
        while (split := split_sse_frame(buffered)) is not None:
            frame, _rest = split
            for line in frame.splitlines():
                if not line.startswith(b"data:"):
                    continue
                try:
                    payload = json.loads(line[5:].lstrip().decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if str(payload.get("type") or "").startswith("response."):
                    return buffered
            buffered = _rest


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
            saw_output = saw_output or _native_sse_payload_has_output(payload)
            if saw_output:
                return buffered
            if payload:
                event_type = str(payload.get("type") or "")
                if event_type in {"response.failed", "response.incomplete"}:
                    error = RuntimeError("native Responses stream ended unsuccessfully")
                    _attach_request_details(error, native_failure_reason=event_type)
                    raise error
                if event_type == "response.completed":
                    response = payload.get("response")
                    if not saw_output and not _native_response_has_output(response):
                        raise _native_empty_output_error(response)
                    return buffered
            buffered = rest


async def _native_response_with_fallbacks(internal, *, stream: bool, required_tool_types: set[str], stateful_markers: list[str] | None = None):
    """Retry only native-capable fallback targets before any client output."""
    primary = RouteTarget(model=internal.target_model, provider_id=internal.provider_id)
    primary = RouteTarget(model=primary.model, provider_id=_fallback_provider_id_for_target(primary))
    targets = [primary]
    stateful_markers = list(stateful_markers or [])
    # Capability mismatch is local routing information, rather than an upstream
    # failure.  Still consult the configured fallback chain: otherwise an
    # advanced Responses request is rejected before it can reach a compatible
    # fallback provider.  An empty trigger intentionally ignores error-trigger
    # gates because no upstream request has been made yet.
    primary_provider, _primary_provider_id = _native_response_target_supported(
        primary, stream=stream, required_tool_types=required_tool_types, is_primary=True,
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
        provider, provider_id = _native_response_target_supported(target, stream=stream, required_tool_types=required_tool_types, is_primary=index == 0)
        if provider is None:
            attempts.append({"index": index, "stage": "primary" if index == 0 else "fallback", "target": target.model, "provider_id": target.provider_id, "status": "skipped", "reason": "capability_mismatch"})
            index += 1
            continue
        attempt = copy.deepcopy(internal)
        attempt.target_model, attempt.provider_id = target.model, provider_id
        try:
            if stream:
                # Do not yield until the first chunk: this preserves the existing
                # stream fallback invariant.
                events = stream_native_response(provider, attempt)
                first = await _wait_for_native_response_output(events)
                async def prefixed():
                    yield first
                    async for chunk in events:
                        yield chunk
                attempts.append({"index": index, "stage": "primary" if index == 0 else "fallback", "target": target.model, "provider_id": provider_id, "status": "success"})
                return prefixed(), target, provider_id, attempts
            response = await post_native_response(provider, attempt)
            if not _native_response_has_output(response):
                raise _native_empty_output_error(response)
            attempts.append({"index": index, "stage": "primary" if index == 0 else "fallback", "target": target.model, "provider_id": provider_id, "status": "success"})
            set_model_responses_capability(
                provider_id, target.model, status="supported",
                streaming=False, streaming_status="unknown",
                error=_RESPONSES_CAPABILITY_PROBE_MARKER,
                expires_at=_responses_capability_expiry("supported"),
            )
            return response, target, provider_id, attempts
        except Exception as exc:
            last_exc = exc
            is_empty_native = bool(getattr(exc, "native_empty_output", False))
            is_protocol_unsupported = _native_error_is_explicitly_unsupported(exc) or (
                getattr(exc, "response", None) is not None
                and getattr(exc.response, "status_code", None) in {400, 422}
            )
            if is_protocol_unsupported:
                set_model_responses_capability(provider_id, target.model, status="unsupported", expires_at=_responses_capability_expiry("unsupported"), error=friendly_error_msg(exc))
            else:
                _mark_model_responses_unknown(provider_id, target.model, exc)
            attempts.append({"index": index, "stage": "primary" if index == 0 else "fallback", "target": target.model, "provider_id": provider_id, "status": "failed", "trigger": classify_upstream_error(exc), "error": friendly_error_msg(exc)})
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
                    if stateful_markers:
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
    if getattr(output, "raw", None) is None:
        output.raw = {}
    if not isinstance(output.raw, dict):
        return
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
    _attach_output_request_details(
        output,
        upstream_endpoint=_upstream_endpoint_for_provider(provider_info),
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
    if not timeout_s or timeout_s <= 0:
        async for event in events:
            yield event
        return

    agen = events.__aiter__()
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


async def _call_nonstream_with_fallbacks(policy, internal, *, temperature, max_tokens, log_label: str):
    original_model = internal.target_model
    original_provider = internal.provider_id
    last_exc = None
    primary = RouteTarget(model=original_model, provider_id=original_provider)
    primary = RouteTarget(model=primary.model, provider_id=_fallback_provider_id_for_target(primary))
    targets = [primary]
    fallback_attempts = []
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
    for index, target in enumerate(targets):
        internal.target_model = target.model
        internal.provider_id = target.provider_id
        fallback_provider_id = _fallback_provider_id_for_target(target)
        try:
            output, provider_info, adapter_provider_id = await _await_with_attempt_timeout(
                _call_nonstream_target(
                    target,
                    internal,
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
                index=index,
                stage="primary",
                target=target,
                provider_id=fallback_provider_id,
                status="success",
            ))
            _attach_output_request_details(
                output,
                fallback_status="unused",
                attempt_index=index,
                fallback_attempts=fallback_attempts,
                upstream_endpoint=_upstream_endpoint_for_provider(provider_info),
            )
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
                friendly_error_msg(last_exc) if last_exc else "",
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


def _normalized_request_details(endpoint: str, details: dict | None) -> dict:
    """Promote image-generation requests to a stable logging dimension."""
    normalized = dict(details or {})
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
            "error_message": friendly_error_msg(exc),
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
        error_message=friendly_error_msg(exc),
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
        "image_count",
        "image_bytes",
        "image_artifact_count",
        "native_attempted",
        "native_failure_endpoint",
        "native_failure_status",
        "native_failure_reason",
        "native_failure_message",
        "native_attempts",
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
    authorization: Optional[str] = Header(None),
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
    user, api_key = verify_api_key(authorization, endpoint="chat_completions")

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
                    base_details=routing_details_from_policy(policy),
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
        success_details = _finalize_success_details(output, policy=policy)
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
    except HTTPException:
        raise
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
    user, api_key = verify_api_key(authorization, endpoint="completions")

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
                    base_details=routing_details_from_policy(policy),
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
        success_details = _finalize_success_details(output, policy=policy)
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
    user, api_key = verify_api_key(authorization, endpoint="messages")

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
                    remember_reasoning_content=_remember_reasoning_content,
                    base_details=routing_details_from_policy(policy),
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
        success_details = _finalize_success_details(output, policy=policy)
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
    user, api_key = verify_api_key(authorization, endpoint="responses")

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
        internal.messages.insert(0, InternalMessage(
            role="system",
            parts=[text_part(
                "Gateway-generated project assets from the current user task follow. "
                "These URLs are available even if the original display message is removed during "
                "conversation normalization. Download and use the originals before continuing; "
                "do not regenerate or replace them merely because they are not yet in the local "
                f"workspace.\n\n{image_asset_context}"
            )],
        ))
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
    image_tool = _responses_image_generation_tool(body)
    explicit_image_choice = isinstance(body.get("tool_choice"), dict) and body["tool_choice"].get("type") == "image_generation"
    image_enabled = bool(provider_info and get_model_image_generation(adapter_provider_id, model))
    system_turn = _responses_is_system_turn(body)
    image_request_intent = is_image_generation_intent(input_data, instructions)
    # Sub2API leaves Codex's client-owned image_gen namespace intact.  The
    # first Responses turn must therefore return a namespaced function_call;
    # Codex will then call /images/generations itself.  Do not replace this
    # protocol with the gateway's synthetic image_generation_call response.
    codex_image_tool = has_codex_image_function_tool(body)
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
        prompt = _responses_image_prompt(input_data, instructions)
        image_results, generator = await _generate_with_configured_backend(prompt, {
            "n": image_tool.get("n") or body.get("n"),
            "size": image_tool.get("size") or body.get("size"),
            "quality": image_tool.get("quality") or body.get("quality"),
            "background": image_tool.get("background") or body.get("background"),
            "output_format": image_tool.get("output_format") or body.get("output_format"),
        })
        image_model = generator.get("model") or generator.get("provider_model") or ""
        details = {**routing_details_from_policy(policy), "request_kind": "image_generation", "responses_mode": "image_generation", "upstream_endpoint": "images/generations", "image_model": image_model, "image_count": len(image_results), "image_bytes": image_results_bytes(image_results)}
        if stream:
            # Streaming requests return before the body iterator runs. Record
            # the completed backend operation here so the admin statistics do
            # not lose successful image requests.
            _log_request(username, api_key_value, model, adapter_provider_id, "responses", True, 0, requested_model, details=details)
            _record_request_log(endpoint="responses", username=username, api_key_value=api_key_value, requested_model=requested_model, final_model=model, final_provider=adapter_provider_id, request_body=body, response_body={"status": "completed", "output_count": len(image_results)}, success=True, status="ok", tokens=0, usage={}, details={**details, "stream": True}, stream=True)
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
        _log_request(username, api_key_value, model, adapter_provider_id, "responses", True, 0, requested_model, details=details)
        _record_request_log(
            endpoint="responses", username=username, api_key_value=api_key_value,
            requested_model=requested_model, final_model=model,
            final_provider=adapter_provider_id, request_body=body,
            response_body={"status": "completed", "output_count": len(image_results)},
            success=True, status="ok", tokens=0, usage={}, details=details,
        )
        _record_success_metrics(username, api_key_value, 0, "ok")
        return rendered

    # Match sub2api's public transform, but project the hosted tool to an
    # internal function because this gateway's configured image backend is
    # separate from the chat provider. Keep the capability on display-result
    # follow-up turns: a project can require several distinct assets over
    # multiple Codex tool rounds. Idempotency and the per-request invocation
    # budget prevent duplicate loops; merely having one result in history must
    # not disable the remaining task's image capability.
    # Model capability is not user intent.  Injecting the private bridge for
    # every request made to an image-capable model causes ordinary Responses
    # traffic (including code/tool turns) to be routed through the image
    # planner and makes its logs look like image-generation requests.
    # Require an explicit image request or an explicit hosted image tool; the
    # two follow-up markers preserve an already-started gateway image flow.
    should_bridge_image = (
        image_request_intent
        or image_tool is not None
        or image_display_followup
        or bool(image_asset_context)
    )
    if (
        image_enabled
        and should_bridge_image
        and not system_turn
        and not has_codex_image_function_tool(body)
    ):
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
                log_label="responses.image_bridge",
            )
            configure_internal_image_bridge(internal, body)
            output, provider_info, adapter_provider_id = await _call_nonstream_with_fallbacks(
                policy, internal, temperature=temperature, max_tokens=max_tokens,
                log_label="responses.image_bridge",
            )
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
            requested_image_invocations = _image_bridge_invocations(output)
            max_image_invocations = 8
            image_invocations = requested_image_invocations[:max_image_invocations]
            skipped_initial_invocations = requested_image_invocations[max_image_invocations:]
            if skipped_initial_invocations:
                _app_log.warning(
                    "[responses image_generation.batch_limited] requested=%d allowed=%d",
                    len(requested_image_invocations), max_image_invocations,
                )
            _app_log.info(
                "[responses image_generation.planner_calls] total=%d image=%d names=%s",
                len(output.tool_calls), len(image_invocations),
                [call.name for call in output.tool_calls],
            )
            # The bridge advertises an optional capability. Never replace the
            # model's chosen text or ordinary tool call with a forced image
            # invocation. Besides overriding agent intent, that behavior made
            # unrelated tasks fail when the global image backend was disabled.
            image_correction_applied = False
            all_initial_calls_are_images = len(requested_image_invocations) == len(output.tool_calls)
            # An explicit image request must not silently degrade into a text
            # completion when the model ignores the advertised image tool.
            # Ordinary image discussion never reaches this branch because
            # ``image_bridge`` is gated by ``image_request_intent`` above.
            # Client-owned Codex image tools are also excluded: their tool
            # loop belongs to the client, not this hosted bridge.
            if (
                image_request_intent
                and not requested_image_invocations
                and not output.tool_calls
                and not codex_image_tool
                and not has_codex_generated_image_exec_tool(body)
                and not image_display_followup
                and not image_already_generated
                and not system_turn
                and (isinstance(input_data, str) or not isinstance(input_data, list) or not any(
                    isinstance(item, dict) and item.get("role") == "assistant"
                    for item in input_data
                ))
            ):
                correction_message = InternalMessage(
                    role="system",
                    parts=[text_part(IMAGE_BRIDGE_CORRECTION_INSTRUCTIONS)],
                )
                internal.messages.append(correction_message)
                internal.tool_choice = {
                    "type": "function",
                    "function": {"name": IMAGE_BRIDGE_TOOL_NAME},
                }
                allowed = internal.extra.setdefault("allowed_openai_params", [])
                if "tool_choice" not in allowed:
                    allowed.append("tool_choice")
                image_correction_applied = True
                _app_log.warning(
                    "[responses image_generation.correction] no image invocation; "
                    "forcing bridge tool choice model=%s provider=%s",
                    internal.target_model, adapter_provider_id or "-",
                )
                correction_output, provider_info, adapter_provider_id = await _call_nonstream_with_fallbacks(
                    policy, internal, temperature=temperature, max_tokens=max_tokens,
                    log_label="responses.image_bridge.correction",
                )
                correction_invocations = _image_bridge_invocations(correction_output)
                if correction_invocations:
                    output = correction_output
                    bridge_upstream_details = _merge_bridge_request_details(
                        bridge_upstream_details, _output_request_details(correction_output),
                    )
                    bridge_final_model = _target_model_for_log(
                        RouteTarget(model=internal.target_model, provider_id=adapter_provider_id),
                        adapter_provider_id,
                    )
                    bridge_final_provider = adapter_provider_id
                    requested_image_invocations = correction_invocations
                    image_invocations = correction_invocations[:max_image_invocations]
                    skipped_initial_invocations = correction_invocations[max_image_invocations:]
                    all_initial_calls_are_images = len(correction_invocations) == len(correction_output.tool_calls)
                    # Continue through the existing backend execution path.
                else:
                    correction_error = HTTPException(
                        status_code=502,
                        detail="The model did not invoke the image-generation tool",
                    )
                    # The correction is a second planner request. Preserve
                    # the first request's authoritative fallback history and
                    # final target, otherwise a failed correction is logged as
                    # if it only touched the primary model.
                    failure_details = {
                        **bridge_upstream_details,
                        "attempted_model": bridge_final_model,
                        "attempted_provider": bridge_final_provider,
                        "request_kind": "image_generation",
                        "responses_mode": "image_generation_failed",
                        "upstream_endpoint": "images/generations",
                        "image_count": 0,
                        "image_failed_count": 1,
                        "image_correction_applied": True,
                        "error_message": "model did not invoke image generation tool after correction",
                    }
                    _attach_request_details(
                        correction_error, **failure_details,
                    )
                    raise correction_error
            if image_invocations:
                image_results = bridge_image_results
                stored_images = bridge_stored_images
                image_artifacts: list[dict[str, str]] = []
                used_asset_filenames: set[str] = set()
                completed_invocations = []
                failed_invocations = []
                generator = {}
                prompt_chars = 0
                image_failure_attempt_count = 0
                image_retried_count = 0
                image_reused_count = 0
                unresolved_failed_keys: set[str] = set()
                image_invocation_attempt_count = 0
                configured_generator = _resolved_image_generator(get_enabled_image_generator() or {})
                image_model = configured_generator.get("model") or configured_generator.get("provider_model") or ""
                image_provider = configured_generator.get("provider_id") or adapter_provider_id
                running_details = {
                    **routing_details_from_policy(policy),
                    "request_kind": "image_generation",
                    "responses_mode": "model_driven_image_generation_running",
                    "upstream_endpoint": "images/generations",
                    "image_model": image_model,
                    "image_requested_count": len(image_invocations),
                    "image_succeeded_count": 0,
                    "image_failed_count": 0,
                    "image_count": 0,
                    "image_bytes": 0,
                    "image_artifact_count": 0,
                    "status": "running",
                }
                image_request_log_id = _record_request_log(
                    endpoint="responses", username=username, api_key_value=api_key_value,
                    requested_model=requested_model, final_model=model,
                    final_provider=image_provider, request_body=body,
                    response_body=None, success=True, status="running", tokens=0,
                    details=running_details, stream=stream,
                )

                def record_image_progress(batch_id, outcomes, total):
                    succeeded = [item for item in outcomes if item.error is None]
                    failed = [item for item in outcomes if item.error is not None]
                    progress_details = {
                        **running_details,
                        "image_batch_id": batch_id,
                        "image_completed_count": len(outcomes),
                        "image_requested_count": total,
                        "image_succeeded_count": len(succeeded),
                        "image_failed_count": len(failed),
                        "image_artifact_count": sum(len(item.stored) for item in succeeded),
                        "image_retried_count": sum(max(0, item.backend_attempts - 1) for item in succeeded),
                        "image_reused_count": sum(1 for item in succeeded if item.reused),
                    }
                    _record_request_log(
                        endpoint="responses", username=username, api_key_value=api_key_value,
                        requested_model=requested_model, final_model=model,
                        final_provider=image_provider, request_body=body,
                        response_body=None, success=True, status="running", tokens=0,
                        details=progress_details, stream=stream, log_id=image_request_log_id,
                    )

                initial_outcomes = await _execute_image_invocations(
                    body,
                    username=username,
                    api_key_value=api_key_value,
                    invocations=image_invocations,
                    progress=record_image_progress,
                )
                image_invocation_attempt_count += len(initial_outcomes)
                for outcome in initial_outcomes:
                    call, args = outcome.call, outcome.arguments
                    prompt = str(args.get("prompt") or latest_user_text(input_data) or "")
                    prompt_chars += len(prompt)
                    if outcome.error is not None:
                        image_failure_attempt_count += 1
                        unresolved_failed_keys.add(_image_prompt_key(args))
                        failed_invocations.append((call, args, friendly_error_msg(outcome.error)))
                        continue
                    generator = outcome.generator
                    image_retried_count += max(0, outcome.backend_attempts - 1)
                    image_reused_count += 1 if outcome.reused else 0
                    invocation_results = await anyio.to_thread.run_sync(
                        partial(
                            generation_results_from_stored,
                            outcome.stored,
                            size=args.get("size"), quality=args.get("quality"),
                            output_format=args.get("output_format"), background=args.get("background"),
                        )
                    )
                    stored_images.extend(item for item in outcome.stored if item not in stored_images)
                    invocation_artifacts = _stored_image_artifacts(
                        request, outcome.stored, arguments=args,
                        start_index=len(image_artifacts) + 1,
                        used_filenames=used_asset_filenames,
                    )
                    image_results.extend(invocation_results)
                    image_artifacts.extend(invocation_artifacts)
                    completed_invocations.append((call, args, invocation_artifacts))
                    unresolved_failed_keys.discard(_image_prompt_key(args))
                if not completed_invocations:
                    first_error = next((item.error for item in initial_outcomes if item.error), None)
                    if first_error is None:
                        first_error = RuntimeError("image batch returned no successful images")
                    _attach_request_details(
                        first_error,
                        request_kind="image_generation",
                        responses_mode="model_driven_image_generation_failed",
                        upstream_endpoint="images/generations",
                        image_model=image_model,
                        attempted_provider=image_provider,
                        image_requested_count=len(image_invocations),
                        image_succeeded_count=0,
                        image_failed_count=len(failed_invocations),
                        image_count=0,
                        image_bytes=0,
                    )
                    raise first_error
                image_model = generator.get("model") or generator.get("provider_model") or ""
                bridge_image_model = image_model
                planner_tokens = output.usage.get("total_tokens", 0)
                tokens = planner_tokens
                details = {
                    **routing_details_from_policy(policy),
                    "request_kind": "image_generation",
                    "responses_mode": "model_driven_image_generation",
                    "upstream_endpoint": "images/generations",
                    "image_model": image_model,
                    "image_count": len(image_results),
                    "image_bytes": image_results_bytes(image_results),
                    "planner_tokens": planner_tokens,
                    "image_invocation_count": len(completed_invocations),
                    "image_requested_count": len(image_invocations),
                    "image_succeeded_count": len(completed_invocations),
                    "image_failed_count": len(unresolved_failed_keys),
                    "image_failure_attempt_count": image_failure_attempt_count,
                    "image_retried_count": image_retried_count,
                    "image_reused_count": image_reused_count,
                    "image_correction_applied": image_correction_applied,
                }
                _app_log.info(
                    "[responses image_generation.model_invoked] invocations=%d prompt_chars=%d image_model=%s",
                    len(completed_invocations), prompt_chars, image_model,
                )
                continuation_tokens = 0
                continuation_usage: dict[str, int] = {}
                continuation_error = ""
                planner_output = output
                if not has_codex_generated_image_exec_tool(body) and all_initial_calls_are_images:
                    _append_image_bridge_results(internal, [
                        *completed_invocations,
                        *((call, args, []) for call, args in skipped_initial_invocations),
                    ], failed=failed_invocations)
                    generated_prompt_keys = {
                        _image_prompt_key(args) for _, args, _ in completed_invocations
                    }
                    continuation = InternalOutputMessage()
                    max_continuation_rounds = 4
                    force_without_image_tool = False
                    for continuation_round in range(1, max_continuation_rounds + 1):
                        try:
                            continuation, provider_info, adapter_provider_id = await _call_nonstream_with_fallbacks(
                                policy, internal, temperature=temperature, max_tokens=max_tokens,
                                log_label="responses.image_bridge.continuation",
                            )
                        except Exception as exc:
                            continuation_error = friendly_error_msg(exc)
                            continuation = InternalOutputMessage(
                                text=(
                                    "Generated image assets are available, but the agent continuation "
                                    "failed. Continue the task in the next turn using the listed originals."
                                ),
                                finish_reason="stop",
                            )
                            _app_log.warning(
                                "[responses image_generation.continuation_failed] round=%d images=%d error=%s",
                                continuation_round, len(image_results), continuation_error,
                            )
                            break
                        for key, value in continuation.usage.items():
                            continuation_usage[key] = continuation_usage.get(key, 0) + int(value or 0)
                        continuation_tokens = continuation_usage.get("total_tokens", 0)
                        pending_images = _image_bridge_invocations(continuation)
                        if not pending_images:
                            break
                        fresh_images = [
                            (call, args) for call, args in pending_images
                            if _image_prompt_key(args) not in generated_prompt_keys
                        ]
                        remaining_image_budget = max_image_invocations - len(completed_invocations)
                        if remaining_image_budget <= 0:
                            fresh_images = []
                            force_without_image_tool = len(pending_images) == len(continuation.tool_calls)
                        else:
                            fresh_images = fresh_images[:remaining_image_budget]
                        _app_log.info(
                            "[responses image_generation.continuation_images] round=%d requested=%d fresh=%d",
                            continuation_round, len(pending_images), len(fresh_images),
                        )
                        if not fresh_images:
                            _append_image_bridge_results(
                                internal, [(call, args, []) for call, args in pending_images]
                            )
                            if len(pending_images) < len(continuation.tool_calls):
                                break
                            if force_without_image_tool:
                                break
                            if continuation_round == max_continuation_rounds:
                                force_without_image_tool = True
                                break
                            continue
                        round_results = []
                        round_completed = []
                        round_failed = []

                        def record_continuation_progress(batch_id, outcomes, total):
                            current_success = sum(1 for item in outcomes if item.error is None)
                            current_failed = sum(1 for item in outcomes if item.error is not None)
                            progress_details = {
                                **running_details,
                                "image_batch_id": batch_id,
                                "image_completed_count": image_invocation_attempt_count + len(outcomes),
                                "image_requested_count": len(image_invocations) + len(fresh_images),
                                "image_succeeded_count": len(completed_invocations) + current_success,
                                "image_failed_count": len(unresolved_failed_keys) + current_failed,
                                "image_artifact_count": len(stored_images) + sum(
                                    len(item.stored) for item in outcomes if item.error is None
                                ),
                                "image_retried_count": image_retried_count + sum(
                                    max(0, item.backend_attempts - 1)
                                    for item in outcomes if item.error is None
                                ),
                                "image_reused_count": image_reused_count + sum(
                                    1 for item in outcomes if item.error is None and item.reused
                                ),
                            }
                            _record_request_log(
                                endpoint="responses", username=username, api_key_value=api_key_value,
                                requested_model=requested_model, final_model=model,
                                final_provider=image_provider, request_body=body,
                                response_body=None, success=True, status="running", tokens=0,
                                details=progress_details, stream=stream, log_id=image_request_log_id,
                            )

                        round_outcomes = await _execute_image_invocations(
                            body,
                            username=username,
                            api_key_value=api_key_value,
                            invocations=fresh_images,
                            progress=record_continuation_progress,
                        )
                        image_invocation_attempt_count += len(round_outcomes)
                        for outcome in round_outcomes:
                            call, args = outcome.call, outcome.arguments
                            prompt = str(args.get("prompt") or latest_user_text(input_data) or "")
                            prompt_chars += len(prompt)
                            if outcome.error is not None:
                                image_failure_attempt_count += 1
                                unresolved_failed_keys.add(_image_prompt_key(args))
                                round_failed.append((call, args, friendly_error_msg(outcome.error)))
                                continue
                            generator = outcome.generator
                            image_retried_count += max(0, outcome.backend_attempts - 1)
                            image_reused_count += 1 if outcome.reused else 0
                            invocation_results = await anyio.to_thread.run_sync(
                                partial(
                                    generation_results_from_stored,
                                    outcome.stored,
                                    size=args.get("size"), quality=args.get("quality"),
                                    output_format=args.get("output_format"), background=args.get("background"),
                                )
                            )
                            stored_images.extend(item for item in outcome.stored if item not in stored_images)
                            invocation_artifacts = _stored_image_artifacts(
                                request, outcome.stored, arguments=args,
                                start_index=len(image_artifacts) + 1,
                                used_filenames=used_asset_filenames,
                            )
                            round_results.extend(invocation_results)
                            image_artifacts.extend(invocation_artifacts)
                            round_completed.append((call, args, invocation_artifacts))
                            generated_prompt_keys.add(_image_prompt_key(args))
                            unresolved_failed_keys.discard(_image_prompt_key(args))
                        image_results.extend(round_results)
                        completed_invocations.extend(round_completed)
                        image_model = generator.get("model") or generator.get("provider_model") or image_model
                        bridge_image_model = image_model
                        completed_call_ids = {call.call_id or call.id for call, _, _ in round_completed}
                        failed_call_ids = {call.call_id or call.id for call, _, _ in round_failed}
                        skipped_invocations = [
                            (call, args, []) for call, args in pending_images
                            if (call.call_id or call.id) not in completed_call_ids | failed_call_ids
                        ]
                        _append_image_bridge_results(
                            internal, [*round_completed, *skipped_invocations], failed=round_failed
                        )
                        if len(pending_images) < len(continuation.tool_calls):
                            break
                        if len(completed_invocations) >= max_image_invocations:
                            force_without_image_tool = True
                            break
                        if continuation_round == max_continuation_rounds:
                            force_without_image_tool = True
                    if force_without_image_tool:
                        internal.tools = [
                            tool for tool in internal.tools if tool.name != IMAGE_BRIDGE_TOOL_NAME
                        ]
                        remaining_tools = internal.chat_tools()
                        if remaining_tools:
                            internal.extra["tools"] = remaining_tools
                        else:
                            internal.extra.pop("tools", None)
                        try:
                            continuation, provider_info, adapter_provider_id = await _call_nonstream_with_fallbacks(
                                policy, internal, temperature=temperature, max_tokens=max_tokens,
                                log_label="responses.image_bridge.continuation.final",
                            )
                            bridge_upstream_details = _merge_bridge_request_details(
                                bridge_upstream_details, _output_request_details(continuation),
                            )
                            bridge_final_model = _target_model_for_log(
                                RouteTarget(model=internal.target_model, provider_id=adapter_provider_id),
                                adapter_provider_id,
                            )
                            bridge_final_provider = adapter_provider_id
                            for key, value in continuation.usage.items():
                                continuation_usage[key] = continuation_usage.get(key, 0) + int(value or 0)
                            continuation_tokens = continuation_usage.get("total_tokens", 0)
                        except Exception as exc:
                            continuation_error = friendly_error_msg(exc)
                            continuation = InternalOutputMessage(
                                text=(
                                    "Generated image assets are available, but the agent continuation "
                                    "failed. Continue the task in the next turn using the listed originals."
                                ),
                                finish_reason="stop",
                            )
                            _app_log.warning(
                                "[responses image_generation.continuation_failed] stage=final images=%d error=%s",
                                len(image_results), continuation_error,
                            )
                    planner_output = continuation
                    continuation.tool_calls = [
                        call for call in continuation.tool_calls
                        if call.name != IMAGE_BRIDGE_TOOL_NAME
                        and not (call.name == "exec" and image_call_arguments_from_exec(call.arguments))
                    ]
                    _app_log.info(
                        "[responses image_generation.continuation] text_chars=%d tool_calls=%d tokens=%d",
                        len(continuation.text or ""), len(continuation.tool_calls), continuation_tokens,
                    )
                combined_usage = {
                    key: int(output.usage.get(key, 0) or 0) + int(continuation_usage.get(key, 0) or 0)
                    for key in set(output.usage) | set(continuation_usage)
                }
                tokens = combined_usage.get("total_tokens", planner_tokens + continuation_tokens)
                image_output, display_mode = await anyio.to_thread.run_sync(
                    _generated_image_client_output,
                    body, image_results, stored_images, image_artifacts, combined_usage,
                )
                image_output = _merge_image_bridge_output(planner_output, image_output)
                details = {
                    **details,
                    **bridge_upstream_details,
                    "responses_mode": f"model_driven_image_generation_{display_mode}",
                    "image_artifact_count": len(stored_images),
                    "continuation_tokens": continuation_tokens,
                    "image_count": len(image_results),
                    "image_bytes": image_results_bytes(image_results),
                    "image_invocation_count": len(completed_invocations),
                    "image_invocation_attempt_count": image_invocation_attempt_count,
                    "image_requested_count": len(image_invocations),
                    "image_succeeded_count": len(completed_invocations),
                    "image_failed_count": len(unresolved_failed_keys),
                    "image_failure_attempt_count": image_failure_attempt_count,
                    "image_retried_count": image_retried_count,
                    "image_reused_count": image_reused_count,
                    "image_continuation_error": continuation_error,
                    "image_correction_applied": image_correction_applied,
                }
                request_status = (
                    "degraded"
                    if (
                        image_failure_attempt_count > 0
                        or image_retried_count > 0
                        or unresolved_failed_keys
                        or continuation_error
                    )
                    else "ok"
                )
                details["status"] = request_status
                details = apply_outcome_to_details(details, success=True)
                request_status = details.get("status", request_status)
                _app_log.info(
                    "[responses image_generation.assistant_message] images=%d artifact_bytes=%d",
                    len(stored_images),
                    sum(item.path.stat().st_size for item in stored_images),
                )
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
                        base_details={**routing_details_from_policy(policy), **_output_request_details(output), "responses_mode": "image_bridge_model_passthrough"},
                        previous_response_id=previous_response_id, conv_key=conv_key,
                        remember_response_chain_key=_remember_response_chain_key,
                        remember_reasoning_content=_remember_reasoning_content,
                        tool_only_turns=_tool_only_turns, render_extra=internal.extra,
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
            _record_success_metrics(username, api_key_value, tokens, "ok")
            return rendered

        native_required = _responses_requires_native(body)
        required_tool_types = _responses_required_tool_types(body)
        stateful_markers = _responses_stateful_tool_markers(body)
        capability = get_model_responses_capability(adapter_provider_id, model) if provider_info else None
        native_downgrade_details = {}
        native_supported = await _native_capability_for_request(provider_info, model)
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
                observed = _observed_response_tool_types(rendered)
                if observed:
                    capability = get_model_responses_capability(adapter_provider_id, model) or {}
                    update_model_responses_tool_types(adapter_provider_id, model, list(set(capability.get("responses_tool_types") or []) | observed))
                native_details = apply_outcome_to_details({**routing_details_from_policy(policy), "responses_mode": "native", "upstream_endpoint": "responses", "fallback_attempts": native_attempts}, success=True)
                status = native_details.get("status", "ok")
                _log_request(username, api_key_value, model, adapter_provider_id, "responses", True, tokens, requested_model, details=native_details)
                _record_request_log(endpoint="responses", username=username, api_key_value=api_key_value, requested_model=requested_model, final_model=model, final_provider=adapter_provider_id, request_body=body, response_body=rendered, success=True, status=status, tokens=tokens, usage=usage, details=native_details)
                _record_success_metrics(username, api_key_value, tokens, status)
                return rendered
            except Exception as native_error:
                native_attempts = list(getattr(native_error, "request_details", {}).get("fallback_attempts", []) or [])
                client_owned_tools = _responses_client_owned_tool_markers(body)
                if native_required or client_owned_tools:
                    # Codex custom/namespace tools and other native-only
                    # features must not silently fall back to Chat.  The Chat
                    # adapter can still serve ordinary text/function requests.
                    if getattr(native_error, "native_capability_unavailable", False) or _native_error_is_explicitly_unsupported(native_error):
                        raise HTTPException(status_code=422, detail=(
                            "No configured provider supports native Responses required by this request: "
                            + ", ".join(native_required or client_owned_tools)
                        )) from native_error
                    raise
                _app_log.warning("[responses native fallback] no native target succeeded; downgrading basic request: %s", native_error)
                native_downgrade_details = _native_downgrade_details(native_error, native_attempts)

        # The initial minimal policy deliberately leaves a native payload untouched.
        # Once native dispatch is ruled out, run the full IR policy required by the
        # Chat/Anthropic compatibility adapters.
        policy = await prepare_request_policy(
            internal, username=username, api_key_value=api_key_value,
            preprocess_request=_policy_preprocess_request, conversation_cache_key=_conversation_cache_key,
            reasoning_context=_reasoning_context if isinstance(input_data, list) else None,
            log_label="responses",
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
                    base_details={**routing_details_from_policy(policy), **native_downgrade_details},
                    previous_response_id=previous_response_id,
                    conv_key=conv_key,
                    remember_response_chain_key=_remember_response_chain_key,
                    remember_reasoning_content=_remember_reasoning_content,
                    tool_only_turns=_tool_only_turns,
                    render_extra=internal.extra,
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
        rendered = render_response(output, model=model, previous_response_id=previous_response_id, response_id=resp_id, extra=internal.extra)
        success_details = _finalize_success_details(
            output, policy=policy,
            extra={"response_id": resp_id, **native_downgrade_details},
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
            tokens=0, details=details, error_message=friendly_error_msg(e),
            log_id=image_request_log_id or None,
        )
        increment_global_stats(success=False, stateful_fallback_blocked=bool(details.get("stateful_fallback_blocked")))
        if username != "legacy":
            increment_user_usage(username, api_key_value, False, 0)
        _error_log.error("FAILED: %s", str(e))
        raise HTTPException(status_code=500, detail=friendly_error_msg(e))


async def _images_generation_request(request: Request, authorization: Optional[str]):
    user, api_key = verify_api_key(authorization, endpoint="images_generations")
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
    generator.setdefault("max_retries", get_default("image_generation_max_retries", 2))
    generator.setdefault("retry_base_seconds", get_default("image_generation_retry_base_seconds", 1.0))
    generator.setdefault("max_retry_delay_seconds", get_default("image_generation_max_retry_delay_seconds", 30.0))
    generator.setdefault("result_max_bytes", get_default("image_generation_result_max_bytes", 25 * 1024 * 1024))
    generator.setdefault("allow_private_download_hosts", get_default("image_download_allow_private_hosts", False))
    image_provider_id = str(generator.get("provider_id") or provider_id or "")
    image_model = str(generator.get("model") or generator.get("provider_model") or "")
    try:
        results = await generate_images(generator, prompt=prompt, model=generator.get("model") or None, n=body.get("n", 1), size=body.get("size"), quality=body.get("quality"), background=body.get("background"), output_format=body.get("output_format"), extra={k: v for k, v in body.items() if k not in {"model", "prompt", "n", "size", "quality", "background", "output_format"}})
    except Exception as exc:
        username = user.get("username", "legacy")
        api_key_value = api_key.get("key", "")
        details = {
            "request_kind": "image_generation", "responses_mode": "image_generation",
            "upstream_endpoint": "images/generations", "image_model": image_model,
            "image_count": 0, "image_bytes": 0, "error_message": friendly_error_msg(exc),
        }
        _log_request(username, api_key_value, requested_model, image_provider_id, "images_generations", False, 0, requested_model, details=details)
        _record_request_log(
            endpoint="images_generations", username=username, api_key_value=api_key_value,
            requested_model=requested_model, final_model=requested_model,
            final_provider=image_provider_id, request_body=body, success=False,
            status="fail", tokens=0, details=details, error_message=friendly_error_msg(exc),
        )
        _record_success_metrics(username, api_key_value, 0, "fail")
        raise HTTPException(status_code=502, detail=friendly_error_msg(exc)) from exc
    data = [{"b64_json": item.data_uri.split(",", 1)[1], "mime_type": item.mime_type} for item in results]
    details = {"request_kind": "image_generation", "responses_mode": "image_generation", "upstream_endpoint": "images/generations", "image_model": image_model, "image_count": len(results), "image_bytes": image_results_bytes(results)}
    username = user.get("username", "legacy")
    api_key_value = api_key.get("key", "")
    _log_request(username, api_key_value, requested_model, image_provider_id, "images_generations", True, 0, requested_model, details=details)
    _record_request_log(
        endpoint="images_generations", username=username, api_key_value=api_key_value,
        requested_model=requested_model, final_model=requested_model,
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
):
    capture_payloads = bool(get_default("request_log_capture_payloads", True))
    payload_details = _normalized_request_details(endpoint, details)
    if usage and 'usage' not in payload_details:
        payload_details['usage'] = usage
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
    try:
        trim_request_logs(get_default('request_log_max', 500))
    except Exception as exc:
        _app_log.warning('trim_request_logs failed: %s', exc)
    return written_id

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
