"""Protocol-agnostic orchestration for the gateway image-generation bridge.

This module lifts the image-bridge execution/continuation loop out of the
``/responses`` endpoint so every protocol endpoint (``/responses``,
``/chat/completions``, ...) can share the same behavior.

The loop itself is protocol-neutral: it operates on the internal
representation (``InternalRequest`` / ``InternalOutputMessage``). Everything
that is endpoint- or client-specific is injected as a callable:

- ``call_model``            -- run one non-streaming model attempt
- ``execute_invocations``   -- run the configured image backend for a batch
- ``build_artifacts``       -- turn stored images into client artifact dicts
- ``render_client_output``  -- render the generated images for THIS client
- ``latest_user_text``      -- the current-turn user prompt (protocol-specific)
- ``record_running``/``on_progress`` -- request-log progress (endpoint-specific)
- ``describe_upstream``/``merge_upstream`` -- fallback/target metadata capture

The pure IR helpers (invocation detection, result appending, output merging,
prompt keying) live here too so both endpoints reuse one implementation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable, Optional

import anyio

from app.core.types import (
    InternalMessage,
    append_system_text,
    text_part,
    tool_call_part,
    tool_result_part,
)
from app.core.output import InternalOutputEvent, InternalOutputMessage, InternalToolCallOutput
from app.core.image_bridge import (
    IMAGE_BRIDGE_TOOL_NAME,
    IMAGE_BRIDGE_CORRECTION_INSTRUCTIONS,
    image_call_arguments,
    image_call_arguments_from_exec,
    image_call_arguments_list_from_exec,
)
from app.core.image_results import StoredImageResult, generation_results_from_stored
from app.adapters.imagegen import image_results_bytes
from app.core.outcome import apply_outcome_to_details
from app.core.text import friendly_error_msg

_log = logging.getLogger("llmgw.app")


def image_generator_identity(generator: dict) -> tuple[str, str]:
    """Return the effective backend identity, not the client's model hint."""
    backend_type = str(generator.get("backend_type") or "existing_model").strip()
    if backend_type == "comfyui":
        return "comfyui", "comfyui"
    return (
        str(generator.get("provider_id") or "").strip(),
        str(generator.get("model") or generator.get("provider_model") or "").strip(),
    )


def image_prompt_key(arguments: dict[str, Any]) -> str:
    prompt = " ".join(str(arguments.get("prompt") or "").lower().split())
    return json.dumps({
        "prompt": prompt,
        "size": arguments.get("size"),
        "quality": arguments.get("quality"),
        "background": arguments.get("background"),
        "output_format": arguments.get("output_format"),
    }, sort_keys=True, ensure_ascii=False)


def image_bridge_invocations(
    output: InternalOutputMessage,
) -> list[tuple[InternalToolCallOutput, dict[str, Any]]]:
    """Extract gateway image invocations (direct + exec-wrapped) from an output."""
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


def generated_image_asset_manifest(artifacts: list[dict[str, str]]) -> str:
    """Build a compact, model-readable handoff that survives conversation history."""
    if not artifacts:
        return ""
    lines = [
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


def append_image_bridge_results(
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
    internal.messages.append(
        _internal_message("assistant", call_parts)
    )
    for call, _, artifacts in invocations:
        if artifacts:
            summary = generated_image_asset_manifest(artifacts)
        else:
            summary = (
                "This image was already generated and displayed earlier in the current task. "
                "Continue without regenerating it."
            )
        internal.messages.append(
            _internal_message("tool",
                              [tool_result_part(call.call_id or call.id, [text_part(summary)])])
        )
    for call, _, error_message in failed:
        summary = (
            "Image generation failed for this invocation after gateway retries. "
            f"Error: {error_message}. Continue the task using successful assets and retry only "
            "this failed prompt if it is still required; do not regenerate successful prompts."
        )
        internal.messages.append(
            _internal_message("tool",
                              [tool_result_part(call.call_id or call.id, [text_part(summary)])])
        )


def _internal_message(role: str, parts: list) -> InternalMessage:
    """Build a fresh IR message for the image-bridge continuation transcript."""
    return InternalMessage(role=role, parts=parts)


def merge_image_bridge_output(
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


def events_to_message(events: list) -> InternalOutputMessage:
    """Aggregate a buffered stream of ``InternalOutputEvent`` into one
    ``InternalOutputMessage`` (text/reasoning/tool_calls/usage/finish).

    Used by the chat streaming image bridge: the upstream stream is buffered
    first, collapsed into a planner message, then handed to
    :func:`run_image_bridge` (which is written against a single
    ``InternalOutputMessage``).
    """
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = "stop"
    usage: dict[str, int] = {}
    tools: dict[int, InternalToolCallOutput] = {}
    for ev in events:
        kind = ev.kind
        if kind == "text_delta" and ev.text:
            text_parts.append(ev.text)
        elif kind == "reasoning_delta" and ev.reasoning:
            reasoning_parts.append(ev.reasoning)
        elif kind in ("tool_call_start", "tool_call_arguments_delta", "tool_call_done"):
            tc = tools.setdefault(
                ev.tool_index,
                InternalToolCallOutput(id=ev.tool_call_id, call_id=ev.call_id, name=ev.name, arguments=""),
            )
            if ev.tool_call_id:
                tc.id = ev.tool_call_id
            if ev.call_id:
                tc.call_id = ev.call_id
            if ev.name:
                tc.name = ev.name
            if kind == "tool_call_arguments_delta":
                tc.arguments += ev.arguments_delta
            elif kind == "tool_call_done" and ev.arguments:
                tc.arguments = ev.arguments
        elif kind in ("message_delta", "message_done"):
            if ev.finish_reason:
                finish_reason = ev.finish_reason
        elif kind == "usage" and ev.usage:
            usage.update(ev.usage)
    return InternalOutputMessage(
        role="assistant",
        text="".join(text_parts),
        reasoning="".join(reasoning_parts),
        tool_calls=[tools[i] for i in sorted(tools)],
        finish_reason=finish_reason,
        usage=usage,
    )


def message_to_events(message: InternalOutputMessage) -> list:
    """Expand one ``InternalOutputMessage`` into an ``InternalOutputEvent``
    stream for the endpoint SSE renderers.

    Used to stream a generated-image result that was produced out-of-band by
    the image bridge (the result is a single message, not a native stream).
    ``usage`` precedes ``message_done`` so renderers that attach a final
    usage block can pick it up.
    """
    events: list[InternalOutputEvent] = []
    events.append(InternalOutputEvent(kind="message_start", role=message.role or "assistant"))
    if message.reasoning:
        events.append(InternalOutputEvent(kind="reasoning_delta", reasoning=message.reasoning))
    if message.text:
        events.append(InternalOutputEvent(kind="text_delta", text=message.text))
    for index, call in enumerate(message.tool_calls):
        events.append(InternalOutputEvent(
            kind="tool_call_start", tool_index=index,
            tool_call_id=call.id, call_id=call.call_id, name=call.name,
        ))
        if call.arguments:
            events.append(InternalOutputEvent(
                kind="tool_call_arguments_delta", tool_index=index,
                tool_call_id=call.id, call_id=call.call_id, name=call.name,
                arguments_delta=call.arguments,
            ))
    if message.usage:
        events.append(InternalOutputEvent(kind="usage", usage=dict(message.usage)))
    events.append(InternalOutputEvent(kind="message_done", finish_reason=message.finish_reason or "stop"))
    return events


@dataclass
class ImageBridgeOutcome:
    """Structured result of a gateway image-bridge run for one request."""

    image_output: InternalOutputMessage
    display_mode: str
    image_results: list
    stored_images: list
    artifacts: list
    tokens: int
    usage: dict
    details: dict
    request_status: str
    bridge_final_model: str
    bridge_final_provider: str
    continuation_error: str = ""
    correction_applied: bool = False


# Type aliases for the injected callables.
_CallModel = Callable[..., Awaitable[tuple[InternalOutputMessage, Any, str]]]
_ExecuteInvocations = Callable[..., Awaitable[list]]
_BuildArtifacts = Callable[[list, dict, int, set], list[dict[str, str]]]
_RenderClientOutput = Callable[..., tuple[InternalOutputMessage, str]]
_LatestUserText = Callable[[], str]
_RecordRunning = Callable[[], int]
_OnProgress = Callable[[dict], None]
_DescribeUpstream = Callable[[InternalOutputMessage, str], tuple[dict, str, str]]
_MergeUpstream = Callable[[dict, dict], dict]


async def run_image_bridge(
    internal,
    *,
    policy,
    model: str,
    temperature: float,
    max_tokens: int,
    base_details: dict,
    running_details: dict,
    configured_generator: dict,
    planner_output: InternalOutputMessage,
    planner_provider_info: Any,
    planner_provider_id: str,
    allow_correction: bool,
    has_client_image_exec_tool: bool,
    call_model: _CallModel,
    execute_invocations: _ExecuteInvocations,
    build_artifacts: _BuildArtifacts,
    render_client_output: _RenderClientOutput,
    latest_user_text: _LatestUserText,
    record_running: _RecordRunning,
    on_progress: _OnProgress,
    describe_upstream: _DescribeUpstream,
    merge_upstream: _MergeUpstream,
    log_label: str = "image_bridge",
) -> Optional[ImageBridgeOutcome]:
    """Run the image-bridge execution/continuation loop.

    Returns an :class:`ImageBridgeOutcome` when at least one image was
    generated, or ``None`` when the model chose not to generate an image
    (the caller then falls through to its normal passthrough rendering).
    """
    output = planner_output
    provider_info = planner_provider_info
    adapter_provider_id = planner_provider_id

    requested_image_invocations = image_bridge_invocations(output)
    max_image_invocations = 8
    image_invocations = requested_image_invocations[:max_image_invocations]
    skipped_initial_invocations = requested_image_invocations[max_image_invocations:]
    if skipped_initial_invocations:
        _log.warning(
            "[image_generation.batch_limited] requested=%d allowed=%d",
            len(requested_image_invocations), max_image_invocations,
        )
    _log.info(
        "[image_generation.planner_calls] total=%d image=%d names=%s",
        len(output.tool_calls), len(image_invocations),
        [call.name for call in output.tool_calls],
    )
    image_correction_applied = False
    all_initial_calls_are_images = len(requested_image_invocations) == len(output.tool_calls)

    bridge_upstream_details, bridge_final_model, bridge_final_provider = describe_upstream(
        output, adapter_provider_id
    )

    if allow_correction and not requested_image_invocations and not output.tool_calls:
        append_system_text(internal.messages, IMAGE_BRIDGE_CORRECTION_INSTRUCTIONS)
        internal.tool_choice = {
            "type": "function",
            "function": {"name": IMAGE_BRIDGE_TOOL_NAME},
        }
        allowed = internal.extra.setdefault("allowed_openai_params", [])
        if "tool_choice" not in allowed:
            allowed.append("tool_choice")
        image_correction_applied = True
        _log.warning(
            "[image_generation.correction] no image invocation; forcing bridge tool choice model=%s provider=%s",
            internal.target_model, adapter_provider_id or "-",
        )
        correction_output, provider_info, adapter_provider_id = await call_model(
            policy, internal, temperature=temperature, max_tokens=max_tokens,
            log_label=f"{log_label}.correction",
        )
        correction_invocations = image_bridge_invocations(correction_output)
        if correction_invocations:
            output = correction_output
            snap = describe_upstream(correction_output, adapter_provider_id)
            bridge_upstream_details = merge_upstream(bridge_upstream_details, snap[0])
            _, bridge_final_model, bridge_final_provider = snap
            requested_image_invocations = correction_invocations
            image_invocations = correction_invocations[:max_image_invocations]
            skipped_initial_invocations = correction_invocations[max_image_invocations:]
            all_initial_calls_are_images = len(correction_invocations) == len(correction_output.tool_calls)
        else:
            from fastapi import HTTPException
            correction_error = HTTPException(
                status_code=502,
                detail="The model did not invoke the image-generation tool",
            )
            _attach_request_details(
                correction_error,
                **{
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
                },
            )
            raise correction_error

    if not image_invocations:
        return None

    image_results = []
    stored_images: list[StoredImageResult] = []
    image_artifacts: list[dict[str, str]] = []
    used_asset_filenames: set[str] = set()
    completed_invocations = []
    failed_invocations = []
    generator = {}
    image_failure_attempt_count = 0
    image_retried_count = 0
    image_reused_count = 0
    unresolved_failed_keys: set[str] = set()
    image_invocation_attempt_count = 0
    image_provider, image_model = image_generator_identity(configured_generator)
    image_provider = image_provider or adapter_provider_id
    image_model = image_model or model
    image_request_log_id = record_running()

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
        on_progress(progress_details)

    initial_outcomes = await execute_invocations(
        invocations=image_invocations, progress=record_image_progress,
    )
    image_invocation_attempt_count += len(initial_outcomes)
    for outcome in initial_outcomes:
        call, args = outcome.call, outcome.arguments
        if outcome.error is not None:
            image_failure_attempt_count += 1
            unresolved_failed_keys.add(image_prompt_key(args))
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
        invocation_artifacts = build_artifacts(
            outcome.stored, args,
            start_index=len(image_artifacts) + 1,
            used_filenames=used_asset_filenames,
        )
        image_results.extend(invocation_results)
        image_artifacts.extend(invocation_artifacts)
        completed_invocations.append((call, args, invocation_artifacts))
        unresolved_failed_keys.discard(image_prompt_key(args))
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
    image_provider, image_model = image_generator_identity(generator)
    planner_tokens = output.usage.get("total_tokens", 0)
    tokens = planner_tokens

    continuation_tokens = 0
    continuation_usage: dict[str, int] = {}
    continuation_error = ""
    planner_output_final = output
    if not has_client_image_exec_tool and all_initial_calls_are_images:
        append_image_bridge_results(
            internal,
            [
                *completed_invocations,
                *((call, args, []) for call, args in skipped_initial_invocations),
            ],
            failed=failed_invocations,
        )
        generated_prompt_keys = {
            image_prompt_key(args) for _, args, _ in completed_invocations
        }
        continuation = InternalOutputMessage()
        max_continuation_rounds = 4
        force_without_image_tool = False
        for continuation_round in range(1, max_continuation_rounds + 1):
            try:
                continuation, provider_info, adapter_provider_id = await call_model(
                    policy, internal, temperature=temperature, max_tokens=max_tokens,
                    log_label=f"{log_label}.continuation",
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
                _log.warning(
                    "[image_generation.continuation_failed] round=%d images=%d error=%s",
                    continuation_round, len(image_results), continuation_error,
                )
                break
            for key, value in continuation.usage.items():
                continuation_usage[key] = continuation_usage.get(key, 0) + int(value or 0)
            continuation_tokens = continuation_usage.get("total_tokens", 0)
            pending_images = image_bridge_invocations(continuation)
            if not pending_images:
                break
            fresh_images = [
                (call, args) for call, args in pending_images
                if image_prompt_key(args) not in generated_prompt_keys
            ]
            remaining_image_budget = max_image_invocations - len(completed_invocations)
            if remaining_image_budget <= 0:
                fresh_images = []
                force_without_image_tool = len(pending_images) == len(continuation.tool_calls)
            else:
                fresh_images = fresh_images[:remaining_image_budget]
            _log.info(
                "[image_generation.continuation_images] round=%d requested=%d fresh=%d",
                continuation_round, len(pending_images), len(fresh_images),
            )
            if not fresh_images:
                append_image_bridge_results(
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
                on_progress(progress_details)

            round_outcomes = await execute_invocations(
                invocations=fresh_images, progress=record_continuation_progress,
            )
            image_invocation_attempt_count += len(round_outcomes)
            for outcome in round_outcomes:
                call, args = outcome.call, outcome.arguments
                if outcome.error is not None:
                    image_failure_attempt_count += 1
                    unresolved_failed_keys.add(image_prompt_key(args))
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
                invocation_artifacts = build_artifacts(
                    outcome.stored, args,
                    start_index=len(image_artifacts) + 1,
                    used_filenames=used_asset_filenames,
                )
                round_results.extend(invocation_results)
                image_artifacts.extend(invocation_artifacts)
                round_completed.append((call, args, invocation_artifacts))
                generated_prompt_keys.add(image_prompt_key(args))
                unresolved_failed_keys.discard(image_prompt_key(args))
            image_results.extend(round_results)
            completed_invocations.extend(round_completed)
            image_model = image_generator_identity(generator)[1]
            completed_call_ids = {call.call_id or call.id for call, _, _ in round_completed}
            failed_call_ids = {call.call_id or call.id for call, _, _ in round_failed}
            skipped_invocations = [
                (call, args, []) for call, args in pending_images
                if (call.call_id or call.id) not in completed_call_ids | failed_call_ids
            ]
            append_image_bridge_results(
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
                continuation, provider_info, adapter_provider_id = await call_model(
                    policy, internal, temperature=temperature, max_tokens=max_tokens,
                    log_label=f"{log_label}.continuation.final",
                )
                snap = describe_upstream(continuation, adapter_provider_id)
                bridge_upstream_details = merge_upstream(bridge_upstream_details, snap[0])
                _, bridge_final_model, bridge_final_provider = snap
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
                _log.warning(
                    "[image_generation.continuation_failed] stage=final images=%d error=%s",
                    len(image_results), continuation_error,
                )
        planner_output_final = continuation
        continuation.tool_calls = [
            call for call in continuation.tool_calls
            if call.name != IMAGE_BRIDGE_TOOL_NAME
            and not (call.name == "exec" and image_call_arguments_from_exec(call.arguments))
        ]
        _log.info(
            "[image_generation.continuation] text_chars=%d tool_calls=%d tokens=%d",
            len(continuation.text or ""), len(continuation.tool_calls), continuation_tokens,
        )

    combined_usage = {
        key: int(output.usage.get(key, 0) or 0) + int(continuation_usage.get(key, 0) or 0)
        for key in set(output.usage) | set(continuation_usage)
    }
    tokens = combined_usage.get("total_tokens", planner_tokens + continuation_tokens)
    image_output, display_mode = render_client_output(
        image_results, stored_images, image_artifacts, combined_usage,
    )
    image_output = merge_image_bridge_output(planner_output_final, image_output)
    details = {
        **base_details,
        **bridge_upstream_details,
        "request_kind": "image_generation",
        "responses_mode": f"model_driven_image_generation_{display_mode}",
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
        "image_artifact_count": len(stored_images),
        "continuation_tokens": continuation_tokens,
        "image_continuation_error": continuation_error,
    }
    # Planner fallback is independent from the configured image backend.
    if "fallback_status" in details:
        details["planner_fallback_status"] = details.pop("fallback_status")
    if "fallback_attempts" in details:
        details["planner_fallback_attempts"] = details.pop("fallback_attempts")
    details["image_fallback_status"] = "unused"
    details["image_backend_provider"] = image_provider
    details["image_backend_model"] = image_model
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
    _log.info(
        "[image_generation.assistant_message] images=%d artifact_bytes=%d",
        len(stored_images),
        _stored_image_bytes(stored_images),
    )

    return ImageBridgeOutcome(
        image_output=image_output,
        display_mode=display_mode,
        image_results=image_results,
        stored_images=stored_images,
        artifacts=image_artifacts,
        tokens=tokens,
        usage=combined_usage,
        details=details,
        request_status=request_status,
        bridge_final_model=bridge_final_model,
        bridge_final_provider=bridge_final_provider,
        continuation_error=continuation_error,
        correction_applied=image_correction_applied,
    )


def _stored_image_bytes(items) -> int:
    total = 0
    for item in items:
        try:
            total += int(getattr(item, "bytes", 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _attach_request_details(exc: Exception, **details: Any) -> None:
    """Attach request-detail metadata to an exception for logging (mirrors proxy)."""
    existing = getattr(exc, "request_details", None)
    if not isinstance(existing, dict):
        existing = {}
    existing.update(details)
    try:
        exc.request_details = existing
    except AttributeError:
        pass
