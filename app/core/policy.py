import re
import json
from dataclasses import dataclass
from typing import Awaitable, Callable

from app.database import get_routing_rules, parse_model_id
from app.core.types import InternalMessage, InternalRequest, reasoning_part
from app.core.text import mask_key, strip_billing_header
from app.core.tool_args import sanitize_args
from app.services.logger import get_logger


_app_log = get_logger("app")


RequestPreprocessFunc = Callable[[InternalRequest, str, str, str], Awaitable[bool]]
ConversationKeyFunc = Callable[[str, list[InternalMessage], str], str]
ReasoningContextFunc = Callable[[str, list[InternalMessage]], tuple[str | None, dict]]


def normalize_messages(request: InternalRequest) -> None:
    if not request.messages:
        return

    merged: list[InternalMessage] = []
    for msg in request.messages:
        if msg.role == "system":
            for part in msg.parts:
                if part.kind == "text" and part.text:
                    part.text = strip_billing_header(part.text)

        if merged and merged[-1].role == msg.role and msg.role in ("system", "user"):
            prev = merged[-1]
            if _message_parts_are_text_only(prev.parts) and _message_parts_are_text_only(msg.parts):
                if prev.parts and msg.parts:
                    prev.parts[-1].text = f"{prev.parts[-1].text}\n\n{msg.parts[0].text}"
                    prev.parts.extend(msg.parts[1:])
                elif msg.parts:
                    prev.parts.extend(msg.parts)
            else:
                prev.parts.extend(msg.parts)
            _merge_message_metadata(prev, msg)
            continue

        merged.append(msg)

    request.messages = merged


def _message_parts_are_text_only(parts: list) -> bool:
    return all(part.kind == "text" for part in parts)


def _merge_message_metadata(target: InternalMessage, source: InternalMessage) -> None:
    if not target.name and source.name:
        target.name = source.name
    for key, value in source.raw.items():
        if key not in ("role", "content") and value and not target.raw.get(key):
            target.raw[key] = value
    for key, value in source.extensions.items():
        if value and not target.extensions.get(key):
            target.extensions[key] = value


def _ir_message_has_text(message: InternalMessage) -> bool:
    return any(part.kind == "text" and part.text.strip() for part in message.parts)


def _ir_message_has_tool_calls(message: InternalMessage) -> bool:
    return any(part.kind == "tool_call" for part in message.parts)


def _ir_message_has_reasoning(message: InternalMessage) -> bool:
    return any(part.kind == "reasoning" for part in message.parts)


def _ir_tool_call_ids(message: InternalMessage) -> list[str]:
    return [part.tool_call_id for part in message.parts if part.kind == "tool_call" and part.tool_call_id]


def request_has_tools(request: InternalRequest) -> bool:
    return bool(request.tools or request.extra.get("tools"))


def strip_tools(request: InternalRequest) -> None:
    request.extra.pop("tools", None)
    request.extra.pop("tool_choice", None)
    request.tools.clear()
    request.tool_choice = None


def fix_tool_args(request: InternalRequest) -> int:
    """Repair malformed tool-call argument JSON in internal messages."""
    fixed = 0
    for message in request.messages:
        for part in message.parts:
            if part.kind != "tool_call":
                continue
            if isinstance(part.raw_arguments, str) and "undefined" in part.raw_arguments:
                repaired = sanitize_args(part.raw_arguments)
                if repaired != part.raw_arguments:
                    part.raw_arguments = repaired
                    try:
                        part.arguments = json.loads(repaired) if repaired else {}
                    except json.JSONDecodeError:
                        part.arguments = None
                    fixed += 1
            elif isinstance(part.arguments, dict):
                repaired_args, changed = _replace_undefined_values(part.arguments)
                if changed:
                    part.arguments = repaired_args
                    part.raw_arguments = json.dumps(repaired_args, ensure_ascii=False)
                    fixed += 1
    return fixed


def _replace_undefined_values(value):
    if value == "undefined":
        return "", True
    if isinstance(value, dict):
        changed = False
        out = {}
        for key, item in value.items():
            out[key], item_changed = _replace_undefined_values(item)
            changed = changed or item_changed
        return out, changed
    if isinstance(value, list):
        changed = False
        out = []
        for item in value:
            replaced, item_changed = _replace_undefined_values(item)
            out.append(replaced)
            changed = changed or item_changed
        return out, changed
    return value, False


def inject_reasoning_content(
    messages: list[InternalMessage],
    cached_rc: str | None,
    tool_map: dict,
    *,
    allow_empty: bool = False,
) -> int:
    """Restore reasoning_content on active assistant tool-call turns in IR v2."""
    candidate_indexes = []
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.role == "assistant" and _ir_message_has_reasoning(msg):
            break
        if msg.role == "assistant" and not _ir_message_has_tool_calls(msg) and _ir_message_has_text(msg):
            break
        if msg.role == "assistant" and _ir_message_has_tool_calls(msg) and not _ir_message_has_reasoning(msg):
            candidate_indexes.append(idx)

    injected = 0
    for idx in reversed(candidate_indexes):
        msg = messages[idx]
        rc = None
        for tid in _ir_tool_call_ids(msg):
            if tid in tool_map:
                rc = tool_map[tid]
                break
        if rc is None and cached_rc is not None:
            rc = cached_rc
        elif rc is None and allow_empty:
            rc = ""
        if rc is not None:
            msg.parts.insert(0, reasoning_part(rc))
            injected += 1
    if injected or candidate_indexes or cached_rc is not None or tool_map:
        _app_log.debug(
            "[policy] reasoning_injection injected=%d candidates=%d cached_len=%d tool_map=%d",
            injected,
            len(candidate_indexes),
            len(cached_rc or ""),
            len(tool_map or {}),
        )
    return injected


@dataclass(slots=True)
class RequestPolicyResult:
    request: InternalRequest
    conv_key: str
    route_model: str
    route_provider: str
    modified_by_preprocessor: bool
    reasoning_injected: int = 0
    tool_only_limited: bool = False


def wildcard_match(pattern: str, value: str) -> bool:
    """Simple glob-style wildcard matching: * matches any sequence."""
    regex = re.escape(pattern).replace(r"\*", ".*")
    return bool(re.fullmatch(regex, value, re.IGNORECASE))


def apply_routing_rules(username: str, api_key_value: str, requested_model: str, resolved_model: str) -> tuple[str, str]:
    """Apply user-defined routing rules. Returns (final_model, provider_id)."""
    rules = get_routing_rules()
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        rule_user = rule.get("username", "")
        if rule_user and rule_user != username:
            continue
        key_pat = rule.get("api_key_pattern", "")
        if key_pat and key_pat not in api_key_value:
            continue
        match_model = rule.get("match_model", "")
        if not match_model:
            continue
        mid = parse_model_id(requested_model)
        if not (wildcard_match(match_model, requested_model) or
                (mid.is_composite and wildcard_match(match_model, mid.model_name))):
            continue
        target = rule.get("target_model", resolved_model)
        provider = rule.get("target_provider", "")
        if target and target != resolved_model:
            _app_log.info("[routing] rule='%s' matched: %s@%s requested '%s', routing to '%s'",
                          rule.get("name", ""), username, mask_key(api_key_value),
                          requested_model, target)
        return target or resolved_model, provider or ""
    return resolved_model, ""


async def prepare_request_policy(
    request: InternalRequest,
    *,
    username: str,
    api_key_value: str,
    preprocess_request: RequestPreprocessFunc,
    conversation_cache_key: ConversationKeyFunc,
    reasoning_context: ReasoningContextFunc | None = None,
    tool_only_turns=None,
    tool_only_limit: int | None = None,
    normalize: bool = True,
    log_label: str = "request",
) -> RequestPolicyResult:
    """Apply request-side policy while keeping endpoint-specific output unchanged."""
    requested_model = request.requested_model
    route_model, route_provider = apply_routing_rules(
        username, api_key_value, requested_model, request.target_model
    )
    if route_model != request.target_model:
        _app_log.debug("[%s] ROUTED model=%s -> %s", log_label, request.target_model, route_model)
        request.target_model = route_model
    if route_provider:
        request.provider_id = route_provider

    if normalize:
        pre_norm = len(request.messages)
        normalize_messages(request)
        _app_log.debug(
            "[%s NORM] messages %d -> %d roles=%s",
            log_label,
            pre_norm,
            len(request.messages),
            [m.role for m in request.messages],
        )

    conv_key = conversation_cache_key(api_key_value, request.messages, request.previous_response_id)

    modified = await preprocess_request(
        request,
        request.target_model,
        request.provider_id,
        requested_model,
    )
    injected = 0
    if reasoning_context is not None:
        cached_rc, tool_map = reasoning_context(conv_key, request.messages)
        injected = inject_reasoning_content(request.messages, cached_rc, tool_map)

    fix_tool_args(request)

    limited = False
    if tool_only_turns is not None and tool_only_limit is not None:
        if request_has_tools(request) and tool_only_turns.get(conv_key, 0) >= tool_only_limit:
            strip_tools(request)
            limited = True

    return RequestPolicyResult(
        request=request,
        conv_key=conv_key,
        route_model=route_model,
        route_provider=route_provider,
        modified_by_preprocessor=modified,
        reasoning_injected=injected,
        tool_only_limited=limited,
    )
