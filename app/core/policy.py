import re
import json
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.database import get_fallback_policies, get_routing_rules, parse_model_id
from app.core.types import InternalMessage, InternalRequest, reasoning_part
from app.core.text import mask_key, strip_billing_header
from app.core.tool_args import coerce_tool_arguments_json, sanitize_args
from app.services.logger import get_logger


_app_log = get_logger("app")


RequestPreprocessFunc = Callable[[InternalRequest, str, str, str], Awaitable[bool]]
ConversationKeyFunc = Callable[[str, list[InternalMessage], str], str]
ReasoningContextFunc = Callable[[str, list[InternalMessage]], tuple[str | None, dict]]


def _merge_compatible_messages(target: InternalMessage, source: InternalMessage) -> None:
    if _message_parts_are_text_only(target.parts) and _message_parts_are_text_only(source.parts):
        if target.parts and source.parts:
            target.parts[-1].text = f"{target.parts[-1].text}\n\n{source.parts[0].text}"
            target.parts.extend(source.parts[1:])
        elif source.parts:
            target.parts.extend(source.parts)
    else:
        target.parts.extend(source.parts)
    _merge_message_metadata(target, source)


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
            _merge_compatible_messages(merged[-1], msg)
            continue

        merged.append(msg)

    systems = [msg for msg in merged if msg.role == "system"]
    if len(systems) > 1:
        combined = systems[0]
        for extra in systems[1:]:
            _merge_compatible_messages(combined, extra)
        merged = [combined] + [msg for msg in merged if msg.role != "system"]

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
            if isinstance(part.raw_arguments, str):
                coerced = coerce_tool_arguments_json(part.raw_arguments)
                if coerced != part.raw_arguments:
                    part.raw_arguments = coerced
                    try:
                        part.arguments = json.loads(coerced) if coerced else {}
                    except json.JSONDecodeError:
                        part.arguments = {"input": coerced}
                    fixed += 1
            elif part.raw_arguments is None and not isinstance(part.arguments, dict):
                coerced = coerce_tool_arguments_json(part.arguments)
                part.raw_arguments = coerced
                try:
                    part.arguments = json.loads(coerced) if coerced else {}
                except json.JSONDecodeError:
                    part.arguments = {}
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


def has_missing_reasoning_content_for_tool_calls(messages: list[InternalMessage]) -> bool:
    """Whether thinking-mode replay would need reasoning unavailable in history."""
    return any(
        message.role == "assistant"
        and _ir_message_has_tool_calls(message)
        and not _ir_message_has_reasoning(message)
        for message in messages
    )


@dataclass(slots=True)
class RouteTarget:
    model: str
    provider_id: str = ""


@dataclass(slots=True)
class RoutingDecision:
    requested_model: str
    resolved_model: str
    target: RouteTarget
    matched: bool = False
    rule_id: int | None = None
    rule_name: str = ""
    source: str = "default"
    reason: str = "no routing rule matched"

    @property
    def target_model(self) -> str:
        return self.target.model

    @property
    def target_provider(self) -> str:
        return self.target.provider_id


@dataclass(slots=True)
class FallbackDecision:
    target: RouteTarget
    trigger: str = ""
    matched: bool = False
    policy_id: str = ""
    policy_name: str = ""
    reason: str = "no fallback policy matched"
    chain: list[RouteTarget] = field(default_factory=list)
    # Seconds to wait on current attempt before proactive timeout→fallback.
    attempt_timeout: int = 60


@dataclass(slots=True)
class RequestPolicyResult:
    request: InternalRequest
    conv_key: str
    route_model: str
    route_provider: str
    routing: RoutingDecision
    modified_by_preprocessor: bool
    reasoning_injected: int = 0
    tool_only_limited: bool = False


def wildcard_match(pattern: str, value: str) -> bool:
    """Simple glob-style wildcard matching: * matches any sequence."""
    regex = re.escape(pattern).replace(r"\*", ".*")
    return bool(re.fullmatch(regex, value, re.IGNORECASE))


def _route_targets(raw_targets) -> list[RouteTarget]:
    targets = []
    for item in raw_targets or []:
        if isinstance(item, str):
            model = item.strip()
            provider_id = ""
        elif isinstance(item, dict):
            model = str(item.get("model") or item.get("target_model") or "").strip()
            provider_id = str(item.get("provider_id") or item.get("target_provider") or "").strip()
        else:
            continue
        if model:
            targets.append(RouteTarget(model=model, provider_id=provider_id))
    return targets


def fallback_trigger_enabled(triggers: dict, trigger: str) -> bool:
    return bool((triggers or {}).get(trigger))


def _target_label(provider_id: str, model: str) -> str:
    mid = parse_model_id(model)
    if provider_id and mid.is_composite and mid.provider_id == provider_id:
        return model
    if provider_id:
        return f"{provider_id}/{model}"
    return model


def apply_fallback_policy(provider_id: str, model: str, trigger: str = "") -> FallbackDecision:
    target = RouteTarget(model=model, provider_id=provider_id)
    policies = sorted(
        get_fallback_policies(),
        key=lambda policy: 1 if (policy.get("match_model", "*") or "*").strip() == "*" else 0,
    )
    for policy in policies:
        if not policy.get("enabled", True):
            continue
        match_provider = policy.get("match_provider", "")
        if match_provider and match_provider != provider_id:
            continue
        match_model = policy.get("match_model", "*") or "*"
        mid = parse_model_id(model)
        if not (
            wildcard_match(match_model, model)
            or (mid.is_composite and wildcard_match(match_model, mid.model_name))
        ):
            continue
        if trigger and not fallback_trigger_enabled(policy.get("triggers", {}), trigger):
            _app_log.info(
                "[fallback] policy_skipped policy_id=%s policy='%s' target=%s provider=%s trigger=%s reason=trigger_disabled",
                policy.get("id"),
                policy.get("name", ""),
                model,
                provider_id or "-",
                trigger,
            )
            continue
        chain = _route_targets(policy.get("chain"))
        try:
            attempt_timeout = int(policy.get("attempt_timeout") or 60)
        except (TypeError, ValueError):
            attempt_timeout = 60
        attempt_timeout = max(5, min(3600, attempt_timeout))
        decision = FallbackDecision(
            target=target,
            trigger=trigger,
            matched=bool(chain),
            policy_id=policy.get("id", ""),
            policy_name=policy.get("name", ""),
            reason=f"matched fallback policy '{policy.get('name', '') or policy.get('id', '')}' for target '{_target_label(provider_id, model)}'",
            chain=chain,
            attempt_timeout=attempt_timeout,
        )
        _app_log.info(
            "[fallback] matched=%s policy_id=%s policy='%s' target=%s provider=%s trigger=%s chain=%d attempt_timeout=%ds reason=%s",
            decision.matched,
            decision.policy_id,
            decision.policy_name,
            model,
            provider_id or "-",
            trigger or "-",
            len(chain),
            decision.attempt_timeout,
            decision.reason,
        )
        return decision
    _app_log.debug(
        "[fallback] matched=False target=%s provider=%s trigger=%s reason=no policy matched",
        model,
        provider_id or "-",
        trigger or "-",
    )
    return FallbackDecision(target=target, trigger=trigger)


def apply_routing_rules(username: str, api_key_value: str, requested_model: str, resolved_model: str) -> RoutingDecision:
    """Apply user-defined routing rules and return a structured decision."""
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
        match_scope = str(rule.get("match_scope") or "any").lower()
        if match_scope == "unqualified":
            model_matches = not mid.is_composite and wildcard_match(match_model, requested_model)
        elif match_scope == "qualified":
            model_matches = mid.is_composite and wildcard_match(match_model, requested_model)
        else:
            # Legacy/default behaviour: a simple model alias also matches the
            # model component of a provider-qualified request.
            model_matches = wildcard_match(match_model, requested_model) or (
                mid.is_composite and wildcard_match(match_model, mid.model_name)
            )
        if not model_matches:
            continue
        target = rule.get("target_model", resolved_model)
        provider = rule.get("target_provider", "")
        target_model = target or resolved_model
        target_provider = provider or ""
        decision = RoutingDecision(
            requested_model=requested_model,
            resolved_model=resolved_model,
            target=RouteTarget(model=target_model, provider_id=target_provider),
            matched=True,
            rule_id=rule.get("id"),
            rule_name=rule.get("name", ""),
            source="routing_rule",
            reason=f"matched rule '{rule.get('name', '') or rule.get('id', '')}' for model '{match_model}'",
        )
        _app_log.info(
            "[routing] matched=%s rule_id=%s rule='%s' user=%s key=%s requested=%s resolved=%s target=%s provider=%s reason=%s",
            decision.matched,
            decision.rule_id,
            decision.rule_name,
            username,
            mask_key(api_key_value),
            decision.requested_model,
            decision.resolved_model,
            decision.target_model,
            decision.target_provider or "-",
            decision.reason,
        )
        return decision
    return RoutingDecision(
        requested_model=requested_model,
        resolved_model=resolved_model,
        target=RouteTarget(model=resolved_model, provider_id=""),
    )


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
    preprocess: bool = True,
    apply_ir_transforms: bool = True,
    log_label: str = "request",
) -> RequestPolicyResult:
    """Apply request-side policy while keeping endpoint-specific output unchanged."""
    requested_model = request.requested_model
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

    modified = False
    if preprocess:
        modified = await preprocess_request(
            request,
            requested_model,
            parse_model_id(requested_model).provider_id,
            requested_model,
        )
    _app_log.info(
        "[%s preprocess.result] requested=%s modified=%s messages=%d",
        log_label,
        requested_model,
        modified,
        len(request.messages),
    )

    routing = apply_routing_rules(
        username, api_key_value, requested_model, request.target_model
    )
    route_model = routing.target_model
    route_provider = routing.target_provider
    if route_model != request.target_model:
        _app_log.debug("[%s] ROUTED model=%s -> %s", log_label, request.target_model, route_model)
        request.target_model = route_model
    if route_provider:
        request.provider_id = route_provider
    _app_log.info(
        "[%s ROUTE] matched=%s source=%s rule_id=%s rule='%s' requested=%s resolved=%s target=%s provider=%s reason=%s",
        log_label,
        routing.matched,
        routing.source,
        routing.rule_id,
        routing.rule_name,
        routing.requested_model,
        routing.resolved_model,
        routing.target_model,
        routing.target_provider or "-",
        routing.reason,
    )
    injected = 0
    if apply_ir_transforms and reasoning_context is not None:
        cached_rc, tool_map = reasoning_context(conv_key, request.messages)
        injected = inject_reasoning_content(request.messages, cached_rc, tool_map)

    if apply_ir_transforms:
        fix_tool_args(request)

    limited = False
    if apply_ir_transforms and tool_only_turns is not None and tool_only_limit is not None:
        if request_has_tools(request) and tool_only_turns.get(conv_key, 0) >= tool_only_limit:
            strip_tools(request)
            limited = True

    return RequestPolicyResult(
        request=request,
        conv_key=conv_key,
        route_model=route_model,
        route_provider=route_provider,
        routing=routing,
        modified_by_preprocessor=modified,
        reasoning_injected=injected,
        tool_only_limited=limited,
    )
