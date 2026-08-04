import json
import time
import uuid

from app.adapters.output import tool_arguments_to_input
from app.core.output import InternalOutputEvent, InternalOutputMessage
from app.protocols.ingress import custom_tool_input_from_arguments
from app.services.logger import get_logger
from app.adapters.imagegen import ImageGenerationResult


_app_log = get_logger("app")
_tool_log = get_logger("tool_calls")


def render_responses_image_generation(results: list[ImageGenerationResult], *, model: str,
                                      previous_response_id: str | None = None,
                                      tool: dict | None = None) -> dict:
    def result_base64(result: ImageGenerationResult) -> str:
        """Return the Responses API image result without a data-URI prefix.

        Responses' image_generation_call.result is documented as base64 data.
        The images endpoint can still expose a data URI, but forwarding that
        wrapper through a Responses stream makes strict Codex clients abort.
        """
        value = result.data_uri or ""
        if value.startswith("data:") and "," in value:
            return value.split(",", 1)[1]
        return value

    output = []
    for result in results:
        item = {"type": "image_generation_call", "id": f"ig_{uuid.uuid4().hex}",
                "status": "completed", "result": result_base64(result),
                "output_format": result.output_format or result.mime_type.removeprefix("image/"),}
        if result.revised_prompt:
            item["revised_prompt"] = result.revised_prompt
        if result.size:
            item["size"] = result.size
        output.append(item)
    now = int(time.time())
    # Keep the bridge response shaped like a normal Responses response.  Codex
    # reads the lifecycle metadata as well as output_item.done; a minimal
    # response is accepted by the OpenAI SDK but can be discarded by stricter
    # Codex clients before it reaches the conversation renderer.
    return {
        "id": f"resp_{uuid.uuid4().hex}", "object": "response", "created_at": now,
        "completed_at": now, "status": "completed", "background": False,
        "error": None, "incomplete_details": None, "model": model,
        "previous_response_id": previous_response_id or None, "output": output,
        "parallel_tool_calls": True, "tool_choice": "auto",
        "tools": [tool] if isinstance(tool, dict) else [],
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "store": False,
        "tool_usage": {"image_gen": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


async def render_responses_image_generation_sse(results: list[ImageGenerationResult], *, model: str,
                                                previous_response_id: str | None = None,
                                                tool: dict | None = None):
    response = render_responses_image_generation(results, model=model,
                                                 previous_response_id=previous_response_id,
                                                 tool=tool)
    initial = {**response, "status": "in_progress", "output": []}
    def frame(payload: dict) -> str:
        # Sub2API forwards data-only Responses SSE frames. Keep the event
        # discriminator in the JSON payload and do not add gateway-specific
        # SSE fields.
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # The image upstream contract used by Sub2API does not require synthetic
    # lifecycle frames before the completed output item. Keep these standard
    # response lifecycle frames only; the image-specific result is emitted by
    # output_item.done below.
    yield frame({'type': 'response.created', 'response': initial})
    yield frame({'type': 'response.in_progress', 'response': initial})
    for index, item in enumerate(response["output"]):
        # This is the authoritative image event consumed by Sub2API and
        # Codex-compatible clients.
        yield frame({'type': 'response.output_item.done', 'output_index': index, 'item': item})
    # Sub2API's Codex Responses bridge preserves the completed image item in
    # both output_item.done and the terminal response snapshot. Its dedicated
    # /images adapter accepts an empty terminal output, but that is a different
    # protocol boundary; using that shape here makes Codex discard the turn.
    yield frame({'type': 'response.completed', 'response': response})
    yield "data: [DONE]\n\n"


def _anthropic_stop_reason(finish_reason: str) -> str:
    return {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}.get(
        finish_reason, "end_turn")


async def render_chat_completions_sse(events, *, model: str):
    chat_id = f"chatcmpl-{int(time.time())}"
    has_text = False
    accumulated_reasoning = ""
    text_chars = 0
    tool_calls = set()

    _app_log.debug("[egress_chat_stream] START model=%s chat_id=%s", model, chat_id)

    async for event in events:
        if event.kind == "reasoning_delta":
            accumulated_reasoning += event.reasoning
            delta = {"reasoning_content": event.reasoning}
        elif event.kind == "text_delta":
            has_text = True
            text_chars += len(event.text)
            delta = {"content": event.text}
        elif event.kind == "message_start" and event.role:
            delta = {"role": event.role}
        elif event.kind == "tool_call_start":
            tool_calls.add(event.tool_index)
            _tool_log.debug("[egress_chat_stream] tool_start index=%d id=%s name=%s", event.tool_index, event.tool_call_id, event.name)
            delta = {
                "tool_calls": [{
                    "index": event.tool_index,
                    "id": event.tool_call_id,
                    "type": "function",
                    "function": {"name": event.name, "arguments": ""},
                }]
            }
        elif event.kind == "tool_call_arguments_delta":
            delta = {
                "tool_calls": [{
                    "index": event.tool_index,
                    "id": event.tool_call_id,
                    "type": "function",
                    "function": {"name": event.name, "arguments": event.arguments_delta},
                }]
            }
        elif event.kind == "message_done":
            if not has_text and accumulated_reasoning:
                yield _chat_chunk(chat_id, model, {"content": accumulated_reasoning}, event.finish_reason or "stop")
            else:
                yield _chat_chunk(chat_id, model, {}, event.finish_reason or ("tool_calls" if tool_calls else "stop"))
            _app_log.debug(
                "[egress_chat_stream] DONE model=%s finish_reason=%s text_chars=%d reasoning_chars=%d tool_calls=%d",
                model,
                event.finish_reason or "stop",
                text_chars,
                len(accumulated_reasoning),
                len(tool_calls),
            )
            yield "data: [DONE]\n\n"
            return
        else:
            continue
        yield _chat_chunk(chat_id, model, delta, None)

    yield "data: [DONE]\n\n"


def _chat_chunk(chat_id: str, model: str, delta: dict, finish_reason):
    return f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]}, ensure_ascii=False)}\n\n"


async def render_completions_sse(events, *, model: str):
    cmpl_id = f"cmpl-{int(time.time())}"
    _app_log.debug("[egress_completions_stream] START model=%s completion_id=%s", model, cmpl_id)
    text_chars = 0
    reasoning_chars = 0

    async for event in events:
        if event.kind == "text_delta" and event.text:
            text_chars += len(event.text)
            yield _completion_chunk(cmpl_id, model, event.text, None)
        elif event.kind == "reasoning_delta" and event.reasoning:
            reasoning_chars += len(event.reasoning)
        elif event.kind == "message_done":
            _app_log.debug(
                "[egress_completions_stream] DONE model=%s finish_reason=%s text_chars=%d reasoning_chars=%d",
                model,
                event.finish_reason or "stop",
                text_chars,
                reasoning_chars,
            )
            yield _completion_chunk(cmpl_id, model, "", event.finish_reason or "stop")
            yield "data: [DONE]\n\n"
            return

    yield "data: [DONE]\n\n"


def _completion_chunk(cmpl_id: str, model: str, text: str, finish_reason):
    return f"data: {json.dumps({'id': cmpl_id, 'object': 'text_completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'text': text, 'finish_reason': finish_reason}]}, ensure_ascii=False)}\n\n"


def _responses_tool_ids(event: InternalOutputEvent) -> tuple[str, str]:
    tool_id = event.tool_call_id or f"fc_{int(time.time())}_{event.tool_index}"
    call_id = event.call_id or (tool_id if str(tool_id).startswith("call_") else f"call_{tool_id}")
    return tool_id, call_id


def _responses_custom_tools(extra: dict | None) -> dict[str, dict[str, str]]:
    custom_tools = (extra or {}).get("responses_custom_tools")
    return custom_tools if isinstance(custom_tools, dict) else {}


def _responses_namespace_tools(extra: dict | None) -> dict[str, dict[str, str]]:
    namespace_tools = (extra or {}).get("responses_namespace_tools")
    return namespace_tools if isinstance(namespace_tools, dict) else {}


def _responses_tool_name(raw_name: str, namespace_tools: dict[str, dict[str, str]]) -> tuple[str | None, str]:
    mapped = namespace_tools.get(raw_name)
    if isinstance(mapped, dict):
        return mapped.get("namespace") or None, mapped.get("name") or raw_name
    if "." in raw_name:
        namespace, name = raw_name.split(".", 1)
        if namespace and name:
            return namespace, name
    if raw_name.startswith("mcp__"):
        rest = raw_name[len("mcp__"):]
        marker = rest.find("__")
        if marker >= 0:
            split_at = len("mcp__") + marker + len("__")
            if split_at < len(raw_name):
                return raw_name[:split_at], raw_name[split_at:]
    return None, raw_name


def _responses_tool_item_from_state(state: dict, *, status: str, extra: dict | None = None) -> dict:
    name = state.get("name") or ""
    custom_tools = _responses_custom_tools(extra)
    custom_tool = custom_tools.get(name)
    item_id = state.get("item_id") or state.get("id") or f"fc_{int(time.time())}"
    call_id = state.get("call_id") or state.get("id") or f"call_{int(time.time())}"
    if custom_tool:
        argument_field = custom_tool.get("argument_field") or ("patch" if name == "apply_patch" else "input")
        input_text = custom_tool_input_from_arguments(state.get("arguments") or "", argument_field)
        return {
            "type": "custom_tool_call",
            "id": item_id if str(item_id).startswith("ctc_") else f"ctc_{item_id}",
            "call_id": call_id,
            "name": custom_tool.get("name") or name,
            "input": "" if status == "in_progress" else input_text,
            "status": status,
        }

    namespace, response_name = _responses_tool_name(name, _responses_namespace_tools(extra))
    item = {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": response_name,
        "arguments": "" if status == "in_progress" else (state.get("arguments") or ""),
        "status": status,
    }
    if namespace:
        item["namespace"] = namespace
    return item


def _custom_tool_state_delta(state: dict, argument_field: str) -> str:
    input_text = custom_tool_input_from_arguments(state.get("arguments") or "", argument_field)
    previous = state.get("custom_input_sent") or ""
    if input_text == previous:
        return ""
    if input_text.startswith(previous):
        delta = input_text[len(previous):]
    else:
        delta = input_text
    state["custom_input_sent"] = input_text
    return delta


async def render_responses_sse(events, *, model: str, previous_response_id: str | None = None, response_id: str | None = None, extra: dict | None = None):
    resp_id = response_id or f"resp_{uuid.uuid4().hex}"
    msg_id = f"msg_{uuid.uuid4().hex}"
    created_at = int(time.time())
    response_base = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "model": model,
        "output": [],
        "previous_response_id": previous_response_id or None,
        "metadata": {},
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    yield f"data: {json.dumps({'type': 'response.created', 'response': response_base})}\n\n"
    yield f"data: {json.dumps({'type': 'response.in_progress', 'response': response_base})}\n\n"

    text_item_added = False
    text_content_added = False
    text_output_index = 0
    output_index_counter = 0
    accumulated_text = ""
    accumulated_reasoning = ""
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    tool_states: dict[int, dict] = {}
    completion_output = []
    saw_message_done = False

    _app_log.debug("[egress_responses_stream] START model=%s response_id=%s previous_response_id=%s", model, resp_id, previous_response_id or "")

    async for event in events:
        if event.kind == "text_delta" and event.text:
            if not text_item_added:
                text_output_index = output_index_counter
                output_index_counter += 1
                text_item_added = True
                yield f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': text_output_index, 'item': {'type': 'message', 'id': msg_id, 'status': 'in_progress', 'role': 'assistant', 'content': []}})}\n\n"
            if not text_content_added:
                text_content_added = True
                yield f"data: {json.dumps({'type': 'response.content_part.added', 'output_index': text_output_index, 'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}})}\n\n"
            accumulated_text += event.text
            yield f"data: {json.dumps({'type': 'response.output_text.delta', 'output_index': text_output_index, 'content_index': 0, 'delta': event.text})}\n\n"
        elif event.kind == "reasoning_delta":
            accumulated_reasoning += event.reasoning
        elif event.kind == "tool_call_start":
            tool_id, call_id = _responses_tool_ids(event)
            state = tool_states.setdefault(event.tool_index, {
                "id": tool_id,
                "call_id": call_id,
                "name": event.name,
                "arguments": "",
                "output_index": output_index_counter,
                "item_added": True,
            })
            output_index_counter += 1
            _tool_log.debug("[egress_responses_stream] tool_start index=%d id=%s call_id=%s name=%s output_index=%d", event.tool_index, state["id"], state["call_id"], state["name"], state["output_index"])
            yield f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': state['output_index'], 'item': _responses_tool_item_from_state(state, status='in_progress', extra=extra)})}\n\n"
        elif event.kind == "tool_call_arguments_delta":
            tool_id, call_id = _responses_tool_ids(event)
            state = tool_states.setdefault(event.tool_index, {
                "id": tool_id,
                "call_id": call_id,
                "name": event.name,
                "arguments": "",
                "output_index": output_index_counter,
                "item_added": False,
            })
            if not state.get("item_added"):
                output_index_counter = max(output_index_counter, state["output_index"] + 1)
                state["item_added"] = True
                yield f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': state['output_index'], 'item': _responses_tool_item_from_state(state, status='in_progress', extra=extra)})}\n\n"
            state["arguments"] = event.arguments or (state["arguments"] + event.arguments_delta)
            custom_tool = _responses_custom_tools(extra).get(state.get("name") or "")
            if custom_tool:
                argument_field = custom_tool.get("argument_field") or ("patch" if state.get("name") == "apply_patch" else "input")
                delta = _custom_tool_state_delta(state, argument_field)
                if delta:
                    yield f"data: {json.dumps({'type': 'response.custom_tool_call_input.delta', 'output_index': state['output_index'], 'call_id': state['call_id'], 'delta': delta})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'response.function_call_arguments.delta', 'output_index': state['output_index'], 'call_id': state['call_id'], 'delta': event.arguments_delta})}\n\n"
        elif event.kind == "usage":
            usage.update({k: v for k, v in event.usage.items() if k in usage})
        elif event.kind == "message_done":
            _app_log.debug("[egress_responses_stream] message_done finish_reason=%s", event.finish_reason or "")
            saw_message_done = True
            break

    if not saw_message_done and not accumulated_text and not accumulated_reasoning and not tool_states:
        raise RuntimeError("upstream closed before sending response output")

    if text_item_added:
        if text_content_added:
            yield f"data: {json.dumps({'type': 'response.output_text.done', 'output_index': text_output_index, 'content_index': 0, 'text': accumulated_text})}\n\n"
            yield f"data: {json.dumps({'type': 'response.content_part.done', 'output_index': text_output_index, 'content_index': 0, 'part': {'type': 'output_text', 'text': accumulated_text, 'annotations': []}})}\n\n"
        msg_out = {"type": "message", "id": msg_id, "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": accumulated_text, "annotations": []}]}
        if accumulated_reasoning:
            msg_out["reasoning_content"] = accumulated_reasoning
        yield f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': text_output_index, 'item': msg_out})}\n\n"
        completion_output.append(msg_out)
    elif accumulated_reasoning:
        completion_output.append({"type": "message", "id": msg_id, "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": accumulated_reasoning, "annotations": []}]})

    for idx in sorted(tool_states):
        state = tool_states[idx]
        custom_tool = _responses_custom_tools(extra).get(state.get("name") or "")
        if custom_tool:
            argument_field = custom_tool.get("argument_field") or ("patch" if state.get("name") == "apply_patch" else "input")
            delta = _custom_tool_state_delta(state, argument_field)
            if delta:
                yield f"data: {json.dumps({'type': 'response.custom_tool_call_input.delta', 'output_index': state['output_index'], 'call_id': state['call_id'], 'delta': delta})}\n\n"
            yield f"data: {json.dumps({'type': 'response.custom_tool_call_input.done', 'output_index': state['output_index'], 'call_id': state['call_id'], 'input': custom_tool_input_from_arguments(state['arguments'], argument_field)})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'response.function_call_arguments.done', 'output_index': state['output_index'], 'call_id': state['call_id'], 'arguments': state['arguments']})}\n\n"
        item = _responses_tool_item_from_state(state, status="completed", extra=extra)
        yield f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': state['output_index'], 'item': item})}\n\n"
        completion_output.append(item)

    completed = {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "model": model,
            "output": completion_output,
            "previous_response_id": previous_response_id or None,
            "metadata": {},
            "usage": usage,
        },
    }
    _app_log.debug(
        "[egress_responses_stream] DONE model=%s response_id=%s text_chars=%d reasoning_chars=%d tool_calls=%d total_tokens=%d",
        model,
        resp_id,
        len(accumulated_text),
        len(accumulated_reasoning),
        len(tool_states),
        usage.get("total_tokens", 0),
    )
    yield f"data: {json.dumps(completed)}\n\n"
    yield "data: [DONE]\n\n"


async def render_anthropic_messages_sse(events, *, model: str):
    msg_id = f"msg_{int(time.time())}"
    _app_log.debug("[egress_messages_stream] START model=%s message_id=%s", model, msg_id)
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    text_started = False
    text_index = 0
    next_block_index = 0
    accumulated_text = ""
    accumulated_reasoning = ""
    input_tokens = 0
    output_tokens = 0
    finish_reason = "stop"
    tool_states: dict[int, dict] = {}

    async for event in events:
        if event.kind == "text_delta" and event.text:
            if not text_started:
                text_started = True
                text_index = next_block_index
                next_block_index += 1
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            accumulated_text += event.text
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': event.text}})}\n\n"
        elif event.kind == "reasoning_delta":
            accumulated_reasoning += event.reasoning
        elif event.kind == "tool_call_start":
            block_index = next_block_index
            next_block_index += 1
            tool_states[event.tool_index] = {
                "block_index": block_index,
                "id": event.tool_call_id or f"toolu_{event.tool_index}",
                "name": event.name,
                "arguments": "",
            }
            _tool_log.debug("[egress_messages_stream] tool_start index=%d block_index=%d id=%s name=%s", event.tool_index, block_index, event.tool_call_id, event.name)
        elif event.kind == "tool_call_arguments_delta":
            state = tool_states.setdefault(event.tool_index, {
                "block_index": next_block_index,
                "id": event.tool_call_id or f"toolu_{event.tool_index}",
                "name": event.name,
                "arguments": "",
            })
            state["arguments"] = event.arguments or (state["arguments"] + event.arguments_delta)
            if state["block_index"] == next_block_index:
                next_block_index += 1
        elif event.kind == "usage":
            input_tokens = event.usage.get("input_tokens", input_tokens) or input_tokens
            output_tokens = event.usage.get("output_tokens", output_tokens) or output_tokens
        elif event.kind == "message_done":
            finish_reason = event.finish_reason or finish_reason
            _app_log.debug("[egress_messages_stream] message_done finish_reason=%s", finish_reason)
            break

    if text_started:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_index})}\n\n"
    elif accumulated_reasoning and not tool_states:
        text_index = next_block_index
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': accumulated_reasoning}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_index})}\n\n"

    for idx in sorted(tool_states):
        state = tool_states[idx]
        tool_input = tool_arguments_to_input(state["arguments"])
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': state['block_index'], 'content_block': {'type': 'tool_use', 'id': state['id'], 'name': state['name'], 'input': {}}})}\n\n"
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': state['block_index'], 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(tool_input, ensure_ascii=False)}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': state['block_index']})}\n\n"

    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': _anthropic_stop_reason(finish_reason), 'stop_sequence': None}, 'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    _app_log.debug(
        "[egress_messages_stream] DONE model=%s message_id=%s text_chars=%d reasoning_chars=%d tool_calls=%d input_tokens=%d output_tokens=%d stop_reason=%s",
        model,
        msg_id,
        len(accumulated_text),
        len(accumulated_reasoning),
        len(tool_states),
        input_tokens,
        output_tokens,
        _anthropic_stop_reason(finish_reason),
    )


async def render_responses_error_sse(*, model: str, message: str, previous_response_id: str | None = None):
    resp_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    yield f"data: {json.dumps({'type': 'error', 'error': {'message': message, 'type': 'server_error'}})}\n\n"
    completed = {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "status": "failed",
            "model": model,
            "output": [],
            "previous_response_id": previous_response_id or None,
            "metadata": {},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "status_details": {"error": {"type": "server_error", "message": message}},
        },
    }
    yield f"data: {json.dumps(completed)}\n\n"
    yield "data: [DONE]\n\n"


def render_chat_completion(output: InternalOutputMessage, *, model: str) -> dict:
    _app_log.debug(
        "[egress_chat_nonstream] model=%s finish_reason=%s text_chars=%d reasoning_chars=%d tool_calls=%d total_tokens=%d",
        model,
        output.finish_reason,
        len(output.text or ""),
        len(output.reasoning or ""),
        len(output.tool_calls),
        output.usage.get("total_tokens", 0),
    )
    resp_msg = {"role": output.role or "assistant", "content": output.text}
    if output.reasoning:
        resp_msg["reasoning_content"] = output.reasoning
    if output.tool_calls:
        resp_msg["tool_calls"] = [
            {
                "id": tool.id,
                "type": "function",
                "function": {"name": tool.name, "arguments": tool.arguments},
            }
            for tool in output.tool_calls
        ]
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": resp_msg, "finish_reason": output.finish_reason or "stop"}],
        "usage": output.usage,
    }


def render_completion(output: InternalOutputMessage, *, model: str) -> dict:
    _app_log.debug(
        "[egress_completions_nonstream] model=%s finish_reason=%s text_chars=%d reasoning_chars=%d total_tokens=%d",
        model,
        output.finish_reason,
        len(output.text or ""),
        len(output.reasoning or ""),
        output.usage.get("total_tokens", 0),
    )
    return {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "text": output.text or "",
            "index": 0,
            "finish_reason": output.finish_reason or "stop",
        }],
        "usage": output.usage,
    }


def render_anthropic_message(output: InternalOutputMessage, *, model: str) -> dict:
    _app_log.debug(
        "[egress_messages_nonstream] model=%s finish_reason=%s text_chars=%d reasoning_chars=%d tool_calls=%d total_tokens=%d",
        model,
        output.finish_reason,
        len(output.text or ""),
        len(output.reasoning or ""),
        len(output.tool_calls),
        output.usage.get("total_tokens", 0),
    )
    content_blocks = []
    if output.text:
        content_blocks.append({"type": "text", "text": output.text})
    for tool in output.tool_calls:
        content_blocks.append({
            "type": "tool_use",
            "id": tool.id,
            "name": tool.name,
            "input": tool_arguments_to_input(tool.arguments),
        })
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    return {
        "id": f"msg_{int(time.time())}",
        "type": "message",
        "role": output.role or "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": _anthropic_stop_reason(output.finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": output.usage.get("prompt_tokens", output.usage.get("input_tokens", 0)),
            "output_tokens": output.usage.get("completion_tokens", output.usage.get("output_tokens", 0)),
        },
    }


def render_response(output: InternalOutputMessage, *, model: str, previous_response_id: str | None = None, response_id: str | None = None, extra: dict | None = None) -> dict:
    resp_id = response_id or f"resp_{uuid.uuid4().hex}"
    msg_id = f"msg_{uuid.uuid4().hex}"
    rendered_output = []
    _app_log.debug(
        "[egress_responses_nonstream] model=%s response_id=%s previous_response_id=%s finish_reason=%s text_chars=%d reasoning_chars=%d tool_calls=%d total_tokens=%d",
        model,
        resp_id,
        previous_response_id or "",
        output.finish_reason,
        len(output.text or ""),
        len(output.reasoning or ""),
        len(output.tool_calls),
        output.usage.get("total_tokens", 0),
    )
    if output.text:
        msg_out = {
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": output.role or "assistant",
            "content": [{"type": "output_text", "text": output.text, "annotations": []}],
        }
        if output.reasoning:
            msg_out["reasoning_content"] = output.reasoning
        rendered_output.append(msg_out)
    for tool in output.tool_calls:
        rendered_output.append(_responses_tool_item_from_state({
            "id": tool.id or f"fc_{int(time.time())}",
            "call_id": tool.call_id or tool.id or f"call_{int(time.time())}",
            "name": tool.name,
            "arguments": tool.arguments,
        }, status="completed", extra=extra))
    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "previous_response_id": previous_response_id or None,
        "output": rendered_output,
        "usage": {
            "input_tokens": output.usage.get("prompt_tokens", output.usage.get("input_tokens", 0)),
            "output_tokens": output.usage.get("completion_tokens", output.usage.get("output_tokens", 0)),
            "total_tokens": output.usage.get("total_tokens", 0),
        },
    }
