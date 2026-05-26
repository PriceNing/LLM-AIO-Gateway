import json
from typing import Any

from app.core.types import (
    InternalMessage,
    InternalPart,
    image_part,
    reasoning_part,
    text_part,
    tool_call_part,
    tool_result_part,
    unknown_part,
)
from app.core.text import strip_billing_header


def _parse_arguments(raw_arguments: Any) -> Any:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, str):
        try:
            return json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            return None
    return raw_arguments


def _text_from_parts(parts: list[InternalPart]) -> str:
    return "\n".join(part.text for part in parts if part.kind == "text" and part.text)


def _parts_to_openai_content(parts: list[InternalPart]) -> Any:
    text = _text_from_parts(parts)
    images = [part for part in parts if part.kind == "image"]
    unknown = [part.raw for part in parts if part.kind == "unknown" and part.raw is not None]
    if not images and not unknown:
        return text
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for part in images:
        source = part.source
        if source.get("kind") == "url":
            image_url = {"url": source.get("url", "")}
            if source.get("detail"):
                image_url["detail"] = source["detail"]
            content.append({"type": "image_url", "image_url": image_url})
        elif source.get("kind") == "base64":
            media = source.get("media_type") or "image/png"
            content.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{source.get('data', '')}"}})
    for raw in unknown:
        content.append(raw if isinstance(raw, dict) else {"type": "text", "text": str(raw)})
    return content


def _parts_from_openai_content(content: Any) -> list[InternalPart]:
    if content is None:
        return []
    if isinstance(content, str):
        return [text_part(content)] if content else []
    if not isinstance(content, list):
        return [text_part(content)]

    parts = []
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append(text_part(item, raw=item))
            continue
        if not isinstance(item, dict):
            parts.append(unknown_part(item))
            continue
        item_type = item.get("type")
        if item_type in ("text", "input_text", "output_text"):
            txt = item.get("text", "")
            if txt:
                parts.append(text_part(txt, raw=dict(item)))
        elif item_type in ("image_url", "input_image"):
            image_url = item.get("image_url", "")
            source = {
                "kind": "url",
                "url": image_url.get("url", "") if isinstance(image_url, dict) else image_url,
            }
            detail = image_url.get("detail") if isinstance(image_url, dict) else item.get("detail")
            if detail:
                source["detail"] = detail
            parts.append(image_part(source, raw=dict(item)))
        elif item_type == "image":
            source_obj = item.get("source")
            if isinstance(source_obj, dict):
                source_type = source_obj.get("type") or source_obj.get("kind")
                if source_type == "base64" or source_obj.get("data"):
                    source = {
                        "kind": "base64",
                        "media_type": source_obj.get("media_type") or source_obj.get("mime_type") or "image/png",
                        "data": source_obj.get("data", ""),
                    }
                else:
                    source = {
                        "kind": "url",
                        "url": source_obj.get("url") or source_obj.get("uri") or "",
                    }
                if item.get("detail"):
                    source["detail"] = item.get("detail")
                parts.append(image_part(source, raw=dict(item)))
            elif isinstance(source_obj, str):
                parts.append(image_part({"kind": "url", "url": source_obj}, raw=dict(item)))
            else:
                parts.append(unknown_part(dict(item)))
        else:
            parts.append(unknown_part(dict(item)))
    return parts


def _parts_from_responses_content(content: Any) -> list[InternalPart]:
    if isinstance(content, str):
        return [text_part(content)] if content else []
    if not isinstance(content, list):
        return []
    parts = []
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append(text_part(item, raw=item))
            continue
        if not isinstance(item, dict):
            parts.append(unknown_part(item))
            continue
        item_type = item.get("type")
        if item_type in ("input_text", "output_text", "text"):
            txt = item.get("text", "")
            if txt:
                parts.append(text_part(txt, raw=dict(item)))
        elif item_type == "input_image":
            image_url = item.get("image_url", "")
            source = {"kind": "url", "url": image_url.get("url", "") if isinstance(image_url, dict) else image_url}
            detail = item.get("detail") or (image_url.get("detail") if isinstance(image_url, dict) else None)
            if detail:
                source["detail"] = detail
            parts.append(image_part(source, raw=dict(item)))
        else:
            parts.append(unknown_part(dict(item)))
    return parts


def _reasoning_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, dict):
                parts.append(part.get("text", "") or part.get("reasoning", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        return value.get("text") or value.get("reasoning") or value.get("summary") or ""
    return ""


def openai_messages_to_ir(messages: list[dict[str, Any]]) -> list[InternalMessage]:
    result = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            result.append(InternalMessage(role="user", parts=[unknown_part(msg)], raw={}))
            continue
        role = msg.get("role", "user")
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        parts = _parts_from_openai_content(msg.get("content"))

        reasoning = msg.get("reasoning_content")
        if reasoning:
            parts.insert(0, reasoning_part(reasoning, raw=reasoning))

        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                parts.append(unknown_part(tc))
                continue
            func = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
            raw_args = func.get("arguments", "")
            parts.append(tool_call_part(
                tc.get("id", ""),
                func.get("name", ""),
                _parse_arguments(raw_args),
                raw_arguments=raw_args,
                raw=dict(tc),
            ))

        if role == "tool":
            parts = [tool_result_part(msg.get("tool_call_id", ""), parts, raw=dict(msg))]

        result.append(InternalMessage(role=role, parts=parts, name=msg.get("name", ""), raw=dict(msg)))
    return result


def _anthropic_content_to_parts(content: Any) -> list[InternalPart]:
    if content is None:
        return []
    if isinstance(content, str):
        return [text_part(content)] if content else []
    if not isinstance(content, list):
        return [text_part(content)]

    parts = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(unknown_part(block))
            continue
        block_type = block.get("type")
        if block_type == "text":
            txt = block.get("text", "")
            if txt:
                extensions = {"cache_control": block.get("cache_control")} if block.get("cache_control") else {}
                parts.append(text_part(txt, raw=dict(block), extensions=extensions))
        elif block_type == "image":
            src = block.get("source", {}) if isinstance(block.get("source"), dict) else {}
            parts.append(image_part({
                "kind": src.get("type", "unknown"),
                "media_type": src.get("media_type"),
                "data": src.get("data"),
            }, raw=dict(block)))
        elif block_type == "tool_use":
            parts.append(tool_call_part(
                block.get("id", ""),
                block.get("name", ""),
                block.get("input", {}),
                raw_arguments=block.get("input", {}),
                raw=dict(block),
            ))
        elif block_type == "tool_result":
            parts.append(tool_result_part(
                block.get("tool_use_id", ""),
                _anthropic_content_to_parts(block.get("content", "")),
                raw=dict(block),
            ))
        elif block_type in ("thinking", "redacted_thinking"):
            parts.append(reasoning_part(
                block.get("thinking", "") or block.get("text", ""),
                raw=dict(block),
                extensions={"redacted": block_type == "redacted_thinking"},
            ))
        else:
            parts.append(unknown_part(dict(block)))
    return parts


def anthropic_messages_to_ir(messages: list[dict[str, Any]], system: Any = "") -> list[InternalMessage]:
    result = []
    system = strip_billing_header(system)
    if system:
        if isinstance(system, list):
            result.append(InternalMessage(role="system", parts=_anthropic_content_to_parts(system), raw={"system": system}))
        else:
            result.append(InternalMessage(role="system", parts=[text_part(system)], raw={"system": system}))
    for msg in messages or []:
        if not isinstance(msg, dict):
            result.append(InternalMessage(role="user", parts=[unknown_part(msg)], raw={}))
            continue
        role = msg.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        result.append(InternalMessage(role=role, parts=_anthropic_content_to_parts(msg.get("content", "")), raw=dict(msg)))
    return result


def responses_input_to_ir(input_data: Any, instructions: str = "") -> list[InternalMessage]:
    if isinstance(input_data, str):
        messages = []
        if instructions:
            messages.append(InternalMessage(role="system", parts=[text_part(instructions)], raw={"instructions": instructions}))
        messages.append(InternalMessage(role="user", parts=[text_part(input_data)], raw={"input": input_data}))
        return messages
    messages = []
    if instructions:
        messages.append(InternalMessage(role="system", parts=[text_part(instructions)], raw={"instructions": instructions}))
    if not isinstance(input_data, list):
        return messages

    last_tool_assistant_idx = None
    tool_call_assistant_idx = {}
    i = 0
    while i < len(input_data):
        item = input_data[i]
        if not isinstance(item, dict):
            i += 1
            continue
        item_type = item.get("type", "")

        if item_type == "message" or (not item_type and "role" in item):
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant", "tool"):
                role = "user"
            parts = _parts_from_responses_content(item.get("content", ""))
            rc = item.get("reasoning_content")
            if role == "assistant" and rc:
                parts.insert(0, reasoning_part(rc, raw=rc))
            messages.append(InternalMessage(role=role, parts=parts, raw=dict(item)))
            i += 1

        elif "role" in item:
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant", "tool"):
                role = "user"
            messages.append(InternalMessage(role=role, parts=_parts_from_responses_content(item.get("content", "")), raw=dict(item)))
            i += 1

        elif item_type == "function_call":
            tool_parts = []
            fc_items = []
            while i < len(input_data) and isinstance(input_data[i], dict) and input_data[i].get("type") == "function_call":
                fc = input_data[i]
                fc_items.append(fc)
                tool_parts.append(tool_call_part(
                    fc.get("call_id", ""),
                    fc.get("name", ""),
                    _parse_arguments(fc.get("arguments", "")),
                    raw_arguments=fc.get("arguments", ""),
                    raw=dict(fc),
                ))
                i += 1
            rc = next((fc.get("reasoning_content") for fc in fc_items if isinstance(fc, dict) and fc.get("reasoning_content")), None)
            if messages and messages[-1].role == "assistant" and not any(part.kind == "tool_call" for part in messages[-1].parts):
                prev = messages[-1]
                prev_text = _text_from_parts(prev.parts).strip()
                prev.parts = [part for part in prev.parts if part.kind != "text"]
                if rc:
                    prev.parts.insert(0, reasoning_part(rc, raw=rc))
                elif prev_text:
                    prev.parts.insert(0, reasoning_part(prev_text, raw=prev_text))
                prev.parts.extend(tool_parts)
                last_tool_assistant_idx = len(messages) - 1
            else:
                parts = []
                if rc:
                    parts.append(reasoning_part(rc, raw=rc))
                parts.extend(tool_parts)
                messages.append(InternalMessage(role="assistant", parts=parts, raw={"type": "function_call_group", "items": fc_items}))
                last_tool_assistant_idx = len(messages) - 1
            for part in tool_parts:
                if part.tool_call_id:
                    tool_call_assistant_idx[str(part.tool_call_id)] = last_tool_assistant_idx

        elif item_type == "function_call_output":
            call_id = str(item.get("call_id", ""))
            rc = item.get("reasoning_content")
            assistant_idx = tool_call_assistant_idx.get(call_id, last_tool_assistant_idx)
            if rc and assistant_idx is not None:
                assistant = messages[assistant_idx]
                if assistant.role == "assistant" and not any(part.kind == "reasoning" for part in assistant.parts):
                    assistant.parts.insert(0, reasoning_part(rc, raw=rc))
            messages.append(InternalMessage(
                role="tool",
                parts=[tool_result_part(call_id, _parts_from_responses_content(item.get("output", "")), raw=dict(item))],
                raw=dict(item),
            ))
            i += 1

        elif item_type == "reasoning":
            rc = _reasoning_text(item.get("reasoning_content") or item.get("summary") or item.get("text") or item.get("content"))
            if rc and last_tool_assistant_idx is not None:
                assistant = messages[last_tool_assistant_idx]
                if assistant.role == "assistant" and not any(part.kind == "reasoning" for part in assistant.parts):
                    assistant.parts.insert(0, reasoning_part(rc, raw=dict(item)))
            i += 1

        elif item_type == "input_image":
            part = _parts_from_responses_content([item])[0]
            for msg in reversed(messages):
                if msg.role == "user":
                    msg.parts.append(part)
                    break
            else:
                messages.append(InternalMessage(role="user", parts=[part], raw=dict(item)))
            i += 1

        else:
            i += 1
    return messages


def ir_to_openai_messages(messages: list[InternalMessage]) -> list[dict[str, Any]]:
    result = []
    for msg in messages or []:
        if msg.role in ("user", "tool") and any(part.kind == "tool_result" for part in msg.parts):
            pending_user_parts = []
            for part in msg.parts:
                if part.kind == "tool_result":
                    if pending_user_parts:
                        result.append({"role": "user", "content": _parts_to_openai_content(pending_user_parts)})
                        pending_user_parts = []
                    content = _parts_to_openai_content(part.parts)
                    result.append({"role": "tool", "tool_call_id": part.tool_call_id, "content": content or "(tool output)"})
                elif msg.role == "user":
                    pending_user_parts.append(part)
            if pending_user_parts:
                result.append({"role": "user", "content": _parts_to_openai_content(pending_user_parts)})
            continue

        out = {"role": msg.role}
        content = _parts_to_openai_content(msg.parts)
        out["content"] = (content or None) if msg.role == "assistant" else content

        tool_calls = []
        for part in msg.parts:
            if part.kind == "tool_call":
                raw_args = part.raw_arguments
                if raw_args is None:
                    raw_args = json.dumps(part.arguments if part.arguments is not None else {}, ensure_ascii=False)
                elif not isinstance(raw_args, str):
                    raw_args = json.dumps(raw_args, ensure_ascii=False)
                tool_calls.append({
                    "id": part.tool_call_id,
                    "type": "function",
                    "function": {"name": part.name, "arguments": raw_args},
                })
        if tool_calls:
            out["tool_calls"] = tool_calls
            if msg.role == "assistant" and not out.get("content"):
                out["content"] = None

        reasoning = "\n".join(part.text for part in msg.parts if part.kind == "reasoning" and part.text)
        if reasoning:
            out["reasoning_content"] = reasoning
        result.append(out)
    return result


def ir_to_anthropic_messages(messages: list[InternalMessage]) -> tuple[list[dict[str, Any]], Any]:
    anthropic_messages = []
    system_parts = []
    for msg in messages or []:
        if msg.role == "system":
            system_parts.extend(_parts_to_anthropic_content(msg.parts))
        elif msg.role in ("user", "assistant"):
            anthropic_messages.append({"role": msg.role, "content": _parts_to_anthropic_content(msg.parts)})
        elif msg.role == "tool":
            for part in msg.parts:
                if part.kind == "tool_result":
                    anthropic_messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": part.tool_call_id, "content": _text_from_parts(part.parts) or "(tool output)"}],
                    })
    system = ""
    if system_parts:
        if all(part.get("type") == "text" and not part.get("cache_control") for part in system_parts):
            system = "\n\n".join(part.get("text", "") for part in system_parts if part.get("text"))
        else:
            system = system_parts
    return anthropic_messages, system


def _parts_to_anthropic_content(parts: list[InternalPart]) -> list[dict[str, Any]]:
    content = []
    for part in parts or []:
        if part.kind == "text":
            block = {"type": "text", "text": part.text}
            if part.extensions.get("cache_control"):
                block["cache_control"] = part.extensions["cache_control"]
            content.append(block)
        elif part.kind == "image":
            source = part.source
            if source.get("kind") == "base64":
                content.append({"type": "image", "source": {"type": "base64", "media_type": source.get("media_type") or "image/png", "data": source.get("data", "")}})
            elif source.get("url", "").startswith("data:"):
                url = source.get("url", "")
                media = url.split(";")[0].replace("data:", "")
                data = url.split(",", 1)[1] if "," in url else ""
                content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
            elif source.get("url"):
                content.append({"type": "text", "text": f"[image URL: {source.get('url')}]"})
        elif part.kind == "tool_call":
            args = part.arguments if part.arguments is not None else _parse_arguments(part.raw_arguments)
            content.append({"type": "tool_use", "id": part.tool_call_id, "name": part.name, "input": args or {}})
        elif part.kind == "tool_result":
            content.append({"type": "tool_result", "tool_use_id": part.tool_call_id, "content": _text_from_parts(part.parts) or "(tool output)"})
        elif part.kind == "reasoning":
            content.append({"type": "thinking", "thinking": part.text})
        elif part.raw is not None:
            content.append(part.raw if isinstance(part.raw, dict) else {"type": "text", "text": str(part.raw)})
    return content or [{"type": "text", "text": ""}]
