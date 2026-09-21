"""Native OpenAI Responses adapter.

This module deliberately keeps the protocol body and SSE payload opaque: Responses
has hosted/Codex tool item types that cannot be represented by Chat Completions.
"""
from typing import Any


from app.database import parse_model_id
from app.services.http_pool import shared_client


def split_sse_frame(buffer: bytes) -> tuple[bytes, bytes] | None:
    """Split one complete SSE frame while preserving its original line endings.

    SSE permits either LF or CRLF line endings.  We forward upstream frames
    unchanged, so normalizing CRLF here would defeat payload transparency.
    """
    lf_end = buffer.find(b"\n\n")
    crlf_end = buffer.find(b"\r\n\r\n")
    endings = [
        (lf_end, 2) if lf_end >= 0 else None,
        (crlf_end, 4) if crlf_end >= 0 else None,
    ]
    complete = [item for item in endings if item is not None]
    if not complete:
        return None
    index, width = min(complete, key=lambda item: item[0])
    return buffer[:index + width], buffer[index + width:]


def iter_sse_frames(chunks):
    """Yield complete SSE frames from an async byte iterator without rewriting bytes."""
    async def _iter():
        buffer = b""
        async for chunk in chunks:
            buffer += chunk
            while (split := split_sse_frame(buffer)) is not None:
                frame, buffer = split
                yield frame
        if buffer:
            yield buffer
    return _iter()


def sse_payload(frame: bytes) -> dict | None:
    data = [line[5:].lstrip() for line in frame.splitlines() if line.startswith(b"data:")]
    if not data:
        return None
    try:
        value = __import__("json").loads(b"\n".join(data).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def responses_url(api_base: str) -> str:
    base = (api_base or "").rstrip("/")
    if base.endswith("/responses"):
        return base
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def responses_headers(provider: dict) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = provider.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    upstream_headers = provider.get("upstream_headers") or {}
    if isinstance(upstream_headers, dict):
        headers.update({str(k): str(v) for k, v in upstream_headers.items() if v not in (None, "")})
    return headers


def native_responses_body(internal, *, stream: bool | None = None) -> dict[str, Any]:
    body = dict((internal.metadata.get("responses_native") or {}).get("request_body") or internal.raw_body)
    body["model"] = parse_model_id(internal.target_model).model_name
    if stream is not None:
        body["stream"] = stream
    return body


async def _raise_for_status_with_body(response) -> None:
    """Raise HTTPStatusError only after the error body is available for logs."""
    if response.status_code < 400:
        return
    try:
        if not getattr(response, "_content", False) and hasattr(response, "aread"):
            await response.aread()
    except Exception:
        pass
    response.raise_for_status()


def _request_timeout(provider: dict) -> int:
    return max(1, int(provider.get("request_timeout") or 120))


async def post_native_response(provider: dict, internal) -> dict[str, Any]:
    timeout = _request_timeout(provider)
    async with shared_client(provider.get("api_base", ""), timeout) as client:
        response = await client.post(responses_url(provider.get("api_base", "")), headers=responses_headers(provider), json=native_responses_body(internal, stream=False))
        await _raise_for_status_with_body(response)
        payload = response.json()
    if not isinstance(payload, dict) or payload.get("object") != "response":
        raise RuntimeError("upstream returned a non-Responses payload")
    return payload


async def stream_native_response(provider: dict, internal):
    timeout = _request_timeout(provider)
    async with shared_client(provider.get("api_base", ""), timeout) as client:
        async with client.stream("POST", responses_url(provider.get("api_base", "")), headers=responses_headers(provider), json=native_responses_body(internal, stream=True)) as response:
            await _raise_for_status_with_body(response)
            async for chunk in response.aiter_raw():
                # Preserve the upstream event framing byte-for-byte.
                yield chunk


# ---------------------------------------------------------------------------
# Native Responses 上游 wire 校验（上游协议边界）。
# 从 router/proxy.py 迁入：这些判断只依赖上游 native Responses 的原始
# JSON/SSE 载荷，属于 native Responses adapter 的协议职责。
# ---------------------------------------------------------------------------
def observed_response_tool_types(response: dict) -> set[str]:
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



class EmptyNativeResponsesError(RuntimeError):
    """The upstream completed a Responses request without client-usable output."""

    native_empty_output = True



def native_completed_output_item(item: dict | None) -> bool:
    """True when a Responses output item is a finished client-visible turn."""
    if not isinstance(item, dict):
        return False
    item_type = str(item.get("type") or "")
    return item_type in {
        "message", "function_call", "custom_tool_call",
        "computer_call", "image_generation_call", "output_text",
    } or item_type.endswith("_call")



def native_response_has_output(response: dict | None) -> bool:
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



def native_sse_payload_has_output(payload: dict | None) -> bool:
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



def native_sse_error_message(payload: dict | None) -> str | None:
    """Return a message for Responses SSE error payloads, including typeless ``event: error`` frames."""
    if not isinstance(payload, dict):
        return None
    event_type = str(payload.get("type") or "")
    error = payload.get("error")
    # OpenAI uses type=error. Some proxies emit ``event: error`` with only an
    # ``error`` object and no type. Ignore response.* frames that happen to
    # contain an error key.
    if event_type not in {"", "error"}:
        return None
    if event_type == "" and not isinstance(error, dict):
        return None
    if isinstance(error, dict):
        return str(error.get("message") or error.get("type") or error)
    if error:
        return str(error)
    if event_type == "error":
        return str(payload.get("message") or "native Responses stream error")
    return None





