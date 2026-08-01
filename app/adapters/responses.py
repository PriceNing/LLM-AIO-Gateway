"""Native OpenAI Responses adapter.

This module deliberately keeps the protocol body and SSE payload opaque: Responses
has hosted/Codex tool item types that cannot be represented by Chat Completions.
"""
from typing import Any

import httpx

from app.database import parse_model_id


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
    extra = provider.get("extra_headers") or {}
    if isinstance(extra, dict):
        headers.update({str(k): str(v) for k, v in extra.items() if k != "thinking"})
    return headers


def native_responses_body(internal, *, stream: bool | None = None) -> dict[str, Any]:
    body = dict((internal.metadata.get("responses_native") or {}).get("request_body") or internal.raw_body)
    body["model"] = parse_model_id(internal.target_model).model_name
    if stream is not None:
        body["stream"] = stream
    return body


async def post_native_response(provider: dict, internal) -> dict[str, Any]:
    timeout = max(1, int(provider.get("request_timeout") or 120))
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(responses_url(provider.get("api_base", "")), headers=responses_headers(provider), json=native_responses_body(internal, stream=False))
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict) or payload.get("object") != "response":
        raise RuntimeError("upstream returned a non-Responses payload")
    return payload


async def stream_native_response(provider: dict, internal):
    timeout = max(1, int(provider.get("request_timeout") or 120))
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", responses_url(provider.get("api_base", "")), headers=responses_headers(provider), json=native_responses_body(internal, stream=True)) as response:
            response.raise_for_status()
            async for chunk in response.aiter_raw():
                # Preserve the upstream event framing byte-for-byte.
                yield chunk
