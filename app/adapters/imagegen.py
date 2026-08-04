"""Adapters for OpenAI-compatible image-generation backends."""

import base64
import binascii
import math
import mimetypes
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from app.database import parse_model_id


@dataclass
class ImageGenerationResult:
    """Provider-neutral image result; protocol renderers add their own shape."""

    data_uri: str
    mime_type: str = "image/png"
    revised_prompt: str | None = None
    size: str | None = None
    quality: str | None = None
    output_format: str | None = None
    background: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


def images_url(api_base: str) -> str:
    base = (api_base or "").rstrip("/")
    if base.endswith("/images/generations"):
        return base
    if base.endswith("/v1"):
        return f"{base}/images/generations"
    return f"{base}/v1/images/generations"


def _headers(config: dict) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.get("api_key"):
        headers["Authorization"] = f"Bearer {config['api_key']}"
    extra = config.get("extra_headers") or {}
    if isinstance(extra, dict):
        headers.update({str(k): str(v) for k, v in extra.items()})
    return headers


def _data_uri(value: str, mime_type: str = "image/png") -> str:
    if value.startswith("data:"):
        return value
    return f"data:{mime_type};base64,{value}"


def data_uri_bytes(value: str) -> int:
    """Return decoded payload bytes for a normalized base64 data URI."""
    if not isinstance(value, str) or not value.startswith("data:") or "," not in value:
        return 0
    try:
        return len(base64.b64decode(value.split(",", 1)[1], validate=True))
    except (binascii.Error, ValueError, TypeError):
        return 0


def image_results_bytes(results: list[ImageGenerationResult]) -> int:
    """Return decoded image payload bytes for statistics and logging."""
    return sum(data_uri_bytes(result.data_uri) for result in results)


def _mime_from_bytes(data: bytes, fallback: str = "image/png") -> str:
    if data.startswith(b"\xFF\xD8\xFF"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return fallback if fallback.startswith("image/") else "image/png"


def _is_grok_image_backend(config: dict) -> bool:
    provider = str(config.get("provider_id") or "").lower()
    model = str(config.get("model") or config.get("provider_model") or "").lower()
    return provider in {"grok", "supergrok", "xai"} or "grok-imagine" in model


def _grok_image_options(size: str | None) -> dict[str, str]:
    """Translate OpenAI size values to xAI Imagine controls."""
    value = str(size or "").strip().lower()
    # Codex's built-in image_gen extension always sends size="auto".  Grok
    # Imagine does not accept that OpenAI sentinel, so let the backend choose
    # its default aspect ratio and resolution.
    if not value or value == "auto":
        return {}
    match = re.fullmatch(r"(\d+)x(\d+)", value)
    if not match:
        raise ValueError(f"unsupported image size: {size}")
    width, height = (int(part) for part in match.groups())
    divisor = math.gcd(width, height)
    ratio = f"{width // divisor}:{height // divisor}"
    supported = {"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2:1", "1:2", "19.5:9", "9:19.5", "20:9", "9:20"}
    if ratio not in supported:
        raise ValueError(f"Grok Imagine does not support aspect ratio derived from size {size}: {ratio}")
    return {"aspect_ratio": ratio, "resolution": "2k" if max(width, height) > 1024 else "1k"}


async def _download_image(client: httpx.AsyncClient, url: str, max_bytes: int = 25 * 1024 * 1024) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("image result URL must use http or https")
    response = await client.get(url, follow_redirects=False)
    response.raise_for_status()
    if len(response.content) > max_bytes:
        raise ValueError("image result exceeds the configured size limit")
    mime_type = (response.headers.get("content-type") or "").split(";", 1)[0].strip()
    mime_type = _mime_from_bytes(response.content, mime_type or mimetypes.guess_type(parsed.path)[0] or "image/png")
    if not mime_type.startswith("image/"):
        raise ValueError("image result URL did not return an image")
    return f"data:{mime_type};base64,{base64.b64encode(response.content).decode('ascii')}", mime_type


async def generate_images(config: dict, *, prompt: str, model: str | None = None,
                          n: int = 1, size: str | None = None, quality: str | None = None,
                          background: str | None = None, output_format: str | None = None,
                          extra: dict[str, Any] | None = None) -> list[ImageGenerationResult]:
    """Call an OpenAI Images-compatible backend and normalize its result."""
    backend_type = config.get("backend_type") or "existing_model"
    # openai_images is retained as a read-only compatibility alias for old
    # configurations; new configuration uses existing_model or external_model.
    if backend_type == "comfyui":
        raise ValueError("ComfyUI image generation backend is not implemented yet")
    if backend_type not in {"existing_model", "external_model", "openai_images"}:
        raise ValueError(f"unsupported image backend type: {config.get('backend_type')}")
    request_model = model or config.get("model") or parse_model_id(config.get("provider_model") or "").model_name
    if not request_model:
        raise ValueError("image generation model is not configured")
    payload: dict[str, Any] = {"model": request_model, "prompt": prompt, "n": max(1, min(int(n or 1), 10)), "response_format": "b64_json"}
    is_grok = _is_grok_image_backend(config)
    if is_grok and size:
        payload.update(_grok_image_options(size))
    for key, value in (("size", size), ("quality", quality), ("background", background), ("output_format", output_format)):
        # xAI Imagine's OpenAI-compatible endpoint accepts aspect_ratio and
        # resolution, but rejects OpenAI image controls such as quality,
        # background and output_format. The response MIME type is detected
        # from the returned bytes, so dropping them is lossless.
        if is_grok:
            continue
        if value not in (None, ""):
            payload[key] = value
    if isinstance(extra, dict):
        payload.update({k: v for k, v in extra.items() if k not in {"model", "prompt", "n", "response_format"} and v is not None})
    timeout = max(1, int(config.get("timeout") or 180))
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(images_url(config.get("api_base") or ""), headers=_headers(config), json=payload)
        status_code = int(getattr(response, "status_code", 200))
        if status_code >= 400:
            detail = response.text[:1000].strip()
            suffix = f": {detail}" if detail else ""
            raise ValueError(f"image backend returned HTTP {response.status_code}{suffix}")
        body = response.json()
        results: list[ImageGenerationResult] = []
        for item in body.get("data") or []:
            if not isinstance(item, dict):
                continue
            revised = item.get("revised_prompt")
            if item.get("b64_json"):
                try:
                    raw = base64.b64decode(str(item["b64_json"]), validate=True)
                except (binascii.Error, ValueError, TypeError):
                    raise ValueError("image backend returned invalid base64 data") from None
                mime_type = _mime_from_bytes(raw)
                data_uri = _data_uri(str(item["b64_json"]), mime_type)
            elif item.get("url"):
                data_uri, mime_type = await _download_image(client, str(item["url"]))
            else:
                continue
            results.append(ImageGenerationResult(data_uri=data_uri, mime_type=mime_type, revised_prompt=revised, size=size, quality=quality, output_format=output_format, background=background, usage=body.get("usage") or {}))
        if not results:
            raise RuntimeError("image backend returned no image data")
        return results
