"""Adapters for OpenAI-compatible image-generation backends."""

import base64
import binascii
import asyncio
import email.utils
import ipaddress
import math
import mimetypes
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from app.database import parse_model_id


class ImageBackendHTTPError(ValueError):
    """HTTP failure from an image backend with retry-relevant metadata."""

    def __init__(self, status_code: int, detail: str = "", retry_after: float | None = None):
        suffix = f": {detail}" if detail else ""
        super().__init__(f"image backend returned HTTP {status_code}{suffix}")
        self.status_code = int(status_code)
        self.detail = detail
        self.retry_after = retry_after


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        return max(0.0, parsed.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


def _is_transient_status(status_code: int) -> bool:
    return status_code == 429 or status_code in {500, 502, 503, 504}


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
    backend_attempts: int = 1


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


# 生图后端参数档位：profile 决定 size 如何发送、以及是否携带 OpenAI 专有参数。
# - "openai"：标准 OpenAI Images 形状（size/quality/background/output_format 全发）。
# - "aspect_ratio"：接受 aspect_ratio+resolution、拒绝 OpenAI 专有参数
#   （xAI Imagine 的 OpenAI 兼容端点即属此档）。
_IMAGE_PARAM_PROFILES = {"openai", "aspect_ratio"}

# 内置启发式表：仅当生成器未显式配置 image_param_profile 时用于推断默认档位，
# 与 model_capabilities 的内置家族表同构——身份判断集中在数据表 + 可被配置覆盖，
# 不散落 if 分支（AGENTS.md 兼容补丁边界规则第 2/3 条，bug-2026-10-05 M-22）。
_BUILTIN_IMAGE_PARAM_PROFILE_RULES = (
    ("aspect_ratio", {"grok", "supergrok", "xai"}, ("grok-imagine",)),
)


def _resolve_image_param_profile(config: dict) -> str:
    """解析生图后端的参数档位（配置优先，内置表兑底，默认 openai）。"""
    explicit = str(config.get("image_param_profile") or "").strip().lower()
    if explicit in _IMAGE_PARAM_PROFILES:
        return explicit
    provider = str(config.get("provider_id") or "").lower()
    model = str(config.get("model") or config.get("provider_model") or "").lower()
    for profile, providers, model_tokens in _BUILTIN_IMAGE_PARAM_PROFILE_RULES:
        if provider in providers or any(token in model for token in model_tokens):
            return profile
    return "openai"


def _is_grok_image_backend(config: dict) -> bool:
    """@deprecated 兼容旧调用：等价于档位为 aspect_ratio。新代码用 _resolve_image_param_profile。"""
    return _resolve_image_param_profile(config) == "aspect_ratio"


def _aspect_ratio_options(size: str | None) -> dict[str, str]:
    """Translate OpenAI size values to aspect_ratio+resolution (aspect_ratio profile)."""
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


def _is_public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


async def _validate_download_host(hostname: str, *, allow_private_hosts: bool) -> str | None:
    """校验下载主机；返回需要钉进连接的已验证 IP 字面量（None = 无需改写）。

    只校验不复用结果等于没校验：getaddrinfo 与 httpx 各自解析一次，两次之间
    DNS 答案可变（rebinding TOCTOU，bug-2026-10-05 M-4②）。这里把校验过的
    那一个地址返回给调用方，由它改写请求 URL，httpx 不再自行解析。

    ``allow_private_hosts`` 只放宽普通私网/回环一档；元数据主机名与恒封
    地址（链路本地/组播/保留/未指定/169.254.169.254）无论配置如何都检查，
    域名也不因该开关跳过解析（bug-2026-10-05 R-2，对齐 url_guard 的
    “元数据地址始终禁止”契约：开启私网下载后 rebinding 到元数据地址
    仍是真实可达的提权路径）。
    """
    from app.services.url_guard import is_blocked_hostname, is_blocked_ip_address

    if not hostname:
        return None
    if is_blocked_hostname(hostname):
        raise ValueError("image result URL points at a blocked metadata endpoint")
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        # URL 本身就是 IP 字面量，httpx 不会再解析，无 TOCTOU 窗口。
        if is_blocked_ip_address(literal):
            raise ValueError("image result URL resolves to a private or unsafe network address")
        if not allow_private_hosts and not _is_public_address(str(literal)):
            raise ValueError("image result URL resolves to a private or unsafe network address")
        return None
    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(f"image result host could not be resolved: {hostname}") from exc
    resolved = {str(item[4][0]).split("%", 1)[0] for item in addresses if item[4]}
    if not resolved:
        raise ValueError(f"image result host could not be resolved: {hostname}")
    # 恒封地址检查与 allow_private 开关无关（R-2）。
    if any(is_blocked_ip_address(address) for address in resolved):
        raise ValueError("image result URL resolves to a private or unsafe network address")
    if not allow_private_hosts and any(not _is_public_address(address) for address in resolved):
        raise ValueError("image result URL resolves to a private or unsafe network address")
    # 双栈优先 IPv4：v6 路由缺失的部署更常见。两种模式下都钉 IP：
    # 私网主机同样存在 rebinding 窗口，只是对地址范围的容忍不同。
    v4 = sorted(item for item in resolved if ":" not in item)
    return (v4 or sorted(resolved))[0]


async def _download_image(
    client: httpx.AsyncClient,
    url: str,
    max_bytes: int = 25 * 1024 * 1024,
    *,
    allow_private_hosts: bool = False,
) -> tuple[str, str]:
    limit = max(64 * 1024, min(100 * 1024 * 1024, int(max_bytes)))
    current_url = url
    for redirect in range(4):
        parsed = urlparse(current_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("image result URL must use http or https")
        if parsed.username or parsed.password:
            raise ValueError("image result URL must not contain credentials")
        pinned_ip = await _validate_download_host(parsed.hostname or "", allow_private_hosts=allow_private_hosts)
        request_url = current_url
        headers: dict[str, str] = {}
        extensions: dict[str, Any] = {}
        if pinned_ip:
            port = f":{parsed.port}" if parsed.port else ""
            literal = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
            request_url = parsed._replace(netloc=f"{literal}{port}").geturl()
            # Host/SNI 保留原主机名：虚拟主机路由与证书校验语义不变；
            # Connection: close 防止同 IP 不同主机的请求复用上一条 TLS 会话
            # （连接池按 origin 分组，改写后的 origin 已不含主机名）。
            headers["Host"] = parsed.netloc
            headers["Connection"] = "close"
            if parsed.scheme == "https":
                extensions["sni_hostname"] = parsed.hostname or ""
        chunks = bytearray()
        async with client.stream(
            "GET", request_url,
            headers=headers or None, extensions=extensions or None,
            follow_redirects=False,
        ) as response:
            if 300 <= response.status_code < 400:
                # raise_for_status() 对 3xx 不报错；不处理重定向会读到空 body
                # 产出空 data URI。手动跟随并逐跳重新校验主机（SSRF 防护）。
                location = response.headers.get("location")
                if not location:
                    raise ValueError("image result URL returned a redirect without Location")
                if redirect == 3:
                    raise ValueError("image result URL exceeded the redirect limit")
                current_url = urljoin(current_url, location)
                continue
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            try:
                declared_length = int(content_length) if content_length is not None else None
            except (TypeError, ValueError):
                declared_length = None
            if declared_length is not None and declared_length > limit:
                raise ValueError("image result exceeds the configured size limit")
            async for chunk in response.aiter_bytes():
                chunks.extend(chunk)
                if len(chunks) > limit:
                    raise ValueError("image result exceeds the configured size limit")
            mime_type = (response.headers.get("content-type") or "").split(";", 1)[0].strip()
        data = bytes(chunks)
        mime_type = _mime_from_bytes(data, mime_type or mimetypes.guess_type(parsed.path)[0] or "image/png")
        if not mime_type.startswith("image/"):
            raise ValueError("image result URL did not return an image")
        return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}", mime_type
    raise ValueError("image result URL exceeded the redirect limit")


async def generate_images(config: dict, *, prompt: str, model: str | None = None,
                          n: int = 1, size: str | None = None, quality: str | None = None,
                          background: str | None = None, output_format: str | None = None,
                          extra: dict[str, Any] | None = None) -> list[ImageGenerationResult]:
    """Call an OpenAI Images-compatible backend and normalize its result."""
    backend_type = config.get("backend_type") or "existing_model"
    # openai_images is retained as a read-only compatibility alias for old
    # configurations; new configuration uses existing_model or external_model.
    if backend_type == "comfyui":
        from app.adapters.comfyui import generate_comfyui_images
        comfy_extra = dict(extra or {})
        if quality not in (None, ""):
            comfy_extra.setdefault("quality", quality)
        if background not in (None, ""):
            comfy_extra.setdefault("background", background)
        if output_format not in (None, ""):
            comfy_extra.setdefault("output_format", output_format)
        return await generate_comfyui_images(
            config, prompt=prompt, n=n, size=size, extra=comfy_extra,
        )
    if backend_type not in {"existing_model", "external_model", "openai_images"}:
        raise ValueError(f"unsupported image backend type: {config.get('backend_type')}")
    request_model = model or config.get("model") or parse_model_id(config.get("provider_model") or "").model_name
    if not request_model:
        raise ValueError("image generation model is not configured")
    payload: dict[str, Any] = {"model": request_model, "prompt": prompt, "n": max(1, min(int(n or 1), 10)), "response_format": "b64_json"}
    param_profile = _resolve_image_param_profile(config)
    if param_profile == "aspect_ratio":
        # 该档位的后端用 aspect_ratio+resolution 表达尺寸，且拒绝 OpenAI 专有
        # 参数（quality/background/output_format）。响应 MIME 由字节嗅探，丢弃无损。
        if size:
            payload.update(_aspect_ratio_options(size))
    else:
        for key, value in (("size", size), ("quality", quality), ("background", background), ("output_format", output_format)):
            if value not in (None, ""):
                payload[key] = value
    if isinstance(extra, dict):
        payload.update({k: v for k, v in extra.items() if k not in {"model", "prompt", "n", "response_format"} and v is not None})
    timeout = max(1, int(config.get("timeout") or 180))
    max_retries = max(0, min(5, int(config["max_retries"] if "max_retries" in config else 2)))
    retry_base = max(0.05, min(30.0, float(config["retry_base_seconds"] if "retry_base_seconds" in config else 1.0)))
    max_retry_delay = max(retry_base, min(120.0, float(config["max_retry_delay_seconds"] if "max_retry_delay_seconds" in config else 30.0)))
    result_max_bytes = max(
        64 * 1024,
        min(100 * 1024 * 1024, int(config.get("result_max_bytes") or 25 * 1024 * 1024)),
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = None
        attempts = 0
        # 请求时钉 IP：生图后端同样是管理员配置的上游（bug-2026-10-05 R-1）。
        from app.services.url_guard import pinned_request

        target = await pinned_request(images_url(config.get("api_base") or ""), field="image_backend_api_base")
        request_kwargs = {"extensions": target.extensions} if target.extensions else {}
        backend_headers = {**_headers(config), **target.headers}
        while True:
            attempts += 1
            try:
                response = await client.post(
                    target.url,
                    headers=backend_headers,
                    json=payload,
                    **request_kwargs,
                )
                status_code = int(getattr(response, "status_code", 200))
                if status_code >= 400:
                    detail = response.text[:1000].strip()
                    error = ImageBackendHTTPError(
                        status_code,
                        detail,
                        _retry_after_seconds(getattr(response, "headers", {}).get("retry-after")),
                    )
                    if not _is_transient_status(status_code) or attempts > max_retries:
                        raise error
                    delay = error.retry_after
                    if delay is None:
                        delay = retry_base * (2 ** (attempts - 1))
                    # 上游 Retry-After 同样封顶：否则一次 429 带 Retry-After: 3600
                    # 就能占住批处理并发信号量数十分钟，后续同批请求全部排队卡死。
                    delay = max(0.0, min(float(delay), max_retry_delay))
                    await asyncio.sleep(delay)
                    continue
                break
            except (httpx.TimeoutException, httpx.TransportError):
                if attempts > max_retries:
                    raise
                await asyncio.sleep(min(max_retry_delay, retry_base * (2 ** (attempts - 1))))
        assert response is not None
        body = response.json()
        results: list[ImageGenerationResult] = []
        for item in body.get("data") or []:
            if not isinstance(item, dict):
                continue
            revised = item.get("revised_prompt")
            if item.get("b64_json"):
                encoded_image = str(item["b64_json"])
                if len(encoded_image) > ((result_max_bytes + 2) // 3) * 4 + 4:
                    raise ValueError("image backend result exceeds the configured size limit")
                try:
                    raw = base64.b64decode(encoded_image, validate=True)
                except (binascii.Error, ValueError, TypeError):
                    raise ValueError("image backend returned invalid base64 data") from None
                if len(raw) > result_max_bytes:
                    raise ValueError("image backend result exceeds the configured size limit")
                mime_type = _mime_from_bytes(raw)
                data_uri = _data_uri(encoded_image, mime_type)
            elif item.get("url"):
                data_uri, mime_type = await _download_image(
                    client,
                    str(item["url"]),
                    max_bytes=result_max_bytes,
                    allow_private_hosts=bool(config.get("allow_private_download_hosts", False)),
                )
            else:
                continue
            results.append(ImageGenerationResult(data_uri=data_uri, mime_type=mime_type, revised_prompt=revised, size=size, quality=quality, output_format=output_format, background=background, usage=body.get("usage") or {}, backend_attempts=attempts))
        if not results:
            raise RuntimeError("image backend returned no image data")
        return results
