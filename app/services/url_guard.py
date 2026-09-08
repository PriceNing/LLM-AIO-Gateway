"""管理员提供的上游地址校验。

网关的正常用法包含指向局域网自建推理服务（llama.cpp、vLLM、Ollama 等），
因此**不能**一刀切禁止私网地址。这里只拦掉真正有害的部分（见「当前问题.md」S6）：

* 非 http(s) scheme：``file://``、``gopher://`` 等不应用作上游基址。
* 云元数据端点与链路本地/组播/保留地址：无论配置如何都拒绝。
* 私网与回环地址：由 ``allow_private_upstream_hosts`` 决定，默认放行以保持
  现有部署可用，生产如需收紧可关闭。
"""

from __future__ import annotations

import asyncio
import ipaddress
from urllib.parse import urlsplit

from app.config import get_default

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# 云厂商元数据服务地址与常见主机名，任何配置下都不允许作为上游。
_BLOCKED_HOSTNAMES = frozenset({
    "metadata",
    "metadata.google.internal",
    "metadata.goog",
    "instance-data",
})

# 解析后命中这些类别的 IP 一律拒绝。
_ALWAYS_BLOCKED_IP_FLAGS = ("is_link_local", "is_multicast", "is_reserved", "is_unspecified")


class UnsafeUpstreamURL(ValueError):
    """Raised when an administrator-supplied upstream address is not usable."""


def default_allow_private_hosts() -> bool:
    return bool(get_default("allow_private_upstream_hosts", True))


def _blocked_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    for flag in _ALWAYS_BLOCKED_IP_FLAGS:
        if getattr(address, flag, False):
            return True
    # 169.254.169.254 在部分实现里不算 link-local，显式按元数据地址处理。
    return address.version == 4 and str(address) == "169.254.169.254"


def _resolve_addresses(hostname: str) -> list[str]:
    import socket

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return []
    return sorted({info[4][0] for info in infos})


def validate_upstream_url(
    url: str,
    *,
    field: str = "api_base",
    allow_private_hosts: bool | None = None,
) -> str:
    """Validate scheme and host of an upstream base URL. Returns the stripped URL.

    同步版本：只做字面校验，不做 DNS 解析，用于无法 await 的路径。
    """
    raw = str(url or "").strip()
    if not raw:
        raise UnsafeUpstreamURL(f"{field} is required")
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        raise UnsafeUpstreamURL(f"{field} is not a valid URL") from exc

    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UnsafeUpstreamURL(f"{field} must use http or https")
    hostname = parsed.hostname or ""
    if not hostname:
        raise UnsafeUpstreamURL(f"{field} must include a host")
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise UnsafeUpstreamURL(f"{field} points at a blocked metadata endpoint")

    allow_private = default_allow_private_hosts() if allow_private_hosts is None else bool(allow_private_hosts)
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if _blocked_ip(literal):
            raise UnsafeUpstreamURL(f"{field} points at a blocked address")
        if not allow_private and (literal.is_private or literal.is_loopback):
            raise UnsafeUpstreamURL(
                f"{field} points at a private address; enable allow_private_upstream_hosts if intentional"
            )
    return raw


async def validate_upstream_url_async(
    url: str,
    *,
    field: str = "api_base",
    allow_private_hosts: bool | None = None,
) -> str:
    """Literal validation plus DNS resolution, so a hostname cannot resolve to
    a metadata or (when disallowed) private address."""
    cleaned = validate_upstream_url(url, field=field, allow_private_hosts=allow_private_hosts)
    hostname = urlsplit(cleaned).hostname or ""
    try:
        ipaddress.ip_address(hostname)
        return cleaned
    except ValueError:
        pass

    allow_private = default_allow_private_hosts() if allow_private_hosts is None else bool(allow_private_hosts)
    addresses = await asyncio.to_thread(_resolve_addresses, hostname)
    for value in addresses:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            continue
        if _blocked_ip(address):
            raise UnsafeUpstreamURL(f"{field} resolves to a blocked address")
        if not allow_private and (address.is_private or address.is_loopback):
            raise UnsafeUpstreamURL(
                f"{field} resolves to a private address; enable allow_private_upstream_hosts if intentional"
            )
    return cleaned
