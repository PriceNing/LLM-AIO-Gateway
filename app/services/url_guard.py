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
import os
import threading
import time
from dataclasses import dataclass
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

# 这些后缀的主机名同样按元数据端点处理（裸标签名 + 限定形式都拦，
# bug-2026-10-05 M-4④：旧实现只做 4 个精确匹配，"abc.metadata.goog"
# 或 FQDN 尾点形式即绕过）。
_BLOCKED_HOST_SUFFIXES = (".metadata.goog", ".metadata.google.internal", ".instance-data")

# 解析后无条件封禁的 IPv6 网段：AWS IMSE 的 IPv6 元数据地址落在 ULA 段内
# （fd00:ec2::/104），allow_private 默认放行时旧实现会把它当普通 ULA 放过。
_ALWAYS_BLOCKED_IPV6_NETWORKS = (
    ipaddress.ip_network("fd00:ec2::/104"),
)

# 解析后命中这些类别的 IP 一律拒绝。
_ALWAYS_BLOCKED_IP_FLAGS = ("is_link_local", "is_multicast", "is_reserved", "is_unspecified")


class UnsafeUpstreamURL(ValueError):
    """Raised when an administrator-supplied upstream address is not usable."""


class UnresolvableHostname(UnsafeUpstreamURL):
    """DNS 解析不出任何地址。

    保存路径按 ``url_guard_require_resolvable`` 决定拒绝/留痕；请求路径
    （``pinned_request``）对它降级为原样放行——httpx 自行解析同样会失败，
    不应把 DNS 故障变成新的错误形态（bug-2026-10-05 R-1）。
    """


def default_allow_private_hosts() -> bool:
    return bool(get_default("allow_private_upstream_hosts", True))


def default_require_resolvable() -> bool:
    """DNS 解析失败是否拒绝保存。

    默认 False（留痕放行），保持既有刻意行为：上游地址保存是管理员配置场景，
    离线/内网/拆分 DNS 环境下硬拒绝会阻断正常配置，且大量部署用不可解析主机名。
    真正的 rebinding 防线在请求时（imagegen 下载钉 IP）；需要收紧保存路径的
    部署可置 True，让 NXDOMAIN 直接拒绝。
    """
    return bool(get_default("url_guard_require_resolvable", False))


def is_blocked_hostname(hostname: str) -> bool:
    """主机名是否命中恒封的元数据名单（精确 + 后缀，含 FQDN 尾点）。"""
    lowered = str(hostname or "").lower().rstrip(".")
    return lowered in _BLOCKED_HOSTNAMES or lowered.endswith(_BLOCKED_HOST_SUFFIXES)


def is_blocked_ip_address(value: "str | ipaddress.IPv4Address | ipaddress.IPv6Address") -> bool:
    """恒封地址判定（元数据/链路本地/组播/保留/未指定，含 IPv4-mapped 解包）。

    无法解析的字符串按封禁处理（宁可拒绝）。供 imagegen 等自有解析路径
    复用同一套绝对阻断规则，避免"配置开关顺手关掉恒封检查"（R-2）。
    """
    if isinstance(value, str):
        try:
            value = ipaddress.ip_address(value)
        except ValueError:
            return True
    return _blocked_ip(value)


def _blocked_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # IPv4-mapped IPv6（::ffff:169.254.169.254）先解包再判，否则所有 v4 规则
    # 对它整体失明（M-4④）。
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    for flag in _ALWAYS_BLOCKED_IP_FLAGS:
        if getattr(address, flag, False):
            return True
    if address.version == 6:
        return any(address in network for network in _ALWAYS_BLOCKED_IPV6_NETWORKS)
    # 169.254.169.254 在部分实现里不算 link-local，显式按元数据地址处理。
    return str(address) == "169.254.169.254"


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
    if is_blocked_hostname(hostname):
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


@dataclass(frozen=True)
class PinnedTarget:
    """一次解析、按解析结果直连的目标：URL 里 host 换成 IP 字面量。

    httpx/httpcore 会再次独立解析 DNS，两次解析之间答案可变（rebinding
    TOCTOU，bug-2026-10-05 M-4②）。钉 IP 后连接只走校验过的那一个地址；
    TLS 的 SNI 与证书主机名用 request extensions 里的 ``sni_hostname`` 保留
    原主机名，证书校验语义不变。
    """

    url: str
    host_header: str
    sni_hostname: str
    # URL 的 host 是否真的被改写成 IP 字面量。未改写（IP 字面量输入、
    # 降级放行）时不得注入 Host/SNI 覆盖，否则会改写调用方语义，且对
    # IP 设 SNI 本身违反 TLS 规范（bug-2026-10-05 R-1）。
    rewritten: bool = False

    @property
    def headers(self) -> dict:
        return {"Host": self.host_header} if (self.rewritten and self.host_header) else {}

    @property
    def extensions(self) -> dict:
        return {"sni_hostname": self.sni_hostname} if (self.rewritten and self.sni_hostname) else {}

    def rewrite_url(self, url: str) -> str:
        """把同一钉住结果应用到同主机的另一个 URL（多端点共用一个 client）。

        未发生改写时原样返回；改写时只替换 netloc，路径/查询/端口保留。
        调用方必须保证传入 URL 的主机与本目标是同一个 origin。
        """
        if not self.rewritten:
            return url
        parsed = urlsplit(str(url))
        pinned = urlsplit(self.url)
        return parsed._replace(netloc=pinned.netloc).geturl()


async def resolve_pinned_target(url: str, *, field: str = "api_base") -> PinnedTarget:
    """校验并解析 URL 主机，返回钉 IP 后的请求目标（字面 IP 主机原样返回）。"""
    cleaned = validate_upstream_url(url, field=field)
    parsed = urlsplit(cleaned)
    hostname = parsed.hostname or ""
    try:
        ipaddress.ip_address(hostname)
        return PinnedTarget(url=cleaned, host_header=parsed.netloc, sni_hostname=hostname)
    except ValueError:
        pass
    addresses = await asyncio.to_thread(_resolve_addresses, hostname)
    if not addresses:
        raise UnresolvableHostname(f"{field} hostname could not be resolved: {hostname}")
    allow_private = default_allow_private_hosts()
    chosen = None
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
        if chosen is None or (chosen.version == 6 and address.version == 4):
            # 双栈时优先 IPv4：v6-only 网络里 v4 会立即失败并由调用方报错，
            # 但常见部署是 v4 可达而 v6 路由缺失，优先 v4 更稳。
            chosen = address
    if chosen is None:
        raise UnsafeUpstreamURL(f"{field} hostname resolved to no usable address")
    port = f":{parsed.port}" if parsed.port else ""
    # ipaddress 对象没有 .ip 属性（本函数此前无调用方，这条路径从未被执行过）。
    literal = f"[{chosen}]" if chosen.version == 6 else str(chosen)
    pinned = parsed._replace(netloc=f"{literal}{port}").geturl()
    return PinnedTarget(url=pinned, host_header=parsed.netloc, sni_hostname=hostname, rewritten=True)


# 请求时钉 IP（管理员配置上游，热路径）：TTL 缓存 + 降级放行。
# 与 imagegen 下载（不可信 URL、逐请求解析）不同，这里的地址由管理员
# 配置，每请求重新解析会平白引入 DNS 抖动；"连接刚校验过的地址"在 TTL
# 窗口内依然成立，rebinding 窗口从"两次解析之间"收敛为"至多 TTL 秒"。
_PIN_CACHE_TTL_SECONDS = 60.0
_PIN_CACHE_MAX = 256
_pinned_cache: dict = {}
_pinned_cache_lock = threading.Lock()

_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def _env_proxy_configured() -> bool:
    """环境配置了 http(s) 代理时不钉 IP：真正连接由代理发起，网关侧改写
    URL 只会破坏代理的路由与证书语义；SSRF 防线随之转移到代理配置本身。"""
    return any(os.environ.get(key, "").strip() for key in _PROXY_ENV_KEYS)


def clear_pinned_target_cache() -> None:
    """清空钉 IP 缓存（测试与配置热变更用）。"""
    with _pinned_cache_lock:
        _pinned_cache.clear()


async def pinned_request(url: str, *, field: str = "api_base") -> PinnedTarget:
    """管理员配置上游的请求时钉 IP：一次解析、按解析结果直连（带缓存）。

    调用方把 ``target.url`` 交给 httpx，并合并 ``target.headers`` /
    ``target.extensions``；未改写时两者为空，请求与原行为完全一致。

    降级与失败语义（bug-2026-10-05 R-1）：
    * 主机名解析失败 → 原样放行（httpx 自行解析同样失败，不改变错误形态）；
    * 解析命中恒封地址（rebinding 已发生）→ 抛 ConnectionError，与连接失败
      同口径进入 fallback 的 connection_error 触发器；
    * 环境存在代理 → 原样放行（见 ``_env_proxy_configured``）。
    """
    passthrough = PinnedTarget(url=url, host_header="", sni_hostname="", rewritten=False)
    if _env_proxy_configured():
        return passthrough
    key = f"{int(default_allow_private_hosts())}|{url}"
    now = time.monotonic()
    with _pinned_cache_lock:
        hit = _pinned_cache.get(key)
        if hit is not None and now - hit[0] < _PIN_CACHE_TTL_SECONDS:
            return hit[1]
    try:
        target = await resolve_pinned_target(url, field=field)
    except UnresolvableHostname:
        return passthrough
    except UnsafeUpstreamURL as exc:
        raise ConnectionError(str(exc)) from exc
    with _pinned_cache_lock:
        _pinned_cache[key] = (time.monotonic(), target)
        if len(_pinned_cache) > _PIN_CACHE_MAX:
            stale = sorted(_pinned_cache, key=lambda k: _pinned_cache[k][0])[: len(_pinned_cache) - _PIN_CACHE_MAX]
            for stale_key in stale:
                _pinned_cache.pop(stale_key, None)
    return target


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
    if not addresses:
        # 旧行为是留痕放行，但这正是 rebinding 的入口：先让域名 NXDOMAIN 通过
        # 保存，事后把 A 记录指向元数据地址，请求时命中（bug-2026-10-05 M-4①）。
        # 默认改为硬拒绝；确实存在"保存期 DNS 不可用"的内网部署时，用
        # url_guard_require_resolvable=false 显式退回旧行为。
        if default_require_resolvable():
            raise UnsafeUpstreamURL(
                f"{field} hostname could not be resolved; fix DNS or set url_guard_require_resolvable=false"
            )
        import logging

        logging.getLogger("llmgw.app").warning(
            "[url_guard] %s hostname could not be resolved during validation: %s",
            field,
            hostname,
        )
        return cleaned
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
