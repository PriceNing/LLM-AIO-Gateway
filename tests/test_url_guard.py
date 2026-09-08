"""上游地址校验的行为测试（「当前问题.md」S6）。"""

import pytest

from app.services.url_guard import (
    UnsafeUpstreamURL,
    validate_upstream_url,
    validate_upstream_url_async,
)


# -- scheme --

@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "gopher://127.0.0.1:11211/",
    "ftp://example.com",
    "http+unix://%2Fvar%2Frun%2Fsock",
])
def test_non_http_scheme_rejected(url):
    with pytest.raises(UnsafeUpstreamURL):
        validate_upstream_url(url)


def test_empty_url_rejected():
    with pytest.raises(UnsafeUpstreamURL):
        validate_upstream_url("")


def test_missing_host_rejected():
    with pytest.raises(UnsafeUpstreamURL):
        validate_upstream_url("http:///v1")


# -- 元数据与保留地址：任何配置下都拒绝 --

@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://metadata.google.internal/computeMetadata/v1/",
    "http://[fe80::1]/v1",
    "http://224.0.0.1/v1",
])
def test_metadata_and_reserved_addresses_always_blocked(url):
    with pytest.raises(UnsafeUpstreamURL):
        validate_upstream_url(url, allow_private_hosts=True)


# -- 局域网自建推理服务：默认必须继续可用 --

@pytest.mark.parametrize("url", [
    "http://192.168.75.202:8080/v1",
    "http://10.0.0.5:11434",
    "http://localhost:8000/v1",
])
def test_private_hosts_allowed_by_default(url):
    assert validate_upstream_url(url) == url


@pytest.mark.parametrize("url", [
    "http://192.168.75.202:8080/v1",
    "http://10.0.0.5:11434",
])
def test_private_hosts_rejected_when_disallowed(url):
    with pytest.raises(UnsafeUpstreamURL):
        validate_upstream_url(url, allow_private_hosts=False)


def test_hostname_is_not_judged_without_dns():
    """字面校验不解析主机名；localhost 的归类由 async 版本负责。"""
    assert validate_upstream_url("http://localhost:8000/v1", allow_private_hosts=False) \
        == "http://localhost:8000/v1"


# -- 公网地址始终可用 --

@pytest.mark.parametrize("url", [
    "https://api.deepseek.com",
    "https://api.openai.com/v1",
])
def test_public_https_accepted(url):
    assert validate_upstream_url(url) == url


def test_returns_stripped_value():
    assert validate_upstream_url("  https://api.test/v1  ") == "https://api.test/v1"


# -- DNS 解析版本：主机名不得解析到元数据地址 --

@pytest.mark.asyncio
async def test_async_validation_rejects_hostname_resolving_to_metadata(monkeypatch):
    import app.services.url_guard as module

    monkeypatch.setattr(module, "_resolve_addresses", lambda host: ["169.254.169.254"])
    with pytest.raises(UnsafeUpstreamURL):
        await validate_upstream_url_async("http://evil.example")


@pytest.mark.asyncio
async def test_async_validation_rejects_hostname_resolving_to_private_when_disallowed(monkeypatch):
    import app.services.url_guard as module

    monkeypatch.setattr(module, "_resolve_addresses", lambda host: ["10.1.2.3"])
    with pytest.raises(UnsafeUpstreamURL):
        await validate_upstream_url_async("http://evil.example", allow_private_hosts=False)


@pytest.mark.asyncio
async def test_async_validation_passes_public_hostname(monkeypatch):
    import app.services.url_guard as module

    monkeypatch.setattr(module, "_resolve_addresses", lambda host: ["93.184.216.34"])
    assert await validate_upstream_url_async("https://example.com/v1") == "https://example.com/v1"


@pytest.mark.asyncio
async def test_async_validation_skips_dns_for_literal_ip():
    assert await validate_upstream_url_async("http://192.168.1.50:8080/v1") == "http://192.168.1.50:8080/v1"


@pytest.mark.asyncio
async def test_async_validation_resolves_localhost_when_private_disallowed(monkeypatch):
    import app.services.url_guard as module

    monkeypatch.setattr(module, "_resolve_addresses", lambda host: ["127.0.0.1"])
    with pytest.raises(UnsafeUpstreamURL):
        await validate_upstream_url_async("http://localhost:8000/v1", allow_private_hosts=False)
