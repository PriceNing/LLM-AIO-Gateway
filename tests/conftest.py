"""全局测试 fixture：确定性 DNS。

url_guard 的请求时钉 IP（``pinned_request``）经 ``_resolve_addresses`` 走真实
getaddrinfo。测试若沿用真实解析器，结果取决于环境：``example.com`` 会真解析并
被改写成 IP（破坏对 mock client 的 URL 断言），``*.invalid`` 则 NXDOMAIN——
两者行为不一致且依赖网络。默认统一为"不可解析"，命中 ``pinned_request`` 的
降级放行路径，mock client 看到的 URL 与钉 IP 接线前完全一致；需要特定解析
结果的测试自行 monkeypatch ``app.services.url_guard._resolve_addresses``
覆盖本 fixture（函数级 setattr 在 autouse 之后执行，后写生效）。

同时清理进程级钉 IP 缓存，防跨测试泄漏。
"""

import pytest


@pytest.fixture(autouse=True)
def deterministic_dns(monkeypatch):
    import app.services.url_guard as url_guard

    url_guard.clear_pinned_target_cache()
    monkeypatch.setattr(url_guard, "_resolve_addresses", lambda hostname: [])
    yield
    url_guard.clear_pinned_target_cache()
