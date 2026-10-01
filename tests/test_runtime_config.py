"""运行时配置推送的测试：把"改了不生效"的固化值变成改了立即生效。

背景：大多数 defaults 键是每次调用现读；但有几类值被固化在对象里
（TTLDict 的 ttl/max_size、内存请求日志的 deque.maxlen、litellm 的全局超时），
必须靠 core.runtime_config 的钩子在写入后推送。这些测试锁的就是这条承诺。
"""

import json
import threading
from collections import deque

import litellm
import pytest

from app.config import load_config
from app.core import runtime_config
from app.core.runtime_config import apply_runtime_config, register_runtime_hook
from app.core.state import (
    TTLDict,
    configure_caches,
    image_generation_budget,
    reasoning_cache,
    reasoning_tool_cache,
    response_chain_cache,
    tool_only_limit,
    tool_only_turns,
)


@pytest.fixture
def temp_config(tmp_path):
    path = tmp_path / "config.json"
    # 必须用 json.dumps：Windows 路径里的反斜杠直接拼进 JSON 字符串会变成非法转义。
    path.write_text(json.dumps({
        "host": "127.0.0.1",
        "port": 8000,
        "database": str(tmp_path / "d.db"),
    }), encoding="utf-8")
    cfg = load_config(str(path), force_reload=True)
    yield cfg
    load_config(str(tmp_path / "teardown.json"), force_reload=True)


@pytest.fixture
def restore_cache_limits():
    """缓存上限是进程级状态，测试结束必须复原，避免污染同进程后续用例。"""
    caches = [tool_only_turns, reasoning_cache, reasoning_tool_cache, response_chain_cache, image_generation_budget]
    snapshot = [(cache.ttl, cache.max_size) for cache in caches]
    yield caches
    for cache, (ttl, max_size) in zip(caches, snapshot):
        cache.configure(ttl, max_size)


def test_register_hook_replaces_same_name():
    calls = []
    register_runtime_hook("probe", lambda cfg: calls.append("first"))
    register_runtime_hook("probe", lambda cfg: calls.append("second"))
    assert runtime_config.registered_hooks().count("probe") == 1
    apply_runtime_config()
    assert calls == ["second"]
    runtime_config._HOOKS[:] = [(n, fn) for n, fn in runtime_config._HOOKS if n != "probe"]


def test_failing_hook_is_reported_not_fatal():
    register_runtime_hook("boom", lambda cfg: 1 / 0)
    register_runtime_hook("ok_probe", lambda cfg: None)
    results = apply_runtime_config()
    assert results["boom"].startswith("failed: ZeroDivisionError")
    assert results["ok_probe"] == "applied"
    runtime_config._HOOKS[:] = [(n, fn) for n, fn in runtime_config._HOOKS if n not in ("boom", "ok_probe")]


def test_ttl_dict_configure_evicts_on_shrink():
    cache = TTLDict(ttl_seconds=100, max_size=10)
    for index in range(8):
        cache[f"k{index}"] = index
    assert cache.configure(50, 3) is True
    assert cache.ttl == 50
    assert len(cache) == 3
    # 未保留的键被淘汰，保留的总是最近写入的
    assert cache.get("k7") == 7
    assert cache.get("k0") is None
    assert cache.configure(50, 3) is False


def test_configure_caches_pushes_ttl_and_size(temp_config, restore_cache_limits):
    changed = configure_caches({"reasoning_cache_ttl": 60, "reasoning_cache_max_size": 12})
    assert "reasoning_cache_ttl/reasoning_cache_max_size" in changed
    # 四个缓存共用同一组键，必须一起跟上
    assert reasoning_cache.ttl == 60 and reasoning_cache.max_size == 12
    assert reasoning_tool_cache.ttl == 60
    assert response_chain_cache.ttl == 60
    assert tool_only_turns.ttl == 600  # 未提交的键保持自己的默认


def test_configure_caches_falls_back_on_dirty_values(temp_config, restore_cache_limits):
    configure_caches({"reasoning_cache_ttl": "abc", "reasoning_cache_max_size": -5})
    assert reasoning_cache.ttl == 1800
    assert reasoning_cache.max_size == 1000


def test_apply_runtime_config_updates_caches_end_to_end(temp_config, restore_cache_limits):
    temp_config.patch("defaults", {"reasoning_cache_ttl": 90})
    results = apply_runtime_config()
    assert results["state_caches"] == "applied"
    assert reasoning_cache.ttl == 90


def test_apply_runtime_config_updates_request_log_deque(temp_config, restore_cache_limits):
    from app.router import proxy

    original = proxy._request_log
    # 模拟真实写入路径：_record_request_log 用 appendleft，最新条目在左端。
    log = deque(maxlen=200)
    for index in range(6):
        log.appendleft({"id": index})
    proxy._request_log = log
    try:
        temp_config.patch("defaults", {"request_log_max": 4})
        assert apply_runtime_config()["request_log"] == "applied"
        assert proxy._request_log.maxlen == 4
        # 缩小上限必须保留“最近”的 4 条（id 5..2），而不是最旧的几条
        assert [entry["id"] for entry in proxy._request_log] == [5, 4, 3, 2]
        # <=0 与重启路径同语义（内存滚动日志关闭），不是回退成 200
        assert proxy.set_request_log_max(0) is True
        assert proxy._request_log.maxlen == 0
        assert len(proxy._request_log) == 0
    finally:
        proxy._request_log = original


def test_set_request_log_max_is_noop_when_unchanged(temp_config):
    from app.router import proxy

    before = proxy._request_log
    assert proxy.set_request_log_max(before.maxlen) is False
    assert proxy._request_log is before


def test_apply_runtime_config_updates_litellm_timeout(temp_config):
    before = litellm.request_timeout
    try:
        temp_config.patch("defaults", {"litellm_request_timeout": 33})
        assert apply_runtime_config()["litellm_timeout"] == "applied"
        assert litellm.request_timeout == 33
    finally:
        litellm.request_timeout = before


def test_lazy_read_keys_follow_config(temp_config, monkeypatch):
    from app.security import session_ttl_hours
    from app.services.lite_llm import min_image_max_tokens

    temp_config.patch("defaults", {
        "tool_only_limit": 3,
        "min_image_max_tokens": 777,
        "session_ttl_hours": 5,
    })
    assert tool_only_limit() == 3
    assert min_image_max_tokens() == 777
    assert session_ttl_hours() == 5


def test_lazy_read_keys_reject_zero_as_disable(temp_config):
    """0/负数 = 关闭该保护，不得被静默改回默认值。"""
    from app.services.lite_llm import min_image_max_tokens

    temp_config.patch("defaults", {"tool_only_limit": 0, "min_image_max_tokens": 0})
    assert tool_only_limit() == 0
    assert min_image_max_tokens() == 0


def test_lazy_read_keys_fall_back_on_garbage(temp_config):
    from app.security import session_ttl_hours
    from app.services.lite_llm import min_image_max_tokens

    temp_config.patch("defaults", {"session_ttl_hours": "abc", "min_image_max_tokens": None})
    assert session_ttl_hours() == 12
    assert min_image_max_tokens() == 2000


def test_session_ttl_only_affects_new_sessions(temp_config):
    """改 TTL 不得把已登录会话拉长/缩短，界面文案依赖这条语义。"""
    from app.security import create_session, get_session_username, session_ttl_hours
    from datetime import datetime, timedelta
    from app import security

    temp_config.patch("defaults", {"session_ttl_hours": 100})
    token = create_session("admin")
    expires = security._sessions[token]["expires_at"]
    temp_config.patch("defaults", {"session_ttl_hours": 1})
    assert security._sessions[token]["expires_at"] == expires
    assert session_ttl_hours() == 1
    assert get_session_username(token) == "admin"
    assert expires > datetime.now(expires.tzinfo) + timedelta(hours=99)


def test_maintenance_interval_is_read_per_iteration(temp_config):
    import main

    temp_config.patch("defaults", {"storage_maintenance_interval_seconds": 120})
    assert main._maintenance_interval() == 120
    temp_config.patch("defaults", {"storage_maintenance_interval_seconds": 1})
    assert main._maintenance_interval() == 5  # 下限，避免维护循环忙等
    temp_config.patch("defaults", {"storage_maintenance_interval_seconds": "junk"})
    assert main._maintenance_interval() == 60


def test_state_cache_locks_are_thread_safe_under_reconfigure(restore_cache_limits):
    """推送上限与并发写入同时发生不得崩，配置页写入不是独占操作。"""
    stop = threading.Event()
    errors = []

    def writer():
        index = 0
        while not stop.is_set():
            try:
                tool_only_turns[f"k{index % 50}"] = index
                tool_only_turns.get(f"k{index % 50}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            index += 1

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        for size in (10, 5, 50, 1000):
            tool_only_turns.configure(600, size)
    finally:
        stop.set()
        thread.join(timeout=2)
    assert errors == []
