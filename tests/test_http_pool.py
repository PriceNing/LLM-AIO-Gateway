"""共享 httpx 客户端池的行为测试（P8，含引用计数边界）。"""

import asyncio

import pytest

import app.services.http_pool as pool


@pytest.fixture(autouse=True)
def isolated_pool():
    """每个用例从空池开始。测试内的客户端不发起请求，因此不持有真实 socket。"""
    pool._entries.clear()
    yield
    pool._entries.clear()


@pytest.mark.asyncio
async def test_same_origin_and_timeout_reuses_client():
    first = await pool.get_shared_client("https://api.anthropic.com/v1", 120)
    second = await pool.get_shared_client("https://api.anthropic.com/v1/messages", 120)
    assert first is second
    assert pool.shared_client_pool_size() == 1


@pytest.mark.asyncio
async def test_different_timeout_creates_separate_client():
    await pool.get_shared_client("https://api.anthropic.com/v1", 120)
    await pool.get_shared_client("https://api.anthropic.com/v1", 300)
    assert pool.shared_client_pool_size() == 2


@pytest.mark.asyncio
async def test_different_origin_creates_separate_client():
    await pool.get_shared_client("https://a.example/v1", 60)
    await pool.get_shared_client("https://b.example/v1", 60)
    assert pool.shared_client_pool_size() == 2


@pytest.mark.asyncio
async def test_context_manager_does_not_close_shared_client():
    async with pool.shared_client("https://api.example.com/v1", 30) as client:
        assert client is not None
    again = await pool.get_shared_client("https://api.example.com/v1", 30)
    assert again is client
    assert again.is_closed is False


@pytest.mark.asyncio
async def test_in_flight_client_survives_idle_eviction(monkeypatch):
    """空闲 TTL 已过期也不能关正在被流使用的客户端。"""
    monkeypatch.setattr(pool, "_IDLE_TTL_SECONDS", 1.0)

    client, key = await pool.acquire("https://slow-stream.example/v1", 30)
    try:
        # 把在用条目的空闲时间显式推过 TTL，不依赖真实时钟分辨率。
        pool._entries[key].last_used -= 1000.0
        other = await pool.get_shared_client("https://other.example/v1", 30)
        assert client.is_closed is False, "在飞客户端被驱逐，会导致流中途断开"
        assert pool.in_use_count() == 1
    finally:
        await pool.release(key)

    assert other.is_closed is False


@pytest.mark.asyncio
async def test_in_flight_client_survives_capacity_pressure(monkeypatch):
    """池满且全部在用：允许临时超限，而不是牺牲在飞请求。"""
    monkeypatch.setattr(pool, "_MAX_CLIENTS", 2)

    held = []
    for index in range(3):
        client, key = await pool.acquire(f"https://busy{index}.example/v1", 30)
        held.append((client, key))

    assert all(not client.is_closed for client, _ in held)
    assert pool.shared_client_pool_size() == 3  # 临时超过上限 2

    for _client, key in held:
        await pool.release(key)


@pytest.mark.asyncio
async def test_released_client_can_be_evicted(monkeypatch):
    monkeypatch.setattr(pool, "_IDLE_TTL_SECONDS", 1.0)
    client, key = await pool.acquire("https://done.example/v1", 30)
    await pool.release(key)
    # 直接回退空闲时间，避免依赖 Windows 下 ~15ms 的单调钟分辨率。
    pool._entries[key].last_used -= 1000.0
    await pool.get_shared_client("https://trigger.example/v1", 30)
    assert client.is_closed is True


@pytest.mark.asyncio
async def test_release_balances_acquire():
    _client, key = await pool.acquire("https://balance.example/v1", 30)
    assert pool.in_use_count() == 1
    await pool.release(key)
    assert pool.in_use_count() == 0
    # 重复 release 不应把计数压成负数。
    await pool.release(key)
    assert pool.in_use_count() == 0


@pytest.mark.asyncio
async def test_nested_contexts_share_one_client_and_close_after_last():
    async with pool.shared_client("https://shared.example/v1", 30) as outer:
        async with pool.shared_client("https://shared.example/v1", 30) as inner:
            assert outer is inner
            assert pool.in_use_count() == 1
        # 内层退出后引用归零，但客户端仍在池中可复用。
        assert outer.is_closed is False
    assert pool.in_use_count() == 0


@pytest.mark.asyncio
async def test_pool_is_bounded_when_idle(monkeypatch):
    monkeypatch.setattr(pool, "_MAX_CLIENTS", 3)
    for index in range(6):
        await pool.get_shared_client(f"https://host{index}.example/v1", 30)
    assert pool.shared_client_pool_size() <= 3


@pytest.mark.asyncio
async def test_aclose_shared_clients_closes_and_empties():
    client = await pool.get_shared_client("https://api.example.com/v1", 30)
    await pool.aclose_shared_clients()
    assert pool.shared_client_pool_size() == 0
    assert client.is_closed is True


# -- 引用计数与上游流关闭（S5）的闭环 --

@pytest.mark.asyncio
async def test_stream_reference_is_released_when_consumer_abandons_generator(monkeypatch):
    """上层丢弃流生成器时必须归还池引用，否则该客户端永久免于驱逐。

    S5 的 aclose() 与 P8 的 release() 互相依赖：只有生成器被关闭，
    ``async with shared_client(...)`` 的 finally 才会执行。
    """
    from app.adapters import anthropic_streaming

    class FakeStream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_lines(self):
            yield "event: message_start"
            yield 'data: {"type":"message_start","message":{"id":"m1","role":"assistant","content":[],"usage":{"input_tokens":1}}}'
            yield ""
            yield "event: content_block_delta"
            yield 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}'
            yield ""
            await asyncio.sleep(3600)  # 之后不再产出，模拟慢上游

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def stream(self, *args, **kwargs):
            return FakeStream()

    monkeypatch.setattr(pool.httpx, "AsyncClient", FakeClient)

    baseline = pool.in_use_count()
    events = anthropic_streaming.iter_anthropic_output_events(
        provider_info={"id": "anth", "api_base": "https://anth.example", "api_key": "k"},
        messages=[{"role": "user", "content": "hi"}],
        body={}, max_tokens=16, temperature=0.2, model="claude-x",
    )
    seen = 0
    async for _ in events:
        seen += 1
        if seen >= 1:
            break

    assert pool.in_use_count() == baseline + 1, "流进行中应持有引用"
    await events.aclose()
    assert pool.in_use_count() == baseline, "丢弃生成器后必须归还引用"
