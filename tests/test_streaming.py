"""Tests for app/core/streaming.py ¡ª record_streaming_events and stream_internal_output."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.output import InternalOutputEvent
from app.core.streaming import record_streaming_events, stream_internal_output
from app.adapters.streaming import iter_stream_async
from app.core.streaming import iter_output_idle_timeout
from app.services.logger import get_request_id, set_request_id


async def _collect_events(events):
    """Collect all events from an async generator."""
    result = []
    async for e in events:
        result.append(e)
    return result


async def _make_events(*kinds_and_data):
    """Helper: yield InternalOutputEvent from (kind, field, value) triples."""
    for kind, field, value in kinds_and_data:
        yield InternalOutputEvent(kind=kind, **{field: value})


@pytest.mark.asyncio
async def test_iter_stream_async_accepts_litellm_style_iterable_without_close(monkeypatch):
    """A stream wrapper may be iterable without exposing generator.close()."""
    import app.adapters.streaming as streaming_adapter

    warnings = []
    monkeypatch.setattr(streaming_adapter._app_log, "warning", lambda *args: warnings.append(args))

    class StreamWrapper:
        def __iter__(self):
            return iter(["one", "two"])

    received = []
    async for chunk in iter_stream_async(StreamWrapper):
        received.append(chunk)

    assert received == ["one", "two"]
    assert warnings == []


@pytest.mark.asyncio
async def test_iter_stream_async_propagates_request_id_to_worker_thread():
    set_request_id("stream-request-123")
    seen = []

    def stream():
        seen.append(get_request_id())
        yield "ok"

    received = [chunk async for chunk in iter_stream_async(stream)]
    assert received == ["ok"]
    assert seen == ["stream-request-123"]


# --- record_streaming_events ---

@pytest.mark.asyncio
async def test_record_text_delta_sets_has_text():
    """Text delta events should be tracked."""
    gen = _make_events(
        ("text_delta", "text", "hello"),
        ("message_done", "finish_reason", "stop"),
    )
    reasoning_tool_ids = []
    tool_only = MagicMock()
    tool_only.reset = MagicMock()

    events = await _collect_events(
        record_streaming_events(gen, conv_key="test", tool_only_turns=tool_only)
    )
    # tool_only_turns.reset should be called because has_text=True
    tool_only.reset.assert_called_once_with("test")


@pytest.mark.asyncio
async def test_record_reasoning_delta_accumulates():
    """Reasoning deltas should accumulate and be stored on message_done."""
    stored = {}

    def fake_remember(conv_key, reasoning, tool_ids):
        stored["key"] = conv_key
        stored["reasoning"] = reasoning
        stored["tool_ids"] = tool_ids

    gen = _make_events(
        ("reasoning_delta", "reasoning", "think "),
        ("reasoning_delta", "reasoning", "more"),
        ("message_done", "finish_reason", "stop"),
    )
    events = await _collect_events(
        record_streaming_events(
            gen,
            conv_key="conv1",
            remember_reasoning_content=fake_remember,
        )
    )
    assert stored["key"] == "conv1"
    assert stored["reasoning"] == "think more"


@pytest.mark.asyncio
async def test_record_tool_only_turn_increments():
    """Tool-only turn (tools but no text) should increment counter."""
    increment_mock = MagicMock(return_value=3)
    tool_only = MagicMock()
    tool_only.increment = increment_mock
    tool_only.reset = MagicMock()

    gen = _make_events(
        ("tool_call_start", "tool_call_id", "tc_1"),
        ("message_done", "finish_reason", "stop"),
    )
    events = await _collect_events(
        record_streaming_events(gen, conv_key="conv2", tool_only_turns=tool_only)
    )
    tool_only.increment.assert_called_once_with("conv2")
    tool_only.reset.assert_not_called()


@pytest.mark.asyncio
async def test_record_mixed_text_and_tools_resets():
    """Both text and tools should reset tool-only counter."""
    tool_only = MagicMock()
    tool_only.reset = MagicMock()

    gen = _make_events(
        ("text_delta", "text", "hi"),
        ("tool_call_start", "tool_call_id", "tc_1"),
        ("message_done", "finish_reason", "stop"),
    )
    events = await _collect_events(
        record_streaming_events(gen, conv_key="conv3", tool_only_turns=tool_only)
    )
    tool_only.reset.assert_called_once_with("conv3")


@pytest.mark.asyncio
async def test_record_finalize_on_stream_end_without_message_done():
    """If stream ends without message_done, finalize should still run."""
    stored = {}

    def fake_remember(conv_key, reasoning, tool_ids):
        stored["reasoning"] = reasoning

    gen = _make_events(
        ("reasoning_delta", "reasoning", "partial"),
    )
    events = await _collect_events(
        record_streaming_events(
            gen,
            conv_key="conv4",
            remember_reasoning_content=fake_remember,
        )
    )
    assert stored["reasoning"] == "partial"


@pytest.mark.asyncio
async def test_record_no_finalize_called_twice():
    """Finalize should only run once even if message_done fires."""
    call_count = [0]

    def counting_remember(conv_key, reasoning, tool_ids):
        call_count[0] += 1

    gen = _make_events(
        ("reasoning_delta", "reasoning", "thinking"),
        ("message_done", "finish_reason", "stop"),
    )
    events = await _collect_events(
        record_streaming_events(
            gen,
            conv_key="conv5",
            remember_reasoning_content=counting_remember,
        )
    )
    # message_done triggers finalize; no second call at stream end
    assert call_count[0] == 1


@pytest.mark.asyncio
async def test_record_no_remember_when_no_reasoning():
    """If no reasoning events fire, remember should not be called."""
    call_count = [0]

    def counting_remember(conv_key, reasoning, tool_ids):
        call_count[0] += 1

    gen = _make_events(
        ("text_delta", "text", "hello"),
        ("message_done", "finish_reason", "stop"),
    )
    events = await _collect_events(
        record_streaming_events(
            gen,
            conv_key="conv6",
            remember_reasoning_content=counting_remember,
        )
    )
    assert call_count[0] == 0


@pytest.mark.asyncio
async def test_record_no_tool_only_turns_when_none():
    """If tool_only_turns is None, no error should occur."""
    gen = _make_events(
        ("text_delta", "text", "hi"),
        ("message_done", "finish_reason", "stop"),
    )
    events = await _collect_events(
        record_streaming_events(gen, conv_key="conv7", tool_only_turns=None)
    )
    assert len(events) == 2


# --- stream_internal_output (focused on metadata/usage routing) ---

@pytest.mark.asyncio
async def test_stream_internal_output_extracts_usage():
    """Usage events should be captured for request logging."""
    logged = {}

    def fake_log(user, key, model, provider, endpoint, success, tokens, requested, **kwargs):
        logged["tokens"] = tokens
        logged["success"] = success

    async def _events():
        yield InternalOutputEvent(kind="text_delta", text="hi")
        yield InternalOutputEvent(kind="usage", usage={"total_tokens": 42})
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")
    events = _events()

    result = []
    async for line in stream_internal_output(
        events=events,
        endpoint="chat_completions",
        model="test-model",
        username="testuser",
        api_key_value="sk-test",
        provider_id="p1",
        requested_model="test-model",
        log_request=fake_log,
        conv_key="conv8",
    ):
        result.append(line)

    assert logged["tokens"] == 42
    assert logged["success"] is True


@pytest.mark.asyncio
async def test_stream_internal_output_updates_model_from_metadata():
    """Metadata events should update the final model used for logging."""
    logged = {}

    def fake_log(user, key, model, provider, endpoint, success, tokens, requested, **kwargs):
        logged["model"] = model
        logged["provider"] = provider

    async def _events():
        yield InternalOutputEvent(kind="metadata", metadata={"model": "real-model", "provider_id": "real-provider"})
        yield InternalOutputEvent(kind="text_delta", text="hi")
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")
    events = _events()

    result = []
    async for line in stream_internal_output(
        events=events,
        endpoint="chat_completions",
        model="original-model",
        username="u",
        api_key_value="k",
        provider_id="p1",
        requested_model="orig",
        log_request=fake_log,
        conv_key="conv9",
    ):
        result.append(line)

    assert logged["model"] == "real-model"
    assert logged["provider"] == "real-provider"


@pytest.mark.asyncio
async def test_stream_internal_output_logs_partial_failure_after_visible_output():
    """Failures after SSE output starts should keep the attempted target and mark partial output."""
    logged = {}

    def fake_log(user, key, model, provider, endpoint, success, tokens, requested, **kwargs):
        logged["model"] = model
        logged["provider"] = provider
        logged["success"] = success
        logged["tokens"] = tokens
        logged["details"] = kwargs.get("details") or {}

    async def _events():
        yield InternalOutputEvent(kind="metadata", metadata={"model": "real-model", "provider_id": "real-provider"})
        yield InternalOutputEvent(kind="text_delta", text="hi")
        exc = RuntimeError("upstream closed")
        exc.request_details = {
            "fallback_status": "skipped",
            "fallback_reason": "client_output_started",
            "error_trigger": "http_5xx",
            "attempted_model": "real-model",
            "attempted_provider": "real-provider",
            "partial_output": True,
        }
        raise exc

    result = []
    async for line in stream_internal_output(
        events=_events(),
        endpoint="responses",
        model="original-model",
        username="u",
        api_key_value="k",
        provider_id="p1",
        requested_model="orig",
        log_request=fake_log,
        conv_key="conv10",
    ):
        result.append(line)

    assert logged["success"] is False
    assert logged["model"] == "real-model"
    assert logged["provider"] == "real-provider"
    assert logged["details"]["status"] == "partial"
    assert logged["details"]["partial_output"] is True
    assert logged["details"]["fallback_status"] == "skipped"


@pytest.mark.asyncio
async def test_stream_internal_output_marks_fallback_success_as_degraded(monkeypatch):
    """Successful completion after fallback should log status=degraded and increment degraded stats."""
    logged = {}
    counters = {"ok": 0, "degraded": 0, "failed": 0, "cancelled": 0}

    def fake_log(user, key, model, provider, endpoint, success, tokens, requested, **kwargs):
        logged["success"] = success
        logged["model"] = model
        logged["details"] = kwargs.get("details") or {}

    def fake_increment(success, *, degraded=False, rejected=False, cancelled=False):
        if success and degraded:
            counters["degraded"] += 1
        elif success:
            counters["ok"] += 1
        elif cancelled:
            counters["cancelled"] += 1
        else:
            counters["failed"] += 1

    monkeypatch.setattr("app.core.streaming.increment_global_stats", fake_increment)
    monkeypatch.setattr("app.core.streaming.increment_user_usage", lambda *a, **k: None)

    async def _events():
        yield InternalOutputEvent(
            kind="metadata",
            metadata={
                "model": "qianye/gpt-5.5",
                "provider_id": "qianye",
                "fallback_status": "used",
                "attempt_index": 1,
                "fallback_attempts": [
                    {"index": 0, "stage": "primary", "status": "failed", "provider": "PixelAPI"},
                    {"index": 1, "stage": "fallback", "status": "success", "provider": "qianye"},
                ],
            },
        )
        yield InternalOutputEvent(kind="text_delta", text="recovered")
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    async for _ in stream_internal_output(
        events=_events(),
        endpoint="chat_completions",
        model="PixelAPI/gpt-5.5",
        username="u",
        api_key_value="k",
        provider_id="PixelAPI",
        requested_model="qianye/gpt-5.5",
        log_request=fake_log,
        conv_key="conv-degraded",
        base_details={"routing_matched": True, "routed_model": "PixelAPI/gpt-5.5"},
    ):
        pass

    assert logged["success"] is True
    assert logged["details"]["status"] == "degraded"
    assert logged["details"]["fallback_status"] == "used"
    assert logged["details"]["routed_model"] == "PixelAPI/gpt-5.5"
    assert logged["model"] == "qianye/gpt-5.5"
    assert counters["degraded"] == 1
    assert counters["failed"] == 0


@pytest.mark.asyncio
async def test_stream_internal_output_preserves_native_downgrade_metadata():
    logged = {}

    def fake_log(*args, **kwargs):
        logged["details"] = kwargs.get("details") or {}

    async def _events():
        yield InternalOutputEvent(
            kind="metadata",
            metadata={
                "upstream_endpoint": "chat_completions",
                "model": "PixelAPI/gpt-5.6-luna",
                "provider_id": "PixelAPI",
            },
        )
        yield InternalOutputEvent(kind="message_done", finish_reason="stop")

    async for _ in stream_internal_output(
        events=_events(), endpoint="responses", model="PixelAPI/gpt-5.6-luna",
        username="u", api_key_value="k", provider_id="PixelAPI",
        requested_model="PixelAPI/gpt-5.6-luna", log_request=fake_log,
        base_details={
            "responses_mode": "compatibility_downgrade",
            "native_attempted": True,
            "native_failure_endpoint": "responses",
            "native_failure_status": 502,
            "native_failure_reason": "http_5xx",
        },
    ):
        pass

    assert logged["details"]["upstream_endpoint"] == "chat_completions"
    assert logged["details"]["responses_mode"] == "compatibility_downgrade"
    assert logged["details"]["native_attempted"] is True
    assert logged["details"]["native_failure_status"] == 502


@pytest.mark.asyncio
async def test_stream_internal_output_marks_client_disconnect_as_cancelled(monkeypatch):
    logged = {}
    counters = {"cancelled": 0, "failed": 0}

    def fake_log(user, key, model, provider, endpoint, success, tokens, requested, **kwargs):
        logged["success"] = success
        logged["details"] = kwargs.get("details") or {}

    def fake_increment(success, *, degraded=False, rejected=False, cancelled=False):
        if cancelled:
            counters["cancelled"] += 1
        elif not success:
            counters["failed"] += 1

    monkeypatch.setattr("app.core.streaming.increment_global_stats", fake_increment)
    monkeypatch.setattr("app.core.streaming.increment_user_usage", lambda *a, **k: None)

    class ClientDisconnect(Exception):
        pass

    async def _events():
        yield InternalOutputEvent(kind="text_delta", text="partial")
        raise ClientDisconnect("client disconnected")

    result = []
    async for line in stream_internal_output(
        events=_events(),
        endpoint="chat_completions",
        model="m",
        username="u",
        api_key_value="k",
        provider_id="p",
        requested_model="m",
        log_request=fake_log,
        conv_key="conv-cancel",
    ):
        result.append(line)

    assert logged["success"] is False
    assert logged["details"]["status"] == "cancelled"
    assert logged["details"]["client_disconnected"] is True
    assert counters["cancelled"] == 1
    assert counters["failed"] == 0


@pytest.mark.asyncio
async def test_stream_internal_output_tool_turn_disconnect_is_ok(monkeypatch):
    logged = {}
    counters = {"ok": 0, "cancelled": 0}

    def fake_log(user, key, model, provider, endpoint, success, tokens, requested, **kwargs):
        logged["success"] = success
        logged["details"] = kwargs.get("details") or {}

    def fake_increment(success, *, degraded=False, rejected=False, cancelled=False):
        if cancelled:
            counters["cancelled"] += 1
        elif success:
            counters["ok"] += 1

    monkeypatch.setattr("app.core.streaming.increment_global_stats", fake_increment)
    monkeypatch.setattr("app.core.streaming.increment_user_usage", lambda *a, **k: None)

    class ClientDisconnect(Exception):
        pass

    async def _events():
        yield InternalOutputEvent(kind="tool_call_start", tool_call_id="call_1", name="exec_command")
        yield InternalOutputEvent(kind="tool_call_done", tool_call_id="call_1", name="exec_command", arguments="{}")
        raise ClientDisconnect("client disconnected")

    async for _line in stream_internal_output(
        events=_events(),
        endpoint="chat_completions",
        model="m",
        username="u",
        api_key_value="k",
        provider_id="p",
        requested_model="m",
        log_request=fake_log,
        conv_key="conv-tool-close",
    ):
        pass

    assert logged["success"] is True
    assert logged["details"]["status"] == "ok"
    assert logged["details"]["stream_closed_after_output"] is True
    assert counters["ok"] == 1
    assert counters["cancelled"] == 0


@pytest.mark.asyncio
async def test_stream_internal_output_records_tool_arguments_for_request_log():
    recorded = {}

    def fake_log(user, key, model, provider, endpoint, success, tokens, requested, **kwargs):
        pass

    def fake_record(**kwargs):
        recorded.update(kwargs)

    async def _events():
        yield InternalOutputEvent(kind="tool_call_start", tool_call_id="call_1", name="lookup")
        yield InternalOutputEvent(
            kind="tool_call_arguments_delta",
            tool_call_id="call_1",
            name="lookup",
            arguments_delta='{"q"',
        )
        yield InternalOutputEvent(
            kind="tool_call_arguments_delta",
            tool_call_id="call_1",
            name="lookup",
            arguments_delta=':"x"}',
        )
        yield InternalOutputEvent(kind="tool_call_done", tool_call_id="call_1", name="lookup")
        yield InternalOutputEvent(kind="message_done", finish_reason="tool_calls")

    async for _ in stream_internal_output(
        events=_events(),
        endpoint="chat_completions",
        model="test-model",
        username="u",
        api_key_value="k",
        provider_id="p",
        requested_model="test-model",
        log_request=fake_log,
        record_request_log=fake_record,
        conv_key="conv-tools",
    ):
        pass

    assert recorded["streamed_tool_calls"] == [
        {"id": "call_1", "name": "lookup", "arguments": '{"q":"x"}'}
    ]


# -- 流桥取消与背压（「当前问题.md」S3/P2/P3）--

import asyncio as _asyncio


def test_iter_stream_async_does_not_use_thread_async_exceptions():
    """取消必须协作式完成，不得向工作线程注入异步异常。

    在 AST 层面检查，避免因文档措辞提及 API 名而误判。
    """
    import ast
    import inspect
    import pathlib

    path = pathlib.Path(inspect.getsourcefile(__import__("app.adapters.streaming", fromlist=["*"])))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned_calls = {"PyThreadState_SetAsyncExc", "pythonapi"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name == "ctypes" for a in node.names), "不得重新引入 ctypes"
        if isinstance(node, ast.Attribute):
            assert node.attr not in banned_calls, f"检测到禁止的线程注入调用：{node.attr}"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in banned_calls


@pytest.mark.asyncio
async def test_iter_stream_async_closes_upstream_on_early_consumer_exit():
    closed = []

    def stream():
        try:
            for i in range(100):
                yield i
        finally:
            closed.append(True)

    got = []
    async for chunk in iter_stream_async(stream, poll_interval=0.005):
        got.append(chunk)
        if len(got) == 2:
            break

    for _ in range(200):
        if closed:
            break
        await _asyncio.sleep(0.01)

    assert got == [0, 1]
    assert closed == [True]


@pytest.mark.asyncio
async def test_iter_stream_async_applies_backpressure_to_producer():
    produced = []

    def stream():
        for i in range(10):
            produced.append(i)
            yield i

    agen = iter_stream_async(stream, maxsize=1, poll_interval=0.005)
    assert await agen.__anext__() == 0
    await _asyncio.sleep(0.1)
    # 有界缓冲生效：未被消费时生产者不得一次性跑完全部 chunk。
    assert len(produced) <= 3
    await agen.aclose()


@pytest.mark.asyncio
async def test_iter_stream_async_propagates_upstream_error():
    def stream():
        yield "ok"
        raise RuntimeError("upstream broke")

    received = []
    with pytest.raises(RuntimeError, match="upstream broke"):
        async for chunk in iter_stream_async(stream):
            received.append(chunk)
    assert received == ["ok"]


@pytest.mark.asyncio
async def test_producer_is_backpressured_when_consumer_never_reads():
    """消费者不读就关闭时，生产者必须被背压+取消拦住，不能跑完整流。

    新投递架构下挂死已从结构上消除：消费者不依赖哨兵送达，而是靠
    producer_done + 队列排空收尾；生产者则在 chunk 边界检查 cancel。
    """
    produced = []

    def stream():
        for index in range(10000):
            produced.append(index)
            yield index

    agen = iter_stream_async(stream, maxsize=2, poll_interval=0.005)
    await agen.aclose()
    await _asyncio.sleep(0.3)

    assert len(produced) < 200, f"生产者未被背压/取消拦住，已产出 {len(produced)} 个 chunk"


@pytest.mark.asyncio
async def test_consumer_exits_without_sentinel_via_producer_done():
    """上游报错且哨兵未能送达时，消费者仍须报错退出而非永久等待。"""
    import app.adapters.streaming as module

    def stream():
        yield "a"
        raise RuntimeError("boom without sentinel")

    received = []
    with pytest.raises(RuntimeError, match="boom without sentinel"):
        async for chunk in iter_stream_async(stream, poll_interval=0.005):
            received.append(chunk)
    assert received == ["a"]


# -- 上游流确定性关闭（「当前问题.md」S5）--

@pytest.mark.asyncio
async def test_first_output_timeout_closes_upstream_generator():
    """超时进入回退时，必须关闭被丢弃的上游流。"""
    from app.core.policy import RouteTarget
    from app.router.proxy import _iter_events_with_first_output_timeout

    closed = []

    async def upstream():
        try:
            yield InternalOutputEvent(kind="usage", usage={})  # 非可见事件，不解除超时
            await _asyncio.sleep(3600)
        finally:
            closed.append(True)

    with pytest.raises(TimeoutError, match="fallback attempt timeout"):
        async for _ in _iter_events_with_first_output_timeout(
            upstream(),
            timeout_s=1,
            target=RouteTarget(model="m", provider_id="p"),
            provider_id="p",
        ):
            pass

    assert closed == [True]


@pytest.mark.asyncio
async def test_stream_internal_output_closes_upstream_on_early_exit():
    """客户端提前断开（上层 aclose 响应流）时，上游流必须被关闭。"""
    closed = []

    async def events():
        try:
            for i in range(50):
                yield InternalOutputEvent(kind="text_delta", text=f"t{i}")
        finally:
            closed.append(True)

    response = stream_internal_output(
        events=events(),
        endpoint="chat_completions",
        model="m",
        username="u",
        api_key_value="k",
        provider_id="p",
        requested_model="m",
        log_request=lambda *args, **kwargs: None,
    )
    seen = 0
    async for _ in response:
        seen += 1
        if seen >= 2:
            break
    await response.aclose()
    await _asyncio.sleep(0)

    assert seen == 2
    assert closed == [True]


@pytest.mark.asyncio
async def test_concurrent_streams_do_not_leak_worker_threads():
    """协作式取消必须能并发收尾：无线程滞留、无死锁。"""
    import threading
    import time as _time

    def make_stream():
        def stream():
            for i in range(500):
                yield i
        return stream

    baseline = threading.active_count()

    async def consume_half():
        seen = 0
        async for _ in iter_stream_async(make_stream(), poll_interval=0.002):
            seen += 1
            if seen >= 3:
                break
        return seen

    results = await _asyncio.wait_for(
        _asyncio.gather(*(consume_half() for _ in range(20))),
        timeout=30,
    )
    assert results == [3] * 20

    deadline = _time.monotonic() + 10
    while _time.monotonic() < deadline and threading.active_count() > baseline:
        await _asyncio.sleep(0.05)
    assert threading.active_count() <= baseline, (
        f"流工作线程未回收：baseline={baseline}, now={threading.active_count()}"
    )


# -- 上游 HTTP 传输层必须随流一起释放（2026-10-03 单槽上游楔死事故）--
#
# 事故根因：网关只关掉 Python 生成器层，而 litellm 1.83 的流式包装器
# （CustomStreamWrapper）没有同步 close()，上游 httpx 响应挂在它的
# completion_stream 上，只能等引用计数回收；异常 traceback 在请求存活期间
# 一直拽着它，被放弃的上游连接因此继续占着单槽推理引擎的执行槽。
# 下面几条断言的都是"显式关闭"，不得依赖 gc.collect()。


class _TransportStream:
    """litellm CustomStreamWrapper 的形状：可迭代、只有异步 aclose、内层流挂 completion_stream。"""

    def __init__(self, events: list, *, fail_at: int | None = None, tick: float = 0.005):
        import time as _time

        self.events = events
        self.completion_stream = _InnerSyncStream(events)
        self._fail_at = fail_at
        self._tick = tick
        self._time = _time

    def __iter__(self):
        for index in range(100):
            if self._fail_at is not None and index == self._fail_at:
                raise RuntimeError("upstream stream broke mid-flight")
            self._time.sleep(self._tick)  # 让生产者停在流中间，而不是抢在消费者之前跑完
            yield index

    async def aclose(self):
        # 工作线程不得调用异步 aclose：消费者此刻可能仍卡在 socket 读上，
        # 跨线程关闭 httpx 同步响应不安全。
        self.events.append("aclose")


class _InnerSyncStream:
    """openai.Stream / httpx.Response 的形状：同步 close()，幂等。"""

    def __init__(self, events: list):
        self.events = events

    def close(self) -> None:
        self.events.append("close")


async def _wait_until(predicate, *, attempts: int = 400, delay: float = 0.005) -> bool:
    for _ in range(attempts):
        if predicate():
            return True
        await _asyncio.sleep(delay)
    return predicate()


@pytest.mark.asyncio
async def test_iter_stream_async_closes_upstream_transport_on_early_consumer_exit():
    """客户端提前断开（消费者 break）必须在工作线程内关掉上游 HTTP 响应。"""
    events: list = []

    agen = iter_stream_async(lambda: _TransportStream(events), maxsize=1, poll_interval=0.005)
    got = []
    async for chunk in agen:
        got.append(chunk)
        if len(got) == 2:
            break
    # 网关的真实收尾：渲染链提前丢弃后由 finally 里的 aclose_async_iterator 关闭。
    await agen.aclose()

    assert await _wait_until(lambda: "close" in events), f"上游传输未被关闭: events={events}"
    assert got == [0, 1]
    assert "aclose" not in events


@pytest.mark.asyncio
async def test_iter_stream_async_closes_upstream_transport_on_midstream_error():
    """中途超时/异常放弃上游（事故路径）同样必须关闭上游 HTTP 响应。"""
    events: list = []

    agen = iter_stream_async(
        lambda: _TransportStream(events, fail_at=40), maxsize=1, poll_interval=0.005
    )
    got = []
    error = None
    try:
        async for chunk in agen:
            got.append(chunk)
    except RuntimeError as exc:
        error = str(exc)

    assert error == "upstream stream broke mid-flight"
    assert await _wait_until(lambda: "close" in events), f"上游传输未被关闭: events={events}"
    assert len(got) == 40
    assert "aclose" not in events


@pytest.mark.asyncio
async def test_iter_stream_async_closes_upstream_transport_on_normal_completion():
    """正常读完也关闭一次（openai.Stream.close 幂等），不得重复或遗漏。"""
    events: list = []

    received = [chunk async for chunk in iter_stream_async(lambda: _TransportStream(events, tick=0.0), poll_interval=0.005)]

    assert received == list(range(100))
    assert events == ["close"]


@pytest.mark.asyncio
async def test_iter_stream_async_reports_transport_close_failure_without_breaking_stream():
    """关闭失败不能把正常流变成错误，但必须留下可取证的 WARNING。"""
    import app.adapters.streaming as streaming_adapter

    warnings: list = []

    class BrokenClose:
        def close(self):
            raise OSError("transport already gone")

    class Wrapper:
        def __init__(self):
            self.completion_stream = BrokenClose()

        def __iter__(self):
            yield from range(3)

    original = streaming_adapter._app_log.warning
    streaming_adapter._app_log.warning = lambda *args: warnings.append(args)
    try:
        received = [chunk async for chunk in iter_stream_async(Wrapper, poll_interval=0.005)]
    finally:
        streaming_adapter._app_log.warning = original

    assert received == [0, 1, 2]
    assert any("upstream transport" in str(arg) for args in warnings for arg in args), warnings


@pytest.mark.asyncio
async def test_iter_stream_async_tolerates_wrapper_without_transport_attribute():
    """既有契约：包装器既没有 close() 也没有 completion_stream 时不得产生告警。"""
    import app.adapters.streaming as streaming_adapter

    warnings: list = []
    original = streaming_adapter._app_log.warning
    streaming_adapter._app_log.warning = lambda *args: warnings.append(args)
    try:
        received = []
        async for chunk in iter_stream_async(lambda: _WrapperNoCloseNoTransport(), poll_interval=0.005):
            received.append(chunk)
    finally:
        streaming_adapter._app_log.warning = original

    assert received == ["one", "two"]
    assert warnings == []


class _WrapperNoCloseNoTransport:
    def __iter__(self):
        return iter(["one", "two"])


@pytest.mark.asyncio
async def test_iter_stream_async_skips_async_only_inner_stream_without_warning():
    """内层流只有 async def close()（openai.AsyncStream 形状）时必须跳过。

    在工作线程里调协程函数只会拿到一个未被 await 的协程：既没关掉连接，
    又留下 RuntimeWarning，看起来像修好了其实没修。
    """
    import app.adapters.streaming as streaming_adapter

    calls: list = []
    warnings: list = []
    original = streaming_adapter._app_log.warning
    streaming_adapter._app_log.warning = lambda *args: warnings.append(args)

    class AsyncOnlyInner:
        async def close(self):
            calls.append("close")

    class Wrapper:
        def __init__(self):
            self.completion_stream = AsyncOnlyInner()

        def __iter__(self):
            yield from range(2)

    try:
        received = [chunk async for chunk in iter_stream_async(Wrapper, poll_interval=0.005)]
    finally:
        streaming_adapter._app_log.warning = original

    assert received == [0, 1]
    assert calls == []
    assert warnings == []
    assert received == [0, 1]
    assert calls == []
    assert warnings == []


# -- 流式块间空闲超时（stream_idle_timeout_seconds）--
#
# 语义：只在已向客户端产出首个可见输出之后计时。provider.request_timeout 透传给
# httpx 后是分段读超时，对"冷 prefill 静默"和"解码间隙"是同一个旋钮；本层负责
# 后者，因此可以用远小于前者的值快速发现死流，而不会砍掉长 prefill。


def _idle_factory(seconds: float) -> TimeoutError:
    exc = TimeoutError(f"upstream stream idle for {int(seconds)}s after output started")
    return exc


def _is_output(item) -> bool:
    return isinstance(item, str) and item.startswith("out:")


async def _collect_idle(events, *, idle_seconds):
    return [
        item
        async for item in iter_output_idle_timeout(
            events, idle_seconds=idle_seconds, is_output=_is_output, error_factory=_idle_factory
        )
    ]


@pytest.mark.asyncio
async def test_output_idle_timeout_disabled_passes_everything_through():
    async def source():
        yield "out:a"
        await _asyncio.sleep(0.05)
        yield "out:b"

    for idle in (0, None):
        assert await _collect_idle(source(), idle_seconds=idle) == ["out:a", "out:b"]


@pytest.mark.asyncio
async def test_output_idle_timeout_fires_after_first_visible_output():
    closed = []

    async def source():
        try:
            yield "out:a"
            await _asyncio.sleep(3600)  # 产出开始后上游再也不出声
        finally:
            closed.append(True)

    with pytest.raises(TimeoutError):
        await _collect_idle(source(), idle_seconds=0.05)
    assert closed == [True]  # 被砍也必须级联关闭上游流


@pytest.mark.asyncio
async def test_output_idle_timeout_never_cuts_silence_before_first_output():
    """prefill 静默归 provider.request_timeout 管，本层不得插手。"""

    async def source():
        await _asyncio.sleep(0.2)
        yield "out:a"

    assert await _collect_idle(source(), idle_seconds=0.05) == ["out:a"]


@pytest.mark.asyncio
async def test_output_idle_timeout_resets_on_invisible_upstream_items():
    """不可见的 usage/metadata 同样证明连接活着：必须重新计时。"""

    async def source():
        yield "out:a"
        for _ in range(20):
            await _asyncio.sleep(0.02)  # 总时长远超 idle 上限
            yield "meta:usage"
        yield "out:b"

    assert await _collect_idle(source(), idle_seconds=0.1) == ["out:a"] + ["meta:usage"] * 20 + ["out:b"]


@pytest.mark.asyncio
async def test_output_idle_timeout_passes_upstream_errors_through():
    async def source():
        yield "out:a"
        raise RuntimeError("upstream broke")

    with pytest.raises(RuntimeError, match="upstream broke"):
        await _collect_idle(source(), idle_seconds=0.05)
