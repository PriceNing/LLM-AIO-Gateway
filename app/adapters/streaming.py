import asyncio
import queue
import threading
from inspect import iscoroutinefunction as _iscoroutinefunction

from app.core.text import error_detail_for_log
from app.services.logger import get_logger, get_request_id, set_request_id


_app_log = get_logger("app")
_error_log = get_logger("error")
_STREAM_SENTINEL = object()

# 有界缓冲：上游产出快于客户端消费时形成背压，避免内存无界增长（P3）。
_STREAM_QUEUE_MAXSIZE = 256
# 取消检查与"暂无数据"时的等待粒度；有数据时立即返回，不阻塞事件循环（P2）。
_STREAM_POLL_INTERVAL = 0.02
# 单次唤醒之间最多搬运的 chunk 数，避免长时间占住事件循环。
_STREAM_DRAIN_BATCH = 64


def _close_upstream_transport(stream_gen: object) -> None:
    """在工作线程内释放被包装的上游 HTTP 传输（S5 不变量在同步流上的落点）。

    “上游流必须被关闭”这条不变量以前只做到 Python 生成器层：
    ``stream_gen.close()`` 对 litellm 1.83 的 ``CustomStreamWrapper`` 是空操作（它只有
    异步 ``aclose()``，没有同步 ``close()``），而上游 HTTP 响应就挂在它的
    ``completion_stream`` 上。结果是：被放弃的上游连接只能等 CPython 引用计数回收，
    而引用又被异常 ``__traceback__`` 拽着（traceback → 上游迭代器帧 → httpx 响应 →
    连接），只要请求本身还活着就不释放——同一请求内的多次尝试会把这段时间叠加成
    分钟级（2026-10-03 事故：三次 120s 尝试共 ~4 分钟，单槽上游被同一次超时楔死）。

    在此期间上游仍在往一个我们不再读取、也没发 FIN 的 socket 里写；对单槽推理引擎，
    一次中途超时足以把整台服务楔死。

    只在工作线程自己的栈里关闭：消费者（事件循环侧）此刻可能仍卡在 ``__next__`` 的
    socket 读上，跨线程关闭 httpx 同步响应不安全。``openai.Stream.close()`` /
    ``httpx.Response.close()`` 都是幂等的同步关闭，正常读完的流重复关闭无副作用。
    """
    inner = getattr(stream_gen, "completion_stream", None)
    if inner is None or inner is stream_gen:
        return
    close = getattr(inner, "close", None)
    if not callable(close) or _iscoroutinefunction(close):
        # 只有 aclose()/async def close() 的内层流属于异步适配层（openai.AsyncStream 的
        # close 是协程函数，在这里调只会得一个未被 await 的协程），由上层 aclose 链负责。
        return
    try:
        close()
    except Exception as close_err:
        _app_log.warning("[iter_stream_async] error closing upstream transport: %s", close_err)
    else:
        _app_log.debug("[iter_stream_async] upstream transport closed (%s)", type(inner).__name__)


async def iter_stream_async(
    stream_func,
    *,
    maxsize: int = _STREAM_QUEUE_MAXSIZE,
    poll_interval: float = _STREAM_POLL_INTERVAL,
):
    """Iterate a sync generator from a worker thread without blocking the loop.

    投递成本：生产者只做线程内有界 ``queue.put``（实测约 1.6 µs），跨线程唤醒
    用 ``armed`` 标志合并，因此每 chunk 约 2 µs，而不是每个 chunk 一次
    ``run_coroutine_threadsafe(...).result()`` 往返（实测约 152 µs）。

    退出契约
    --------
    * 取消只设置 ``cancel`` 标志，由生产者线程在 chunk 边界自行退出并在**其自身
      线程内** ``close()`` 上游生成器。不再向工作线程注入异步异常
      （``PyThreadState_SetAsyncExc`` 在 C 扩展执行期间不生效且可能损坏解释器
      状态，见「当前问题.md」S3）。
    * 同一线程内还会显式释放被包装的上游 HTTP 传输层（见
      ``_close_upstream_transport``）：只关生成器不够，litellm 的流式包装器根本没有
      同步 ``close()``。
    * 生产者退出时**无条件**置位 ``producer_done``；消费者在哨兵丢失时靠它收尾，
      不会永久等待。
    * 上游卡死时本函数无法提前收回控制权，由 ``litellm.request_timeout`` 兜底；
      这是放弃线程注入后明确的取舍，不宣称可即时中断。
    """
    loop = asyncio.get_running_loop()
    handoff: queue.Queue = queue.Queue(maxsize=max(1, int(maxsize)))
    cancel = threading.Event()
    producer_done = threading.Event()
    armed = threading.Event()          # 已排了一次唤醒，避免每 chunk 都唤醒事件循环
    data_ready = asyncio.Event()       # 事件循环侧的唤醒信号
    error: BaseException | None = None
    request_id = get_request_id()

    def _notify() -> None:
        if armed.is_set():
            return
        armed.set()
        try:
            loop.call_soon_threadsafe(data_ready.set)
        except RuntimeError:
            # 事件循环已关闭：消费者不存在了，producer_done 会负责收尾。
            pass

    def _offer(item) -> bool:
        """Blocking hand-off with cancellation checks; False when consumer is gone."""
        while not cancel.is_set():
            try:
                handoff.put(item, timeout=poll_interval)
                return True
            except queue.Full:
                continue
        return False

    def _run():
        nonlocal error
        stream_gen = None
        chunk_idx = 0
        # ContextVars 不会自动进入手工创建的线程，保留 request_id 以便
        # 上游流错误能与本次 HTTP 请求的日志关联。
        set_request_id(request_id)
        try:
            stream_gen = stream_func()
            for chunk in stream_gen:
                chunk_idx += 1
                if cancel.is_set():
                    break
                if not _offer(chunk):
                    break
                _notify()
            _app_log.debug("[iter_stream_async] generator finished, total_chunks=%d", chunk_idx)
        except GeneratorExit:
            pass
        except BaseException as exc:
            error = exc
            _error_log.error("[iter_stream_async] type=%s msg=%s", type(exc).__name__, error_detail_for_log(exc))
        finally:
            if stream_gen is not None:
                close = getattr(stream_gen, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as close_err:
                        _app_log.warning("[iter_stream_async] error closing generator: %s", close_err)
                # 生成器层关闭不等于连接释放：litellm 包装器没有同步 close()，
                # 必须再显式关掉它包装的上游 HTTP 响应（见 _close_upstream_transport）。
                _close_upstream_transport(stream_gen)
            if not cancel.is_set():
                _offer(_STREAM_SENTINEL)
                _notify()
            # 无条件置位：哨兵没送到时，消费者靠它退出而不是永久等待。
            producer_done.set()
            _notify()

    worker = threading.Thread(target=_run, daemon=True, name="llmgw-stream")
    worker.start()

    try:
        ended = False
        while not ended:
            drained = 0
            while drained < _STREAM_DRAIN_BATCH:
                try:
                    item = handoff.get_nowait()
                except queue.Empty:
                    break
                drained += 1
                if item is _STREAM_SENTINEL:
                    ended = True
                    break
                yield item
            if ended:
                break
            if drained >= _STREAM_DRAIN_BATCH:
                # 还有数据就继续搬，但先让事件循环喘一口气。
                await asyncio.sleep(0)
                continue
            # 队列已空：先解除唤醒锁存，再决定是否收尾，避免丢唤醒。
            armed.clear()
            if not handoff.empty():
                continue
            if producer_done.is_set():
                ended = True
                break
            try:
                await asyncio.wait_for(data_ready.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                pass
            data_ready.clear()
        # 上游异常必须向上传播，不能被当成正常结束（S5 同类语义）。
        if error:
            raise error
    finally:
        cancel.set()
        if worker.is_alive():
            _app_log.debug("[iter_stream_async] client disconnected, cancel requested")
