import ctypes
import queue
import threading

import anyio

from app.services.logger import get_logger


_app_log = get_logger("app")
_error_log = get_logger("error")
_STREAM_SENTINEL = object()


async def iter_stream_async(stream_func):
    """Iterate a sync generator in a background thread without blocking the event loop."""
    chunk_queue = queue.Queue()
    error = None
    done = threading.Event()
    cancel = threading.Event()
    stream_gen = None
    bg_thread = None

    def _run():
        nonlocal error, stream_gen
        try:
            stream_gen = stream_func()
            chunk_idx = 0
            for chunk in stream_gen:
                if cancel.is_set():
                    break
                chunk_idx += 1
                chunk_queue.put(chunk)
            _app_log.debug("[iter_stream_async] generator finished, total_chunks=%d", chunk_idx)
        except GeneratorExit:
            pass
        except Exception as e:
            import traceback as tb
            try:
                _error_log.error("[iter_stream_async] type=%s msg=%s", type(e).__name__, str(e)[:200])
                _error_log.error("[iter_stream_async] %s", tb.format_exc())
            except Exception:
                _app_log.warning("streaming: error in sync generator %s", e)
            error = e
        finally:
            if stream_gen is not None:
                try:
                    stream_gen.close()
                except Exception:
                    _app_log.warning("streaming: error putting sentinel %s", e)
            chunk_queue.put(_STREAM_SENTINEL)
            done.set()

    bg_thread = threading.Thread(target=_run, daemon=True)
    bg_thread.start()

    try:
        while True:
            try:
                chunk = chunk_queue.get(timeout=0.01)
                if chunk is _STREAM_SENTINEL:
                    break
                yield chunk
            except queue.Empty:
                if done.is_set():
                    break
                await anyio.sleep(0)

        if error:
            raise error
    finally:
        cancel.set()
        if bg_thread is not None and bg_thread.is_alive():
            try:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(bg_thread.ident),
                    ctypes.py_object(GeneratorExit),
                )
            except Exception:
                _app_log.warning("streaming: error in async generator %s", e)
