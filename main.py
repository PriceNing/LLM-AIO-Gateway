import sys
import io
import asyncio
import contextlib
import time
# Windows cmd.exe uses GBK by default, which can't encode emoji (e.g. OK).
# Reconfigure stdout/stderr to UTF-8 so diagnostic prints don't crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from app import __version__
from app.config import load_config, get_config
from app.core.body_limit import RequestBodyLimitMiddleware
from app.services.logger import get_logger, init_logging, set_request_id, generate_request_id
from app.router import admin, auth, proxy

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "app" / "web" / "static"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign a unique request_id to every HTTP request for log tracing."""
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or generate_request_id()
        set_request_id(rid)
        start = time.perf_counter()
        access_log = get_logger("access")
        error_log = get_logger("error")
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            error_log.exception(
                "[http.exception] method=%s path=%s elapsed_ms=%d error=%s",
                request.method,
                request.url.path,
                elapsed_ms,
                exc,
            )
            raise
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = rid
        access_log.info(
            "[http] method=%s path=%s status=%d elapsed_ms=%d",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.database import init_db, run_storage_maintenance, close_thread_connection
    cfg = load_config()
    db_path = cfg.config.get("database", "data.db")
    init_db(db_path)

    init_logging(cfg.config.get("logging"))

    logger = get_logger("app")
    stop_maintenance = asyncio.Event()

    async def _maintenance_loop():
        """周期裁剪请求日志并清理过期历史记录（P7/Q4）。"""
        interval = 60
        try:
            interval = int(cfg.config.get("defaults", {}).get("storage_maintenance_interval_seconds", 60))
        except (AttributeError, TypeError, ValueError):
            interval = 60
        interval = max(5, interval)
        while not stop_maintenance.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop_maintenance.wait(), timeout=interval)
            if stop_maintenance.is_set():
                break
            try:
                result = await asyncio.to_thread(run_storage_maintenance)
                logger.debug("[storage.maintenance] %s", result)
            except Exception as exc:
                logger.warning("[storage.maintenance] failed: %s", exc)

    maintenance_task = asyncio.create_task(_maintenance_loop())
    try:
        yield
    finally:
        stop_maintenance.set()
        maintenance_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await maintenance_task
        # 后台线程与主线程缓存的 SQLite 连接需显式释放（Q7）。
        from app.security import stop_session_cleanup
        stop_session_cleanup()
        from app.services.http_pool import aclose_shared_clients
        await aclose_shared_clients()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(close_thread_connection)
        close_thread_connection()


app = FastAPI(title="LLM AIO Gateway", version=__version__, lifespan=lifespan)

app.add_middleware(RequestBodyLimitMiddleware)
app.add_middleware(RequestIdMiddleware)
# 默认仍为 "*" 以免破坏既有部署；生产环境应在 config.json 的
# cors_allow_origins 里填写具体来源（S9）。
_cors_origins = get_config().config.get("cors_allow_origins") or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in _cors_origins],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(proxy.router, prefix="/v1", tags=["proxy"])

# Also mount proxy routes at root level for SDKs (OpenCode, etc.) that resolve
# {baseURL}/chat/completions via JS URL semantics, which drops the base path.
app.include_router(proxy.router, prefix="", tags=["proxy-root"])

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    cfg = load_config(force_reload=True)
    uvicorn.run(
        "main:app",
        host=cfg.config.get("host", "0.0.0.0"),
        port=int(cfg.config.get("port", 8000)),
        reload=bool(cfg.config.get("reload", False)),
    )
