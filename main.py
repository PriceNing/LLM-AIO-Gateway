import sys
import io
# Windows cmd.exe uses GBK by default, which can't encode emoji (e.g. OK).
# Reconfigure stdout/stderr to UTF-8 so diagnostic prints don't crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import load_config, get_config
from app.services.logger import init_logging, set_request_id, generate_request_id
from app.router import admin, auth, proxy

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "app" / "web" / "static"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign a unique request_id to every HTTP request for log tracing."""
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or generate_request_id()
        set_request_id(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.database import init_db
    cfg = load_config()
    db_path = cfg.config.get("database", "data.db")
    init_db(db_path)

    init_logging(cfg.config.get("logging"))
    yield


app = FastAPI(title="LLM AIO Gateway", version="1.0.0", lifespan=lifespan)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
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
        reload=True,
    )
