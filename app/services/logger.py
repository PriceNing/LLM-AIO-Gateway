"""
Structured logging system with date-based directories, automatic cleanup,
and per-channel log level override.

Config (in config.json):
  "logging": {
    "enabled": true,         // master switch
    "level": "INFO",         // DEBUG | INFO | WARNING | ERROR (default for all channels)
    "log_dir": "logs",       // root directory for log files
    "retention_days": 30,    // auto-delete directories older than this
    "console": false,        // also print to stderr
    "channels": {            // optional per-channel level override
      "tool_calls": "DEBUG",
      "request": "DEBUG"
    }
  }

Log files per day (under logs/YYYY-MM-DD/):
  access.log    - HTTP request summaries (with elapsed_ms)
  error.log     - errors and exceptions
  app.log       - general application messages + liteLLM internal logs
  tool_calls.log - raw tool-call data for debugging
  request.log   - raw request metadata for debugging (by default at DEBUG level)

Each line is JSON with keys: ts, request_id, level, logger, msg
"""

import json
import logging
import os
import sys
import time
import uuid
import threading
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Per-request tracing ID, set by middleware in main.py
_request_id: ContextVar[str] = ContextVar("request_id", default="")

_LOG_MANAGER: Optional["LogManager"] = None

LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def _default_logging_config() -> dict:
    return {
        "enabled": True,
        "level": "INFO",
        "log_dir": "logs",
        "retention_days": 30,
        "console": False,
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "request_id": _request_id.get(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exc"] = str(record.exc_info[1])
        return json.dumps(entry, ensure_ascii=False, default=str)


class SafeFileHandler(logging.FileHandler):
    """FileHandler that recreates missing parent directories before opening."""

    def _open(self):
        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        return super()._open()


class LogManager:
    # Map logger name suffix -> log file name
    _CHANNELS = {
        "access": "access.log",
        "error": "error.log",
        "app": "app.log",
        "tool_calls": "tool_calls.log",
        "request": "request.log",
    }

    @classmethod
    def channels(cls) -> dict[str, str]:
        return dict(cls._CHANNELS)

    def __init__(self):
        self._lock = threading.Lock()
        self._handlers: dict[str, logging.FileHandler] = {}
        self._date_str = ""
        self._log_dir = "logs"
        self._retention_days = 30
        self._enabled = True
        self._level = logging.INFO
        self._console = False
        self._channel_levels: dict[str, int] = {}
        self._litellm_captured = False

    def configure(self, config: Optional[dict] = None) -> None:
        with self._lock:
            if config is None:
                config = {}
            self._cfg = {**_default_logging_config(), **config}
            self._log_dir = self._cfg["log_dir"]
            self._retention_days = int(self._cfg["retention_days"])
            self._enabled = bool(self._cfg["enabled"])
            self._level = LEVEL_MAP.get(str(self._cfg["level"]).upper(), logging.INFO)
            self._console = bool(self._cfg.get("console", False))

            # Per-channel level overrides (e.g. {"tool_calls": "DEBUG"})
            self._channel_levels = {}
            for ch_name, ch_level in self._cfg.get("channels", {}).items():
                if ch_name in self._CHANNELS:
                    self._channel_levels[ch_name] = LEVEL_MAP.get(
                        str(ch_level).upper(), logging.DEBUG)

            if self._enabled:
                self._ensure_handlers()
                self._cleanup_old_logs()
                self._capture_litellm_logs()
                self._emit_startup_diagnostics()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def log_dir(self) -> str:
        return self._log_dir

    def get_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(f"llmgw.{name}")
        logger.propagate = False

        # Resolve effective level: per-channel override > global > disabled
        channel_key = name.split(".")[-1] if "." in name else name
        if not self._enabled:
            logger.setLevel(logging.CRITICAL + 10)
        elif channel_key in self._channel_levels:
            logger.setLevel(self._channel_levels[channel_key])
        else:
            logger.setLevel(self._level)

        if not self._enabled:
            return logger

        with self._lock:
            self._ensure_handlers()
            handler = self._handlers.get(channel_key, self._handlers.get("app"))
        if handler and handler not in logger.handlers:
            # Remove previous FileHandlers because configure() may have created new handlers,
            # while old handlers may still be attached to loggers, causing duplicate output.
            for h in list(logger.handlers):
                if isinstance(h, logging.FileHandler):
                    logger.removeHandler(h)
                    h.close()
            logger.addHandler(handler)

        if self._console:
            if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stderr
                       for h in logger.handlers):
                ch = logging.StreamHandler(sys.stderr)
                ch.setFormatter(JsonFormatter())
                logger.addHandler(ch)

        return logger

    def _today_dir(self) -> Path:
        return Path(self._log_dir) / datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _ensure_handlers(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today == self._date_str:
            return

        for h in self._handlers.values():
            h.close()
        self._handlers.clear()

        day_dir = self._today_dir()
        day_dir.mkdir(parents=True, exist_ok=True)

        for channel, filename in self._CHANNELS.items():
            path = day_dir / filename
            handler = SafeFileHandler(str(path), encoding="utf-8")
            handler.setFormatter(JsonFormatter())
            handler.setLevel(logging.DEBUG)  # let logger level control filtering
            self._handlers[channel] = handler

        self._date_str = today

    def _cleanup_old_logs(self) -> None:
        root = Path(self._log_dir)
        if not root.exists():
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            try:
                dir_date = datetime.strptime(entry.name, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if dir_date < cutoff:
                    import shutil
                    shutil.rmtree(entry, ignore_errors=True)
            except ValueError:
                pass

    def _capture_litellm_logs(self) -> None:
        """Route liteLLM's own log output into our app.log channel."""
        if self._litellm_captured:
            return
        try:
            handler = self._handlers.get("app")
            if not handler:
                return
            for logger_name in ("LiteLLM", "litellm", "LiteLLM Router", "LiteLLM Proxy"):
                litellm_logger = logging.getLogger(logger_name)
                litellm_logger.setLevel(self._level)
                if handler not in litellm_logger.handlers:
                    litellm_logger.addHandler(handler)
        except Exception:
            pass  # liteLLM logger capture is best-effort
        self._litellm_captured = True

    def _emit_startup_diagnostics(self) -> None:
        try:
            app_logger = logging.getLogger("llmgw.app")
            app_logger.debug(
                "[logging] configured enabled=%s level=%s log_dir=%s console=%s channels=%s",
                self._enabled,
                logging.getLevelName(self._level),
                self._log_dir,
                self._console,
                {k: logging.getLevelName(v) for k, v in self._channel_levels.items()},
            )
            for channel, handler in self._handlers.items():
                app_logger.debug("[logging] handler channel=%s file=%s", channel, getattr(handler, "baseFilename", ""))
        except Exception:
            pass  # liteLLM handler removal is best-effort


def init_logging(config: Optional[dict] = None) -> LogManager:
    """Initialize the logging system. Called once at app startup."""
    global _LOG_MANAGER
    mgr = get_log_manager()  # Reuse the instance created at import time to avoid stale handlers
    mgr.configure(config)
    return mgr


def get_log_manager() -> LogManager:
    global _LOG_MANAGER
    if _LOG_MANAGER is None:
        _LOG_MANAGER = LogManager()
    return _LOG_MANAGER


def get_logger(name: str) -> logging.Logger:
    return get_log_manager().get_logger(name)


def available_log_channels() -> dict[str, str]:
    return LogManager.channels()


def _configured_log_dir() -> str:
    try:
        from app.config import get_config
        logging_config = get_config().config.get("logging") or {}
        return str(logging_config.get("log_dir") or get_log_manager().log_dir)
    except Exception:
        return get_log_manager().log_dir


def list_log_dates() -> list[str]:
    root = Path(_configured_log_dir())
    if not root.exists():
        return []
    dates = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            datetime.strptime(entry.name, "%Y-%m-%d")
        except ValueError:
            continue
        dates.append(entry.name)
    return sorted(dates, reverse=True)


def _log_file_path(date: str, channel: str) -> Path:
    channels = available_log_channels()
    if channel not in channels:
        raise ValueError("invalid log channel")
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("invalid log date") from exc

    root = Path(_configured_log_dir()).resolve()
    path = (root / date / channels[channel]).resolve()
    if root != path and root not in path.parents:
        raise ValueError("invalid log path")
    return path


def read_log_entries(
    date: str,
    channel: str,
    *,
    limit: int = 200,
    offset: int = 0,
    level: str = "",
    q: str = "",
) -> dict:
    path = _log_file_path(date, channel)
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    level = str(level or "").upper().strip()
    q = str(q or "").lower().strip()

    if not path.exists():
        return {"items": [], "total": 0, "limit": limit, "offset": offset, "path": str(path)}

    items = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            raw = line.rstrip("\r\n")
            if not raw:
                continue
            try:
                entry = json.loads(raw)
                if not isinstance(entry, dict):
                    entry = {"msg": raw}
            except json.JSONDecodeError:
                entry = {"msg": raw}
            entry.setdefault("ts", "")
            entry.setdefault("request_id", "")
            entry.setdefault("level", "")
            entry.setdefault("logger", "")
            entry.setdefault("msg", raw)
            entry["line"] = line_no
            entry["raw"] = raw
            if level and str(entry.get("level") or "").upper() != level:
                continue
            if q and q not in raw.lower():
                continue
            items.append(entry)

    items.reverse()
    total = len(items)
    return {
        "items": items[offset:offset + limit],
        "total": total,
        "limit": limit,
        "offset": offset,
        "path": str(path),
    }


def set_request_id(rid: str = "") -> None:
    _request_id.set(rid)


def get_request_id() -> str:
    return _request_id.get() or ""


def generate_request_id() -> str:
    return uuid.uuid4().hex[:12]
