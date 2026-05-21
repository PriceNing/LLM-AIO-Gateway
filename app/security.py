import hashlib
import hmac
import secrets
import threading
import time as _time
from datetime import UTC, datetime, timedelta
from typing import Optional

from app.config import get_default

_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()
SESSION_TTL_HOURS = get_default("session_ttl_hours", 12)


def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        scheme, salt, expected = password_hash.split("$", 2)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    actual = hash_password(password, salt).split("$", 2)[2]
    return hmac.compare_digest(actual, expected)


def new_api_key(prefix: str = "sk-aio") -> str:
    return f"{prefix}-{secrets.token_urlsafe(32)}"


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    with _sessions_lock:
        _sessions[token] = {
            "username": username,
            "expires_at": datetime.now(UTC) + timedelta(hours=SESSION_TTL_HOURS),
        }
    return token


def get_session_username(token: str) -> Optional[str]:
    with _sessions_lock:
        session = _sessions.get(token)
        if not session:
            return None
        if session["expires_at"] < datetime.now(UTC):
            _sessions.pop(token, None)
            return None
        return session["username"]


def delete_session(token: str) -> None:
    with _sessions_lock:
        _sessions.pop(token, None)


_stop_cleanup = threading.Event()


def _cleanup_expired_sessions() -> None:
    """Periodically remove expired sessions to prevent memory leaks."""
    while not _stop_cleanup.is_set():
        try:
            now = datetime.now(UTC)
            with _sessions_lock:
                expired = [t for t, s in _sessions.items() if s["expires_at"] < now]
                for t in expired:
                    _sessions.pop(t, None)
        except Exception:
            pass
        _stop_cleanup.wait(300)  # Every 5 minutes


_cleanup_thread = threading.Thread(target=_cleanup_expired_sessions, daemon=True)
_cleanup_thread.start()
