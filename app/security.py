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
_login_attempts: dict[str, list[float]] = {}
_login_blocked_until: dict[str, float] = {}
_login_attempts_lock = threading.Lock()
_login_last_prune = 0.0
# 同一时间刻度内入定的身份时间戳完全相同，仅按时间排序会退化为 set 的
# 迭代顺序（受哈希扰动影响），导致“刚记录的身份”反而被当成最旧删除。
# 用单调序号做 tie-break，保证保留的总是最近更新的身份。
_login_seq: dict[str, int] = {}
_login_next_seq = 0


def _note_login_touch_locked(identity: str) -> None:
    global _login_next_seq
    _login_next_seq += 1
    _login_seq[identity] = _login_next_seq


def _login_rank_locked(identity: str) -> tuple[float, int]:
    last_seen = max(
        _login_attempts.get(identity, [0.0])[-1],
        _login_blocked_until.get(identity, 0.0),
    )
    return (last_seen, _login_seq.get(identity, 0))


def _prune_login_throttle_locked(now: float, window: int, max_identities: int) -> None:
    """Expire stale identities and cap memory used by attacker-controlled keys."""
    global _login_last_prune
    if now - _login_last_prune >= min(30, window):
        for identity, values in list(_login_attempts.items()):
            recent = [value for value in values if now - value <= window]
            if recent:
                _login_attempts[identity] = recent
            else:
                _login_attempts.pop(identity, None)
                _login_seq.pop(identity, None)
        for identity, blocked_until in list(_login_blocked_until.items()):
            if blocked_until <= now:
                _login_blocked_until.pop(identity, None)
                _login_seq.pop(identity, None)
        _login_last_prune = now

    # An identity moves from attempts to blocked state, so these mappings are
    # disjoint during normal operation and their lengths can be added cheaply.
    overflow = len(_login_attempts) + len(_login_blocked_until) - max_identities
    if overflow <= 0:
        return
    identities = set(_login_attempts) | set(_login_blocked_until)
    oldest = sorted(identities, key=_login_rank_locked)
    for identity in oldest[:overflow]:
        _login_attempts.pop(identity, None)
        _login_blocked_until.pop(identity, None)
        _login_seq.pop(identity, None)


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


def login_retry_after(identity: str) -> int:
    """Return lockout seconds remaining for an admin login identity."""
    now = _time.monotonic()
    window = max(10, int(get_default("login_attempt_window_seconds", 300)))
    max_identities = max(100, int(get_default("login_attempt_max_identities", 10000)))
    with _login_attempts_lock:
        _prune_login_throttle_locked(now, window, max_identities)
        blocked_until = _login_blocked_until.get(identity, 0.0)
        if blocked_until <= now:
            _login_blocked_until.pop(identity, None)
            return 0
        return max(1, int(blocked_until - now))


def record_login_failure(identity: str) -> int:
    now = _time.monotonic()
    window = max(10, int(get_default("login_attempt_window_seconds", 300)))
    limit = max(1, int(get_default("login_attempt_limit", 10)))
    lockout = max(10, int(get_default("login_lockout_seconds", 900)))
    max_identities = max(100, int(get_default("login_attempt_max_identities", 10000)))
    with _login_attempts_lock:
        _prune_login_throttle_locked(now, window, max_identities)
        recent = [value for value in _login_attempts.get(identity, []) if now - value <= window]
        recent.append(now)
        _login_attempts[identity] = recent
        _note_login_touch_locked(identity)
        if len(recent) >= limit:
            _login_attempts.pop(identity, None)
            _login_blocked_until[identity] = now + lockout
            _note_login_touch_locked(identity)
            _prune_login_throttle_locked(now, window, max_identities)
            return lockout
        _prune_login_throttle_locked(now, window, max_identities)
    return 0


def clear_login_failures(identity: str) -> None:
    with _login_attempts_lock:
        _login_attempts.pop(identity, None)
        _login_blocked_until.pop(identity, None)
        _login_seq.pop(identity, None)


def _cleanup_expired_sessions() -> None:
    """Periodically remove expired sessions to prevent memory leaks."""
    while not _stop_cleanup.is_set():
        try:
            now = datetime.now(UTC)
            with _sessions_lock:
                expired = [t for t, s in _sessions.items() if s["expires_at"] < now]
                for t in expired:
                    _sessions.pop(t, None)
        except Exception as exc:  # 清理是尽力而为，但失败必须可见
            import logging
            logging.getLogger("llmgw.app").warning("[sessions.cleanup] failed: %s", exc)
        _stop_cleanup.wait(300)  # Every 5 minutes


_stop_cleanup = threading.Event()


_cleanup_thread = threading.Thread(target=_cleanup_expired_sessions, daemon=True)
_cleanup_thread.start()


def stop_session_cleanup(timeout: float = 1.0) -> bool:
    """Stop the background session-cleanup thread. Returns True when it exited."""
    _stop_cleanup.set()
    if not _cleanup_thread.is_alive():
        return True
    _cleanup_thread.join(timeout=timeout)
    return not _cleanup_thread.is_alive()
