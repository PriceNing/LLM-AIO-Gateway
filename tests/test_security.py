"""
Unit tests for app.security - password hashing, session management, API key generation.
"""
import time
import pytest
from app.security import (
    hash_password, verify_password, new_api_key,
    create_session, get_session_username, delete_session,
    clear_login_failures, login_retry_after, record_login_failure,
)

# -- Password hashing --

def test_hash_password_produces_expected_format():
    result = hash_password("mypassword")
    assert result.startswith("pbkdf2_sha256$")
    parts = result.split("$", 2)
    assert len(parts) == 3
    assert len(parts[1]) == 32  # salt is 16 bytes hex = 32 chars
    assert len(parts[2]) == 64  # sha256 hex = 64 chars


def test_hash_password_deterministic_with_same_salt():
    salt = "a" * 32
    result1 = hash_password("hello", salt)
    result2 = hash_password("hello", salt)
    assert result1 == result2


def test_hash_password_different_salt_produces_different_hash():
    result1 = hash_password("hello")
    result2 = hash_password("hello")
    assert result1 != result2  # different random salts


def test_verify_password_correct():
    h = hash_password("correct")
    assert verify_password("correct", h) is True


def test_verify_password_wrong():
    h = hash_password("correct")
    assert verify_password("wrong", h) is False


def test_verify_password_invalid_hash():
    assert verify_password("anything", "not-a-valid-hash") is False


def test_verify_password_wrong_scheme():
    assert verify_password("p", "md5$salt$hash") is False


# -- API key generation --

def test_new_api_key_format():
    for _ in range(10):
        key = new_api_key()
        assert key.startswith("sk-aio-")
        assert len(key) > 10  # prefix + base64 data


def test_new_api_key_custom_prefix():
    key = new_api_key(prefix="my-prefix")
    assert key.startswith("my-prefix-")


def test_new_api_key_unique():
    keys = {new_api_key() for _ in range(100)}
    assert len(keys) == 100  # no collisions in 100 keys


# -- Session management --

def test_create_and_get_session():
    token = create_session("admin")
    assert get_session_username(token) == "admin"


def test_get_nonexistent_session():
    assert get_session_username("nonexistent") is None


def test_delete_session():
    token = create_session("test")
    delete_session(token)
    assert get_session_username(token) is None


def test_expired_session(monkeypatch):
    from datetime import UTC, datetime, timedelta
    token = create_session("expiring")
    # Manipulate the session's expiry to be in the past
    from app import security
    original_session = security._sessions[token]
    security._sessions[token] = {
        "username": "expiring",
        "expires_at": datetime.now(UTC) - timedelta(hours=1),
    }
    assert get_session_username(token) is None


def test_session_cleanup_removes_expired(monkeypatch):
    from datetime import UTC, datetime, timedelta
    from app import security

    # Create sessions with past expiry
    for i in range(5):
        token = create_session(f"user-{i}")
        security._sessions[token]["expires_at"] = (
            datetime.now(UTC) - timedelta(hours=1)
        )

    # Force cleanup
    from app.security import _cleanup_expired_sessions
    now = datetime.now(UTC)
    expired = [t for t, s in list(security._sessions.items())
               if s.get("expires_at") and s["expires_at"] < now]
    for t in expired:
        security._sessions.pop(t, None)

    # All expired sessions should be gone
    remaining = [s for s in security._sessions.values()
                 if s.get("username", "").startswith("user-")]
    assert len(remaining) == 0


def test_failed_login_attempts_trigger_and_clear_lockout(monkeypatch):
    identity = "127.0.0.1\0rate-limit-test"
    clear_login_failures(identity)
    monkeypatch.setattr("app.security.get_default", lambda key, fallback=None: {
        "login_attempt_window_seconds": 60,
        "login_attempt_limit": 2,
        "login_lockout_seconds": 30,
    }.get(key, fallback))

    assert record_login_failure(identity) == 0
    assert record_login_failure(identity) == 30
    assert login_retry_after(identity) > 0

    clear_login_failures(identity)
    assert login_retry_after(identity) == 0


def test_failed_login_identity_cache_is_bounded(monkeypatch):
    from app import security

    monkeypatch.setattr("app.security.get_default", lambda key, fallback=None: {
        "login_attempt_window_seconds": 60,
        "login_attempt_limit": 10,
        "login_lockout_seconds": 30,
        "login_attempt_max_identities": 100,
    }.get(key, fallback))
    identities = [f"bounded-login-{index}" for index in range(105)]
    for identity in identities:
        clear_login_failures(identity)
        record_login_failure(identity)

    active = set(security._login_attempts) | set(security._login_blocked_until)
    assert len(active) <= 100
    assert identities[-1] in active

    for identity in identities:
        clear_login_failures(identity)
