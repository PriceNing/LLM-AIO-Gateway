import asyncio
from fastapi import APIRouter, Header, HTTPException
from typing import Optional
from app.database import get_admins, get_admin, add_admin
from app.security import create_session, delete_session, get_session_username, hash_password, verify_password

router = APIRouter()


def get_bearer_token(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing session token")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing session token")
    return token


async def require_admin_session(authorization: Optional[str] = Header(None)) -> str:
    token = get_bearer_token(authorization)
    username = get_session_username(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    admin = get_admin(username)
    if not admin or not admin.get("enabled", True):
        raise HTTPException(status_code=401, detail="Admin disabled")
    return username


@router.get("/status")
async def auth_status():
    return {"has_admin": bool(get_admins())}


@router.post("/setup")
async def setup_admin(payload: dict):
    if get_admins():
        raise HTTPException(status_code=409, detail="Admin already initialized")
    username = payload.get("username", "").strip()
    password = payload.get("password", "")
    display_name = payload.get("display_name", username)
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password are required")
    password_hash = await asyncio.to_thread(hash_password, password)
    add_admin(username, password_hash, display_name)
    token = create_session(username)
    return {"token": token, "username": username, "display_name": display_name}


@router.post("/login")
async def login(payload: dict):
    username = payload.get("username", "").strip()
    password = payload.get("password", "")
    admin = get_admin(username)
    if not admin or not admin.get("enabled", True):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    valid = await asyncio.to_thread(verify_password, password, admin.get("password_hash", ""))
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_session(username)
    return {"token": token, "username": username, "display_name": admin.get("display_name", username)}


@router.get("/me")
async def me(authorization: Optional[str] = Header(None)):
    username = await require_admin_session(authorization)
    admin = get_admin(username)
    return {"username": username, "display_name": admin.get("display_name", username)}


@router.post("/logout")
async def logout(authorization: Optional[str] = Header(None)):
    token = get_bearer_token(authorization)
    delete_session(token)
    return {"status": "ok"}
