"""Admin authentication.

The password is never stored: ADMIN_PASSWORD_HASH holds an Argon2id hash
(make one with `python -m scripts.hash_password`). A login issues a random
session token, kept in an HttpOnly, SameSite=Strict cookie scoped to
/api/admin; only its SHA-256 is stored. State-changing admin requests also
need an X-CSRF-Token header derived from the session.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Cookie, Header, HTTPException, Request, Response

from app import storage
from app.auth import _check_window, client_ip
from app.settings import settings

COOKIE = "pg_admin"
COOKIE_PATH = "/api/admin"
LOGIN_ATTEMPTS = 5
LOGIN_WINDOW_S = 15 * 60

_hasher = PasswordHasher()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def csrf_for(token: str) -> str:
    return hmac.new(settings.session_secret.encode(), token.encode(), hashlib.sha256).hexdigest()


def verify_password(password: str) -> bool:
    if not settings.admin_password_hash:
        raise HTTPException(503, "Admin login is not configured (ADMIN_PASSWORD_HASH is unset).")
    try:
        return _hasher.verify(settings.admin_password_hash, password)
    except VerifyMismatchError:
        return False
    except (InvalidHashError, VerificationError) as exc:
        raise HTTPException(503, "ADMIN_PASSWORD_HASH is not a valid Argon2 hash.") from exc


def login(request: Request, response: Response, password: str) -> dict[str, str]:
    ip = client_ip(request)
    if not _check_window("login:" + ip, LOGIN_ATTEMPTS, LOGIN_WINDOW_S):
        storage.audit("admin.login_blocked", ip)
        raise HTTPException(429, "Too many login attempts. Wait 15 minutes and try again.")
    if not verify_password(password):
        storage.audit("admin.login_failed", ip)
        raise HTTPException(401, "Incorrect password.")
    token = secrets.token_urlsafe(32)
    max_age = settings.admin_session_hours * 3600
    storage.create_admin_session(_hash_token(token), time.time() + max_age)
    response.set_cookie(COOKIE, token, max_age=max_age, path=COOKIE_PATH, httponly=True,
                        secure=settings.cookie_secure, samesite="strict")
    storage.audit("admin.login", ip)
    return {"csrf_token": csrf_for(token)}


def logout(request: Request, response: Response, token: str | None) -> None:
    if token:
        storage.delete_admin_session(_hash_token(token))
    response.delete_cookie(COOKIE, path=COOKIE_PATH, httponly=True, secure=settings.cookie_secure,
                           samesite="strict")
    storage.audit("admin.logout", client_ip(request))


def require_admin(pg_admin: str | None = Cookie(default=None)) -> str:
    if not pg_admin or not storage.admin_session_valid(_hash_token(pg_admin)):
        raise HTTPException(401, "Admin sign-in required.")
    return pg_admin


def require_csrf(pg_admin: str | None = Cookie(default=None),
                 x_csrf_token: str | None = Header(default=None)) -> None:
    token = require_admin(pg_admin)
    if not x_csrf_token or not hmac.compare_digest(x_csrf_token, csrf_for(token)):
        raise HTTPException(403, "Missing or invalid CSRF token.")
