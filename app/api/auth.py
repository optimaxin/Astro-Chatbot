"""Who is calling /chat — derived on the server, never taken from the body.

Signed-in users
    The browser sends the same access token it uses for every Tredev API
    call (`Authorization: Bearer …`). It is verified by asking the Tredev
    backend (`GET /api/auth/me`), which checks signature, expiry, suspension
    and session revocation in one place. ai-core never holds the JWT secret,
    and a revoked session stops working here exactly when it stops working
    everywhere else. Results are cached in-process for `cache_ttl` seconds,
    keyed by a hash of the token, so verification costs ~0 ms on most turns.

    Why not Firebase ID tokens directly: in this platform Firebase is used
    only at sign-in (Google / phone); the backend then issues its own session
    token, and email/password users never hold a Firebase token at all. The
    backend token is the one identity every signed-in user has.

Anonymous visitors (the public Kundli page)
    Get a server-issued identity `anon:<random>` plus an HMAC-signed
    `session_token` returned in the response; the client echoes it back. A
    browser-invented id is never an identity. Anonymous turns may only use a
    chart computed from birth details they submitted themselves.

`AUTH_MODE`: optional (default) | required (no anonymous chat) | off (legacy:
trust the body's user_id — rollback only).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

from app.config.settings import settings
from app.logging_setup import get_logger

log = get_logger("api.auth")

_cache: dict[str, tuple[str | None, float]] = {}
_cache_lock = threading.Lock()
_ephemeral_secret = secrets.token_bytes(32)


class AuthRequired(Exception):
    """401: no acceptable identity for this request."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class AuthUnavailable(Exception):
    """503: the identity provider could not be asked. Fails closed."""


class Forbidden(Exception):
    """403: authenticated, but not allowed to touch this resource."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True)
class Identity:
    kind: str            # "user" | "anon" | "legacy"
    user_id: str         # for usage attribution and chart lookup
    key: str             # namespace for history, limits and ownership
    new_session_token: str | None = None

    @property
    def authenticated(self) -> bool:
        return self.kind == "user"


# ── signed anonymous tokens ───────────────────────────────────────────
def _secret() -> bytes:
    configured = settings.auth.session_secret
    return configured.encode() if configured else _ephemeral_secret


def _sign(anon_id: str) -> str:
    return hmac.new(_secret(), anon_id.encode(), hashlib.sha256).hexdigest()[:32]


def issue_anon_token() -> tuple[str, str]:
    anon_id = secrets.token_urlsafe(12)
    return anon_id, f"{anon_id}.{_sign(anon_id)}"


def parse_anon_token(token: str | None) -> str | None:
    if not token or "." not in token or len(token) > 200:
        return None
    anon_id, _, signature = token.rpartition(".")
    if anon_id and hmac.compare_digest(signature, _sign(anon_id)):
        return anon_id
    return None


# ── signed-in users ───────────────────────────────────────────────────
def _verify_with_backend(token: str) -> str | None:
    """User id for a valid token, None for an invalid one. Raises
    AuthUnavailable when the backend cannot be asked."""
    import httpx

    url = settings.auth.backend_url.rstrip("/") + settings.auth.verify_path
    try:
        response = httpx.get(url, headers={"Authorization": f"Bearer {token}"},
                             timeout=settings.auth.verify_timeout_seconds)
    except httpx.HTTPError as exc:
        raise AuthUnavailable(type(exc).__name__) from exc
    if response.status_code in (401, 403):
        return None
    if response.status_code != 200:
        raise AuthUnavailable(f"verify returned {response.status_code}")
    data: Any = (response.json() or {}).get("data") or {}
    user_id = data.get("id")
    return str(user_id) if user_id else None


def verify_bearer(token: str) -> str | None:
    digest = hashlib.sha256(token.encode()).hexdigest()
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(digest)
        if hit and hit[1] > now:
            return hit[0]
    user_id = _verify_with_backend(token)
    # Valid tokens are cached for the full TTL; invalid ones briefly, so a
    # client stuck in a retry loop cannot hammer the backend.
    ttl = settings.auth.cache_ttl_seconds if user_id else 10.0
    with _cache_lock:
        if len(_cache) > 10_000:
            _cache.clear()
        _cache[digest] = (user_id, now + ttl)
    return user_id


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def _bearer(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        return token or None
    return None


def resolve_identity(*, authorization: str | None, session_token: str | None,
                     body_user_id: str, has_birth: bool) -> Identity:
    mode = settings.auth.mode
    if mode == "off":
        return Identity("legacy", body_user_id, f"legacy:{body_user_id}")

    token = _bearer(authorization)
    if token:
        user_id = verify_bearer(token)
        if user_id:
            if body_user_id and body_user_id != user_id and not body_user_id.startswith("web-"):
                log.warning("body user_id differs from the authenticated user; ignored")
            return Identity("user", user_id, f"user:{user_id}")
        if mode == "required":
            raise AuthRequired("Your session has expired. Please sign in again.")
        # optional: an expired token degrades to anonymous, like the
        # backend's optionalAuth — never to the body's user_id.

    if mode == "required":
        raise AuthRequired("Please sign in to chat with the astrology assistant.")
    if not has_birth:
        raise AuthRequired("Please sign in, or generate your Kundli first, to chat about your chart.")

    anon_id = parse_anon_token(session_token)
    new_token = None
    if anon_id is None:
        anon_id, new_token = issue_anon_token()
    return Identity("anon", f"anon:{anon_id}", f"anon:{anon_id}", new_token)


def scoped_session(identity: Identity, session_id: str) -> str:
    """Conversation key in Redis: the caller's identity plus their session id.
    A guessed or reused session id can only ever reach the caller's own
    history — session ownership by construction."""
    if identity.kind == "legacy":
        return session_id
    return f"{identity.key}|{session_id}"


__all__ = ["Identity", "AuthRequired", "AuthUnavailable", "Forbidden", "resolve_identity",
           "scoped_session", "verify_bearer", "issue_anon_token", "parse_anon_token", "clear_cache"]
