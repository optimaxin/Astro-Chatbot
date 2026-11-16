"""Tester access: optional shared token, anonymous client ids, rate and spend limits.

With PLAYGROUND_ACCESS_TOKEN unset the tester endpoints are public. Each
browser then identifies itself with a random anonymous client id
(X-Playground-Client), which scopes history so testers never see each
other's experiments. It is not a verified identity. Provider keys never
leave the server.
"""

from __future__ import annotations

import hmac
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import Header, HTTPException, Request

from app import storage
from app.settings import settings

_CLIENT_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.access_token:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), settings.access_token):
        raise HTTPException(401, "Invalid or missing access token.")


def client_id(x_playground_client: str | None = Header(default=None)) -> str:
    value = (x_playground_client or "").strip()
    if not _CLIENT_RE.fullmatch(value):
        raise HTTPException(400, "Missing or malformed X-Playground-Client header.")
    return value


def client_ip(request: Request) -> str:
    # Uvicorn runs with --proxy-headers on Render, so this is the caller's IP.
    return request.client.host if request.client else "unknown"


_hits: dict[str, deque[float]] = defaultdict(deque)
_hits_lock = threading.Lock()


def _check_window(key: str, limit: int, window_s: float) -> bool:
    now = time.monotonic()
    with _hits_lock:
        window = _hits[key]
        while window and now - window[0] > window_s:
            window.popleft()
        if len(window) >= limit:
            return False
        window.append(now)
        return True


def rate_limit(request: Request) -> None:
    limit = settings.rate_limit_per_minute
    if limit > 0 and not _check_window("gen:" + client_ip(request), limit, 60):
        raise HTTPException(429, f"Playground rate limit reached ({limit} generations/minute).")


def start_of_today() -> float:
    tz = ZoneInfo(settings.report_timezone)
    now = datetime.now(tz)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def spend_guard() -> None:
    """Stops all generation for the day once the global request or USD cap is hit."""
    if settings.daily_request_limit <= 0 and settings.daily_cost_limit_usd <= 0:
        return
    used = storage.usage_since(start_of_today())
    if 0 < settings.daily_request_limit <= used["requests"]:
        raise HTTPException(429, "The playground's daily request limit has been reached. Try again tomorrow.")
    if 0 < settings.daily_cost_limit_usd <= used["cost_usd"]:
        raise HTTPException(429, "The playground's daily spending limit has been reached. Try again tomorrow.")


def reset_rate_limits() -> None:
    with _hits_lock:
        _hits.clear()
