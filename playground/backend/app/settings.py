"""Environment-driven settings. The only module that reads os.environ."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")
# Optional second env file (e.g. another service's) so local dev can reuse
# existing provider keys without copying them. Never overrides values above.
_shared = os.getenv("PLAYGROUND_SHARED_ENV_FILE", "").strip()
if _shared:
    load_dotenv((BASE_DIR / _shared).resolve(), override=False)


def _str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name) or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name) or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = _str(name).lower()
    return default if not raw else raw in {"1", "true", "yes", "on"}


def _production() -> bool:
    return _str("NODE_ENV").lower() == "production" or _str("PLAYGROUND_ENV").lower() == "production"


@dataclass(frozen=True)
class Settings:
    production: bool = field(default_factory=_production)
    # Empty means the tester endpoints are public (no login), as on Render.
    access_token: str = field(default_factory=lambda: _str("PLAYGROUND_ACCESS_TOKEN"))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in _str("PLAYGROUND_CORS_ORIGINS", "http://localhost:5180").split(",") if o.strip()
    ))
    rate_limit_per_minute: int = field(default_factory=lambda: _int("PLAYGROUND_RATE_LIMIT_PER_MINUTE", 20))
    daily_request_limit: int = field(default_factory=lambda: _int("PLAYGROUND_DAILY_REQUEST_LIMIT", 1000))
    daily_cost_limit_usd: float = field(default_factory=lambda: _float("PLAYGROUND_DAILY_COST_LIMIT_USD", 5.0))
    db_path: Path = field(default_factory=lambda: (BASE_DIR / _str("PLAYGROUND_DB_PATH", "data/playground.db")).resolve())
    models_file: Path = field(default_factory=lambda: (BASE_DIR / _str("PLAYGROUND_MODELS_FILE", "config/models.json")).resolve())
    static_dir: Path = field(default_factory=lambda: (BASE_DIR / _str("PLAYGROUND_STATIC_DIR", "../frontend/dist")).resolve())
    timeout_seconds: float = field(default_factory=lambda: _float("PLAYGROUND_LLM_TIMEOUT_SECONDS", 90.0))
    max_prompt_chars: int = field(default_factory=lambda: _int("PLAYGROUND_MAX_PROMPT_CHARS", 20000))
    max_body_bytes: int = field(default_factory=lambda: _int("PLAYGROUND_MAX_BODY_BYTES", 2_000_000))
    report_timezone: str = field(default_factory=lambda: _str("PLAYGROUND_REPORT_TIMEZONE", "Asia/Kolkata"))
    log_level: str = field(default_factory=lambda: _str("PLAYGROUND_LOG_LEVEL", "INFO").upper())

    admin_password_hash: str = field(default_factory=lambda: _str("ADMIN_PASSWORD_HASH"))
    # Random per process when unset: admin sessions then survive a restart, but
    # the CSRF token is re-issued (the dashboard re-reads it from /me).
    session_secret: str = field(default_factory=lambda: _str("SESSION_SECRET") or secrets.token_urlsafe(32))
    admin_session_hours: int = field(default_factory=lambda: _int("ADMIN_SESSION_HOURS", 12))
    cookie_secure: bool = field(default_factory=lambda: _bool("PLAYGROUND_COOKIE_SECURE", _production()))

    def api_key(self, env_name: str) -> str:
        return _str(env_name) if env_name else ""


settings = Settings()
