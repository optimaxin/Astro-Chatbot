"""Credential redaction for everything this service writes down.

Two layers, because neither alone is enough:

* known secrets  - the exact values loaded into `settings` (API keys, tokens, the usage-store DSN
  password) are removed wherever they appear, whatever format the provider uses;
* patterns       - well-known credential shapes (NVIDIA / OpenRouter / OpenAI / Anthropic / Google /
  Pinecone / GitHub / AWS / Slack keys, JWTs, `Bearer` tokens, `user:password@host` URLs and
  `api_key=...` pairs), which catch a key that was never in `settings` (a rotated key, a key echoed
  back in a provider's error body, a key pasted into a request).

Applied by the JSON log formatter (message, fields and tracebacks), the console formatter and the
usage ledger's error text, so a secret cannot reach a log line, the JSONL ledger, Redis or the Token
Meter table through an exception message. Pure string work: it never raises and never logs.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "***redacted***"
MIN_SECRET_LEN = 8

_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"nvapi-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-or-(?:v1-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"pcsk_[A-Za-z0-9_]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}"),     # JWT
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-]{12,}=*"),
)
# scheme://user:password@host  ->  scheme://user:***@host
_URL_CREDENTIALS = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^\s:/@]+):([^\s@/]+)@")
# api_key=..., "token": "...", password: ..., Authorization=...
_PAIR = re.compile(
    r"(?i)((?:api[_\-]?key|apikey|access[_\-]?token|auth[_\-]?token|token|secret|client[_\-]?secret|password|passwd|"
    r"authorization)[\"']?\s*[:=]\s*[\"']?)([^\s\"',;&)}\]]{4,})")


def _known_secrets() -> set[str]:
    """Secret values currently loaded into settings (computed per call: settings can be replaced)."""
    try:
        from app.config.settings import settings

        values: set[str] = set()
        for group in vars(settings).values():
            for name, value in (vars(group).items() if hasattr(group, "__dict__") else ()):
                if isinstance(value, str) and len(value) >= MIN_SECRET_LEN and any(
                        m in name.lower() for m in ("api_key", "token", "secret", "password")):
                    values.add(value)
        dsn = getattr(getattr(settings, "usage_store", None), "dsn", "") or ""
        match = _URL_CREDENTIALS.search(dsn)
        if match and len(match.group(2)) >= MIN_SECRET_LEN:
            values.add(match.group(2))
        return values
    except Exception:  # noqa: BLE001 — redaction must never be the thing that fails
        return set()


def redact_text(text: Any) -> Any:
    """`text` with every credential replaced. Non-strings are returned unchanged."""
    if not isinstance(text, str) or not text:
        return text
    out = text
    for secret in sorted(_known_secrets(), key=len, reverse=True):
        if secret in out:
            out = out.replace(secret, REDACTED)
    for pattern in _PATTERNS:
        out = pattern.sub(REDACTED, out)
    out = _URL_CREDENTIALS.sub(lambda m: f"{m.group(1)}:{REDACTED}@", out)
    return _PAIR.sub(lambda m: m.group(1) + REDACTED, out)


def redact_value(value: Any, _depth: int = 0) -> Any:
    """`redact_text` applied through dicts, lists and tuples (bounded depth)."""
    if _depth > 6:
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact_value(v, _depth + 1) for v in value)
    return value


__all__ = ["redact_text", "redact_value", "REDACTED"]
