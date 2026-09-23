"""Per-request AI usage, written to the Tredev Postgres for the admin panel.

This is the third sink behind `usage_ledger.write()`, alongside the JSONL
ledger and the Redis counters, and it is the one the Token Meter reads. The
three answer different questions and none replaces the others:

* JSONL   — complete, append-only, needs no infrastructure. The fallback if a
            row never reaches Postgres, and what a backfill would replay.
* Redis   — O(1) running totals for quotas and rate limits.
* Postgres— queryable history: per user, per conversation, per model, per day.
            Only this one can answer "what did last Tuesday cost".

Direction of traffic matters: ai-core *writes*, the Node backend only *reads*.
Routing these rows through an HTTP endpoint on the backend would add a hop and
a failure mode to something that is already best-effort, for no gain — both
services already share this database.

Everything here is best-effort and silent on failure. A metering row is worth
strictly less than a user's answer, so a Postgres outage degrades the admin
dashboard and must never surface in a chat reply.
"""

from __future__ import annotations

import threading
from typing import Any

from app.config.settings import settings
from app.logging_setup import get_logger

log = get_logger("usage.store")

_pool: Any = None
_pool_lock = threading.Lock()
# Set once when the driver or the database is unreachable, so a broken
# deployment logs a single warning instead of one per LLM call.
_disabled = False

INSERT_SQL = """
INSERT INTO ai_usage (
    request_id, role, user_id, conversation_id,
    provider, model,
    input_tokens, output_tokens, total_tokens,
    cached_input_tokens, reasoning_tokens,
    input_cost_usd, output_cost_usd, total_cost_usd, actual_cost_usd,
    input_price_per_million, output_price_per_million, fx_rate_usd_inr,
    latency_ms, status, error_code, error_message, fallback_used,
    created_at
) VALUES (
    %(request_id)s, %(role)s, %(user_id)s, %(conversation_id)s,
    %(provider)s, %(model)s,
    %(input_tokens)s, %(output_tokens)s, %(total_tokens)s,
    %(cached_input_tokens)s, %(reasoning_tokens)s,
    %(input_cost_usd)s, %(output_cost_usd)s, %(total_cost_usd)s, %(actual_cost_usd)s,
    %(input_price_per_million)s, %(output_price_per_million)s, %(fx_rate_usd_inr)s,
    %(latency_ms)s, %(status)s, %(error_code)s, %(error_message)s, %(fallback_used)s,
    to_timestamp(%(created_at)s)
)
ON CONFLICT (request_id, role) DO NOTHING
"""


def _get_pool() -> Any:
    """Lazily open a small connection pool. Returns None when unavailable."""
    global _pool, _disabled
    if _disabled:
        return None
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool
        if not settings.usage_store.enabled:
            _disabled = True
            return None
        try:
            from psycopg_pool import ConnectionPool

            # `open=True` connects eagerly so a bad DSN is reported here, once,
            # rather than on the first chat turn. `check` reconnects silently
            # after the database restarts.
            _pool = ConnectionPool(
                settings.usage_store.dsn,
                min_size=0,
                max_size=settings.usage_store.pool_size,
                timeout=settings.usage_store.timeout_seconds,
                open=True,
            )
        except Exception as exc:  # noqa: BLE001 — metering must never break a turn
            _disabled = True
            log.warning(
                "ai_usage postgres sink unavailable; usage will still reach the "
                "JSONL ledger and Redis counters",
                extra={"error_type": type(exc).__name__, "error_message": str(exc)[:200]},
            )
            return None
    return _pool


def _error_code(record: dict[str, Any]) -> str | None:
    """The exception type from the stored 'TypeName: message' error string.

    Kept separate from the message so the admin panel can group failures by
    kind without parsing free text.
    """
    error = record.get("error")
    if not error:
        return None
    head, _, _ = str(error).partition(":")
    return head.strip()[:120] or None


def to_row(record: dict[str, Any]) -> dict[str, Any]:
    """Map one ledger record onto the ai_usage column set.

    Pure and importable, so the mapping can be tested without a database.

    Two deliberate choices:

    * `session_id` becomes `conversation_id`. A chat session *is* the
      conversation in this application; inventing a separate id would create
      a second key for one thing.
    * Token counts pass through as-is, including None. The provider not
      reporting a count is a fact worth storing — replacing it with 0 would
      quietly understate real usage.
    """
    prices = settings.pricing.for_model(str(record.get("model") or ""))
    success = bool(record.get("success"))
    return {
        "request_id": record.get("request_id") or "unknown",
        "role": record.get("role") or "unknown",
        "user_id": record.get("user_id"),
        "conversation_id": record.get("session_id"),
        "provider": record.get("provider") or "unknown",
        "model": record.get("model") or "unknown",
        "input_tokens": record.get("input_tokens"),
        "output_tokens": record.get("output_tokens"),
        "total_tokens": record.get("total_tokens"),
        "cached_input_tokens": record.get("cached_tokens"),
        "reasoning_tokens": record.get("reasoning_tokens"),
        "input_cost_usd": record.get("input_cost_usd") or 0.0,
        "output_cost_usd": record.get("output_cost_usd") or 0.0,
        "total_cost_usd": record.get("total_cost_usd") or 0.0,
        # None, not 0: on a free tier the provider charges nothing *and*
        # reports nothing, and the UI must be able to tell those apart.
        "actual_cost_usd": record.get("native_cost_usd"),
        "input_price_per_million": prices[0],
        "output_price_per_million": prices[1],
        "fx_rate_usd_inr": settings.usage_store.usd_to_inr,
        "latency_ms": record.get("latency_ms"),
        "status": "success" if success else "failed",
        "error_code": None if success else _error_code(record),
        "error_message": None if success else (str(record.get("error") or "")[:500] or None),
        "fallback_used": bool(record.get("fallback_used")),
        "created_at": record.get("timestamp") or 0.0,
    }


def write(record: dict[str, Any]) -> bool:
    """Insert one usage row. Never raises."""
    pool = _get_pool()
    if pool is None:
        return False
    try:
        with pool.connection() as conn:
            conn.execute(INSERT_SQL, to_row(record))
        return True
    except Exception as exc:  # noqa: BLE001 — see module docstring
        log.warning(
            "ai_usage insert failed",
            extra={
                "request_id": record.get("request_id"),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:200],
            },
        )
        return False


def ping() -> tuple[bool, str]:
    """Used by scripts/test_connections.py."""
    if not settings.usage_store.enabled:
        return False, "AI_USAGE_DATABASE_URL is not set — admin Token Meter will have no data"
    pool = _get_pool()
    if pool is None:
        return False, "could not open a connection pool"
    try:
        with pool.connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM ai_usage").fetchone()
        return True, f"ai_usage reachable, {row[0]} rows recorded"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"


__all__ = ["write", "to_row", "ping"]
