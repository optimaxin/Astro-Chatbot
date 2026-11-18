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
-- One row per (request_id, role) is the table's contract (see migration
-- 045), so a role's row holds its FINAL outcome. A failed attempt is written
-- first; if a retry or the fallback model then succeeds, that success
-- replaces the failure instead of being dropped — previously DO NOTHING
-- discarded every successful fallback, so the Token Meter under-reported
-- exactly the turns where the fallback mattered. A success is never
-- overwritten, which keeps a replayed ledger idempotent. The primary
-- failure's code is kept on the success row (fallback_used = TRUE).
ON CONFLICT (request_id, role) DO UPDATE SET
    provider = EXCLUDED.provider,
    model = EXCLUDED.model,
    input_tokens = EXCLUDED.input_tokens,
    output_tokens = EXCLUDED.output_tokens,
    total_tokens = EXCLUDED.total_tokens,
    cached_input_tokens = EXCLUDED.cached_input_tokens,
    reasoning_tokens = EXCLUDED.reasoning_tokens,
    input_cost_usd = EXCLUDED.input_cost_usd,
    output_cost_usd = EXCLUDED.output_cost_usd,
    total_cost_usd = EXCLUDED.total_cost_usd,
    actual_cost_usd = EXCLUDED.actual_cost_usd,
    input_price_per_million = EXCLUDED.input_price_per_million,
    output_price_per_million = EXCLUDED.output_price_per_million,
    fx_rate_usd_inr = EXCLUDED.fx_rate_usd_inr,
    latency_ms = EXCLUDED.latency_ms,
    status = EXCLUDED.status,
    error_code = COALESCE(EXCLUDED.error_code, ai_usage.error_code),
    error_message = COALESCE(EXCLUDED.error_message, ai_usage.error_message),
    fallback_used = EXCLUDED.fallback_used OR ai_usage.fallback_used,
    created_at = EXCLUDED.created_at
WHERE ai_usage.status <> 'success'
"""


_columns: dict[str, set[str]] = {}


def table_columns(table: str) -> set[str]:
    """Columns that exist on `table`, read once per process. Lets ai-core
    write a new column only after the backend migration has added it, so the
    two can be deployed in either order."""
    if table in _columns:
        return _columns[table]
    pool = _get_pool()
    if pool is None:
        return set()
    try:
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table,)
            ).fetchall()
        _columns[table] = {r[0] for r in rows}
    except Exception:  # noqa: BLE001 — unknown schema: write only the base columns
        return set()
    return _columns[table]


def _insert_sql() -> str:
    if "traffic_source" not in table_columns("ai_usage"):
        return INSERT_SQL
    return (INSERT_SQL
            .replace("    created_at\n) VALUES", "    created_at, traffic_source\n) VALUES", 1)
            .replace("to_timestamp(%(created_at)s)\n)", "to_timestamp(%(created_at)s), %(traffic_source)s\n)", 1))


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
    """The failure class, kept separate from the message so the admin panel
    can group failures by kind without parsing free text.

    Prefers the gateway's normalised code (`rate_limited:429`, `timeout`, ...),
    which groups the same failure identically across providers; falls back to
    the exception type from the 'TypeName: message' error string for records
    written before normalisation existed.
    """
    if record.get("error_code"):
        return str(record["error_code"])[:120]
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
        "role": role_key(record),
        "base_role": record.get("role") or "unknown",
        "call_id": call_id_of(record),
        "purpose": record.get("purpose"),
        "attempt": int(record.get("attempt") or 0),
        "route": record.get("route"),
        "prompt_version": record.get("prompt_version"),
        "requested_model": record.get("requested_model"),
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
        # A successful *fallback* keeps the code of the primary failure that
        # triggered it; any other success carries no error fields.
        "error_code": (
            _error_code(record) if not success
            else (str(record.get("error_code"))[:120]
                  if record.get("fallback_used") and record.get("error_code") else None)
        ),
        "error_message": None if success else (str(record.get("error") or "")[:500] or None),
        "fallback_used": bool(record.get("fallback_used")),
        "created_at": record.get("timestamp") or 0.0,
        "traffic_source": traffic_source_of(record),
    }


# Calls made outside a chat turn (scripts, probes, direct gateway calls)
# carry no user. /chat always attaches the caller's identity, so a row with
# no user is never real-user traffic, whatever the process default says.
NO_USER = {None, "", "-"}


def traffic_source_of(record: dict[str, Any]) -> str:
    source = record.get("traffic_source") or "user"
    if source == "user" and record.get("user_id") in NO_USER:
        return "debug"
    return source


# ── per-call rows (migration 059) ─────────────────────────────────────
# Purposes recorded under their own summary role key, so role 'job-main-turn'
# keeps meaning "the answer" for the dashboard and the ai_evaluations join.
SECONDARY_PURPOSES = frozenset({"repair", "language_rewrite", "grammar_correction", "validation_fallback",
                                "grounding_verify"})

_CALL_COLUMNS = (
    "call_id", "request_id", "role", "purpose", "attempt", "route", "prompt_version",
    "user_id", "conversation_id", "provider", "model", "requested_model",
    "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens",
    "input_cost_usd", "output_cost_usd", "total_cost_usd", "actual_cost_usd",
    "input_price_per_million", "output_price_per_million", "fx_rate_usd_inr",
    "latency_ms", "status", "error_code", "error_message", "fallback_used", "traffic_source",
)
CALL_INSERT_SQL = (
    f"INSERT INTO ai_usage_calls ({', '.join(_CALL_COLUMNS)}, created_at)\n"
    # The call row stores the real role; the purpose has its own column.
    f"VALUES ({', '.join('%(base_role)s' if c == 'role' else f'%({c})s' for c in _CALL_COLUMNS)}, "
    "to_timestamp(%(created_at)s))\n"
    "ON CONFLICT (call_id) DO NOTHING\nRETURNING id"
)


def _add(col: str) -> str:
    """Sum two nullable counts: NULL only when neither call reported one."""
    return (f"{col} = CASE WHEN ai_usage.{col} IS NULL AND EXCLUDED.{col} IS NULL THEN NULL "
            f"ELSE COALESCE(ai_usage.{col}, 0) + COALESCE(EXCLUDED.{col}, 0) END")


# The first success in a (request_id, role) summary decides its model and latency;
# a failure never replaces a success. Tokens and cost are SUMMED over every call.
_PROMOTE = "(ai_usage.status <> 'success' AND EXCLUDED.status = 'success')"
SUMMARY_ACCUMULATE_SQL = (
    INSERT_SQL.split("-- One row per (request_id, role)")[0]
    .replace("    created_at\n) VALUES", "    created_at, traffic_source, purpose, calls\n) VALUES", 1)
    .replace("to_timestamp(%(created_at)s)\n)", "to_timestamp(%(created_at)s), %(traffic_source)s, %(purpose)s, 1\n)", 1)
    + "ON CONFLICT (request_id, role) DO UPDATE SET\n    "
    + ",\n    ".join([
        _add("input_tokens"), _add("output_tokens"), _add("total_tokens"), _add("cached_input_tokens"),
        _add("reasoning_tokens"), _add("actual_cost_usd"),
        "input_cost_usd = ai_usage.input_cost_usd + EXCLUDED.input_cost_usd",
        "output_cost_usd = ai_usage.output_cost_usd + EXCLUDED.output_cost_usd",
        "total_cost_usd = ai_usage.total_cost_usd + EXCLUDED.total_cost_usd",
        f"provider = CASE WHEN {_PROMOTE} THEN EXCLUDED.provider ELSE ai_usage.provider END",
        f"model = CASE WHEN {_PROMOTE} THEN EXCLUDED.model ELSE ai_usage.model END",
        f"latency_ms = CASE WHEN {_PROMOTE} THEN EXCLUDED.latency_ms ELSE ai_usage.latency_ms END",
        "status = CASE WHEN ai_usage.status = 'success' OR EXCLUDED.status = 'success' "
        "THEN 'success' ELSE 'failed' END",
        "error_code = COALESCE(ai_usage.error_code, EXCLUDED.error_code)",
        "error_message = COALESCE(ai_usage.error_message, EXCLUDED.error_message)",
        "fallback_used = ai_usage.fallback_used OR EXCLUDED.fallback_used",
        "purpose = COALESCE(ai_usage.purpose, EXCLUDED.purpose)",
        "calls = ai_usage.calls + 1",
    ])
)


def role_key(record: dict[str, Any]) -> str:
    role = record.get("role") or "unknown"
    purpose = record.get("purpose")
    return f"{role}:{purpose}" if purpose in SECONDARY_PURPOSES else role


def call_id_of(record: dict[str, Any]) -> str:
    """The record's own id; for ledger lines written before call ids existed, a
    deterministic one, so replaying the same line twice is still idempotent."""
    if record.get("call_id"):
        return str(record["call_id"])
    import hashlib

    # Beyond identity, the figures that tell two calls apart: the ledger's timestamps were
    # whole seconds, so two calls of one request and role in the same second collided on
    # identity alone (108 of 21,012 historical lines, measured on a scratch database).
    key = "|".join(str(record.get(k)) for k in ("request_id", "role", "model", "timestamp", "attempt", "success",
                                                  "latency_ms", "input_tokens", "output_tokens"))
    return "legacy-" + hashlib.sha256(key.encode()).hexdigest()[:24]


def _per_call_schema() -> bool:
    return "call_id" in table_columns("ai_usage_calls") and "calls" in table_columns("ai_usage")


def write(record: dict[str, Any]) -> bool:
    """Record one invocation. Never raises.

    After migration 059: its own ai_usage_calls row (never updated) plus its
    tokens/cost added into the ai_usage summary row, in ONE transaction; a call
    already recorded (replayed ledger) changes nothing. Before 059: the
    previous one-row-per-(request, role) upsert, so either deploy order works.
    """
    pool = _get_pool()
    if pool is None:
        return False
    try:
        row = to_row(record)
        with pool.connection() as conn:
            if _per_call_schema():
                with conn.transaction():
                    if conn.execute(CALL_INSERT_SQL, row).fetchone() is None:
                        return True                       # already counted
                    conn.execute(SUMMARY_ACCUMULATE_SQL, row)
            else:
                conn.execute(_insert_sql(), row)
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
