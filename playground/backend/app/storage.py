"""Experiment storage (SQLite).

One row per experiment. Nested or evolving data (params, the effective
request, evaluation, metadata, pricing snapshot) is stored as JSON so new
metrics need no migration. A separate database from production, on purpose.

Schema changes are applied by `_MIGRATIONS`, tracked in PRAGMA user_version,
so an existing database on the persistent disk is upgraded in place.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from app.settings import settings

_MIGRATIONS: list[str] = [
    # 1 — the original experiments table.
    """
    CREATE TABLE IF NOT EXISTS experiments (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at          REAL NOT NULL,
        session_id          TEXT,
        provider            TEXT NOT NULL,
        model               TEXT NOT NULL,
        system_prompt       TEXT,
        user_prompt         TEXT NOT NULL,
        context             TEXT,
        params              TEXT,
        effective_request   TEXT,
        status              TEXT NOT NULL,
        error               TEXT,
        response            TEXT,
        humanized_response  TEXT,
        input_tokens        INTEGER,
        output_tokens       INTEGER,
        total_tokens        INTEGER,
        input_cost          REAL,
        output_cost         REAL,
        total_cost          REAL,
        currency            TEXT,
        latency_ms          INTEGER,
        provider_latency_ms INTEGER,
        evaluation          TEXT,
        evaluation_overall  REAL,
        human_rating        TEXT,
        human_feedback      TEXT,
        edited_response     TEXT,
        metadata            TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_experiments_session ON experiments(session_id);
    """,
    # 2 — public testing: request ids, anonymous client ids, pricing snapshots,
    #     admin sessions and the audit log.
    """
    ALTER TABLE experiments ADD COLUMN request_id TEXT;
    ALTER TABLE experiments ADD COLUMN client_id TEXT;
    ALTER TABLE experiments ADD COLUMN tester_name TEXT;
    ALTER TABLE experiments ADD COLUMN model_version TEXT;
    ALTER TABLE experiments ADD COLUMN usage_source TEXT;
    ALTER TABLE experiments ADD COLUMN pricing_snapshot TEXT;
    ALTER TABLE experiments ADD COLUMN feedback_at REAL;
    CREATE UNIQUE INDEX IF NOT EXISTS idx_experiments_request ON experiments(request_id)
        WHERE request_id IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_experiments_client ON experiments(client_id, id DESC);
    CREATE INDEX IF NOT EXISTS idx_experiments_model ON experiments(provider, model);
    CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
    CREATE TABLE IF NOT EXISTS admin_sessions (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        session_token_hash TEXT NOT NULL UNIQUE,
        expires_at         REAL NOT NULL,
        created_at         REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS audit_logs (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        action    TEXT NOT NULL,
        timestamp REAL NOT NULL,
        ip        TEXT,
        details   TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_logs(timestamp DESC);
    """,
]

_JSON_FIELDS = ("params", "effective_request", "evaluation", "metadata", "pricing_snapshot")
_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for number, script in enumerate(_MIGRATIONS, start=1):
        if number <= version:
            continue
        with conn:
            for statement in (s.strip() for s in script.split(";")):
                if not statement:
                    continue
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError as exc:
                    # A column added by hand earlier is not a failure.
                    if "duplicate column" not in str(exc):
                        raise
            conn.execute(f"PRAGMA user_version = {number}")


def _connection(path: Path | None = None) -> sqlite3.Connection:
    global _conn
    if _conn is None:
        db = path or settings.db_path
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        _migrate(conn)
        _conn = conn
    return _conn


def init() -> None:
    with _lock:
        _connection()


def reset_for_tests(path: Path) -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None
        _connection(path)


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None


def _encode(values: dict[str, Any]) -> dict[str, Any]:
    return {k: (json.dumps(v, ensure_ascii=False) if k in _JSON_FIELDS and v is not None else v)
            for k, v in values.items()}


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    for key in _JSON_FIELDS:
        if out.get(key):
            out[key] = json.loads(out[key])
    return out


# ── experiments ──────────────────────────────────────────────────────
def find_by_request_id(request_id: str) -> dict[str, Any] | None:
    with _lock:
        row = _connection().execute("SELECT * FROM experiments WHERE request_id = ?", (request_id,)).fetchone()
    return _decode(row) if row else None


def create_experiment(values: dict[str, Any]) -> dict[str, Any]:
    """Insert one experiment. A repeated request_id returns the first record
    instead of creating a duplicate."""
    values = _encode({"created_at": time.time(), **values})
    cols = ", ".join(values)
    marks = ", ".join(f":{k}" for k in values)
    with _lock:
        conn = _connection()
        try:
            with conn:
                cur = conn.execute(f"INSERT INTO experiments ({cols}) VALUES ({marks})", values)
        except sqlite3.IntegrityError:
            existing = find_by_request_id(values.get("request_id") or "")
            if existing is None:
                raise
            return existing
        new_id = cur.lastrowid
    return get_experiment(new_id)  # type: ignore[arg-type,return-value]


_UPDATABLE = {"evaluation", "evaluation_overall", "human_rating", "human_feedback",
              "edited_response", "humanized_response", "metadata", "feedback_at"}


def update_experiment(experiment_id: int, values: dict[str, Any]) -> dict[str, Any] | None:
    values = {k: v for k, v in values.items() if k in _UPDATABLE}
    if values:
        encoded = _encode(values)
        assignments = ", ".join(f"{k} = :{k}" for k in encoded)
        with _lock:
            conn = _connection()
            with conn:
                conn.execute(f"UPDATE experiments SET {assignments} WHERE id = :_id",
                             {**encoded, "_id": experiment_id})
    return get_experiment(experiment_id)


def get_experiment(experiment_id: int) -> dict[str, Any] | None:
    with _lock:
        row = _connection().execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    return _decode(row) if row else None


_SUMMARY = ("id, created_at, session_id, provider, model, user_prompt, status, total_tokens, "
            "total_cost, currency, latency_ms, evaluation_overall, human_rating")


def list_experiments(limit: int = 50, offset: int = 0, client_id: str | None = None) -> list[dict[str, Any]]:
    where, args = ("WHERE client_id = ?", [client_id]) if client_id else ("", [])
    with _lock:
        rows = _connection().execute(
            f"SELECT {_SUMMARY} FROM experiments {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            (*args, limit, offset)).fetchall()
    return [dict(r) for r in rows]


def usage_since(since_epoch: float) -> dict[str, float]:
    """Request count and USD spend since a moment — feeds the global spend guard."""
    with _lock:
        row = _connection().execute(
            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN currency = 'USD' THEN total_cost END), 0) "
            "FROM experiments WHERE created_at >= ?", (since_epoch,)).fetchone()
    return {"requests": row[0], "cost_usd": float(row[1] or 0)}


# ── admin queries ────────────────────────────────────────────────────
_FILTER_COLUMNS = {"provider": "provider", "model": "model", "status": "status",
                   "session_id": "session_id", "client_id": "client_id"}


def _where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    args: list[Any] = []
    if filters.get("since") is not None:
        clauses.append("created_at >= ?")
        args.append(filters["since"])
    if filters.get("until") is not None:
        clauses.append("created_at < ?")
        args.append(filters["until"])
    for key, column in _FILTER_COLUMNS.items():
        if filters.get(key):
            clauses.append(f"{column} = ?")
            args.append(filters[key])
    if filters.get("q"):
        clauses.append("(user_prompt LIKE ? ESCAPE '\\' OR response LIKE ? ESCAPE '\\' OR tester_name LIKE ? ESCAPE '\\')")
        needle = "%" + filters["q"].replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        args.extend([needle, needle, needle])
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", args


def search_experiments(filters: dict[str, Any], limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
    where, args = _where(filters)
    with _lock:
        conn = _connection()
        total = conn.execute(f"SELECT COUNT(*) FROM experiments {where}", args).fetchone()[0]
        rows = conn.execute(
            "SELECT id, created_at, request_id, session_id, client_id, tester_name, provider, model, "
            "substr(user_prompt, 1, 160) AS prompt_preview, substr(response, 1, 160) AS response_preview, "
            "input_tokens, output_tokens, total_tokens, total_cost, currency, latency_ms, status, error, "
            f"evaluation_overall, human_rating FROM experiments {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            (*args, limit, offset)).fetchall()
    return [dict(r) for r in rows], total


def all_experiments(filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Every matching record, oldest first — for analytics and Excel export."""
    where, args = _where(filters)
    with _lock:
        rows = _connection().execute(f"SELECT * FROM experiments {where} ORDER BY id ASC", args).fetchall()
    return [_decode(r) for r in rows]


def distinct_values() -> dict[str, list[str]]:
    with _lock:
        conn = _connection()
        providers = [r[0] for r in conn.execute("SELECT DISTINCT provider FROM experiments ORDER BY 1")]
        models = [r[0] for r in conn.execute("SELECT DISTINCT model FROM experiments ORDER BY 1")]
    return {"providers": providers, "models": models}


# ── admin sessions and audit log ─────────────────────────────────────
def create_admin_session(token_hash: str, expires_at: float) -> None:
    with _lock:
        conn = _connection()
        with conn:
            conn.execute("DELETE FROM admin_sessions WHERE expires_at < ?", (time.time(),))
            conn.execute("INSERT INTO admin_sessions (session_token_hash, expires_at, created_at) VALUES (?, ?, ?)",
                         (token_hash, expires_at, time.time()))


def admin_session_valid(token_hash: str) -> bool:
    with _lock:
        row = _connection().execute("SELECT expires_at FROM admin_sessions WHERE session_token_hash = ?",
                                    (token_hash,)).fetchone()
    return bool(row) and row[0] > time.time()


def delete_admin_session(token_hash: str) -> None:
    with _lock:
        conn = _connection()
        with conn:
            conn.execute("DELETE FROM admin_sessions WHERE session_token_hash = ?", (token_hash,))


def audit(action: str, ip: str | None = None, details: dict[str, Any] | None = None) -> None:
    with _lock:
        conn = _connection()
        with conn:
            conn.execute("INSERT INTO audit_logs (action, timestamp, ip, details) VALUES (?, ?, ?, ?)",
                         (action, time.time(), ip, json.dumps(details or {}, ensure_ascii=False)))


def audit_logs(limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
    with _lock:
        conn = _connection()
        total = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
        rows = conn.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ? OFFSET ?",
                            (limit, offset)).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        item["details"] = json.loads(item["details"]) if item.get("details") else {}
        out.append(item)
    return out, total


# ── backup ───────────────────────────────────────────────────────────
def backup_to(dest: Path) -> Path:
    """A consistent copy of the live database (SQLite online backup API)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        target = sqlite3.connect(dest)
        try:
            _connection().backup(target)
        finally:
            target.close()
    return dest
