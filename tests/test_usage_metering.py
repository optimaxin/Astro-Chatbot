"""Usage metering: the bounded in-memory tracker and the Token Meter writer.

Database behaviour (migration 059, the upsert) is verified against a scratch
Postgres separately; everything here is what can be proven without one: the
ring buffer, the persistent ledger staying complete, row mapping, the SQL
that is generated, and the writer's two schema paths.
"""

from __future__ import annotations

import json
import threading

import pytest

from app.llm import usage_ledger, usage_store
from app.llm.usage_tracker import LLMUsage, UsageTracker

# conftest replaces usage_store.write with a no-op for every test (nothing may reach the real
# database); the writer's own logic is tested through the original, captured at import time.
_REAL_WRITE = usage_store.write


def _usage(i: int, request_id: str = "r", purpose: str = "answer", role: str = "job-main-turn", **kw) -> LLMUsage:
    return LLMUsage(request_id=request_id, session_id="s", user_id="u", role=role, provider="nvidia",
                    model="nvidia/nemotron-3.5-lightning-30b-a3b", latency_ms=100 + i, success=True,
                    timestamp=1000.0 + i, input_tokens=10, output_tokens=5, total_tokens=15,
                    purpose=purpose, **kw)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "usage.jsonl"
    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: path)
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)
    monkeypatch.setattr(usage_store, "write", lambda record: False)
    return path


# ── 11. the in-memory window is bounded ───────────────────────────────
def test_the_tracker_keeps_only_the_most_recent_records(ledger) -> None:
    tracker = UsageTracker(persist=False, max_records=50)
    for i in range(120):
        tracker.record(_usage(i, request_id=f"r{i}"))
    assert len(tracker.records) == 50 and tracker.evicted == 70
    assert tracker.records[0].request_id == "r70" and tracker.records[-1].request_id == "r119"
    assert tracker.for_request("r0") == [] and len(tracker.for_request("r119")) == 1
    assert tracker.totals()["calls"] == 50


def test_eviction_never_touches_the_persistent_ledger(ledger) -> None:
    tracker = UsageTracker(persist=True, max_records=10)
    for i in range(40):
        tracker.record(_usage(i, request_id=f"r{i}"))
    lines = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(tracker.records) == 10
    assert [r["request_id"] for r in lines] == [f"r{i}" for i in range(40)], "every call is in the ledger"


def test_the_tracker_is_safe_under_concurrent_writers_and_readers(ledger) -> None:
    tracker = UsageTracker(persist=False, max_records=200)
    errors: list[BaseException] = []

    def writer(n: int) -> None:
        try:
            for i in range(300):
                tracker.record(_usage(i, request_id=f"w{n}-{i}"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(300):
                tracker.totals()
                tracker.for_request("w0-1")
                tracker.by_model()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)] + \
              [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(tracker.records) == 200 and tracker.evicted == 6 * 300 - 200


def test_the_default_bound_is_configurable_from_settings() -> None:
    from app.config.settings import settings

    assert settings.usage_tracker_max_records >= 1
    assert UsageTracker(persist=False).records == []


def test_a_call_with_no_purpose_is_visibly_unlabelled(ledger) -> None:
    tracker = UsageTracker(persist=False, max_records=5)
    tracker.record(LLMUsage(request_id="x", session_id="s", user_id="u", role="job-main-turn", provider="nvidia",
                            model="m", latency_ms=1, success=True, timestamp=1.0))
    assert tracker.records[0].purpose == "unlabelled:job-main-turn"


# ── 5. Token Meter: every call is its own record ──────────────────────
def test_secondary_purposes_get_their_own_summary_role_but_keep_the_real_role() -> None:
    answer = usage_store.to_row(_usage(0, purpose="answer").to_dict())
    repair = usage_store.to_row(_usage(1, purpose="repair").to_dict())
    classify = usage_store.to_row(_usage(2, purpose="classify", role="job-classify-intent").to_dict())
    assert answer["role"] == "job-main-turn", "the dashboard's role = 'job-main-turn' still means the answer"
    assert repair["role"] == "job-main-turn:repair" and repair["base_role"] == "job-main-turn"
    assert classify["role"] == "job-classify-intent"
    for purpose in ("language_rewrite", "grammar_correction", "validation_fallback"):
        assert usage_store.to_row(_usage(3, purpose=purpose).to_dict())["role"] == f"job-main-turn:{purpose}"


def test_every_call_has_a_stable_unique_call_id() -> None:
    a, b = _usage(0), _usage(0)
    assert a.call_id != b.call_id and len(a.call_id) == 32
    row = usage_store.to_row(a.to_dict())
    assert row["call_id"] == a.call_id and usage_store.to_row(a.to_dict())["call_id"] == a.call_id


def test_a_ledger_line_without_a_call_id_gets_a_deterministic_one() -> None:
    legacy = {k: v for k, v in _usage(5).to_dict().items() if k != "call_id"}
    first, second = usage_store.call_id_of(legacy), usage_store.call_id_of(dict(legacy))
    assert first == second and first.startswith("legacy-")
    assert usage_store.call_id_of({**legacy, "attempt": 1}) != first, "a retry is a different call"


def test_a_failed_attempt_and_the_success_are_separate_rows() -> None:
    failed = usage_store.to_row(_usage(0, attempt=0).to_dict() | {"success": False, "error_code": "timeout",
                                                                   "error": "APITimeoutError: x"})
    ok = usage_store.to_row(_usage(1, attempt=1, fallback_used=True).to_dict())
    assert failed["status"] == "failed" and failed["error_code"] == "timeout"
    assert ok["status"] == "success" and failed["call_id"] != ok["call_id"]


def test_the_call_insert_never_updates_an_existing_call() -> None:
    sql = usage_store.CALL_INSERT_SQL
    assert "ON CONFLICT (call_id) DO NOTHING" in sql and "DO UPDATE" not in sql
    assert "%(base_role)s" in sql and "%(purpose)s" in sql


def test_the_summary_upsert_adds_instead_of_overwriting() -> None:
    sql = usage_store.SUMMARY_ACCUMULATE_SQL
    assert "ON CONFLICT (request_id, role) DO UPDATE" in sql
    assert "WHERE ai_usage.status <> 'success'" not in sql, "a later success is no longer discarded"
    for col in ("input_tokens", "output_tokens", "total_tokens", "total_cost_usd"):
        assert f"{col} = " in sql
    assert "calls = ai_usage.calls + 1" in sql
    assert "ai_usage.input_tokens IS NULL AND EXCLUDED.input_tokens IS NULL THEN NULL" in sql, \
        "unreported tokens stay NULL, never an invented 0"
    assert sql.count("%(") == sql.count("%(") and "-- One row per" not in sql


class _Conn:
    def __init__(self, calls, rows):
        self.calls, self.rows = calls, rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def transaction(self):
        return self

    def execute(self, sql, params=None):
        self.calls.append(("tx" if "ai_usage_calls" in sql else "summary", sql.split()[2] if "INSERT" in sql else sql[:20]))
        return self

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def connection(self):
        return self.conn


def _writer(monkeypatch, columns, fetch_rows):
    calls: list = []
    monkeypatch.setattr(usage_store, "write", _REAL_WRITE)
    monkeypatch.setattr(usage_store, "_get_pool", lambda: _Pool(_Conn(calls, fetch_rows)))
    monkeypatch.setattr(usage_store, "table_columns", lambda table: columns.get(table, set()))
    return calls


def test_after_the_migration_each_call_writes_a_call_row_then_adds_into_the_summary(monkeypatch) -> None:
    calls = _writer(monkeypatch, {"ai_usage_calls": {"call_id"}, "ai_usage": {"calls", "traffic_source"}}, [(1,)])
    assert usage_store.write(_usage(0).to_dict()) is True
    assert [c[0] for c in calls] == ["tx", "summary"]


def test_a_replayed_call_changes_nothing(monkeypatch) -> None:
    calls = _writer(monkeypatch, {"ai_usage_calls": {"call_id"}, "ai_usage": {"calls"}}, [])   # RETURNING gave no row
    assert usage_store.write(_usage(0).to_dict()) is True
    assert [c[0] for c in calls] == ["tx"], "the summary is not touched twice for one call"


def test_before_the_migration_the_previous_writer_is_used(monkeypatch) -> None:
    calls = _writer(monkeypatch, {"ai_usage": {"traffic_source"}}, [])
    assert usage_store.write(_usage(0).to_dict()) is True
    assert [c[0] for c in calls] == ["summary"], "either deploy order works"


def test_a_database_failure_never_raises(monkeypatch) -> None:
    class Boom:
        def connection(self):
            raise RuntimeError("db down")

    monkeypatch.setattr(usage_store, "write", _REAL_WRITE)
    monkeypatch.setattr(usage_store, "_get_pool", lambda: Boom())
    assert usage_store.write(_usage(0).to_dict()) is False


def test_the_migration_is_additive_and_its_rollback_destroys_nothing() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "backend" / "migrations"
    up = (root / "059_ai_usage_calls.sql").read_text(encoding="utf-8").upper()
    down = (root / "rollback" / "059_ai_usage_calls.down.sql").read_text(encoding="utf-8").upper()
    for forbidden in ("DROP TABLE", "DROP COLUMN", "DELETE FROM AI_USAGE", "TRUNCATE", "UPDATE AI_USAGE"):
        assert forbidden not in up, forbidden
    assert "IF NOT EXISTS" in up and "ADD COLUMN IF NOT EXISTS CALLS" in up.replace("\n", " ")
    for forbidden in ("DROP TABLE", "DROP COLUMN", "TRUNCATE", "DELETE FROM AI_USAGE"):
        assert forbidden not in down, forbidden
    assert "RENAME TO AI_USAGE_CALLS_ARCHIVED_059" in down and down.strip().endswith("COMMIT;")
    assert not (root / "rollback").glob("*.sql") or all(p.parent.name == "rollback" for p in (root / "rollback").glob("*.sql"))
    # The runner only reads top-level migrations/*.sql, so the rollback is never auto-applied.
    assert not [p for p in root.glob("*.sql") if "down" in p.name]
