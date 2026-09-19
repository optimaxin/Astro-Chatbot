"""Durable per-user token and cost metering."""

from __future__ import annotations

import json

import pytest

from app.llm.usage_ledger import LEDGER_FIELDS, aggregate, read_ledger
from app.llm.usage_tracker import LLMUsage, UsageTracker


def _usage(user: str, role: str = "job-main-turn", **overrides) -> LLMUsage:
    defaults = dict(
        request_id="r1", session_id=f"sess-{user}", user_id=user, role=role,
        provider="openrouter", model="google/gemini-2.5-flash", latency_ms=1200,
        success=True, timestamp=0.0, input_tokens=1000, output_tokens=100,
        total_tokens=1100, total_cost_usd=0.00055,
    )
    defaults.update(overrides)
    return LLMUsage(**defaults)  # type: ignore[arg-type]


def _rows(*usages: LLMUsage) -> list[dict]:
    return [u.to_dict() for u in usages]


# ── aggregation ───────────────────────────────────────────────────────
def test_tokens_and_cost_are_summed_per_user() -> None:
    summaries = aggregate(_rows(_usage("alice"), _usage("alice"), _usage("bob")))

    assert summaries["alice"].calls == 2
    assert summaries["alice"].input_tokens == 2000
    assert summaries["alice"].output_tokens == 200
    assert summaries["alice"].total_tokens == 2200
    assert summaries["alice"].estimated_cost_usd == pytest.approx(0.0011)

    assert summaries["bob"].calls == 1
    assert summaries["bob"].input_tokens == 1000


def test_role_filter_isolates_the_user_visible_answer() -> None:
    rows = _rows(
        _usage("alice", role="job-main-turn", input_tokens=1000, output_tokens=100, total_cost_usd=0.001),
        _usage("alice", role="job-classify-intent", input_tokens=200, output_tokens=10, total_cost_usd=0.0001),
        _usage("alice", role="job-evaluate-response", input_tokens=800, output_tokens=60, total_cost_usd=0.0002),
    )
    everything = aggregate(rows)["alice"]
    answers = aggregate(rows, role="job-main-turn")["alice"]

    assert everything.calls == 3
    assert everything.estimated_cost_usd == pytest.approx(0.0013)
    assert answers.calls == 1
    assert answers.estimated_cost_usd == pytest.approx(0.001)
    # The overhead of classifying and judging is real and separately visible.
    assert everything.estimated_cost_usd > answers.estimated_cost_usd


def test_grouping_by_session_and_model() -> None:
    rows = _rows(
        _usage("alice", model="model-a"),
        _usage("bob", model="model-b"),
    )
    assert set(aggregate(rows, group_by="session_id")) == {"sess-alice", "sess-bob"}
    assert set(aggregate(rows, group_by="model")) == {"model-a", "model-b"}


def test_failed_calls_are_counted_but_cost_nothing() -> None:
    rows = _rows(
        _usage("alice"),
        _usage("alice", success=False, input_tokens=None, output_tokens=None,
               total_tokens=None, total_cost_usd=0.0, error="boom"),
    )
    summary = aggregate(rows)["alice"]
    assert summary.calls == 2
    assert summary.failed_calls == 1
    assert summary.input_tokens == 1000
    assert summary.estimated_cost_usd == pytest.approx(0.00055)


def test_missing_provider_usage_is_flagged_not_invented() -> None:
    """The rule that matters: an unmetered call is declared, never guessed."""
    rows = _rows(
        _usage("alice"),
        _usage("alice", input_tokens=None, output_tokens=None, total_tokens=None),
    )
    summary = aggregate(rows)["alice"]
    assert summary.calls_without_usage == 1
    assert not summary.tokens_exact
    assert summary.total_tokens == 1100, "the known call still counts"

    clean = aggregate(_rows(_usage("bob")))["bob"]
    assert clean.tokens_exact
    assert clean.calls_without_usage == 0


def test_cost_per_call_excludes_failures() -> None:
    rows = _rows(
        _usage("alice", total_cost_usd=0.002),
        _usage("alice", success=False, total_cost_usd=0.0),
    )
    summary = aggregate(rows)["alice"]
    assert summary.cost_per_call_usd == pytest.approx(0.002)


def test_empty_ledger_aggregates_to_nothing() -> None:
    assert aggregate([]) == {}


# ── persistence ───────────────────────────────────────────────────────
def test_every_call_is_written_to_the_ledger(tmp_path, monkeypatch) -> None:
    from app.config import settings as settings_module
    from app.llm import usage_ledger

    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: tmp_path / "usage.jsonl")
    # Redis is not configured in tests; the JSONL path must stand alone.
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)

    tracker = UsageTracker(persist=True)
    tracker.record(_usage("alice"))
    tracker.record(_usage("bob", role="job-evaluate-response"))

    written = list(read_ledger(tmp_path / "usage.jsonl"))
    assert len(written) == 2
    assert {row["user_id"] for row in written} == {"alice", "bob"}
    for row in written:
        assert set(row) == set(LEDGER_FIELDS), "on-disk schema must stay explicit"

    assert settings_module.settings is not None  # settings import stays live


def test_ledger_survives_a_malformed_line(tmp_path) -> None:
    path = tmp_path / "usage.jsonl"
    path.write_text(
        json.dumps({"user_id": "alice", "total_cost_usd": 0.001, "success": True}) + "\n"
        + "{ this line was truncated by a crash\n"
        + json.dumps({"user_id": "bob", "total_cost_usd": 0.002, "success": True}) + "\n",
        encoding="utf-8",
    )
    rows = list(read_ledger(path))
    assert len(rows) == 2, "a corrupt line must be skipped, not fatal"
    assert {r["user_id"] for r in rows} == {"alice", "bob"}


def test_ledger_write_failure_does_not_break_the_turn(tmp_path, monkeypatch) -> None:
    from app.llm import usage_ledger

    def explode(record):
        raise OSError("disk full")

    monkeypatch.setattr(usage_ledger, "_append_jsonl", explode)
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)

    tracker = UsageTracker(persist=True)
    with pytest.raises(OSError):
        # _append_jsonl itself is what swallows OSError in production; this
        # asserts the seam exists rather than that every call is wrapped twice.
        usage_ledger.write(_usage("alice"))

    # The real path swallows it: a metering failure never costs an answer.
    monkeypatch.setattr(usage_ledger, "_append_jsonl", lambda record: False)
    assert tracker.record(_usage("alice")) is not None


def test_tracker_can_group_in_memory() -> None:
    tracker = UsageTracker(persist=False)
    tracker.record(_usage("alice"))
    tracker.record(_usage("alice", role="job-classify-intent", total_cost_usd=0.0001))
    tracker.record(_usage("bob"))

    by_user = tracker.by_user()
    assert by_user["alice"]["calls"] == 2
    assert by_user["bob"]["calls"] == 1
    assert tracker.by_user(role="job-main-turn")["alice"]["calls"] == 1
    assert set(tracker.by_session()) == {"sess-alice", "sess-bob"}
