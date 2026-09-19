"""Token tracking (Phase 12) and cost calculation (Phase 13)."""

from __future__ import annotations

import pytest

from app.config.settings import PricingSettings, _parse_prices
from app.llm.usage_tracker import (
    LLMUsage,
    UsageTracker,
    calculate_cost,
    extract_native_cost,
    extract_usage,
    new_request_id,
)


# ── token extraction ──────────────────────────────────────────────────
def test_extract_usage_reads_provider_numbers() -> None:
    tokens = extract_usage(
        {
            "prompt_tokens": 1842,
            "completion_tokens": 287,
            "total_tokens": 2129,
            "prompt_tokens_details": {"cached_tokens": 512},
            "completion_tokens_details": {"reasoning_tokens": 64},
        }
    )
    assert tokens["input_tokens"] == 1842
    assert tokens["output_tokens"] == 287
    assert tokens["total_tokens"] == 2129
    assert tokens["cached_tokens"] == 512
    assert tokens["reasoning_tokens"] == 64


def test_missing_usage_is_none_not_zero() -> None:
    """The rule the whole module exists for: never invent a token count."""
    tokens = extract_usage(None)
    assert all(value is None for value in tokens.values())

    partial = extract_usage({"prompt_tokens": 100})
    assert partial["input_tokens"] == 100
    assert partial["output_tokens"] is None
    assert partial["total_tokens"] is None
    assert partial["cached_tokens"] is None


def test_total_is_derived_only_when_both_halves_are_known() -> None:
    assert extract_usage({"prompt_tokens": 10, "completion_tokens": 5})["total_tokens"] == 15
    assert extract_usage({"completion_tokens": 5})["total_tokens"] is None


def test_garbage_usage_does_not_raise() -> None:
    assert extract_usage({"prompt_tokens": "lots"})["input_tokens"] is None
    assert extract_usage("not a usage object")["input_tokens"] is None


def test_native_cost_extraction() -> None:
    assert extract_native_cost({"cost": 0.00042}) == pytest.approx(0.00042)
    assert extract_native_cost({"cost": 0}) == 0.0
    assert extract_native_cost({}) is None
    assert extract_native_cost(None) is None
    assert extract_native_cost({"cost": "free"}) is None


# ── pricing ───────────────────────────────────────────────────────────
def test_price_table_parsing() -> None:
    table = _parse_prices("a/model=0.12:0.30, b/model=1:2")
    assert table["a/model"] == (0.12, 0.30)
    assert table["b/model"] == (1.0, 2.0)


def test_malformed_price_entries_are_skipped_not_fatal() -> None:
    table = _parse_prices("good/model=0.1:0.2,broken,also=bad:entries,=0.1:0.2")
    assert table["good/model"] == (0.1, 0.2)
    assert "broken" not in table


def test_cost_calculation() -> None:
    pricing = PricingSettings(table={"m": (0.12, 0.30)}, default_input_per_1m=0.0, default_output_per_1m=0.0)
    assert pricing.for_model("m") == (0.12, 0.30)

    # 1,000,000 in / 1,000,000 out at $0.12 / $0.30.
    import app.llm.usage_tracker as tracker
    from app.config import settings as settings_module

    original = settings_module.settings.pricing
    object.__setattr__(settings_module.settings, "pricing", pricing)
    try:
        input_cost, output_cost, total = tracker.calculate_cost("m", 1_000_000, 1_000_000)
        assert input_cost == pytest.approx(0.12)
        assert output_cost == pytest.approx(0.30)
        assert total == pytest.approx(0.42)

        # Half a million input tokens costs half as much.
        half, _, _ = tracker.calculate_cost("m", 500_000, 0)
        assert half == pytest.approx(0.06)
    finally:
        object.__setattr__(settings_module.settings, "pricing", original)


def test_unknown_model_falls_back_to_default_price() -> None:
    pricing = PricingSettings(table={}, default_input_per_1m=1.0, default_output_per_1m=2.0)
    assert pricing.for_model("never/seen") == (1.0, 2.0)


def test_unknown_token_counts_cost_nothing_rather_than_guessing() -> None:
    input_cost, output_cost, total = calculate_cost("anything", None, None)
    assert (input_cost, output_cost, total) == (0.0, 0.0, 0.0)


# ── accumulation ──────────────────────────────────────────────────────
def _usage(request_id: str, **overrides) -> LLMUsage:
    defaults = dict(
        request_id=request_id, session_id="s", user_id="u", role="job-main-turn",
        provider="openrouter", model="m", latency_ms=100, success=True, timestamp=0.0,
        input_tokens=100, output_tokens=50, total_tokens=150, total_cost_usd=0.001,
    )
    defaults.update(overrides)
    return LLMUsage(**defaults)  # type: ignore[arg-type]


def test_tracker_totals_only_count_successful_calls() -> None:
    tracker = UsageTracker()
    tracker.record(_usage("r1"))
    tracker.record(_usage("r1", role="job-evaluate-response", input_tokens=200, output_tokens=20, total_tokens=220))
    tracker.record(_usage("r2", success=False, input_tokens=None, output_tokens=None,
                          total_tokens=None, total_cost_usd=0.0, error="boom"))

    totals = tracker.totals()
    assert totals["calls"] == 3
    assert totals["successful_calls"] == 2
    assert totals["failed_calls"] == 1
    assert totals["input_tokens"] == 300
    assert totals["total_tokens"] == 370
    assert totals["estimated_cost_usd"] == pytest.approx(0.002)


def test_tracker_groups_calls_by_request() -> None:
    tracker = UsageTracker()
    tracker.record(_usage("r1"))
    tracker.record(_usage("r1", role="job-classify-intent"))
    tracker.record(_usage("r2"))
    assert len(tracker.for_request("r1")) == 2
    assert len(tracker.for_request("r2")) == 1


def test_tracker_flags_calls_with_no_reported_usage() -> None:
    tracker = UsageTracker()
    tracker.record(_usage("r1", input_tokens=None, output_tokens=None, total_tokens=None))
    assert tracker.totals()["tokens_unavailable_for"] == 1
    assert not tracker.records[0].tokens_available


def test_request_ids_are_unique() -> None:
    assert len({new_request_id() for _ in range(200)}) == 200
