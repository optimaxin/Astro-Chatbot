"""Token, latency and cost accounting for every LLM call.

Two rules this module exists to enforce:

1. Token counts are never guessed. They come from the provider's `usage`
   block. When the provider omits usage, the count is recorded as `None` and
   surfaces as "unavailable" — never as a plausible-looking invented number.
2. Prices live in configuration (`MODEL_PRICES`), not in business logic, so
   the same code can price a free model at $0 and a paid one correctly.

OpenRouter can also return its own authoritative `usage.cost`. When present
that is kept as `native_cost_usd` alongside the config-table estimate, so a
free model shows a real cost of $0 while still producing a comparable
hypothetical estimate for benchmarking (Phase 13 / Phase 20).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from app.config.settings import settings


@dataclass
class LLMUsage:
    """One LLM call's full accounting record."""

    request_id: str
    session_id: str
    user_id: str
    role: str
    provider: str
    model: str
    latency_ms: int
    success: bool
    timestamp: float

    # `None` means "the provider did not report it", never zero-by-assumption.
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None

    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    native_cost_usd: float | None = None

    fallback_used: bool = False
    error: str | None = None

    @property
    def tokens_available(self) -> bool:
        return self.input_tokens is not None or self.output_tokens is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_usage(raw_usage: Any) -> dict[str, int | None]:
    """Pull token counts out of an OpenAI-shaped `usage` object.

    Anything the provider did not send stays `None`.
    """
    if raw_usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cached_tokens": None,
            "reasoning_tokens": None,
        }

    data = raw_usage if isinstance(raw_usage, dict) else getattr(raw_usage, "model_dump", dict)()
    if not isinstance(data, dict):
        data = {}

    prompt_details = data.get("prompt_tokens_details") or {}
    completion_details = data.get("completion_tokens_details") or {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    if not isinstance(completion_details, dict):
        completion_details = {}

    input_tokens = _as_int(data.get("prompt_tokens", data.get("input_tokens")))
    output_tokens = _as_int(data.get("completion_tokens", data.get("output_tokens")))
    total_tokens = _as_int(data.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": _as_int(prompt_details.get("cached_tokens")),
        "reasoning_tokens": _as_int(completion_details.get("reasoning_tokens")),
    }


def extract_native_cost(raw_usage: Any) -> float | None:
    """OpenRouter's own charged cost, when the provider reports it."""
    if raw_usage is None:
        return None
    data = raw_usage if isinstance(raw_usage, dict) else getattr(raw_usage, "model_dump", dict)()
    if not isinstance(data, dict):
        return None
    cost = data.get("cost")
    if cost is None:
        return None
    try:
        return float(cost)
    except (TypeError, ValueError):
        return None


def calculate_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> tuple[float, float, float]:
    """(input_cost, output_cost, total_cost) in USD from the configured price table.

    Unknown token counts contribute 0 — an unpriceable call is reported as
    $0.000000 next to an explicit "tokens unavailable" note rather than
    having its cost fabricated.
    """
    input_price, output_price = settings.pricing.for_model(model)
    input_cost = (input_tokens or 0) / 1_000_000 * input_price
    output_cost = (output_tokens or 0) / 1_000_000 * output_price
    return input_cost, output_cost, input_cost + output_cost


class UsageTracker:
    """Accumulator for the current process, plus the write path to durable storage.

    In-memory records answer "what did this turn cost"; the ledger
    (`usage_ledger.py`) answers "what has this user cost us in total", which
    has to survive a restart. Every call goes to both from here, so no call
    site can record a turn without also metering it.
    """

    def __init__(self, persist: bool = True) -> None:
        self.records: list[LLMUsage] = []
        self.persist = persist

    def record(self, usage: LLMUsage) -> LLMUsage:
        self.records.append(usage)
        if self.persist:
            # Best-effort: a metering failure must never cost a user their answer.
            from app.llm import usage_ledger

            usage_ledger.write(usage)
        return usage

    def for_request(self, request_id: str) -> list[LLMUsage]:
        return [r for r in self.records if r.request_id == request_id]

    def by_user(self, role: str | None = None) -> dict[str, dict[str, Any]]:
        """Per-user totals for this process. Pass a role to isolate one job."""
        return self._grouped("user_id", role)

    def by_session(self, role: str | None = None) -> dict[str, dict[str, Any]]:
        return self._grouped("session_id", role)

    def by_model(self, role: str | None = None) -> dict[str, dict[str, Any]]:
        return self._grouped("model", role)

    def _grouped(self, attribute: str, role: str | None) -> dict[str, dict[str, Any]]:
        from app.llm.usage_ledger import aggregate

        rows = [r.to_dict() for r in self.records]
        summaries = aggregate(rows, group_by=attribute, role=role)
        return {key: summary.to_dict() for key, summary in summaries.items()}

    def totals(self) -> dict[str, Any]:
        priced = [r for r in self.records if r.success]
        known_tokens = [r for r in priced if r.total_tokens is not None]
        return {
            "calls": len(self.records),
            "successful_calls": len(priced),
            "failed_calls": len(self.records) - len(priced),
            "input_tokens": sum(r.input_tokens or 0 for r in priced),
            "output_tokens": sum(r.output_tokens or 0 for r in priced),
            "total_tokens": sum(r.total_tokens or 0 for r in priced),
            "tokens_unavailable_for": len(priced) - len(known_tokens),
            "estimated_cost_usd": sum(r.total_cost_usd for r in priced),
            "native_cost_usd": sum(r.native_cost_usd or 0.0 for r in priced),
            "avg_latency_ms": (
                int(sum(r.latency_ms for r in priced) / len(priced)) if priced else 0
            ),
        }


@dataclass
class Stopwatch:
    started_at: float = field(default_factory=time.perf_counter)

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)


usage_tracker = UsageTracker()

__all__ = [
    "LLMUsage",
    "UsageTracker",
    "Stopwatch",
    "usage_tracker",
    "new_request_id",
    "extract_usage",
    "extract_native_cost",
    "calculate_cost",
]
