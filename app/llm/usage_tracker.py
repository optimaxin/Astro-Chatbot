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


import collections
import contextlib
import contextvars
import threading

# Who generated this traffic: "user" (real chat) unless an internal script
# says otherwise (evaluation, profiling, security, benchmark, debug). Lets
# the Token Meter report real usage without scripts and probes mixed in.
_traffic_source: contextvars.ContextVar[str | None] = contextvars.ContextVar("traffic_source", default=None)


def set_traffic_source(source: str) -> None:
    _traffic_source.set(source)


def current_traffic_source() -> str:
    return _traffic_source.get() or settings.traffic_source


# WHY a model was called (answer, repair, language_rewrite, grammar_correction,
# validation_fallback, classify, evaluate, ...) and which prompt version it was
# given. One role (job-main-turn) serves several purposes, so the role alone
# cannot tell a repair from the answer. Set by the caller with `llm_purpose`;
# context variables follow the call into the hedging threads (copy_context).
_purpose: contextvars.ContextVar[tuple[str, str | None] | None] = contextvars.ContextVar("llm_purpose", default=None)
_route: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_route", default=None)


@contextlib.contextmanager
def llm_purpose(purpose: str, prompt_version: str | None = None):
    token = _purpose.set((purpose, prompt_version))
    try:
        yield
    finally:
        _purpose.reset(token)


def current_purpose() -> str | None:
    value = _purpose.get()
    return value[0] if value else None


def current_prompt_version() -> str | None:
    value = _purpose.get()
    return value[1] if value else None


def set_route(route: str | None) -> contextvars.Token:
    return _route.set(route)


def reset_route(token: contextvars.Token) -> None:
    _route.reset(token)


def current_route() -> str | None:
    return _route.get()


def prompt_version_of(text: str) -> str:
    """A short, stable fingerprint of a STATIC prompt (never of user content)."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


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
    # Normalised failure class from `reliability.classify_error`, e.g.
    # "rate_limited:429". On a *successful fallback* call it carries why the
    # primary failed, so the Token Meter can show what triggered the fallback.
    error_code: str | None = None
    # 0 for the first try of a model, 1.. for gateway retries of it.
    attempt: int = 0
    streamed: bool = False

    # Where the token counts came from:
    #   "provider"    — the upstream's own usage block (authoritative)
    #   "estimated"   — the provider sent none; `estimated_*` hold a ~4
    #                   chars/token approximation and input/output_tokens
    #                   stay None, so an estimate can never be summed as fact
    #   "unavailable" — no usage and nothing to estimate from (failed call)
    token_source: str = "unavailable"
    estimated_input_tokens: int | None = None
    estimated_output_tokens: int | None = None
    traffic_source: str = field(default_factory=current_traffic_source)
    # Invocation reason, prompt version and routing decision (see llm_purpose).
    purpose: str | None = field(default_factory=current_purpose)
    prompt_version: str | None = field(default_factory=current_prompt_version)
    route: str | None = field(default_factory=current_route)
    # The model id that was REQUESTED; `model` is what the provider reported it
    # served (NVIDIA has returned e.g. "private/nvidia/..." for a requested id).
    requested_model: str | None = None
    # One id per invocation: the Postgres call row's idempotency key, so a
    # replayed ledger can never count the same call twice.
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex)

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


CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens(text: str) -> int:
    """Rough token count for text a provider did not meter. Never stored as
    a real count — see `LLMUsage.token_source`."""
    return max(1, -(-len(text) // CHARS_PER_TOKEN_ESTIMATE)) if text else 0


def message_text(messages: list[dict[str, Any]]) -> str:
    """Concatenated text of a chat request, for estimating its input size.
    Image parts are not counted (their token cost is provider-specific)."""
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(p.get("text", "")) for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
    return "\n".join(parts)


def token_accounting(
    tokens: dict[str, int | None], messages: list[dict[str, Any]], output_text: str
) -> dict[str, Any]:
    """`token_source` plus estimates, given what the provider reported."""
    if tokens.get("input_tokens") is not None or tokens.get("output_tokens") is not None:
        return {"token_source": "provider"}
    return {
        "token_source": "estimated",
        "estimated_input_tokens": estimate_tokens(message_text(messages)),
        "estimated_output_tokens": estimate_tokens(output_text),
    }


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

    Retention: the in-memory window is a ring buffer of the most recent
    `USAGE_TRACKER_MAX_RECORDS` calls (default 5000, ~4 MB). It exists for
    per-turn reads and the in-process summaries below; once full, the oldest
    record is dropped from MEMORY only — the JSONL ledger, the Redis counters and
    Postgres keep every call. `by_user/by_session/by_model/totals` therefore
    describe the retained window (`evicted` counts what fell out), not all time:
    use the ledger or the Token Meter for history.
    """

    def __init__(self, persist: bool = True, max_records: int | None = None) -> None:
        self._lock = threading.Lock()
        self._records: collections.deque[LLMUsage] = collections.deque(
            maxlen=max(1, max_records if max_records is not None else settings.usage_tracker_max_records))
        self.evicted = 0                    # records dropped from MEMORY (never from the ledger or Postgres)
        self.persist = persist

    @property
    def records(self) -> list[LLMUsage]:
        """A snapshot of the retained records, oldest first."""
        with self._lock:
            return list(self._records)

    def record(self, usage: LLMUsage) -> LLMUsage:
        if usage.purpose is None:
            # A caller that did not say why it called a model is visible as such.
            usage.purpose = f"unlabelled:{usage.role}"
        with self._lock:
            if len(self._records) == self._records.maxlen:
                self.evicted += 1           # the deque drops its oldest entry on append
            self._records.append(usage)
        if self.persist:
            # Best-effort: a metering failure must never cost a user their answer.
            from app.llm import usage_ledger

            usage_ledger.write(usage)
        return usage

    def for_request(self, request_id: str) -> list[LLMUsage]:
        """This request's retained calls, oldest first. A turn reads them within
        seconds of writing them, so the window only has to outlast one turn."""
        with self._lock:
            return [r for r in self._records if r.request_id == request_id]

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
    "estimate_tokens",
    "message_text",
    "token_accounting",
    "set_traffic_source",
    "current_traffic_source",
    "llm_purpose",
    "current_purpose",
    "prompt_version_of",
]
