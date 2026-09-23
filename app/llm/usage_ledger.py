"""Durable per-user token and cost accounting.

`UsageTracker` (in `usage_tracker.py`) is a per-process accumulator — it is
gone the moment the terminal closes. That is fine for a turn's metrics block
and useless for the question this module exists to answer: *how many tokens
has each user consumed, and what has each user cost us?*

Two stores, written from one place, because they answer different questions:

* **JSONL ledger** (`logs/usage.jsonl`) — one append-only line per LLM call.
  Complete, auditable, needs no infrastructure, and survives restarts. This is
  what `scripts/usage_report.py` aggregates and what a Postgres migration
  would later replay.
* **Redis counters** (when Upstash is configured) — running per-user totals
  under `usage:user:{user_id}`, incremented atomically. This is the O(1)
  lookup a production system needs for quotas, rate limits and billing, and
  it is shared across processes, which the JSONL file is not.

Both are best-effort. A ledger write failure is logged and swallowed: losing a
metering row must never cost a user their answer.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from app.config.settings import BASE_DIR, settings
from app.logging_setup import get_logger

log = get_logger("usage.ledger")

# Fields persisted per call. Explicit rather than `asdict()` so a change to
# LLMUsage cannot silently alter the on-disk schema.
LEDGER_FIELDS = (
    "request_id", "session_id", "user_id", "role", "provider", "model",
    "input_tokens", "output_tokens", "total_tokens", "cached_tokens",
    "reasoning_tokens", "input_cost_usd", "output_cost_usd", "total_cost_usd",
    "native_cost_usd", "latency_ms", "success", "fallback_used", "timestamp",
)

_write_lock = threading.Lock()


def ledger_path() -> Path:
    path = Path(settings.log_file).parent / "usage.jsonl"
    return path if path.is_absolute() else BASE_DIR / path


# ── writing ───────────────────────────────────────────────────────────
def _append_jsonl(record: dict[str, Any]) -> bool:
    path = ledger_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str, ensure_ascii=False)
        # Serialised so two threads cannot interleave a partial line.
        with _write_lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return True
    except OSError:
        log.warning("usage ledger write failed", extra={"path": str(path)})
        return False


def _increment_redis(record: dict[str, Any]) -> bool:
    """Bump this user's running totals. No-op when Redis is not configured."""
    from app.memory.redis_memory import memory

    client = memory._connect()  # noqa: SLF001 — same package, deliberate reuse
    if client is None:
        return False

    user_id = record.get("user_id") or "unknown"
    key = f"usage:user:{user_id}"
    # Cost is stored in micro-dollars as an integer: Redis HINCRBY is integer
    # only, and accumulating floats across millions of calls would drift.
    micro_dollars = int(round(float(record.get("total_cost_usd") or 0.0) * 1_000_000))

    try:
        client.hincrby(key, "calls", 1)
        client.hincrby(key, "input_tokens", int(record.get("input_tokens") or 0))
        client.hincrby(key, "output_tokens", int(record.get("output_tokens") or 0))
        client.hincrby(key, "total_tokens", int(record.get("total_tokens") or 0))
        client.hincrby(key, "cost_micro_usd", micro_dollars)
        if not record.get("success"):
            client.hincrby(key, "failed_calls", 1)
        # Tokens the provider never reported, so the totals above are known
        # to be an undercount by this many calls rather than silently wrong.
        if record.get("total_tokens") is None:
            client.hincrby(key, "calls_without_usage", 1)
        client.sadd("usage:users", user_id)
        return True
    except Exception:  # noqa: BLE001 — metering must never break a turn
        log.warning("redis usage counters failed", extra={"user_id": user_id})
        return False


def write(usage: Any) -> None:
    """Persist one LLM call. Called by `UsageTracker.record`."""
    record = {field: getattr(usage, field, None) for field in LEDGER_FIELDS}
    # Plus the error text, which the JSONL schema does not carry but the admin
    # panel needs in order to explain a failed call.
    record["error"] = getattr(usage, "error", None)
    _append_jsonl(record)
    _increment_redis(record)
    _insert_postgres(record)


def _insert_postgres(record: dict[str, Any]) -> bool:
    """Per-request row for the admin Token Meter. No-op when unconfigured."""
    from app.llm import usage_store

    return usage_store.write(record)


# ── reading ───────────────────────────────────────────────────────────
def read_ledger(path: Path | None = None) -> Iterator[dict[str, Any]]:
    """Stream the ledger, skipping any line a crash left malformed."""
    target = path or ledger_path()
    if not target.exists():
        return
    with open(target, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping malformed ledger line", extra={"line": line_number})


@dataclass
class UsageSummary:
    """Aggregated usage for one grouping key (a user, session or model)."""

    key: str
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    estimated_cost_usd: float = 0.0
    native_cost_usd: float = 0.0
    # Calls where the provider reported no usage, so the token totals above
    # are a known undercount rather than an exact figure.
    calls_without_usage: int = 0
    latency_ms_total: int = 0
    by_role: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    models: set[str] = field(default_factory=set)

    @property
    def avg_latency_ms(self) -> int:
        successful = self.calls - self.failed_calls
        return int(self.latency_ms_total / successful) if successful else 0

    @property
    def cost_per_call_usd(self) -> float:
        successful = self.calls - self.failed_calls
        return self.estimated_cost_usd / successful if successful else 0.0

    @property
    def tokens_exact(self) -> bool:
        return self.calls_without_usage == 0

    def add(self, record: dict[str, Any]) -> None:
        self.calls += 1
        self.by_role[record.get("role") or "unknown"] += 1
        if record.get("model"):
            self.models.add(str(record["model"]))

        if not record.get("success"):
            self.failed_calls += 1
            return

        if record.get("total_tokens") is None and record.get("input_tokens") is None:
            self.calls_without_usage += 1

        self.input_tokens += int(record.get("input_tokens") or 0)
        self.output_tokens += int(record.get("output_tokens") or 0)
        self.total_tokens += int(record.get("total_tokens") or 0)
        self.cached_tokens += int(record.get("cached_tokens") or 0)
        self.estimated_cost_usd += float(record.get("total_cost_usd") or 0.0)
        self.native_cost_usd += float(record.get("native_cost_usd") or 0.0)
        self.latency_ms_total += int(record.get("latency_ms") or 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "calls": self.calls,
            "failed_calls": self.failed_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "native_cost_usd": round(self.native_cost_usd, 8),
            "cost_per_call_usd": round(self.cost_per_call_usd, 8),
            "avg_latency_ms": self.avg_latency_ms,
            "calls_without_usage": self.calls_without_usage,
            "tokens_exact": self.tokens_exact,
            "models": sorted(self.models),
            "by_role": dict(self.by_role),
        }


def aggregate(
    records: Iterable[dict[str, Any]],
    group_by: str = "user_id",
    role: str | None = None,
) -> dict[str, UsageSummary]:
    """Group ledger rows by a field.

    `role` filters to a single LLM role — pass `job-main-turn` to see what the
    user-visible answers cost, excluding classification and evaluation
    overhead.
    """
    summaries: dict[str, UsageSummary] = {}
    for record in records:
        if role is not None and record.get("role") != role:
            continue
        key = str(record.get(group_by) or "unknown")
        summaries.setdefault(key, UsageSummary(key=key)).add(record)
    return summaries


def user_totals(user_id: str) -> dict[str, Any] | None:
    """Live per-user counters straight from Redis, without reading the ledger.

    This is the O(1) path a quota check or billing lookup would use.
    Returns None when Redis is not configured or the user is unknown.
    """
    from app.memory.redis_memory import memory

    client = memory._connect()  # noqa: SLF001
    if client is None:
        return None
    try:
        raw = client.hgetall(f"usage:user:{user_id}") or {}
    except Exception:  # noqa: BLE001
        log.warning("redis usage read failed", extra={"user_id": user_id})
        return None
    if not raw:
        return None

    counters = {key: int(value) for key, value in raw.items() if str(value).lstrip("-").isdigit()}
    return {
        "user_id": user_id,
        "calls": counters.get("calls", 0),
        "failed_calls": counters.get("failed_calls", 0),
        "input_tokens": counters.get("input_tokens", 0),
        "output_tokens": counters.get("output_tokens", 0),
        "total_tokens": counters.get("total_tokens", 0),
        "estimated_cost_usd": counters.get("cost_micro_usd", 0) / 1_000_000,
        "calls_without_usage": counters.get("calls_without_usage", 0),
        "source": "redis",
    }


def known_users() -> list[str]:
    from app.memory.redis_memory import memory

    client = memory._connect()  # noqa: SLF001
    if client is None:
        return []
    try:
        return sorted(client.smembers("usage:users") or [])
    except Exception:  # noqa: BLE001
        return []


__all__ = [
    "write",
    "read_ledger",
    "aggregate",
    "user_totals",
    "known_users",
    "ledger_path",
    "UsageSummary",
    "LEDGER_FIELDS",
]
