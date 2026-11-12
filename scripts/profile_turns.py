"""Profile real chat turns end to end and report latency per stage.

    python -m scripts.profile_turns --label baseline --rounds 2
    python -m scripts.profile_turns --report logs/profiles/baseline.json logs/profiles/after.json

Runs the fixed question set in `tests/test_questions.py` through the real
pipeline (real LLM, Redis, Pinecone) and aggregates the per-request traces
from `app.observability.trace`. Comparing two labels is a like-for-like
before/after: same questions, same chart, same process settings apart from
the flags under test.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.llm.usage_tracker import set_traffic_source  # noqa: E402

# Internal traffic: kept out of the real-user view of the Token Meter.
set_traffic_source("profiling")
from app.graph.chatbot_graph import run_turn  # noqa: E402
from app.memory.redis_memory import memory  # noqa: E402
from app.observability import trace  # noqa: E402
from tests.test_questions import QUESTIONS  # noqa: E402


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 1)


def aggregate(traces: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [float(t["total_ms"]) for t in traces]
    spans: dict[str, list[float]] = {}
    for t in traces:
        for name, ms in t["by_span_ms"].items():
            spans.setdefault(name, []).append(float(ms))

    def stats(values: list[float]) -> dict[str, float]:
        return {"n": len(values), "avg": round(statistics.fmean(values), 1),
                "p50": pct(values, .5), "p95": pct(values, .95), "p99": pct(values, .99)}

    n = max(1, len(traces))
    return {
        "turns": len(traces),
        "failure_rate": round(sum(1 for t in traces if t.get("failed")) / n, 4),
        "fallback_rate": round(sum(1 for t in traces if t.get("fallback_used")) / n, 4),
        # main_error_code is why the primary did not answer ("slow_primary" = hedge won)
        "primary_error_codes": dict(Counter(t.get("main_error_code") for t in traces if t.get("main_error_code"))),
        "llm_calls": _ledger_stats({t.get("session") for t in traces}),
        "total_ms": stats(totals),
        "spans": {name: stats(v) for name, v in sorted(spans.items(),
                                                      key=lambda kv: -statistics.fmean(kv[1]))},
    }


def _ledger_stats(sessions: set[str | None]) -> dict[str, Any]:
    """Every LLM call these turns made, from the usage ledger: error codes
    (incl. NVIDIA 429 = rate_limited:429) and how many were fallbacks."""
    from app.llm import usage_ledger

    path = usage_ledger.ledger_path()
    calls = errors = fallbacks = 0
    codes: Counter = Counter()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("session_id") not in sessions:
                continue
            calls += 1
            fallbacks += bool(rec.get("fallback_used"))
            if rec.get("error_code"):
                errors += 1
                codes[rec["error_code"]] += 1
    return {"calls": calls, "failed_calls": errors, "fallback_calls": fallbacks,
            "rate_limited_429": sum(v for k, v in codes.items() if "429" in k or k.startswith("rate_limited")),
            "error_codes": dict(codes)}


def run(label: str, rounds: int) -> Path:
    traces: list[dict[str, Any]] = []
    for r in range(rounds):
        for q in QUESTIONS:
            session = f"profile-{uuid.uuid4().hex[:8]}"
            for role, content in q.context:
                memory.save_message(session, role, content)  # type: ignore[arg-type]
            with trace.request("pending", persist=False) as t:
                state = run_turn(q.question, session, settings.default_user_id)
            summary = t.summary()
            metrics = state.get("metrics") or {}
            summary["failed"] = bool(metrics.get("generation_failed"))
            summary["main_error_code"] = metrics.get("main_error_code")
            summary["session"] = session
            traces.append(summary)
            memory.clear_session(session)
            print(f"  round {r + 1} {q.id:<24} {summary['total_ms']:>6} ms", flush=True)
    out = settings.base_dir / "logs" / "profiles" / f"{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": label, "ts": time.time(), "summary": aggregate(traces),
                               "traces": traces}, indent=2), encoding="utf-8")
    return out


def report(paths: list[str]) -> None:
    data = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    names = sorted({n for d in data for n in d["summary"]["spans"]},
                   key=lambda n: -max(d["summary"]["spans"].get(n, {"avg": 0})["avg"] for d in data))
    header = f"{'stage':<38}" + "".join(f"{d['label'][:22]:>24}" for d in data)
    print(header + "\n" + "-" * len(header))
    for key in ("avg", "p50", "p95", "p99"):
        print(f"{'TOTAL ' + key:<38}" + "".join(f"{d['summary']['total_ms'][key]:>21.0f} ms" for d in data))
    print(f"{'failure rate':<38}" + "".join(f"{d['summary']['failure_rate'] * 100:>22.1f} %" for d in data))
    print(f"{'fallback rate':<38}" + "".join(f"{d['summary'].get('fallback_rate', 0) * 100:>22.1f} %" for d in data))
    for d in data:
        if "llm_calls" in d["summary"]:
            print(f"{d['label']}: primary errors {d['summary']['primary_error_codes']}; "
                  f"LLM calls {d['summary']['llm_calls']}")
    print()
    for n in names:
        cells = []
        for d in data:
            s = d["summary"]["spans"].get(n)
            cells.append(f"{s['avg']:>8.0f} / {s['p95']:>6.0f} ms (n={s['n']:>3})" if s else f"{'—':>24}")
        print(f"{n:<38}" + "".join(f"{c:>24}" for c in cells))
    print("\n(cells: avg / p95 per turn; n = turns in which the stage ran)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--report", nargs="*")
    args = parser.parse_args()
    if args.report:
        report(args.report)
        return 0
    path = run(args.label, args.rounds)
    report([str(path)])
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
