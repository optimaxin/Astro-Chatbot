"""Coverage, reply status and length-target metrics for
scripts/kundli_quality_bench.py (kept separate so the bench stays small).

Every turn gets ONE status, and quality is never credited to a turn that did
not produce a real answer:

    ok               the primary model answered
    fallback_model   the backup model answered (a real answer, reported apart)
    safe_fallback    the user got a canned "please ask again" / identity /
                     no-chart message — counted as a FAILED language-quality
                     result, never as a clean reply
    failed           generation failed

Length targets are measured against the depth the pipeline ASSIGNED to the
turn (`answer_depth`), and separately against the benchmark's own label.
Turns that are not chart questions (a language switch, "theek hai", a
greeting) have no length target and are listed as not applicable.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

BUDGET_WORDS = {"simple": (15, 30), "normal": (30, 60), "detailed": (60, 100), "complex": (80, 120)}
SLACK = 1.25                     # same soft band as kq_concise_metrics
BUBBLE_BUCKETS = ((0, 10), (11, 15), (16, 25), (26, 36), (37, 60), (61, 10_000))


def _canned_replies() -> set[str]:
    from app.evaluation.response_quality import IDENTITY_REPLY
    from app.nodes.validate import NO_CHART_MESSAGE, SAFE_FALLBACK, _SAFE_FALLBACK_BY_STYLE

    texts = {SAFE_FALLBACK, *_SAFE_FALLBACK_BY_STYLE.values(), *NO_CHART_MESSAGE.values(), *IDENTITY_REPLY.values()}
    return {" ".join(t.split()) for t in texts}


_CANNED: set[str] | None = None


_WHO_ARE_YOU = r"\b(?:who|what)\s+are\s+you\b|\baap\s+kaun\b|\btum\s+kaun\b|आप\s+कौन|तुम\s+कौन"


def row_status(row: dict[str, Any]) -> str:
    import re

    global _CANNED
    if _CANNED is None:
        _CANNED = _canned_replies()
    if row.get("failed"):
        return "failed"
    # The identity reply IS the answer to "who are you?"; anywhere else it replaced one.
    asks_identity = bool(re.search(_WHO_ARE_YOU, row.get("question") or "", re.I))
    if (" ".join((row.get("answer") or "").split()) in _CANNED and not asks_identity) \
            or row.get("language_gate") == "fallback":
        return "safe_fallback"
    return "fallback_model" if row.get("fallback_used") else "ok"


def length_applicable(row: dict[str, Any]) -> bool:
    from app.prompts.answer_depth import _ACKNOWLEDGEMENT

    return not (row.get("switch") or row.get("intent") in ("smalltalk", "greeting")
                or _ACKNOWLEDGEMENT.match(row.get("question") or ""))


def _within(words: int, depth: str | None) -> bool | None:
    if depth not in BUDGET_WORDS:
        return None
    low, high = BUDGET_WORDS[depth]
    return low / SLACK <= words <= high * SLACK


def summarize(rows: list[dict[str, Any]], avg) -> dict[str, Any]:
    statuses = Counter(row_status(r) for r in rows)
    n = max(1, len(rows))
    real = [r for r in rows if row_status(r) in ("ok", "fallback_model")]
    sized = [r for r in real if length_applicable(r)]
    words = {id(r): sum(len(m.split()) for m in r["messages"] if m.strip()) for r in rows}
    out: dict[str, Any] = {
        "status_ok_%": round(100.0 * statuses["ok"] / n, 1),
        "status_fallback_model_%": round(100.0 * statuses["fallback_model"] / n, 1),
        "status_safe_fallback_%": round(100.0 * statuses["safe_fallback"] / n, 1),
        "status_failed_%": round(100.0 * statuses["failed"] / n, 1),
        "length_applicable_turns": len(sized),
        "length_not_applicable_turns": len(real) - len(sized),
        "within_assigned_target_%": round(100.0 * sum(1 for r in sized if _within(words[id(r)], r.get("answer_depth")))
                                          / max(1, len(sized)), 1),
        "over_assigned_target_%": round(100.0 * sum(1 for r in sized if r.get("answer_depth") in BUDGET_WORDS and
                                                    words[id(r)] > BUDGET_WORDS[r["answer_depth"]][1] * SLACK)
                                        / max(1, len(sized)), 1),
        "under_assigned_target_%": round(100.0 * sum(1 for r in sized if r.get("answer_depth") in BUDGET_WORDS and
                                                     words[id(r)] < BUDGET_WORDS[r["answer_depth"]][0] / SLACK)
                                         / max(1, len(sized)), 1),
        "within_labelled_target_%": round(100.0 * sum(1 for r in sized if _within(words[id(r)], r.get("depth")))
                                          / max(1, len(sized)), 1),
    }
    for depth in BUDGET_WORDS:
        sub = [words[id(r)] for r in sized if r.get("answer_depth") == depth]
        out[f"assigned_{depth}_turns"] = len(sub)
        out[f"assigned_{depth}_words_avg"] = avg(sub)
    bubbles = [len(m.split()) for r in real for m in r["messages"] if m.strip()]
    total = max(1, len(bubbles))
    for low, high in BUBBLE_BUCKETS:
        label = f"{low}-{high}" if high < 10_000 else f">{low - 1}"
        out[f"bubbles_{label}w_%"] = round(100.0 * sum(1 for w in bubbles if low <= w <= high) / total, 1)
    return out


def judge_coverage(rows: list[dict[str, Any]], status_key: str) -> dict[str, Any]:
    from benchmarks.judge_runner import coverage

    eligible = [r for r in rows if row_status(r) in ("ok", "fallback_model")]
    excluded = Counter(row_status(r) for r in rows if row_status(r) not in ("ok", "fallback_model"))
    return coverage([r.get(status_key) for r in eligible], dict(excluded))


__all__ = ["row_status", "length_applicable", "summarize", "judge_coverage", "BUDGET_WORDS"]
