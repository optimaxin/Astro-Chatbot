"""Scoring: three signals -> five 1–5 dimensions -> one weighted score.

Signals
  rules  deterministic findings (rules.py)          — precise, narrow
  facts  chart contradictions / invented dates /    — the meaning check
         facts dropped versus the reference (facts.py)
  judge  LLM verdict per dimension (judge.py)       — broad, noisy

Per dimension
  start at 5; each rule finding subtracts its penalty (floor 1).
  If a rule found an error there: score = min(rule score, judge score).
  If no rule fired there:        score = mean(5, judge score) — so the judge
  alone can pull a dimension down to 3 at most: no example fails on an LLM's
  opinion only.
  Without a judge verdict the rule score stands (recorded as judge=None).

Overall (weights favour meaning and grammar, as the brief requires)
  meaning 0.30, grammar 0.25, vocabulary 0.15, naturalness 0.15, consistency 0.15.

Fact gate
  A changed or invented Kundli fact (or a machinery/unsafe claim) caps the
  overall score at 2.0 — natural phrasing never compensates for a wrong fact.

Pass
  overall >= 4.0, the fact gate not triggered, and no rule/fact finding
  (a clean reply). Judge-only errors lower the score but never fail alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmarks.hinglish_quality.facts import FactReport
from benchmarks.hinglish_quality.rules import Finding
from benchmarks.hinglish_quality.schema import CATEGORIES, DIMENSIONS

SCORING_VERSION = "hqb-scoring-1.0"
WEIGHTS = {"meaning": 0.30, "grammar": 0.25, "vocabulary": 0.15, "naturalness": 0.15, "consistency": 0.15}
PENALTY = {"grammar": 1.5, "vocabulary": 1.0, "naturalness": 1.0, "meaning": 1.5, "consistency": 1.5}
# Findings that floor their dimension at 1 on their own.
SEVERE = {"fact_change", "machinery", "unsafe_claim", "language_switch"}
GATE_CAP = 2.0
PASS_MARK = 4.0
_DROPPED_PENALTY, _DROPPED_MAX = 1.0, 3.0


@dataclass
class Score:
    dimensions: dict[str, float]
    overall: float
    gated: bool
    passed: bool
    rule_dimensions: dict[str, float]
    judge_dimensions: dict[str, float] | None
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"dimensions": self.dimensions, "overall": self.overall, "gated": self.gated, "passed": self.passed,
                "rule_dimensions": self.rule_dimensions, "judge_dimensions": self.judge_dimensions,
                "errors": self.errors}


def rule_dimensions(findings: list[Finding], facts: FactReport) -> dict[str, float]:
    dims = {d: 5.0 for d in DIMENSIONS}
    for f in findings:
        dim = CATEGORIES.get(f.category)
        if dim is None:
            continue
        dims[dim] = 1.0 if f.category in SEVERE else max(1.0, dims[dim] - PENALTY[dim])
    if facts.fact_change:
        dims["meaning"] = 1.0
    if facts.dropped:
        dims["meaning"] = max(1.0, dims["meaning"] - min(_DROPPED_MAX, _DROPPED_PENALTY * len(facts.dropped)))
    return dims


def score(findings: list[Finding], facts: FactReport, verdict: dict[str, Any] | None) -> Score:
    rules = rule_dimensions(findings, facts)
    judged = None
    if verdict:
        judged = {d: float(max(1, min(5, verdict.get(d) or 5))) for d in DIMENSIONS}
    dims: dict[str, float] = {}
    for d in DIMENSIONS:
        if judged is None:
            dims[d] = rules[d]
        elif rules[d] < 5:
            dims[d] = min(rules[d], judged[d])
        else:
            dims[d] = (5.0 + judged[d]) / 2
    overall = sum(WEIGHTS[d] * dims[d] for d in DIMENSIONS)
    gated = facts.fact_change or any(f.category in {"fact_change", "machinery", "unsafe_claim"} for f in findings)
    if gated:
        overall = min(overall, GATE_CAP)
    errors = [{"source": "rule", "category": f.category, "dimension": CATEGORIES[f.category], "span": f.span,
               "message": f.message} for f in findings]
    errors += [{"source": "facts", "category": "fact_change", "dimension": "meaning", "span": s, "message": "chart"}
               for s in facts.contradictions]
    errors += [{"source": "facts", "category": "fact_change", "dimension": "meaning", "span": y,
                "message": "invented year"} for y in facts.invented_years]
    errors += [{"source": "facts", "category": "fact_dropped", "dimension": "meaning", "span": t,
                "message": "chart fact of the reference missing"} for t in facts.dropped]
    if verdict:
        for e in verdict.get("errors") or []:
            if isinstance(e, dict) and e.get("category") in CATEGORIES:
                errors.append({"source": "judge", "category": e["category"], "dimension": CATEGORIES[e["category"]],
                               "span": str(e.get("span", ""))[:80], "message": str(e.get("fix", ""))[:80]})
    # A pass is a clean reply: good overall, no fact gate, and no error a
    # deterministic check is certain about (judge-only errors lower the score
    # but cannot fail a reply on their own).
    certain = any(e["source"] in ("rule", "facts") for e in errors)
    return Score({d: round(v, 2) for d, v in dims.items()}, round(overall, 3), gated,
                 overall >= PASS_MARK and not gated and not certain, rules, judged, errors)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate scored rows (each has row['score'] from Score.to_dict())."""
    from collections import Counter

    if not rows:
        return {"examples": 0}
    n = len(rows)
    dims = {d: round(sum(r["score"]["dimensions"][d] for r in rows) / n, 3) for d in DIMENSIONS}
    categories = Counter(e["category"] for r in rows for e in r["score"]["errors"])
    sources = Counter(e["source"] for r in rows for e in r["score"]["errors"])
    judged = sum(1 for r in rows if r["score"]["judge_dimensions"])
    return {
        "examples": n,
        "overall": round(sum(r["score"]["overall"] for r in rows) / n, 3),
        **{f"{d}": v for d, v in dims.items()},
        "pass_rate_%": round(100.0 * sum(1 for r in rows if r["score"]["passed"]) / n, 1),
        "fact_gate_%": round(100.0 * sum(1 for r in rows if r["score"]["gated"]) / n, 1),
        "errors_total": sum(categories.values()),
        "examples_with_errors": sum(1 for r in rows if r["score"]["errors"]),
        "errors_by_source": dict(sources),
        "top_error_categories": categories.most_common(10),
        "judged_%": round(100.0 * judged / n, 1),
        # The two signals apart: rules+facts on every row, the judge only where it answered.
        "rule_only": {d: round(sum(r["score"]["rule_dimensions"][d] for r in rows) / n, 3) for d in DIMENSIONS},
        "judge_only": ({d: round(sum(r["score"]["judge_dimensions"][d] for r in rows if r["score"]["judge_dimensions"])
                                 / judged, 3) for d in DIMENSIONS} if judged else None),
    }


__all__ = ["score", "summarize", "Score", "WEIGHTS", "SCORING_VERSION", "PASS_MARK", "GATE_CAP"]
