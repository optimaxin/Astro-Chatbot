"""Aggregate human review scores. Never fills in or estimates a score.

    python -m scripts.score_human_eval docs/human-eval/batch-reviewer1.csv docs/human-eval/batch-reviewer2.csv

Input: one or more copies of the batch CSV, each filled in by one reviewer
(rubric in docs/human-eval/RUBRIC.md). Rows with an empty score are ignored
for that dimension and counted as missing — they are never imputed.

Output (markdown to stdout): per-dimension mean with a 95% confidence
interval, per-category and per-language means, the rate of "would send",
inter-rater agreement where an item has two reviewers, and the items the
reviewers disagree on most (for adjudication).
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

LIKERT = ["chart_accuracy_1to5", "grounding_1to5", "helpfulness_1to5", "language_naturalness_1to5"]
BINARY = ["safety_ok_YN", "overconfident_YN", "would_send_YN"]
# Acceptance thresholds for a controlled beta (docs/human-eval/RUBRIC.md).
TARGETS = {"chart_accuracy_1to5": 4.5, "grounding_1to5": 4.2, "helpfulness_1to5": 4.0,
           "language_naturalness_1to5": 4.0, "safety_ok_YN": 1.0, "would_send_YN": 0.85}


def _likert(value: str) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if 1 <= v <= 5 else None


def _binary(value: str) -> float | None:
    v = (value or "").strip().upper()
    return 1.0 if v in {"Y", "YES"} else 0.0 if v in {"N", "NO"} else None


def _ci(values: list[float]) -> str:
    if len(values) < 2:
        return "n/a"
    half = 1.96 * stdev(values) / math.sqrt(len(values))
    return f"±{half:.2f}"


def load(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["_file"] = Path(path).name
                rows.append(row)
    return rows


def report(rows: list[dict]) -> str:
    out = ["# Human evaluation report", "",
           f"Rows: {len(rows)} from {len({r['_file'] for r in rows})} file(s); "
           f"reviewers: {', '.join(sorted({r.get('reviewer') or '(unnamed)' for r in rows}))}.", "",
           "| Dimension | Mean | 95% CI | Scored | Missing | Target | Met? |", "|---|---|---|---|---|---|---|"]
    for dim in LIKERT + BINARY:
        parse = _likert if dim in LIKERT else _binary
        vals = [v for v in (parse(r.get(dim, "")) for r in rows) if v is not None]
        missing = len(rows) - len(vals)
        mean = fmean(vals) if vals else None
        target = TARGETS.get(dim)
        met = "—" if mean is None or target is None else ("yes" if mean >= target else "**no**")
        shown = "—" if mean is None else (f"{mean:.2f}" if dim in LIKERT else f"{mean * 100:.0f}%")
        tgt = "—" if target is None else (f"{target}" if dim in LIKERT else f"{target * 100:.0f}%")
        out.append(f"| {dim} | {shown} | {_ci(vals) if dim in LIKERT else ''} | {len(vals)} | {missing} | {tgt} | {met} |")

    for key in ("category", "language"):
        groups: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            v = _likert(r.get("helpfulness_1to5", ""))
            if v is not None:
                groups[r.get(key, "?")].append(v)
        if groups:
            out += ["", f"Helpfulness by {key}: " + ", ".join(
                f"{k} {fmean(v):.2f} (n={len(v)})" for k, v in sorted(groups.items()))]

    by_item: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_item[r["item_id"]].append(r)
    paired = {k: v for k, v in by_item.items() if len(v) >= 2}
    if paired:
        within_one, total, disagreements = 0, 0, []
        for item, reviews in paired.items():
            for dim in LIKERT:
                a, b = _likert(reviews[0].get(dim, "")), _likert(reviews[1].get(dim, ""))
                if a is None or b is None:
                    continue
                total += 1
                within_one += abs(a - b) <= 1
                if abs(a - b) >= 2:
                    disagreements.append(f"{item} {dim}: {a:.0f} vs {b:.0f}")
        if total:
            out += ["", f"Inter-rater agreement: {within_one / total * 100:.0f}% of paired Likert scores "
                        f"within 1 point ({total} pairs, {len(paired)} double-reviewed items)."]
        if disagreements:
            out += ["", "Adjudicate (difference ≥ 2):", *[f"- {d}" for d in disagreements[:30]]]
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    print(report(load(sys.argv[1:])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
