"""Summarise the two shadow roles from their logs (no model calls, no reply text).

    python -m scripts.shadow_report [--since-hours 24]

Quality checker (logs/quality_checker.jsonl): verdict mix, how often the independent checker and the
deterministic validators agree, latency. Humanizer (logs/humanizer_shadow.jsonl): how many replies it was
shown, how many rewrites passed the fact guard, and on the clean-reply sample whether a blind judge
preferred the rewrite, with the mean change per dimension.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
import time
from pathlib import Path

LOGS = Path(__file__).resolve().parents[1] / "logs"


def read(name: str, since: float) -> list[dict]:
    path = LOGS / name
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("ts", 0) >= since:
            rows.append(row)
    return rows


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def quality(rows: list[dict]) -> dict:
    status = collections.Counter(r["status"] for r in rows)
    judged = [r for r in rows if r["status"] != "UNAVAILABLE"]
    agree = collections.Counter(r.get("agreement") for r in rows)
    lat = sorted(r["latency_ms"] for r in rows if r.get("latency_ms"))
    return {
        "checked": len(rows), "verdicts": dict(status), "unavailable_rate": pct(status["UNAVAILABLE"], len(rows)),
        "flag_rate_of_judged": pct(status["REPAIR"] + status["REJECT"], len(judged)),
        "agreement_with_deterministic": dict(agree),
        "checker_only_rate": pct(agree["checker_only"], len(judged)),
        "both_flagged_rate": pct(agree["both_flagged"], len(judged)),
        "mean_quality_score": round(st.mean([r["scores"]["quality_score"] for r in judged if r.get("scores")] or [0]), 2),
        "unavailable_reasons": dict(collections.Counter(r.get("reason") for r in rows if r["status"] == "UNAVAILABLE")),
        "p50_ms": lat[len(lat) // 2] if lat else None, "p95_ms": lat[int(len(lat) * .95) - 1] if lat else None,
    }


def humanizer(rows: list[dict]) -> dict:
    issue = [r for r in rows if r.get("sample") != "clean"]
    clean = [r for r in rows if r.get("sample") == "clean"]
    compared = [r["comparison"] for r in clean if r.get("comparison", {}).get("status") == "JUDGED"]
    prefs = collections.Counter(c["preferred"] for c in compared)
    dims = ("grammar", "naturalness", "readability", "conversational")
    return {
        "replies_with_issues": {"shown": len(issue), "status": dict(collections.Counter(r["status"] for r in issue)),
                                "would_apply_rate": pct(sum(r["status"] == "shadow_would_apply" for r in issue), len(issue))},
        "clean_sample": {
            "shown": len(clean), "status": dict(collections.Counter(r["status"] for r in clean)),
            "fact_guard_rejected": pct(sum(r["status"] == "discarded" for r in clean), len(clean)),
            "judged": len(compared), "preferred": dict(prefs),
            "rewrite_preferred_rate": pct(prefs["rewrite"], len(compared)),
            "judge_says_facts_changed": sum(not c["facts_preserved_by_judge"] for c in compared),
            "mean_delta_rewrite_minus_original": {d: round(st.mean([c["delta"][d] for c in compared]), 2) for d in dims}
            if compared else {},
            "judge_unavailable": sum(r.get("comparison", {}).get("status") == "UNAVAILABLE" for r in clean),
        },
        "tokens": {"humanizer_in": sum(r.get("input_tokens", 0) for r in rows), "humanizer_out": sum(r.get("output_tokens", 0) for r in rows),
                   "judge_in": sum(r.get("judge_input_tokens", 0) for r in rows), "judge_out": sum(r.get("judge_output_tokens", 0) for r in rows)},
        "p50_ms": int(st.median([r["ms"] for r in rows])) if rows else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-hours", type=float, default=24 * 365)
    since = time.time() - ap.parse_args().since_hours * 3600
    report = {"quality_checker": quality(read("quality_checker.jsonl", since)),
              "humanizer": humanizer(read("humanizer_shadow.jsonl", since))}
    json.dump(report, sys.stdout, indent=1, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
