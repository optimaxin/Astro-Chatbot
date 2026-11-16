"""Reply status, judge coverage and the native-speaker review sheet for
chatbot runs of the Hinglish Quality Benchmark.

Status of every chatbot turn (never credited as a language-quality success
unless it is a real answer):
    ok / fallback_model     a real answer (by the primary / the backup model)
    safe_fallback           a canned "please ask again", identity or no-chart reply
    failed                  generation failed

The review sheet is for INDEPENDENT Hinglish/Hindi speakers. Every row says
plainly that its scores are automated (rules + Kundli facts + an LLM judge)
and NOT a native-speaker review; only a reviewer's own verdict, imported
with `import-review`, is recorded as native review.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

AUTOMATED_NOTE = "automated (rules + Kundli facts + LLM judge) - NOT native-speaker reviewed"
REVIEW_COLUMNS = ["run", "id", "language", "question", "original_response", "corrected_response",
                  "changed_by_pipeline", "detected_issues", "benchmark_score", "benchmark_passed", "reply_status",
                  "automated_status", "reviewer_name", "reviewer_native_language",
                  "verdict (ok / wrong / unsure)", "reviewer_correction", "reviewer_comments"]


def _canned() -> set[str]:
    from app.evaluation.response_quality import IDENTITY_REPLY
    from app.nodes.validate import NO_CHART_MESSAGE, SAFE_FALLBACK, _SAFE_FALLBACK_BY_STYLE

    return {" ".join(t.split()) for t in (SAFE_FALLBACK, *_SAFE_FALLBACK_BY_STYLE.values(),
                                         *NO_CHART_MESSAGE.values(), *IDENTITY_REPLY.values())}


def reply_status(row: dict[str, Any], canned: set[str] | None = None) -> str:
    bot = row.get("chatbot") or {}
    if bot.get("failed"):
        return "failed"
    import re

    asks_identity = re.search(r"\b(?:who|what)\s+are\s+you\b|\baap\s+kaun\b|आप\s+कौन|तुम\s+कौन", row.get("question") or "",
                              re.I)
    if " ".join((row.get("text") or "").split()) in (canned if canned is not None else _canned()) and not asks_identity:
        return "safe_fallback"
    return "fallback_model" if bot.get("fallback_used") else "ok"


def chatbot_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    canned = _canned()
    statuses = Counter(reply_status(r, canned) for r in rows)
    real = [r for r in rows if reply_status(r, canned) in ("ok", "fallback_model")]
    judge = Counter(r.get("judge_status") or "not_requested" for r in real)
    n = max(1, len(rows))
    return {
        "turns": len(rows), **{f"status_{k}": statuses[k] for k in ("ok", "fallback_model", "safe_fallback", "failed")},
        # Strict: a turn passes only if the PRIMARY model gave a real answer that passed.
        "pass_rate_strict_%": round(100.0 * sum(1 for r in rows if reply_status(r, canned) == "ok"
                                                and r["score"]["passed"]) / n, 1),
        "pass_rate_real_answers_%": round(100.0 * sum(1 for r in real if r["score"]["passed"]) / max(1, len(real)), 1),
        "judge_coverage": {"eligible": len(real), "with_verdict": judge["judged"] + judge["cached"],
                           "coverage_%": round(100.0 * (judge["judged"] + judge["cached"]) / max(1, len(real)), 1),
                           "failed": {k: v for k, v in judge.items() if k.startswith("failed")},
                           "not_requested": judge["not_requested"],
                           "excluded_from_judge": {k: statuses[k] for k in ("safe_fallback", "failed") if statuses[k]}},
    }


def _issues(row: dict[str, Any]) -> str:
    parts = [f"{e['source']}:{e['category']}: {e['span']}" + (f" -> {e['message']}" if e.get("message") else "")
             for e in row["score"]["errors"]]
    layer = ((row.get("chatbot") or {}).get("language_quality") or {})
    if layer.get("categories"):
        parts.append("pipeline-detected: " + ", ".join(layer["categories"]))
    return " | ".join(parts)


def export_chatbot_review(runs: list[Path], out: Path, languages: tuple[str, ...] = ("hinglish", "hindi")) -> int:
    import json

    canned = _canned()
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(REVIEW_COLUMNS)
        for run in runs:
            data = json.loads(Path(run).read_text(encoding="utf-8"))
            label = data["meta"]["label"]
            for r in data["rows"]:
                if r["language"] not in languages:
                    continue
                original = (r.get("chatbot") or {}).get("draft") or ""
                w.writerow([label, r["id"], r["language"], r["question"], original or r["text"], r["text"],
                            "yes" if original and original.strip() != r["text"].strip() else "no",
                            _issues(r), r["score"]["overall"], r["score"]["passed"], reply_status(r, canned),
                            AUTOMATED_NOTE, "", "", "", "", ""])
                count += 1
    return count


__all__ = ["reply_status", "chatbot_coverage", "export_chatbot_review", "AUTOMATED_NOTE", "REVIEW_COLUMNS"]
