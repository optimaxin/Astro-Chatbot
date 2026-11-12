"""The human-eval aggregation script, on synthetic sheets (arithmetic only —
these are not evaluation results)."""

from __future__ import annotations

import csv

from scripts.build_human_eval import FOLLOW_UPS, RUBRIC_COLUMNS, SINGLE
from scripts.score_human_eval import load, report

BASE = {"item_id": "", "category": "career", "language": "english", "question": "q",
        "conversation_context": "", "chart_facts_available": "", "answer": "a", "citations": "",
        "model_hidden_from_reviewers": "m", "latency_ms": "1"}


def _sheet(path, reviewer, scores):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BASE) + RUBRIC_COLUMNS)
        writer.writeheader()
        for item, (acc, send) in scores.items():
            writer.writerow({**BASE, "item_id": item, "reviewer": reviewer, "chart_accuracy_1to5": acc,
                             "grounding_1to5": "", "helpfulness_1to5": 4, "language_naturalness_1to5": 5,
                             "safety_ok_YN": "Y", "overconfident_YN": "N", "would_send_YN": send, "notes": ""})


def test_batch_has_at_least_fifty_items_across_required_categories() -> None:
    items = len(SINGLE) + len(FOLLOW_UPS)
    assert items >= 50
    cats = {c for c, _ in SINGLE} | {c for c, *_ in FOLLOW_UPS}
    assert {"career", "marriage", "finance", "general", "hindi", "hinglish", "follow-up"} <= cats


def test_scores_are_aggregated_missing_values_never_imputed_and_disagreements_listed(tmp_path) -> None:
    a, b = tmp_path / "r1.csv", tmp_path / "r2.csv"
    _sheet(a, "asha", {"H001": (5, "Y"), "H002": (5, "Y")})
    _sheet(b, "ravi", {"H001": (5, "Y"), "H002": (2, "N")})
    text = report(load([str(a), str(b)]))
    assert "| chart_accuracy_1to5 | 4.25 |" in text          # (5+5+5+2)/4
    assert "| grounding_1to5 | — |" in text and "| 0 | 4 |" in text   # all missing, none imputed
    assert "| would_send_YN | 75% |" in text
    assert "H002 chart_accuracy_1to5: 5 vs 2" in text
