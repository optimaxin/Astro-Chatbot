"""Conciseness metrics for scripts/kundli_quality_bench.py (kept separate so
the bench script stays small).

Deterministic: words per reply by depth, bubbles in the 15–25-word target,
over-36-word bubbles, sentences per bubble, within-budget share, redundant
sentence pairs, Dasha dates repeated within a reply, depth classification
agreement with the case labels.

Judge (a separate, focused call so the earlier verdicts stay comparable):
redundant information, missing essential context, concision 1–5.
"""

from __future__ import annotations

import re
from typing import Any

BUDGET_WORDS = {"simple": (15, 30), "normal": (30, 60), "detailed": (60, 100), "complex": (80, 120)}
# A soft target: within 25% of the band counts as on budget.
_SLACK = 1.25

CONCISE_JUDGE_PROMPT = """You are reviewing one reply from a Vedic astrology chat assistant for CONCISENESS.
Return strict JSON: {"redundant": true|false, "missing_key_context": true|false, "concise": 1-5}
- redundant: true if the reply says the same thing twice in different words, explains the same placement
  more than once, or repeats the same Dasha dates.
- missing_key_context: true if, to be brief, it leaves out something essential to answer THIS question
  (e.g. a timing question with no period named, a career question with no career factor).
- concise: 5 = every sentence earns its place; 1 = padded, repetitive, long-winded."""


def judge_concise(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """(verdict, status) through the paced, cached, retrying judge runner."""
    from benchmarks.judge_runner import get_runner as runner

    user = f"User question: {row['question']}\n\nChatbot reply:\n{row['answer']}"
    return runner("kq-concise-judge", "kq-concise-1.0").run(
        [row["question"], row["answer"]],
        [{"role": "system", "content": CONCISE_JUDGE_PROMPT}, {"role": "user", "content": user}],
        valid=lambda v: isinstance(v.get("concise"), (int, float)))


def _sentences(text: str) -> int:
    return max(1, len([s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s.strip()]))


def row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    from app.evaluation.conciseness import redundant_pairs
    from app.prompts.answer_depth import classify_depth

    msgs = [m for m in row["messages"] if m.strip()]
    words = [len(m.split()) for m in msgs]
    total = sum(words)
    low, high = BUDGET_WORDS.get(row.get("depth") or "normal", (30, 60))
    ranges = re.findall(r"\(\D{0,12}\d{4}\s*(?:to|se|से|until|–|-)\s*\D{0,12}\d{4}", row["answer"] or "")
    return {
        "total_words": total,
        "within_budget": low / _SLACK <= total <= high * _SLACK,
        "over_budget": total > high * _SLACK,
        "bubbles_15_25": sum(1 for w in words if 15 <= w <= 25),
        "bubbles_le_25": sum(1 for w in words if w <= 25),
        "bubbles_over_36": sum(1 for w in words if w > 36),
        "sentences_per_bubble": [_sentences(m) for m in msgs],
        "redundant_pairs": redundant_pairs(row["answer"] or ""),
        "repeated_date_range": len(ranges) != len(set(re.sub(r"\D", "", r) for r in ranges)),
        "depth_predicted": classify_depth(row["question"], row.get("intent"), bool(row.get("switch"))),
    }


def summarize(rows: list[dict[str, Any]], avg, pct) -> dict[str, Any]:
    def m(r):
        return r["score"].get("concise") or row_metrics(r)

    bubbles = sum(len([x for x in r["messages"] if x.strip()]) for r in rows) or 1
    out: dict[str, Any] = {}
    for depth in ("simple", "normal", "detailed", "complex"):
        sub = [r for r in rows if r.get("depth") == depth]
        out[f"words_{depth}_avg"] = avg([m(r)["total_words"] for r in sub])
    judged = [r["judge_concise"] for r in rows if r.get("judge_concise")]
    out.update({
        "words_per_reply_avg": avg([m(r)["total_words"] for r in rows]),
        "within_length_budget_%": round(100.0 * sum(1 for r in rows if m(r)["within_budget"]) / max(1, len(rows)), 1),
        "over_length_budget_%": round(100.0 * sum(1 for r in rows if m(r)["over_budget"]) / max(1, len(rows)), 1),
        "bubbles_15to25w_%": round(100.0 * sum(m(r)["bubbles_15_25"] for r in rows) / bubbles, 1),
        "bubbles_le25w_%": round(100.0 * sum(m(r)["bubbles_le_25"] for r in rows) / bubbles, 1),
        "bubbles_over36w_%": round(100.0 * sum(m(r)["bubbles_over_36"] for r in rows) / bubbles, 1),
        "sentences_per_bubble_avg": avg([s for r in rows for s in m(r)["sentences_per_bubble"]]),
        "detailed_in_3to4_%": round(100.0 * sum(1 for r in rows if r.get("depth") == "detailed"
                                                and 3 <= len(r["messages"]) <= 4)
                                    / max(1, sum(1 for r in rows if r.get("depth") == "detailed")), 1),
        "replies_with_redundant_pair_%": round(100.0 * sum(1 for r in rows if m(r)["redundant_pairs"])
                                               / max(1, len(rows)), 1),
        "replies_repeating_dasha_dates_%": round(100.0 * sum(1 for r in rows if m(r)["repeated_date_range"])
                                                 / max(1, len(rows)), 1),
        "depth_classifier_agreement_%": round(100.0 * sum(1 for r in rows if m(r)["depth_predicted"] == r.get("depth"))
                                              / max(1, len(rows)), 1),
        "judge_redundant_%": round(100.0 * sum(1 for j in judged if j.get("redundant")) / len(judged), 1)
        if judged else None,
        "judge_missing_key_context_%": round(100.0 * sum(1 for j in judged if j.get("missing_key_context"))
                                             / len(judged), 1) if judged else None,
        "judge_concise": avg([j.get("concise") for j in judged if isinstance(j.get("concise"), (int, float))]),
    })
    return out
