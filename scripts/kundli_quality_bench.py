"""Kundli chatbot quality benchmark — before vs after, on the REAL pipeline.

Two steps, so both versions are scored by the SAME scorer:

1. run   — send the 100 turns of benchmarks/kundli_quality_cases.py through
           `run_turn` of the code under test and save the raw replies:

    # after (this checkout)
    python -m scripts.kundli_quality_bench run --label after
    # before (a worktree of the previous commit; env read from this .env)
    python scripts/kundli_quality_bench.py run --root <worktree> --label before

2. score — deterministic metrics (with the CURRENT validators) plus an LLM
           judge, for one or more raw files, and a comparison table:

    python -m scripts.kundli_quality_bench score logs/evals/kq-before.json logs/evals/kq-after.json

Traffic is tagged "evaluation", so it stays out of the real-user Token Meter.
The chart is the one from the production screenshots (see
tests/test_screenshot_regressions.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import kq_concise_metrics  # noqa: E402 — sibling module in scripts/
import kq_coverage  # noqa: E402

HERE = Path(__file__).resolve().parents[1]          # this ai-core checkout
OUT_DIR = HERE / "logs" / "evals"


def _load_cases():
    spec = importlib.util.spec_from_file_location("kq_cases", HERE / "benchmarks" / "kundli_quality_cases.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kq_cases"] = module
    spec.loader.exec_module(module)
    return module


def screenshot_chart(KundliContext):
    def planet(pid, rashi, house):
        return {"id": pid, "rashi": rashi, "house": house, "degreeInSign": 10.0, "nakshatra": "Magha",
                "nakshatraPada": 1, "retrograde": False}
    ads = [("mercury", "2021-01-10", "2023-06-07"), ("ketu", "2023-06-07", "2024-06-03"),
           ("venus", "2024-06-03", "2027-04-03"), ("sun", "2027-04-03", "2028-02-09"),
           ("moon", "2028-02-09", "2029-07-09"), ("mars", "2029-07-09", "2030-07-06"),
           ("rahu", "2030-07-06", "2033-01-24"), ("jupiter", "2033-01-24", "2035-04-30"),
           ("saturn", "2035-04-30", "2038-01-10")]
    return KundliContext(
        user_id="kq-bench", source="benchmark", available=True,
        birth={"date": "2001-03-02", "time": "06:10"},
        ascendant={"rashi": "Pisces", "degreeInSign": 12.0},
        planets=[planet("sun", "Leo", 6), planet("moon", "Pisces", 1), planet("mars", "Aries", 2),
                 planet("mercury", "Leo", 6), planet("jupiter", "Leo", 6), planet("venus", "Virgo", 7),
                 planet("saturn", "Gemini", 4), planet("rahu", "Taurus", 3), planet("ketu", "Scorpio", 9)],
        current_mahadasha={"lord": "mercury", "startsAt": "2021-01-10", "endsAt": "2038-01-10",
                           "antardashas": [{"lord": l, "startsAt": s, "endsAt": e} for l, s, e in ads]})


# ── step 1: run ───────────────────────────────────────────────────────
def run(root: Path, label: str, workers: int) -> Path:
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")                       # never printed; a worktree has no .env of its own
    os.environ["AI_TRAFFIC_SOURCE"] = "evaluation"
    os.environ.setdefault("EVALUATION_ASYNC", "true")
    sys.path.insert(0, str(root))
    cases = _load_cases()

    from app.api.server import _to_response
    from app.graph.chatbot_graph import run_turn
    from app.kundli.kundli_service import KundliContext
    from app.llm.usage_tracker import set_traffic_source
    from app.observability import background

    try:                                              # the upgraded code resolves language in the graph
        import app.prompts.language_policy  # noqa: F401
        new_language_path = True
    except ImportError:
        new_language_path = False
    from app.config.settings import settings
    from app.prompts.response_style import resolve_style

    chart = screenshot_chart(KundliContext)
    run_id = uuid.uuid4().hex[:8]

    def one_conversation(conv) -> list[dict[str, Any]]:
        set_traffic_source("evaluation")
        session = f"kq-{label}-{run_id}-{conv.id}"
        earlier: list[str] = []
        rows = []
        for index, turn in enumerate(conv.turns):
            if new_language_path:
                style = "auto"
            else:                                     # exactly what the old /chat handler did
                style = resolve_style(None, turn.question, settings.api.default_response_style,
                                      history=earlier[-6:] or None).id
            started = time.perf_counter()
            state = run_turn(turn.question, session, "kq-bench-user", "vedic_general", kundli=chart,
                             chat_bubbles=True, response_style=style)
            wall_ms = int((time.perf_counter() - started) * 1000)
            resolved = state.get("response_style") or style
            body = _to_response(dict(state), session, "vedic_general", resolved)
            metrics = state.get("metrics") or {}
            rows.append({
                "conversation": conv.id, "group": conv.group, "turn": index, "question": turn.question,
                "depth": turn.expected_depth,
                "expected_language": turn.language, "topic": turn.topic, "life_fact": turn.life_fact,
                "beyond_chart": turn.beyond_chart, "switch": turn.switch,
                "previous_question": conv.turns[index - 1].question if index else None,
                "answer": body.answer, "messages": body.messages, "response_style": resolved,
                "intent": metrics.get("intent"), "model": metrics.get("model"),
                "fallback_used": metrics.get("fallback_used"), "failed": bool(metrics.get("generation_failed")),
                "turn_latency_ms": metrics.get("turn_latency_ms") or wall_ms, "wall_ms": wall_ms,
                "input_tokens": metrics.get("input_tokens"), "output_tokens": metrics.get("output_tokens"),
                "turn_total_tokens": metrics.get("turn_total_tokens"), "llm_calls": metrics.get("llm_calls"),
                "rag_chunks": metrics.get("rag_chunks"), "sources": [s.get("domain") for s in metrics.get("sources") or []],
                "validation_action": (metrics.get("validation") or {}).get("action"),
                "language_gate": (metrics.get("validation") or {}).get("language_gate"),
                "followup_allowed": (metrics.get("followup") or {}).get("allowed"),
                "answer_depth": metrics.get("answer_depth"),
                "prompt_version": metrics.get("prompt_version", "v1"),
                "trace_ms": metrics.get("trace_ms") or {},
            })
            earlier.append(turn.question)
        return rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [row for rows in pool.map(one_conversation, cases.CONVERSATIONS) for row in rows]
    background.drain(timeout=30) if hasattr(background, "drain") else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"kq-{label}.json"
    path.write_text(json.dumps({"label": label, "root": str(root), "run_id": run_id, "rows": results},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(results)} turns)")
    return path


# ── step 2: score ─────────────────────────────────────────────────────
_STYLE = {"english": "english", "hinglish": "whatsapp_hinglish", "hindi": "hindi"}

JUDGE_PROMPT = """You are grading one reply from a Vedic astrology chatbot that explains the user's own birth chart.
Grade ONLY what is asked. Return strict JSON:
{"relevance": 1-5, "completeness": 1-5, "naturalness": 1-5, "language_quality": 1-5,
 "answers_question": true|false, "unneeded_question": true|false, "followup_relevance": 1-5 or null}
- relevance: does it address the user's actual (latest) question?
- completeness: is it a complete, non-fragmentary answer proportional to the question?
- naturalness: does it read like a knowledgeable, respectful human, without filler or repetition?
- language_quality: spelling and grammar as a fluent speaker of the reply's language would write it. For Roman-script
  Hinglish: English words in correct English spelling (partner, commitment — not paartanar, committment), natural
  Roman Hindi (aapko, batate hain ki — not aapako, bataate hain kee), correct verb agreement and postpositions.
- answers_question: true if it answers the question asked (for questions a chart cannot answer, saying so clearly counts as answering).
- unneeded_question: true if it asks the user a question that was not necessary to answer.
- followup_relevance: if the reply ends with a follow-up question, how relevant and useful that question is
  as a next step for THIS user's question (5 = specific and natural, 1 = generic or intrusive); null if none."""


JUDGE_VERSION = "kq-judge-1.0"


def runner(name: str, version: str):
    """One paced, cached, retrying judge per prompt (benchmarks/judge_runner.py)."""
    from benchmarks.judge_runner import get_runner

    return get_runner(name, version)


def _judge(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    context = f"Previous user message: {row['previous_question']}\n" if row.get("previous_question") else ""
    user = f"{context}User question: {row['question']}\n\nChatbot reply:\n{row['answer']}"
    return runner("kq-judge", JUDGE_VERSION).run(
        [row.get("previous_question"), row["question"], row["answer"]],
        [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
        valid=lambda v: isinstance(v.get("relevance"), (int, float)))


def _judge_rows(rows: list[dict[str, Any]], key: str, fn, workers: int) -> None:
    """Judge every real answer that has no verdict yet; canned and failed
    replies are not judged (kq_coverage reports them as excluded)."""
    todo = [r for r in rows if not r.get(key) and kq_coverage.row_status(r) in ("ok", "fallback_model")]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for row, (verdict, status) in zip(todo, pool.map(fn, todo)):
            row[key], row[f"{key}_status"] = verdict, status
    for r in rows:
        if r.get(key) and not r.get(f"{key}_status"):
            r[f"{key}_status"] = "cached"


def score(paths: list[Path], judge: bool, workers: int) -> None:
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")
    os.environ["AI_TRAFFIC_SOURCE"] = "evaluation"
    sys.path.insert(0, str(HERE))
    from app.evaluation.chart_claims import validate_reply
    from app.evaluation.response_quality import (filler_phrases, mentions_machinery, repeated_sentences,
                                                 trailing_questions)
    from app.kundli.chart_facts import build_chart_facts
    from app.kundli.kundli_service import KundliContext
    from app.llm.usage_tracker import set_traffic_source
    from app.prompts.language import reply_matches, script_shares
    from app.prompts.followup import is_good_followup
    from app.prompts.hinglish import hinglish_issues
    from app.prompts.segments import _CLOSING_RANGE_START, _OPEN_RANGE_END, is_complete
    from app.prompts.terminology import hybrid_tokens

    set_traffic_source("evaluation")
    facts = build_chart_facts(screenshot_chart(KundliContext))
    reports = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = data["rows"]
        if judge:
            # A *.scored.json keeps its earlier verdicts; only new rows are judged.
            _judge_rows(rows, "judge", _judge, workers)
            _judge_rows(rows, "judge_concise", kq_concise_metrics.judge_concise, workers)
            from benchmarks.judge_runner import save_all

            print("judge calls:", json.dumps(save_all()))
        for row in rows:
            text = row["answer"] or ""
            result = validate_reply(text, facts)
            tail = trailing_questions(text)
            clarifying_ok = row["intent"] in ("unknown", "smalltalk") or row["switch"]
            real = kq_coverage.row_status(row) in ("ok", "fallback_model")
            row["score"] = {
                # A canned or failed reply is never a language-quality success.
                "language_ok": real and reply_matches(_STYLE[row["expected_language"]], text),
                "followup": bool(tail) and not clarifying_ok and not row.get("followup_allowed"),
                "followup_asked": bool(row["messages"]) and row["messages"][-1].rstrip().endswith("?"),
                "followup_good": bool(row["messages"]) and is_good_followup(row["messages"][-1]),
                "words_per_bubble": [len(m.split()) for m in row["messages"]],
                "oversized_bubbles": sum(1 for m in row["messages"] if len(m.split()) > 60),
                "contradictions": len(result.contradictions),
                "unsupported": len(result.unsupported),
                "fragmented": len(row["messages"]) > 1,
                "messages": len(row["messages"]),
                "machinery": mentions_machinery(text),
                "filler": bool(filler_phrases(text)),
                "repetition": bool(repeated_sentences(text)),
                "hybrid_terms": bool(hybrid_tokens(text)),
                "devanagari_in_latin": row["expected_language"] != "hindi" and script_shares(text)[0] > 0.03,
            }
            msgs = [m for m in row["messages"] if m.strip()]
            issues = hinglish_issues(text) if row["expected_language"] != "hindi" else {}
            row["score"].update({
                "incomplete_segment": any(not is_complete(m) for m in msgs),
                "date_split": any(_OPEN_RANGE_END.search(a.strip().rstrip(".")) or
                                  (_CLOSING_RANGE_START.match(b) and not is_complete(a))
                                  for a, b in zip(msgs, msgs[1:])),
                "duplicate_segment": len({" ".join(m.lower().split()) for m in msgs}) < len(msgs),
                "vocab_misspelling": bool(issues.get("misspelled")),
                "suspect_transliteration": bool(issues.get("suspect_transliteration")),
                "hinglish_issue": row["expected_language"] == "hinglish" and (not real or bool(
                    {k: v for k, v in issues.items() if k != "repeated_phrase"})),
                "concise": kq_concise_metrics.row_metrics(row),
            })
        reports.append((data["label"], rows))
        Path(path).with_suffix(".scored.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    _print_table(reports)
    if judge:
        calibrate_judge(workers)


def calibrate_judge(workers: int) -> dict[str, Any]:
    """Can the judge tell awkward Hinglish from natural Hinglish? Scores both
    halves of every reference pair (benchmarks/hinglish_reference.py)."""
    spec = importlib.util.spec_from_file_location("hinglish_ref", HERE / "benchmarks" / "hinglish_reference.py")
    ref = importlib.util.module_from_spec(spec)
    sys.modules["hinglish_ref"] = ref
    spec.loader.exec_module(ref)
    rows = []
    for pair in ref.PAIRS:
        for kind, text in (("awkward", pair.awkward), ("natural", pair.natural)):
            rows.append({"kind": kind, "pair": pair, "question": "Mere bare mein kundli kya kehti hai?",
                         "previous_question": None, "answer": text})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = [v for v, _ in pool.map(_judge, rows)]
    scores: dict[int, dict[str, float]] = {}
    for i, (row, v) in enumerate(zip(rows, verdicts)):
        scores.setdefault(i // 2, {})[row["kind"]] = (v or {}).get("language_quality")
    valid = [x for x in scores.values() if isinstance(x.get("awkward"), (int, float))
             and isinstance(x.get("natural"), (int, float))]
    agree = sum(1 for x in valid if x["natural"] > x["awkward"])
    ties = sum(1 for x in valid if x["natural"] == x["awkward"])
    result = {"pairs": len(ref.PAIRS), "scored": len(valid), "natural_scored_higher": agree, "ties": ties,
              "natural_avg": _avg([x["natural"] for x in valid]), "awkward_avg": _avg([x["awkward"] for x in valid]),
              "review_status": ref.REVIEW_STATUS}
    print("\njudge calibration on Hinglish reference pairs:", json.dumps(result))
    (OUT_DIR / "kq-judge-calibration.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def _rate(rows, key) -> float:
    return 100.0 * sum(1 for r in rows if r["score"][key]) / max(1, len(rows))


def _avg(values) -> float | None:
    values = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(values), 2) if values else None


def _pct(values, q) -> float | None:
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def _share(rows, depth: str, ok) -> float | None:
    sub = [r for r in rows if r.get("depth") == depth]
    return round(100.0 * sum(1 for r in sub if ok(r["score"]["messages"])) / len(sub), 1) if sub else None


def summarize(rows) -> dict[str, Any]:
    judged = [r["judge"] for r in rows if r.get("judge")]
    return {
        "turns": len(rows),
        "failed": sum(1 for r in rows if r["failed"]),
        "language_consistency_%": round(_rate(rows, "language_ok"), 1),
        "unneeded_followup_%": round(_rate(rows, "followup"), 1),
        "replies_with_chart_contradiction_%": round(100.0 * sum(1 for r in rows if r["score"]["contradictions"])
                                                    / max(1, len(rows)), 1),
        "chart_contradictions_total": sum(r["score"]["contradictions"] for r in rows),
        "unsupported_chart_claims_total": sum(r["score"]["unsupported"] for r in rows),
        "multi_bubble_replies_%": round(_rate(rows, "fragmented"), 1),
        "words_per_bubble_avg": _avg([w for r in rows for w in r["score"].get("words_per_bubble") or
                                      [len(m.split()) for m in r["messages"]]]),
        "words_per_bubble_p90": _pct([w for r in rows for w in r["score"].get("words_per_bubble") or
                                      [len(m.split()) for m in r["messages"]]], 0.9),
        "oversized_bubbles_%_(>60w)": round(100.0 * sum(r["score"].get("oversized_bubbles", 0) for r in rows)
                                            / max(1, sum(len(r["messages"]) for r in rows)), 1),
        "followup_asked_%": round(_rate(rows, "followup_asked"), 1),
        "followup_allowed_turns_%": round(100.0 * sum(1 for r in rows if r.get("followup_allowed"))
                                          / max(1, len(rows)), 1),
        "followup_specific_offer_%_of_asked": round(100.0 * sum(1 for r in rows if r["score"].get("followup_good"))
                                                    / max(1, sum(1 for r in rows if r["score"].get("followup_asked"))),
                                                    1),
        "judge_followup_relevance": _avg([r["judge"].get("followup_relevance") for r in rows if r.get("judge")
                                          and isinstance(r["judge"].get("followup_relevance"), (int, float))]),
        "avg_bubbles_per_reply": _avg([r["score"]["messages"] for r in rows]),
        "bubbles_simple_avg": _avg([r["score"]["messages"] for r in rows if r.get("depth") == "simple"]),
        "bubbles_normal_avg": _avg([r["score"]["messages"] for r in rows if r.get("depth") == "normal"]),
        "bubbles_detailed_avg": _avg([r["score"]["messages"] for r in rows if r.get("depth") == "detailed"]),
        "bubbles_complex_avg": _avg([r["score"]["messages"] for r in rows if r.get("depth") == "complex"]),
        "simple_in_1_bubble_%": _share(rows, "simple", lambda n: n == 1),
        "detailed_in_2to4_%": _share(rows, "detailed", lambda n: 2 <= n <= 4),
        "complex_in_3to4_%": _share(rows, "complex", lambda n: 3 <= n <= 4),
        "max_bubbles": max((r["score"]["messages"] for r in rows), default=0),
        "incomplete_segment_%": round(_rate(rows, "incomplete_segment"), 1),
        "date_range_split_%": round(_rate(rows, "date_split"), 1),
        "duplicate_segment_%": round(_rate(rows, "duplicate_segment"), 1),
        "vocab_misspelling_%": round(_rate(rows, "vocab_misspelling"), 1),
        "suspect_transliteration_%": round(_rate(rows, "suspect_transliteration"), 1),
        "hinglish_replies_with_issue_%": round(100.0 * sum(1 for r in rows if r["score"]["hinglish_issue"])
                                               / max(1, sum(1 for r in rows if r["expected_language"] == "hinglish")),
                                               1),
        "machinery_mentions_%": round(_rate(rows, "machinery"), 1),
        "filler_%": round(_rate(rows, "filler"), 1),
        "repetition_%": round(_rate(rows, "repetition"), 1),
        "hybrid_terms_%": round(_rate(rows, "hybrid_terms"), 1),
        "devanagari_in_latin_reply_%": round(_rate(rows, "devanagari_in_latin"), 1),
        "judge_relevance": _avg([j.get("relevance") for j in judged]),
        "judge_completeness": _avg([j.get("completeness") for j in judged]),
        "judge_naturalness": _avg([j.get("naturalness") for j in judged]),
        "judge_language_quality": _avg([j.get("language_quality") for j in judged]),
        "judge_hinglish_language_quality": _avg([r["judge"].get("language_quality") for r in rows
                                                 if r.get("judge") and r["expected_language"] == "hinglish"]),
        "judge_answers_question_%": round(100.0 * sum(1 for j in judged if j.get("answers_question"))
                                          / max(1, len(judged)), 1) if judged else None,
        "judge_unneeded_question_%": round(100.0 * sum(1 for j in judged if j.get("unneeded_question"))
                                           / max(1, len(judged)), 1) if judged else None,
        "judged_turns": len(judged),
        "latency_ms_avg": _avg([r["turn_latency_ms"] for r in rows]),
        "latency_ms_p50": _pct([r["turn_latency_ms"] for r in rows], 0.5),
        "latency_ms_p95": _pct([r["turn_latency_ms"] for r in rows], 0.95),
        "tokens_per_turn_avg": _avg([r["turn_total_tokens"] for r in rows]),
        "llm_calls_per_turn_avg": _avg([r["llm_calls"] for r in rows]),
        "fallback_used_%": round(100.0 * sum(1 for r in rows if r["fallback_used"]) / max(1, len(rows)), 1),
        **kq_concise_metrics.summarize(rows, _avg, _pct),
        **kq_coverage.summarize(rows, _avg),
        "language_quality_pass_strict_%": round(100.0 * sum(
            1 for r in rows if kq_coverage.row_status(r) == "ok" and r["score"]["language_ok"]
            and not r["score"]["hinglish_issue"] and not r["score"]["contradictions"]) / max(1, len(rows)), 1),
        "judge_coverage": kq_coverage.judge_coverage(rows, "judge_status"),
        "judge_concise_coverage": kq_coverage.judge_coverage(rows, "judge_concise_status"),
    }


def _print_table(reports) -> None:
    summaries = [(label, summarize(rows)) for label, rows in reports]
    keys = list(summaries[0][1].keys())
    print("\n| metric | " + " | ".join(label for label, _ in summaries) + " |")
    print("|---|" + "---|" * len(summaries))
    for key in keys:
        print(f"| {key} | " + " | ".join(str(s[key]) for _, s in summaries) + " |")
    for label, rows in reports:
        spans: dict[str, list[float]] = {}
        for r in rows:
            for name, ms in (r.get("trace_ms") or {}).items():
                spans.setdefault(name, []).append(ms)
        top = sorted(((statistics.mean(v), k) for k, v in spans.items() if k.startswith("node.")), reverse=True)
        if top:
            print(f"\n[{label}] avg ms per node: " + ", ".join(f"{k[5:]} {int(m)}" for m, k in top[:7]))
    for label, rows in reports:
        print(f"\n[{label}] by group:")
        for group in ("english", "hinglish", "hindi", "multiturn", "extended"):
            sub = [r for r in rows if r["group"] == group]
            if not sub:
                continue
            s = summarize(sub)
            print(f"  {group:9s} lang {s['language_consistency_%']:5.1f}%  followup {s['unneeded_followup_%']:5.1f}%  "
                  f"contradiction {s['replies_with_chart_contradiction_%']:5.1f}%  bubbles {s['avg_bubbles_per_reply']}  "
                  f"incomplete {s['incomplete_segment_%']:4.1f}%  hinglish-issue {s['hinglish_replies_with_issue_%']:4.1f}%")
    out = OUT_DIR / "kq-summary.json"
    out.write_text(json.dumps({label: s for label, s in summaries}, indent=1), encoding="utf-8")
    print(f"\nsummary -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", default=str(HERE))
    r.add_argument("--label", required=True)
    r.add_argument("--workers", type=int, default=2)
    s = sub.add_parser("score")
    s.add_argument("paths", nargs="+")
    s.add_argument("--no-judge", action="store_true")
    s.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.cmd == "run":
        run(Path(args.root).resolve(), args.label, args.workers)
    else:
        score([Path(p) for p in args.paths], judge=not args.no_judge, workers=args.workers)


if __name__ == "__main__":
    main()
