"""Hinglish Quality Benchmark runner (benchmarks/hinglish_quality).

Modes (each writes logs/hinglish-bench/<label>.json with full metadata):

  dataset     score the 250 examples themselves: gold replies, negatives and
              their corrections. Validates the EVALUATOR (does it pass gold
              text, catch negatives, rank a correction above its original?).
  chatbot     ask the LIVE pipeline every example's question (in the
              example's language, against the benchmark chart) and score the
              replies. This is the chatbot's language-quality score.
  correction  run the production post-generation language processing on
              every example's text: negatives should come out fixed, gold
              text should come out unchanged (no unnecessary rewriting).

Then:
  compare BASE NEW [--fail-on-regression]   regression gate (exit 1 on drop)
  report RUN [RUN ...] --out FILE.html      dashboard, filterable by language
  export-review / import-review             native-speaker sign-off sheets

Usage (from ai-core/):
  python scripts/hinglish_quality_bench.py dataset --label ds-1 [--judge]
  python scripts/hinglish_quality_bench.py chatbot --label cb-before --judge --workers 2
  python scripts/hinglish_quality_bench.py report logs/hinglish-bench/cb-before.json --out docs/hqb-report.html
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
OUT_DIR = HERE / "logs" / "hinglish-bench"
REVIEW_DIR = HERE / "docs" / "human-eval" / "hinglish"
_STYLE = {"english": "english", "hinglish": "whatsapp_hinglish", "hindi": "hindi"}
# Regression gate: a drop larger than these fails `compare --fail-on-regression`.
THRESHOLDS = {"overall": 0.10, "dimension": 0.20, "pass_rate_%": 5.0, "fact_gate_%": 2.0}


def _env() -> None:
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")
    os.environ["AI_TRAFFIC_SOURCE"] = "evaluation"


def _meta(mode: str, label: str, use_judge: bool) -> dict[str, Any]:
    from benchmarks.hinglish_quality import BENCHMARK_VERSION, dataset_hash, load_examples
    from benchmarks.hinglish_quality.judge import JUDGE_VERSION, judge_model
    from benchmarks.hinglish_quality.scoring import SCORING_VERSION, WEIGHTS

    overrides = {k: os.environ[k] for k in ("HINGLISH_GRAMMAR", "ENABLE_EVALUATION", "LLM_HEDGE_AFTER_SECONDS",
                                             "HQB_TURN_DELAY", "PREMIUM_RAG_URL") if k in os.environ}
    meta: dict[str, Any] = {
        "env_overrides": overrides,
        "benchmark_version": BENCHMARK_VERSION, "scoring_version": SCORING_VERSION, "weights": WEIGHTS,
        "dataset_hash": dataset_hash(load_examples()), "mode": mode, "label": label,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "judge": {"used": use_judge, "version": JUDGE_VERSION, "model": judge_model() if use_judge else None},
    }
    try:
        meta["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True,
                                            text=True, timeout=10).stdout.strip()
        meta["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], cwd=HERE, capture_output=True,
                                                text=True, timeout=20).stdout.strip())
    except Exception:  # noqa: BLE001
        meta["git_commit"] = None
    try:
        from app.evaluation.response_quality import VALIDATOR_VERSION
        from app.llm.llm_factory import describe_roles
        from app.prompts.builder import PROMPT_VERSION

        meta["chatbot"] = {"prompt_version": PROMPT_VERSION, "validator_version": VALIDATOR_VERSION,
                           "models": {role: {k: v for k, v in cfg.items() if k in ("provider", "model", "fallback")}
                                      for role, cfg in describe_roles().items()}}
    except Exception as exc:  # noqa: BLE001
        meta["chatbot"] = {"error": type(exc).__name__}
    return meta


def evaluate(example, text: str, use_judge: bool, reference: str | None, source: str | None = None,
             variant: str = "text") -> dict[str, Any]:
    from benchmarks.hinglish_quality.facts import check_facts
    from benchmarks.hinglish_quality.judge import judge_with_status
    from benchmarks.hinglish_quality.rules import check
    from benchmarks.hinglish_quality.scoring import score

    findings = check(text, example.language)
    facts = check_facts(text, reference, source)
    verdict, status = (judge_with_status(example.language, example.question, text, reference) if use_judge
                       else (None, "not_requested"))
    result = score(findings, facts, verdict)
    return {"id": example.id, "variant": variant, "language": example.language, "domain": example.domain,
            "topic": example.topic, "kind": example.kind, "split": example.split, "review": example.review,
            "question": example.question, "text": text, "reference": reference,
            "expected_errors": [e.category for e in example.errors] if variant == "text" else [],
            "facts": facts.to_dict(), "score": result.to_dict(), "judge_status": status}


def _run_pool(fn, items, workers: int) -> list:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(fn, items))


# ── modes ─────────────────────────────────────────────────────────────
def mode_dataset(args) -> list[dict[str, Any]]:
    from benchmarks.hinglish_quality import load_examples

    jobs = []
    for e in load_examples():
        jobs.append((e, e.text, e.reference if e.kind == "negative" else e.text, "text"))
        if e.kind == "negative":
            jobs.append((e, e.corrected, e.corrected, "corrected"))
    return _run_pool(lambda j: evaluate(j[0], j[1], args.judge, j[2], variant=j[3]), jobs, args.workers)


def mode_chatbot(args) -> list[dict[str, Any]]:
    import importlib.util
    import uuid

    sys.path.insert(0, str(HERE / "scripts"))
    spec = importlib.util.spec_from_file_location("_kqb", HERE / "scripts" / "kundli_quality_bench.py")
    kqb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kqb)
    from app.graph.chatbot_graph import run_turn
    from app.kundli.kundli_service import KundliContext
    from app.llm.usage_tracker import set_traffic_source
    from benchmarks.hinglish_quality import load_examples

    chart = kqb.screenshot_chart(KundliContext)
    examples = [e for e in load_examples() if not args.languages or e.language in args.languages]
    if args.limit:
        examples = examples[:args.limit]
    run_id = uuid.uuid4().hex[:6]

    import time

    import app.graph.chatbot_graph as graph_module

    # Benchmark-only instrumentation: keep the model's draft (what validation
    # received) so the review sheet can show original vs delivered text.
    drafts: dict[str, str] = {}
    validate_node = graph_module.validate_reply_node

    def recording_validate(state):
        drafts[state["session_id"]] = state.get("response") or ""
        return validate_node(state)

    graph_module.validate_reply_node = recording_validate
    graph_module._compiled = None
    delay = float(os.environ.get("HQB_TURN_DELAY", "2"))

    def one(e, attempt: int = 0):
        set_traffic_source("evaluation")
        time.sleep(delay)            # the provider rate-limits; a 429 storm turns replies into fallbacks
        session = f"hqb-{args.label}-{run_id}-{e.id}-{attempt}"
        state = run_turn(e.question, session, "hqb-bench-user",
                         "vedic_general", kundli=chart, chat_bubbles=True, response_style=_STYLE[e.language])
        reply = state.get("response") or ""
        metrics = state.get("metrics") or {}
        validation = metrics.get("validation") or {}
        row = evaluate(e, reply, args.judge, None, variant="chatbot")
        row["chatbot"] = {"failed": bool(metrics.get("generation_failed")), "model": metrics.get("model"),
                          "fallback_used": metrics.get("fallback_used"),
                          "latency_ms": metrics.get("turn_latency_ms"),
                          "validation": validation.get("action"), "draft": drafts.pop(session, None),
                          "language_quality": {k: v for k, v in (validation.get("language_quality") or {}).items()
                                               if k in ("categories", "issues_before", "issues_after",
                                                        "llm_accepted", "llm_rejected", "dropped_sentences")}}
        return row

    rows = _run_pool(one, examples, args.workers)
    # One sequential retry of failed turns after the provider's circuit has
    # closed again; what still fails is reported and excluded from quality.
    failed = [i for i, r in enumerate(rows) if r["chatbot"]["failed"]]
    if failed:
        time.sleep(40)
        by_id = {e.id: e for e in examples}
        for i in failed:
            rows[i] = one(by_id[rows[i]["id"]], attempt=1)
    return rows


_INTENT = {"career": "career", "marriage": "marriage", "relationship": "relationship", "health": "health",
           "chat": "smalltalk"}


def production_postprocess(text: str, language: str, question: str, intent: str = "general_astrology") -> str:
    """The production post-generation language processing, applied to a
    finished reply as the pipeline applies it: the deterministic quality pass,
    then the language-quality layer (HINGLISH_GRAMMAR decides how much of it
    runs; 'correct' includes its one targeted model call)."""
    from app.evaluation.response_quality import apply_safe_fixes

    out, _ = apply_safe_fixes(text, language, intent, question)
    try:
        from app.nodes.language_quality import apply_language_quality
    except ImportError:          # before the layer existed
        return out
    state = {"response": out, "response_style": _STYLE[language], "session_id": "hqb-correction",
             "user_id": "hqb-bench-user", "request_id": "hqb-correction"}
    return apply_language_quality(state, {"response": out}).get("response", out)


def mode_correction(args) -> list[dict[str, Any]]:
    from benchmarks.hinglish_quality import load_examples

    def one(e):
        out = production_postprocess(e.text, e.language, e.question, _INTENT.get(e.topic, "general_astrology"))
        row = evaluate(e, out, args.judge, e.reference, source=e.text, variant="correction")
        row["original"] = e.text
        row["changed"] = out.strip() != e.text.strip()
        row["exact_fix"] = e.kind == "negative" and _norm(out) == _norm(e.corrected or "")
        return row

    return _run_pool(one, load_examples(), args.workers)


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("।", ".").split())


# ── summaries ─────────────────────────────────────────────────────────
def summarize_run(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    from benchmarks.hinglish_quality.scoring import summarize

    all_rows = rows
    if mode == "chatbot":
        # A failed turn's reply is the canned safe message: scoring it would
        # reward an outage with a perfect language score.
        rows = [r for r in rows if not (r.get("chatbot") or {}).get("failed")]
    out: dict[str, Any] = {"all": summarize(rows)}
    for lang in ("hinglish", "hindi", "english"):
        out[lang] = summarize([r for r in rows if r["language"] == lang])
    rows_for_extra = all_rows
    if mode == "dataset":
        gold = [r for r in rows if r["kind"] == "gold" or r["variant"] == "corrected"]
        neg = [r for r in rows if r["kind"] == "negative" and r["variant"] == "text"]
        by_id = {(r["id"], r["variant"]): r for r in rows}
        pairs = [(by_id[(r["id"], "text")], by_id[(r["id"], "corrected")]) for r in neg]
        out["evaluator"] = {
            "gold_flagged_%": round(100.0 * sum(1 for r in gold if any(e["source"] != "judge" for e in r["score"]["errors"]))
                                    / max(1, len(gold)), 1),
            "gold_pass_%": round(100.0 * sum(1 for r in gold if r["score"]["passed"]) / max(1, len(gold)), 1),
            "negatives_detected_%": round(100.0 * sum(1 for r in neg if r["score"]["errors"]) / max(1, len(neg)), 1),
            "negatives_failed_%": round(100.0 * sum(1 for r in neg if not r["score"]["passed"]) / max(1, len(neg)), 1),
            "correction_ranked_higher_%": round(100.0 * sum(1 for a, b in pairs if b["score"]["overall"] > a["score"]["overall"])
                                                / max(1, len(pairs)), 1),
            "category_recall_%": _category_recall(neg),
            "holdout_negatives_detected_%": round(
                100.0 * sum(1 for r in neg if r["split"] == "holdout" and r["score"]["errors"])
                / max(1, sum(1 for r in neg if r["split"] == "holdout")), 1),
        }
    if mode == "correction":
        neg = [r for r in rows if r["kind"] == "negative"]
        gold = [r for r in rows if r["kind"] == "gold"]
        out["correction"] = {
            "negatives_changed_%": round(100.0 * sum(1 for r in neg if r["changed"]) / max(1, len(neg)), 1),
            "negatives_exactly_fixed_%": round(100.0 * sum(1 for r in neg if r["exact_fix"]) / max(1, len(neg)), 1),
            "negatives_passing_after_%": round(100.0 * sum(1 for r in neg if r["score"]["passed"]) / max(1, len(neg)), 1),
            "gold_unchanged_%": round(100.0 * sum(1 for r in gold if not r["changed"]) / max(1, len(gold)), 1),
        }
    if mode == "chatbot":
        from benchmarks.hinglish_quality.review import chatbot_coverage

        out["coverage"] = chatbot_coverage(rows_for_extra)
        for lang in ("hinglish", "hindi", "english"):
            out[lang]["coverage"] = chatbot_coverage([r for r in rows_for_extra if r["language"] == lang])
        out["chatbot"] = {"turns": len(rows_for_extra),
                          "failed_turns": sum(1 for r in rows_for_extra if (r.get("chatbot") or {}).get("failed")),
                          "fallback_model_%": round(100.0 * sum(1 for r in rows if (r.get("chatbot") or {}).get("fallback_used"))
                                                    / max(1, len(rows)), 1),
                          "latency_ms_avg": round(sum((r.get("chatbot") or {}).get("latency_ms") or 0 for r in rows)
                                                  / max(1, len(rows)))}
    return out


def _category_recall(neg: list[dict[str, Any]]) -> float:
    from benchmarks.hinglish_quality.schema import CATEGORIES

    hit = total = 0
    for r in neg:
        found_dims = {e["dimension"] for e in r["score"]["errors"]}
        for category in r["expected_errors"]:
            total += 1
            hit += CATEGORIES[category] in found_dims
    return round(100.0 * hit / max(1, total), 1)


def save(label: str, meta: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "summary": summarize_run(rows, meta["mode"]), "rows": rows}
    path = OUT_DIR / f"{label}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def add_judge(path: Path, workers: int) -> Path:
    """Judge an existing run (paced, cached) and rescore it. Lets a live run
    and its judging happen at different times — the judge shares the
    provider's rate limit with the chatbot."""
    from benchmarks.hinglish_quality import load_examples
    from benchmarks.hinglish_quality.facts import FactReport
    from benchmarks.hinglish_quality.judge import judge_model, judge_with_status, save_cache
    from benchmarks.hinglish_quality.review import reply_status
    from benchmarks.hinglish_quality.rules import check
    from benchmarks.hinglish_quality.scoring import score

    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {e.id: e for e in load_examples()}

    def one(row):
        e = by_id[row["id"]]
        if data["meta"]["mode"] == "chatbot" and reply_status(row) in ("safe_fallback", "failed"):
            verdict, row["judge_status"] = None, f"excluded:{reply_status(row)}"
        else:
            verdict, row["judge_status"] = judge_with_status(e.language, e.question, row["text"], row.get("reference"))
        facts = FactReport(**row["facts"])
        row["score"] = score(check(row["text"], e.language), facts, verdict).to_dict()
        return row

    data["rows"] = _run_pool(one, data["rows"], workers)
    save_cache()
    data["meta"]["judge"] = {**data["meta"]["judge"], "used": True, "model": judge_model()}
    data["summary"] = summarize_run(data["rows"], data["meta"]["mode"])
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def compare(base_path: str, new_path: str, fail: bool) -> int:
    base, new = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (base_path, new_path))
    if base["meta"]["dataset_hash"] != new["meta"]["dataset_hash"]:
        print("WARNING: different dataset versions — the comparison is not like-for-like")
    regressions = []
    print(f"| metric | {base['meta']['label']} | {new['meta']['label']} | delta |\n|---|---|---|---|")
    for lang in ("all", "hinglish", "hindi", "english"):
        b, n = base["summary"].get(lang, {}), new["summary"].get(lang, {})
        for key in ("overall", "grammar", "naturalness", "vocabulary", "meaning", "consistency", "pass_rate_%",
                    "fact_gate_%", "errors_total"):
            if key not in b or key not in n:
                continue
            delta = round(n[key] - b[key], 3)
            print(f"| {lang}.{key} | {b[key]} | {n[key]} | {delta:+} |")
            limit = THRESHOLDS["overall"] if key == "overall" else THRESHOLDS.get(key, THRESHOLDS["dimension"])
            worse = delta > limit if key == "fact_gate_%" else (key != "errors_total" and -delta > limit)
            if worse:
                regressions.append(f"{lang}.{key}: {b[key]} -> {n[key]}")
    if regressions:
        print("\nREGRESSIONS:\n  " + "\n  ".join(regressions))
    else:
        print("\nno regression beyond thresholds", THRESHOLDS)
    return 1 if regressions and fail else 0


def export_review(out: Path) -> None:
    from benchmarks.hinglish_quality import load_examples

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "language", "kind", "question", "text", "proposed_correction", "annotated_errors",
                    "reviewer", "verdict (ok / wrong / unsure)", "your_correction", "notes"])
        for e in load_examples():
            if e.language in ("hinglish", "hindi"):
                w.writerow([e.id, e.language, e.kind, e.question, e.text, e.corrected or "",
                            "; ".join(f"{x.category}: {x.span} -> {x.fix}" for x in e.errors), "", "", "", ""])
    print(f"wrote {out}")


def import_review(sheet: Path) -> None:
    path = HERE / "benchmarks" / "hinglish_quality" / "reviews.json"
    reviews = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    with sheet.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            verdict = (row.get("verdict (ok / wrong / unsure)") or "").strip().lower()
            reviewer = (row.get("reviewer") or row.get("reviewer_name") or "").strip()
            if reviewer and verdict:
                # Chatbot-reply sheets are keyed by run, dataset sheets by example.
                key = f"{row['run']}:{row['id']}" if row.get("run") else row["id"]
                reviews[key] = {"status": f"native:{reviewer}:{verdict}",
                                "native_language": row.get("reviewer_native_language", ""),
                                "correction": row.get("your_correction") or row.get("reviewer_correction", ""),
                                "notes": row.get("notes") or row.get("reviewer_comments", "")}
    path.write_text(json.dumps(reviews, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(reviews)} reviewed examples in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("dataset", "chatbot", "correction"):
        p = sub.add_parser(name)
        p.add_argument("--label", required=True)
        p.add_argument("--judge", action="store_true", help="add the LLM judge (cached)")
        p.add_argument("--workers", type=int, default=2)
        if name == "chatbot":
            p.add_argument("--languages", nargs="*", choices=("hinglish", "hindi", "english"))
            p.add_argument("--limit", type=int, default=0)
    j = sub.add_parser("judge")
    j.add_argument("run")
    j.add_argument("--workers", type=int, default=1)
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--fail-on-regression", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("runs", nargs="+")
    r.add_argument("--out", required=True)
    e = sub.add_parser("export-review")
    e.add_argument("--out", default=str(REVIEW_DIR / "review-sheet.csv"))
    e.add_argument("--runs", nargs="*", help="chatbot run files: export their replies (original vs corrected) "
                                             "instead of the dataset")
    i = sub.add_parser("import-review")
    i.add_argument("sheet")
    args = parser.parse_args()

    if args.cmd in ("dataset", "chatbot", "correction"):
        _env()
        from benchmarks.hinglish_quality.judge import save_cache

        rows = {"dataset": mode_dataset, "chatbot": mode_chatbot, "correction": mode_correction}[args.cmd](args)
        save_cache()
        path = save(args.label, _meta(args.cmd, args.label, args.judge), rows)
        summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        print(f"wrote {path}")
    elif args.cmd == "judge":
        _env()
        path = add_judge(Path(args.run), args.workers)
        print(json.dumps(json.loads(path.read_text(encoding="utf-8"))["summary"]["all"], ensure_ascii=False, indent=1))
    elif args.cmd == "compare":
        sys.exit(compare(args.base, args.new, args.fail_on_regression))
    elif args.cmd == "report":
        from benchmarks.hinglish_quality.report import write_report

        print(f"wrote {write_report([Path(p) for p in args.runs], Path(args.out))}")
    elif args.cmd == "export-review" and args.runs:
        from benchmarks.hinglish_quality.review import export_chatbot_review

        print(f"wrote {export_chatbot_review([Path(p) for p in args.runs], Path(args.out))} rows to {args.out}")
    elif args.cmd == "export-review":
        export_review(Path(args.out))
    else:
        import_review(Path(args.sheet))


if __name__ == "__main__":
    main()
