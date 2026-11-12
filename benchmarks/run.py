"""Benchmark the NVIDIA models on the 220-case suite.

    python -m benchmarks.run                         # all 3 models, all cases
    python -m benchmarks.run --limit 3 --judge none  # quick look
    python -m benchmarks.run --categories rag,function_calling
    python -m benchmarks.run --mock                  # offline: validates the harness only

Reproducibility rules:

* Identical messages, tools, temperature and max_tokens for every model; the
  dataset's SHA-256 is written into the report.
* Cases are run case-major (case 1 on every model, then case 2 ...), so a
  slow minute on the endpoint hits all models alike.
* No fallback and no routing: `llm_for_model` pins the model under test, so a
  failure shows as a failure instead of being answered by another model.
* The judge, when used, is a model that is NOT being benchmarked.
* Benchmark calls are kept out of the usage ledger / Token Meter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.llm.usage_tracker import set_traffic_source  # noqa: E402

# Internal traffic: kept out of the real-user view of the Token Meter.
set_traffic_source("benchmark")
from app.llm import llm_factory  # noqa: E402
from app.llm.usage_tracker import new_request_id, usage_tracker  # noqa: E402
from benchmarks.cases import CASES, TOOLS  # noqa: E402
from benchmarks.images import to_data_url  # noqa: E402
from benchmarks.scoring import judge_messages, parse_judge, score_case, similarity  # noqa: E402

BASE_SYSTEM = ("You are Tredev Astro's AI assistant: a warm, honest Vedic astrology guide "
               "for users in India. Be concise.")
TEMPERATURE, MAX_TOKENS = 0.2, 900
# Not under test, and reachable on this account (nemotron-3-super-120b and
# mistral-large-2 return 404 "Function not found" for it, verified 2026-11-12).
DEFAULT_JUDGE = "nvidia/nemotron-3.5-super-vl-120b-a12b"


# "raw" = the case's own instruction only (model ability).
# "production" = the same sources plus the production grounding protocol
# (app/rag/grounding.py) and citation validation — what users get.
GROUNDING = "raw"
# Adds the verification pass (app/rag/grounding.verification_messages).
VERIFY = False


def build_messages(case: dict[str, Any]) -> list[dict[str, Any]]:
    system = BASE_SYSTEM + (f"\n\n{case['system']}" if case.get("system") else "")
    if case.get("context"):
        system += "\n\nSOURCES:\n" + "\n".join(f"[{c['id']}] {c['text']}" for c in case["context"])
        if GROUNDING == "production":
            from app.rag.grounding import grounding_rules

            system += "\n\n" + grounding_rules("S")
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages += [dict(m) for m in case.get("history") or []]
    if case.get("image"):
        messages.append({"role": "user", "content": [
            {"type": "text", "text": case["prompt"]},
            {"type": "image_url", "image_url": {"url": to_data_url(case["image"])}},
        ]})
    else:
        messages.append({"role": "user", "content": case["prompt"]})
    return messages


class Pacer:
    """Spaces call starts to stay under the free tier's requests-per-minute."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last = 0.0

    def wait(self) -> None:
        delay = self.last + self.interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


def call_model(model: str, case: dict[str, Any], messages: list, pacer: Pacer) -> dict[str, Any]:
    pacer.wait()
    rid = f"bench-{new_request_id()}"
    tools = [TOOLS[name] for name in case.get("tools") or []] or None
    started = time.perf_counter()
    out: dict[str, Any] = {"model": model, "case_id": case["id"], "category": case["category"]}
    try:
        result = llm_factory.llm_for_model(
            model, role="job-benchmark", temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
        ).complete(messages, request_id=rid, tools=tools, tool_choice="auto" if tools else None)
        u = result.usage
        text = result.text
        if VERIFY and case.get("context"):
            from app.rag.grounding import verification_messages

            # The same model checks its own draft against the same sources —
            # mirrors GROUNDING_VERIFIER in production.
            pacer.wait()
            try:
                checked = llm_factory.llm_for_model(
                    model, role="job-benchmark-verify", temperature=0.0, max_tokens=MAX_TOKENS,
                ).complete(verification_messages(case["prompt"],
                                                 [(c["id"], c["text"]) for c in case["context"]],
                                                 text, tag="S"))
                text = checked.text
                out["verified"] = True
                out["verify_ms"] = checked.usage.latency_ms
            except Exception as exc:  # noqa: BLE001 — keep the unverified draft
                out["verify_error"] = type(exc).__name__
        if GROUNDING == "production" and case.get("context"):
            from app.rag.grounding import process_citations

            # Same post-processing users get: tags pointing at no real source
            # are removed (and counted) before scoring.
            cited = process_citations(text, len(case["context"]), tag="S", keep_inline=True)
            out["invalid_citations"] = cited.invalid
            text = cited.text
        out.update(text=text, tool_calls=result.tool_calls, error=None,
                   finish_reason=result.finish_reason, latency_ms=u.latency_ms,
                   input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                   token_source=u.token_source,
                   estimated_output_tokens=u.estimated_output_tokens)
    except Exception as exc:  # noqa: BLE001 — a failure is a data point
        info = getattr(exc, "info", None)
        out.update(text="", tool_calls=[], error=(info.label() if info else type(exc).__name__),
                   error_detail=str(exc)[:300], latency_ms=None, input_tokens=None,
                   output_tokens=None, token_source="unavailable")
    out["wall_ms"] = int((time.perf_counter() - started) * 1000)
    out["attempts"] = len(usage_tracker.for_request(rid))
    return out


def judge(judge_model: str, case: dict, text: str, tool_calls: list, pacer: Pacer) -> dict | None:
    pacer.wait()
    try:
        raw = llm_factory.llm_for_model(judge_model, role="job-benchmark-judge",
                                        temperature=0.0, max_tokens=200
                                        ).complete(judge_messages(case, text, tool_calls),
                                                   response_format={"type": "json_object"})
        return parse_judge(raw.text)
    except Exception:  # noqa: BLE001
        return None


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarise(rows: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model in models:
        mine = [r for r in rows if r["model"] == model and r["run"] == 0]
        ok = [r for r in mine if not r["error"]]
        lat = [float(r["latency_ms"]) for r in ok if r["latency_ms"] is not None]
        cats: dict[str, Any] = {}
        for cat in sorted({r["category"] for r in mine}):
            cr = [r for r in mine if r["category"] == cat]
            cats[cat] = {"cases": len(cr), "pass_rate": _mean([float(r["score"]["passed"]) for r in cr])}
        consistency = [r["consistency"] for r in mine if r.get("consistency")]
        judged = [r["judge"] for r in mine if r.get("judge")]
        summary[model] = {
            "cases": len(mine),
            "errors": len(mine) - len(ok),
            "error_codes": sorted({r["error"] for r in mine if r["error"]}),
            "pass_rate": _mean([float(r["score"]["passed"]) for r in mine]),
            "by_category": cats,
            "rag_groundedness": _mean([float(r["score"]["grounded"]) for r in mine
                                       if r["score"]["grounded"] is not None]),
            "tool_call_accuracy": _mean([float(r["score"]["tool_correct"]) for r in mine
                                         if r["score"]["tool_correct"] is not None]),
            "language_accuracy": _mean([float(r["score"]["language_ok"]) for r in mine
                                        if r["score"]["language_ok"] is not None]),
            "latency_ms": {"mean": _mean(lat), "p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95)},
            "output_tokens_reported_mean": _mean([float(r["output_tokens"]) for r in ok
                                                  if r["output_tokens"] is not None]),
            "calls_with_estimated_tokens": sum(r["token_source"] == "estimated" for r in ok),
            "judge": {k: _mean([j[k] for j in judged if j.get(k) is not None])
                      for k in ("quality", "factual", "grounded")} | {"judged": len(judged)},
            "consistency": {
                "cases": len(consistency),
                "pass_agreement": _mean([float(c["pass_agreement"]) for c in consistency]),
                "text_similarity": _mean([c["similarity"] for c in consistency]),
            },
        }
    return summary


def write_report(path: Path, meta: dict, summary: dict) -> None:
    models = list(summary)
    lines = [f"# NVIDIA model benchmark — {meta['started_at']}", "",
             f"Dataset: {meta['cases']} cases, sha256 `{meta['dataset_sha256'][:16]}`; "
             f"temperature {TEMPERATURE}, max_tokens {MAX_TOKENS}; judge: `{meta['judge']}`; "
             f"mode: {'MOCK (harness check only — not model results)' if meta['mock'] else 'live'}.", "",
             "| Metric | " + " | ".join(f"`{m}`" for m in models) + " |",
             "|---|" + "---|" * len(models)]

    def row(label: str, fn) -> None:
        lines.append(f"| {label} | " + " | ".join(fn(summary[m]) for m in models) + " |")

    fmt = lambda v, pct=False: "—" if v is None else (f"{v * 100:.1f}%" if pct else f"{v:,.0f}" if v > 50 else f"{v:.2f}")  # noqa: E731
    row("Overall pass rate", lambda s: fmt(s["pass_rate"], True))
    for cat in sorted({c for s in summary.values() for c in s["by_category"]}):
        row(f"· {cat}", lambda s, c=cat: fmt(s["by_category"].get(c, {}).get("pass_rate"), True))
    row("RAG groundedness", lambda s: fmt(s["rag_groundedness"], True))
    row("Tool-call accuracy", lambda s: fmt(s["tool_call_accuracy"], True))
    row("Reply-language accuracy", lambda s: fmt(s["language_accuracy"], True))
    row("Judge quality (1-5)", lambda s: fmt(s["judge"]["quality"]))
    row("Judge factual (1-5)", lambda s: fmt(s["judge"]["factual"]))
    row("Judge grounded (1-5)", lambda s: fmt(s["judge"]["grounded"]))
    row("Consistency: pass agreement", lambda s: fmt(s["consistency"]["pass_agreement"], True))
    row("Consistency: text similarity", lambda s: fmt(s["consistency"]["text_similarity"]))
    row("Latency mean (ms)", lambda s: fmt(s["latency_ms"]["mean"]))
    row("Latency p50 / p95 (ms)", lambda s: f"{fmt(s['latency_ms']['p50'])} / {fmt(s['latency_ms']['p95'])}")
    row("Output tokens (provider-reported, mean)", lambda s: fmt(s["output_tokens_reported_mean"]))
    row("Calls with estimated tokens", lambda s: str(s["calls_with_estimated_tokens"]))
    row("Errors", lambda s: f"{s['errors']} {', '.join(s['error_codes'])}".strip())
    lines += ["", "Costs are not reported: these are free hosted endpoints, so any figure "
              "would be hypothetical. Pass rates come from deterministic checks in "
              "`benchmarks/cases_*.py`; judge scores are an extra signal, not ground truth."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _install_mock() -> None:
    """Deterministic offline endpoint, for validating the harness in CI."""
    import dataclasses

    from tests.llm_fakes import FakeClient, reply

    class Echo(FakeClient):
        def _create(self, **kwargs):
            self.script.append(reply('{"quality": 3, "factual": 3, "grounded": null, "reason": "mock"}'
                                     if kwargs.get("response_format") else "Mock answer [S1]. Answer: 0"))
            return super()._create(**kwargs)

    patched = dataclasses.replace(settings, llm_provider="nvidia", nvidia=dataclasses.replace(
        settings.nvidia, api_key="mock"))
    llm_factory.settings = patched
    client = Echo([])
    llm_factory._openai_client = lambda: client


def rejudge(out_dir: Path, judge_model: str, rpm: float) -> int:
    """Judge an existing run from its saved answers — no regeneration, so
    the answers being judged are exactly the ones already scored."""
    results = out_dir / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {c["id"]: c for c in CASES}
    pacer = Pacer(rpm)
    todo = [r for r in rows if r["run"] == 0 and not r.get("error") and not r.get("judge")]
    for i, r in enumerate(todo, 1):
        r["judge"] = judge(judge_model, by_id[r["case_id"]], r["text"], r["tool_calls"], pacer)
        print(f"[{i:>3}/{len(todo)}] judged {r['case_id']:<9} {r['model']:<42} {r['judge']}", flush=True)
    results.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n",
                       encoding="utf-8")
    meta = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["meta"]
    meta["judge"] = judge_model
    models = list(dict.fromkeys(r["model"] for r in rows))
    summary = summarise(rows, models)
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2),
                                          encoding="utf-8")
    write_report(out_dir / "report.md", meta, summary)
    print(f"\nReport: {out_dir / 'report.md'}")
    return 0


def rescore(out_dir: Path) -> int:
    """Re-apply the current deterministic checks to saved answers (no model
    calls). The previous report is kept as report.v1.md."""
    results = out_dir / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {c["id"]: c for c in CASES}
    changed = 0
    for r in rows:
        new = score_case(by_id[r["case_id"]], r["text"], r["tool_calls"])
        changed += new["passed"] != r["score"]["passed"]
        r["score"] = new
    results.write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n",
                       encoding="utf-8")
    report = out_dir / "report.md"
    if report.exists() and not (out_dir / "report.v1.md").exists():
        report.rename(out_dir / "report.v1.md")
    meta = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["meta"]
    meta["rescored"] = True
    models = list(dict.fromkeys(r["model"] for r in rows))
    summary = summarise(rows, models)
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2),
                                          encoding="utf-8")
    write_report(report, meta, summary)
    print(f"rescored {len(rows)} rows, {changed} verdicts changed -> {report}")
    return 0


def main() -> int:
    n = settings.nvidia
    parser = argparse.ArgumentParser(description="Benchmark NVIDIA models on identical prompts")
    parser.add_argument("--models", default=",".join([n.main, n.fallback, n.benchmark]))
    parser.add_argument("--categories", default="", help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=0, help="first N cases per category")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="model id, or 'none'")
    parser.add_argument("--repeats", type=int, default=2, help="runs for consistency cases")
    parser.add_argument("--rpm", type=float, default=30.0, help="max requests per minute")
    parser.add_argument("--out", default="", help="output directory")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--rejudge", default="", help="judge an existing output dir without regenerating")
    parser.add_argument("--rescore", default="", help="re-apply current checks to an existing output dir")
    parser.add_argument("--grounding", choices=["raw", "production"], default="raw",
                        help="production = add the production grounding protocol to RAG cases")
    parser.add_argument("--verify", action="store_true", help="add the grounding verification pass")
    args = parser.parse_args()

    if args.rejudge:
        usage_tracker.persist = False
        return rejudge(Path(args.rejudge), args.judge, args.rpm)
    if args.rescore:
        return rescore(Path(args.rescore))
    global GROUNDING, VERIFY
    GROUNDING = args.grounding
    VERIFY = args.verify

    if args.mock:
        _install_mock()
        args.rpm = 0
    elif llm_factory.active_provider() != "nvidia" or not settings.nvidia.configured:
        print("Set LLM_PROVIDER=nvidia and NVIDIA_API_KEY in ai-core/.env (or use --mock).")
        return 1
    usage_tracker.persist = False  # keep benchmark traffic out of the Token Meter

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    judge_model = None if args.judge.lower() == "none" else args.judge
    if judge_model in models:
        print(f"Judge {judge_model} is also under test; pick a neutral judge.")
        return 1
    wanted = {c for c in args.categories.split(",") if c}
    cases, seen = [], {}
    for case in CASES:
        if wanted and case["category"] not in wanted:
            continue
        seen[case["category"]] = seen.get(case["category"], 0) + 1
        if args.limit and seen[case["category"]] > args.limit:
            continue
        cases.append(case)

    started = datetime.now(timezone.utc)
    out_dir = Path(args.out) if args.out else settings.base_dir / "logs" / "benchmarks" / f"{started:%Y%m%dT%H%M%SZ}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"started_at": started.isoformat(), "models": models, "judge": judge_model,
            "cases": len(cases), "mock": args.mock, "repeats": args.repeats, "grounding": GROUNDING, "verify": VERIFY,
            "base_url": settings.nvidia.base_url, "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "dataset_sha256": hashlib.sha256(json.dumps(CASES, sort_keys=True,
                                                        ensure_ascii=False).encode()).hexdigest()}
    pacer = Pacer(args.rpm)
    rows: list[dict[str, Any]] = []
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as handle:
        for i, case in enumerate(cases, 1):
            messages = build_messages(case)
            for model in models:
                runs = []
                for run in range(args.repeats if case.get("consistency") else 1):
                    r = call_model(model, case, messages, pacer)
                    r["run"] = run
                    r["score"] = score_case(case, r["text"], r["tool_calls"])
                    runs.append(r)
                first = runs[0]
                if len(runs) > 1:
                    first["consistency"] = {
                        "pass_agreement": len({x["score"]["passed"] for x in runs}) == 1,
                        "similarity": _mean([similarity(first["text"], x["text"]) for x in runs[1:]]),
                    }
                if judge_model and not first["error"]:
                    first["judge"] = judge(judge_model, case, first["text"], first["tool_calls"], pacer)
                for r in runs:
                    handle.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                rows.extend(runs)
                mark = "ok" if first["score"]["passed"] else ("ERR " + first["error"] if first["error"] else "fail")
                print(f"[{i:>3}/{len(cases)}] {case['id']:<9} {model:<42} {mark}", flush=True)
            handle.flush()

    summary = summarise(rows, models)
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2),
                                          encoding="utf-8")
    write_report(out_dir / "report.md", meta, summary)
    print(f"\nReport: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
