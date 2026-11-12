"""End-to-end quality + latency evaluation of the REAL chat pipeline.

    # before (every optimisation flag off) and after (defaults):
    STRUCTURED_CHART_CONTEXT=false CHART_VALIDATION_MODE=log LANGUAGE_GUARD=false \
    CLASSIFIER_MODE=llm EVALUATION_ASYNC=false METERING_ASYNC=false REDIS_PIPELINE=false \
        python -m scripts.eval_pipeline --label before
    python -m scripts.eval_pipeline --label after
    python -m scripts.eval_pipeline --compare logs/evals/before.json logs/evals/after.json

Unlike `benchmarks/run.py` (raw model ability on a fixed system prompt), this
sends questions through `run_turn` — production prompt, chart context,
validation, guards — and measures what a user would actually receive:

* chart-fact accuracy — claims in the FINAL reply checked deterministically
  against the Kundli engine's chart (app.evaluation.chart_claims)
* language/script preservation — reply language vs the language asked in
* latency — per-turn trace totals, avg/p50/p95/p99
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.graph.chatbot_graph import run_turn  # noqa: E402
from app.kundli.chart_facts import build_chart_facts  # noqa: E402
from app.kundli.kundli_service import SampleFileKundliProvider  # noqa: E402
from app.memory.redis_memory import memory  # noqa: E402
from app.observability import background, trace  # noqa: E402
from app.prompts.response_style import resolve_style  # noqa: E402
from benchmarks.cases import CASES  # noqa: E402
from benchmarks.scoring import language_ok  # noqa: E402
from app.evaluation.chart_claims import validate_reply  # noqa: E402

# Chart-specific questions: these are where a placement can be misstated.
CHART_QUESTIONS: list[tuple[str, str]] = [
    ("english", "How does my career look? I'm a software developer."),
    ("english", "Is my 10th house strong?"),
    ("english", "Which house is my Jupiter in, and what does it mean for me?"),
    ("english", "What does my chart say about marriage?"),
    ("english", "Which mahadasha am I running and what does it mean?"),
    ("english", "Is Saturn troubling me right now?"),
    ("english", "What is my ascendant and moon sign?"),
    ("english", "Do I have any doshas in my chart?"),
    ("english", "Will I make good money? Tell me from my chart."),
    ("english", "What do my 7th house and its lord say?"),
    ("hinglish", "Meri career growth kab achhi hogi?"),
    ("hinglish", "Mera Guru kis ghar mein baitha hai?"),
    ("hinglish", "Meri shaadi ke yog kaise hain mere chart mein?"),
    ("hinglish", "Shani ka mere upar kya asar hai abhi?"),
    ("hinglish", "Mera lagna aur rashi kya hai?"),
    ("hinglish", "Paisa kab aayega? Chart dekh ke batao."),
    ("hinglish", "Mere 10th house mein kaunsa planet hai?"),
    ("hindi", "मेरे करियर के बारे में मेरी कुंडली क्या कहती है?"),
    ("hindi", "मेरा गुरु किस भाव में है?"),
    ("hindi", "मेरी शादी कब होगी?"),
    ("hindi", "अभी मेरी कौन सी महादशा चल रही है?"),
    ("hindi", "क्या मेरी कुंडली में कोई दोष है?"),
]
# Multi-turn: continuity and not re-asking known details.
CONVERSATIONS: list[list[tuple[str, str]]] = [
    [("english", "I work as a software developer and I'm 30."),
     ("english", "Is this a good time for a job change?"),
     ("english", "And what about next year?")],
    [("hinglish", "Main teacher hoon, Delhi mein."),
     ("hinglish", "Kya mujhe naya job dhundhna chahiye?"),
     ("hinglish", "Aur shaadi ke baare mein?")],
]
_LANG_OF = {"english": "english", "hinglish": "hinglish", "hindi": "hindi_devanagari"}


def _turn(question: str, session: str, lang: str, facts: Any) -> dict[str, Any]:
    # Exactly what POST /chat does: the reply style is resolved from the
    # question (auto -> English / Hinglish / Hindi) before the turn runs.
    style = resolve_style(None, question, settings.api.default_response_style)
    with trace.request("pending", persist=False) as t:
        state = run_turn(question, session, settings.default_user_id, chat_bubbles=True,
                         response_style=style.id)
    reply = state.get("response", "") or ""
    metrics = state.get("metrics") or {}
    final = validate_reply(reply, facts)
    validation = metrics.get("validation") or {}
    return {
        "question": question, "lang": lang, "reply": reply,
        "total_ms": t.total_ms, "by_span_ms": t.summary()["by_span_ms"],
        "failed": bool(metrics.get("generation_failed")),
        "model": metrics.get("model"), "intent": metrics.get("intent"),
        "intent_method": metrics.get("intent_method"), "fallback": metrics.get("fallback_used"),
        "claims": len(final.claims),
        "contradictions": [c.describe() for c in final.contradictions],
        "unsupported": len(final.unsupported),
        "draft_contradictions": len((validation.get("draft") or {}).get("contradictions") or []),
        "validation_action": validation.get("action", "none"),
        "language_ok": language_ok(reply, _LANG_OF[lang]),
    }


def run(label: str, multilingual_cases: int) -> Path:
    facts = build_chart_facts(SampleFileKundliProvider().get_kundli(settings.default_user_id))
    turns: list[dict[str, Any]] = []
    for lang, question in CHART_QUESTIONS:
        session = f"eval-{uuid.uuid4().hex[:8]}"
        turns.append({"kind": "chart", **_turn(question, session, lang, facts)})
        memory.clear_session(session)
        print(f"  chart  {turns[-1]['total_ms']:>6}ms lang_ok={turns[-1]['language_ok']!s:<5} "
              f"contra={len(turns[-1]['contradictions'])} :: {question[:50]}", flush=True)
    for convo in CONVERSATIONS:
        session = f"eval-{uuid.uuid4().hex[:8]}"
        for lang, question in convo:
            turns.append({"kind": "conversation", **_turn(question, session, lang, facts)})
            print(f"  convo  {turns[-1]['total_ms']:>6}ms :: {question[:50]}", flush=True)
        memory.clear_session(session)
    ml = [c for c in CASES if c["category"] == "multilingual"
          and c["checks"].get("language") in {"hinglish", "hindi_devanagari"}][:multilingual_cases]
    for case in ml:
        session = f"eval-{uuid.uuid4().hex[:8]}"
        lang = "hinglish" if case["checks"]["language"] == "hinglish" else "hindi"
        turns.append({"kind": "multilingual", "case_id": case["id"],
                      **_turn(case["prompt"], session, lang, facts)})
        memory.clear_session(session)
        print(f"  ml     {turns[-1]['total_ms']:>6}ms lang_ok={turns[-1]['language_ok']!s:<5} :: "
              f"{case['id']}", flush=True)
    background.drain(30)
    out = settings.base_dir / "logs" / "evals" / f"{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": label, "ts": time.time(), "flags": settings.flags.__dict__,
                               "summary": summarise(turns), "turns": turns},
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo))


def summarise(turns: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [t for t in turns if not t["failed"]]
    lat = [float(t["total_ms"]) for t in turns]
    claims = sum(t["claims"] for t in ok)
    contra = sum(len(t["contradictions"]) for t in ok)
    by_lang: dict[str, list[bool]] = {}
    for t in ok:
        by_lang.setdefault(t["lang"], []).append(t["language_ok"])
    return {
        "turns": len(turns), "failures": len(turns) - len(ok),
        "replies_with_contradiction": sum(1 for t in ok if t["contradictions"]),
        "chart_claims_checked": claims, "contradicted_claims": contra,
        "chart_fact_accuracy": round(1 - contra / claims, 4) if claims else None,
        "drafts_with_contradiction": sum(1 for t in ok if t["draft_contradictions"]),
        "unsupported_claims": sum(t["unsupported"] for t in ok),
        "language_accuracy": round(statistics.fmean([t["language_ok"] for t in ok]), 4) if ok else None,
        "language_by_lang": {k: f"{sum(v)}/{len(v)}" for k, v in by_lang.items()},
        "llm_classifier_skipped": sum(1 for t in turns if t.get("intent_method") == "keyword"),
        "latency_ms": {"avg": round(statistics.fmean(lat)) if lat else 0, "p50": _pct(lat, .5),
                       "p95": _pct(lat, .95), "p99": _pct(lat, .99)},
    }


def compare(paths: list[str]) -> None:
    data = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    keys = ["turns", "failures", "replies_with_contradiction", "chart_claims_checked",
            "contradicted_claims", "chart_fact_accuracy", "drafts_with_contradiction",
            "unsupported_claims", "language_accuracy", "language_by_lang", "llm_classifier_skipped"]
    print(f"{'metric':<30}" + "".join(f"{d['label']:>22}" for d in data))
    for k in keys:
        print(f"{k:<30}" + "".join(f"{str(d['summary'].get(k)):>22}" for d in data))
    for q in ("avg", "p50", "p95", "p99"):
        print(f"{'latency ' + q + ' (ms)':<30}" + "".join(f"{d['summary']['latency_ms'][q]:>22}" for d in data))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--multilingual", type=int, default=23, help="Hinglish/Hindi benchmark prompts to include")
    parser.add_argument("--compare", nargs="*")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare)
        return 0
    path = run(args.label, args.multilingual)
    compare([str(path)])
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
