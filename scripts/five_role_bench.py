"""Candidate-model benchmarks for the Fallback and Humanizer roles.

    python -m scripts.five_role_bench fallback-run  --models A B ... [--timeout 25] [--limit N]
    python -m scripts.five_role_bench humanizer-run --models A B ... [--timeout 25] [--limit N]
    python -m scripts.five_role_bench judge         --kind fallback|humanizer --models A B ...
    python -m scripts.five_role_bench report        --kind fallback|humanizer --models A B ...

Everything runs through the production gateway (request building, retries,
timeouts), is tagged traffic_source="evaluation" and kept out of the usage
ledger. Raw outputs go to logs/benchmarks/five-role/ and runs resume where they
stopped. An unavailable model records its error and is skipped; nothing is ever
substituted. Scoring is deterministic (validators from the production pipeline)
plus an INDEPENDENT judge, Nemotron 3 Ultra, which is none of the candidates.

Fallback: the 75 single-turn questions of benchmarks/kundli_quality_cases.py (25
English, 25 Hinglish, 25 Hindi) on the production prompt (no retrieval).
Humanizer: the Hinglish Quality Benchmark's annotated "awkward" replies plus gold
replies that carry dates and placements (fact-preservation traps).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

import app.llm.llm_factory as gw  # noqa: E402
from app.config import settings as settings_module  # noqa: E402
from app.llm.usage_tracker import llm_purpose, set_traffic_source, usage_tracker  # noqa: E402

OUT = HERE / "logs" / "benchmarks" / "five-role"
ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"
PACE = 2.3
STYLE = {"english": "english", "hinglish": "whatsapp_hinglish", "hindi": "hindi"}
usage_tracker.persist = False
set_traffic_source("evaluation")


def configure(timeout: float, retries: int = 1) -> None:
    base = settings_module.settings
    gw.settings = dataclasses.replace(
        base, nvidia=dataclasses.replace(base.nvidia, timeout_seconds=timeout),
        reliability=dataclasses.replace(base.reliability, max_retries=retries, hedge_after_seconds=0.0))
    gw._client = None


def chart():
    from app.kundli.kundli_service import KundliContext
    from scripts.kundli_quality_bench import screenshot_chart

    return screenshot_chart(KundliContext)


def slug(model: str) -> str:
    return model.replace("/", "__")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def call(model: str, messages: list[dict], *, temperature: float, max_tokens: int, purpose: str) -> dict:
    """One pinned call through the gateway. Never raises: errors are data."""
    llm = gw.llm_for_model(model, role=gw.ROLE_MAIN, fallback_model=None, temperature=temperature, max_tokens=max_tokens)
    t0 = time.perf_counter()
    rec: dict = {"model": model}
    try:
        with llm_purpose(purpose):
            res = llm.complete(messages, session_id="bench", user_id="bench", request_id=f"b-{time.time_ns()}")
        rec.update(text=res.text, finish=res.finish_reason, served=res.usage.model, in_tokens=res.usage.input_tokens,
                   out_tokens=res.usage.output_tokens)
    except Exception as exc:  # noqa: BLE001
        info = getattr(exc, "info", None)
        rec.update(text="", error=(info.label() if info else type(exc).__name__), error_detail=str(exc)[:200])
    rec["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    time.sleep(PACE)
    return rec


# ── datasets ─────────────────────────────────────────────────────────
def fallback_cases() -> list[dict]:
    spec = importlib.util.spec_from_file_location("kq_cases", HERE / "benchmarks" / "kundli_quality_cases.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kq_cases"] = mod
    spec.loader.exec_module(mod)
    out = []
    for conv in mod.SINGLE:
        t = conv.turns[0]
        out.append({"id": conv.id, "language": t.language, "topic": t.topic, "question": t.question,
                    "life_fact": t.life_fact, "beyond_chart": t.beyond_chart})
    return out


def pick(cases: list[dict], n: int | None) -> list[dict]:
    """All cases, or n of them evenly spaced (so a small sample still spans every language)."""
    if not n or n >= len(cases):
        return cases
    return [cases[int(i * len(cases) / n)] for i in range(n)]


def production_messages(case: dict, kundli) -> list[dict]:
    """The exact message list the fallback receives in production (build_context_node), minus retrieval."""
    from app.nodes.classifier import classify_by_keywords, classify_unambiguous
    from app.persona.registry import get_persona
    from app.prompts.answer_depth import classify_depth, is_fact_lookup, length_line, turn_length_note
    from app.prompts.builder import build_messages
    from app.prompts.followup import decide, prompt_line
    from app.rag.retriever import RetrievalResult

    q = case["question"]
    intent = (classify_unambiguous(q, False) or classify_by_keywords(q)).intent
    follow = decide(q, intent, case["language"], None)
    depth = classify_depth(q, intent, False)
    return build_messages(
        question=q, intent=intent, kundli=kundli, retrieval=RetrievalResult.unavailable("benchmark"), history=[],
        session_state={}, persona=get_persona(None), chat_bubbles=True, response_style=STYLE[case["language"]],
        switch_only=False, followup_line=prompt_line(follow), length_line=length_line(depth, q, follow.allowed),
        length_note=turn_length_note(depth, q), length_note_always=is_fact_lookup(depth, q))


def humanizer_cases(per_group: int = 10, gold_per_language: int = 8) -> list[dict]:
    from benchmarks.hinglish_quality import load_examples

    ex = load_examples()
    cases, seen = [], set()
    for lang, domain in (("hinglish", "astrology"), ("hinglish", "conversational"), ("hindi", "astrology"),
                         ("english", "astrology")):
        group = [e for e in ex if (e.language, e.domain, e.kind) == (lang, domain, "negative")][:per_group]
        cases += [{"id": e.id, "language": e.language, "kind": "awkward", "text": e.text, "reference": e.corrected,
                   "question": e.question} for e in group]
        seen.update(e.id for e in group)
    for lang in ("hinglish", "hindi", "english"):
        gold = [e for e in ex if e.language == lang and e.kind == "gold" and e.domain == "astrology"
                and any(ch.isdigit() for ch in e.text)][:gold_per_language]
        cases += [{"id": e.id, "language": lang, "kind": "fact-trap", "text": e.text, "reference": e.text,
                   "question": e.question} for e in gold]
    return cases


# ── runs ─────────────────────────────────────────────────────────────
def run_fallback(models: list[str], timeout: float, limit: int | None) -> None:
    configure(timeout)
    kundli, cases = chart(), pick(fallback_cases(), limit)
    for model in models:
        path = OUT / f"fallback__{slug(model)}__t{int(timeout)}.jsonl"
        done = {r["id"] for r in read(path)}
        for c in cases:
            if c["id"] in done:
                continue
            rec = call(model, production_messages(c, kundli), temperature=0.7, max_tokens=700, purpose="answer")
            rec.update(id=c["id"], language=c["language"], topic=c["topic"], question=c["question"])
            append(path, rec)
            print(f"{model.split('/')[-1][:24]:24} {c['id']} {rec['latency_ms']:6}ms out={rec.get('out_tokens')} "
                  f"{rec.get('error', '')}", flush=True)


def run_humanizer(models: list[str], timeout: float, limit: int | None) -> None:
    from app.prompts.roles.humanizer import (HUMANIZER_MAX_TOKENS, HUMANIZER_PROMPT_VERSION, HUMANIZER_TEMPERATURE,
                                             humanizer_messages)

    configure(timeout)
    cases = pick(humanizer_cases(), limit)
    for model in models:
        path = OUT / f"humanizer__{slug(model)}__t{int(timeout)}.jsonl"
        done = {r["id"] for r in read(path)}
        for c in cases:
            if c["id"] in done:
                continue
            rec = call(model, humanizer_messages(c["text"], c["language"]), temperature=HUMANIZER_TEMPERATURE,
                       max_tokens=HUMANIZER_MAX_TOKENS, purpose="humanize")
            rec.update(id=c["id"], language=c["language"], kind=c["kind"], original=c["text"], reference=c["reference"],
                       question=c["question"], prompt_version=HUMANIZER_PROMPT_VERSION)
            append(path, rec)
            print(f"{model.split('/')[-1][:24]:24} {c['id']} {rec['latency_ms']:6}ms {rec.get('error', '')}", flush=True)


# ── the independent judge (Ultra) ────────────────────────────────────
FALLBACK_JUDGE = None


def judge_call(system: str, user: str) -> dict | None:
    cfg = gw.RoleConfig(role=gw.ROLE_EVALUATE, model=ULTRA, temperature=0.0, max_tokens=500, fallback_model=None,
                        response_format={"type": "json_object"})
    for attempt in range(4):
        try:
            with llm_purpose("evaluate"):
                raw = gw.LLM(cfg).complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                           session_id="bench", user_id="bench", request_id=f"j-{time.time_ns()}").text
            time.sleep(PACE)
            return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except Exception:  # noqa: BLE001
            time.sleep(8 * (attempt + 1))
    return None


HUMANIZER_JUDGE = """You are a fluent native speaker of Hindi, English and Roman-script Hinglish judging ONE chat reply from a Vedic astrology assistant. Judge the LANGUAGE only, not the astrology.
Score 1-5 (5 flawless, 4 minor slip, 3 noticeable error, 2 several errors, 1 broken):
- grammar: agreement, verb forms, postpositions, sentence structure
- naturalness: would a fluent speaker say it this way in a friendly chat? Penalise literal translation and bookish register
- vocabulary: correct word choice and spelling (aapko not aapako; career not kareer; real words only)
- tone: warm, conversational, respectful ("aap" throughout)
- readability: short clear sentences, easy to follow on a phone
Expected language: {language}. Natural Hinglish keeps common English words (career, partnership, business, Mahadasha).
If an ORIGINAL is given, also say whether the reply CHANGED any astrological fact (planet, house, sign, date, period, past/current/future status, dosha, prediction or its certainty): "meaning_changed": true or false.
Return strict JSON only: {{"grammar": n, "naturalness": n, "vocabulary": n, "tone": n, "readability": n, "meaning_changed": false}}"""


def judge_humanizer(models: list[str], timeout: float) -> None:
    configure(60.0)
    cache = OUT / "judge__humanizer.jsonl"
    done = {(r["kind"], r["model"], r["id"]) for r in read(cache)}
    for model in models:
        for r in read(OUT / f"humanizer__{slug(model)}__t{int(timeout)}.jsonl"):
            for kind, text in (("original", r["original"]), ("output", r.get("text"))):
                key = (kind, model if kind == "output" else "-", r["id"])
                if key in done or not text:
                    continue
                user = f"Question: {r['question']}\n\nREPLY:\n{text}" + (
                    f"\n\nORIGINAL (the reply before editing):\n{r['original']}" if kind == "output" else "")
                v = judge_call(HUMANIZER_JUDGE.format(language=r["language"]), user)
                append(cache, {"kind": kind, "model": key[1], "id": r["id"], "language": r["language"], "verdict": v})
                done.add(key)
                print("judged", key, bool(v), flush=True)


def judge_fallback(models: list[str], timeout: float) -> None:
    from app.evaluation.evaluator import JUDGE_SYSTEM_PROMPT, _parse_judge
    from app.rag.retriever import RetrievalResult

    configure(60.0)
    kundli = chart()
    cache = OUT / "judge__fallback.jsonl"
    done = {(r["model"], r["id"], r["language"]) for r in read(cache)}      # ids repeat across languages
    for model in models:
        for r in read(OUT / f"fallback__{slug(model)}__t{int(timeout)}.jsonl"):
            if (model, r["id"], r["language"]) in done or not r.get("text"):
                continue
            payload = "\n\n".join(["CLASSIFIED INTENT: general", kundli.to_prompt_block("general"),
                                   RetrievalResult.unavailable("benchmark").to_prompt_block(),
                                   f"USER ASKED:\n{r['question']}", f"ASSISTANT REPLIED:\n{r['text']}"])
            v = judge_call(JUDGE_SYSTEM_PROMPT, payload)
            append(cache, {"model": model, "id": r["id"], "language": r["language"], "verdict": v})
            done.add((model, r["id"], r["language"]))
            print("judged", model.split("/")[-1], r["id"], bool(v), flush=True)


# ── reports ──────────────────────────────────────────────────────────
def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def mean(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    return f"{st.mean(xs):.2f}" if xs else "-"


def report_fallback(models: list[str], timeout: float) -> dict:
    from app.evaluation.chart_claims import validate_reply
    from app.evaluation.fact_claims import HARD_KINDS
    from app.evaluation.hinglish_grammar import detect
    from app.kundli.chart_facts import build_chart_facts
    from app.llm.llm_factory import looks_malformed
    from app.prompts.language import reply_matches
    from app.prompts.segments import is_complete, segment_reply

    facts = build_chart_facts(chart())
    judged = {(r["model"], r["id"], r["language"]): r["verdict"] for r in read(OUT / "judge__fallback.jsonl")}
    table: dict = {}
    for model in models:
        rows = read(OUT / f"fallback__{slug(model)}__t{int(timeout)}.jsonl")
        if not rows:
            continue
        ok = [r for r in rows if r.get("text")]
        errs: dict = {}
        for r in rows:
            if r.get("error"):
                errs[r["error"]] = errs.get(r["error"], 0) + 1
        m = {"n": len(rows), "answered": len(ok), "errors": errs,
             "malformed": sum(looks_malformed(r["text"]) for r in ok),
             "truncated": sum(r.get("finish") == "length" for r in ok)}
        bad = lang_bad = dosha = shape_bad = followups = issues = 0
        for r in ok:
            v = validate_reply(r["text"], facts)
            hard = v.contradictions or [u for u in v.unsupported if u.kind in HARD_KINDS]
            bad += bool(hard)
            dosha += any(c.kind == "dosha_verdict" for c in v.claims)
            lang_bad += not reply_matches(STYLE[r["language"]], r["text"])
            segs = segment_reply(r["text"])
            shape_bad += not (segs and all(is_complete(s) for s in segs))
            followups += r["text"].rstrip().endswith("?")
            if r["language"] != "english":
                issues += len(detect(r["text"], "hindi" if r["language"] == "hindi" else "hinglish"))
        lat = sorted(r["latency_ms"] for r in ok)
        m.update(factual_failures=bad, dosha_verdicts=dosha, wrong_language=lang_bad, incomplete_bubbles=shape_bad,
                 followup_rate=pct(followups, len(ok)), hinglish_issues_per_reply=round(issues / max(1, len(ok)), 2),
                 p50_ms=lat[len(lat) // 2] if lat else None, p95_ms=lat[int(len(lat) * .95) - 1] if lat else None,
                 avg_in=int(st.mean([r["in_tokens"] for r in ok if r.get("in_tokens")] or [0])),
                 avg_out=int(st.mean([r["out_tokens"] for r in ok if r.get("out_tokens")] or [0])))
        verdicts = [judged[(model, r["id"], r["language"])] for r in ok if judged.get((model, r["id"], r["language"]))]
        m["judged"] = len(verdicts)
        for dim in ("relevance", "groundedness", "completeness", "human_likeness", "hinglish_quality", "clarity", "safety"):
            m[f"judge_{dim}"] = mean([v.get(dim) for v in verdicts])
        by_lang = {}
        for lang in ("english", "hinglish", "hindi"):
            vs = [judged[(model, r["id"], r["language"])] for r in ok if r["language"] == lang and judged.get((model, r["id"], r["language"]))]
            by_lang[lang] = {"groundedness": mean([v.get("groundedness") for v in vs]),
                             "human_likeness": mean([v.get("human_likeness") for v in vs]), "n": len(vs)}
        m["judge_by_language"] = by_lang
        table[model] = m
    return table


def report_humanizer(models: list[str], timeout: float) -> dict:
    from app.evaluation.chart_claims import validate_reply
    from app.evaluation.fact_claims import HARD_KINDS
    from app.evaluation.hinglish_grammar import detect
    from app.evaluation.rewrite_guard import compare_rewrite
    from app.kundli.chart_facts import build_chart_facts
    from app.prompts.language import reply_matches

    facts = build_chart_facts(chart())
    judged = {(r["kind"], r["model"], r["id"]): r["verdict"] for r in read(OUT / "judge__humanizer.jsonl") if r.get("verdict")}
    table: dict = {}
    for model in models:
        rows = read(OUT / f"humanizer__{slug(model)}__t{int(timeout)}.jsonl")
        if not rows:
            continue
        ok = [r for r in rows if r.get("text")]
        errs: dict = {}
        for r in rows:
            if r.get("error"):
                errs[r["error"]] = errs.get(r["error"], 0) + 1
        accepted = changed = script_bad = 0
        reasons: dict = {}
        issues_before = issues_after = 0
        for r in ok:
            verdict = compare_rewrite(r["original"], r["text"], facts)
            accepted += verdict.ok
            for reason in verdict.reasons:
                key = reason.split(":")[0][:40]
                reasons[key] = reasons.get(key, 0) + 1
            changed += r["text"].strip() != r["original"].strip()
            script_bad += not reply_matches(STYLE[r["language"]], r["text"])
            if r["language"] != "english":
                lang = "hindi" if r["language"] == "hindi" else "hinglish"
                issues_before += len(detect(r["original"], lang))
                issues_after += len(detect(r["text"], lang))
        lat = sorted(r["latency_ms"] for r in ok)
        m = {"n": len(rows), "answered": len(ok), "errors": errs, "guard_accepted": accepted,
             "guard_rejected": len(ok) - accepted, "guard_reasons": reasons, "changed_text": changed,
             "wrong_script_or_language": script_bad, "hinglish_issues_before": issues_before,
             "hinglish_issues_after": issues_after, "p50_ms": lat[len(lat) // 2] if lat else None,
             "avg_in": int(st.mean([r["in_tokens"] for r in ok if r.get("in_tokens")] or [0])),
             "avg_out": int(st.mean([r["out_tokens"] for r in ok if r.get("out_tokens")] or [0]))}
        for dim in ("grammar", "naturalness", "vocabulary", "tone", "readability"):
            before = [judged[("original", "-", r["id"])][dim] for r in ok if ("original", "-", r["id"]) in judged and dim in judged[("original", "-", r["id"])]]
            after = [judged[("output", model, r["id"])][dim] for r in ok if ("output", model, r["id"]) in judged and dim in judged[("output", model, r["id"])]]
            m[f"judge_{dim}"] = {"before": mean(before), "after": mean(after)}
        outs = [judged[("output", model, r["id"])] for r in ok if ("output", model, r["id"]) in judged]
        m["judge_meaning_changed"] = sum(bool(v.get("meaning_changed")) for v in outs)
        m["judged"] = len(outs)
        table[model] = m
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fallback-run", "humanizer-run", "judge", "report"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--kind", choices=["fallback", "humanizer"], default="fallback")
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "fallback-run":
        run_fallback(a.models, a.timeout, a.limit)
    elif a.cmd == "humanizer-run":
        run_humanizer(a.models, a.timeout, a.limit)
    elif a.cmd == "judge":
        (judge_fallback if a.kind == "fallback" else judge_humanizer)(a.models, a.timeout)
    else:
        table = (report_fallback if a.kind == "fallback" else report_humanizer)(a.models, a.timeout)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"report-{a.kind}.json").write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(table, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
