"""Humanizer: language polish for Hinglish and Hindi replies, never astrology.

Runs after the deterministic validation and language fixes, on a reply that is
already approved. One model call (ROLE_HUMANIZE, its own prompt, temperature and
metering), only when the reply still has language issues — "only when it provides
meaningful improvement". The rewrite ships only if EVERY check below passes;
otherwise the original, already-validated reply is kept. There is no second
attempt: a failed rewrite is discarded, never retried until it passes.

    off      the node does nothing (default)
    shadow   runs in the BACKGROUND after delivery and records what it would have done
             (logs/humanizer_shadow.jsonl, no text); the reply is unchanged and never waits
    on       ships a rewrite that passes the guard

Accepted only if the rewrite: is non-empty, not a JSON/template fragment, does not
recite instructions; keeps the paragraph structure, every system notice and the
closing question; stays in the user's language and script, every bubble complete;
keeps all verified chart facts, Dasha dates and past/current/future status, adds
no claim, raises no certainty (`rewrite_guard.compare_rewrite`); has fewer language
issues than the original; and is of similar length.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any

from app.config.settings import BASE_DIR, settings
from app.evaluation.chart_claims import split_sentences
from app.evaluation.hinglish_grammar import detect
from app.evaluation.rewrite_guard import compare_rewrite
from app.kundli.chart_facts import ChartFacts, build_chart_facts
from app.llm.llm_factory import ROLE_HUMANIZE, ROLE_HUMANIZE_COMPARE, LLMNotConfigured, get_llm, looks_malformed
from app.llm.usage_tracker import llm_purpose, usage_tracker
from app.logging_setup import get_logger
from app.prompts.language import leaks_prompt, reply_matches
from app.prompts.roles.humanizer import (COMPARE_PROMPT_VERSION, DIMENSIONS, HUMANIZER_PROMPT_VERSION,
                                         compare_messages, humanizer_messages, parse_comparison)
from app.prompts.segments import is_complete, segment_reply
from app.safety.notices import is_system_sentence
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.humanize")

SHADOW_LOG = Path("logs") / "humanizer_shadow.jsonl"
LANGUAGES = {"hinglish": "hinglish", "hindi": "hindi"}      # English replies are not humanized
LENGTH_RATIO = (0.6, 1.4)


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _clean(raw: str) -> str:
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:\w+)?\s*(.*?)\s*```", text, re.DOTALL)
    text = fenced.group(1) if fenced else text
    return text.strip().strip('"“”').strip()


def check_rewrite(original: str, rewritten: str, *, facts: ChartFacts, style_id: str | None, language: str,
                  issues_before: int, require_improvement: bool = True) -> tuple[bool, str, int]:
    """(accepted, reason, issues_after). Pure: every acceptance rule in one place.

    `require_improvement=False` is for the shadow sample of already-clean replies, where there is
    no known issue to fix: the rewrite must then simply not add one."""
    from app.nodes.validate import _segments_ok

    if not rewritten:
        return False, "empty", issues_before
    if looks_malformed(rewritten) or leaks_prompt(rewritten):
        return False, "malformed_or_leaked", issues_before
    if len(_paragraphs(rewritten)) != len(_paragraphs(original)):
        return False, "paragraph_structure_changed", issues_before
    kept = {s for s in split_sentences(rewritten)}
    if any(is_system_sentence(s) and s not in kept for s in split_sentences(original)):
        return False, "system_notice_changed", issues_before
    if original.rstrip().endswith(("?", "？")) != rewritten.rstrip().endswith(("?", "？")):
        return False, "followup_question_changed", issues_before
    ratio = len(rewritten.split()) / max(1, len(original.split()))
    if not LENGTH_RATIO[0] <= ratio <= LENGTH_RATIO[1]:
        return False, f"length_ratio_{ratio:.2f}", issues_before
    if not _segments_ok(style_id, rewritten):
        return False, "language_or_script_changed", issues_before
    if not all(is_complete(b) for b in segment_reply(rewritten)):
        return False, "incomplete_bubble", issues_before
    guard = compare_rewrite(original, rewritten, facts)
    if not guard.ok:
        return False, "facts_changed:" + guard.reasons[0][:90], issues_before
    after = len(detect(rewritten, LANGUAGES[language]))
    if require_improvement and after >= issues_before:
        return False, "not_better", after
    if not require_improvement and after > issues_before:
        return False, "adds_issues", after
    return True, "ok", after


def _sample_clean() -> bool:
    rate = settings.flags.humanizer_clean_sample_rate
    return rate >= 1.0 or (rate > 0.0 and random.random() < rate)


def skip_reason(state: ChatbotState, issues: int, language: str | None) -> str | None:
    if state.get("generation_failed") or not (state.get("response") or "").strip():
        return "no_answer"
    if language not in LANGUAGES:
        return "language_not_supported"
    if state.get("intent") in ("smalltalk", "off_topic", "clarification", "safety_gated") \
            or state.get("answer_depth") == "simple":
        return "simple_or_nonastro"
    v = state.get("validation") or {}
    if str(v.get("action", "")).endswith("safe_fallback") or v.get("language_gate") == "fallback":
        return "safe_reply"
    if str(state.get("model_used") or "").startswith(("rule:", "safety-gate")):
        return "deterministic_reply"
    if issues < settings.flags.humanizer_min_issues:
        return "already_natural"
    return None


def _compare(*, question: str, original: str, rewritten: str, language: str, request_id: str, session_id: str,
             user_id: str) -> dict[str, Any]:
    """Blind A/B of the rewrite against the original by the independent judge. Never raises. The
    order is randomised so the judge's position bias does not favour either side."""
    swap = random.random() < 0.5
    a, b = (rewritten, original) if swap else (original, rewritten)
    out: dict[str, Any] = {"prompt_version": COMPARE_PROMPT_VERSION, "rewrite_was": "A" if swap else "B"}
    try:
        with llm_purpose("humanize_compare", COMPARE_PROMPT_VERSION):
            raw = get_llm(ROLE_HUMANIZE_COMPARE).complete(
                compare_messages(question, a, b, language), session_id=session_id, user_id=user_id,
                request_id=request_id).text
    except LLMNotConfigured:
        return {**out, "status": "UNAVAILABLE", "reason": "not_configured"}
    except Exception as exc:  # noqa: BLE001 — advisory
        info = getattr(exc, "info", None)
        return {**out, "status": "UNAVAILABLE", "reason": info.label() if info else type(exc).__name__}
    verdict = parse_comparison(raw)
    if verdict is None:
        return {**out, "status": "UNAVAILABLE", "reason": "unparseable_output"}
    orig_side, new_side = ("B", "A") if swap else ("A", "B")
    preferred = verdict["preferred"]
    out.update(status="JUDGED", original={d: verdict[orig_side][d] for d in DIMENSIONS},
               rewrite={d: verdict[new_side][d] for d in DIMENSIONS},
               facts_preserved_by_judge=verdict["facts_preserved"],
               preferred="tie" if preferred == "tie" else ("rewrite" if preferred == new_side else "original"))
    out["delta"] = {d: round(out["rewrite"][d] - out["original"][d], 2) for d in DIMENSIONS}
    return out


def _attempt(*, mode: str, text: str, language: str, issues: int, facts: ChartFacts, style_id: str | None,
             request_id: str, session_id: str, user_id: str, question: str = "",
             clean_sample: bool = False) -> tuple[dict[str, Any], str]:
    """One humanizer call and the decision on its result. Never raises.
    Returns (report, rewritten_text); the text is only meaningful when status is applied."""
    started = time.perf_counter()
    report: dict[str, Any] = {"mode": mode, "issues_before": issues, "prompt_version": HUMANIZER_PROMPT_VERSION}
    rewritten = ""
    try:
        with llm_purpose("humanize", HUMANIZER_PROMPT_VERSION):
            rewritten = _clean(get_llm(ROLE_HUMANIZE).complete(
                humanizer_messages(text, language), session_id=session_id, user_id=user_id,
                request_id=request_id).text)
    except LLMNotConfigured:
        report.update(status="skipped", reason="not_configured")
    except Exception as exc:  # noqa: BLE001 — a failed polish never costs the user their answer
        info = getattr(exc, "info", None)
        report.update(status="failed", reason=info.label() if info else type(exc).__name__)
    if "status" not in report:
        ok, reason, after = check_rewrite(text, rewritten, facts=facts, style_id=style_id, language=language,
                                          issues_before=issues, require_improvement=not clean_sample)
        report.update(issues_after=after, reason=reason)
        if clean_sample:
            # A reply with no known issue: the question is whether the rewrite is BETTER, which only the
            # independent judge can say. A rewrite that fails the fact guard is recorded and not judged.
            report["sample"] = "clean"
            report["changed"] = rewritten.strip() != text.strip()
            if not ok:
                report["status"] = "discarded"
            elif not report["changed"]:
                report["status"] = "shadow_clean_unchanged"
            else:
                report["comparison"] = _compare(question=question, original=text, rewritten=rewritten,
                                                language=language, request_id=request_id, session_id=session_id,
                                                user_id=user_id)
                report["status"] = "shadow_clean_compared"
        else:
            report["status"] = "discarded" if not ok else "shadow_would_apply" if mode == "shadow" else "applied"
    report["ms"] = int((time.perf_counter() - started) * 1000)
    return report, rewritten


def _record_shadow(request_id: str, report: dict[str, Any], language: str) -> None:
    """Shadow outcome, ids and counts only (no reply text), to logs/humanizer_shadow.jsonl."""
    mine = [c for c in usage_tracker.for_request(request_id) if c.purpose == "humanize"]
    judge = [c for c in usage_tracker.for_request(request_id) if c.purpose == "humanize_compare"]
    line = {"ts": time.time(), "request_id": request_id, "language": language, **report,
            "input_tokens": sum(c.input_tokens or 0 for c in mine), "output_tokens": sum(c.output_tokens or 0 for c in mine),
            "judge_input_tokens": sum(c.input_tokens or 0 for c in judge),
            "judge_output_tokens": sum(c.output_tokens or 0 for c in judge)}
    log.info("humanizer shadow", extra={k: line.get(k) for k in ("status", "reason", "issues_before", "issues_after", "ms")})
    try:
        path = SHADOW_LOG if SHADOW_LOG.is_absolute() else BASE_DIR / SHADOW_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError:
        pass


def humanize_node(state: ChatbotState) -> dict[str, Any]:
    mode = settings.flags.humanizer_mode
    if mode == "off":
        return {}
    text = state.get("response") or ""
    language = (state.get("language") or {}).get("language")
    issues = len(detect(text, LANGUAGES[language])) if language in LANGUAGES and text else 0
    validation = dict(state.get("validation") or {})
    why = skip_reason(state, issues, language)
    clean_sample = False
    if why == "already_natural" and mode == "shadow" and _sample_clean():
        why, clean_sample = None, True                  # measure the humanizer on a reply with no known issue
    if why is not None:
        validation["humanizer"] = {"status": "skipped", "reason": why}
        return {"validation": validation}

    kundli = state.get("kundli")
    kwargs = dict(mode=mode, text=text, language=language, issues=issues,
                  facts=build_chart_facts(kundli) if kundli is not None else ChartFacts(available=False),
                  style_id=state.get("response_style"), request_id=state["request_id"],
                  session_id=state["session_id"], user_id=state["user_id"], question=state.get("question") or "",
                  clean_sample=clean_sample)

    if mode == "shadow":
        # Shadow never touches the reply, so it never waits for the model: the call runs
        # in the background after delivery and only its outcome is recorded.
        def work() -> None:
            report, _ = _attempt(**kwargs)
            _record_shadow(kwargs["request_id"], report, language)

        if settings.flags.evaluation_async:
            from app.observability import background

            ok = background.try_submit(work)
            validation["humanizer"] = ({"status": "shadow_pending", "mode": mode} if ok
                                       else {"status": "skipped", "reason": "queue_full"})
        else:
            work()
            validation["humanizer"] = {"status": "shadow_recorded", "mode": mode}
        return {"validation": validation}

    report, rewritten = _attempt(**kwargs)
    validation["humanizer"] = report
    log.info("humanizer", extra={k: report.get(k) for k in ("status", "reason", "issues_before", "issues_after", "ms")})
    if report["status"] == "applied":
        return {"response": rewritten, "validation": validation}
    return {"validation": validation}


__all__ = ["humanize_node", "check_rewrite", "skip_reason"]
