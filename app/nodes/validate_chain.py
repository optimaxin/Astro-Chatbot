"""Judging a candidate answer, and the Bolt step of the factual-failure chain.

When an answer fails FACTUAL validation the order is fixed:

    1. one repair by the primary model          (validate.py)
    2. if that is still wrong: one fresh answer from the fallback model (Bolt)
    3. if that is wrong too: the localized safe reply

Every candidate is judged by the same `check_candidate`, against the same
engine facts. Nothing is retried until it passes, and no step can ship text
that was not judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.evaluation.chart_claims import ValidationResult, validate_reply
from app.evaluation.fact_claims import HARD_KINDS
from app.evaluation.rewrite_guard import check_repair
from app.kundli.chart_facts import ChartFacts
from app.llm.llm_factory import ROLE_MAIN, fallback_llm
from app.logging_setup import get_logger
from app.prompts.language import leaks_prompt, reply_matches

log = get_logger("nodes.validate_chain")


def without_dosha(result: ValidationResult) -> ValidationResult:
    """Dosha verdicts the engine cannot back are removed by the dosha policy,
    not repaired: they must not trigger a repair call or fail a candidate."""
    return ValidationResult(claims=[c for c in result.claims if c.kind != "dosha_verdict"
                                    and not (c.kind == "dosha" and c.truth == "not computed")])


def factual_failure(result: ValidationResult) -> bool:
    """A contradicted claim, or an unconfirmable placement/lord/Dasha/Sade Sati claim."""
    return bool(result.contradictions) or any(c.kind in HARD_KINDS for c in result.unsupported)


@dataclass
class Candidate:
    text: str
    ok: bool
    factual: bool = False                      # failed on chart facts (as opposed to language only)
    reasons: list[str] = field(default_factory=list)


def check_candidate(style_id: str | None, text: str | None, facts: ChartFacts, draft: str | None = None) -> Candidate:
    """Is `text` fit to ship? Chart facts, prompt leakage and reply language;
    when it replaces `draft`, also: no risky claim the draft did not make."""
    if not text or not text.strip():
        return Candidate("", False, reasons=["empty"])
    reasons: list[str] = []
    factual = False
    if leaks_prompt(text):
        reasons.append("recites internal instructions")
    result = without_dosha(validate_reply(text, facts))
    if factual_failure(result):
        factual = True
        reasons += [c.describe() for c in [*result.contradictions, *[u for u in result.unsupported
                                                                      if u.kind in HARD_KINDS]]]
    if draft is not None:
        verdict = check_repair(draft, text, facts)
        if not verdict.ok:
            factual = True
            reasons += verdict.reasons
    if not reply_matches(style_id, text):
        reasons.append("wrong language")
    return Candidate(text, not reasons, factual, reasons)


def _bolt_messages(state: dict[str, Any], reasons: list[str], language_name: str) -> list[dict[str, Any]]:
    """The original prompt, plus what was wrong. The rejected text itself is not
    passed on, so its claims cannot be repeated."""
    note = ("[Automatic check, not written by the user: an earlier answer to this message was rejected because "
            + "; ".join(reasons[:4]) + ". Write your own answer using only the USER KUNDLI block. State a "
            "placement, house, sign, lord, Dasha or date only if it is listed there; if something is not listed, "
            f"say it is not available. Do not give a verdict on any dosha. Reply in {language_name}.]")
    messages = [dict(m) for m in state.get("messages") or []]
    if messages and messages[-1].get("role") == "user" and isinstance(messages[-1].get("content"), str):
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{note}"
    else:
        messages.append({"role": "user", "content": note})
    return messages


def bolt_answer(state: dict[str, Any], reasons: list[str]) -> str | None:
    """One answer from the fallback model, or None if it cannot be reached."""
    from app.nodes.repair_prompts import _LANGUAGE_NAME

    llm = fallback_llm(ROLE_MAIN)
    if llm is None or not state.get("messages"):
        return None
    from app.llm.usage_tracker import llm_purpose
    from app.prompts.builder import PROMPT_VERSION

    language = _LANGUAGE_NAME.get(state.get("response_style") or "", "the user's language")
    try:
        with llm_purpose("validation_fallback", f"{PROMPT_VERSION}+bolt"):
            return llm.complete(_bolt_messages(state, reasons, language), session_id=state["session_id"],
                                user_id=state["user_id"], request_id=state["request_id"]).text
    except Exception as exc:  # noqa: BLE001 — the chain ends in the safe reply
        log.warning("fallback-model answer failed", extra={"error_type": type(exc).__name__})
        return None


def resolve_after_repair(state: dict[str, Any], draft: str, final: str, facts: ChartFacts, mode: str,
                         action: str, safe_reply) -> tuple[str, str, dict[str, Any]]:
    """What ships once the primary repair (if any) is done. Returns
    (text, action, notes).

    repair/auto: a still-wrong answer goes to Bolt for one fresh answer; if
    Bolt's is wrong too, the localized safe reply. strip: the wrong sentences
    are dropped. log: nothing changes. Language-only problems are not handled
    here: they keep the language gate's own single rewrite."""
    from app.evaluation.chart_claims import split_sentences, strip_sentences
    from app.prompts.language import has_placeholder

    style_id = state.get("response_style")
    notes: dict[str, Any] = {}
    if mode == "log":
        return final, action, notes
    if mode in ("repair", "auto"):
        verdict = check_candidate(style_id, final, facts, draft if final != draft else None)
        if verdict.factual:
            notes["primary_rejected"] = verdict.reasons[:4]
            bolt = bolt_answer(state, verdict.reasons)
            second = check_candidate(style_id, bolt, facts)
            if second.ok:
                final, action = bolt or "", f"{action}+bolt" if action != "none" else "bolt"
            else:
                notes["bolt_rejected"] = second.reasons[:4] if bolt else ["unavailable"]
                final, action = safe_reply(style_id), f"{action}+safe_fallback" if action != "none" else "safe_fallback"
    else:
        after = without_dosha(validate_reply(final, facts))
        drop = {c.sentence_index for c in after.contradictions}
        drop |= {c.sentence_index for c in after.unsupported if c.kind in HARD_KINDS}
        if drop:
            stripped = strip_sentences(final, drop)
            final = stripped if len(stripped) >= 25 else safe_reply(style_id)
            action = f"{action}+stripped" if action != "none" else "stripped"
    # A template slot that survived is removed with its sentence.
    slots = {i for i, s in enumerate(split_sentences(final)) if has_placeholder(s)}
    if slots:
        stripped = strip_sentences(final, slots)
        final = stripped if len(stripped) >= 25 else safe_reply(style_id)
        action = f"{action}+stripped" if action != "none" else "stripped"
    return final, action, notes


__all__ = ["Candidate", "check_candidate", "factual_failure", "without_dosha", "bolt_answer",
           "resolve_after_repair"]
