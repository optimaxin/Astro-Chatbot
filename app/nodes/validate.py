"""Validate the draft reply before anyone sees it — synchronously, every turn.

Two deterministic checks run on the generated draft:

1. **Chart claims** (`app.evaluation.chart_claims`) against the Kundli
   engine's facts. A contradicted claim must not reach the user.
2. **Script** (`app.prompts.language`): the reply is in the script the user
   wrote in (e.g. not Devanagari for a Roman-script Hinglish question).

What happens on a failure is `CHART_VALIDATION_MODE`:

* `repair` (default) — ONE corrective regeneration with the main model, told
  exactly what was wrong and what the chart says. The rewrite is validated
  again; any contradiction that survives is stripped sentence by sentence.
* `strip`  — contradicted sentences are removed; no extra model call.
* `log`    — nothing is changed; the result is recorded for evaluation.

Validation itself has no "off" switch. Unsupported (not contradicted)
claims are rewritten only in repair mode, and never stripped: they are not
known to be false, only not verifiable.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import settings
from app.evaluation.chart_claims import (
    ValidationResult,
    split_sentences,
    strip_sentences,
    validate_reply,
)
from app.kundli.chart_facts import build_chart_facts
from app.llm.llm_factory import ROLE_MAIN, get_llm
from app.logging_setup import get_logger
from app.observability.trace import span
from app.prompts.language import deflection, has_placeholder, leaks_prompt, reply_matches
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.validate")

MIN_KEEP_CHARS = 25
# Shown only if repair and stripping together leave nothing usable.
SAFE_FALLBACK = ("I want to be precise about your chart, and I couldn't phrase that answer reliably "
                 "just now. Could you ask it once more?")


def _correction_note(result: ValidationResult, fix_language: bool, style_id: str | None,
                     fix_unsupported: bool, fix_placeholder: bool = False) -> str:
    lines = ["CORRECTION REQUIRED (an automatic check wrote this, not the user). Your previous reply has problems:"]
    for claim in result.contradictions:
        lines.append(f"- It {claim.describe()}. Use the chart value.")
    if fix_unsupported:
        for claim in result.unsupported:
            lines.append(f"- It asserts something the chart data does not contain ({claim.kind}: "
                         f"{claim.subject} -> {claim.value}). Remove it or mark it clearly as general astrology.")
    if fix_placeholder:
        lines.append("- It contains an unfilled template slot in <angle brackets>. Replace it with the real "
                     "chart fact from the USER KUNDLI block, or remove it.")
    if fix_language:
        lines.append("- It is in the wrong language or script.")
    # Always restate the language: a repair once fixed a placement and drifted
    # from Hinglish into English.
    lines.append("Required language: " + _LANGUAGE_NAME.get(style_id or "", "the user's language") + ".")
    lines.append("Rewrite the reply fixing ONLY these problems. Keep the same meaning, tone and length, "
                 "and use only facts from the USER KUNDLI block. Output only the reply.")
    return "\n".join(lines)


_LANGUAGE_NAME = {
    "whatsapp_hinglish": "Roman-script Hinglish (Hindi and English mixed, Latin letters only, no Devanagari)",
    "hindi": "Hindi in Devanagari script",
    "english": "English", "formal": "English", "detailed": "English",
}


def validate_reply_node(state: ChatbotState) -> dict[str, Any]:
    if state.get("generation_failed"):
        return {}
    draft = state.get("response") or ""
    style_id = state.get("response_style")

    # Prompt/instruction leakage is never repaired by asking the same model
    # again — the reply is replaced outright with a polite deflection.
    if leaks_prompt(draft):
        log.warning("reply recited internal instructions; replaced")
        return {"response": deflection(style_id),
                "validation": {"mode": settings.flags.chart_validation_mode, "action": "prompt_leak_blocked"}}
    mode = settings.flags.chart_validation_mode
    facts = build_chart_facts(state["kundli"]) if state.get("kundli") is not None else None

    with span("validate.chart_claims"):
        result = validate_reply(draft, facts) if facts else ValidationResult()
    language_ok = reply_matches(style_id, draft)
    report: dict[str, Any] = {"mode": mode, "draft": result.to_dict(), "language_ok": language_ok,
                              "action": "none"}

    fix_language = settings.flags.language_guard and not language_ok
    fix_unsupported = mode == "repair" and bool(result.unsupported)
    fix_placeholder = has_placeholder(draft)
    report["placeholder"] = fix_placeholder
    needs_fix = bool(result.contradictions) or fix_language or fix_unsupported or fix_placeholder
    if not needs_fix or mode == "log" and not (fix_language or fix_placeholder):
        return {"validation": report}

    final = draft
    if mode == "repair" or fix_language or fix_placeholder:
        note = _correction_note(result, fix_language, style_id,
                                fix_unsupported and mode == "repair", fix_placeholder)
        messages = [*(state.get("messages") or []), {"role": "assistant", "content": draft},
                    {"role": "user", "content": note}]
        try:
            with span("validate.repair_llm"):
                # The main model (not the light route) does repairs: the light
                # model is the one most likely to have produced the problem.
                repaired = get_llm(role=ROLE_MAIN).complete(
                    messages, session_id=state["session_id"], user_id=state["user_id"],
                    request_id=state["request_id"]).text
            final = repaired
            report["action"] = "regenerated"
        except Exception as exc:  # noqa: BLE001 — fall through to stripping
            log.warning("repair regeneration failed", extra={"error_type": type(exc).__name__})
            report["repair_error"] = type(exc).__name__

    if mode != "log" and facts:
        after = validate_reply(final, facts)
        report["after"] = after.to_dict()
        drop = {c.sentence_index for c in after.contradictions}
        # A template slot that survived the repair is removed with its sentence.
        drop |= {i for i, s in enumerate(split_sentences(final)) if has_placeholder(s)}
        if drop:
            stripped = strip_sentences(final, drop)
            final = stripped if len(stripped) >= MIN_KEEP_CHARS else SAFE_FALLBACK
            report["action"] = f"{report['action']}+stripped" if report["action"] != "none" else "stripped"
    report["language_ok_after"] = reply_matches(style_id, final)

    log.warning("reply validation intervened", extra={
        "mode": mode, "action": report["action"],
        "contradictions": len(result.contradictions), "unsupported": len(result.unsupported),
        "language_ok": language_ok,
        "remaining_contradictions": len((report.get("after") or {}).get("contradictions") or []),
    })
    return {"response": final, "validation": report}


__all__ = ["validate_reply_node", "SAFE_FALLBACK"]
