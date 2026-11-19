"""A follow-up must answer the NEW question, not re-send the previous answer.

Live 2026-11-19: "और इस साल कैसा रहेगा?" after a career answer came back as that
career answer, word for word; "और बताइए।" came back as a 92%-same paraphrase.
Partial copies are already dropped sentence by sentence (conciseness.
repeated_from_previous). A reply that is MOSTLY the previous one gets ONE
regeneration told exactly that, re-checked against the chart and the language;
if that fails too, a short honest reply offers to go deeper instead. Never
retried more than once.
"""

from __future__ import annotations

import re
from typing import Any

from app.evaluation.chart_claims import split_sentences
from app.logging_setup import get_logger

log = get_logger("nodes.repeat_guard")

VERBATIM_SHARE = 0.5      # of the reply's content sentences copied from the previous reply
WORD_SHARE = 0.9          # of the reply's distinct words already in the previous reply

FALLBACK = {
    "english": "I've already shared what your chart shows on this above. Which part would you like me to go "
               "deeper into?",
    "hinglish": "Iske baare mein jo aapki Kundli dikhati hai, woh maine upar bata diya hai. Aap kis hisse ko aur "
                "detail mein jaanna chahenge?",
    "hindi": "इस बारे में आपकी कुंडली जो दिखाती है, वह मैंने ऊपर बता दिया है। आप किस हिस्से को और विस्तार से जानना "
             "चाहेंगे?",
}


def _key(sentence: str) -> str:
    return " ".join(re.findall(r"[\wऀ-ॿ]+", sentence.lower()))


def repeats_previous(text: str, previous: str | None) -> bool:
    if not previous or not text:
        return False
    content = [s for s in split_sentences(text) if not s.rstrip().endswith(("?", "？"))]
    if not content:
        return False
    said = {_key(s) for s in split_sentences(previous)}
    verbatim = sum(1 for s in content if _key(s) in said) / len(content)
    words = set(_key(" ".join(content)).split())
    overlap = len(words & set(_key(previous).split())) / max(1, len(words))
    return verbatim >= VERBATIM_SHARE or (len(words) >= 12 and overlap >= WORD_SHARE)


def _note(question: str) -> str:
    return ("CORRECTION REQUIRED (an automatic check wrote this, not the user). Your reply repeats your previous "
            f"answer. Answer ONLY the latest message — \"{question[:200]}\" — with what you have NOT already said, "
            "from the USER KUNDLI block. If the chart adds nothing new for it, say so in one short sentence and "
            "offer one specific thing you can explain next. Same language, same short chat-message format. Output "
            "only the reply.")


def apply_repeat_guard(state: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    text = out.get("response", state.get("response") or "")
    previous = next((m.content for m in reversed(state.get("history") or []) if getattr(m, "role", "") == "assistant"),
                    None)
    if state.get("generation_failed") or not repeats_previous(text, previous):
        return out
    from app.evaluation.chart_claims import validate_reply
    from app.kundli.chart_facts import ChartFacts, build_chart_facts
    from app.llm.llm_factory import ROLE_MAIN, get_llm
    from app.llm.usage_tracker import llm_purpose
    from app.prompts.language import leaks_prompt, reply_matches
    from app.prompts.language_policy import LANGUAGE_OF_STYLE

    style_id = state.get("response_style")
    kundli = state.get("kundli")
    facts = build_chart_facts(kundli) if kundli is not None else ChartFacts(available=False)
    validation = dict(out.get("validation") or {})
    messages = [*(state.get("messages") or []), {"role": "assistant", "content": text},
                {"role": "user", "content": _note(state.get("question") or "")}]
    fresh, reason = "", "ok"
    try:
        with llm_purpose("repeat_repair"):
            fresh = get_llm(ROLE_MAIN).complete(messages, session_id=state["session_id"], user_id=state["user_id"],
                                                request_id=state["request_id"]).text.strip()
    except Exception as exc:  # noqa: BLE001 — the honest fallback below still answers
        reason = f"failed:{type(exc).__name__}"
    if not reason.startswith("failed"):
        if not fresh or leaks_prompt(fresh):
            reason = "empty_or_leaked"
        elif repeats_previous(fresh, previous):
            reason = "still_repeats"
        elif not reply_matches(style_id, fresh):
            reason = "language"
        elif validate_reply(fresh, facts).contradictions:
            reason = "contradicts_chart"
    if reason == "ok":
        validation["repeat_guard"] = {"action": "regenerated"}
        log.info("follow-up repeated the previous answer; regenerated")
        return {**out, "response": fresh, "validation": validation}
    language = LANGUAGE_OF_STYLE.get((style_id or "").lower(), "english")
    validation["repeat_guard"] = {"action": "fallback", "reason": reason}
    log.warning("follow-up repeated the previous answer; sent the offer to go deeper", extra={"reason": reason})
    return {**out, "response": FALLBACK.get(language, FALLBACK["english"]), "validation": validation}


__all__ = ["apply_repeat_guard", "repeats_previous"]
