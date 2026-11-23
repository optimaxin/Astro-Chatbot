"""Turn-level gate that needs the conversation: crisis context, then closings.

Runs right after load_context, so it can see history; the per-message safety
gate (nodes/safety.py) runs before that and cannot.

1. Crisis context (audit 2026-11-20). After "I'm going to kill myself" or
   "maine zeher kha liya", the next message — "I already died", "bye", "too
   late", "when will I die?", "ok" — was judged on its own words and answered
   with astrology. While a self-harm, poisoning or violence disclosure is
   live, EVERY message keeps the safety flow until the user explicitly
   resolves it ("I'm safe now", "I haven't taken anything", "main theek hoon")
   — a farewell, an astrology question or a code request does not. Each held
   reply asks whether they are safe, so the way out is one sentence.
2. Closings ("bye", "thank you", "dhanyawad", "bas itna hi") get one short
   localized line with no model call (app/prompts/closing.py). Safety is
   always checked first, so "bye, I am going to die" is never a goodbye.
"""

from __future__ import annotations

import re
from typing import Any

from app.logging_setup import get_logger
from app.safety.block_messages import block_message, resolve_lang
from app.safety.categories import SafetyCategory
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.conversation_gate")

CONTEXT_TURNS = 3

_SAFE_NOW = re.compile(
    r"\b(?:i'?m|i\s+am)\s+(?:safe|okay|ok|fine|alright|better)\b|\b(?:i\s+was|just)\s+(?:joking|kidding)\b|"
    r"\b(?:main|mai|mein)\s+(?:theek|thik|safe|thk)\s+(?:hoon|hu|hun)\b|\bmazaa?k\s+(?:tha|kar\s+rah)|"
    r"\bsab\s+theek\s+hai\b|मैं\s+(?:ठीक|सुरक्षित)\s+(?:हूँ|हूं)|"
    # answers to "are you safe, have you taken anything?" (safety regression review 2026-11-20)
    r"\b(?:yes|yeah|haan|han)\b[^.?!]{0,20}\b(?:safe|theek|thik|fine|okay)\b|\bsafe\s+(?:hoon|hu|hun)\b|"
    r"\bab\s+(?:main\s+)?(?:theek|thik|safe)\b|\bi\s+(?:have\s*n'?t|have\s+not|did\s*n'?t|did\s+not)\s+"
    r"(?:taken?|took|had|swallowed|drank|eaten|ate)\s+anything\b|\b(?:maine|mene|maine\s+to)\s+kuch\s+nahi\s+"
    r"(?:liya|khaya|piya)\b|\bi'?m\s+not\s+(?:going|planning)\s+to\s+(?:hurt|harm|kill|do\s+anything)\b|"
    r"\bi\s+(?:did\s*n'?t|do\s*n'?t)\s+(?:really\s+)?mean\s+it\b|\bi\s+was\s+(?:just\s+)?(?:venting|frustrated|angry)\b|"
    r"मैं\s+सुरक्षित|अब\s+(?:मैं\s+)?ठीक|"
    # help has arrived (a medical emergency or an injury reported earlier)
    r"\bambulance\s+(?:aa\s*gayi|a\s*gayi|pahunch\s*gayi|has\s+arrived|is\s+here|came|reached)\b|\b(?:paramedics|doctors?)\s+"
    r"(?:are\s+here|have\s+arrived|arrived|came)\b|\bhospital\s+(?:mein|me)\s+(?:hain|hai|hu|hoon)\b|\bat\s+the\s+hospital\b|"
    r"\b(?:he|she)\s+is\s+(?:okay|ok|fine|stable|safe)\s+now\b|\b(?:wo|woh|vo)\s+(?:ab\s+)?(?:theek|thik|safe)\s+hai\b", re.I)
_CONTINUATION = re.compile(
    r"\b(?:already|now)\s+(?:dead|died|gone)|\bi(?:'?m|\s+am)\s+dead\b|\bi\s+(?:died|did\s+it)\b|"
    r"\bit'?s\s+(?:done|over|too\s+late)\b|\btoo\s+late\b|\bgood\s*bye\b|\bbye\b|\balvida\b|\blast\s+(?:message|time)\b|"
    r"\bforever\b|\b(?:die|dying|death|dead|maut|marunga|marungi|mar\s+(?:gaya|gayi|chuka|chuki|jaunga|jaungi))\b|"
    r"\b(?:kha|pi|le)\s+(?:liya|li)\b|\bho\s+gaya\b|\bkhatam\b|\bnothing\s+matters\b|\bdoesn'?t\s+matter\b|"
    r"\bkuch\s+nahi\s+ho\s+sakta|\bkya\s+fayda|\bno\s+point|\bnothing\s+can\s+be\s+done|\bder\s+ho\s+(?:gayi|chuki)|"
    r"\b(?:still|will)\s+(?:do\s+it|kill)|\bkar\s+(?:dunga|dungi)\b|\bchhod(?:unga|ungi)\s+nahi\b|\bdeserves?\b|"
    r"मर\s*(?:गया|गई|चुका|चुकी|जाऊं|जाऊँ)|खा\s*लिया|अलविदा|मौत|बाय", re.I)
_WORD = re.compile(r"[\wऀ-ॿ']+")
_SAFETY_CONTEXT = (SafetyCategory.self_harm_crisis, SafetyCategory.poisoning_emergency,
                   SafetyCategory.crisis_followup, SafetyCategory.violence_risk, SafetyCategory.violence_aftermath,
                   SafetyCategory.medical_emergency)


def _flagged(text: str) -> SafetyCategory | None:
    """The safety category a single user message triggers, if any."""
    from app.safety.screen import screen_message

    result = screen_message(text)
    return result.category if result.blocked and result.category in _SAFETY_CONTEXT else None


def safety_context(history: list[Any]) -> SafetyCategory | None:
    """A self-harm / poisoning / violence disclosure among the user's last
    CONTEXT_TURNS messages, or a safety reply as the assistant's last message."""
    users = [m.content for m in history if getattr(m, "role", "") == "user"][-CONTEXT_TURNS:]
    for text in reversed(users):
        category = _flagged(text)
        if category is not None:
            return category
        if _SAFE_NOW.search(text):          # the user resolved it ("I'm safe now"): not re-triggered by what came before
            return None
    last = next((m.content for m in reversed(history) if getattr(m, "role", "") == "assistant"), "")
    for category in _SAFETY_CONTEXT:
        for lang in ("en", "hi", "hinglish"):
            if last and last.strip() == block_message(category, lang).strip():
                return category
    return None


def continues_crisis(text: str, context: SafetyCategory | None = None, previous_user: str | None = None) -> bool:
    """Does this message continue a safety flow? Only an explicit resolution ends a
    live context ("I'm safe now", "I haven't taken anything", "main theek hoon"):
    a farewell, an astrology question, a code request or silence never does — the
    context is held, and the follow-up reply asks the user whether they are safe.
    With no context given (a bare message) the old lenient reading applies: finality,
    death, an ingestion, a farewell, or a very short message continues; a
    substantive new question after a farewell/finality message is not recovery."""
    t = text or ""
    if _SAFE_NOW.search(t):
        return False
    if context is not None:
        return True
    if _CONTINUATION.search(t) or len(_WORD.findall(t)) <= 4:
        return True
    return bool(previous_user and _CONTINUATION.search(previous_user) and not _SAFE_NOW.search(previous_user))


def safety_context_update(question: str, history: list[Any], lang: str, session_id: str | None = None) -> dict[str, Any] | None:
    """The gated-turn update when this message continues a live safety context, else None.
    Used by the conversation gate and by the input gate, so a live crisis outranks
    every domain rule (injection, programming redirect, scope) as well as a farewell."""
    context = safety_context(history)
    previous_user = next((m.content for m in reversed(history) if getattr(m, "role", "") == "user"), None)
    if context is None or not continues_crisis(question, context, previous_user):
        return None
    # The follow-up repeats the reply that fits the situation: harm to others, an injured person, or
    # (for self-harm and poisoning) the "are you safe, have you taken anything" check-in.
    category = context if context in (SafetyCategory.violence_risk, SafetyCategory.violence_aftermath,
                                      SafetyCategory.medical_emergency) else SafetyCategory.crisis_followup
    log.warning("safety context continues", extra={"category": category.value, "context": context.value,
                                                   "session_id": session_id})
    return {"response": block_message(category, lang), "model_used": "safety-gate", "fallback_used": False,
            "generation_failed": False, "intent": "safety_gated", "intent_confidence": 1.0,
            "intent_method": "safety-context",
            "safety": {"blocked": True, "category": category.value, "rule": f"context_{context.value}"}}


def conversation_gate_node(state: ChatbotState) -> dict[str, Any]:
    question = state.get("question") or ""
    lang = resolve_lang(state.get("response_style"))
    held = safety_context_update(question, state.get("history") or [], lang, state.get("session_id"))
    if held is not None:
        return held

    from app.prompts.closing import closing_language, closing_reply, detect_closing

    kind = detect_closing(question)
    if kind is not None:
        reply_lang = closing_language(question, lang)
        return {"response": closing_reply(kind, question, reply_lang, seed=state.get("session_id") or ""),
                "model_used": "rule:closing", "fallback_used": False, "generation_failed": False,
                "intent": "closing", "intent_confidence": 1.0, "intent_method": "rule:closing",
                "closing": {"kind": kind, "method": "deterministic"}}
    return {}


def closing_reply_node(state: ChatbotState) -> dict[str, Any]:
    """The intent classifier judged an ambiguous message a closing: the same short
    reply as a deterministic closing, still with no main-model call."""
    from app.prompts.closing import closing_language, closing_reply, infer_closing_kind

    question = state.get("question") or ""
    kind = infer_closing_kind(question)
    lang = closing_language(question, resolve_lang(state.get("response_style")))
    return {"response": closing_reply(kind, question, lang, seed=state.get("session_id") or ""),
            "model_used": "rule:closing", "fallback_used": False, "generation_failed": False,
            "closing": {"kind": kind, "method": "classifier"}}


def route_after_conversation_gate(state: ChatbotState) -> str:
    """A turn the gate answered goes straight to persistence; otherwise the usual route."""
    if state.get("model_used") in ("safety-gate", "rule:closing"):
        return "save_metrics"
    from app.nodes.clarify import route_after_load_context

    return route_after_load_context(state)


__all__ = ["conversation_gate_node", "closing_reply_node", "route_after_conversation_gate", "safety_context",
           "safety_context_update", "continues_crisis"]
