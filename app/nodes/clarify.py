"""A follow-up with nothing to follow: ask, don't guess.

"Kab tak?", "Aur batao", "Why?" mean something only against an earlier
question. With no conversation history there is no earlier question, and
answering anyway meant inventing a topic (the 2026-11-18 live run answered a
cold "Kab tak?" from the user's current Dasha). Such a message is answered with
one fixed clarifying question, in the user's language, with no model call.

With history the message goes through the normal pipeline, which resolves the
follow-up from the conversation, exactly as before.
"""

from __future__ import annotations

import re
from typing import Any

from app.logging_setup import get_logger
from app.safety.notices import clarification
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.clarify")

MAX_WORDS = 5

# Words that carry no topic of their own: question particles, "tell me more",
# "go on", "why". A message made ONLY of these refers to something unsaid.
_FOLLOWUP_WORDS = frozenset("""
kab tak kabtak kitna kitne kitni kitna kyun kyu kyon kaise kaisa kaisi kya aur or batao bataiye btao bata
bataye do dijiye phir fir iska uska iski uski iske uske isme usme is us ye yeh woh wo ka ki ke ko mein me se
bhi aage agla agle next ahead then and why when how long until till what else more tell explain elaborate
continue go on reason matlab mtlb meaning hmm hm ok okay accha acha theek thik sahi hai hain hoga hogi honge
please pls plz bhai sir ji so but then now
कब तक कितना कितने कितनी क्यों क्यूं कैसे कैसा क्या और बताओ बताइए बताएं बताइये फिर इसका उसका इसकी उसकी इसमें
उसमें आगे अगला मतलब कारण भी से में को का की के है हैं होगा होगी
""".split())

_TOKEN = re.compile(r"[\wऀ-ॿ']+")


def is_ambiguous_followup(text: str | None) -> bool:
    words = [w.lower() for w in _TOKEN.findall(text or "")]
    return bool(words) and len(words) <= MAX_WORDS and all(w in _FOLLOWUP_WORDS for w in words)


def route_after_load_context(state: ChatbotState) -> str:
    """A bare follow-up with no earlier turn is asked about, not answered."""
    if not state.get("history") and is_ambiguous_followup(state.get("question")):
        return "clarify"
    return "classify_intent"


def clarify_node(state: ChatbotState) -> dict[str, Any]:
    log.info("cold follow-up; asking what it refers to",
             extra={"session_id": state.get("session_id"), "words": len(_TOKEN.findall(state["question"]))})
    return {
        "response": clarification(state.get("response_style")),
        "model_used": "rule:clarify",
        "fallback_used": False,
        "generation_failed": False,
        "intent": "clarification",
        "intent_confidence": 1.0,
        "intent_method": "rule:clarify",
    }


__all__ = ["clarify_node", "route_after_load_context", "is_ambiguous_followup"]
