"""Intent classification.

An LLM classifier with a deterministic keyword fallback. The fallback is not
decoration: it keeps the pipeline working when the classifier model is down,
rate-limited, or returns unparseable JSON, and it makes the classifier
testable without a network call.

The node is kept deliberately separate from generation so a dedicated safety
or moderation classifier can be slotted in beside it later without touching
the generation path.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.llm.llm_factory import ROLE_CLASSIFY, get_llm
from app.logging_setup import get_logger
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.classifier")

INTENTS = (
    "career",
    "love",
    "relationship",
    "marriage",
    "finance",
    "health",
    "personal_growth",
    "general_astrology",
    "smalltalk",
)

# Ordered most- to least- specific: `marriage` must win over `relationship`
# for "when will I get married", so it is checked first.
KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("marriage", ("marriage", "married", "marry", "wedding", "spouse", "husband", "wife",
                  "shaadi", "kundli matching", "guna milan", "manglik", "matchmaking")),
    ("career", ("career", "job", "work", "profession", "promotion", "boss", "office",
                "business", "startup", "interview", "salary hike", "resign", "switch",
                "employment", "appraisal", "company", "developer", "engineer")),
    ("finance", ("money", "finance", "financial", "wealth", "income", "savings", "invest",
                 "investment", "loan", "debt", "property", "stock", "profit", "loss",
                 "salary", "rich", "earning")),
    ("health", ("health", "illness", "disease", "sick", "surgery", "hospital", "anxiety",
                "depression", "stress", "sleep", "energy", "body", "medical", "doctor")),
    ("love", ("love", "crush", "romance", "romantic", "dating", "girlfriend", "boyfriend",
              "attraction", "breakup", "ex ")),
    ("relationship", ("relationship", "partner", "compatibility", "fight", "argument",
                      "family", "parents", "friend", "trust")),
    ("personal_growth", ("purpose", "growth", "myself", "personality", "confidence",
                         "spiritual", "meditation", "life path", "who am i", "improve",
                         "direction in life", "meaning")),
    ("general_astrology", ("dasha", "mahadasha", "antardasha", "nakshatra", "rashi", "sign",
                           "house", "planet", "transit", "yoga", "dosha", "sade sati",
                           "ascendant", "lagna", "navamsa", "kundli", "chart", "horoscope",
                           "astrology", "retrograde", "saturn", "jupiter", "venus", "mars",
                           "mercury", "rahu", "ketu")),
    ("smalltalk", ("hello", "hi ", "hey", "namaste", "thanks", "thank you", "good morning",
                   "good evening", "how are you", "bye", "ok", "okay")),
)


class IntentResult(BaseModel):
    """Validated classifier output — the boundary where LLM JSON enters."""

    intent: str = Field(default="general_astrology")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("intent")
    @classmethod
    def known_intent(cls, value: str) -> str:
        normalised = value.strip().lower().replace(" ", "_").replace("-", "_")
        if normalised not in INTENTS:
            raise ValueError(f"unknown intent {value!r}")
        return normalised


CLASSIFIER_SYSTEM_PROMPT = f"""You classify a user's message to a Vedic astrology assistant into exactly one intent.

Intents: {", ".join(INTENTS)}

Rules:
- "marriage" for marriage, spouse and matchmaking; "love" for romance and dating; "relationship" for existing partnerships, family and friends.
- "general_astrology" when they ask how astrology itself works, or about a chart feature, rather than about their own life situation.
- "smalltalk" for greetings, thanks and chit-chat carrying no question.
- Use the recent conversation to resolve a short follow-up such as "what about next year?" — it usually carries the previous intent.

Reply with JSON only: {{"intent": "<one of the above>", "confidence": <0.0-1.0>}}"""


def classify_by_keywords(question: str) -> IntentResult:
    """Deterministic fallback. Also used directly by the tests."""
    text = f" {question.lower().strip()} "
    for intent, markers in KEYWORDS:
        if any(marker in text for marker in markers):
            return IntentResult(intent=intent, confidence=0.55)
    return IntentResult(intent="general_astrology", confidence=0.3)


def _parse(raw: str) -> IntentResult | None:
    """Tolerate a model that wraps its JSON in prose or a code fence."""
    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    braces = re.search(r"\{.*\}", raw, re.DOTALL)
    if braces:
        candidates.append(braces.group(0))

    for candidate in candidates:
        try:
            return IntentResult.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError):
            continue
    return None


def classify(question: str, history_text: str = "", *, session_id: str = "-", user_id: str = "-",
             request_id: str | None = None) -> tuple[IntentResult, str]:
    """Returns (result, method) where method is "llm" or "keyword-fallback"."""
    user_content = question if not history_text else f"Recent conversation:\n{history_text}\n\nLatest message: {question}"
    try:
        llm = get_llm(role=ROLE_CLASSIFY)
        result = llm.complete(
            [
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            session_id=session_id,
            user_id=user_id,
            request_id=request_id,
        )
    except Exception as exc:  # noqa: BLE001 — classification must never end a turn
        log.warning(
            "classifier llm unavailable, using keywords",
            extra={"error_type": type(exc).__name__, "error_message": str(exc)[:300]},
        )
        return classify_by_keywords(question), "keyword-fallback"

    parsed = _parse(result.text)
    if parsed is None:
        log.warning("classifier returned unparseable output, using keywords",
                    extra={"raw": result.text[:200]})
        return classify_by_keywords(question), "keyword-fallback"
    return parsed, "llm"


def classify_intent_node(state: ChatbotState) -> dict[str, Any]:
    history = state.get("history") or []
    history_text = "\n".join(f"{m.role}: {m.content}" for m in history[-4:])

    result, method = classify(
        state["question"],
        history_text,
        session_id=state["session_id"],
        user_id=state["user_id"],
        request_id=state["request_id"],
    )

    log.info(
        "intent classified",
        extra={"intent": result.intent, "confidence": result.confidence, "method": method},
    )
    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "intent_method": method,
    }


__all__ = ["classify_intent_node", "classify", "classify_by_keywords", "IntentResult", "INTENTS"]
