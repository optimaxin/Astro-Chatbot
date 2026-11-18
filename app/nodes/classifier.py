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
    # Not about the user's chart or life at all (phone repair, cricket, trivia). Only the LLM
    # classifier assigns it; the pre-LLM rules in app/safety/offtopic_rules.py catch the
    # unambiguous cases first.
    "off_topic",
)
ROUTING = ("answer", "clarify", "scope_reply")

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
    """Validated classifier output — the boundary where LLM JSON enters.

    Beyond `intent`/`confidence` the classifier reports whether the message is about
    astrology at all, whether it continues the conversation, the language it is written
    in (advisory: the reply language is decided by the deterministic policy in
    app.prompts.language_policy) and what to do with it. Every extra field is optional
    with a safe default, so a model that returns only intent/confidence still validates.
    """

    intent: str = Field(default="general_astrology")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    sub_intent: str | None = None
    is_astrology_related: bool = True
    is_follow_up: bool = False
    language: str | None = None
    routing_decision: str = "answer"

    @field_validator("intent")
    @classmethod
    def known_intent(cls, value: str) -> str:
        normalised = value.strip().lower().replace(" ", "_").replace("-", "_")
        if normalised not in INTENTS:
            raise ValueError(f"unknown intent {value!r}")
        return normalised

    @field_validator("routing_decision")
    @classmethod
    def known_routing(cls, value: str) -> str:
        normalised = str(value).strip().lower().replace(" ", "_")
        return normalised if normalised in ROUTING else "answer"

    @field_validator("sub_intent", "language")
    @classmethod
    def short_text(cls, value: str | None) -> str | None:
        return (str(value).strip()[:40] or None) if value else None


CLASSIFIER_SYSTEM_PROMPT = f"""You classify a user's message to a Vedic astrology assistant. You never answer the message and never state anything about a chart.

Intents: {", ".join(INTENTS)}

Rules:
- "marriage" for marriage, spouse and matchmaking; "love" for romance and dating; "relationship" for existing partnerships, family and friends.
- "general_astrology" when they ask how astrology itself works, or about a chart feature, rather than about their own life situation.
- "smalltalk" for greetings, thanks and chit-chat carrying no question.
- "off_topic" for a message that is not about astrology or the user's own life at all: phone or computer help, sports scores, general trivia, recipes, entertainment, shopping. A question about the user's own career, money, health, marriage or future is NEVER off_topic, even when it mentions a phone, a match, a score or a house.
- Use the recent conversation to resolve a short follow-up such as "what about next year?": it keeps the previous intent and is_follow_up is true.
- If you cannot tell what the message is about, use routing_decision "clarify".

Reply with JSON only:
{{"intent": "<one of the above>", "confidence": <0.0-1.0>, "sub_intent": "<2-3 word topic or null>", "is_astrology_related": <true|false>, "is_follow_up": <true|false>, "language": "<english|hindi|hinglish|other>", "routing_decision": "<answer|clarify|scope_reply>"}}
Use "scope_reply" only with intent "off_topic"."""


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


# Words that appear in KEYWORDS but are too common outside their intent to be
# trusted on their own ("will it work out", "my energy", "a sign from God").
_AMBIGUOUS_MARKERS = frozenset({"work", "switch", "company", "office", "energy", "body", "sign",
                                "house", "chart", "loss", "profit", "ok", "okay", "trust",
                                "meaning", "improve", "myself", "stress", "sleep", "rich"})
_FOLLOW_UP = re.compile(r"\b(?:what about|and|also|then|uske|iske|aur|phir|next year|that|this|it)\b", re.I)


def classify_unambiguous(question: str, has_history: bool) -> IntentResult | None:
    """Keyword classification, only when it cannot reasonably be wrong.

    Whole-word matching (unlike `classify_by_keywords`, whose substring test
    lets "ok" fire inside "look"). Returns None — meaning "ask the LLM" — when
    no intent or more than one intent matches, or when a short message in an
    ongoing conversation may be a follow-up whose intent lives in history.
    """
    text = question.lower()
    matched: list[str] = []
    for intent, markers in KEYWORDS:
        markers = tuple(m for m in markers if m.strip() not in _AMBIGUOUS_MARKERS)
        if any(re.search(rf"(?<![a-z]){re.escape(m.strip())}(?![a-z])", text) for m in markers):
            matched.append(intent)
    # "marriage" is listed before "relationship"/"love" precisely because it
    # is the more specific reading of the same words; treat that pair as one.
    if set(matched) <= {"marriage", "relationship", "love"} and "marriage" in matched:
        matched = ["marriage"]
    # Chart vocabulary ("dasha", "house") co-occurring with a life area does
    # not make the question ambiguous: the life area is the intent.
    if len(matched) == 2 and "general_astrology" in matched:
        matched = [m for m in matched if m != "general_astrology"]
    if len(matched) != 1:
        return None
    words = len(re.findall(r"\w+", text))
    if has_history and words <= 6 and _FOLLOW_UP.search(text) and matched[0] != "smalltalk":
        return None
    return IntentResult(intent=matched[0], confidence=0.8)


def classify(question: str, history_text: str = "", *, session_id: str = "-", user_id: str = "-",
             request_id: str | None = None) -> tuple[IntentResult, str]:
    """Returns (result, method): "llm", "keyword" or "keyword-fallback"."""
    from app.config.settings import settings

    mode = settings.flags.classifier_mode
    if mode in {"hybrid", "keyword"}:
        quick = classify_unambiguous(question, has_history=bool(history_text))
        if quick is not None:
            return quick, "keyword"
        if mode == "keyword":
            return classify_by_keywords(question), "keyword-fallback"

    user_content = question if not history_text else f"Recent conversation:\n{history_text}\n\nLatest message: {question}"
    from app.llm.usage_tracker import llm_purpose, prompt_version_of

    try:
        llm = get_llm(role=ROLE_CLASSIFY)
        with llm_purpose("classify", prompt_version_of(CLASSIFIER_SYSTEM_PROMPT)):
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


OFFTOPIC_MIN_CONFIDENCE = 0.7      # below this an off_topic label is asked about, not acted on
CLARIFY_BELOW = 0.35               # an LLM verdict this unsure is asked about, not answered


def decide_routing(result: IntentResult, method: str, has_history: bool) -> str:
    """answer | clarify | scope_reply, from a validated classification.

    Off-topic goes to the existing localized scope reply WITHOUT an astrology model call,
    unless it is a follow-up inside a live conversation, or the model is not sure (then the
    user is asked, not blocked). Only an LLM verdict can ask for clarification: the keyword
    fallback's low confidence just means "no keyword matched", and answering is the safe
    behaviour there."""
    if result.intent == "off_topic":
        if result.is_follow_up and has_history:
            return "answer"
        return "scope_reply" if result.confidence >= OFFTOPIC_MIN_CONFIDENCE else "clarify"
    if method == "llm" and not has_history and (result.routing_decision == "clarify"
                                                 or result.confidence < CLARIFY_BELOW):
        return "clarify"
    return "answer"


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
    routing = decide_routing(result, method, bool(history))

    log.info(
        "intent classified",
        extra={"intent": result.intent, "confidence": result.confidence, "method": method, "routing": routing},
    )
    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "intent_method": method,
        "routing_decision": routing,
        "classification": {
            "intent": result.intent, "sub_intent": result.sub_intent, "confidence": result.confidence,
            "is_astrology_related": result.intent not in ("off_topic", "smalltalk") and result.is_astrology_related,
            "is_follow_up": result.is_follow_up, "language": result.language,
            "routing_decision": routing, "method": method,
        },
    }


def route_after_classify(state: ChatbotState) -> str:
    return {"scope_reply": "scope_reply", "clarify": "clarify"}.get(state.get("routing_decision") or "",
                                                                    "retrieve_knowledge")


def scope_reply_node(state: ChatbotState) -> dict[str, Any]:
    """An off-topic message gets the existing localized scope reply, with no model call."""
    from app.safety.block_messages import block_message, resolve_lang
    from app.safety.categories import SafetyCategory

    style = (state.get("language") or {}).get("style") or state.get("response_style")
    log.info("off-topic message; scope reply", extra={"session_id": state.get("session_id")})
    return {
        "response": block_message(SafetyCategory.out_of_scope, resolve_lang(style)),
        "model_used": "rule:scope",
        "fallback_used": False,
        "generation_failed": False,
        "safety": {"blocked": True, "category": SafetyCategory.out_of_scope.value, "rule": "classifier_off_topic"},
    }


__all__ = ["classify_intent_node", "classify", "classify_by_keywords", "IntentResult", "INTENTS", "ROUTING",
           "decide_routing", "route_after_classify", "scope_reply_node"]
