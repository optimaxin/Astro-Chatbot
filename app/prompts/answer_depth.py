"""How long THIS answer should be — decided in code, per turn, before generation.

Root cause it fixes (2026-11 conciseness audit): the prompt gave one size for
every message ("1–3 short sentences, ~20–40 words") and left the number of
messages to the model, which treated almost every topical question as
"detailed". Normal answers averaged 77 words (target 30–60) and bubbles 27
words (target 15–25). The model obeys an explicit per-turn budget far better
than a table of budgets it has to apply itself, so the depth is classified
here and the prompt states a single target for this turn.

The targets are soft: the prompt says never to drop an essential chart fact to
meet them. The segmenter (app.prompts.segments) still guarantees integrity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SIMPLE, NORMAL, DETAILED, COMPLEX = "simple", "normal", "detailed", "complex"

# Broad life-area questions: answered from several chart factors.
_BROAD_INTENTS = {"career", "marriage", "finance", "relationship", "love"}

# The user explicitly asks for depth or a full analysis.
_DEPTH_REQUEST = re.compile(
    r"\b(?:in\s+detail|detailed|details|detail\s+(?:mein|me|se)|complete|full|thorough|in\s+depth|"
    r"analysis|analyse|analyze|poora|pura|puri|poori|vistaar|vistar|sab\s+kuch)\b"
    r"|विस्तार|विश्लेषण|पूरा|पूरी|सब\s+कुछ",
    re.I)

# One plain fact: a sign, a house, a lord. Answer it and stop.
_FACT_LOOKUP = re.compile(
    r"\b(?:what|which)\s+is\s+my\s+(?:lagna|ascendant|rashi|moon\s+sign|sun\s+sign|nakshatra)\s*\??$"
    r"|\bwhich\s+house\s+is\s+my\s+\w+\s+in\b|\bwhere\s+is\s+my\s+\w+\s*\??$"
    r"|^\s*(?:mera|meri)\s+(?:lagna|rashi|nakshatra)\s+(?:kya|kaun\w*)\s+(?:hai|h)\s*\??$"
    r"|^\s*\w+\s+(?:kis|kaunse)\s+(?:house|bhav|ghar)\s+(?:mein|me)\s+(?:hai|h)\s*\??$"
    r"|^\s*(?:मेरा|मेरी)\s+(?:लग्न|राशि|नक्षत्र)\s+(?:क्या|कौन\S*)\s+है\s*[?।]?$"
    r"|^\s*\S+\s+किस\s+भाव\s+में\s+है\s*[?।]?$",
    re.I)

_ACKNOWLEDGEMENT = re.compile(
    r"^\s*(?:ok(?:ay)?|theek\s+hai|thik\s+hai|thanks?|thank\s+you|dhanyavaad|shukriya|accha|achha|"
    r"hmm+|great|nice|ठीक\s+है|धन्यवाद|अच्छा)\s*[.!?।]*\s*$",
    re.I)

# Two questions in one message: "partner kaisa hoga aur shaadi kab hogi?"
_SECOND_QUESTION = re.compile(
    r"\b(?:kaisa|kaisi|kaise|kya)\b[^?]{0,60}\b(?:aur|and|or)\b[^?]{0,60}\b(?:kab|when|kitna|kaun)\b"
    r"|\b(?:when|kab)\b[^?]{0,60}\b(?:aur|and)\b[^?]{0,60}\b(?:how|kaisa|kaisi|what|kya)\b"
    r"|कैसा[^?]{0,60}और[^?]{0,60}कब|कब[^?]{0,60}और[^?]{0,60}कैसा",
    re.I)

# A question about ONE named factor (a house, a lord, the Navamsa, a period)
# is normal depth even inside a broad life area.
_SPECIFIC_FACTOR = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)\s+(?:house|lord|bhav)\b|\blord\b|\bnavamsa\b|\bD9\b|\bwhich\s+period\b"
    r"|भाव|स्वामी",
    re.I)
# Tag questions are not a second question: "My Mercury is in the 7th, right?"
_TAG_QUESTION = re.compile(r"\b(?:right|correct|na|hai\s+na)\s*\?", re.I)


@dataclass(frozen=True)
class Budget:
    messages: str
    words: str


BUDGETS: dict[str, Budget] = {
    SIMPLE: Budget("ONE message", "15–30 words"),
    NORMAL: Budget("TWO messages", "30–60 words in total"),
    DETAILED: Budget("THREE messages (four only if truly needed)", "60–100 words in total"),
    COMPLEX: Budget("THREE or FOUR messages", "80–120 words in total"),
}


def classify_depth(question: str, intent: str | None = None, switch_only: bool = False) -> str:
    """Pure. simple | normal | detailed | complex."""
    q = (question or "").strip()
    if switch_only or not q or _ACKNOWLEDGEMENT.match(q) or (intent or "") in ("smalltalk", "greeting"):
        return SIMPLE
    if _FACT_LOOKUP.search(q):
        return SIMPLE
    questions = q.count("?") - len(_TAG_QUESTION.findall(q))
    if _DEPTH_REQUEST.search(q) or questions >= 2 or _SECOND_QUESTION.search(q):
        return COMPLEX
    if (intent or "") in _BROAD_INTENTS and not _SPECIFIC_FACTOR.search(q):
        return DETAILED
    return NORMAL


def length_line(depth: str, question: str | None = None) -> str:
    """The per-turn length instruction placed after the shape rules."""
    budget = BUDGETS.get(depth, BUDGETS[NORMAL])
    # A bare fact ("Your Moon sign is Pisces.") was 5–7 words, half the
    # simple budget: the one line of meaning is what makes it an answer.
    # Acknowledgements and language switches stay one short sentence.
    lookup = is_fact_lookup(depth, question)
    shape = (" — the fact, plus one short clause on what it traditionally means or one related placement from "
             "the chart" if lookup else
             " — one or two chart factors, each named once, no Dasha unless the question is about timing"
             if depth == NORMAL else "")
    return (f"LENGTH FOR THIS ANSWER: {budget.messages}, about {budget.words}{shape}. Each message 15–25 words, "
            "at most two short sentences. These are targets, not limits on accuracy: never leave out a chart "
            "fact the answer needs, and never invent or alter one to save words.")


# A question that asks only WHICH/WHERE/WHO/UNTIL WHEN ("Who is the lord of my
# 10th house and where is it placed?") — answered with bare facts in 11–16
# words against a 30–60-word normal target (final quality gate, c5).
_ONLY_FACTS = re.compile(r"\b(?:who|which|where|until\s+when|kaun\w*|kahan|kis|kab\s+tak)\b|कौन|कहाँ|कहां|किस|कब\s+तक",
                         re.I)
_ASKS_MEANING = re.compile(r"\b(?:mean\w*|matlab|asar|effect|impact|say|says|kaisa|kaisi|kaise|kya\s+kehta|"
                           r"tell\s+me|batao|bataiye|how|why|good|achha|shubh)\b|अर्थ|असर|प्रभाव|कैसा|कैसी|बताइए|क्यों",
                           re.I)


def is_fact_lookup(depth: str, question: str | None) -> bool:
    q = (question or "").strip()
    if depth == SIMPLE:
        return bool(_FACT_LOOKUP.search(q))
    return depth == NORMAL and bool(_ONLY_FACTS.search(q)) and not _ASKS_MEANING.search(q)


def turn_length_note(depth: str, question: str | None = None) -> str:
    """The same target, restated in one bracketed line on the current question.

    Placed when the conversation already has replies — after a few long
    answers the model matched their length rather than the system-prompt
    target (live browser test: a "detailed" career question answered in six
    bubbles and 173 words, the same question in a fresh session in 74) — and
    on every fact lookup, which the model otherwise answered in 5–7 words
    ("Your moon sign is Pisces.") against a 15–30-word target."""
    budget = BUDGETS.get(depth, BUDGETS[NORMAL])
    if is_fact_lookup(depth, question):
        return (f"[Answer length: {budget.messages} of {budget.words}: state the fact, then add in your own words "
                "what it traditionally means for the user — not the bare fact alone.]")
    return f"[Answer length: {budget.messages}, about {budget.words}, whatever the length of earlier replies.]"


__all__ = ["classify_depth", "length_line", "turn_length_note", "is_fact_lookup", "BUDGETS", "SIMPLE", "NORMAL",
           "DETAILED", "COMPLEX"]
