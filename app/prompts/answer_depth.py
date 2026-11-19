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


# Multi-bubble product shape (2026-11-18 brief): an advice question is answered
# in about 4–5 chat bubbles — four content messages plus an optional follow-up
# — and a simple question in fewer. The follow-up is decided separately
# (app.prompts.followup) and is not counted here.
# Global response policy (2026-11-19 audit): live advice answers were 4–5
# bubbles of 33–49 words, three of four bubbles listing placements. Bubbles are
# 15–30 words; an advice answer is about 4 × 20 words, with the human meaning
# and the advice taking two of its four messages.
MESSAGE_SIZE = "1–2 short sentences (15–30 words)"

BUDGETS: dict[str, Budget] = {
    SIMPLE: Budget("ONE message", "15–30 words"),
    NORMAL: Budget("TWO or THREE messages", "40–80 words in total"),
    DETAILED: Budget("FOUR messages", "70–110 words in total"),
    COMPLEX: Budget("FOUR messages", "100–140 words in total"),
}

# The order of the content messages, per depth. Simple answers have one.
_STRUCTURE = {
    NORMAL: ("the direct answer to their concern, warmly and in plain words; then the chart fact behind it and "
             "what it means for them; then, if it helps, one practical suggestion"),
    DETAILED: ("(1) the direct answer: the main thing their chart shows for this question, in warm, simple words — "
               "not a greeting; (2) what that means in their everyday life — the human side, no new placement; "
               "(3) one supporting chart factor, or the running period when it matters for this question — leave "
               "this message out if it adds nothing new; (4) practical guidance — one or two concrete things they "
               "can do or be careful about"),
    COMPLEX: ("(1) the direct answer: the main thing their chart shows for each part of the question, in warm, "
              "simple words; (2) the most important chart factor and what it means in their everyday life; (3) one "
              "more relevant chart factor, or the running period when timing matters; (4) practical guidance — "
              "what they can do, what to consider, what to be careful about"),
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


def length_line(depth: str, question: str | None = None, followup: bool = False) -> str:
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
    order = f" In this order: {_STRUCTURE[depth]}." if depth in _STRUCTURE and not lookup else ""
    # Live 2026-11-18: with only "FOUR messages" here the model wrote four and
    # stopped; the follow-up (0/11 turns) has to be part of the count it sees.
    if followup and depth != SIMPLE:
        order += " Then one more, separate message: the FOLLOW-UP question."
    return (f"LENGTH FOR THIS ANSWER: {budget.messages}, about {budget.words}{shape}.{order} Each message is "
            f"one complete thought in {MESSAGE_SIZE}. These are targets, not limits on accuracy: "
            "never leave out a chart fact the answer needs, and never invent or alter one to fill a message.")


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

    Placed on EVERY chat turn. The primary model follows the structure it is
    shown but not a word count placed mid-way through a ~2,700-word system
    prompt (2026-11-19 audit: 153-word raw advice answers against a 90–140
    target, 189 after the shape rules were tightened); a target on the
    question itself is what it obeys (fact lookups, and turns with history,
    where earlier long replies otherwise set the length)."""
    budget = BUDGETS.get(depth, BUDGETS[NORMAL])
    if is_fact_lookup(depth, question):
        return (f"[Answer length: {budget.messages} of {budget.words}: state the fact, then add in your own words "
                "what it traditionally means for the user — not the bare fact alone.]")
    if depth == SIMPLE:
        return f"[Answer length: {budget.messages}, about {budget.words}.]"
    # Exact wording matters: this enumerated one-sentence form averaged 102 and
    # 114 raw words over two 21-sample runs; the same content prefixed
    # "Answer length:" with a follow-up clause and a history clause, 175 and 143.
    if depth == DETAILED:
        return ("[Reply as FOUR short chat messages separated by blank lines, each ONE sentence of at most 25 "
                "words: 1) the main thing their chart shows for this, 2) what it means in their daily life, 3) one "
                "supporting chart factor — skip it if it adds nothing new, 4) one practical tip. Name at most two "
                "chart factors in the whole reply.]")
    if depth == NORMAL:
        return ("[Reply as TWO or THREE short chat messages separated by blank lines, each ONE sentence of at most "
                "25 words: the direct answer, the chart fact behind it, then one practical suggestion if it helps. "
                "Name at most two chart factors in the whole reply.]")
    return (f"[Answer length: {budget.messages}, about {budget.words}, whatever the length of earlier replies. "
            f"Each message {MESSAGE_SIZE}, separated by a blank line; name at most two chart factors.]")


__all__ = ["classify_depth", "length_line", "turn_length_note", "is_fact_lookup", "BUDGETS", "MESSAGE_SIZE", "SIMPLE",
           "NORMAL", "DETAILED", "COMPLEX"]
