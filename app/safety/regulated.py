"""Deterministic medical / legal / financial directive detector.

Tuned to **directive** language — the user asking to be told what to do — and
not to mere topic mention. That distinction is the whole design. An astrology
service is supposed to field "will I win my legal battle?" and "will money
come this year?"; those are chart questions that happen to touch a regulated
domain. What it must not field is "should I stop taking my tablets?" or
"which mutual fund should I buy?".

So a category fires on either a high-confidence standalone phrase (`*_STRONG`)
or a domain noun co-occurring with advice framing (`_ADVICE`). A bare keyword
is never enough.

Ported from the recovered service, where this ran as the classifier-independent
half of a fail-closed pair: the LLM classifier's verdict was OR'd with this so
a classifier error or a plain miss still gated. The same reasoning applies here
— `app/nodes/classifier.py` falls back to keywords when the LLM is down, and a
regulated directive must gate even then.
"""

from __future__ import annotations

import re
from typing import Literal

RegulatedDirective = Literal["none", "medical", "legal", "financial"]

# Advice / action framing — an interrogative or a request to be told what to
# DO. Includes the common Hinglish forms.
_ADVICE = re.compile(
    r"\b(should i|shall i|can i|do i need to|need to|which|what (kind|sort|type)? ?of|"
    r"how (much|many)|recommend|suggest|advise|advice|"
    r"kya mujhe|mujhe|chahiye|lena|leni|lun|loon|karu|karoon)\b",
    re.IGNORECASE,
)
_ADVICE_DEVANAGARI = ("चाहिए", "क्या", "लूँ", "लूं", "करूँ", "करूं")

# ── medical ───────────────────────────────────────────────────────────
_MEDICAL_NOUN = re.compile(
    r"\b(medicine|medication|drug|tablet|tablets|pill|pills|antibiotic|antibiotics|"
    r"dose|dosage|insulin|chemo|chemotherapy|prescription|surgery|operation|"
    r"dawa|dawai|goli|\d+\s*mg)\b",
    re.IGNORECASE,
)
_MEDICAL_STRONG = re.compile(
    r"(prescrib\w*|how many mg|what dose|diagnos\w+ (me|my)|stop taking my|"
    r"should i (stop|start|continue) (taking|my)|is it safe to take)",
    re.IGNORECASE,
)
_MEDICAL_DEVANAGARI = ("दवा", "दवाई", "गोली", "खुराक", "इलाज")

# ── legal ─────────────────────────────────────────────────────────────
_LEGAL_NOUN = re.compile(
    r"\b(lawsuit|sue|attorney|lawyer|contract|fir|police complaint|legal action|"
    r"legally binding|legal advice|mukadma|mukadama|kanooni)\b",
    re.IGNORECASE,
)
_LEGAL_STRONG = re.compile(
    r"(file a (lawsuit|case|complaint|fir)|should i sue|\bsue (him|her|them|my|the)\b|"
    r"take .{0,15}to court|legally binding|legal advice)",
    re.IGNORECASE,
)
_LEGAL_DEVANAGARI = ("मुकदमा", "कानूनी")

# ── financial ─────────────────────────────────────────────────────────
# Securities-only nouns. Bare "invest" and "share" are deliberately excluded:
# "should I SHARE my chart?" and "should I INVEST in this relationship?" are
# not financial directives. The financial object is required via _STRONG.
_FINANCIAL_NOUN = re.compile(
    r"\b(stock|stocks|shares|crypto|bitcoin|mutual fund|sip|fixed deposit|\bfd\b|"
    r"forex|portfolio|equity|equities)\b",
    re.IGNORECASE,
)
_FINANCIAL_STRONG = re.compile(
    r"(should i invest .{0,25}(stock|share|crypto|bitcoin|market|mutual fund|gold|"
    r"savings|money|property|real estate|\bfd\b|sip)|invest all my|invest my savings|"
    r"where (should i |to )?invest|which (stock|shares|crypto|mutual fund|fund)|"
    r"(buy|sell) .{0,15}(stock|stocks|share|shares|crypto|bitcoin))",
    re.IGNORECASE,
)
_FINANCIAL_DEVANAGARI: tuple[str, ...] = ()


def _has_advice(text: str) -> bool:
    return bool(_ADVICE.search(text)) or any(m in text for m in _ADVICE_DEVANAGARI)


def _fires(
    text: str,
    *,
    strong: re.Pattern[str],
    noun: re.Pattern[str],
    devanagari: tuple[str, ...],
    has_advice: bool,
) -> bool:
    if strong.search(text):
        return True
    noun_present = bool(noun.search(text)) or any(m in text for m in devanagari)
    return noun_present and has_advice


def detect_regulated_directive(text: str | None) -> RegulatedDirective:
    """Return the regulated domain the text asks for a directive in, else "none"."""
    if not text:
        return "none"
    has_advice = _has_advice(text)
    for domain, strong, noun, deva in (
        ("medical", _MEDICAL_STRONG, _MEDICAL_NOUN, _MEDICAL_DEVANAGARI),
        ("legal", _LEGAL_STRONG, _LEGAL_NOUN, _LEGAL_DEVANAGARI),
        ("financial", _FINANCIAL_STRONG, _FINANCIAL_NOUN, _FINANCIAL_DEVANAGARI),
    ):
        if _fires(text, strong=strong, noun=noun, devanagari=deva, has_advice=has_advice):
            return domain  # type: ignore[return-value]
    return "none"


__all__ = ["RegulatedDirective", "detect_regulated_directive"]
