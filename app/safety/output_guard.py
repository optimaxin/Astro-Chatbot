"""Structural checks on what the model actually said.

Kept small on purpose. Subjective judgements about tone, persona drift or
helpfulness belong to the LLM judge in `app/evaluation/` — regex auditors were
tried for those in the service this came from and rejected, because they only
catch the phrasings their author anticipated while the model invents new ones
every turn.

What stays here is the set of failures that are loud, deterministic and
consequential:

* `leak_meta` — the "as an AI language model" tell, which contradicts the
  persona and reads as a malfunction.
* `hard_certainty` — "you will definitely", which turns interpretation into a
  promise. `BASE_SYSTEM_PROMPT` forbids it; this is what makes it stick.
* `pii_phone` — a phone number in the answer. The model has no real number to
  give, so any number here is invented.
* `disease_prediction` — a chart-to-illness claim. This one repairs rather
  than replaces: offending sentences are dropped and the rest of the reply
  ships, falling back to a canned line only if dropping would gut it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── whole-reply structural rules ──────────────────────────────────────
_LEAK_META_RE = re.compile(
    r"\bas an (ai|artificial intelligence) (language )?(model|assistant)\b",
    re.IGNORECASE,
)
# No trailing \b: "100%" ends on a non-word char, which \b would reject.
_CERTAINTY_RE = re.compile(
    r"\byou will (definitely|certainly|absolutely|surely|100%)",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{8,}\d")
_PHONE_MIN_DIGITS = 10
_PHONE_MAX_DIGITS = 13

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।\n])\s+")
_MIN_KEPT_CHARS = 40

# ── disease prediction ────────────────────────────────────────────────
_DISEASE_RE = re.compile(
    r"\b(cancer|tumou?r|diabetes|heart\s+(?:attack|disease)|stroke|kidney|"
    r"liver\s+disease|tb|tuberculosis|asthma|paralysis|blood\s+cancer|"
    r"disease|illness|ailment|"
    # Reproductive and gendered-health anatomy. A chart must never forecast
    # these; the source carried a regression note about a turn-1 reply warning
    # a user about "uterus-related health issues".
    r"uterus|uterine|ovar(?:y|ies|ian)|cervix|cervical|prostate|"
    r"miscarriage|infertilit|fibroid|pcos|"
    r"bimari|beemari|rog)\b"
    r"|बीमारी|कैंसर|रोग|गर्भाशय",
    re.IGNORECASE,
)
_CHART_FACTOR_RE = re.compile(
    r"\b(mars|saturn|rahu|ketu|jupiter|venus|mercury|sun|moon|shani|mangal|"
    r"dasha|mahadasha|transit|gochar|house|grah|kundli|chart|planet|lagna|rashi)\b",
    re.IGNORECASE,
)
_PROGNOSIS_RE = re.compile(
    r"\b(will\s+(get|have|cause|develop|lead\s+to|result\s+in)|"
    r"causes?|indicates?|leads?\s+to|shows?|suggests?|"
    r"hoga|ho\s+jayega|ho\s+jayegi)\b|होगा",
    re.IGNORECASE,
)
# The reply is declining, negating or referring out — not asserting.
_SAFE_RE = re.compile(
    r"\b(consult\s+an?\s+(doctor|physician|specialist|medical)|"
    r"see\s+an?\s+(doctor|physician|specialist)|"
    r"cannot\s+(predict|diagnose|tell)|can'?t\s+(predict|diagnose|tell)|"
    r"not\s+(able\s+to\s+)?(predict|diagnose)|"
    r"does?\s*n'?o?t\s+indicate|no\s+(disease|illness|condition)|"
    r"please\s+(see|visit|consult)\s+an?\s+(doctor|physician|specialist)|"
    r"nahi\s+(bata|keh)|doctor\s+se\s+(mile|milein|consult))\b",
    re.IGNORECASE,
)

DISEASE_SAFE_FALLBACK = (
    "I'm not able to read health conditions or predict illness from your chart — "
    "that's firmly a doctor's domain. If you have health concerns, please consult "
    "a qualified medical professional. I'm happy to look at other areas of your "
    "chart where astrology can genuinely help."
)


@dataclass(frozen=True, slots=True)
class OutputCheck:
    """What the guard found, and what should actually be sent.

    `text` is always safe to ship: unchanged when nothing fired, repaired when
    sentences were dropped, or the canned fallback when repair was not enough.
    """

    flagged: bool
    rule: str | None = None
    text: str = ""
    repaired: bool = False


def _has_phone(text: str) -> bool:
    for match in _PHONE_RE.finditer(text):
        digits = sum(1 for c in match.group(0) if c.isdigit())
        if _PHONE_MIN_DIGITS <= digits <= _PHONE_MAX_DIGITS:
            return True
    return False


def _is_disease_claim(sentence: str) -> bool:
    """A positive chart-to-illness assertion needs all three signals."""
    return bool(
        _DISEASE_RE.search(sentence)
        and _CHART_FACTOR_RE.search(sentence)
        and _PROGNOSIS_RE.search(sentence)
    )


def detect_disease_prediction(text: str) -> bool:
    """True when any sentence asserts illness from the chart and none walks it back."""
    if not text or _SAFE_RE.search(text):
        return False
    return any(_is_disease_claim(s) for s in _SENTENCE_SPLIT_RE.split(text))


def _drop_disease_sentences(text: str) -> str | None:
    """Remove the offending sentences, or None if too little would remain."""
    kept = [s for s in _SENTENCE_SPLIT_RE.split(text) if not _is_disease_claim(s)]
    result = " ".join(part.strip() for part in kept if part.strip())
    return result if len(result) >= _MIN_KEPT_CHARS else None


def check_output(text: str) -> OutputCheck:
    """Screen the generated reply. Never returns text that failed a rule."""
    if not text:
        return OutputCheck(flagged=False, text=text)

    if _LEAK_META_RE.search(text):
        return OutputCheck(flagged=True, rule="leak_meta", text=text)
    if _CERTAINTY_RE.search(text):
        return OutputCheck(flagged=True, rule="hard_certainty", text=text)
    if _has_phone(text):
        return OutputCheck(flagged=True, rule="pii_phone", text=text)

    if detect_disease_prediction(text):
        repaired = _drop_disease_sentences(text)
        if repaired is not None:
            return OutputCheck(
                flagged=True, rule="disease_prediction", text=repaired, repaired=True
            )
        return OutputCheck(
            flagged=True,
            rule="disease_prediction",
            text=DISEASE_SAFE_FALLBACK,
            repaired=True,
        )

    return OutputCheck(flagged=False, text=text)


__all__ = [
    "DISEASE_SAFE_FALLBACK",
    "OutputCheck",
    "check_output",
    "detect_disease_prediction",
]
