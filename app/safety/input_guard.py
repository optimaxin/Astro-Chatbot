"""Input screen that runs before a single token is spent.

Two different responses, because the two problems are not alike:

* **Block** — prompt injection and card numbers. Neither has an innocent
  reading in an astrology chat, so the turn stops and a canned reply ships.
* **Redact** — phone numbers. This is deliberately *not* a block, unlike the
  service this was ported from. That service received birth details through a
  structured form; here they arrive as free text, and "born 15 04 1990 10 30
  in Delhi" is twelve digits separated by spaces — indistinguishable from a
  phone number to any regex simple enough to be trustworthy. Blocking it would
  reject a perfectly normal first message. So the digits are masked out of the
  question before it reaches the model or the logs, and the turn continues.

The date/time guard below is what keeps the redaction itself from eating a
birth date, which would be a quieter version of the same bug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.safety.block_messages import block_message
from app.safety.categories import SafetyCategory

_PROMPT_INJECTION_RE = re.compile(
    r"(ignore (all )?previous (instructions|prompts)|"
    r"disregard (the )?(above|prior) (instructions|prompt)|"
    r"system\s*prompt\s*:|"
    r"reveal (your|the) (system )?(prompt|instructions)|"
    r"you are now [^.]{0,40}(developer|jailbroken|unfiltered|dan)\b)",
    re.IGNORECASE,
)

# 13-19 digits in the grouping a card number uses. No Luhn check: a false
# positive costs one canned reply, and asking the user to re-send without the
# card number is the right outcome either way.
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# A run of digits with phone-ish separators only. Slashes and dots are absent
# on purpose so `15/04/1990` and `15.04.1990` never enter the candidate set.
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{8,}\d")
_PHONE_MIN_DIGITS = 10
_PHONE_MAX_DIGITS = 13

# Spans that look like a birth date or a birth time. A phone candidate
# overlapping one of these is left alone.
_DATE_LIKE_RE = re.compile(r"\b\d{1,2}\s*[-/.\s]\s*\d{1,2}\s*[-/.\s]\s*\d{2,4}\b")
_YEAR_RE = re.compile(r"\b(1[89]|20)\d{2}\b")
_TIME_LIKE_RE = re.compile(r"\b\d{1,2}\s*[:.]\s*\d{2}\s*(am|pm|a\.m\.|p\.m\.)?", re.IGNORECASE)

REDACTION = "[redacted]"


@dataclass(frozen=True, slots=True)
class InputCheck:
    """Outcome of the pre-LLM input screen.

    `question` is what the rest of the pipeline should use — the original text
    with any phone numbers masked. When `blocked` is true it is unchanged and
    irrelevant, because nothing downstream will read it.
    """

    blocked: bool
    category: SafetyCategory | None = None
    reply: str = ""
    question: str = ""
    redacted: tuple[str, ...] = ()


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges holding a date, a year or a clock time."""
    spans: list[tuple[int, int]] = []
    for pattern in (_DATE_LIKE_RE, _YEAR_RE, _TIME_LIKE_RE):
        spans.extend(m.span() for m in pattern.finditer(text))
    return spans


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < s_end and s_start < end for s_start, s_end in spans)


def redact_phone_numbers(text: str) -> tuple[str, tuple[str, ...]]:
    """Mask phone-shaped digit runs, leaving birth dates and times intact.

    Returns the cleaned text and the raw matches that were masked, so the
    caller can log *that* a number was seen without logging the number.
    """
    if not text:
        return text, ()

    protected = _protected_spans(text)
    found: list[str] = []
    out: list[str] = []
    cursor = 0

    for match in _PHONE_RE.finditer(text):
        digits = sum(1 for c in match.group(0) if c.isdigit())
        if not (_PHONE_MIN_DIGITS <= digits <= _PHONE_MAX_DIGITS):
            continue
        if _overlaps(match.span(), protected):
            continue
        found.append(match.group(0).strip())
        out.append(text[cursor : match.start()])
        out.append(REDACTION)
        cursor = match.end()

    if not found:
        return text, ()
    out.append(text[cursor:])
    return "".join(out), tuple(found)


def check_input(text: str, lang: str | None = None) -> InputCheck:
    """Screen the user's message. Blocking rules run before redaction."""
    if not text:
        return InputCheck(blocked=False, question=text)

    if _PROMPT_INJECTION_RE.search(text):
        return InputCheck(
            blocked=True,
            category=SafetyCategory.prompt_injection,
            reply=block_message(SafetyCategory.prompt_injection, lang),
            question=text,
        )

    if _CREDIT_CARD_RE.search(text):
        return InputCheck(
            blocked=True,
            category=SafetyCategory.pii_credit_card,
            reply=block_message(SafetyCategory.pii_credit_card, lang),
            question=text,
        )

    cleaned, redacted = redact_phone_numbers(text)
    return InputCheck(
        blocked=False,
        category=SafetyCategory.pii_phone if redacted else None,
        question=cleaned,
        redacted=redacted,
    )


__all__ = ["REDACTION", "InputCheck", "check_input", "redact_phone_numbers"]
