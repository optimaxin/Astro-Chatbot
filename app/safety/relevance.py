"""Meaningless or clearly unrelated input: "blah blah blah", "who won
yesterday's match?", general trivia with no astrology angle at all.

Distinct from `app.safety.scope`, which declines an explicit request to do a
different *task* (write code, solve an equation, translate text). This module
catches a message that is not a task request either — it is simply noise, or
a question about the world that has nothing to do with the user or their
chart. Both are deterministic and precision-tuned for the same reason
`scope.py` is: a missed off-topic message just gets answered (a wasted model
call, not a harm), while a wrongly blocked real question is a worse user
experience. A short, ambiguous message ("Kab tak?", "Why?", "aur batao") must
always fall through to the classifier, which can read conversation history
that this pre-context gate cannot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.safety.block_messages import block_message
from app.safety.categories import SafetyCategory


@dataclass(frozen=True, slots=True)
class RelevanceCheck:
    blocked: bool
    category: SafetyCategory | None = None
    rule: str | None = None
    reply: str = ""


# A word (or short phrase) repeated three or more times with nothing else of
# substance — "blah blah blah", "test test test" — carries no question to
# answer. `\1` backreference keeps this to literal repeats, not just any two
# short words next to each other.
_REPEATED_FILLER = re.compile(r"\b(\w{2,})\b(?:\s+\1\b){2,}", re.IGNORECASE)

# Known placeholder/nonsense tokens that occasionally get pasted by mistake
# or sent as a connectivity check, independent of how many times they repeat.
_NONSENSE_TOKENS = re.compile(
    r"^\s*(?:blah+|lorem\s+ipsum|asdf+\w*|qwerty\w*|xyz+\s*abc+|hjkl+|zzz+)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

# Sports/election results, weather and general trivia that has no chart
# reading to it. Narrow on purpose: "Will my startup get funding?" or "Is my
# match compatible?" must never match these.
_SPORTS_RESULT = re.compile(
    r"\bwho\s+won\b.*\b(match|game|cup|series|tournament|championship|election)\b|"
    r"\b(score|result)\s+of\s+(yesterday'?s?|today'?s?|the)\s+\w*\s*(match|game)\b",
    re.IGNORECASE,
)

_WEATHER = re.compile(
    r"\bwhat'?s?\s+the\s+weather\b|\bweather\s+(today|tomorrow|forecast|report)\b|"
    r"\btemperature\s+(today|right\s+now|outside)\b",
    re.IGNORECASE,
)

_NEWS = re.compile(
    r"\b(latest|today'?s|breaking)\s+news\b|\bwhat'?s?\s+happening\s+in\s+the\s+news\b",
    re.IGNORECASE,
)

_GENERAL_TRIVIA = re.compile(
    r"\bcapital\s+of\s+\w+|\bpopulation\s+of\s+\w+|\bcurrency\s+of\s+\w+|"
    r"\bwho\s+is\s+the\s+(president|prime\s+minister|ceo)\s+of\b",
    re.IGNORECASE,
)


def check_relevance(text: str | None, lang: str | None = None) -> RelevanceCheck:
    """Decline input that is noise or an unrelated real-world question.

    Reuses `SafetyCategory.out_of_scope`'s existing canned reply rather than
    adding a new category: both cases resolve to the same honest answer —
    "that's outside what I read charts for" — in the user's own language.
    """
    if not text or not text.strip():
        return RelevanceCheck(blocked=False)

    for rule, pattern in (
        ("gibberish_repeated", _REPEATED_FILLER),
        ("gibberish_token", _NONSENSE_TOKENS),
        ("sports_result", _SPORTS_RESULT),
        ("weather", _WEATHER),
        ("news", _NEWS),
        ("general_trivia", _GENERAL_TRIVIA),
    ):
        if pattern.search(text):
            return RelevanceCheck(
                blocked=True,
                category=SafetyCategory.out_of_scope,
                rule=rule,
                reply=block_message(SafetyCategory.out_of_scope, lang),
            )

    # Phone troubleshooting, cricket scores, trivia, recipes, … (offtopic_rules.py): a
    # message with astrology vocabulary or a personal-life framing is never matched.
    from app.safety.offtopic_rules import offtopic_rule

    rule = offtopic_rule(text)
    if rule:
        return RelevanceCheck(blocked=True, category=SafetyCategory.out_of_scope, rule=f"offtopic_{rule}",
                              reply=block_message(SafetyCategory.out_of_scope, lang))
    return RelevanceCheck(blocked=False)


__all__ = ["RelevanceCheck", "check_relevance"]
