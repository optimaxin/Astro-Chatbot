"""Out-of-domain requests: write my code, do my maths, translate this.

The service this was ported from left scope to its LLM classifier, reasoning
that there was "no clean deterministic detector, and a missed off-topic answer
is low-liability vs a regulated directive". That is right about the stakes, so
this module is the one gate here tuned for **precision** rather than recall:
every pattern below names an explicit request to perform a non-astrology task.
A missed off-topic question just gets answered, which is a wasted call and not
a harm.

The narrow framing matters because astrology chat is full of words that look
off-domain. "Will my startup get funding?" contains business vocabulary and is
squarely a chart question; "write a Python script" is not. Only the imperative
task request fires.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.safety.block_messages import block_message
from app.safety.categories import SafetyCategory


@dataclass(frozen=True, slots=True)
class ScopeCheck:
    blocked: bool
    category: SafetyCategory | None = None
    rule: str | None = None
    reply: str = ""


_CODING = re.compile(
    r"\b(write|generate|debug|fix|refactor|explain)\s+(me\s+)?(a|an|this|my|some)?\s*"
    r"(python|java|javascript|typescript|c\+\+|sql|html|css|react|code|script|"
    r"function|program|query|regex)\b|"
    r"\b(code|script|program)\s+(for|to)\s+\w+|"
    r"\bstack\s?trace\b|\bcompile\s+error\b",
    re.IGNORECASE,
)

_MATH = re.compile(
    r"\b(solve|calculate|compute)\s+(this|the|my)?\s*"
    r"(equation|integral|derivative|sum|maths?|problem|expression)\b|"
    r"\bwhat\s+is\s+\d+\s*[-+*/x]\s*\d+",
    re.IGNORECASE,
)

_TRANSLATE = re.compile(
    r"\btranslate\s+(this|that|the following|it)\b|"
    r"\b(translate|convert)\s+.{0,30}\b(into|to)\s+"
    r"(english|hindi|spanish|french|german|tamil|telugu|marathi|bengali)\b",
    re.IGNORECASE,
)

_DRAFTING = re.compile(
    r"\bwrite\s+(me\s+)?(a|an|my)\s+"
    r"(essay|email|resume|cv|cover letter|blog|article|poem|story|speech|"
    r"assignment|report|letter of)\b|"
    r"\b(draft|compose)\s+(me\s+)?(a|an)\s+(email|letter|essay|message to my)\b",
    re.IGNORECASE,
)

_DIET_FITNESS = re.compile(
    r"\b(diet|meal|workout|exercise|training)\s+(plan|chart|routine|schedule)\b|"
    r"\bhow many calories\b|\bgym routine\b",
    re.IGNORECASE,
)

_TECH_SUPPORT = re.compile(
    r"\b(my|the)\s+(phone|laptop|computer|wifi|printer|app)\s+"
    r"(is\s+)?(not working|won'?t|broken|crashed|keeps)\b|"
    r"\bhow (do|to) i\s+(install|uninstall|reset|reboot|update)\b",
    re.IGNORECASE,
)


def check_scope(text: str | None, lang: str | None = None) -> ScopeCheck:
    """Decline an explicit non-astrology task request and redirect."""
    if not text:
        return ScopeCheck(blocked=False)

    for rule, pattern in (
        ("coding", _CODING),
        ("math", _MATH),
        ("translation", _TRANSLATE),
        ("drafting", _DRAFTING),
        ("diet_fitness", _DIET_FITNESS),
        ("tech_support", _TECH_SUPPORT),
    ):
        if pattern.search(text):
            return ScopeCheck(
                blocked=True,
                category=SafetyCategory.out_of_scope,
                rule=rule,
                reply=block_message(SafetyCategory.out_of_scope, lang),
            )

    return ScopeCheck(blocked=False)


__all__ = ["ScopeCheck", "check_scope"]
