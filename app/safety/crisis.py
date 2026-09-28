"""Self-harm and immediate-danger detection, before the model is called.

The service this was ported from decided crisis with an LLM classifier field
(`risk_level == "crisis"`). That is a good signal and a bad single point of
failure: it costs a call, it is unavailable exactly when the provider is
down, and `app/nodes/classifier.py` already degrades to keyword matching when
the LLM is unreachable — which would silently degrade crisis detection too.

So this layer is deterministic. It runs in microseconds, it cannot fail open
because of an outage, and it is deliberately tuned for **recall**: a false
positive shows a helpline to someone who did not need it, which is mildly
jarring; a false negative sends a chart reading to someone describing suicide.
Those costs are not symmetric and the thresholds here reflect that.

The idiom guard exists because English is full of casual death metaphors --
"dying to know", "this commute is killing me", "I'd die for that job". Those
are matched and excluded explicitly rather than by trying to write cleverer
patterns, because an idiom list is reviewable and a clever regex is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.safety.block_messages import block_message
from app.safety.categories import SafetyCategory


@dataclass(frozen=True, slots=True)
class CrisisCheck:
    blocked: bool
    category: SafetyCategory | None = None
    rule: str | None = None
    reply: str = ""


# ── idioms that look like self-harm and are not ───────────────────────
# Checked first. Only the softer patterns defer to this list; an explicit
# "I want to kill myself" fires regardless of anything else in the message.
_IDIOMS = re.compile(
    r"(dying to\s|die for\s|to die for|killing me|kill me now|"
    r"dead tired|dead serious|drop dead gorgeous|"
    r"i could(?:'ve| have)? died|died laughing|kill myself laughing|"
    r"deadline|dead end|dead weight)",
    re.IGNORECASE,
)

# ── unambiguous self-harm statements ──────────────────────────────────
# These fire even alongside an idiom: the cost of ignoring them is too high.
_CRISIS_STRONG = re.compile(
    r"("
    r"kill\s*(my ?self|me\s+off)|"
    r"\bsuicid(e|al)\b|"
    r"end(ing)?\s+(my|this)\s+life|end\s+it\s+all|"
    r"take\s+my\s+own\s+life|"
    r"self[\s-]*harm|harm\s+my ?self|hurt\s+my ?self|"
    r"cut(ting)?\s+my ?self|"
    r"overdose|\bod\s+on\b|"
    r"want\s+to\s+be\s+dead|better\s+off\s+dead|"
    r"don'?t\s+want\s+to\s+(live|be\s+alive|exist)|"
    r"no\s+(point|reason)\s+(in\s+)?living|"
    r"not\s+worth\s+living|"
    r"everyone\s+would\s+be\s+better\s+off\s+without\s+me"
    r")",
    re.IGNORECASE,
)

# Hinglish / romanised Hindi. Written out rather than transliterated on the
# fly because the spellings users actually type are irregular.
_CRISIS_STRONG_HINGLISH = re.compile(
    r"("
    r"aatmahatya|atmahatya|khudkushi|khudkhushi|"
    r"jaan\s+de\s*(dun|doon|du|dena|deni)|"
    r"mar\s*(jaun|jaoon|jau|jana\s+chahta|jana\s+chahti)|"
    r"marna\s+chahta|marna\s+chahti|"
    r"jeena\s+nahi(?:n)?\s+chahta|jeena\s+nahi(?:n)?\s+chahti|"
    r"jeene\s+ka\s+mann\s+nahi|"
    r"zindagi\s+khatam\s+kar"
    r")",
    re.IGNORECASE,
)

# Devanagari. No IGNORECASE needed; the script has no case.
_CRISIS_STRONG_DEVANAGARI: tuple[str, ...] = (
    "आत्महत्या",
    "खुदकुशी",
    "जान दे दूं",
    "जान दे दूँ",
    "मर जाऊं",
    "मर जाऊँ",
    "मरना चाहता",
    "मरना चाहती",
    "जीना नहीं चाहता",
    "जीना नहीं चाहती",
    "जीने का मन नहीं",
    "खुद को नुकसान",
)

# ── softer signals: fire unless the message reads as an idiom ─────────
_CRISIS_SOFT = re.compile(
    r"("
    r"want\s+to\s+die|wanna\s+die|wish\s+i\s+was\s+dead|wish\s+i\s+were\s+dead|"
    r"can'?t\s+(go\s+on|take\s+(it|this)\s+any\s*more|do\s+this\s+any\s*more)|"
    r"give\s+up\s+on\s+life|tired\s+of\s+living|"
    r"nothing\s+left\s+to\s+live\s+for"
    r")",
    re.IGNORECASE,
)

# ── abuse / immediate danger from another person ──────────────────────
# A different category with its own reply: the referral is a helpline and the
# police, not a mental-health line.
_ABUSE = re.compile(
    r"("
    r"(he|she|they|husband|wife|father|mother|partner|in.?laws?)\s+"
    r"(beats?|beat|hits?|hit|abuses?|abused|threatens?|threatened)\s+me|"
    r"beating\s+me|hitting\s+me|"
    r"domestic\s+(violence|abuse)|"
    r"(am|feel)\s+(i\s+)?(not\s+)?safe\s+at\s+home|"
    r"afraid\s+(for\s+my\s+life|he'?ll\s+kill\s+me|she'?ll\s+kill\s+me)|"
    r"threatening\s+to\s+kill\s+me|"
    r"marpeet|ghar\s+mein\s+maar|pati\s+marta\s+hai|"
    r"jaan\s+se\s+maarne\s+ki\s+dhamki"
    r")",
    re.IGNORECASE,
)

_ABUSE_DEVANAGARI: tuple[str, ...] = (
    "मारपीट",
    "घरेलू हिंसा",
    "जान से मारने की धमकी",
    "मुझे मारता है",
    "मुझे मारती है",
)


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def check_crisis(text: str | None, lang: str | None = None) -> CrisisCheck:
    """Screen for self-harm and immediate danger.

    Self-harm is checked before abuse: a message describing both is a
    self-harm message first, and the crisis reply carries the 24x7 lines.
    """
    if not text:
        return CrisisCheck(blocked=False)

    strong = (
        _CRISIS_STRONG.search(text)
        or _CRISIS_STRONG_HINGLISH.search(text)
        or _has_any(text, _CRISIS_STRONG_DEVANAGARI)
    )
    if strong:
        return CrisisCheck(
            blocked=True,
            category=SafetyCategory.self_harm_crisis,
            rule="self_harm_explicit",
            reply=block_message(SafetyCategory.self_harm_crisis, lang),
        )

    if _CRISIS_SOFT.search(text) and not _IDIOMS.search(text):
        return CrisisCheck(
            blocked=True,
            category=SafetyCategory.self_harm_crisis,
            rule="self_harm_distress",
            reply=block_message(SafetyCategory.self_harm_crisis, lang),
        )

    if _ABUSE.search(text) or _has_any(text, _ABUSE_DEVANAGARI):
        return CrisisCheck(
            blocked=True,
            category=SafetyCategory.abuse_coercion,
            rule="abuse_disclosure",
            reply=block_message(SafetyCategory.abuse_coercion, lang),
        )

    return CrisisCheck(blocked=False)


__all__ = ["CrisisCheck", "check_crisis"]
