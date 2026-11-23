"""One ordered screen for every emergency rule, run before intent classification and every domain rule.

`screen_message(text, lang)` is the single place the order is decided; the input gate, the
conversation-context gate and the final output guard all call it, so a message can never be an
emergency in one of them and an off-topic question in another.

Order (most urgent first): poisoning and lethality, self-harm, harm to others, a violence admission,
an injured person, abuse, intent that is not specific, then dangerous-material requests.

Every rule set runs over the raw text and over its spelling variants (app.safety.normalize), so
"su1c1de", "s u i c i d e" and "p0ison" are screened too. Nothing here calls a model.
"""

from __future__ import annotations

from app.safety.categories import SafetyCategory
from app.safety.crisis import CrisisCheck, check_crisis
from app.safety.danger import check_dangerous, check_medical_emergency, check_poisoning, check_violence
from app.safety.emergency_rules import (check_injury_report, check_lethality, check_poison_intent, check_self_harm_variants,
                                        check_uncertain, check_violence_admission, check_violence_possible)
from app.safety.normalize import variants

#: Categories that are an emergency: nothing may replace their reply with a domain refusal or a rewrite.
EMERGENCY_CATEGORIES: frozenset[SafetyCategory] = frozenset({
    SafetyCategory.poisoning_emergency, SafetyCategory.self_harm_crisis, SafetyCategory.crisis_followup,
    SafetyCategory.violence_risk, SafetyCategory.violence_aftermath, SafetyCategory.medical_emergency,
    SafetyCategory.abuse_coercion,
})

_NOT_BLOCKED = CrisisCheck(False)


def _first(checks, text: str, lang: str | None, only: SafetyCategory | None = None) -> CrisisCheck | None:
    for check in checks:
        result = check(text, lang)
        if result.blocked and (only is None or result.category == only):
            return result
    return None


def _self_harm(text: str, lang: str | None) -> CrisisCheck | None:
    crisis = check_crisis(text, lang)
    return crisis if crisis.category == SafetyCategory.self_harm_crisis else None


def _other_crisis(text: str, lang: str | None) -> CrisisCheck | None:
    crisis = check_crisis(text, lang)
    return crisis if crisis.blocked else None


# (stage, rule set) in priority order; every stage is tried on every spelling variant before the next.
_STAGES = (
    lambda t, lang: _first((check_poisoning, check_poison_intent, check_lethality), t, lang, SafetyCategory.poisoning_emergency),
    _self_harm,
    lambda t, lang: _first((check_self_harm_variants,), t, lang),
    lambda t, lang: _first((check_lethality,), t, lang, SafetyCategory.self_harm_crisis),
    lambda t, lang: _first((check_violence, check_violence_possible), t, lang),
    lambda t, lang: _first((check_violence_admission,), t, lang),
    lambda t, lang: _first((check_injury_report, check_medical_emergency), t, lang),
    _other_crisis,
    lambda t, lang: _first((check_uncertain,), t, lang),
    lambda t, lang: _first((check_dangerous,), t, lang),
)


def screen_message(text: str | None, lang: str | None = None) -> CrisisCheck:
    """The first emergency or dangerous-request rule that fires on `text` or one of its spelling
    variants, else a non-blocked check. Self-harm outranks harm to others, as in the original gate order."""
    forms = variants(text)
    for stage in _STAGES:
        for form in forms:
            hit = stage(form, lang)
            if hit is not None and hit.blocked:
                return hit
    return _NOT_BLOCKED


__all__ = ["EMERGENCY_CATEGORIES", "screen_message"]
