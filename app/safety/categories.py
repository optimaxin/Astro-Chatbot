"""The categories a turn can be gated under.

One enum shared by every gate so a category name is spelled exactly once. The
value is what lands in logs and metrics, so it is a stable snake_case string
rather than the member name's repr.
"""

from __future__ import annotations

from enum import Enum


class SafetyCategory(str, Enum):
    """Why a turn was gated.

    Ordered by the precedence the router applies: a message that is both a
    crisis disclosure and a medical question is a crisis first.
    """

    # ── pre-LLM, highest stakes first ─────────────────────────────────
    poisoning_emergency = "poisoning_emergency"
    self_harm_crisis = "self_harm_crisis"
    crisis_followup = "crisis_followup"
    violence_risk = "violence_risk"
    violence_aftermath = "violence_aftermath"
    abuse_coercion = "abuse_coercion"
    medical_emergency = "medical_emergency"
    dangerous_request = "dangerous_request"
    dangerous_safety = "dangerous_safety"
    medical = "medical"
    legal = "legal"
    financial = "financial"
    boundary = "boundary"
    out_of_scope = "out_of_scope"

    # ── input screen ──────────────────────────────────────────────────
    prompt_injection = "prompt_injection"
    pii_phone = "pii_phone"
    pii_credit_card = "pii_credit_card"

    # ── post-LLM ──────────────────────────────────────────────────────
    leak_meta = "leak_meta"
    hard_certainty = "hard_certainty"
    disease_prediction = "disease_prediction"


#: Categories that must never be answered by the model. A hit short-circuits
#: the turn: the canned reply ships and `generate_response` never runs.
BLOCKING_CATEGORIES: frozenset[SafetyCategory] = frozenset(
    {
        SafetyCategory.poisoning_emergency,
        SafetyCategory.self_harm_crisis,
        SafetyCategory.crisis_followup,
        SafetyCategory.violence_risk,
        SafetyCategory.violence_aftermath,
        SafetyCategory.medical_emergency,
        SafetyCategory.dangerous_request,
        SafetyCategory.dangerous_safety,
        SafetyCategory.abuse_coercion,
        SafetyCategory.medical,
        SafetyCategory.legal,
        SafetyCategory.financial,
        SafetyCategory.prompt_injection,
        SafetyCategory.pii_phone,
        SafetyCategory.pii_credit_card,
    }
)

__all__ = ["BLOCKING_CATEGORIES", "SafetyCategory"]
