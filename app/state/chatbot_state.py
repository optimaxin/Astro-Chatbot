"""The state object that flows through the LangGraph pipeline.

A plain `TypedDict` rather than a Pydantic model: LangGraph merges node
return values into this dict, and every node returns only the keys it owns,
so there is no single construction point for a validator to guard. Validation
happens at the real boundaries instead — `IntentResult` and `QualityScores`
are Pydantic-validated where the LLM's JSON enters the system.
"""

from __future__ import annotations

from typing import Any, TypedDict

from app.kundli.kundli_service import KundliContext
from app.memory.redis_memory import Message
from app.persona.registry import DEFAULT_PERSONA_ID
from app.prompts.response_style import DEFAULT_STYLE_ID
from app.rag.retriever import RetrievalResult


class ChatbotState(TypedDict, total=False):
    # ── inputs ────────────────────────────────────────────────────────
    request_id: str
    session_id: str
    user_id: str
    question: str
    persona_id: str
    # Set when the caller already resolved the chart (the web API computes
    # it from birth details). load_context uses it instead of a lookup.
    preset_kundli: KundliContext
    chat_bubbles: bool
    response_style: str

    # ── input_guardrail / guardrail_out ───────────────────────────────
    # {"blocked": bool, "category": str, "rule": str, ...}. A blocked turn
    # carries its own `response` and never reaches the model.
    safety: dict[str, Any]

    # ── load_context ──────────────────────────────────────────────────
    history: list[Message]
    kundli: KundliContext
    session_state: dict[str, Any]

    # Language decision for this turn (app.prompts.language_policy):
    # {"language", "style", "source", "persisted", "switch_only"}.
    # load_context overwrites `response_style` with the resolved style.
    language: dict[str, Any]
    # Follow-up decision for this turn (app.prompts.followup): {"allowed", "reason"}.
    followup: dict[str, Any]
    # Length budget for this turn (app.prompts.answer_depth): simple|normal|detailed|complex.
    answer_depth: str

    # ── classify_intent ───────────────────────────────────────────────
    intent: str
    intent_confidence: float
    intent_method: str  # "llm" | "keyword-fallback"
    # answer | clarify | scope_reply, and the full structured classification
    # (intent, sub_intent, is_astrology_related, is_follow_up, language, confidence).
    routing_decision: str
    classification: dict[str, Any]

    # ── retrieve_knowledge ────────────────────────────────────────────
    retrieval: RetrievalResult

    # ── build_context ─────────────────────────────────────────────────
    messages: list[dict[str, str]]

    # ── generate_response ─────────────────────────────────────────────
    response: str
    model_used: str
    fallback_used: bool
    generation_failed: bool

    # ── validate_reply ────────────────────────────────────────────────
    # Chart-claim and language validation of the draft, and what was done
    # about it. See app/nodes/validate.py.
    validation: dict[str, Any]
    # Knowledge-base passages the reply cited, validated against what was
    # retrieved: [{"id": "document::topic", "title": "..."}].
    citations: list[dict[str, Any]]

    # ── quality_check (independent checker, shadow mode) ──────────────
    # {"status": PASS|REPAIR|REJECT|UNAVAILABLE|SKIPPED|PENDING, ...}. Advisory only.
    quality_check: dict[str, Any]

    # ── evaluate_response ─────────────────────────────────────────────
    quality: dict[str, Any]
    reward: float

    # ── save_metrics ──────────────────────────────────────────────────
    metrics: dict[str, Any]

    # ── cross-cutting ─────────────────────────────────────────────────
    errors: list[str]
    started_at: float


def new_state(
    request_id: str,
    session_id: str,
    user_id: str,
    question: str,
    started_at: float,
    persona_id: str = DEFAULT_PERSONA_ID,
    preset_kundli: KundliContext | None = None,
    chat_bubbles: bool = False,
    response_style: str = DEFAULT_STYLE_ID,
) -> ChatbotState:
    return ChatbotState(
        request_id=request_id,
        session_id=session_id,
        user_id=user_id,
        question=question,
        persona_id=persona_id,
        started_at=started_at,
        chat_bubbles=chat_bubbles,
        response_style=response_style,
        errors=[],
        **({"preset_kundli": preset_kundli} if preset_kundli is not None else {}),
    )


__all__ = ["ChatbotState", "new_state"]
