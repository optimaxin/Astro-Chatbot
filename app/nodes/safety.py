"""The two safety nodes: one before the model runs, one after.

`input_guardrail_node` is the first node in the graph — ahead of
`load_context`, so a gated turn costs no Redis round-trip, no Pinecone query
and no LLM call. It short-circuits to `save_metrics`, which still writes the
exchange to history: a canned reply is part of the conversation and the next
turn needs to see it.

`guardrail_out_node` runs on whatever the model produced, before the judge
grades it. Ordering matters here too — the judge should score the reply that
was actually sent, not a draft the guard went on to repair.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import settings
from app.logging_setup import get_logger
from app.nodes.generate import FAILURE_MESSAGE
from app.prompts.response_style import resolve_style
from app.safety.block_messages import block_message, resolve_lang
from app.safety.boundary import detect_boundary_probe
from app.safety.categories import SafetyCategory
from app.safety.crisis import check_crisis
from app.safety.input_guard import check_input
from app.safety.output_guard import check_output
from app.safety.regulated import detect_regulated_directive
from app.safety.scope import check_scope
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.safety")


def _lang_for(state: ChatbotState, question: str) -> str:
    """The canned-reply language, mirroring what the user wrote in."""
    style = resolve_style(
        state.get("response_style"), question, settings.api.default_response_style
    )
    return resolve_lang(style.id)


def input_guardrail_node(state: ChatbotState) -> dict[str, Any]:
    """Screen the user's message and gate the turn before anything is spent."""
    original = state.get("question", "") or ""
    lang = _lang_for(state, original)

    # Runs first for its side effect: any phone number is masked out of the
    # question every downstream node, log line and prompt will see.
    screened = check_input(original, lang)
    question = screened.question or original

    if screened.redacted:
        log.info(
            "phone number redacted from question",
            extra={"count": len(screened.redacted)},
        )

    # Crisis is evaluated ahead of the input screen's own block. A message can
    # trip both — a self-harm disclosure that happens to contain a long number
    # — and when it does, the helpline reply is the one that has to ship.
    crisis = check_crisis(question, lang)
    if crisis.blocked:
        return _blocked(state, question, crisis.category, crisis.rule, crisis.reply,
                        level="warning")

    if screened.blocked:
        rule = screened.category.value if screened.category else None
        return _blocked(state, question, screened.category, rule, screened.reply)

    directive = detect_regulated_directive(question)
    if directive != "none":
        category = SafetyCategory(directive)
        return _blocked(state, question, category, f"regulated_{directive}",
                        block_message(category, lang))

    boundary = detect_boundary_probe(question, lang)
    if boundary.blocked:
        return _blocked(state, question, boundary.category, boundary.rule, boundary.reply)

    scope = check_scope(question, lang)
    if scope.blocked:
        return _blocked(state, question, scope.category, scope.rule, scope.reply)

    return {
        "question": question,
        "safety": {"blocked": False, "redacted": len(screened.redacted)},
    }


def _blocked(
    state: ChatbotState,
    question: str,
    category: SafetyCategory | None,
    rule: str | None,
    reply: str,
    *,
    level: str = "info",
) -> dict[str, Any]:
    """Build the state update for a gated turn.

    `generation_failed` stays False on purpose: the turn did not fail, it was
    answered by a gate. That keeps the canned reply in history and out of the
    error metrics.
    """
    name = category.value if category else "unknown"
    getattr(log, level)(
        "safety gate engaged",
        extra={"category": name, "rule": rule, "session_id": state.get("session_id")},
    )
    return {
        "question": question,
        "response": reply,
        "model_used": "safety-gate",
        "fallback_used": False,
        "generation_failed": False,
        "intent": "safety_gated",
        "intent_confidence": 1.0,
        "intent_method": "safety-gate",
        "safety": {"blocked": True, "category": name, "rule": rule},
    }


def route_after_input_guardrail(state: ChatbotState) -> str:
    """A gated turn has its answer already; skip straight to persistence."""
    safety = state.get("safety") or {}
    return "save_metrics" if safety.get("blocked") else "load_context"


def guardrail_out_node(state: ChatbotState) -> dict[str, Any]:
    """Check the generated reply, repairing or replacing it when a rule fires."""
    if state.get("generation_failed"):
        return {}

    response = state.get("response", "") or ""
    result = check_output(response)
    if not result.flagged:
        return {}

    log.warning(
        "output guardrail fired",
        extra={"rule": result.rule, "repaired": result.repaired,
               "session_id": state.get("session_id")},
    )

    safety = dict(state.get("safety") or {})
    safety["output_rule"] = result.rule
    safety["output_repaired"] = result.repaired

    update: dict[str, Any] = {"safety": safety}
    # A repaired reply is safe to send. An unrepaired one is not, so the whole
    # answer is withheld rather than shipped with a known defect in it.
    if result.repaired:
        update["response"] = result.text
    else:
        update["response"] = FAILURE_MESSAGE
        update["errors"] = [
            *state.get("errors", []),
            f"output guardrail blocked the reply: {result.rule}",
        ]
    return update


__all__ = [
    "guardrail_out_node",
    "input_guardrail_node",
    "route_after_input_guardrail",
]
