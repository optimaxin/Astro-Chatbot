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
from app.prompts.response_style import resolve_style
from app.safety.block_messages import block_message, resolve_lang
from app.safety.boundary import detect_boundary_probe
from app.safety.categories import SafetyCategory
from app.safety.input_guard import check_input
from app.safety.output_guard import check_output
from app.safety.regulated import detect_regulated_directive
from app.safety.relevance import check_relevance
from app.safety.scope import check_scope
from app.safety.screen import EMERGENCY_CATEGORIES, screen_message
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

    # Safety is evaluated ahead of the input screen's own block, intent classification and every
    # domain rule. A message can trip both — a self-harm disclosure that happens to contain a long
    # number — and when it does, the safety reply has to ship. One ordered screen decides
    # (app.safety.screen): poisoning, self-harm, harm to others, an admission of violence, an
    # injured person, abuse, intent that is not specific, then dangerous-material requests; each
    # rule also runs over spelling variants ("su1c1de", "p0ison").
    emergency = screen_message(question, lang)
    if emergency.blocked:
        return _blocked(state, question, emergency.category, emergency.rule, emergency.reply, level="warning")

    # Domain rules (injection, regulated advice, boundary probes, programming/off-topic).
    # A live crisis outranks all of them: "ignore previous instructions" or "write me a
    # python program" right after a disclosure still gets the safety follow-up.
    domain = _domain_block(question, screened, lang)
    if domain is not None:
        held = _live_crisis_reply(state, question, lang)
        if held is not None:
            return {"question": question, **held}
        return _blocked(state, question, *domain)

    return {
        "question": question,
        "safety": {"blocked": False, "redacted": len(screened.redacted)},
    }


def _domain_block(question: str, screened: Any, lang: str) -> tuple[SafetyCategory | None, str | None, str] | None:
    """The first domain rule that blocks this message: (category, rule, reply), or None."""
    if screened.blocked:
        return screened.category, (screened.category.value if screened.category else None), screened.reply
    directive = detect_regulated_directive(question)
    if directive != "none":
        category = SafetyCategory(directive)
        return category, f"regulated_{directive}", block_message(category, lang)
    for result in (detect_boundary_probe(question, lang), check_scope(question, lang),
                   check_relevance(question, lang)):
        if result.blocked:
            return result.category, result.rule, result.reply
    return None


def _live_crisis_reply(state: ChatbotState, question: str, lang: str) -> dict[str, Any] | None:
    """The safety follow-up when an unresolved disclosure is live in this session, else None.
    Reads history only for a turn a domain rule is about to block (one Redis read)."""
    session_id = state.get("session_id")
    if not session_id:
        return None
    from app.memory.redis_memory import memory
    from app.nodes.conversation_gate import safety_context_update

    try:
        history = memory.get_history(session_id)
    except Exception:  # noqa: BLE001 — a history failure must never turn into a missing safety reply
        return None
    return safety_context_update(question, history, lang, session_id)


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
        # Not an outage, so not the outage text — and in the user's language
        # (the English FAILURE_MESSAGE reached Hindi/Hinglish users here).
        from app.nodes.validate import safe_fallback

        update["response"] = safe_fallback(state.get("response_style"))
        update["errors"] = [
            *state.get("errors", []),
            f"output guardrail blocked the reply: {result.rule}",
        ]
    return update


def _canned_emergency_texts(category: SafetyCategory) -> set[str]:
    return {block_message(category, lang).strip() for lang in ("en", "hi", "hinglish")}


def enforce_emergency_reply(state: ChatbotState) -> dict[str, Any]:
    """The last check before a turn is saved and shipped: an emergency reply is never replaced.

    1. A turn the safety gate answered must still carry its canned reply. Whatever later node wrote
       something else over it (a rewrite, a scope refusal, a fallback) is undone.
    2. A turn answered by anything else — an off-topic refusal from the intent classifier, a model
       answer — is checked again against the same deterministic screen; an emergency message is
       answered with the emergency reply, never with a domain refusal.

    Returns {} when nothing needs to change (the common case, a few microseconds)."""
    safety = state.get("safety") or {}
    response = (state.get("response") or "").strip()
    gated = safety.get("category")
    lang = _lang_for(state, state.get("question") or "")
    if safety.get("blocked") and gated in {c.value for c in EMERGENCY_CATEGORIES}:
        category = SafetyCategory(gated)
        if response in _canned_emergency_texts(category):
            return {}
        log.error("emergency reply was overwritten; restoring it",
                  extra={"category": gated, "session_id": state.get("session_id")})
        return {"response": block_message(category, lang), "model_used": "safety-gate",
                "safety": {**safety, "restored": True}}
    emergency = screen_message(state.get("question") or "", lang)
    if emergency.blocked and emergency.category in EMERGENCY_CATEGORIES:
        if response in _canned_emergency_texts(emergency.category):
            return {}
        log.error("emergency message reached a non-safety reply; overriding it",
                  extra={"category": emergency.category.value, "rule": emergency.rule,
                         "session_id": state.get("session_id")})
        return {"response": emergency.reply, "model_used": "safety-gate", "fallback_used": False,
                "generation_failed": False, "intent": "safety_gated", "intent_method": "safety-final-guard",
                "safety": {"blocked": True, "category": emergency.category.value,
                           "rule": f"final_guard_{emergency.rule}"}}
    return {}


def save_metrics_guarded_node(state: ChatbotState) -> dict[str, Any]:
    """`save_metrics`, preceded by the emergency check, so the reply that is persisted and returned
    is the reply that passed it."""
    from app.nodes.generate import save_metrics_node

    fix = enforce_emergency_reply(state)
    if fix:
        state = {**state, **fix}
    return {**fix, **save_metrics_node(state)}


__all__ = [
    "enforce_emergency_reply",
    "guardrail_out_node",
    "save_metrics_guarded_node",
    "input_guardrail_node",
    "route_after_input_guardrail",
]
