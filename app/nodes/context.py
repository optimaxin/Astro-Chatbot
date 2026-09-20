"""Context loading and prompt assembly nodes.

`load_context_node` is the graph's first step: it gathers the three distinct
kinds of context — conversation history (Redis), the user's kundli (Tredev
backend or sample file), and any remembered session facts — and puts them on
the state. Each is loaded independently so one failing source degrades that
one input rather than the turn.

`build_context_node` is the last step before generation: it turns everything
gathered so far into the exact message list the LLM will receive.
"""

from __future__ import annotations

from typing import Any

from app.kundli.kundli_service import KundliContext, get_kundli
from app.logging_setup import get_logger
from app.memory.redis_memory import memory
from app.persona.registry import get_persona
from app.prompts.builder import build_messages
from app.rag.retriever import RetrievalResult
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.context")


def load_context_node(state: ChatbotState) -> dict[str, Any]:
    session_id = state["session_id"]
    user_id = state["user_id"]
    errors: list[str] = []

    try:
        history = memory.get_history(session_id)
    except Exception as exc:  # noqa: BLE001 — a memory outage must not end the turn
        log.warning("history load failed", extra={"error_type": type(exc).__name__})
        errors.append(f"history unavailable: {type(exc).__name__}")
        history = []

    try:
        session_state = memory.get_session_state(session_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("session state load failed", extra={"error_type": type(exc).__name__})
        errors.append(f"session state unavailable: {type(exc).__name__}")
        session_state = {}

    # The chart is resolved from the most authoritative source available, in
    # order, and each step is skipped once one has produced a chart:
    #
    #   1. one the caller already resolved  (the web API computes it from the
    #      birth details a logged-out visitor just submitted)
    #   2. the per-user Redis cache         (avoids re-hitting the calculator
    #      on every turn of a chatty session)
    #   3. the configured provider          (sample file or Tredev backend)
    #
    # Step 1 must win outright: falling through to a lookup when a chart was
    # handed in would quietly answer about somebody else's chart.
    kundli = state.get("preset_kundli")
    chart_preset = kundli is not None
    chart_cached = False

    if kundli is None:
        try:
            cached = memory.get_chart_facts(user_id)
            if cached:
                kundli = KundliContext.from_facts(user_id, cached)
                chart_cached = kundli is not None
        except Exception as exc:  # noqa: BLE001 — the cache is an optimisation
            log.warning("chart cache read failed", extra={"error_type": type(exc).__name__})

    if kundli is None:
        try:
            kundli = get_kundli(user_id)
            if not kundli.available:
                errors.append(f"kundli unavailable: {kundli.error}")
            else:
                memory.save_chart_facts(user_id, kundli.to_facts())
        except Exception as exc:  # noqa: BLE001
            log.exception("kundli load failed")
            kundli = KundliContext.unavailable(user_id, "unknown", f"{type(exc).__name__}: {exc}")
            errors.append(f"kundli unavailable: {type(exc).__name__}")

    if not kundli.available and not errors:
        errors.append(f"kundli unavailable: {kundli.error}")

    log.info(
        "context loaded",
        extra={
            "history_messages": len(history),
            "kundli_available": kundli.available,
            "kundli_source": kundli.source,
            "chart_cached": chart_cached,
            "chart_preset": chart_preset,
            "session_state_keys": sorted(session_state.keys()),
        },
    )

    return {
        "history": history,
        "kundli": kundli,
        "session_state": session_state,
        "errors": [*state.get("errors", []), *errors],
    }


def build_context_node(state: ChatbotState) -> dict[str, Any]:
    retrieval = state.get("retrieval") or RetrievalResult.unavailable("retrieval node did not run")
    persona = get_persona(state.get("persona_id"))

    # A specialised persona reads an unclassifiable question in its own frame
    # rather than falling back to a generic reading.
    intent = state.get("intent", "general_astrology")
    if intent in {"unknown", ""}:
        intent = persona.default_intent

    messages = build_messages(
        question=state["question"],
        intent=intent,
        kundli=state["kundli"],
        retrieval=retrieval,
        history=state.get("history") or [],
        session_state=state.get("session_state") or {},
        persona=persona,
        chat_bubbles=bool(state.get("chat_bubbles")),
    )

    system_chars = len(messages[0]["content"])
    log.info(
        "prompt assembled",
        extra={
            "message_count": len(messages),
            "system_prompt_chars": system_chars,
            "history_turns": len(messages) - 2,
            "intent": intent,
            "persona": persona.id,
        },
    )
    return {"messages": messages}


__all__ = ["load_context_node", "build_context_node"]
