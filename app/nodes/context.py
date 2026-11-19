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

    # One pipelined Redis request for history + session state + chart cache
    # when available (REDIS_PIPELINE); otherwise the per-call reads below.
    prefetched_chart: dict[str, Any] | None = None
    batched = None
    try:
        batched = memory.load_turn(session_id, user_id,
                                   want_chart=state.get("preset_kundli") is None)
    except Exception:  # noqa: BLE001 — fall back to the per-call path
        batched = None

    if batched is not None:
        history, session_state, prefetched_chart = batched
    else:
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
            cached = prefetched_chart if batched is not None else memory.get_chart_facts(user_id)
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

    # The reply language, decided once per turn by the central policy:
    # explicit request > saved preference > this message > history > English.
    from app.config.settings import settings
    from app.prompts.language_policy import PREFERENCE_KEY, resolve_language

    decision = resolve_language(
        state["question"],
        saved_preference=(session_state or {}).get(PREFERENCE_KEY),
        earlier_user_messages=[m.content for m in history if m.role == "user"],
        requested_style=state.get("response_style"),
        deployment_default=settings.api.default_response_style,
    )

    log.info(
        "context loaded",
        extra={
            "history_messages": len(history),
            "kundli_available": kundli.available,
            "kundli_source": kundli.source,
            "chart_cached": chart_cached,
            "chart_preset": chart_preset,
            "session_state_keys": sorted(session_state.keys()),
            "language": decision.language,
            "language_source": decision.source,
        },
    )

    return {
        "history": history,
        "kundli": kundli,
        "session_state": session_state,
        "response_style": decision.style_id,
        "language": decision.to_dict(),
        "errors": [*state.get("errors", []), *errors],
    }


def clean_history(history: list, kundli: KundliContext | None) -> tuple[list, int]:
    """Earlier assistant replies with chart-contradicting sentences removed.

    History is context, never a source of chart facts: a placement the model
    got wrong two turns ago must not be fed back as if it were true (it was,
    in production, and the error repeated). Deterministic; user messages are
    never touched. Returns (history, sentences removed).
    """
    if not history or kundli is None or not kundli.available:
        return history, 0
    from app.evaluation.chart_claims import strip_sentences, validate_reply
    from app.kundli.chart_facts import build_chart_facts
    from app.memory.redis_memory import Message

    facts = build_chart_facts(kundli)
    out, removed = [], 0
    for message in history:
        if message.role != "assistant":
            out.append(message)
            continue
        bad = {c.sentence_index for c in validate_reply(message.content, facts).contradictions}
        if not bad:
            out.append(message)
            continue
        kept = strip_sentences(message.content, bad)
        removed += len(bad)
        if kept.strip():
            out.append(Message(role="assistant", content=kept, timestamp=message.timestamp))
    return out, removed


def build_context_node(state: ChatbotState) -> dict[str, Any]:
    retrieval = state.get("retrieval") or RetrievalResult.unavailable("retrieval node did not run")
    persona = get_persona(state.get("persona_id"))

    # A specialised persona reads an unclassifiable question in its own frame
    # rather than falling back to a generic reading.
    intent = state.get("intent", "general_astrology")
    if intent in {"unknown", ""}:
        intent = persona.default_intent

    history, scrubbed = clean_history(state.get("history") or [], state.get("kundli"))
    from app.prompts.followup import decide, prompt_line

    language_info = state.get("language") or {}
    previous_assistant = next((m.content for m in reversed(state.get("history") or []) if m.role == "assistant"),
                              None)
    followup = decide(state["question"], state.get("intent"), language_info.get("language") or "english",
                      previous_assistant, switch_only=bool(language_info.get("switch_only")))
    from app.prompts.answer_depth import classify_depth, length_line, turn_length_note

    depth = classify_depth(state["question"], state.get("intent"), bool(language_info.get("switch_only")))
    messages = build_messages(
        question=state["question"],
        intent=intent,
        kundli=state["kundli"],
        retrieval=retrieval,
        history=history,
        session_state=state.get("session_state") or {},
        persona=persona,
        chat_bubbles=bool(state.get("chat_bubbles")),
        response_style=state.get("response_style"),
        switch_only=bool((state.get("language") or {}).get("switch_only")),
        followup_line=prompt_line(followup),
        length_line=length_line(depth, state["question"], followup.allowed),
        length_note=turn_length_note(depth, state["question"]),
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
            "response_style": state.get("response_style"),
            "history_sentences_scrubbed": scrubbed,
            "answer_depth": depth,
        },
    )
    return {"messages": messages, "followup": followup.to_dict(), "answer_depth": depth}


__all__ = ["load_context_node", "build_context_node"]
