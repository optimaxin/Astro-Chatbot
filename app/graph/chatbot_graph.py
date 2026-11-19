"""LangGraph orchestration for one conversational turn.

    START -> input_guardrail -(gated)-------------------------> save_metrics
                            -(clean)-> load_context -(cold follow-up)-> clarify -> save_metrics
                                                    -> classify_intent -(off-topic)-> scope_reply -> save_metrics
                                                                       -(unsure)---> clarify -> save_metrics
          -> retrieve_knowledge -> build_context -> generate_response
          -> validate_reply -> concise_rewrite -> humanize -> guardrail_out -> quality_check -> evaluate_response
          -> save_metrics -> END

Near-linear on purpose. `input_guardrail` sits ahead of `load_context` rather
than after it so a gated turn costs nothing at all — no history fetch, no
vector query, no model call. The remaining extension points the design has to
accommodate later — tool calls, regeneration on a low reward, a humanizer pass
— stay edge changes in `build_graph()`, because every node reads and writes
only the state keys it owns and holds no state of its own.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.config.settings import settings
from app.llm.usage_tracker import new_request_id
from app.logging_setup import get_logger, request_id_var
from app.observability import trace
from app.observability.trace import traced
from app.nodes.clarify import clarify_node, route_after_load_context
from app.nodes.classifier import classify_intent_node, route_after_classify, scope_reply_node
from app.nodes.concise import concise_rewrite_node
from app.nodes.context import build_context_node, load_context_node
from app.nodes.generate import generate_response_node, save_metrics_node
from app.nodes.humanize import humanize_node
from app.nodes.quality import evaluate_response_node
from app.nodes.quality_check import quality_check_node
from app.nodes.rag import retrieve_knowledge_node
from app.nodes.validate import validate_reply_node
from app.nodes.safety import (
    guardrail_out_node,
    input_guardrail_node,
    route_after_input_guardrail,
)
from app.kundli.kundli_service import KundliContext
from app.persona.registry import DEFAULT_PERSONA_ID
from app.prompts.response_style import DEFAULT_STYLE_ID
from app.state.chatbot_state import ChatbotState, new_state

log = get_logger("graph")


def should_evaluate(state: ChatbotState) -> str:
    """Skip the judge when there is nothing worth grading."""
    if state.get("generation_failed"):
        return "save_metrics"
    if not settings.enable_evaluation:
        return "save_metrics"
    return "evaluate_response"


def build_graph() -> Any:
    graph = StateGraph(ChatbotState)

    graph.add_node("input_guardrail", traced("node.input_guardrail")(input_guardrail_node))
    graph.add_node("load_context", traced("node.load_context")(load_context_node))
    graph.add_node("clarify", traced("node.clarify")(clarify_node))
    graph.add_node("scope_reply", traced("node.scope_reply")(scope_reply_node))
    graph.add_node("classify_intent", traced("node.classify_intent")(classify_intent_node))
    graph.add_node("retrieve_knowledge", traced("node.retrieve_knowledge")(retrieve_knowledge_node))
    graph.add_node("build_context", traced("node.build_context")(build_context_node))
    graph.add_node("generate_response", traced("node.generate_response")(generate_response_node))
    graph.add_node("validate_reply", traced("node.validate_reply")(validate_reply_node))
    graph.add_node("concise_rewrite", traced("node.concise_rewrite")(concise_rewrite_node))
    graph.add_node("humanize", traced("node.humanize")(humanize_node))
    graph.add_node("guardrail_out", traced("node.guardrail_out")(guardrail_out_node))
    graph.add_node("quality_check", traced("node.quality_check")(quality_check_node))
    graph.add_node("evaluate_response", traced("node.evaluate_response")(evaluate_response_node))
    graph.add_node("save_metrics", traced("node.save_metrics")(save_metrics_node))

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {"load_context": "load_context", "save_metrics": "save_metrics"},
    )
    # A bare follow-up with no earlier turn is asked about, not answered.
    graph.add_conditional_edges(
        "load_context",
        route_after_load_context,
        {"classify_intent": "classify_intent", "clarify": "clarify"},
    )
    graph.add_edge("clarify", "save_metrics")
    # An off-topic message gets the scope reply and an unsure one is asked about,
    # in both cases without retrieval or an astrology model call.
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"retrieve_knowledge": "retrieve_knowledge", "scope_reply": "scope_reply", "clarify": "clarify"},
    )
    graph.add_edge("scope_reply", "save_metrics")
    graph.add_edge("retrieve_knowledge", "build_context")
    graph.add_edge("build_context", "generate_response")
    # Chart-fact and language validation sits between generation and the
    # safety guard, so the guard always sees the reply that will be sent.
    graph.add_edge("generate_response", "validate_reply")
    # Concision runs on the validated reply and re-checks its own output;
    # CONCISE_REWRITE_MODE=off (default) makes it a no-op.
    graph.add_edge("validate_reply", "concise_rewrite")
    graph.add_edge("concise_rewrite", "humanize")
    graph.add_edge("humanize", "guardrail_out")
    graph.add_edge("guardrail_out", "quality_check")
    graph.add_conditional_edges(
        "quality_check",
        should_evaluate,
        {"evaluate_response": "evaluate_response", "save_metrics": "save_metrics"},
    )
    graph.add_edge("evaluate_response", "save_metrics")
    graph.add_edge("save_metrics", END)

    return graph.compile()


_compiled: Any | None = None


def get_graph() -> Any:
    """Compile once; the graph is stateless across turns."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
        log.info("graph compiled")
    return _compiled


def run_turn(
    question: str,
    session_id: str,
    user_id: str,
    persona_id: str = DEFAULT_PERSONA_ID,
    kundli: KundliContext | None = None,
    chat_bubbles: bool = False,
    response_style: str = DEFAULT_STYLE_ID,
    on_progress: Any | None = None,
) -> ChatbotState:
    """Run one full turn. Returns the final state, including `metrics`.

    Never raises: an unexpected failure anywhere in the graph is turned into a
    state carrying an apology and the error, because a crashed process loses
    the whole conversation. The HTTP layer relies on this — a `/chat` handler
    that has to catch graph exceptions would leak stack traces.
    """
    request_id = new_request_id()
    token = request_id_var.set(request_id)
    started_at = time.perf_counter()

    state = new_state(
        request_id, session_id, user_id, question, started_at, persona_id, kundli,
        chat_bubbles, response_style,
    )
    log.info("turn started", extra={"session_id": session_id, "user_id": user_id,
                                    "persona": persona_id,
                                    "question_chars": len(question)})
    # Reuse the API's trace when there is one (so rate limiting and chart
    # resolution are in it too); otherwise this turn gets its own.
    outer = trace.current()
    if outer is not None:
        outer.request_id = request_id
        trace_cm = contextlib.nullcontext(outer)
    else:
        trace_cm = trace.request(request_id)
    try:
        with trace_cm as t:
            if on_progress is None:
                final = get_graph().invoke(state)
            else:
                # Same graph, streamed node by node so a caller (the SSE
                # endpoint) can report progress. The final state is identical.
                final = dict(state)
                for update in get_graph().stream(state, stream_mode="updates"):
                    for node, delta in (update or {}).items():
                        if isinstance(delta, dict):
                            final.update(delta)
                        try:
                            on_progress(node)
                        except Exception:  # noqa: BLE001 — progress is best-effort
                            pass
            metrics = final.get("metrics") or {}
            trace.annotate(intent=metrics.get("intent"), model=metrics.get("model"),
                           fallback_used=metrics.get("fallback_used"),
                           failed=bool(metrics.get("generation_failed")))
            if isinstance(final.get("metrics"), dict):
                final["metrics"]["trace_ms"] = dict(t.summary()["by_span_ms"])
            return final
    except Exception as exc:  # noqa: BLE001 — last line of defence for the REPL
        log.exception("turn failed")
        from app.nodes.generate import FAILURE_MESSAGE

        state["response"] = FAILURE_MESSAGE
        state["generation_failed"] = True
        state["errors"] = [*state.get("errors", []), f"graph failure: {type(exc).__name__}: {exc}"]
        state["metrics"] = {
            "request_id": request_id,
            "session_id": session_id,
            "user_id": user_id,
            "persona": persona_id,
            "intent": "unknown",
            "generation_failed": True,
            "errors": state["errors"],
            "turn_latency_ms": int((time.perf_counter() - started_at) * 1000),
        }
        return state
    finally:
        request_id_var.reset(token)


__all__ = ["build_graph", "get_graph", "run_turn", "should_evaluate"]
