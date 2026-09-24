"""LangGraph orchestration for one conversational turn.

    START -> load_context -> classify_intent -> retrieve_knowledge
          -> build_context -> generate_response -> evaluate_response
          -> save_metrics -> END

Linear on purpose. The extension points the design has to accommodate later —
a safety classifier before generation, tool calls, regeneration on a low
reward, a humanizer pass — are all edge changes in `build_graph()`, because
every node already reads and writes only the state keys it owns and holds no
state of its own. One conditional edge is included (`should_evaluate`) so the
shape for adding more is established rather than merely asserted.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.config.settings import settings
from app.llm.usage_tracker import new_request_id
from app.logging_setup import get_logger, request_id_var
from app.nodes.classifier import classify_intent_node
from app.nodes.context import build_context_node, load_context_node
from app.nodes.generate import generate_response_node, save_metrics_node
from app.nodes.quality import evaluate_response_node
from app.nodes.rag import retrieve_knowledge_node
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

    graph.add_node("load_context", load_context_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_knowledge", retrieve_knowledge_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("evaluate_response", evaluate_response_node)
    graph.add_node("save_metrics", save_metrics_node)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "classify_intent")
    graph.add_edge("classify_intent", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "build_context")
    graph.add_edge("build_context", "generate_response")
    graph.add_conditional_edges(
        "generate_response",
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
    try:
        return get_graph().invoke(state)
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
