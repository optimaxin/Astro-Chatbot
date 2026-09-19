"""Response generation and metric persistence nodes."""

from __future__ import annotations

import time
from typing import Any

from app.config.settings import settings
from app.llm.llm_factory import ROLE_MAIN, LLMCallFailed, LLMNotConfigured, get_llm
from app.llm.usage_tracker import usage_tracker
from app.logging_setup import get_logger
from app.memory.redis_memory import memory
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.generate")

# Shown when the primary and fallback models have both failed. Deliberately
# says nothing astrological — inventing a reading to cover an outage would be
# exactly the failure mode this service is built to avoid.
FAILURE_MESSAGE = (
    "I'm having trouble reaching my astrology model right now, so I'd rather not "
    "guess at an answer. Could you try again in a moment?"
)


def generate_response_node(state: ChatbotState) -> dict[str, Any]:
    messages = state.get("messages") or []
    if not messages:
        return {
            "response": FAILURE_MESSAGE,
            "model_used": "none",
            "fallback_used": False,
            "generation_failed": True,
            "errors": [*state.get("errors", []), "generation skipped: no prompt was assembled"],
        }

    try:
        llm = get_llm(role=ROLE_MAIN)
        result = llm.complete(
            messages,
            session_id=state["session_id"],
            user_id=state["user_id"],
            request_id=state["request_id"],
        )
    except (LLMCallFailed, LLMNotConfigured) as exc:
        log.error("generation failed", extra={"error_message": str(exc)[:400]})
        return {
            "response": FAILURE_MESSAGE,
            "model_used": "none",
            "fallback_used": False,
            "generation_failed": True,
            "errors": [*state.get("errors", []), f"generation failed: {exc}"],
        }

    log.info(
        "response generated",
        extra={
            "model": result.usage.model,
            "response_chars": len(result.text),
            "latency_ms": result.usage.latency_ms,
            "fallback_used": result.usage.fallback_used,
        },
    )
    return {
        "response": result.text,
        "model_used": result.usage.model,
        "fallback_used": result.usage.fallback_used,
        "generation_failed": False,
    }


def save_metrics_node(state: ChatbotState) -> dict[str, Any]:
    """Persist the turn and assemble the metrics the terminal renders.

    Only a successful turn is written to history — storing a failure message
    as if the assistant had said it would corrupt the conversation for every
    later turn.
    """
    request_id = state["request_id"]
    session_id = state["session_id"]
    failed = state.get("generation_failed", False)

    if not failed:
        memory.save_message(session_id, "user", state["question"])
        memory.save_message(session_id, "assistant", state["response"])
        memory.save_session_state(
            session_id,
            {"last_intent": state.get("intent", "unknown"), "last_turn_at": time.time()},
        )

    # The turn's LLM calls: generation, plus classification and evaluation
    # when those ran. They are reported separately because only the main call
    # is the user-visible answer, but all of them cost money.
    calls = usage_tracker.for_request(request_id)
    main_calls = [c for c in calls if c.role == ROLE_MAIN and c.success]
    main = main_calls[-1] if main_calls else None

    retrieval = state.get("retrieval")
    quality = state.get("quality") or {}
    kundli = state.get("kundli")

    # What the answer was actually grounded in. Carried on the metrics rather
    # than assembled in the API layer, so the terminal, the logs and the HTTP
    # response all report the same thing.
    sources = [
        {
            "id": f"{chunk.document}::{chunk.topic}",
            "document": chunk.document,
            "domain": chunk.domain,
            "topic": chunk.topic,
            "score": round(chunk.score, 4),
        }
        for chunk in (retrieval.chunks if retrieval else [])
    ]

    metrics: dict[str, Any] = {
        "request_id": request_id,
        "session_id": session_id,
        "user_id": state["user_id"],
        "persona": state.get("persona_id", "unknown"),
        "sources": sources,
        "chart_facts_used": kundli.facts_used() if kundli else [],
        "intent": state.get("intent", "unknown"),
        "intent_confidence": state.get("intent_confidence", 0.0),
        "intent_method": state.get("intent_method", "unknown"),
        "kundli_source": state["kundli"].source if state.get("kundli") else "unknown",
        "kundli_available": state["kundli"].available if state.get("kundli") else False,
        "rag_chunks": retrieval.count if retrieval else 0,
        "rag_available": retrieval.available if retrieval else False,
        "model": main.model if main else state.get("model_used", "none"),
        "fallback_used": state.get("fallback_used", False),
        # None, not 0 — an unreported token count is never invented.
        "input_tokens": main.input_tokens if main else None,
        "output_tokens": main.output_tokens if main else None,
        "total_tokens": main.total_tokens if main else None,
        "cached_tokens": main.cached_tokens if main else None,
        "reasoning_tokens": main.reasoning_tokens if main else None,
        "tokens_available": main.tokens_available if main else False,
        "latency_ms": main.latency_ms if main else 0,
        "turn_latency_ms": int((time.perf_counter() - state["started_at"]) * 1000),
        "estimated_cost_usd": main.total_cost_usd if main else 0.0,
        "native_cost_usd": main.native_cost_usd if main else None,
        "turn_cost_usd": sum(c.total_cost_usd for c in calls if c.success),
        "turn_total_tokens": sum(c.total_tokens or 0 for c in calls if c.success),
        "llm_calls": len(calls),
        "quality": quality,
        "reward": state.get("reward", 0.0),
        "errors": state.get("errors", []),
        "generation_failed": failed,
        "history_persisted": not failed,
        "redis_configured": settings.redis.configured,
    }

    log.info(
        "turn complete",
        extra={
            **{k: v for k, v in metrics.items()
               if k not in {"quality", "sources", "chart_facts_used"}},
            # Ids only — the full chunk text does not belong in a log line.
            "source_ids": [s["id"] for s in sources],
            "chart_facts_count": len(metrics["chart_facts_used"]),
        },
    )
    return {"metrics": metrics}


__all__ = ["generate_response_node", "save_metrics_node", "FAILURE_MESSAGE"]
