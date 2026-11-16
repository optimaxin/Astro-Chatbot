"""Response generation and metric persistence nodes."""

from __future__ import annotations

import time
from typing import Any

from app.config.settings import settings
from app.llm.llm_factory import ROLE_MAIN, LLMCallFailed, LLMNotConfigured, get_llm
from app.llm.usage_tracker import usage_tracker
from app.logging_setup import get_logger
from app.evaluation.response_quality import VALIDATOR_VERSION
from app.memory.redis_memory import memory
from app.prompts.builder import PROMPT_VERSION
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.generate")

# Shown when the primary and fallback models have both failed. Deliberately
# says nothing astrological — inventing a reading to cover an outage would be
# exactly the failure mode this service is built to avoid.
FAILURE_MESSAGE = (
    "I'm having trouble reaching my astrology model right now, so I'd rather not "
    "guess at an answer. Could you try again in a moment?"
)


_FAILURE_BY_STYLE = {
    "whatsapp_hinglish": ("Abhi mera astrology model respond nahi kar pa raha hai, isliye main andaaze se jawab "
                          "nahi dunga. Kripya thodi der baad phir try kijiye."),
    "hindi": ("अभी मेरा ज्योतिष मॉडल जवाब नहीं दे पा रहा है, इसलिए मैं अनुमान से उत्तर नहीं दूँगा। कृपया थोड़ी देर "
              "बाद फिर से कोशिश करें।"),
}


def failure_message(style_id: str | None) -> str:
    """The outage reply, in the user's language (an English notice was sent to
    Hindi and Hinglish users before 2026-11-15)."""
    return _FAILURE_BY_STYLE.get((style_id or "").lower(), FAILURE_MESSAGE)


def generate_response_node(state: ChatbotState) -> dict[str, Any]:
    messages = state.get("messages") or []
    if not messages:
        return {
            "response": failure_message(state.get("response_style")),
            "model_used": "none",
            "fallback_used": False,
            "generation_failed": True,
            "errors": [*state.get("errors", []), "generation skipped: no prompt was assembled"],
        }

    language = state.get("language") or {}
    if language.get("switch_only"):
        from app.prompts.language_policy import SWITCH_CONFIRMATION, wants_reexplain

        if not (wants_reexplain(state.get("question")) and any(
                m.role == "user" for m in state.get("history") or [])):
            # "I want you to reply in english": nothing to read from the chart.
            # A fixed confirmation in the new language, with no model call —
            # the model's own attempts ranged from a Hinglish follow-up question
            # (production) to a looping apology (benchmark).
            return {"response": SWITCH_CONFIRMATION.get(language.get("language"), SWITCH_CONFIRMATION["english"]),
                    "model_used": "rule:language-switch", "fallback_used": False, "generation_failed": False}

    try:
        # The classifier's intent picks the model (see app.llm.model_router):
        # greetings may go to the light model, anything chart-related to main.
        llm = get_llm(role=ROLE_MAIN, intent=state.get("intent"))
        result = llm.complete(
            messages,
            session_id=state["session_id"],
            user_id=state["user_id"],
            request_id=state["request_id"],
        )
    except (LLMCallFailed, LLMNotConfigured) as exc:
        log.error("generation failed", extra={"error_message": str(exc)[:400]})
        return {
            "response": failure_message(state.get("response_style")),
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

    language = state.get("language") or {}
    if not failed:
        session_update: dict[str, Any] = {"last_intent": state.get("intent", "unknown"),
                                          "last_turn_at": time.time()}
        if language.get("language"):
            session_update["last_language"] = language["language"]
        if language.get("persisted"):
            # An explicit "reply in English from now on" holds for the rest of
            # the conversation (app.prompts.language_policy).
            from app.prompts.language_policy import PREFERENCE_KEY

            session_update[PREFERENCE_KEY] = language["persisted"]
        # One pipelined round trip when available; same writes either way.
        memory.save_turn(session_id, state["question"], state["response"], session_update)

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
        "provider": main.provider if main else None,
        # Why a fallback answered, when one did ("slow_primary" = hedged).
        "main_error_code": main.error_code if main else None,
        "fallback_used": state.get("fallback_used", False),
        "token_source": main.token_source if main else None,
        "estimated_total_tokens": (
            (main.estimated_input_tokens or 0) + (main.estimated_output_tokens or 0)
            if main and main.token_source == "estimated" else None
        ),
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
        "validation": state.get("validation") or {},
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "language": language.get("language"),
        "language_source": language.get("source"),
        "followup": state.get("followup") or {},
        "answer_depth": state.get("answer_depth"),
        "response_style": state.get("response_style"),
        "regenerations": 1 if str((state.get("validation") or {}).get("action", "")).startswith("regenerated") else 0,
        "citations": state.get("citations") or [],
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
    # Grounding/citation quality for the admin panel, every answered turn
    # (the LLM judge is sampled; this is deterministic and always written).
    if not failed:
        from app.evaluation import evaluation_store
        from app.llm.usage_tracker import current_traffic_source

        args = (request_id, state["user_id"], session_id, dict(metrics), current_traffic_source())
        if settings.flags.metering_async:
            from app.observability import background

            background.submit(evaluation_store.write_grounding, *args)
        else:
            evaluation_store.write_grounding(*args)

    return {"metrics": metrics}


__all__ = ["generate_response_node", "save_metrics_node", "FAILURE_MESSAGE", "failure_message"]
