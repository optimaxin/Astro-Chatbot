"""Response evaluation node.

Kept strictly separate from generation: the generator never sees its own
scores, and the evaluator uses a different model, so nothing here can be
turned into a way for the main model to grade itself favourably.
"""

from __future__ import annotations

from typing import Any

from app.evaluation.evaluator import evaluate
from app.evaluation.reward import calculate_reward, overall_score
from app.logging_setup import get_logger
from app.rag.retriever import RetrievalResult
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.quality")


def evaluate_response_node(state: ChatbotState) -> dict[str, Any]:
    # A failed generation already reported why; grading the fallback message
    # would only produce a meaningless score.
    if state.get("generation_failed"):
        return {"quality": {"skipped": "generation failed"}, "reward": 0.0}

    retrieval = state.get("retrieval") or RetrievalResult.unavailable("retrieval did not run")

    result = evaluate(
        question=state["question"],
        response=state.get("response", ""),
        intent=state.get("intent", "general_astrology"),
        kundli=state["kundli"],
        retrieval=retrieval,
        session_id=state["session_id"],
        user_id=state["user_id"],
        request_id=state["request_id"],
    )

    scores = result["scores"]
    reward = calculate_reward(scores, result.get("deterministic_failures"))
    result["overall"] = overall_score(scores)

    log.info("reward calculated", extra={"reward": reward, "overall": result["overall"]})

    errors = list(state.get("errors", []))
    if result.get("judge_error"):
        errors.append(f"evaluator degraded: {result['judge_error']}")

    return {"quality": result, "reward": reward, "errors": errors}


__all__ = ["evaluate_response_node"]
