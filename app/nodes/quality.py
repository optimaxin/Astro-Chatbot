"""Response evaluation node.

Kept strictly separate from generation: the generator never sees its own
scores, and the evaluator uses a different model, so nothing here can be
turned into a way for the main model to grade itself favourably.
"""

from __future__ import annotations

import random
from typing import Any

from app.config.settings import settings
from app.evaluation import evaluation_store
from app.evaluation.evaluator import evaluate
from app.evaluation.reward import calculate_reward, overall_score
from app.logging_setup import get_logger
from app.rag.retriever import RetrievalResult
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.quality")


def _should_judge() -> bool:
    """Whether this turn is in the LLM judge's sample.

    Random rather than deterministic (every Nth turn) so the sample cannot
    correlate with anything about the traffic — a fixed stride would
    systematically over-sample whichever user happens to land on it.
    """
    rate = settings.evaluation_sample_rate
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate


def evaluate_response_node(state: ChatbotState) -> dict[str, Any]:
    # A failed generation already reported why; grading the fallback message
    # would only produce a meaningless score.
    if state.get("generation_failed"):
        return {"quality": {"skipped": "generation failed"}, "reward": 0.0}

    retrieval = state.get("retrieval") or RetrievalResult.unavailable("retrieval did not run")

    # The LLM judge is a second model call per turn, so production samples it
    # rather than paying for a metric nobody reads live. The deterministic
    # checks cost nothing and always run, so an unsampled turn still catches a
    # prompt leak or an invented placement — sampling reduces the *judge*, not
    # the safety net.
    use_judge = _should_judge()

    result = evaluate(
        question=state["question"],
        response=state.get("response", ""),
        intent=state.get("intent", "general_astrology"),
        kundli=state["kundli"],
        retrieval=retrieval,
        session_id=state["session_id"],
        user_id=state["user_id"],
        request_id=state["request_id"],
        use_judge=use_judge,
    )

    scores = result["scores"]
    reward = calculate_reward(scores, result.get("deterministic_failures"))
    result["overall"] = overall_score(scores)
    result["sampled"] = use_judge

    log.info(
        "reward calculated",
        extra={"reward": reward, "overall": result["overall"],
               "judge_used": result.get("judge_used"), "sampled": use_judge},
    )

    # Persist for the admin dashboard. Best-effort and deliberately last: a
    # storage failure must not cost the user their answer, and the score is
    # already on the state either way.
    evaluation_store.write_evaluation(
        request_id=state["request_id"],
        user_id=state.get("user_id"),
        conversation_id=state.get("session_id"),
        quality=result,
        reward=reward,
    )

    errors = list(state.get("errors", []))
    if result.get("judge_error"):
        errors.append(f"evaluator degraded: {result['judge_error']}")

    return {"quality": result, "reward": reward, "errors": errors}


__all__ = ["evaluate_response_node"]
