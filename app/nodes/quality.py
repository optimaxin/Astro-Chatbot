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
    run_async = settings.flags.evaluation_async
    kwargs = dict(
        question=state["question"],
        response=state.get("response", ""),
        intent=state.get("intent", "general_astrology"),
        kundli=state["kundli"],
        retrieval=retrieval,
        session_id=state["session_id"],
        user_id=state["user_id"],
        request_id=state["request_id"],
    )

    # Deterministic checks are free and always synchronous. With
    # EVALUATION_ASYNC the LLM judge — a second model call that no user-facing
    # field depends on — runs after the reply has been returned.
    result, reward = _score(use_judge=use_judge and not run_async, **kwargs)
    result["sampled"] = use_judge
    if run_async:
        from app.observability import background

        result["judge_pending"] = use_judge
        background.submit(_finish_in_background, use_judge, dict(kwargs))
    else:
        _persist(state["request_id"], state.get("user_id"), state.get("session_id"), result, reward)

    errors = list(state.get("errors", []))
    if result.get("judge_error"):
        errors.append(f"evaluator degraded: {result['judge_error']}")

    return {"quality": result, "reward": reward if reward is not None else 0.0, "errors": errors}


def _score(*, use_judge: bool, **kwargs) -> tuple[dict[str, Any], float | None]:
    result = evaluate(use_judge=use_judge, **kwargs)
    scores = result["scores"]
    if result.get("evaluation_status") == "unavailable":
        # No judge answer: no reward and no overall score — "unavailable" must never read
        # as a pass (or as a mediocre 5/10). Deterministic failures are still recorded.
        result["overall"] = None
        log.warning("evaluation unavailable", extra={"judge_error": str(result.get("judge_error"))[:160]})
        return result, None
    reward = calculate_reward(scores, result.get("deterministic_failures"))
    result["overall"] = overall_score(scores)
    log.info(
        "reward calculated",
        extra={"reward": reward, "overall": result["overall"],
               "judge_used": result.get("judge_used"), "sampled": use_judge},
    )
    return result, reward


def _persist(request_id: str, user_id: str | None, session_id: str | None,
             result: dict[str, Any], reward: float | None) -> None:
    # Best-effort: a storage failure must not cost the user their answer.
    evaluation_store.write_evaluation(
        request_id=request_id,
        user_id=user_id,
        conversation_id=session_id,
        quality=result,
        reward=reward,
    )


def _finish_in_background(use_judge: bool, kwargs: dict[str, Any]) -> None:
    """Judge (when sampled) and persist, off the request path. The stored
    row is the judged one, so the admin dashboard is unchanged."""
    import time

    started = time.perf_counter()
    result, reward = _score(use_judge=use_judge, **kwargs)
    result["sampled"] = use_judge
    result["evaluation_ms"] = int((time.perf_counter() - started) * 1000)
    _persist(kwargs["request_id"], kwargs.get("user_id"), kwargs.get("session_id"), result, reward)


__all__ = ["evaluate_response_node"]
