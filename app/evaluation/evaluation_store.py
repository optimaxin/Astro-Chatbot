"""Evaluation scores and user feedback, persisted for the admin dashboard.

Sits alongside `llm/usage_store.py` and follows the same contract: ai-core
writes, the Node backend only reads, and everything here is best-effort. A
metering or scoring row is worth strictly less than a user's answer, so every
failure path returns False rather than raising.

Three things are kept deliberately distinct, because collapsing them would
destroy the comparison the whole system exists to make:

    reward_score   an internal estimate, produced by a model
    feedback       what a real person said
    tokens / cost  what it took to produce  (in ai_usage)

Joined on `request_id`, those answer "is the expensive answer the good one".
Merged into a single "score", they answer nothing.
"""

from __future__ import annotations

import threading
from typing import Any

from app.config.settings import settings
from app.logging_setup import get_logger

log = get_logger("evaluation.store")

# Bumped when the dimensions or the weights change, so scores produced under
# different rubrics are never averaged together.
EVALUATION_VERSION = "v1"

_pool: Any = None
_pool_lock = threading.Lock()
_disabled = False

# Re-running the evaluator on a turn replaces the old verdict rather than
# adding a second one, which would double-count it in every average.
UPSERT_EVALUATION_SQL = """
INSERT INTO ai_evaluations (
    request_id, user_id, conversation_id,
    relevance_score, grounding_score, personalization_score, completeness_score,
    human_likeness_score, hinglish_quality_score, whatsapp_style_score,
    clarity_score, conciseness_score, warmth_score, safety_score,
    reward_score, deterministic_failures, judge_used,
    evaluator_provider, evaluator_model, evaluation_version
) VALUES (
    %(request_id)s, %(user_id)s, %(conversation_id)s,
    %(relevance)s, %(groundedness)s, %(personalization)s, %(completeness)s,
    %(human_likeness)s, %(hinglish_quality)s, %(whatsapp_style)s,
    %(clarity)s, %(conciseness)s, %(warmth)s, %(safety)s,
    %(reward_score)s, %(deterministic_failures)s, %(judge_used)s,
    %(evaluator_provider)s, %(evaluator_model)s, %(evaluation_version)s
)
ON CONFLICT (request_id) DO UPDATE SET
    relevance_score        = EXCLUDED.relevance_score,
    grounding_score        = EXCLUDED.grounding_score,
    personalization_score  = EXCLUDED.personalization_score,
    completeness_score     = EXCLUDED.completeness_score,
    human_likeness_score   = EXCLUDED.human_likeness_score,
    hinglish_quality_score = EXCLUDED.hinglish_quality_score,
    whatsapp_style_score   = EXCLUDED.whatsapp_style_score,
    clarity_score          = EXCLUDED.clarity_score,
    conciseness_score      = EXCLUDED.conciseness_score,
    warmth_score           = EXCLUDED.warmth_score,
    safety_score           = EXCLUDED.safety_score,
    reward_score           = EXCLUDED.reward_score,
    deterministic_failures = EXCLUDED.deterministic_failures,
    judge_used             = EXCLUDED.judge_used,
    evaluator_provider     = EXCLUDED.evaluator_provider,
    evaluator_model        = EXCLUDED.evaluator_model,
    evaluation_version     = EXCLUDED.evaluation_version
"""

# One verdict per user per turn. Clicking 👎 after 👍 is a correction, not a
# second vote, so it updates in place.
UPSERT_FEEDBACK_SQL = """
INSERT INTO ai_feedback (
    request_id, user_id, conversation_id,
    feedback_type, rating, feedback_reason, feedback_comment
) VALUES (
    %(request_id)s, %(user_id)s, %(conversation_id)s,
    %(feedback_type)s, %(rating)s, %(feedback_reason)s, %(feedback_comment)s
)
ON CONFLICT (request_id, user_id) DO UPDATE SET
    feedback_type    = EXCLUDED.feedback_type,
    rating           = EXCLUDED.rating,
    feedback_reason  = EXCLUDED.feedback_reason,
    feedback_comment = EXCLUDED.feedback_comment,
    updated_at       = NOW()
"""


def _get_pool() -> Any:
    """Shares the usage store's pool settings; opened lazily, once."""
    global _pool, _disabled
    if _disabled:
        return None
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool
        if not settings.usage_store.enabled:
            _disabled = True
            return None
        try:
            from psycopg_pool import ConnectionPool

            _pool = ConnectionPool(
                settings.usage_store.dsn,
                min_size=0,
                max_size=settings.usage_store.pool_size,
                timeout=settings.usage_store.timeout_seconds,
                open=True,
            )
        except Exception as exc:  # noqa: BLE001 — never break a turn
            _disabled = True
            log.warning(
                "evaluation store unavailable; scores will not reach the admin dashboard",
                extra={"error_type": type(exc).__name__, "error_message": str(exc)[:200]},
            )
            return None
    return _pool


def _normalise(value: Any) -> float | None:
    """0-10 judge scale -> the 0-1 scale the database and dashboard use.

    None in, None out: a dimension the judge did not return is absent, not
    zero, and must not be averaged as if it scored nothing.
    """
    if value is None:
        return None
    try:
        return round(max(0.0, min(10.0, float(value))) / 10.0, 3)
    except (TypeError, ValueError):
        return None


DIMENSIONS = (
    "relevance", "groundedness", "personalization", "completeness",
    "human_likeness", "hinglish_quality", "whatsapp_style",
    "clarity", "conciseness", "warmth", "safety",
)


def to_row(
    request_id: str,
    user_id: str | None,
    conversation_id: str | None,
    quality: dict[str, Any],
    reward: float | None,
) -> dict[str, Any]:
    """Map a `quality` metrics block onto the ai_evaluations column set.

    Pure and importable, so the mapping is testable without a database.
    """
    scores = quality.get("scores") or {}
    row: dict[str, Any] = {
        "request_id": request_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "reward_score": None if reward is None else round(float(reward), 3),
        "deterministic_failures": list(quality.get("deterministic_failures") or []),
        "judge_used": bool(quality.get("judge_used")),
        "evaluator_provider": quality.get("evaluator_provider"),
        "evaluator_model": quality.get("evaluator_model"),
        "evaluation_version": EVALUATION_VERSION,
    }
    for dimension in DIMENSIONS:
        row[dimension] = _normalise(scores.get(dimension))
    return row


def write_evaluation(
    request_id: str,
    user_id: str | None,
    conversation_id: str | None,
    quality: dict[str, Any],
    reward: float | None,
) -> bool:
    """Persist one evaluation. Never raises."""
    pool = _get_pool()
    if pool is None:
        return False
    try:
        with pool.connection() as conn:
            conn.execute(
                UPSERT_EVALUATION_SQL,
                to_row(request_id, user_id, conversation_id, quality, reward),
            )
        return True
    except Exception as exc:  # noqa: BLE001 — see module docstring
        log.warning(
            "ai_evaluations write failed",
            extra={"request_id": request_id, "error_type": type(exc).__name__,
                   "error_message": str(exc)[:200]},
        )
        return False


def write_feedback(
    request_id: str,
    user_id: str,
    feedback_type: str,
    conversation_id: str | None = None,
    rating: int | None = None,
    feedback_reason: str | None = None,
    feedback_comment: str | None = None,
) -> bool:
    """Record (or correct) what the user said about a reply. Never raises."""
    pool = _get_pool()
    if pool is None:
        return False
    try:
        with pool.connection() as conn:
            conn.execute(UPSERT_FEEDBACK_SQL, {
                "request_id": request_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "feedback_type": feedback_type,
                "rating": rating,
                "feedback_reason": feedback_reason,
                "feedback_comment": feedback_comment,
            })
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "ai_feedback write failed",
            extra={"request_id": request_id, "error_type": type(exc).__name__,
                   "error_message": str(exc)[:200]},
        )
        return False


def ping() -> tuple[bool, str]:
    """Used by scripts/test_connections.py."""
    if not settings.usage_store.enabled:
        return False, "AI_USAGE_DATABASE_URL is not set — evaluation will not be stored"
    pool = _get_pool()
    if pool is None:
        return False, "could not open a connection pool"
    try:
        with pool.connection() as conn:
            evaluations = conn.execute("SELECT COUNT(*) FROM ai_evaluations").fetchone()[0]
            feedback = conn.execute("SELECT COUNT(*) FROM ai_feedback").fetchone()[0]
        return True, f"{evaluations} evaluation(s), {feedback} feedback row(s)"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"


__all__ = [
    "write_evaluation",
    "write_feedback",
    "to_row",
    "ping",
    "DIMENSIONS",
    "EVALUATION_VERSION",
]
