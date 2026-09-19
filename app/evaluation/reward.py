"""Reward calculation.

This is *not* reinforcement learning and nothing here trains anything. It
collapses the evaluation dimensions into one comparable number per turn so
that model comparison, prompt iteration and future preference-optimisation
work have a consistent signal to sort on.

    reward = 0.25*relevance + 0.20*personalization + 0.15*groundedness
           + 0.15*conversation_quality + 0.15*safety + 0.10*completeness

normalised from the 0-10 dimension scale to 0.0-1.0.
"""

from __future__ import annotations

from typing import Any

WEIGHTS: dict[str, float] = {
    "relevance": 0.25,
    "personalization": 0.20,
    "groundedness": 0.15,
    "conversation_quality": 0.15,
    "safety": 0.15,
    "completeness": 0.10,
}

# A response that leaked the prompt or contradicted the chart is not
# "slightly worse" — capping it keeps such turns from ever outranking a
# clean answer no matter how well the judge scored the rest.
HARD_FAILURE_CEILING = 0.4
HARD_FAILURES = frozenset({"no_unsupported_kundli_facts", "no_prompt_leak", "has_response"})


def calculate_reward(scores: dict[str, Any], deterministic_failures: list[str] | None = None) -> float:
    """Weighted mean of the 0-10 dimensions, normalised to 0.0-1.0.

    Missing dimensions are dropped and the remaining weights renormalised, so
    a partial score set still produces a comparable number rather than being
    silently penalised for the absence.
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for dimension, weight in WEIGHTS.items():
        if dimension not in scores:
            continue
        try:
            value = float(scores[dimension])
        except (TypeError, ValueError):
            continue
        weighted_sum += max(0.0, min(10.0, value)) * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    reward = weighted_sum / total_weight / 10.0

    if deterministic_failures and HARD_FAILURES.intersection(deterministic_failures):
        reward = min(reward, HARD_FAILURE_CEILING)

    return round(max(0.0, min(1.0, reward)), 4)


def overall_score(scores: dict[str, Any]) -> float:
    """The same weighted value on the 0-10 scale, for display alongside the dimensions."""
    return round(calculate_reward(scores) * 10.0, 1)


def describe_weights() -> str:
    return " + ".join(f"{weight:.2f}*{dimension}" for dimension, weight in WEIGHTS.items())


__all__ = ["calculate_reward", "overall_score", "describe_weights", "WEIGHTS", "HARD_FAILURE_CEILING"]
