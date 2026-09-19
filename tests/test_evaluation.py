"""Quality evaluation (Phase 14) and reward calculation (Phase 15)."""

from __future__ import annotations

import pytest

from app.evaluation.evaluator import (
    check_kundli_contradictions,
    run_deterministic_checks,
)
from app.evaluation.reward import HARD_FAILURE_CEILING, WEIGHTS, calculate_reward, overall_score
from app.kundli.kundli_service import KundliContext

GOOD = (
    "That's a fair thing to be unsure about. Your chart puts Saturn in the 6th house, "
    "which is traditionally read as the stamina to outlast competition rather than a quick "
    "rise. Are you thinking about growth where you are, or a move somewhere new?"
)


# ── deterministic checks ──────────────────────────────────────────────
def test_good_response_passes_every_check(kundli: KundliContext) -> None:
    checks = run_deterministic_checks(GOOD, "I am confused about my career.", kundli)
    assert checks.all_passed, checks.failures()


def test_empty_response_fails(kundli: KundliContext) -> None:
    for text in ["", "   ", "\n"]:
        checks = run_deterministic_checks(text, "anything", kundli)
        assert not checks.has_response
        assert "has_response" in checks.failures()


def test_length_bounds(kundli: KundliContext) -> None:
    assert not run_deterministic_checks("Yes.", "Will I get a job?", kundli).within_length
    assert not run_deterministic_checks("word " * 1000, "Will I get a job?", kundli).within_length


def test_prompt_leak_is_caught(kundli: KundliContext) -> None:
    leaked = "RETRIEVED ASTROLOGY KNOWLEDGE (ground your reasoning in this): the 10th house..."
    checks = run_deterministic_checks(leaked, "career?", kundli)
    assert not checks.no_prompt_leak


def test_overclaiming_is_caught(kundli: KundliContext) -> None:
    for text in [
        "You will definitely get the promotion this year, no question about it at all here.",
        "This is guaranteed to happen for you within the next six months, without fail.",
        "It is certain that you will marry before the end of next year, absolutely.",
    ]:
        assert not run_deterministic_checks(text, "career?", kundli).no_overclaiming

    assert run_deterministic_checks(
        "This is traditionally read as a supportive period, though nothing is fixed.",
        "career?",
        kundli,
    ).no_overclaiming


def test_a_clarifying_question_counts_as_addressing_the_intent(kundli: KundliContext) -> None:
    """Asking one question instead of answering is the behaviour we ask for."""
    checks = run_deterministic_checks(
        "Happy to look into that. Are you working or studying at the moment?",
        "I am confused about my future.",
        kundli,
    )
    assert checks.addresses_intent


# ── kundli contradiction detection ────────────────────────────────────
def test_real_placements_are_not_flagged(kundli: KundliContext) -> None:
    moon = kundli.planet("moon")
    assert moon is not None
    text = f"Your Moon sits in {moon['rashi']}, in the {moon['house']}th house of your chart."
    assert check_kundli_contradictions(text, kundli) == []


def test_wrong_sign_is_flagged(kundli: KundliContext) -> None:
    moon = kundli.planet("moon")
    assert moon is not None
    wrong = "Leo" if moon["rashi"].lower() != "leo" else "Taurus"
    problems = check_kundli_contradictions(f"Your Moon is in {wrong}, which softens things.", kundli)
    assert problems, "a false sign claim should be caught"
    assert "moon" in problems[0]


def test_wrong_house_is_flagged(kundli: KundliContext) -> None:
    sun = kundli.planet("sun")
    assert sun is not None
    wrong = 1 if sun["house"] != 1 else 2
    problems = check_kundli_contradictions(f"Your Sun is in the {wrong}st house.", kundli)
    assert problems


def test_contradiction_fails_the_grounding_check(kundli: KundliContext) -> None:
    moon = kundli.planet("moon")
    assert moon is not None
    wrong = "Leo" if moon["rashi"].lower() != "leo" else "Taurus"
    checks = run_deterministic_checks(
        f"Your Moon is in {wrong}, which makes you naturally expressive and warm with others.",
        "tell me about my moon",
        kundli,
    )
    assert not checks.no_unsupported_kundli_facts
    assert checks.contradictions


def test_no_chart_means_nothing_to_contradict(empty_kundli: KundliContext) -> None:
    assert check_kundli_contradictions("Your Moon is in Leo.", empty_kundli) == []


def test_lord_of_a_house_is_not_a_claim_about_that_body(kundli: KundliContext) -> None:
    """Regression: "the lord of your Ascendant, Mercury, is in the 12th house"
    is a statement about MERCURY. Vedic writing leans on this construction
    constantly, so mis-parsing it flagged correct answers as hallucinations."""
    mercury = kundli.planet("mercury")
    assert mercury is not None
    text = (
        f"With your Ascendant in {kundli.ascendant['rashi']}, you are analytical. "
        f"The lord of your Ascendant, Mercury, is in the {mercury['house']}th house, "
        "which suggests an introspective quality."
    )
    assert check_kundli_contradictions(text, kundli) == []


def test_ascendant_is_never_flagged_for_a_house(kundli: KundliContext) -> None:
    """The Ascendant defines house 1, so it has no house placement to contradict."""
    assert check_kundli_contradictions("Your Ascendant is in the 12th house.", kundli) == []


def test_an_intervening_planet_owns_the_placement(kundli: KundliContext) -> None:
    venus = kundli.planet("venus")
    assert venus is not None
    text = f"Your 9th house is ruled by Venus, and Venus is placed in the {venus['house']}th house."
    assert check_kundli_contradictions(text, kundli) == []


def test_a_genuine_contradiction_still_gets_caught_after_the_fix(kundli: KundliContext) -> None:
    """The exclusions must not blunt the check itself."""
    saturn = kundli.planet("saturn")
    assert saturn is not None
    wrong = 1 if saturn["house"] != 1 else 2
    assert check_kundli_contradictions(f"Your Saturn is in the {wrong}st house.", kundli)


# ── reward ────────────────────────────────────────────────────────────
def test_weights_sum_to_one() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_perfect_and_zero_scores() -> None:
    assert calculate_reward({k: 10 for k in WEIGHTS}) == pytest.approx(1.0)
    assert calculate_reward({k: 0 for k in WEIGHTS}) == pytest.approx(0.0)


def test_reward_matches_the_documented_formula() -> None:
    scores = {
        "relevance": 8, "personalization": 9, "groundedness": 8,
        "conversation_quality": 9, "safety": 10, "completeness": 7,
    }
    expected = (
        0.25 * 8 + 0.20 * 9 + 0.15 * 8 + 0.15 * 9 + 0.15 * 10 + 0.10 * 7
    ) / 10.0
    assert calculate_reward(scores) == pytest.approx(expected, abs=1e-4)
    assert 0.0 <= calculate_reward(scores) <= 1.0


def test_reward_is_always_normalised() -> None:
    assert calculate_reward({k: 999 for k in WEIGHTS}) == 1.0
    assert calculate_reward({k: -5 for k in WEIGHTS}) == 0.0


def test_missing_dimensions_renormalise_rather_than_penalise() -> None:
    """A partial score set stays comparable instead of being dragged toward zero."""
    assert calculate_reward({"relevance": 10, "safety": 10}) == pytest.approx(1.0)
    assert calculate_reward({}) == 0.0
    assert calculate_reward({"relevance": "bad"}) == 0.0


def test_hard_failures_cap_the_reward() -> None:
    perfect = {k: 10 for k in WEIGHTS}
    assert calculate_reward(perfect) == pytest.approx(1.0)
    assert calculate_reward(perfect, ["no_prompt_leak"]) <= HARD_FAILURE_CEILING
    assert calculate_reward(perfect, ["no_unsupported_kundli_facts"]) <= HARD_FAILURE_CEILING
    # A soft failure is not capped — it is already reflected in the scores.
    assert calculate_reward(perfect, ["within_length"]) == pytest.approx(1.0)


def test_overall_score_is_the_reward_on_a_ten_scale() -> None:
    scores = {k: 8 for k in WEIGHTS}
    assert overall_score(scores) == pytest.approx(calculate_reward(scores) * 10, abs=0.05)
