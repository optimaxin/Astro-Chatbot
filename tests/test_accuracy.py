"""Chart-fact accuracy regression suite.

Ground truth is the sample chart produced by the real Kundli engine
(data/sample_kundli.json): Lagna Virgo; Sun Cancer/11; Moon Pisces/7;
Mars Virgo/1; Mercury Leo/12; Jupiter Scorpio/3; Venus Cancer/11;
Saturn Aquarius/6 (R); Rahu Libra/2; Ketu Aries/8; Moon nakshatra Uttara
Bhadrapada; Mahadasha Venus; Kaal Sarp absent; Gajakesari absent.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from app.evaluation.chart_claims import strip_sentences, validate_reply
from app.evaluation.evaluator import check_kundli_contradictions
from app.kundli.chart_facts import build_chart_facts, render_prompt_block
from app.kundli.kundli_service import KundliContext

# The exact reply produced live by Nemotron Lightning on 2026-11-12. It copied
# a style example ("Jupiter sits there") into a fact about a chart whose
# Jupiter is in the 3rd. Must never pass validation again.
LIVE_JUPITER_INCIDENT = ("Good news — your career house is strong. Jupiter sits there, and that's "
                         "the planet of growth.\n\nVenus is also in your 11th, which adds income potential.")


@pytest.fixture
def facts(kundli):
    return build_chart_facts(kundli)


def _bad(text, facts):
    return [c.describe() for c in validate_reply(text, facts).contradictions]


# ── the permanent regression ──────────────────────────────────────────
def test_live_jupiter_in_3rd_incident_is_caught(facts) -> None:
    problems = _bad(LIVE_JUPITER_INCIDENT, facts)
    assert problems == ["says jupiter is in the 10th house; chart: 3rd house"]


def test_live_incident_is_also_caught_by_the_evaluator_check(kundli) -> None:
    assert check_kundli_contradictions(LIVE_JUPITER_INCIDENT, kundli)


def test_style_examples_never_contain_a_concrete_placement() -> None:
    """Root cause of the incident: a few-shot example asserted a placement."""
    from app.prompts.response_style import STYLES

    claim = re.compile(r"(?i)\b(sun|moon|mars|mercury|jupiter|venus|saturn|rahu|ketu|shani|guru)\b"
                       r"[^.\n<]{0,40}\b(\d{1,2}(st|nd|rd|th)\s*(house|bhav)|sits there|"
                       r"in (aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|"
                       r"capricorn|aquarius|pisces))|(\d{1,2}(st|nd|rd|th) house (में|mein)\s+\w+)")
    for style in STYLES.values():
        assert not claim.search(style.instruction), f"{style.id} example asserts a placement"


# ── wrong facts, many phrasings ───────────────────────────────────────
@pytest.mark.parametrize("text", [
    "Jupiter is in your 10th house.",
    "Jupiter occupies the tenth Bhava.",
    "Jupiter is positioned in your professional house.",
    "Your tenth house contains Jupiter.",
    "Your career house is strong. Jupiter sits there.",
    "Guru aapke dasve ghar mein baithe hain.",
    "Jupiter tumhare 10th house mein hai.",
    "आपके दसवें भाव में गुरु बैठे हैं।",
    "Saturn sits in your 1st house.",
])
def test_incorrect_house_placement(text, facts) -> None:
    assert _bad(text, facts)


@pytest.mark.parametrize("text", ["Your Moon is in Leo.", "Shani Makar rashi mein hai, aapke chart mein.",
                                  "Your Scorpio Mars gives intensity."])
def test_incorrect_sign(text, facts) -> None:
    assert _bad(text, facts)


@pytest.mark.parametrize("text", ["Your birth nakshatra is Rohini.", "Jupiter is in Ashwini nakshatra."])
def test_incorrect_nakshatra(text, facts) -> None:
    assert _bad(text, facts)


@pytest.mark.parametrize("text", ["Your Lagna is Leo.", "With a Libra ascendant you seek balance.",
                                  "Aapka lagna Tula hai."])
def test_incorrect_ascendant(text, facts) -> None:
    assert _bad(text, facts)


@pytest.mark.parametrize("text", ["You are currently running Saturn mahadasha.",
                                  "Abhi aapki Rahu mahadasha chal rahi hai."])
def test_incorrect_dasha(text, facts) -> None:
    assert _bad(text, facts)


@pytest.mark.parametrize("text", ["Your 7th house is ruled by Venus.", "Mercury rules your 7th house.",
                                  "Mercury is retrograde in your chart.", "You have Kaal Sarp dosha.",
                                  "Your chart has Gajakesari yoga."])
def test_incorrect_lordship_retrograde_yoga_dosha(text, facts) -> None:
    assert _bad(text, facts)


def test_contradictory_interpretation_within_one_reply(facts) -> None:
    text = "Jupiter sits in your 3rd house. That is why Jupiter in your 10th house gives you fame."
    assert _bad(text, facts) == ["says jupiter is in the 10th house; chart: 3rd house"]


def test_conflicting_conversation_context_does_not_override_the_chart(facts) -> None:
    """The user earlier claimed a Leo Lagna; echoing it is still a contradiction."""
    assert _bad("As you mentioned, your Lagna is Leo, so leadership comes naturally.", facts)


# ── unsupported, not false ────────────────────────────────────────────
@pytest.mark.parametrize("text", ["Jupiter influences your career house.", "You are Manglik.",
                                  "Your chart forms a strong Raj yoga."])
def test_unsupported_claims_are_flagged_but_not_called_false(text, facts) -> None:
    result = validate_reply(text, facts)
    assert not result.contradictions and result.unsupported


# ── correct facts, many phrasings: must not be flagged ────────────────
@pytest.mark.parametrize("text", [
    "Jupiter sits in your 3rd house in Scorpio.",
    "Your third house contains Jupiter.",
    "Guru aapke teesre ghar mein hai.",
    "आपके तीसरे भाव में गुरु हैं।",
    "The lord of your Ascendant, Mercury, is in the 12th house.",
    "The lord of your 7th house, Jupiter, sits in the 3rd house.",
    "Your 9th house is ruled by Venus, and Venus is placed in the 11th house.",
    "Your 10th house lord Mercury sits in the 12th house.",
    "Your Moon sign is Pisces and your Lagna is Virgo.",
    "Your birth nakshatra is Uttara Bhadrapada.",
    "You are running Venus mahadasha right now.",
    "Saturn is retrograde, which is traditionally read as delayed results.",
    "Sun and Venus together in your 11th house point to gains.",
    "Jupiter is not in your 10th house; it sits in the 3rd.",
    "Traditionally, Jupiter in the 10th house is read as a sign of recognition.",
    "Your Ascendant is in the 12th house.",
    "Career ke liye 10th house aur uske lord ki position dekhni hogi.",
])
def test_correct_or_generic_statements_pass(text, facts) -> None:
    assert _bad(text, facts) == []


# ── missing birth information ─────────────────────────────────────────
def test_missing_chart_means_no_claims_and_an_explicit_prompt_limit(empty_kundli) -> None:
    assert validate_reply("Your Moon is in Leo.", build_chart_facts(empty_kundli)).claims == []
    block = render_prompt_block(empty_kundli)
    assert "not available" in block and "Do not state or guess" in block


# ── structured context ────────────────────────────────────────────────
def test_prompt_house_table_states_what_the_10th_contains(kundli) -> None:
    block = render_prompt_block(kundli, "career")
    assert "10th (career, status, karma): Gemini, lord Mercury (in 12th); occupants: none" in block
    assert "3rd (courage, siblings, communication): Scorpio, lord Mars (in 1st); occupants: Jupiter" in block
    assert "NOT computed" in block and "Manglik" in block


def test_house_table_is_only_derived_when_whole_sign_is_proven(kundli) -> None:
    broken = dataclasses.replace(kundli, planets=[dict(p) for p in kundli.planets])
    broken.planets[0]["house"] = (broken.planets[0]["house"] % 12) + 1   # not whole-sign any more
    facts = build_chart_facts(broken)
    assert not facts.whole_sign and facts.houses == {}
    assert "HOUSES" not in render_prompt_block(broken)


def test_cached_chart_keeps_degrees_navamsa_and_absent_yogas(kundli) -> None:
    cached = KundliContext.from_facts("u", kundli.to_facts())
    a, b = build_chart_facts(kundli), build_chart_facts(cached)
    assert a.planets == b.planets and a.navamsa == b.navamsa
    assert a.yogas_absent == b.yogas_absent and a.kaal_sarp == b.kaal_sarp is False


def test_stripping_keeps_paragraph_breaks() -> None:
    text = "First line here.\n\nJupiter sits there. Second sentence stays.\n\nLast paragraph."
    assert strip_sentences(text, {1}) == "First line here.\n\nSecond sentence stays.\n\nLast paragraph."
