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


def test_human_eval_batch_findings_are_caught(facts) -> None:
    """Found reviewing the 2026-11-13 human-eval batch (H001, H030)."""
    assert _bad("Mercury in 11th, lord of the 10th.", facts) == \
        ["says mercury is in the 11th house; chart: 12th house"]
    transit = validate_reply("Jupiter in 3rd transiting your 11th.", facts)
    assert [c.kind for c in transit.unsupported] == ["transit"]
    # …while general rules phrased the same way stay generic.
    assert _bad("Jupiter in the 10th house brings recognition.", facts) == []


@pytest.mark.parametrize("text", [
    # Found in the final live evaluation: a correct sentence flagged twice
    # because a planet was paired with another clause's house.
    "7th house lord Jupiter 3rd mein hai, aur 7th mein Moon hai.",
    # General rules about signs are not placements in this chart.
    "The Moon is exalted in Taurus, so Moon placements there are read as strong.",
    "Mars rules Aries and Scorpio.",
    # A reversed "H mein P" pairing must not cross a clause boundary.
    "Lagna lord Mercury 10th lord bhi 12th mein hai, aur Venus-Sun conjunction 11th mein.",
    # Saturn's transit sign is computed (Sade Sati data): this one is correct.
    "Saturn is transiting Pisces, over your Moon, so Sade Sati is at its peak.",
])
def test_known_false_positives_stay_fixed(text, facts) -> None:
    assert _bad(text, facts) == []


def test_transits_are_checked_where_computed_and_unsupported_elsewhere(facts) -> None:
    assert _bad("Saturn is transiting Aries right now.", facts)            # engine: Pisces
    jupiter = validate_reply("Jupiter is transiting your 11th house.", facts)
    assert not jupiter.contradictions and [c.kind for c in jupiter.unsupported] == ["transit"]


def test_contradictory_interpretation_within_one_reply(facts) -> None:
    text = "Jupiter sits in your 3rd house. That is why Jupiter in your 10th house gives you fame."
    assert _bad(text, facts) == ["says jupiter is in the 10th house; chart: 3rd house"]


def test_conflicting_conversation_context_does_not_override_the_chart(facts) -> None:
    """The user earlier claimed a Leo Lagna; echoing it is still a contradiction."""
    assert _bad("As you mentioned, your Lagna is Leo, so leadership comes naturally.", facts)


# ── unsupported, not false ────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "Jupiter influences your career house.", "You are Manglik.", "Your chart forms a strong Raj yoga.",
    # Live E2E 2026-11-18: negative Manglik verdicts and a Devanagari aspect
    # with a Sanskrit ordinal (and a negation in another clause) all passed.
    "Aapki Kundli mein Manglik dosha nahi hai.",
    "Your chart has no Manglik dosha.",
    "आप मांगलिक नहीं हैं।",
    "शनि की सप्तम पर दृष्टि विलंब का संकेत देती है, लेकिन विवाह की संभावना से इंकार नहीं करती।",
])
def test_unsupported_claims_are_flagged_but_not_called_false(text, facts) -> None:
    result = validate_reply(text, facts)
    assert not result.contradictions and result.unsupported


@pytest.mark.parametrize("text", [
    "Currently Sade Sati is active, and Saturn's transit over your 10th house brings restructuring.",
    "Saturn is transiting your 10th house right now.",
    "Shani ka 10th house par gochar chal raha hai.",
])
def test_saturn_transit_house_is_checked_against_the_computed_transit(text, facts) -> None:
    """Live E2E 2026-11-18: a false premise ("transit over my 10th") was
    repeated as fact. The engine's Saturn transit sign is known, so the house
    it falls in from the Lagna is too."""
    assert validate_reply(text, facts).contradictions


@pytest.mark.parametrize("text", [
    "Saturn is transiting your 7th house right now.",
    "Traditionally, Saturn transiting the 10th house brings restructuring.",
])
def test_saturn_transit_true_or_generic_passes(text, facts) -> None:
    assert _bad(text, facts) == []


def test_hindi_digit_ordinal_aspect_is_unsupported(facts) -> None:
    result = validate_reply("पारंपरिक रूप से, शनि की 7वें भाव पर दृष्टि विवाह में विलंब का संकेत देती है।", facts)
    assert result.unsupported and not result.contradictions


@pytest.mark.parametrize("text", [
    # Conjoined subjects: only the planet next to the house used to be checked.
    "Venus and Mercury are both in your 12th house.",
    "Venus aur Mercury dono aapke 12th house mein hain.",
    "शुक्र और बुध दोनों ही आपके द्वादश भाव में स्थित हैं।",
])
def test_conjoined_planets_are_each_checked(text, facts) -> None:
    assert _bad(text, facts) == ["says venus is in the 12th house; chart: 11th house"]


@pytest.mark.parametrize("text", [
    "Sun and Venus are in your 11th house.",
    "Jupiter aur Venus dono shubh hain, Venus 11th mein hai.",
    "Mercury 12th mein hai, aur Venus-Sun 11th mein.",
])
def test_conjoined_planets_true_placements_pass(text, facts) -> None:
    assert _bad(text, facts) == []


def test_coreference_to_a_house_covers_a_conjoined_pair(facts) -> None:
    assert len(_bad("Aapki Kundli mein 10th house Gemini hai. Shukra aur Surya usme baithi hain.", facts)) == 2
    assert _bad("Aapki Kundli mein 11th house Cancer hai. Shukra aur Surya usme baithe hain.", facts) == []


def test_conflicting_chart_records_are_never_resolved_by_picking_one(kundli) -> None:
    """Two records for one graha: neither is trusted, the prompt says so, and
    any statement of its placement is unsupported (stripped or repaired)."""
    chart = dataclasses.replace(kundli, planets=[*kundli.planets,
                                                 {**kundli.planet("jupiter"), "rashi": "Leo", "house": 12}])
    facts = build_chart_facts(chart)
    assert facts.conflicts == {"jupiter": ["Scorpio/3", "Leo/12"]} and "jupiter" not in facts.planets
    assert "DATA INTEGRITY PROBLEM" in render_prompt_block(chart)
    for text in ("Jupiter aapke 12th house mein hai.", "Jupiter aapke 3rd house mein hai."):
        result = validate_reply(text, facts)
        assert result.unsupported and not result.contradictions
    same = dataclasses.replace(kundli, planets=[*kundli.planets, dict(kundli.planet("jupiter"))])
    assert build_chart_facts(same).conflicts == {}


@pytest.mark.parametrize("text", [
    "Saturn does not aspect your 7th house.",
    "शनि की सातवें भाव पर दृष्टि नहीं है।",
    "आपकी कुंडली में सप्तमेश बृहस्पति 3वें भाव में है।",
])
def test_disclaimers_and_negated_aspects_are_not_flagged(text, facts) -> None:
    result = validate_reply(text, facts)
    assert not result.contradictions and not result.unsupported


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
