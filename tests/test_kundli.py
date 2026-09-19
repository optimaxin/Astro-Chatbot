"""Kundli loading and rendering (Phase 6)."""

from __future__ import annotations

from app.kundli.kundli_service import (
    HOUSE_THEMES,
    VEDIC_BODIES,
    KundliContext,
    SampleFileKundliProvider,
)


def test_sample_kundli_loads(kundli: KundliContext) -> None:
    assert kundli.available
    assert kundli.source == "sample-file"
    assert kundli.ascendant["rashi"]
    assert kundli.moon_nakshatra["name"]


def test_all_vedic_bodies_present(kundli: KundliContext) -> None:
    for body in VEDIC_BODIES:
        placement = kundli.planet(body)
        assert placement is not None, f"{body} missing from the sample chart"
        assert placement["rashi"], f"{body} has no sign"
        assert 1 <= placement["house"] <= 12, f"{body} has an impossible house"


def test_prompt_block_states_only_real_placements(kundli: KundliContext) -> None:
    block = kundli.to_prompt_block("career")
    moon = kundli.planet("moon")
    assert moon is not None
    assert moon["rashi"] in block
    assert kundli.ascendant["rashi"] in block
    # The chart is presented as fact, and the model is told so explicitly.
    assert "treat as fact" in block


def test_prompt_block_is_intent_aware(kundli: KundliContext) -> None:
    career = kundli.to_prompt_block("career")
    marriage = kundli.to_prompt_block("marriage")
    assert career != marriage
    assert "'career' question" in career
    assert "'marriage' question" in marriage
    # The house sets actually differ, which is the point of the branch.
    assert HOUSE_THEMES["career"] != HOUSE_THEMES["marriage"]


def test_unavailable_kundli_tells_the_model_not_to_guess(empty_kundli: KundliContext) -> None:
    block = empty_kundli.to_prompt_block("career")
    assert "not available" in block
    assert "Do not state or guess" in block
    assert empty_kundli.fact_index() == {}


def test_fact_index_matches_the_chart(kundli: KundliContext) -> None:
    facts = kundli.fact_index()
    moon = kundli.planet("moon")
    assert moon is not None
    assert moon["rashi"].lower() in facts["moon"]
    assert str(moon["house"]) in facts["moon"]
    assert kundli.ascendant["rashi"].lower() in facts["ascendant"]


def test_missing_file_degrades_instead_of_raising(tmp_path) -> None:
    provider = SampleFileKundliProvider(path=tmp_path / "does-not-exist.json")
    context = provider.get_kundli("nobody")
    assert not context.available
    assert context.error is not None
    assert "not found" in context.error


def test_malformed_file_degrades_instead_of_raising(tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    context = SampleFileKundliProvider(path=broken).get_kundli("nobody")
    assert not context.available
    assert context.error is not None
    assert "not valid JSON" in context.error
