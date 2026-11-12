"""The benchmark suite itself: dataset integrity, image rendering and the
scoring rules, so a broken harness cannot produce a plausible-looking report."""

from __future__ import annotations

from collections import Counter

import pytest

from benchmarks.cases import CASES, TOOLS
from benchmarks.images import render
from benchmarks.run import build_messages
from benchmarks.scoring import language_ok, parse_judge, score_case, tool_ok


def test_dataset_has_220_cases_in_the_planned_mix() -> None:
    assert len(CASES) == 220
    assert Counter(c["category"] for c in CASES) == {
        "general": 35, "rag": 40, "reasoning": 30, "coding": 25,
        "multilingual": 35, "function_calling": 30, "image": 25,
    }


@pytest.mark.parametrize("case", [c for c in CASES if c["category"] == "image"], ids=lambda c: c["id"])
def test_every_image_renders(case) -> None:
    image = render(case["image"])
    assert image.width > 100 and image.height > 50


def test_messages_are_identical_for_every_model_and_carry_sources() -> None:
    rag = next(c for c in CASES if c["category"] == "rag")
    assert build_messages(rag) == build_messages(rag)
    assert "[S1]" in build_messages(rag)[0]["content"]


def test_function_tools_are_valid_openai_definitions() -> None:
    for name, tool in TOOLS.items():
        assert tool["type"] == "function" and tool["function"]["name"] == name


def test_rag_scoring_requires_citation_and_abstention() -> None:
    answerable = next(c for c in CASES if c["category"] == "rag" and c["checks"].get("cite"))
    assert not score_case(answerable, "Something vague.", [])["grounded"]
    unanswerable = next(c for c in CASES if c["category"] == "rag"
                        and c["checks"].get("answerable") is False)
    good = "The sources do not contain that information, so I can't say."
    assert score_case(unanswerable, good, [])["grounded"]


def test_tool_scoring() -> None:
    expected = {"name": "get_panchang", "args": {"date": "2026-03-14", "city": "Varanasi"}}
    call = {"name": "get_panchang", "parsed_arguments": {"date": "2026-03-14", "city": "varanasi"}}
    assert tool_ok(expected, [call])[0]
    assert not tool_ok(expected, [])[0]
    assert not tool_ok(None, [call])[0]
    assert tool_ok(None, [])[0]


def test_language_detection() -> None:
    assert language_ok("आपकी कुंडली में शनि दसवें भाव में है।", "hindi_devanagari")
    assert language_ok("Aapki shaadi ke yog 2027 mein hai, lekin dhairya rakhiye.", "hinglish")
    assert not language_ok("Your marriage prospects look good next year.", "hinglish")
    assert language_ok("உங்கள் ஜாதகத்தில் சனி உள்ளது", "tamil")


def test_judge_output_is_validated() -> None:
    assert parse_judge('{"quality": 4, "factual": 5, "grounded": null}')["quality"] == 4.0
    assert parse_judge('{"quality": 9}')["quality"] is None
    assert parse_judge("not json") is None
