"""The Hinglish guard: English replies to a Hinglish question are caught, including short
factual ones, and genuine Hinglish is never flagged."""

from __future__ import annotations

import pytest

from app.nodes.validate import _correction_note
from app.evaluation.chart_claims import ValidationResult
from app.prompts.language import is_english_not_hinglish, language_lock, reply_matches


@pytest.mark.parametrize("text", [
    # Observed live on 2026-11-13 for Hinglish questions (6-word replies were unchecked).
    "Saturn is in the 6th house.",
    "Your Lagna is Virgo and your Moon sign is Pisces.",
    "Jupiter is in the 3rd house in Scorpio.",
    "Saturn (Shani) is in the 6th house in Aquarius, at 29.76 degrees. It is retrograde.",
    "Your 10th house is Gemini, and it has no planet sitting in it. The 10th lord, Mercury, is placed in the 12th house.",
])
def test_plain_english_is_caught_for_hinglish_users(text) -> None:
    assert is_english_not_hinglish(text)
    assert not reply_matches("whatsapp_hinglish", text)


@pytest.mark.parametrize("text", [
    "Aapka Saturn 6th house mein hai.",
    "Shani 6th house mein hai, Aquarius rashi mein.",
    "Aapka lagna Virgo hai aur Moon Pisces mein hai.",
    "Haan, ye ho sakta hai.",
    "Abhi Venus mahadasha chal rahi hai.",
    "6th house health, enemies, aur service se judi hoti hai. Shani yahan retrograde chal rahi hai.",
    "Virgo.",                      # too short to judge
    "Haan.",
])
def test_genuine_hinglish_is_never_flagged(text) -> None:
    assert not is_english_not_hinglish(text)
    assert reply_matches("whatsapp_hinglish", text)


def test_the_guard_does_not_apply_to_english_style() -> None:
    assert reply_matches("english", "Saturn is in the 6th house.")


def test_lock_and_repair_note_say_short_answers_are_not_exempt() -> None:
    assert "one-line factual answer" in language_lock("whatsapp_hinglish")
    note = _correction_note(ValidationResult(), True, "whatsapp_hinglish", False)
    assert "plain English" in note and "hai, mein" in note
    other = _correction_note(ValidationResult(), True, "hindi", False)
    assert "wrong language or script" in other
