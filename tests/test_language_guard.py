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
    assert "every sentence" in language_lock("whatsapp_hinglish")
    note = _correction_note(ValidationResult(), True, "whatsapp_hinglish", False)
    assert "Roman-script Hinglish" in note and "hai, mein" in note
    other = _correction_note(ValidationResult(), True, "hindi", False)
    assert "Devanagari" in other
    english = _correction_note(ValidationResult(), True, "english", False)
    assert "natural English" in english


# ── a Hinglish question after a Devanagari Hindi answer (live browser test) ──
_HINDI_REPLY = "आपकी कुंडली में 7वां भाव कन्या है, और उसका स्वामी बुध 6वें भाव में है।"
_HINGLISH_REPLY = "Aapki Kundli mein 7th house Virgo hai aur uska lord Mercury 6th house mein hai."


def test_turn_note_only_when_the_previous_reply_is_in_another_language() -> None:
    from app.prompts.language import turn_language_note

    assert "Latin letters only" in turn_language_note("whatsapp_hinglish", _HINDI_REPLY)
    assert turn_language_note("whatsapp_hinglish", _HINGLISH_REPLY) is None
    assert turn_language_note("whatsapp_hinglish", None) is None
    assert "देवनागरी" in turn_language_note("hindi", _HINGLISH_REPLY)
    assert turn_language_note("hindi", _HINDI_REPLY) is None
    assert "English only" in turn_language_note("english", _HINDI_REPLY)


def test_build_messages_attaches_the_note_to_the_current_question_only(kundli, retrieval) -> None:
    from app.memory.redis_memory import Message
    from app.prompts.builder import build_messages

    history = [Message(role="user", content="मुझे मेरी शादी के बारे में बताइए।", timestamp=0.0),
               Message(role="assistant", content=_HINDI_REPLY, timestamp=1.0)]
    msgs = build_messages("Saturn ka asar kya hai?", "general_astrology", kundli,
                          retrieval, history, chat_bubbles=True,
                          response_style="whatsapp_hinglish")
    assert msgs[-1]["content"].startswith("Saturn ka asar kya hai?")
    assert "Latin letters only" in msgs[-1]["content"]
    assert msgs[2]["content"] == _HINDI_REPLY                 # history itself untouched
    same = build_messages("Aur career?", "career", kundli, retrieval,
                          [Message(role="assistant", content=_HINGLISH_REPLY, timestamp=1.0)], chat_bubbles=True,
                          response_style="whatsapp_hinglish")
    assert same[-1]["content"] == "Aur career?"


def test_wrong_script_draft_is_regenerated_not_rewritten() -> None:
    from app.prompts.language import language_repair_messages

    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "old"},
            {"role": "assistant", "content": _HINDI_REPLY}, {"role": "user", "content": "Saturn ka asar?"}]
    fresh = language_repair_messages(msgs, _HINDI_REPLY + " " + _HINDI_REPLY, "NOTE", "whatsapp_hinglish")
    assert [m["role"] for m in fresh] == ["system", "user"]
    assert fresh[1]["content"].startswith("Saturn ka asar?") and "Latin letters only" in fresh[1]["content"]
    assert msgs[-1]["content"] == "Saturn ka asar?"           # caller's list not mutated
    # Right script, wrong language (English in a Hinglish reply): still a rewrite, without old turns.
    english = "Saturn is in the 4th house and affects your home life and health over time."
    rewrite = language_repair_messages(msgs, english, "NOTE", "whatsapp_hinglish")
    assert [m["content"] for m in rewrite] == ["SYS", "Saturn ka asar?", english, "NOTE"]
