"""Conversation closings: one short localized line, no main/fallback/humanizer
call; anything that opens something new goes through the normal pipeline; safety
always wins."""

from __future__ import annotations

import pytest

from app.nodes.classifier import INTENTS, route_after_classify
from app.nodes.conversation_gate import closing_reply_node
from app.prompts.closing import (
    COMBINED, CONVERSATION_END, GOOD_WISHES, GOODBYE, GRATITUDE, closing_language, closing_reply, detect_closing,
    infer_closing_kind,
)
from app.prompts.reply_format import reply_messages
from tests.llm_fakes import reply, use_nvidia
from tests.test_model_roles import turn_env  # noqa: F401 — fixture: real graph, fake Redis, fixed chart

BOLT = "nvidia/nemotron-3.5-bolt-e4b"


@pytest.mark.parametrize("text,kind", [
    ("bye", GOODBYE), ("Bye bye", GOODBYE), ("ok bye", GOODBYE), ("alvida", GOODBYE), ("चलो फिर मिलेंगे", GOODBYE),
    ("thank you", GRATITUDE), ("Thanks a lot!", GRATITUDE), ("dhanyawad ji 🙏", GRATITUDE), ("धन्यवाद", GRATITUDE),
    ("bahut bahut shukriya", GRATITUDE), ("thank you for the help", GRATITUDE),
    ("good night", GOOD_WISHES), ("shubh ratri", GOOD_WISHES), ("take care", GOOD_WISHES),
    ("bas itna hi", CONVERSATION_END), ("that's all", CONVERSATION_END), ("और कुछ नहीं", CONVERSATION_END),
    ("thank you, bye", COMBINED), ("Thanks, good night!", COMBINED), ("dhanyawad, phir milenge", COMBINED),
    ("ok thanks, I will ask about my career tomorrow", COMBINED),
])
def test_clear_closings_are_detected(text, kind) -> None:
    assert detect_closing(text) == kind


@pytest.mark.parametrize("text", [
    "Thank you, but ek aur sawaal hai", "thanks, what about my marriage?", "bye the way, when will I get a job?",
    "thanks to Saturn my career is stuck", "thanks for nothing, your prediction was wrong", "ok", "achha",
    "Thank you. Mujhe ek baat poochni hai", "good morning, meri shaadi kab hogi", "bas itna batao ki job kab milegi",
    "I am going to die, bye", "kya aap bye bol sakte ho?",
])
def test_anything_else_is_not_a_closing(text) -> None:
    assert detect_closing(text) is None


@pytest.mark.parametrize("text,conversation,expected", [
    ("thank you", "en", "en"), ("dhanyawad", "en", "hinglish"), ("धन्यवाद", "en", "hi"),
    ("thank you", "hinglish", "hinglish"), ("bye", "hi", "hi"),
])
def test_the_reply_keeps_the_conversation_language(text, conversation, expected) -> None:
    assert closing_language(text, conversation) == expected


@pytest.mark.parametrize("kind", [GOODBYE, GRATITUDE, GOOD_WISHES, CONVERSATION_END, COMBINED])
@pytest.mark.parametrize("lang", ["en", "hinglish", "hi"])
def test_every_reply_is_one_short_bubble(kind, lang) -> None:
    text = closing_reply(kind, "x", lang, seed="s")
    assert len(reply_messages(text)) == 1 and len(text) < 120
    if lang == "hinglish":
        assert not any(w in text.lower() for w in ("you're", "welcome!", "have a lovely"))   # no English sentence
    if lang == "hi":
        assert any("ऀ" <= ch <= "ॿ" for ch in text)


def test_time_of_day_wishes_are_answered_in_kind() -> None:
    assert "night" in closing_reply(GOOD_WISHES, "good night", "en").lower()
    assert "ratri" in closing_reply(GOOD_WISHES, "gn", "hinglish").lower()


def test_the_reply_is_stable_for_a_session() -> None:
    assert closing_reply(GRATITUDE, "thanks", "en", seed="a") == closing_reply(GRATITUDE, "thanks", "en", seed="a")


# ── the classifier path for ambiguous messages ────────────────────────
def test_closing_is_a_classifier_intent_routed_to_the_closing_reply() -> None:
    assert "closing" in INTENTS
    assert route_after_classify({"intent": "closing", "routing_decision": "answer"}) == "closing_reply"


def test_infer_kind_for_an_ambiguous_closing() -> None:
    assert infer_closing_kind("thanks, that reading really helped me today") == GRATITUDE
    assert infer_closing_kind("you've been so helpful, bye") == GOODBYE
    assert infer_closing_kind("ok I'm done for now, will ask later") == CONVERSATION_END
    assert infer_closing_kind("you've been so helpful") == GRATITUDE


def test_the_closing_reply_node_makes_no_call(monkeypatch) -> None:
    _, client = use_nvidia(monkeypatch, [])
    out = closing_reply_node({"question": "that was really helpful, shukriya", "response_style": "english",
                              "session_id": "s"})
    assert out["model_used"] == "rule:closing" and out["closing"]["method"] == "classifier"
    assert not client.requests


# ── end to end through the real graph ─────────────────────────────────
def _turn(question: str, session: str, style: str = "english"):
    from app.graph.chatbot_graph import run_turn

    return run_turn(question, session, "u1", response_style=style)


@pytest.mark.parametrize("text,style", [("thank you", "english"), ("dhanyawad ji", "whatsapp_hinglish"),
                                        ("bye", "hindi"), ("good night", "english")])
def test_a_clear_closing_makes_no_model_call(monkeypatch, turn_env, text, style) -> None:  # noqa: F811
    _, client = use_nvidia(monkeypatch, [])
    state = _turn(text, f"close-{abs(hash(text)) % 10**6}", style)
    assert not client.requests and state["metrics"]["llm_calls"] == 0
    assert state["intent"] == "closing" and state["metrics"]["closing"]["method"] == "deterministic"
    assert state["metrics"]["response_source"] == "deterministic"
    assert len(reply_messages(state["response"])) == 1


def test_an_ambiguous_closing_costs_only_the_classifier(monkeypatch, turn_env) -> None:  # noqa: F811
    from app.config import settings as settings_module
    import dataclasses

    s = settings_module.settings
    original = s.flags
    object.__setattr__(s, "flags", dataclasses.replace(s.flags, classifier_mode="llm"))
    try:
        _, client = use_nvidia(monkeypatch, [reply('{"intent": "closing", "confidence": 0.9, '
                                                   '"is_astrology_related": true, "routing_decision": "answer"}')])
        state = _turn("you have been really helpful today, I feel much better", "close-amb")
    finally:
        object.__setattr__(s, "flags", original)
    assert client.models == [BOLT], "only the classifier ran: no primary, fallback or humanizer call"
    assert state["metrics"]["closing"] == {"kind": GRATITUDE, "method": "classifier"}


def test_a_closing_with_a_new_question_is_answered_normally(monkeypatch, turn_env) -> None:  # noqa: F811
    use_nvidia(monkeypatch, [])
    state = _turn("Thank you, but ek aur sawaal hai — meri shaadi kab hogi?", "close-q", "whatsapp_hinglish")
    assert state.get("intent") != "closing"
