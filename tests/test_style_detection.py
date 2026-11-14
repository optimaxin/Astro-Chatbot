"""Which language a reply should be in: detection from the message, and from the
conversation when the message alone gives no signal."""

from __future__ import annotations

import pytest

from app.prompts.response_style import detect_style, language_signal, needs_history, resolve_style

HINGLISH = "whatsapp_hinglish"


@pytest.mark.parametrize("text", [
    "bhaii koi naam",                 # reported live 2026-11-14: was answered in English
    "bhai koi naam batao", "koi naam", "yaarr shaadi kab", "naukri kab lagegi",
    "Mera lagna kya hai?", "mujhe career ke baare mein batao", "acha thik hai",
])
def test_hinglish_messages_are_detected(text) -> None:
    assert detect_style(text) == HINGLISH


@pytest.mark.parametrize("text", [
    "What is my current dasha about?", "Give me a name", "Is Saturn bad for me?",
    "Who is the main person in my life?", "Tell me more", "How will my career go this year?",
    "Explain my 10th house", "Hi", "Thanks", "My rashi and dasha please",
])
def test_english_messages_stay_english(text) -> None:
    assert detect_style(text) == "english"


def test_devanagari_and_explicit_requests_still_win() -> None:
    assert detect_style("मेरी शादी कब होगी?") == "hindi"
    assert detect_style("English mein batao, shaadi kab hogi") == "english"


@pytest.mark.parametrize("text", ["ok", "hmm", "5th house?", "??"])
def test_messages_without_a_signal_follow_the_conversation(text) -> None:
    assert language_signal(text) is None
    assert detect_style(text) == "english"                                   # nothing to go on
    assert detect_style(text, ["Meri shaadi kab hogi?"]) == HINGLISH         # keeps Hinglish
    assert detect_style(text, ["Meri shaadi kab hogi?", "What about my career?"]) == "english"  # latest wins


def test_history_is_only_needed_when_auto_detecting_an_ambiguous_message() -> None:
    assert needs_history(None, "ok") is True
    assert needs_history(None, "bhai koi naam") is False
    assert needs_history("english", "ok") is False                          # explicit style
    assert needs_history(None, "ok", default="english") is False            # deployment pins a style
    assert resolve_style(None, "ok", history=["Mera lagna kya hai?"]).id == HINGLISH


def test_chat_endpoint_keeps_hinglish_for_a_short_follow_up(monkeypatch) -> None:
    """End to end through /chat: the second, signal-less message is answered in
    the language of the first."""
    from app.api import server
    from app.memory.redis_memory import RedisMemory, memory
    from tests.test_chatbot_mvp import FakeRedis

    monkeypatch.setattr(memory, "_client", FakeRedis())
    monkeypatch.setattr(RedisMemory, "configured", property(lambda self: True))
    seen = []

    def fake_run_turn(question, session_id, user_id, persona_id, **kwargs):
        seen.append(kwargs["response_style"])
        memory.save_message(session_id, "user", question)
        return {"response": "ok", "metrics": {"request_id": f"r{len(seen)}", "generation_failed": False,
                                              "errors": []}}

    monkeypatch.setattr(server, "run_turn", fake_run_turn)
    client = pytest.importorskip("fastapi.testclient").TestClient(server.app)
    body = {"user_id": "u1", "session_id": "s-lang", "question": "Meri shaadi kab hogi?"}
    assert client.post("/chat", json=body).status_code == 200
    assert client.post("/chat", json={**body, "question": "ok"}).status_code == 200
    assert seen == [HINGLISH, HINGLISH]
