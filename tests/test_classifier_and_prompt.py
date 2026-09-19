"""Intent classification (Phase 7) and prompt construction (Phase 10)."""

from __future__ import annotations

import pytest

from app.kundli.kundli_service import KundliContext
from app.memory.redis_memory import Message
from app.nodes.classifier import INTENTS, classify_by_keywords
from app.prompts.builder import DOMAIN_MODULES, build_messages, build_system_prompt
from app.rag.retriever import RetrievalResult
from tests.conftest import needs_openrouter
from tests.test_questions import QUESTIONS


# ── classification ────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I am confused about my career", "career"),
        ("Will I get a promotion this year?", "career"),
        ("When will I get married?", "marriage"),
        ("Is he the right partner for me romantically?", "love"),
        ("I keep fighting with my family", "relationship"),
        ("Will I ever have enough savings?", "finance"),
        ("I've had a lot of anxiety lately", "health"),
        ("What is my life purpose?", "personal_growth"),
        ("What is a nakshatra?", "general_astrology"),
        ("hello there", "smalltalk"),
    ],
)
def test_keyword_classifier(text: str, expected: str) -> None:
    result = classify_by_keywords(text)
    assert result.intent == expected
    assert 0.0 <= result.confidence <= 1.0


def test_keyword_classifier_always_returns_a_known_intent() -> None:
    for text in ["", "asdkjhasd", "???", "tell me something"]:
        assert classify_by_keywords(text).intent in INTENTS


def test_marriage_beats_relationship_for_marriage_wording() -> None:
    """Ordering in KEYWORDS matters — these overlap and marriage must win."""
    assert classify_by_keywords("will my marriage partner be supportive?").intent == "marriage"


@needs_openrouter
def test_llm_classifier_agrees_on_unambiguous_questions() -> None:
    from app.nodes.classifier import classify

    for question in QUESTIONS:
        if question.context:
            continue  # follow-ups need history; covered end to end elsewhere
        result, method = classify(question.question, session_id="test", user_id="test")
        assert result.intent in INTENTS
        if method == "llm":
            assert result.intent == question.expected_intent, (
                f"{question.id}: expected {question.expected_intent}, got {result.intent}"
            )


# ── prompt construction ───────────────────────────────────────────────
def test_system_prompt_includes_only_the_matching_domain_module(
    kundli: KundliContext, retrieval: RetrievalResult
) -> None:
    prompt = build_system_prompt("career", kundli, retrieval)
    assert DOMAIN_MODULES["career"].strip() in prompt
    assert DOMAIN_MODULES["marriage"].strip() not in prompt
    assert DOMAIN_MODULES["health"].strip() not in prompt


def test_system_prompt_carries_chart_and_knowledge(
    kundli: KundliContext, retrieval: RetrievalResult
) -> None:
    prompt = build_system_prompt("career", kundli, retrieval)
    assert kundli.ascendant["rashi"] in prompt
    assert "10th house governs profession" in prompt
    assert "not a human astrologer" in prompt or "not a human" in prompt


def test_system_prompt_replays_known_session_facts(
    kundli: KundliContext, retrieval: RetrievalResult
) -> None:
    prompt = build_system_prompt(
        "career", kundli, retrieval, session_state={"occupation": "software developer"}
    )
    assert "software developer" in prompt
    assert "do not ask again" in prompt.lower()


def test_missing_retrieval_tells_the_model_to_be_honest(kundli: KundliContext) -> None:
    prompt = build_system_prompt("career", kundli, RetrievalResult.unavailable("pinecone down"))
    assert "none available" in prompt
    assert "not certain" in prompt


def test_history_is_sent_as_turns_not_pasted_into_the_system_prompt(
    kundli: KundliContext, retrieval: RetrievalResult, history: list[Message]
) -> None:
    messages = build_messages(
        "What about the next two years?", "career", kundli, retrieval, history
    )
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "What about the next two years?"
    assert len(messages) == len(history) + 2
    assert [m["role"] for m in messages[1:-1]] == [m.role for m in history]
    # The prior turns must not ALSO be duplicated into the system prompt.
    assert "software developer" not in messages[0]["content"]
