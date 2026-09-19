"""Live-service tests: Redis memory, Pinecone retrieval, and a full turn.

Each is skipped with a clear reason when its service is not configured, so
this file is informative rather than noisy on a machine without keys.
"""

from __future__ import annotations

import uuid

import pytest

from app.graph.chatbot_graph import build_graph, run_turn, should_evaluate
from app.memory.redis_memory import RedisMemory
from app.nodes.classifier import INTENTS
from app.rag.retriever import INTENT_DOMAINS, retrieve
from tests.conftest import needs_openrouter, needs_pinecone, needs_redis
from tests.test_questions import QUESTIONS, by_id


# ── Redis (Phase 5) ───────────────────────────────────────────────────
@pytest.fixture
def session_id() -> str:
    return f"pytest-{uuid.uuid4().hex[:12]}"


@needs_redis
def test_save_and_get_history(session_id: str) -> None:
    store = RedisMemory()
    try:
        assert store.save_message(session_id, "user", "I want to know about my career.")
        assert store.save_message(session_id, "assistant", "Are you working or studying?")
        assert store.save_message(session_id, "user", "I am working as a software developer.")

        history = store.get_history(session_id)
        assert [m.role for m in history] == ["user", "assistant", "user"]
        assert history[-1].content == "I am working as a software developer."
        assert all(m.timestamp > 0 for m in history)
        assert store.count_messages(session_id) == 3
    finally:
        store.clear_session(session_id)


@needs_redis
def test_history_window_limits_what_is_returned(session_id: str) -> None:
    store = RedisMemory()
    try:
        for i in range(10):
            store.save_message(session_id, "user", f"message {i}")
        recent = store.get_history(session_id, limit=3)
        assert len(recent) == 3
        assert recent[-1].content == "message 9", "must return the NEWEST messages"
        assert recent[0].content == "message 7"
    finally:
        store.clear_session(session_id)


@needs_redis
def test_sessions_are_isolated(session_id: str) -> None:
    store = RedisMemory()
    other = f"{session_id}-other"
    try:
        store.save_message(session_id, "user", "mine")
        store.save_message(other, "user", "theirs")
        assert [m.content for m in store.get_history(session_id)] == ["mine"]
        assert [m.content for m in store.get_history(other)] == ["theirs"]
    finally:
        store.clear_session(session_id)
        store.clear_session(other)


@needs_redis
def test_session_state_round_trip(session_id: str) -> None:
    store = RedisMemory()
    try:
        assert store.save_session_state(session_id, {"occupation": "software developer", "turns": 3})
        state = store.get_session_state(session_id)
        assert state["occupation"] == "software developer"
        assert state["turns"] == 3
    finally:
        store.clear_session(session_id)


@needs_redis
def test_clear_session_removes_everything(session_id: str) -> None:
    store = RedisMemory()
    store.save_message(session_id, "user", "hello")
    store.save_session_state(session_id, {"k": "v"})
    assert store.clear_session(session_id)
    assert store.get_history(session_id) == []
    assert store.get_session_state(session_id) == {}


def test_unconfigured_redis_degrades_instead_of_raising(monkeypatch) -> None:
    """The no-Redis path must be safe — it is what runs before setup."""
    from app.config import settings as settings_module
    from app.config.settings import RedisSettings

    original = settings_module.settings.redis
    object.__setattr__(settings_module.settings, "redis", RedisSettings(url="", token=""))
    try:
        store = RedisMemory()
        assert not store.configured
        assert store.save_message("s", "user", "hi") is False
        assert store.get_history("s") == []
        assert store.get_session_state("s") == {}
        assert store.count_messages("s") == 0
        ok, reason = store.ping()
        assert not ok and "not set" in reason
    finally:
        object.__setattr__(settings_module.settings, "redis", original)


# ── Pinecone (Phase 4) ────────────────────────────────────────────────
@needs_pinecone
def test_retrieval_returns_relevant_chunks() -> None:
    result = retrieve("What does my chart say about my career?", intent="career")
    assert result.available, result.error
    assert result.chunks, "no chunks — has scripts/ingest_knowledge.py been run?"
    assert all(0.0 <= c.score <= 1.0 for c in result.chunks)
    # The intent filter should surface career/house knowledge, not nakshatra trivia.
    assert {c.domain for c in result.chunks} & set(INTENT_DOMAINS["career"])


@needs_pinecone
def test_retrieval_is_intent_specific() -> None:
    career = retrieve("Will I get a promotion?", intent="career")
    marriage = retrieve("When will I get married?", intent="marriage")
    assert career.chunks and marriage.chunks
    assert {c.topic for c in career.chunks} != {c.topic for c in marriage.chunks}


@needs_pinecone
def test_retrieval_block_is_prompt_ready() -> None:
    block = retrieve("What is a Mahadasha?", intent="general_astrology").to_prompt_block()
    assert "RETRIEVED ASTROLOGY KNOWLEDGE" in block
    assert "[1]" in block


def test_retrieval_failure_is_reported_not_raised(monkeypatch) -> None:
    from app.config import settings as settings_module

    original = settings_module.settings.enable_rag
    object.__setattr__(settings_module.settings, "enable_rag", False)
    try:
        result = retrieve("anything", intent="career")
        assert not result.available
        assert result.chunks == []
        assert "none available" in result.to_prompt_block()
    finally:
        object.__setattr__(settings_module.settings, "enable_rag", original)


# ── graph wiring (Phase 8) ────────────────────────────────────────────
def test_graph_compiles_with_every_node() -> None:
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    for expected in (
        "load_context", "classify_intent", "retrieve_knowledge",
        "build_context", "generate_response", "evaluate_response", "save_metrics",
    ):
        assert expected in nodes


def test_evaluation_is_skipped_for_a_failed_generation() -> None:
    assert should_evaluate({"generation_failed": True}) == "save_metrics"  # type: ignore[arg-type]
    assert should_evaluate({"generation_failed": False}) == "evaluate_response"  # type: ignore[arg-type]


def test_turn_never_raises_even_with_nothing_configured() -> None:
    """A missing key must produce an apology and metrics, not a traceback."""
    state = run_turn("Hello", f"pytest-safe-{uuid.uuid4().hex[:8]}", "test-user-001")
    assert state.get("response"), "a turn must always produce something to show the user"
    assert "metrics" in state
    assert state["metrics"]["intent"] in {*INTENTS, "unknown"}


# ── end to end (Phases 11-16) ─────────────────────────────────────────
@needs_openrouter
def test_single_turn_end_to_end(session_id: str) -> None:
    question = by_id("career-open")
    state = run_turn(question.question, session_id, "test-user-001")

    assert not state.get("generation_failed"), state["metrics"].get("errors")
    assert state["response"].strip()

    metrics = state["metrics"]
    assert metrics["intent"] == question.expected_intent
    assert metrics["kundli_available"] is True
    assert metrics["latency_ms"] > 0
    # Token counts must come from the provider, so they are real integers here.
    assert metrics["tokens_available"], "provider reported no usage — investigate"
    assert metrics["input_tokens"] > 0
    assert metrics["output_tokens"] > 0
    assert metrics["total_tokens"] >= metrics["input_tokens"]
    assert metrics["estimated_cost_usd"] >= 0.0
    assert 0.0 <= metrics["reward"] <= 1.0

    lowered = state["response"].lower()
    for banned in question.must_not_contain:
        assert banned.lower() not in lowered


@needs_openrouter
@needs_redis
def test_multi_turn_conversation_remembers(session_id: str) -> None:
    """The load-bearing behaviour: turn 2 must see what was said in turn 1."""
    from app.memory.redis_memory import memory

    try:
        run_turn("I want to know about my career.", session_id, "test-user-001")
        history = memory.get_history(session_id)
        assert len(history) >= 2

        second = run_turn("I am working as a software developer.", session_id, "test-user-001")
        assert not second.get("generation_failed")
        # The prompt for turn 2 carried turn 1's exchange as real chat turns.
        prior = [m["content"] for m in second["messages"][1:-1]]
        assert any("career" in content.lower() for content in prior)
    finally:
        memory.clear_session(session_id)


@needs_openrouter
def test_response_never_contradicts_the_chart(session_id: str) -> None:
    from app.evaluation.evaluator import check_kundli_contradictions

    state = run_turn(by_id("kundli-specific").question, session_id, "test-user-001")
    if state.get("generation_failed"):
        pytest.skip("generation unavailable")
    problems = check_kundli_contradictions(state["response"], state["kundli"])
    assert not problems, f"response contradicts the real chart: {problems}"


@needs_openrouter
@pytest.mark.parametrize("question", QUESTIONS, ids=lambda q: q.id)
def test_every_question_produces_a_usable_answer(question, session_id: str) -> None:
    """The full question set, one isolated session each."""
    from app.memory.redis_memory import memory

    scoped = f"{session_id}-{question.id}"
    try:
        for role, content in question.context:
            memory.save_message(scoped, role, content)  # type: ignore[arg-type]

        state = run_turn(question.question, scoped, "test-user-001")
        if state.get("generation_failed"):
            pytest.skip(f"generation unavailable: {state['metrics'].get('errors')}")

        assert state["response"].strip()
        lowered = state["response"].lower()
        for banned in question.must_not_contain:
            assert banned.lower() not in lowered, f"{question.id} said {banned!r}"

        quality = state["metrics"].get("quality") or {}
        failures = quality.get("deterministic_failures") or []
        assert "no_prompt_leak" not in failures, f"{question.id} leaked the system prompt"
        assert "no_unsupported_kundli_facts" not in failures, (
            f"{question.id} contradicted the chart: {quality.get('deterministic', {}).get('contradictions')}"
        )
    finally:
        memory.clear_session(scoped)
