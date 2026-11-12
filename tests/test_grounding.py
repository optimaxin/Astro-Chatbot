"""Source-grounded answers: citation validation, titles from real metadata,
and the footer kept out of conversation history."""

from __future__ import annotations

import json

import pytest

from app.rag.grounding import citation_footer, grounding_rules, process_citations, source_title
from app.rag.retriever import RetrievalResult, RetrievedChunk
from tests.llm_fakes import reply, use_nvidia


def _chunk(doc, topic, text="t"):
    return RetrievedChunk(text=text, score=0.5, domain="d", topic=topic, document=doc, source="kb")


TWO = RetrievalResult(chunks=[_chunk("career.txt", "job_change_switching_and_starting_a_business"),
                              _chunk("dashas.txt", "sade_sati_and_saturn_transits")],
                      used_filter=True, widened=False, available=True)


def test_valid_tags_become_citations_and_disappear_from_text() -> None:
    r = process_citations("Switching is read from the 10th [K1]. Sade Sati is a transit [K2][K1].", 2)
    assert r.text == "Switching is read from the 10th. Sade Sati is a transit."
    assert r.cited == [1, 2] and r.invalid == []


def test_a_tag_pointing_at_nothing_is_removed_and_counted() -> None:
    r = process_citations("Holi falls on 25 March [K7].", 2)
    assert "[K7]" not in r.text and r.cited == [] and r.invalid == ["K7"]


def test_inline_mode_keeps_only_valid_tags() -> None:
    r = process_citations("A [K1]. B [K9].", 2, keep_inline=True)
    assert r.text == "A [K1]. B." and r.invalid == ["K9"]


def test_grouped_tags_are_understood() -> None:
    assert process_citations("Both [K1, K2].", 2).cited == [1, 2]


def test_other_bracketed_text_is_left_alone() -> None:
    assert process_citations("Your Moon [Pisces] is calm.", 2).text == "Your Moon [Pisces] is calm."


def test_titles_come_only_from_real_metadata() -> None:
    assert source_title("career.txt", "job_change_switching_and_starting_a_business") == \
        "Career guide — Job change switching and starting a business"
    assert source_title("new_upload.txt", "") == "New Upload"


def test_prompt_tags_each_passage_and_states_the_rules() -> None:
    block = TWO.to_prompt_block()
    assert "[K1] Career guide — Job change" in block and "[K2] Dasha guide — Sade sati" in block
    assert "never turn their silence into a yes or a no" in block
    assert grounding_rules("S").count("[S") >= 2


def test_validation_node_emits_validated_citations(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    use_nvidia(monkeypatch, [])
    state = {"response": "Job changes are read from the 10th house [K1]. Also [K5].", "kundli": kundli,
             "response_style": "english", "retrieval": TWO, "messages": [],
             "session_id": "s", "user_id": "u", "request_id": "r"}
    out = validate_reply_node(state)
    assert out["response"] == "Job changes are read from the 10th house. Also."
    assert out["citations"] == [{"id": "career.txt::job_change_switching_and_starting_a_business",
                                 "title": "Career guide — Job change switching and starting a business"}]
    assert out["validation"]["citations"]["invalid"] == ["K5"]


def test_api_shows_sources_as_a_footer_bubble_but_history_stays_clean(monkeypatch, kundli, retrieval) -> None:
    """End to end on the real graph with a fake model."""
    import dataclasses

    from app.api import server
    from app.memory.redis_memory import RedisMemory, memory
    from app.nodes import context as context_node, rag as rag_node
    from tests.test_chatbot_mvp import FakeRedis

    store = FakeRedis()
    monkeypatch.setattr(memory, "_client", store)
    monkeypatch.setattr(RedisMemory, "configured", property(lambda self: True))
    monkeypatch.setattr(context_node, "get_kundli", lambda user_id: kundli)
    monkeypatch.setattr(rag_node, "retrieve", lambda **k: retrieval)
    s = server.settings
    object.__setattr__(s, "flags", dataclasses.replace(s.flags, classifier_mode="keyword"))
    use_nvidia(monkeypatch, [reply("Your 10th house governs profession [K1].\n\nKeep going.")])
    testclient = pytest.importorskip("fastapi.testclient")
    body = testclient.TestClient(server.app).post(
        "/chat", json={"user_id": "u1", "session_id": "s1", "question": "How is my career?",
                       "chat_bubbles": True}).json()
    assert body["citations"] == [{"id": "houses.txt::the_10th_house", "title": "Houses guide — The 10th house"}]
    assert body["messages"][-1].startswith("📚 Source (Tredev knowledge base): Houses guide")
    assert "[K1]" not in body["answer"]
    saved = [json.loads(m)["content"] for m in store.lists["chat:s1:messages"]]
    assert saved[-1] == "Your 10th house governs profession.\n\nKeep going.", "no footer in history"


def _verifier_state(kundli, text):
    return {"response": text, "kundli": kundli, "response_style": "english", "retrieval": TWO,
            "messages": [], "question": "Can I attend the pooja in person?",
            "session_id": "s", "user_id": "u", "request_id": "r"}


def _verifier_mode(mode):
    import dataclasses

    from app.config import settings as settings_module

    s = settings_module.settings
    object.__setattr__(s, "flags", dataclasses.replace(s.flags, grounding_verifier=mode))


def test_verifier_removes_what_the_passages_do_not_state(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _verifier_mode("cited")
    _, client = use_nvidia(monkeypatch, [reply("The passages don't say whether you can attend in person [K1].")])
    out = validate_reply_node(_verifier_state(kundli, "No, you cannot attend in person [K1]."))
    assert out["response"] == "The passages don't say whether you can attend in person."
    assert out["validation"]["grounding_verifier"]["changed"] is True
    assert "SOURCES" in client.requests[0]["messages"][-1]["content"]


def test_verifier_output_that_breaks_the_chart_is_discarded(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _verifier_mode("cited")
    use_nvidia(monkeypatch, [reply("Jupiter sits in your 10th house [K1].")])
    out = validate_reply_node(_verifier_state(kundli, "Career is read from the 10th house [K1]."))
    assert out["response"] == "Career is read from the 10th house."
    assert out["validation"]["grounding_verifier"]["used"] is False


def test_cited_mode_skips_answers_without_citations_and_off_never_runs(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _, client = use_nvidia(monkeypatch, [])
    _verifier_mode("cited")
    validate_reply_node(_verifier_state(kundli, "Namaste! How can I help?"))
    _verifier_mode("off")
    validate_reply_node(_verifier_state(kundli, "Career is read from the 10th house [K1]."))
    assert client.requests == []


def test_traffic_source_is_written_only_once_the_migration_added_the_column(monkeypatch) -> None:
    from app.llm import usage_store

    monkeypatch.setattr(usage_store, "table_columns", lambda table: {"request_id"})
    assert "traffic_source" not in usage_store._insert_sql()
    monkeypatch.setattr(usage_store, "table_columns", lambda table: {"request_id", "traffic_source"})
    sql = usage_store._insert_sql()
    assert "created_at, traffic_source" in sql and "%(traffic_source)s" in sql


def test_internal_scripts_tag_their_traffic() -> None:
    from app.llm.usage_tracker import LLMUsage, set_traffic_source

    set_traffic_source("profiling")
    try:
        u = LLMUsage(request_id="r", session_id="s", user_id="u", role="x", provider="p",
                     model="m", latency_ms=1, success=True, timestamp=0.0)
        assert u.traffic_source == "profiling"
    finally:
        set_traffic_source("user")


def test_footer_follows_the_users_language() -> None:
    c = [{"id": "x", "title": "Career guide — Job change"}]
    assert citation_footer(c, "hindi").startswith("📚 स्रोत")
    assert citation_footer([], "english") == ""
