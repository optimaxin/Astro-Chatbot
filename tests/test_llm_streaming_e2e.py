"""Streaming, RAG regression, fallback-without-duplicates, rate limits and
an end-to-end `/chat` turn on the NVIDIA provider — all against fakes."""

from __future__ import annotations

import dataclasses

import pytest

from app.llm import reliability
from app.llm.llm_factory import ROLE_MAIN
from app.llm.streaming import ThinkFilter
from app.llm.usage_tracker import usage_tracker
from tests.llm_fakes import http_error, reply, stream, timeout_error, use_nvidia
from tests.test_chatbot_mvp import FakeRedis

MAIN = "nvidia/nemotron-3.5-lightning-30b-a3b"  # configured primary
BOLT = "nvidia/nemotron-3.5-bolt-e4b"
MSG = [{"role": "user", "content": "hello"}]


@pytest.fixture(autouse=True)
def _ledger_to_tmp(tmp_path, monkeypatch):
    from app.llm import usage_ledger

    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: tmp_path / "usage.jsonl")
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)


# ── streaming ─────────────────────────────────────────────────────────
def test_stream_yields_deltas_and_records_reported_usage(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [stream(["Nam", "aste", "!"])])
    s = factory.get_llm(ROLE_MAIN).stream(MSG, request_id="s-ok")
    assert list(s) == ["Nam", "aste", "!"]
    assert s.result.text == "Namaste!"
    assert s.result.usage.token_source == "provider" and s.result.usage.streamed
    assert client.requests[0]["stream"] is True
    assert client.requests[0]["stream_options"] == {"include_usage": True}


def test_stream_without_usage_is_marked_estimated(monkeypatch) -> None:
    factory, _ = use_nvidia(monkeypatch, [stream(["hi"], usage=None)])
    s = factory.get_llm(ROLE_MAIN).stream(MSG)
    list(s)
    assert s.result.usage.token_source == "estimated"
    assert s.result.usage.output_tokens is None


def test_stream_falls_back_only_before_the_first_token(monkeypatch) -> None:
    factory, client = use_nvidia(
        monkeypatch, [http_error(503)] * 3 + [stream(["from ", "bolt"])],
    )
    s = factory.get_llm(ROLE_MAIN).stream(MSG)
    assert "".join(s) == "from bolt"
    assert client.models == [MAIN, MAIN, MAIN, BOLT]
    assert s.result.usage.fallback_used


def test_mid_stream_failure_does_not_splice_a_second_model(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [stream(["a", "b", "c"], fail_after=2),
                                               stream(["never"])])
    s = factory.get_llm(ROLE_MAIN).stream(MSG)
    received = []
    with pytest.raises(factory.LLMCallFailed):
        for delta in s:
            received.append(delta)
    assert received == ["a", "b"]
    assert client.models == [MAIN], "no retry or fallback after output began"


def test_cancelling_a_stream_closes_upstream_and_is_not_a_model_failure(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [stream(["one ", "two ", "three"])])
    s = factory.get_llm(ROLE_MAIN).stream(MSG, request_id="s-cancel")
    assert next(iter(s)) == "one "
    s.close()
    assert client.streams[0].closed
    row = usage_tracker.for_request("s-cancel")[-1]
    assert row.error_code == reliability.CANCELLED and row.success is False
    assert reliability.breaker.state(("nvidia", MAIN), 30) == "closed"


@pytest.mark.parametrize("pieces, expected", [
    (["<thi", "nk>plan", "</think>", "Hello"], "Hello"),
    (["Hello ", "world"], "Hello world"),
    (["<think>never closed"], ""),
])
def test_think_filter(pieces, expected) -> None:
    f = ThinkFilter(enabled=True)
    out = "".join(f.feed(p) for p in pieces) + f.flush()
    assert out == expected


# ── full turn: RAG, routing, fallback, history ────────────────────────
@pytest.fixture
def turn_env(monkeypatch, kundli, retrieval):
    """Real graph, fake Redis, fixed chart and retrieval, judge off."""
    from app.graph import chatbot_graph
    from app.memory.redis_memory import RedisMemory, memory
    from app.nodes import context as context_node, rag as rag_node

    fake = FakeRedis()
    monkeypatch.setattr(memory, "_client", fake)
    monkeypatch.setattr(RedisMemory, "configured", property(lambda self: True))
    monkeypatch.setattr(context_node, "get_kundli", lambda user_id: kundli)
    monkeypatch.setattr(rag_node, "retrieve", lambda *a, **k: retrieval)
    monkeypatch.setattr(chatbot_graph, "settings",
                        dataclasses.replace(chatbot_graph.settings, enable_evaluation=False))
    return fake


def _responder(main_steps):
    """Classifier calls get JSON; main-answer calls replay `main_steps`."""
    steps = list(main_steps)

    def respond(kwargs):
        system = str(kwargs["messages"][0].get("content", ""))
        if system.startswith("You classify"):
            return reply('{"intent": "career", "confidence": 0.9}')
        return steps.pop(0)

    return respond


def _install(monkeypatch, main_steps):
    factory, client = use_nvidia(monkeypatch, [])
    respond = _responder(main_steps)
    original = client._create

    def create(**kwargs):
        client.script.append(respond(kwargs))
        return original(**kwargs)

    client.chat.completions.create = create
    return factory, client


def _history(fake, session_id):
    import json

    return [json.loads(m)["role"] for m in fake.lists.get(f"chat:{session_id}:messages", [])]


def test_turn_uses_the_primary_with_retrieved_knowledge_and_keeps_sources(monkeypatch, turn_env) -> None:
    from app.graph.chatbot_graph import run_turn

    _, client = _install(monkeypatch, [reply("Your 10th house speaks to career.")])
    state = run_turn("How is my career looking?", "e2e-1", "u1")
    metrics = state["metrics"]

    main_request = [r for r in client.requests if r["model"] == MAIN][0]
    prompt = " ".join(str(m["content"]) for m in main_request["messages"])
    assert "10th house governs profession" in prompt, "RAG context reaches the NVIDIA prompt"
    assert metrics["provider"] == "nvidia" and metrics["model"] == MAIN
    assert [s["id"] for s in metrics["sources"]] == ["houses.txt::the_10th_house"]
    assert _history(turn_env, "e2e-1") == ["user", "assistant"]


def test_fallback_turn_saves_the_user_message_exactly_once(monkeypatch, turn_env) -> None:
    from app.graph.chatbot_graph import run_turn

    # One timeout on the primary hands straight to the fallback (no retry of a
    # stalled model).
    _install(monkeypatch, [timeout_error(), reply("Fallback answer.")])
    state = run_turn("Career?", "e2e-2", "u1")
    assert state["response"] == "Fallback answer."
    assert state["metrics"]["fallback_used"] is True
    assert state["metrics"]["model"] == BOLT
    assert _history(turn_env, "e2e-2") == ["user", "assistant"], "no duplicate user message"


def test_total_failure_degrades_gracefully_and_writes_no_history(monkeypatch, turn_env) -> None:
    from app.graph.chatbot_graph import run_turn
    from app.nodes.generate import FAILURE_MESSAGE

    _install(monkeypatch, [http_error(503)] * 6)
    state = run_turn("Career?", "e2e-3", "u1")
    assert state["response"] == FAILURE_MESSAGE
    assert state["metrics"]["generation_failed"] is True
    assert _history(turn_env, "e2e-3") == []


# ── HTTP: response contract, health, rate limit ───────────────────────
@pytest.fixture
def api(monkeypatch, turn_env):
    testclient = pytest.importorskip("fastapi.testclient")
    from app.api import server

    monkeypatch.setattr(server, "settings", dataclasses.replace(
        server.settings,
        api=dataclasses.replace(server.settings.api, rate_limit_enabled=True,
                                rate_limit_per_minute=2),
    ))
    return testclient.TestClient(server.app)


def test_chat_endpoint_contract_is_unchanged_for_the_frontend(monkeypatch, api) -> None:
    _install(monkeypatch, [reply("Career looks steady.\n\nKeep going.")])
    body = api.post("/chat", json={"user_id": "web-1", "question": "Career?",
                                   "chat_bubbles": True}).json()
    # The fields KundliChatWidget reads.
    assert body["messages"] and body["answer"] and body["request_id"]
    assert body["degraded"] is False
    assert body["usage"]["provider"] == "nvidia" and body["usage"]["token_source"] == "provider"
    assert body["sources"][0]["document"] == "houses.txt"
    assert "nvapi" not in str(body), "credentials never reach the browser"


def test_per_user_rate_limit_returns_429(monkeypatch, api) -> None:
    _install(monkeypatch, [reply("a"), reply("b"), reply("c")])
    codes = [api.post("/chat", json={"user_id": "rl-user", "question": "Career?"}).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]


def test_health_reports_the_active_provider_and_circuits(monkeypatch, api) -> None:
    from app.api import server
    from app.llm import llm_factory

    use_nvidia(monkeypatch, [])
    monkeypatch.setattr(server, "provider_configured", llm_factory.provider_configured)
    llm = api.get("/health").json()["dependencies"]["llm"]
    assert llm["provider"] == "nvidia"
    assert llm["models"]["job-main-turn"] == MAIN
    assert "circuits" in llm
