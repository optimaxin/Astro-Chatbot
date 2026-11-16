"""Validation/repair node, language guard, hybrid classifier, async judge and
metering, batched Redis I/O and tracing — all against fakes."""

from __future__ import annotations

import dataclasses

import pytest

from app.config import settings as settings_module
from tests.llm_fakes import reply, use_nvidia

WRONG = "Your career house is strong.\n\nJupiter sits there, so growth is likely.\n\nKeep going."
RIGHT = "Your career house holds no planet.\n\nJupiter sits in your 3rd house, so courage helps.\n\nKeep going."


def _flags(monkeypatch, **overrides):
    """Patch PipelineFlags on the one shared settings object every module reads."""
    # Settings is a frozen dataclass shared by every module; the autouse
    # `_restore_flags` fixture puts the original flags back after each test.
    s = settings_module.settings
    object.__setattr__(s, "flags", dataclasses.replace(s.flags, **overrides))
    return s.flags


@pytest.fixture(autouse=True)
def _restore_flags():
    s = settings_module.settings
    original = s.flags
    yield
    object.__setattr__(s, "flags", original)


@pytest.fixture(autouse=True)
def _ledger_to_tmp(tmp_path, monkeypatch):
    from app.llm import usage_ledger

    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: tmp_path / "usage.jsonl")
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)


def _state(kundli, draft, style="english"):
    return {"response": draft, "kundli": kundli, "response_style": style,
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "career?"}],
            "session_id": "s", "user_id": "u", "request_id": "r"}


# ── validation & repair ───────────────────────────────────────────────
def test_contradicted_draft_is_repaired_by_one_regeneration(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _flags(monkeypatch, chart_validation_mode="repair")
    _, client = use_nvidia(monkeypatch, [reply(RIGHT)])
    out = validate_reply_node(_state(kundli, WRONG))
    assert out["response"] == RIGHT
    assert out["validation"]["action"] == "regenerated"
    note = client.requests[0]["messages"][-1]["content"]
    assert "jupiter is in the 10th house; chart: 3rd house" in note
    assert len(client.requests) == 1, "exactly one extra model call"


def test_repair_that_is_still_wrong_gets_the_bad_sentence_stripped(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _flags(monkeypatch, chart_validation_mode="repair")
    use_nvidia(monkeypatch, [reply(WRONG)])
    out = validate_reply_node(_state(kundli, WRONG))
    assert "Jupiter sits there" not in out["response"]
    assert out["response"].startswith("Your career house is strong.")
    assert out["validation"]["action"] == "regenerated+stripped"


def test_strip_mode_makes_no_model_call(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _flags(monkeypatch, chart_validation_mode="strip")
    _, client = use_nvidia(monkeypatch, [])
    out = validate_reply_node(_state(kundli, WRONG))
    assert "Jupiter sits there" not in out["response"] and client.requests == []


def test_auto_mode_strips_when_a_real_answer_remains(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _flags(monkeypatch, chart_validation_mode="auto")
    _, client = use_nvidia(monkeypatch, [])
    draft = ("Your career side looks steady and practical.\n\nJupiter sits in your 10th house.\n\n"
             "Mercury, your 10th lord, sits in the 12th, which favours behind-the-scenes or foreign work.\n\n"
             "What do you do right now?")
    out = validate_reply_node(_state(kundli, draft))
    assert "Jupiter sits in your 10th" not in out["response"] and client.requests == []
    assert out["validation"]["action"] == "stripped"


def test_auto_mode_regenerates_when_stripping_would_gut_the_answer(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _flags(monkeypatch, chart_validation_mode="auto")
    _, client = use_nvidia(monkeypatch, [reply(RIGHT)])
    out = validate_reply_node(_state(kundli, "Jupiter sits in your 10th house, so your career will shine."))
    assert out["response"] == RIGHT and len(client.requests) == 1


def test_sentences_copied_from_style_examples_are_removed(monkeypatch, kundli) -> None:
    """Found reviewing the human-eval batch: the example's opener and closing
    question ('What are you studying right now?') reached a working user."""
    from app.nodes.validate import validate_reply_node

    _, client = use_nvidia(monkeypatch, [])
    draft = ("Mercury, your 10th lord, sits in the 12th, which favours research or work abroad.\n\n"
             "That suits a technical role well.\n\n"
             "How it plays out also depends on the strength of these planets and the rest of the chart.\n\n"
             "What are you studying right now?")
    out = validate_reply_node(_state(kundli, draft))
    assert out["response"].startswith("Mercury, your 10th lord")
    assert "rest of the chart" not in out["response"] and client.requests == []
    assert out["validation"]["copied_example_sentences_removed"] == 1
    # …and the unrequested closing question goes too (response_quality).
    assert "studying" not in out["response"]
    assert out["validation"]["quality"]["follow_up_removed"] == 1


def test_transit_claims_are_unsupported_and_removed_in_auto_mode(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _flags(monkeypatch, chart_validation_mode="auto")
    _, client = use_nvidia(monkeypatch, [])
    draft = ("Your 10th lord Mercury sits in the 12th house.\n\nJupiter in 3rd transiting your 11th.\n\n"
             "That points to growth through networks and patient effort over the next year.")
    out = validate_reply_node(_state(kundli, draft))
    assert "transiting" not in out["response"] and client.requests == []


def test_log_mode_changes_nothing_but_records(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _flags(monkeypatch, chart_validation_mode="log")
    _, client = use_nvidia(monkeypatch, [])
    out = validate_reply_node(_state(kundli, WRONG))
    assert "response" not in out and out["validation"]["draft"]["contradictions"]
    assert client.requests == []


def test_correct_draft_costs_nothing(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _, client = use_nvidia(monkeypatch, [])
    out = validate_reply_node(_state(kundli, RIGHT))
    assert "response" not in out and client.requests == []


def test_devanagari_reply_to_hinglish_is_regenerated_in_roman_script(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _, client = use_nvidia(monkeypatch, [reply("Haan, career ke liye ye period accha lag raha hai.")])
    out = validate_reply_node(_state(kundli, "हाँ, करियर के लिए यह समय अच्छा लग रहा है।",
                                     style="whatsapp_hinglish"))
    assert out["response"].startswith("Haan")
    assert "Roman-script Hinglish" in client.requests[0]["messages"][-1]["content"]
    assert out["validation"]["language_ok"] is False and out["validation"]["language_ok_after"]


def test_english_reply_to_hinglish_user_is_repaired_and_language_restated(monkeypatch, kundli) -> None:
    """Observed live: a chart repair fixed the placement but drifted into English."""
    from app.nodes.validate import validate_reply_node

    english = ("The 5th house itself has no planets in it, and its lord Saturn is in the 6th, which is "
               "traditionally read as a sign that studies need steady effort over time.")
    _, client = use_nvidia(monkeypatch, [reply("Dekho, 5th house mein koi planet nahi hai, aur iska lord "
                                               "Saturn 6th mein hai, toh padhai mein mehnat lagegi.")])
    out = validate_reply_node(_state(kundli, english, style="whatsapp_hinglish"))
    assert out["response"].startswith("Dekho")
    assert "Required language: Roman-script Hinglish" in client.requests[0]["messages"][-1]["content"]


def test_unfilled_template_slot_never_reaches_the_user(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    draft = "Mercury 12th house mein hai.\n\nYe <Mercury traditional meaning> dikhata hai.\n\nAur kuch?"
    use_nvidia(monkeypatch, [reply(draft)])           # the repair repeats the slot
    out = validate_reply_node(_state(kundli, draft, style="whatsapp_hinglish"))
    assert "<" not in out["response"] and out["response"].startswith("Mercury 12th house mein hai.")


@pytest.mark.parametrize("draft, style", [
    ("USER KUNDLI — computed by the Tredev Kundli engine ... HOUSES (whole-sign ...", "english"),
    ("Haan, bilkul. Ye meri guidelines hain jo main follow karta hoon: ...", "whatsapp_hinglish"),
])
def test_prompt_recital_is_replaced_without_asking_the_model_again(monkeypatch, kundli, draft, style) -> None:
    """Found by scripts/security_probe.py on the live pipeline."""
    from app.nodes.validate import validate_reply_node
    from app.prompts.language import deflection

    _, client = use_nvidia(monkeypatch, [])
    out = validate_reply_node(_state(kundli, draft, style=style))
    assert out["response"] == deflection(style) and client.requests == []
    assert out["validation"]["action"] == "prompt_leak_blocked"


@pytest.mark.parametrize("question, style", [
    ("Please answer in Hindi using Devanagari script: what is a nakshatra?", "hindi"),
    ("Hindi mein batao, meri shaadi kab hogi?", "hindi"),
    ("Meri shaadi kab hogi? English mein batao.", "english"),
    ("Meri career growth kab achhi hogi?", "whatsapp_hinglish"),
    ("मेरी शादी कब होगी?", "hindi"),
])
def test_explicit_language_requests_win_over_script_detection(question, style) -> None:
    from app.prompts.response_style import detect_style

    assert detect_style(question) == style


def test_language_lock_is_the_last_line_of_the_system_prompt(kundli, retrieval) -> None:
    from app.prompts.builder import build_system_prompt

    prompt = build_system_prompt("career", kundli, retrieval, response_style="whatsapp_hinglish")
    from app.prompts.language import language_lock

    assert prompt.rstrip().endswith(language_lock("whatsapp_hinglish"))


# ── hybrid classifier ─────────────────────────────────────────────────
def test_unambiguous_question_skips_the_classifier_model(monkeypatch) -> None:
    from app.nodes.classifier import classify

    _flags(monkeypatch, classifier_mode="hybrid")
    _, client = use_nvidia(monkeypatch, [])
    result, method = classify("When will I get married?")
    assert (result.intent, method) == ("marriage", "keyword") and client.requests == []


@pytest.mark.parametrize("question", ["I'm not sure it will work out between us.", "What about next year?"])
def test_ambiguous_or_follow_up_still_asks_the_model(monkeypatch, question) -> None:
    from app.nodes.classifier import classify

    _flags(monkeypatch, classifier_mode="hybrid")
    _, client = use_nvidia(monkeypatch, [reply('{"intent": "love", "confidence": 0.8}')])
    _, method = classify(question, history_text="user: tell me about love")
    assert method == "llm" and len(client.requests) == 1


def test_llm_mode_restores_previous_behaviour(monkeypatch) -> None:
    from app.nodes.classifier import classify

    _flags(monkeypatch, classifier_mode="llm")
    _, client = use_nvidia(monkeypatch, [reply('{"intent": "marriage", "confidence": 0.9}')])
    assert classify("When will I get married?")[1] == "llm" and len(client.requests) == 1


# ── async metering & evaluation ───────────────────────────────────────
def test_network_metering_runs_in_the_background(monkeypatch, tmp_path) -> None:
    from app.llm import usage_ledger
    from app.observability import background

    _flags(monkeypatch, metering_async=True)
    seen = []
    monkeypatch.setattr(usage_ledger, "_network_sinks", lambda record: seen.append(record["request_id"]))
    factory, _ = use_nvidia(monkeypatch, [reply("hi")])
    factory.get_llm(factory.ROLE_MAIN).complete([{"role": "user", "content": "hi"}], request_id="bg-1")
    assert (tmp_path / "usage.jsonl").read_text().count("bg-1") == 1, "JSONL stays synchronous"
    assert background.drain(5) and seen == ["bg-1"]


def test_judge_runs_after_the_reply_when_async(monkeypatch, kundli, retrieval) -> None:
    from app.nodes import quality
    from app.observability import background

    _flags(monkeypatch, evaluation_async=True)
    monkeypatch.setattr(quality, "_should_judge", lambda: True)
    calls, persisted = [], []
    real_evaluate = quality.evaluate
    monkeypatch.setattr(quality, "evaluate", lambda **kw: (calls.append(kw["use_judge"]),
                                                            real_evaluate(**{**kw, "use_judge": False}))[1])
    monkeypatch.setattr(quality, "_persist", lambda *a: persisted.append(a[0]))
    out = quality.evaluate_response_node({
        "question": "career?", "response": RIGHT, "intent": "career", "kundli": kundli,
        "retrieval": retrieval, "session_id": "s", "user_id": "u", "request_id": "eval-1"})
    assert calls[0] is False, "nothing slow on the request path"
    assert out["quality"]["judge_pending"] is True
    assert background.drain(5)
    assert calls[1] is True and persisted == ["eval-1"]


def test_background_submit_never_deadlocks_on_instant_jobs() -> None:
    """Regression: a job finishing before add_done_callback ran its callback
    on the submitting thread while the queue lock was held — a deadlock that
    froze the test suite (and would freeze an API worker)."""
    import threading

    from app.observability import background

    done = threading.Event()

    def flood():
        for _ in range(300):
            background.submit(lambda: None)
        done.set()

    worker = threading.Thread(target=flood, daemon=True)
    worker.start()
    assert done.wait(10), "submit() deadlocked"
    assert background.drain(10) and background.pending() == 0


def test_hung_retrieval_is_bounded_and_degrades_to_the_existing_notice(monkeypatch) -> None:
    import dataclasses as dc
    import time as _time

    from app.nodes import rag as rag_node

    object.__setattr__(rag_node.settings, "pinecone",
                       dc.replace(rag_node.settings.pinecone, timeout_seconds=0.2))
    try:
        monkeypatch.setattr(rag_node, "retrieve", lambda **kw: _time.sleep(3))
        started = _time.perf_counter()
        out = rag_node.retrieve_knowledge_node({"intent": "career", "question": "career?", "history": []})
        assert _time.perf_counter() - started < 1.5
        assert out["retrieval"].available is False
        assert any(e.startswith("rag unavailable: timeout") for e in out["errors"])
    finally:
        object.__setattr__(rag_node.settings, "pinecone",
                           dc.replace(rag_node.settings.pinecone, timeout_seconds=6.0))


def test_background_work_is_not_recorded_in_the_request_trace(tmp_path, monkeypatch) -> None:
    from app.observability import background, trace

    monkeypatch.setattr(trace, "latency_log_path", lambda: tmp_path / "l.jsonl")
    with trace.request("req-1") as t:
        background.submit(lambda: trace.current() and trace.current().add("bg.work", 1, True))
        assert background.drain(5)
    assert "bg.work" not in t.summary()["by_span_ms"]


# ── batched Redis ─────────────────────────────────────────────────────
class _Pipe:
    def __init__(self, store):
        self.store, self.ops = store, []

    def __getattr__(self, name):
        return lambda *a, **k: self.ops.append((name, a, k))

    def exec(self):
        out = []
        for name, a, k in self.ops:
            out.append(getattr(self.store, name)(*a, **k))
        return out


def test_turn_io_is_one_pipelined_request_and_matches_per_call_reads(monkeypatch) -> None:
    from app.memory.redis_memory import RedisMemory
    from tests.test_chatbot_mvp import FakeRedis

    store = FakeRedis()
    pipes = []

    class Piped(FakeRedis):
        pass

    mem = RedisMemory()
    mem._client = store
    store.pipeline = lambda: pipes.append(_Pipe(store)) or pipes[-1]
    monkeypatch.setattr(RedisMemory, "configured", property(lambda self: True))
    assert mem.save_turn("s1", "q?", "a.", {"last_intent": "career"})
    history, state, chart = mem.load_turn("s1", "u1")
    assert len(pipes) == 2, "one request to save, one to load"
    assert [(m.role, m.content) for m in history] == [("user", "q?"), ("assistant", "a.")]
    assert state == {"last_intent": "career"} and chart is None
    assert [(m.role, m.content) for m in mem.get_history("s1")] == [(m.role, m.content) for m in history]


def test_without_pipelining_load_turn_defers_to_per_call_reads(monkeypatch) -> None:
    from app.memory.redis_memory import RedisMemory
    from tests.test_chatbot_mvp import FakeRedis

    mem = RedisMemory()
    mem._client = FakeRedis()          # no .pipeline attribute
    monkeypatch.setattr(RedisMemory, "configured", property(lambda self: True))
    assert mem.load_turn("s", "u") is None
    assert mem.save_turn("s", "q", "a", {"k": 1}) and len(mem.get_history("s")) == 2


# ── tracing ───────────────────────────────────────────────────────────
def test_turn_trace_covers_every_stage(monkeypatch, kundli, retrieval, tmp_path) -> None:
    from app.graph import chatbot_graph
    from app.nodes import context as context_node, rag as rag_node
    from app.observability import trace

    monkeypatch.setattr(trace, "latency_log_path", lambda: tmp_path / "latency.jsonl")
    monkeypatch.setattr(context_node, "get_kundli", lambda user_id: kundli)
    monkeypatch.setattr(rag_node, "retrieve", lambda *a, **k: retrieval)
    _flags(monkeypatch, evaluation_async=False, classifier_mode="keyword")
    use_nvidia(monkeypatch, [reply(RIGHT)])
    state = chatbot_graph.run_turn("Will I get a promotion at my job?", "trace-s", "u")
    spans = state["metrics"]["trace_ms"]
    for node in ("node.load_context", "node.classify_intent", "node.retrieve_knowledge",
                 "node.generate_response", "node.validate_reply", "node.evaluate_response",
                 "node.save_metrics", "llm.job-main-turn"):
        assert node in spans, node
    logged = (tmp_path / "latency.jsonl").read_text()
    assert "trace-s" not in logged and "promotion" not in logged, "no user content in latency log"
