"""Model-role audit: each role runs on its configured model, for its own
purpose, and the per-call trace says so. Scripted endpoint, no network.

Configured (NVIDIA): main Lightning (fallback Bolt); light route for
smalltalk -> Bolt (fallback Lightning); classifier Bolt (fallback Lightning);
evaluator Bolt (fallback Bolt = the same model); validation fallback Bolt.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.llm.llm_factory import ROLE_CLASSIFY, ROLE_EVALUATE, ROLE_MAIN
from app.llm.usage_tracker import usage_tracker
from tests.llm_fakes import http_error, reply, timeout_error, use_nvidia
from tests.test_chatbot_mvp import FakeRedis

MAIN = "nvidia/nemotron-3.5-lightning-30b-a3b"
BOLT = "nvidia/nemotron-3.5-bolt-e4b"
GOOD = "Your 7th lord Jupiter sits in your 3rd house, which traditionally favours meeting someone through your circle."
WRONG = "Your Jupiter sits in your 10th house, so your career will grow.\n\nVenus is in your 12th house."


@pytest.fixture(autouse=True)
def _ledger_to_tmp(tmp_path, monkeypatch):
    from app.llm import usage_ledger

    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: tmp_path / "usage.jsonl")
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)


@pytest.fixture(autouse=True)
def _flags_restored():
    from app.config import settings as settings_module

    s = settings_module.settings
    original = s.flags
    yield
    object.__setattr__(s, "flags", original)


def _flags(**overrides):
    from app.config import settings as settings_module

    s = settings_module.settings
    object.__setattr__(s, "flags", dataclasses.replace(s.flags, **overrides))


def _rows(request_id):
    return usage_tracker.for_request(request_id)


# ── each role on its own model, with its own purpose and prompt version ──
def test_classifier_runs_on_bolt_with_its_own_purpose(monkeypatch) -> None:
    from app.nodes.classifier import classify

    _flags(classifier_mode="llm")
    _, client = use_nvidia(monkeypatch, [reply('{"intent": "marriage", "confidence": 0.9}')])
    result, method = classify("Shaadi?", request_id="r-cls")
    assert (result.intent, method) == ("marriage", "llm") and client.models == [BOLT]
    (row,) = _rows("r-cls")
    assert (row.role, row.purpose, row.model, row.fallback_used) == (ROLE_CLASSIFY, "classify", BOLT, False)
    assert row.prompt_version and len(row.prompt_version) == 10


def test_classifier_falls_back_to_lightning_only_when_bolt_fails(monkeypatch) -> None:
    from app.nodes.classifier import classify

    _flags(classifier_mode="llm")
    _, client = use_nvidia(monkeypatch, [http_error(400, "x")])            # 400 does not fall back
    assert classify("Shaadi?", request_id="r-c2")[1] == "keyword-fallback" and client.models == [BOLT]
    _, client = use_nvidia(monkeypatch, [timeout_error(), reply('{"intent": "career", "confidence": 0.8}')])
    assert classify("Naukri?", request_id="r-c3")[0].intent == "career" and client.models == [BOLT, MAIN]
    assert [r.fallback_used for r in _rows("r-c3")] == [False, True]


def test_hybrid_mode_does_not_call_the_classifier_for_an_unambiguous_message(monkeypatch) -> None:
    from app.nodes.classifier import classify

    _, client = use_nvidia(monkeypatch, [])
    assert classify("When will I get married?")[1] == "keyword" and client.models == []


def test_the_judge_runs_on_bolt_with_lightning_as_its_fallback(monkeypatch, kundli, retrieval) -> None:
    """Audit 2026-11-18 found the judge's fallback was Bolt itself; it is now Lightning."""
    from app.evaluation.evaluator import run_judge
    from app.llm.llm_factory import get_llm

    llm = get_llm(ROLE_EVALUATE)
    assert llm.config.model == BOLT and llm.config.fallback_model == MAIN
    assert llm.attempts() == [(BOLT, False, "same"), (MAIN, True, "same")]
    _, client = use_nvidia(monkeypatch, [http_error(429)] * 6)
    scores, error = run_judge("q", GOOD, "marriage", kundli, retrieval, request_id="r-judge")
    assert scores is None and error and client.models == [BOLT] * 3 + [MAIN] * 3
    assert {r.purpose for r in _rows("r-judge")} == {"evaluate"}


def test_an_evaluator_rejection_scores_the_turn_but_never_changes_the_reply(monkeypatch, kundli, retrieval) -> None:
    from app.evaluation.evaluator import evaluate

    low = ('{"relevance": 1, "personalization": 1, "groundedness": 1, "completeness": 1, "human_likeness": 1, '
           '"hinglish_quality": 1, "whatsapp_style": 1, "clarity": 1, "conciseness": 1, "warmth": 1, "safety": 1, '
           '"notes": "wrong"}')
    use_nvidia(monkeypatch, [reply(low)])
    result = evaluate("When will I marry?", GOOD, "marriage", kundli, retrieval, request_id="r-ev")
    assert result["judge_used"] and result["evaluator_model"] == BOLT
    assert max(result["scores"].values()) <= 1.0
    # The judge has no path back into the reply: the graph's evaluate node only
    # records quality/reward, and runs after the reply is final (EVALUATION_ASYNC).


# ── main role: success and every technical failure ───────────────────
def _answer(monkeypatch, script, *, intent="marriage", rid):
    from app.nodes.generate import generate_response_node

    _, client = use_nvidia(monkeypatch, script)
    out = generate_response_node({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
                                  "intent": intent, "session_id": "s", "user_id": "u", "request_id": rid})
    return out, client


def test_main_success_is_lightning_with_purpose_answer(monkeypatch) -> None:
    out, client = _answer(monkeypatch, [reply(GOOD)], rid="r-m1")
    assert out["model_used"] == MAIN and out["fallback_used"] is False and client.models == [MAIN]
    (row,) = _rows("r-m1")
    assert (row.purpose, row.route, row.success) == ("answer", "default", True) and row.prompt_version


@pytest.mark.parametrize("failure,models,code", [
    (timeout_error(), [MAIN, BOLT], "timeout"),
    (reply('{"status": "succ'), [MAIN, BOLT], "malformed_response"),
    (reply(""), [MAIN, BOLT], "empty_response"),
])
def test_main_failures_that_go_straight_to_bolt(monkeypatch, failure, models, code) -> None:
    out, client = _answer(monkeypatch, [failure, reply(GOOD)], rid=f"r-{code}")
    assert client.models == models and out["model_used"] == BOLT and out["fallback_used"] is True
    rows = _rows(f"r-{code}")
    assert [r.purpose for r in rows] == ["answer", "answer"] and rows[-1].error_code == code


@pytest.mark.parametrize("status,code", [(500, "upstream_error:500"), (429, "rate_limited:429")])
def test_main_5xx_and_429_are_retried_on_lightning_then_go_to_bolt(monkeypatch, status, code) -> None:
    script = [http_error(status)] * 3 + [reply(GOOD)]
    out, client = _answer(monkeypatch, script, rid=f"r-roles-{status}")
    assert client.models == [MAIN, MAIN, MAIN, BOLT] and out["fallback_used"] is True
    rows = _rows(f"r-roles-{status}")
    assert [r.attempt for r in rows] == [0, 1, 2, 0] and rows[-1].error_code == code


def test_auth_and_bad_request_errors_never_reach_bolt(monkeypatch) -> None:
    for status in (400, 401):
        out, client = _answer(monkeypatch, [http_error(status)], rid=f"r-a{status}")
        assert client.models == [MAIN] and out["generation_failed"] is True


def test_both_models_unavailable_gives_the_localized_outage_reply(monkeypatch) -> None:
    out, client = _answer(monkeypatch, [timeout_error(), timeout_error()], rid="r-both")
    assert client.models == [MAIN, BOLT] and out["generation_failed"] is True and out["model_used"] == "none"


def test_a_smalltalk_answer_runs_on_bolt_by_configuration(monkeypatch) -> None:
    out, client = _answer(monkeypatch, [reply("Namaste! Kundli ke baare mein kya jaanna chahenge?")],
                          intent="smalltalk", rid="r-light")
    assert client.models == [BOLT] and out["fallback_used"] is False
    assert _rows("r-light")[0].route == "light-intent:smalltalk"


# ── repair / validation fallback: who runs, and nothing else ──────────
def _validate(monkeypatch, kundli, script, rid):
    from app.nodes.validate import validate_reply_node

    _flags(chart_validation_mode="repair")
    _, client = use_nvidia(monkeypatch, script)
    out = validate_reply_node({"response": WRONG, "kundli": kundli, "response_style": "english",
                               "question": "career?", "intent": "career",
                               "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "career?"}],
                               "session_id": "s", "user_id": "u", "request_id": rid})
    return out, client


def test_repair_runs_on_lightning_and_bolt_only_after_it_fails(monkeypatch, kundli) -> None:
    out, client = _validate(monkeypatch, kundli, [reply(WRONG), reply(GOOD)], "r-rep")
    assert client.models == [MAIN, BOLT] and out["response"] == GOOD
    rows = _rows("r-rep")
    assert [(r.purpose, r.model, r.fallback_used) for r in rows] == [
        ("repair", MAIN, False), ("validation_fallback", BOLT, True)]
    assert rows[1].route == "validation-fallback"


def test_a_successful_repair_never_calls_bolt(monkeypatch, kundli) -> None:
    out, client = _validate(monkeypatch, kundli, [reply(GOOD)], "r-rep2")
    assert client.models == [MAIN] and [r.purpose for r in _rows("r-rep2")] == ["repair"]


def test_the_repair_model_cannot_change_a_verified_fact_or_dasha_tense(monkeypatch, kundli) -> None:
    """Whatever the repair model writes is judged; one that adds Sade Sati or
    states a wrong placement never ships (Bolt, then the safe reply)."""
    smuggled = GOOD + " Your Sade Sati is also running now."
    out, client = _validate(monkeypatch, kundli, [reply(smuggled), reply(WRONG)], "r-rep3")
    assert "Sade Sati" not in out["response"] and "10th house" not in out["response"]
    assert out["validation"]["primary_rejected"] and out["validation"]["bolt_rejected"]


def test_the_grammar_correction_may_return_json(monkeypatch) -> None:
    """Role audit 2026-11-18: the prose check rejected this JSON pass on BOTH
    models (malformed_response x2 per turn), silently disabling the layer."""
    from app.llm.llm_factory import get_llm
    from app.llm.usage_tracker import llm_purpose

    _, client = use_nvidia(monkeypatch, [reply('{"1": "Aapki Mercury Mahadasha chal rahi hai."}')])
    with llm_purpose("grammar_correction", "v"):
        text = get_llm(ROLE_MAIN).complete([{"role": "user", "content": "x"}], request_id="r-gram").text
    assert text.startswith("{") and client.models == [MAIN]
    (row,) = _rows("r-gram")
    assert row.success and row.requested_model == MAIN and row.purpose == "grammar_correction"


def test_a_json_fragment_as_an_answer_is_still_malformed(monkeypatch) -> None:
    out, client = _answer(monkeypatch, [reply('{"1": "x"}'), reply(GOOD)], rid="r-json-answer")
    assert client.models == [MAIN, BOLT] and out["model_used"] == BOLT


# ── the per-turn trace on a full graph run ────────────────────────────
@pytest.fixture
def turn_env(monkeypatch, kundli, retrieval):
    from app.graph import chatbot_graph
    from app.memory.redis_memory import RedisMemory, memory
    from app.nodes import context as context_node, rag as rag_node

    monkeypatch.setattr(memory, "_client", FakeRedis())
    monkeypatch.setattr(RedisMemory, "configured", property(lambda self: True))
    monkeypatch.setattr(context_node, "get_kundli", lambda user_id: kundli)
    monkeypatch.setattr(rag_node, "retrieve", lambda *a, **k: retrieval)
    monkeypatch.setattr(chatbot_graph, "settings", dataclasses.replace(chatbot_graph.settings, enable_evaluation=False))


def test_the_turn_trace_names_every_call_and_its_verdict(monkeypatch, turn_env) -> None:
    from app.graph.chatbot_graph import run_turn

    _flags(classifier_mode="llm", chart_validation_mode="repair")
    use_nvidia(monkeypatch, [reply('{"intent": "career", "confidence": 0.9}'), reply(WRONG), reply(WRONG), reply(GOOD)])
    state = run_turn("Mera kaam kaisa?", "trace-1", "u1", response_style="english")
    trace = state["metrics"]["llm_trace"]
    assert [(t["purpose"], t["model"], t["verdict"]) for t in trace] == [
        ("classify", BOLT, "intent=career via llm"),
        ("answer", MAIN, "intervened:regenerated+bolt"),
        ("repair", MAIN, "rejected"),
        ("validation_fallback", BOLT, "accepted"),
    ]
    assert all(t["prompt_version"] for t in trace)
    assert state["metrics"]["validation_outcome"]["action"] == "regenerated+bolt"
    flat = repr(trace)
    assert "Jupiter" not in flat and "Mera kaam" not in flat          # no prompt or answer text in the trace


def test_every_production_call_site_is_labelled(monkeypatch, turn_env) -> None:
    from app.graph.chatbot_graph import run_turn

    use_nvidia(monkeypatch, [reply(GOOD)])
    state = run_turn("When will I get married?", "trace-2", "u1", response_style="english")
    assert not [t for t in state["metrics"]["llm_trace"] if str(t["purpose"]).startswith("unlabelled")]


# ── audit finding: the turn's reported model/fallback can be wrong ────
def test_the_reported_answer_model_and_fallback_flag_agree(monkeypatch, turn_env) -> None:
    """Audit 2026-11-18, fixed (P0): a Bolt validation-fallback answer is reported as
    Bolt / fallback, and a rejected candidate is never reported as the source."""
    from app.graph.chatbot_graph import run_turn

    _flags(chart_validation_mode="repair")
    use_nvidia(monkeypatch, [reply(WRONG), reply(WRONG), reply(GOOD)])
    metrics = run_turn("When will I get married?", "trace-3", "u1", response_style="english")["metrics"]
    assert metrics["model"] == BOLT and metrics["fallback_used"] is True
    assert (metrics["response_source"], metrics["response_purpose"]) == ("fallback", "validation_fallback")


# ── P1: judge reliability (fallback to Lightning, unavailable is never a pass) ──
def test_the_judge_falls_back_to_lightning_not_to_itself(monkeypatch, kundli, retrieval) -> None:
    from app.evaluation.evaluator import evaluate
    from app.llm.llm_factory import get_llm

    llm = get_llm(ROLE_EVALUATE)
    assert llm.config.model == BOLT and llm.config.fallback_model == MAIN
    assert llm.attempts() == [(BOLT, False, "same"), (MAIN, True, "same")]
    ok = ('{"relevance": 8, "personalization": 7, "groundedness": 9, "completeness": 7, "human_likeness": 7, '
          '"hinglish_quality": 7, "whatsapp_style": 7, "clarity": 8, "conciseness": 8, "warmth": 7, "safety": 10, '
          '"notes": "fine"}')
    _, client = use_nvidia(monkeypatch, [http_error(429), http_error(429), http_error(429), reply(ok)])
    result = evaluate("When will I marry?", GOOD, "marriage", kundli, retrieval, request_id="r-jfb")
    assert client.models == [BOLT, BOLT, BOLT, MAIN], "bounded retries on Bolt, then ONE try on Lightning"
    assert result["evaluation_status"] == "judged" and result["evaluator_model"] == MAIN
    assert [r.purpose for r in _rows("r-jfb")] == ["evaluate"] * 4


def test_both_judge_models_unavailable_is_unavailable_not_a_score(monkeypatch, kundli, retrieval) -> None:
    from app.evaluation.evaluator import evaluate
    from app.nodes.quality import _score

    _, client = use_nvidia(monkeypatch, [http_error(429)] * 6)
    result, reward = _score(use_judge=True, question="q", response=GOOD, intent="marriage", kundli=kundli,
                            retrieval=retrieval, request_id="r-judge-down")
    assert len(client.requests) == 6, "bounded: 3 tries per model, never indefinite"
    assert result["evaluation_status"] == "unavailable" and result["judge_used"] is False
    assert result["scores"] == {} and result["overall"] is None and reward is None
    assert result["judge_error"] and result["deterministic"]["has_response"], "deterministic checks still ran"
    from app.evaluation.evaluation_store import to_row

    row = to_row("r", "u", "s", result, reward)
    assert row["reward_score"] is None and row["relevance"] is None and row["safety"] is None
    assert row["judge_used"] is False


def test_an_unsampled_turn_is_not_reported_as_unavailable(monkeypatch, kundli, retrieval) -> None:
    from app.evaluation.evaluator import evaluate

    use_nvidia(monkeypatch, [])
    result = evaluate("q", GOOD, "marriage", kundli, retrieval, request_id="r-unsampled", use_judge=False)
    assert result["evaluation_status"] == "unsampled" and result["scores"]


def test_a_failed_judge_never_blocks_the_reply(monkeypatch, turn_env) -> None:
    """The judge is advisory and runs after the reply: with both models down the
    user still gets the validated answer."""
    from app.graph.chatbot_graph import run_turn
    from app.nodes import quality

    monkeypatch.setattr(quality, "_should_judge", lambda: True)
    _flags(evaluation_async=False)
    from app.graph import chatbot_graph

    monkeypatch.setattr(chatbot_graph, "settings", dataclasses.replace(chatbot_graph.settings, enable_evaluation=True))
    use_nvidia(monkeypatch, [reply(GOOD)] + [http_error(429)] * 8)
    state = run_turn("When will I get married?", "judge-down", "u1", response_style="english")
    assert state["response"].startswith("Your 7th lord Jupiter") and not state["generation_failed"]
    assert state["quality"]["evaluation_status"] == "unavailable"
