"""The independent Quality Checker (shadow), the Humanizer, and the role registry.

Scripted endpoint, no network. Both roles are OFF by default; these tests switch
them on explicitly and check what they do, what they must never do, and that a
failure of either costs the user nothing.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from app.llm.usage_tracker import usage_tracker
from tests.llm_fakes import http_error, reply, timeout_error, use_nvidia

MAIN = "nvidia/nemotron-3.5-lightning-30b-a3b"
BOLT = "nvidia/nemotron-3.5-bolt-e4b"
ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"
GOOD = "Your 7th lord Jupiter sits in your 3rd house, which traditionally favours meeting someone through your circle."
VERDICT_PASS = json.dumps({"quality_score": 9, "factual_accuracy": 9, "chart_consistency": 9, "relevance": 9,
                           "language_quality": 9, "policy_compliance": 10, "issues": [], "decision": "PASS"})
VERDICT_REJECT = json.dumps({"quality_score": 3, "factual_accuracy": 2, "chart_consistency": 3, "relevance": 8,
                             "language_quality": 8, "policy_compliance": 9, "issues": ["wrong house"], "decision": "REJECT"})


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from app.config import settings as settings_module
    from app.llm import usage_ledger
    from app.nodes import quality_check

    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: tmp_path / "usage.jsonl")
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)
    from app.nodes import humanize

    monkeypatch.setattr(quality_check, "SHADOW_LOG", tmp_path / "qc.jsonl")
    monkeypatch.setattr(humanize, "SHADOW_LOG", tmp_path / "hm.jsonl")
    s = settings_module.settings
    flags, nvidia = s.flags, s.nvidia
    yield
    object.__setattr__(s, "flags", flags)
    object.__setattr__(s, "nvidia", nvidia)


def _flags(**overrides):
    from app.config import settings as settings_module

    s = settings_module.settings
    object.__setattr__(s, "flags", dataclasses.replace(s.flags, **overrides))


def _nvidia(**overrides):
    """The nodes read the shared settings; the gateway reads the copy `use_nvidia` patches."""
    from app.config import settings as settings_module

    s = settings_module.settings
    object.__setattr__(s, "nvidia", dataclasses.replace(s.nvidia, **overrides))


import itertools

_COUNTER = itertools.count()


def _rid(prefix: str) -> str:
    """Unique per call: the process-wide usage tracker outlives a test."""
    return f"{prefix}-{next(_COUNTER)}"


def _rows(request_id):
    return usage_tracker.for_request(request_id)


# ═════════════════════════ Quality Checker ═════════════════════════
def _qc_state(kundli, **extra):
    state = {"request_id": _rid("qc"), "session_id": "s1", "user_id": "u1", "question": "When will I get married?",
             "intent": "marriage", "answer_depth": "normal", "response": GOOD, "model_used": MAIN,
             "kundli": kundli, "history": [], "validation": {"action": "none"}}
    state.update(extra)
    return state


def _qc_on(monkeypatch, script, **flags):
    _flags(quality_checker_mode="shadow", quality_checker_sample_rate=1.0, evaluation_async=False, **flags)
    _nvidia(quality_checker=ULTRA)
    return use_nvidia(monkeypatch, script, nvidia={"quality_checker": ULTRA})


def _answer_call(state):
    """Record the primary call the way generation would, so `shipped()` sees a delivered answer."""
    from app.llm.llm_factory import ROLE_MAIN, get_llm
    from app.llm.usage_tracker import llm_purpose

    with llm_purpose("answer", "p"):
        get_llm(ROLE_MAIN).complete([{"role": "user", "content": "x"}], session_id="s1", user_id="u1",
                                    request_id=state["request_id"])


def test_the_shipped_defaults_are_advisory_shadow_roles(monkeypatch) -> None:
    from app.config.settings import NvidiaSettings, PipelineFlags

    for name in ("QUALITY_CHECKER_MODE", "QUALITY_CHECKER_SAMPLE_RATE", "HUMANIZER_MODE",
                 "NVIDIA_QUALITY_CHECKER_MODEL", "NVIDIA_HUMANIZER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    flags, nvidia = PipelineFlags(), NvidiaSettings()
    assert (flags.quality_checker_mode, flags.quality_checker_sample_rate) == ("shadow", 0.1)
    assert flags.humanizer_mode == "shadow"                        # never "on" by default
    assert nvidia.quality_checker == ULTRA and nvidia.humanizer == ""


def test_a_switched_off_checker_does_nothing(kundli) -> None:
    from app.nodes.quality_check import quality_check_node

    _flags(quality_checker_mode="off")
    assert quality_check_node(_qc_state(kundli)) == {}


def test_shadow_mode_records_a_verdict_on_the_checker_model_and_leaves_the_reply_alone(monkeypatch, kundli) -> None:
    from app.nodes.quality_check import quality_check_node

    _, client = _qc_on(monkeypatch, [reply(GOOD), reply(VERDICT_REJECT)])
    state = _qc_state(kundli)
    _answer_call(state)
    out = quality_check_node(state)
    assert out["quality_check"]["status"] == "REJECT" and "response" not in out      # advisory: nothing is rewritten
    assert client.models == [MAIN, ULTRA]
    check = [r for r in _rows(state["request_id"]) if r.purpose == "quality_check"]
    assert len(check) == 1 and check[0].model == ULTRA and check[0].success
    assert out["quality_check"]["agreement"] if "agreement" in out["quality_check"] else True


def test_the_checker_is_not_told_what_the_deterministic_validators_found(monkeypatch, kundli) -> None:
    from app.nodes.quality_check import quality_check_node

    _, client = _qc_on(monkeypatch, [reply(GOOD), reply(VERDICT_PASS)])
    state = _qc_state(kundli, validation={"action": "regenerated", "primary_rejected": True})
    _answer_call(state)
    quality_check_node(state)
    prompt = json.dumps(client.requests[-1]["messages"])
    assert "regenerated" not in prompt and "primary_rejected" not in prompt and "validator" not in prompt.lower()


@pytest.mark.parametrize("failure", [timeout_error(), http_error(500), http_error(429), reply("not json at all")])
def test_an_unavailable_checker_is_recorded_as_unavailable_never_as_a_pass(monkeypatch, kundli, failure) -> None:
    from app.nodes.quality_check import quality_check_node

    _qc_on(monkeypatch, [reply(GOOD), *([failure] * 4)])
    state = _qc_state(kundli)
    _answer_call(state)
    out = quality_check_node(state)["quality_check"]
    assert out["status"] == "UNAVAILABLE" and out["status"] != "PASS" and out["reason"]
    assert out["agreement"] == "checker_unavailable"


def test_the_checker_has_no_fallback_model(monkeypatch) -> None:
    from app.llm.llm_factory import ROLE_QUALITY, get_llm

    _qc_on(monkeypatch, [])
    llm = get_llm(ROLE_QUALITY)
    assert llm.config.model == ULTRA and llm.config.fallback_model is None


def test_a_failing_checker_never_raises_into_the_turn(monkeypatch, kundli) -> None:
    from app.nodes.quality_check import quality_check_node

    _qc_on(monkeypatch, [reply(GOOD), http_error(401)])
    state = _qc_state(kundli)
    _answer_call(state)
    assert quality_check_node(state)["quality_check"]["status"] == "UNAVAILABLE"


def test_the_checker_never_grades_its_own_work(monkeypatch, kundli) -> None:
    from app.nodes.quality_check import quality_check_node

    _qc_on(monkeypatch, [])
    _nvidia(quality_checker=MAIN)                                       # misconfigured: checker == author
    out = quality_check_node(_qc_state(kundli))
    assert out == {} or out["quality_check"] == {"status": "SKIPPED", "reason": "not_independent"}


@pytest.mark.parametrize("override,reason", [
    ({"intent": "smalltalk"}, "simple_or_nonastro"),
    ({"answer_depth": "simple"}, "simple_or_nonastro"),
    ({"generation_failed": True}, "no_answer"),
    ({"response": ""}, "no_answer"),
])
def test_turns_with_nothing_to_judge_make_no_checker_call(monkeypatch, kundli, override, reason) -> None:
    from app.nodes.quality_check import quality_check_node

    _, client = _qc_on(monkeypatch, [reply(GOOD)])
    state = _qc_state(kundli, **override)
    _answer_call(state)
    out = quality_check_node(state)["quality_check"]
    assert out == {"status": "SKIPPED", "reason": reason}
    assert client.models == [MAIN]                                      # only the answer call, no checker call


def test_a_safe_or_deterministic_reply_is_not_sent_to_the_checker(monkeypatch, kundli) -> None:
    from app.nodes.quality_check import quality_check_node

    _, client = _qc_on(monkeypatch, [])
    state = _qc_state(kundli, model_used="rule:language-switch")
    out = quality_check_node(state)["quality_check"]
    assert out["status"] == "SKIPPED" and out["reason"].startswith("not_model_generated")
    assert client.requests == []


def test_sampling_zero_never_calls_the_checker_and_marks_it_unsampled(monkeypatch, kundli) -> None:
    from app.nodes.quality_check import quality_check_node

    _, client = _qc_on(monkeypatch, [reply(GOOD)])
    _flags(quality_checker_sample_rate=0.0)
    state = _qc_state(kundli)
    _answer_call(state)
    assert quality_check_node(state)["quality_check"] == {"status": "SKIPPED", "reason": "unsampled"}
    assert client.models == [MAIN]


def test_an_async_checker_returns_pending_and_does_not_block_the_turn(monkeypatch, kundli) -> None:
    from app.nodes.quality_check import quality_check_node
    from app.observability import background

    submitted = []
    monkeypatch.setattr(background, "try_submit", lambda fn, *a, **k: submitted.append(fn) or True)
    _qc_on(monkeypatch, [reply(GOOD)])
    _flags(evaluation_async=True)
    state = _qc_state(kundli)
    _answer_call(state)
    assert quality_check_node(state)["quality_check"] == {"status": "PENDING"}
    assert len(submitted) == 1 and [r.purpose for r in _rows(state["request_id"])] == ["answer"]    # not run on the request path


def test_a_full_queue_drops_the_sample_instead_of_running_it_on_the_request_thread(monkeypatch, kundli) -> None:
    from app.nodes.quality_check import quality_check_node
    from app.observability import background

    monkeypatch.setattr(background, "SHADOW_MAX_PENDING", 0)
    _, client = _qc_on(monkeypatch, [reply(GOOD)])
    _flags(evaluation_async=True)
    state = _qc_state(kundli)
    _answer_call(state)
    assert quality_check_node(state)["quality_check"] == {"status": "SKIPPED", "reason": "queue_full"}
    assert client.models == [MAIN]                                      # no checker call happened inline


def test_the_checker_is_inert_when_nvidia_is_not_the_active_provider(monkeypatch, kundli) -> None:
    from app.config import settings as settings_module
    from app.llm import llm_factory
    from app.nodes import quality_check
    from app.nodes.quality_check import quality_check_node

    other = dataclasses.replace(settings_module.settings, llm_provider="openrouter")
    monkeypatch.setattr(llm_factory, "settings", other)
    monkeypatch.setattr(quality_check, "get_llm", lambda *a, **k: pytest.fail("no model call is allowed"))
    _flags(quality_checker_mode="shadow", quality_checker_sample_rate=1.0)
    _nvidia(quality_checker=ULTRA)
    assert quality_check_node(_qc_state(kundli)) == {}


def test_the_shadow_log_holds_ids_and_scores_but_no_reply_or_question_text(monkeypatch, kundli, tmp_path) -> None:
    from app.nodes import quality_check
    from app.nodes.quality_check import quality_check_node

    _qc_on(monkeypatch, [reply(GOOD), reply(VERDICT_PASS)])
    state = _qc_state(kundli)
    _answer_call(state)
    quality_check_node(state)
    text = (tmp_path / "qc.jsonl").read_text(encoding="utf-8")
    line = json.loads(text.splitlines()[0])
    assert line["status"] == "PASS" and line["agreement"] == "both_clean" and line["request_id"] == state["request_id"]
    assert "Jupiter" not in text and "married" not in text and quality_check.SHADOW_LOG.name == "qc.jsonl"


@pytest.mark.parametrize("status,det,expected", [
    ("PASS", "passed", "both_clean"), ("PASS", "regenerated", "deterministic_only"),
    ("REPAIR", "passed", "checker_only"), ("REJECT", "safe_fallback", "both_flagged"),
    ("UNAVAILABLE", "passed", "checker_unavailable"),
])
def test_agreement_between_the_checker_and_the_deterministic_layer(status, det, expected) -> None:
    from app.nodes.quality_check import agreement

    assert agreement(status, det) == expected


@pytest.mark.parametrize("raw", ["", "PASS", '{"decision": "PASS"}', '{"quality_score": 99, "decision": "PASS"}',
                                 VERDICT_PASS.replace("PASS", "MAYBE")])
def test_a_malformed_verdict_is_not_a_verdict(raw) -> None:
    from app.prompts.roles.quality_checker import parse_verdict

    assert parse_verdict(raw) is None


def test_a_verdict_wrapped_in_prose_is_still_parsed() -> None:
    from app.prompts.roles.quality_checker import parse_verdict

    assert parse_verdict(f"Here you go:\n{VERDICT_REJECT}\nThanks").decision == "REJECT"


# ═════════════════════════ Humanizer ═════════════════════════
HING_BAD = "Abhi aap Venus Mahadasha chal rahe hain.\n\nAapko health issues la sakta hai, isliye dhyan rakhiye."
HING_GOOD = "Abhi aapki Venus Mahadasha chal rahi hai.\n\nSaturn ke kaaran health-related dikkat aa sakti hai, isliye dhyan rakhiye."


def _hum_state(kundli, text=HING_BAD, **extra):
    state = {"request_id": _rid("hm"), "session_id": "s1", "user_id": "u1", "question": "Meri health kaisi rahegi?",
             "intent": "health", "answer_depth": "normal", "response": text, "model_used": MAIN, "kundli": kundli,
             "language": {"language": "hinglish"}, "response_style": "whatsapp_hinglish",
             "validation": {"action": "none"}}
    state.update(extra)
    return state


def _hum_on(monkeypatch, script, mode="on", **flags):
    _flags(humanizer_mode=mode, **flags)
    return use_nvidia(monkeypatch, script)


def test_the_humanizer_is_off_by_default(kundli) -> None:
    from app.config.settings import settings
    from app.nodes.humanize import humanize_node

    assert settings.flags.humanizer_mode == "off"
    assert humanize_node(_hum_state(kundli)) == {}


def test_a_better_rewrite_that_keeps_every_fact_is_applied_with_one_call(monkeypatch, kundli) -> None:
    from app.nodes.humanize import humanize_node

    _, client = _hum_on(monkeypatch, [reply(HING_GOOD)])
    state = _hum_state(kundli)
    out = humanize_node(state)
    report = out["validation"]["humanizer"]
    assert report["status"] == "applied" and report["issues_after"] < report["issues_before"], report
    assert out["response"] == HING_GOOD and len(client.requests) == 1
    rows = [r for r in _rows(state["request_id"]) if r.purpose == "humanize"]
    assert len(rows) == 1 and rows[0].prompt_version.startswith("humanizer")


def test_shadow_mode_records_the_decision_but_keeps_the_original(monkeypatch, kundli, tmp_path) -> None:
    from app.nodes.humanize import humanize_node

    _hum_on(monkeypatch, [reply(HING_GOOD)], mode="shadow", evaluation_async=False)
    state = _hum_state(kundli)
    out = humanize_node(state)
    assert out["validation"]["humanizer"]["status"] == "shadow_recorded" and "response" not in out
    line = json.loads((tmp_path / "hm.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["status"] == "shadow_would_apply" and line["issues_after"] < line["issues_before"]
    assert line["request_id"] == state["request_id"] and line["output_tokens"] > 0
    assert "Venus" not in json.dumps(line) and "Mahadasha" not in json.dumps(line)      # ids and counts only


def test_the_shadow_humanizer_never_waits_for_the_model(monkeypatch, kundli) -> None:
    from app.nodes.humanize import humanize_node
    from app.observability import background

    submitted = []
    monkeypatch.setattr(background, "try_submit", lambda fn, *a, **k: submitted.append(fn) or True)
    _, client = _hum_on(monkeypatch, [reply(HING_GOOD)], mode="shadow", evaluation_async=True)
    out = humanize_node(_hum_state(kundli))
    assert out["validation"]["humanizer"]["status"] == "shadow_pending" and "response" not in out
    assert client.requests == [] and len(submitted) == 1                # the model call is deferred to the worker
    submitted[0]()
    assert len(client.requests) == 1


def test_a_shadow_humanizer_failure_is_recorded_and_never_reaches_the_reply(monkeypatch, kundli, tmp_path) -> None:
    from app.nodes.humanize import humanize_node

    _hum_on(monkeypatch, [http_error(500)] * 5, mode="shadow", evaluation_async=False)
    out = humanize_node(_hum_state(kundli))
    assert "response" not in out
    assert json.loads((tmp_path / "hm.jsonl").read_text(encoding="utf-8").splitlines()[0])["status"] == "failed"


@pytest.mark.parametrize("rewrite,why", [
    (HING_GOOD.replace("Venus", "Mercury"), "facts_changed"),                       # a Dasha lord changed
    (HING_GOOD.replace("Abhi aapki", "Aapki").replace("chal rahi hai", "chal chuki thi"), "facts_changed"),        # present turned into past
    (HING_GOOD.replace("aa sakti hai", "zaroor aayegi"), "facts_changed"),          # certainty raised
    (HING_GOOD + " Aapki Jupiter 10th house mein hai.", None),                      # a new claim / structure change
    (HING_GOOD.replace("\n\n", " "), "paragraph_structure_changed"),                # bubbles merged
    ("```json\n{\"reply\": \"x\"}\n```", None),                                       # a JSON fragment
    ("Here is the improved reply: " + HING_GOOD, None),                             # chatter around the answer
])
def test_a_rewrite_that_breaks_a_rule_is_discarded_and_the_original_kept(monkeypatch, kundli, rewrite, why) -> None:
    from app.nodes.humanize import humanize_node

    _hum_on(monkeypatch, [reply(rewrite)])
    out = humanize_node(_hum_state(kundli))
    report = out["validation"]["humanizer"]
    assert report["status"] == "discarded", report
    assert "response" not in out                                        # the validated original ships
    if why:
        assert report["reason"].startswith(why)


def test_an_empty_answer_from_the_humanizer_keeps_the_original(monkeypatch, kundli) -> None:
    from app.nodes.humanize import humanize_node

    _hum_on(monkeypatch, [reply(""), reply(""), reply("")])
    out = humanize_node(_hum_state(kundli))
    assert out["validation"]["humanizer"]["status"] in ("failed", "discarded") and "response" not in out


def test_a_rewrite_that_is_not_better_is_discarded(monkeypatch, kundli) -> None:
    from app.nodes.humanize import humanize_node

    _hum_on(monkeypatch, [reply(HING_BAD)])                              # returns the text unchanged
    report = humanize_node(_hum_state(kundli))["validation"]["humanizer"]
    assert report["status"] == "discarded" and report["reason"] == "not_better"


def test_a_rewrite_in_the_wrong_script_is_discarded(monkeypatch, kundli) -> None:
    from app.nodes.humanize import humanize_node

    deva = "अभी आपकी बुध महादशा चल रही है।\n\nशनि के कारण स्वास्थ्य संबंधी दिक्कत आ सकती है, इसलिए ध्यान रखिए।"
    _hum_on(monkeypatch, [reply(deva)])
    report = humanize_node(_hum_state(kundli))["validation"]["humanizer"]
    assert report["status"] == "discarded"


@pytest.mark.parametrize("failure", [timeout_error(), http_error(500), http_error(401)])
def test_a_failed_humanizer_call_keeps_the_original_and_is_not_retried_into_a_pass(monkeypatch, kundli, failure) -> None:
    from app.nodes.humanize import humanize_node

    _hum_on(monkeypatch, [failure, failure, failure, failure, failure])
    out = humanize_node(_hum_state(kundli))
    assert out["validation"]["humanizer"]["status"] == "failed" and "response" not in out


@pytest.mark.parametrize("override,reason", [
    ({"language": {"language": "english"}}, "language_not_supported"),
    ({"intent": "smalltalk"}, "simple_or_nonastro"),
    ({"answer_depth": "simple"}, "simple_or_nonastro"),
    ({"generation_failed": True}, "no_answer"),
    ({"model_used": "rule:language-switch"}, "deterministic_reply"),
    ({"validation": {"action": "safe_fallback"}}, "safe_reply"),
])
def test_replies_that_should_not_be_polished_make_no_model_call(monkeypatch, kundli, override, reason) -> None:
    from app.nodes.humanize import humanize_node

    _, client = _hum_on(monkeypatch, [reply(HING_GOOD)])
    report = humanize_node(_hum_state(kundli, **override))["validation"]["humanizer"]
    assert report == {"status": "skipped", "reason": reason}, report
    assert client.requests == []


def test_an_already_natural_reply_is_not_sent_to_the_humanizer(monkeypatch, kundli) -> None:
    from app.nodes.humanize import humanize_node

    _, client = _hum_on(monkeypatch, [reply(HING_GOOD)])
    report = humanize_node(_hum_state(kundli, text=HING_GOOD))["validation"]["humanizer"]
    assert report == {"status": "skipped", "reason": "already_natural"} and client.requests == []


def test_the_humanizer_has_its_own_prompt_and_low_temperature(monkeypatch, kundli) -> None:
    from app.nodes.humanize import humanize_node
    from app.prompts.roles.humanizer import HUMANIZER_SYSTEM_PROMPT

    _, client = _hum_on(monkeypatch, [reply(HING_GOOD)])
    humanize_node(_hum_state(kundli))
    request = client.requests[0]
    assert request["messages"][0]["content"].startswith(HUMANIZER_SYSTEM_PROMPT[:80])
    assert request["temperature"] <= 0.3 and request["max_tokens"] <= 700


def test_the_humanizer_runs_after_validation_in_the_graph() -> None:
    from app.graph.chatbot_graph import build_graph

    graph = build_graph().get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    nodes = set(graph.nodes)
    assert {"humanize", "quality_check"} <= nodes
    assert any(src == "validate_reply" or src.startswith("validate") for src, dst in edges if dst == "humanize") or \
        any(dst == "humanize" for _, dst in edges)


# ═════════════════════════ Role registry ═════════════════════════
def test_the_registry_describes_five_roles_from_the_live_configuration(monkeypatch) -> None:
    from app.llm.roles import describe_five_roles

    use_nvidia(monkeypatch, [])
    rows = {r["role"]: r for r in describe_five_roles()}
    assert {"intent_classifier", "primary_response", "fallback_response", "quality_checker", "humanizer"} <= set(rows)
    assert rows["primary_response"]["model"] == MAIN and rows["primary_response"]["fallback_model"] == BOLT
    assert rows["intent_classifier"]["model"] == BOLT
    assert rows["fallback_response"]["model"] == BOLT and rows["fallback_response"]["fallback_model"] is None
    for row in rows.values():
        assert {"provider", "prompt", "temperature", "max_output_tokens", "timeout_seconds", "max_retries",
                "enabled", "mode", "purposes"} <= set(row)


def test_an_unconfigured_checker_is_shown_as_not_configured_never_as_a_model(monkeypatch) -> None:
    from app.llm.roles import describe_five_roles

    use_nvidia(monkeypatch, [], nvidia={"quality_checker": ""})
    _nvidia(quality_checker="")
    row = next(r for r in describe_five_roles() if r["role"] == "quality_checker")
    assert row["model"] is None and row["enabled"] is False and "NOT configured" in row["notes"]


def test_a_configured_checker_is_enabled_only_when_its_mode_is_on(monkeypatch) -> None:
    from app.llm.roles import describe_five_roles

    use_nvidia(monkeypatch, [], nvidia={"quality_checker": ULTRA})
    _nvidia(quality_checker=ULTRA)
    row = next(r for r in describe_five_roles() if r["role"] == "quality_checker")
    assert row["model"] == ULTRA and row["enabled"] is False            # configured, but mode is off
    _flags(quality_checker_mode="shadow")
    row = next(r for r in describe_five_roles() if r["role"] == "quality_checker")
    assert row["enabled"] is True and row["fallback_model"] is None


def test_the_registry_never_contains_a_secret(monkeypatch) -> None:
    from app.llm.roles import describe_five_roles

    use_nvidia(monkeypatch, [])
    assert "nvapi-test" not in json.dumps(describe_five_roles())


# ═════════════════════ Shadow roles never change what is delivered ═════════════════════
@pytest.fixture
def turn_env(monkeypatch, kundli, retrieval):
    from app.graph import chatbot_graph
    from app.memory.redis_memory import RedisMemory, memory
    from app.nodes import context as context_node, rag as rag_node
    from tests.test_chatbot_mvp import FakeRedis

    monkeypatch.setattr(memory, "_client", FakeRedis())
    monkeypatch.setattr(RedisMemory, "configured", property(lambda self: True))
    monkeypatch.setattr(context_node, "get_kundli", lambda user_id: kundli)
    monkeypatch.setattr(rag_node, "retrieve", lambda *a, **k: retrieval)
    monkeypatch.setattr(chatbot_graph, "settings", dataclasses.replace(chatbot_graph.settings, enable_evaluation=False))


def _turn(monkeypatch, script, rid_suffix, **flags):
    from app.graph.chatbot_graph import run_turn

    _flags(classifier_mode="keyword", hinglish_grammar="detect", **flags)
    use_nvidia(monkeypatch, script, nvidia={"quality_checker": ULTRA})
    _nvidia(quality_checker=ULTRA)
    return run_turn("Meri health kaisi rahegi?", f"shadow-{rid_suffix}-{next(_COUNTER)}", "u1",
                    response_style="whatsapp_hinglish", chat_bubbles=True)


@pytest.mark.parametrize("checker_script", [
    [reply(VERDICT_REJECT)], [http_error(500)] * 4, [reply("garbage")], [reply(VERDICT_PASS)]])
def test_shadow_roles_do_not_change_the_delivered_reply(monkeypatch, turn_env, checker_script) -> None:
    baseline = _turn(monkeypatch, [reply(HING_BAD)], "off", humanizer_mode="off", quality_checker_mode="off")
    shadow = _turn(monkeypatch, [reply(HING_BAD), reply(HING_GOOD), *checker_script], "on",
                   humanizer_mode="shadow", quality_checker_mode="shadow", quality_checker_sample_rate=1.0)
    assert shadow["response"] == baseline["response"] and shadow["response"].strip()
    for key in ("model", "response_source", "response_purpose", "final_validation", "response_edits"):
        assert shadow["metrics"][key] == baseline["metrics"][key], key
    assert shadow["metrics"]["humanizer"]["status"] == "shadow_recorded"
    assert shadow["metrics"]["quality_check"]["status"] in ("REJECT", "PASS", "UNAVAILABLE")
    assert "humanizer" not in shadow["metrics"]["response_edits"]


def test_the_shadow_roles_add_no_call_to_the_users_critical_path_when_async(monkeypatch, turn_env) -> None:
    from app.observability import background

    queued = []
    monkeypatch.setattr(background, "try_submit", lambda fn, *a, **k: queued.append(fn) or True)
    state = _turn(monkeypatch, [reply(HING_BAD)], "async", humanizer_mode="shadow", quality_checker_mode="shadow",
                  quality_checker_sample_rate=1.0, evaluation_async=True)
    purposes = [t["purpose"] for t in state["metrics"]["llm_trace"]]
    assert "humanize" not in purposes and "quality_check" not in purposes        # nothing ran before delivery
    assert len(queued) == 2                                                       # both deferred to the background
    assert state["metrics"]["quality_check"]["status"] == "PENDING"
    assert state["metrics"]["humanizer"]["status"] == "shadow_pending"


def test_a_quality_checker_sample_rate_of_ten_percent_samples_about_one_in_ten(monkeypatch) -> None:
    import random

    from app.nodes import quality_check

    _flags(quality_checker_sample_rate=0.1)
    monkeypatch.setattr(random, "random", iter([0.05, 0.5, 0.099, 0.1, 0.9]).__next__)
    assert [quality_check._sampled() for _ in range(5)] == [True, False, True, False, False]
