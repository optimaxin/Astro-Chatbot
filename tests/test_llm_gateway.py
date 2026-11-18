"""NVIDIA provider, routing, reliability and token accounting.

All upstream calls go to `tests.llm_fakes.FakeClient`; nothing here touches
the network. Real-endpoint checks live in `scripts/smoke_nvidia.py`.
"""

from __future__ import annotations

import json

import pytest

from app.llm import reliability
from app.llm.llm_factory import ROLE_CLASSIFY, ROLE_MAIN
from app.llm.model_registry import get_spec, learn_unsupported
from app.llm.usage_tracker import usage_tracker
from tests.llm_fakes import http_error, reply, timeout_error, use_nvidia

MISTRAL = "mistralai/mistral-large-4-1t-a52b"
BOLT = "nvidia/nemotron-3.5-bolt-e4b"
LIGHTNING = "nvidia/nemotron-3.5-lightning-30b-a3b"
MAIN = LIGHTNING  # the configured primary (decided 2026-11-12)
MSG = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]


@pytest.fixture(autouse=True)
def _ledger_to_tmp(tmp_path, monkeypatch):
    from app.llm import usage_ledger

    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: tmp_path / "usage.jsonl")
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)
    return tmp_path / "usage.jsonl"


def _rows(request_id: str):
    return usage_tracker.for_request(request_id)


# ── configuration & providers ─────────────────────────────────────────
def test_nvidia_is_the_default_provider(monkeypatch) -> None:
    from app.config.settings import Settings

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **k: None, raising=False)
    assert Settings().llm_provider == "nvidia"


def test_nvidia_role_models_match_the_beta_plan(monkeypatch) -> None:
    factory, _ = use_nvidia(monkeypatch, [])
    roles = factory.describe_roles()
    assert roles[ROLE_MAIN]["model"] == MAIN
    assert roles[ROLE_MAIN]["fallback"] == BOLT
    assert roles[ROLE_MAIN]["provider"] == "nvidia"
    assert roles[ROLE_CLASSIFY]["model"] == BOLT
    assert MISTRAL not in {r["model"] for r in roles.values()}, "benchmark-only model"


def test_model_ids_are_configurable_from_env(monkeypatch) -> None:
    from app.config.settings import NvidiaSettings

    monkeypatch.setenv("NVIDIA_MODEL", "vendor/other-model")
    monkeypatch.setenv("NVIDIA_LIGHT_INTENTS", "smalltalk, General_Astrology")
    s = NvidiaSettings()
    assert s.main == "vendor/other-model"
    assert s.light_intents == ("smalltalk", "general_astrology")


def test_nvidia_client_points_at_nvidia_with_sdk_retries_off(monkeypatch) -> None:
    from app.config.settings import settings
    import dataclasses

    from app.llm.providers import get_provider

    patched = dataclasses.replace(settings, nvidia=dataclasses.replace(settings.nvidia, api_key="k"))
    client = get_provider("nvidia").build_client(patched)
    assert "integrate.api.nvidia.com" in str(client.base_url)
    assert client.max_retries == 0, "the gateway owns retries"


def test_missing_nvidia_key_fails_with_an_actionable_message(monkeypatch) -> None:
    import dataclasses

    from app.llm import llm_factory

    patched = dataclasses.replace(
        llm_factory.settings, llm_provider="nvidia",
        nvidia=dataclasses.replace(llm_factory.settings.nvidia, api_key=""),
    )
    monkeypatch.setattr(llm_factory, "settings", patched)
    monkeypatch.setattr(llm_factory, "_client", None)
    with pytest.raises(llm_factory.LLMNotConfigured) as excinfo:
        llm_factory._openai_client()
    assert "NVIDIA_API_KEY" in str(excinfo.value)
    assert "build.nvidia.com" in str(excinfo.value)


def test_openrouter_stays_available_and_keeps_its_extras(monkeypatch) -> None:
    import dataclasses

    from app.llm import llm_factory

    patched = dataclasses.replace(llm_factory.settings, llm_provider="openrouter")
    monkeypatch.setattr(llm_factory, "settings", patched)
    request = llm_factory.build_request(
        provider="openrouter", spec=get_spec("meta-llama/llama-3.3-70b-instruct"),
        messages=MSG, temperature=0.1, max_tokens=10, response_format=None,
        tools=None, tool_choice=None, streaming=False,
    )
    assert request["extra_body"] == {"usage": {"include": True}}


def test_openrouter_extras_are_never_sent_to_nvidia(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [reply("hi")])
    factory.get_llm(ROLE_MAIN).complete(MSG)
    assert "usage" not in (client.requests[0].get("extra_body") or {})


def test_each_model_gets_the_reasoning_switch_it_actually_accepts(monkeypatch) -> None:
    """Verified live: Nemotron honours enable_thinking=false; Mistral Large 4
    ignores it and rejects reasoning_effort='none', so it gets 'low'."""
    factory, client = use_nvidia(monkeypatch, [reply("hi"), reply("hi"), reply("hi")])
    factory.llm_for_model(BOLT).complete(MSG)
    factory.llm_for_model(LIGHTNING).complete(MSG)
    factory.llm_for_model(MISTRAL).complete(MSG)
    no_think = {"enable_thinking": False}
    assert client.requests[0]["extra_body"]["chat_template_kwargs"] == no_think
    assert client.requests[1]["extra_body"]["chat_template_kwargs"] == no_think
    assert client.requests[2]["extra_body"] == {"reasoning_effort": "low"}


def test_lightning_is_known_text_only(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [])
    image = [{"role": "user", "content": [{"type": "text", "text": "?"},
                                          {"type": "image_url", "image_url": {"url": "data:x"}}]}]
    with pytest.raises(factory.LLMCallFailed):
        factory.llm_for_model(LIGHTNING).complete(image)
    assert client.requests == [], "refused locally instead of a 400 from NVIDIA"


# ── routing ───────────────────────────────────────────────────────────
def test_smalltalk_goes_to_the_light_model_with_main_as_fallback(monkeypatch) -> None:
    factory, _ = use_nvidia(monkeypatch, [])
    llm = factory.get_llm(ROLE_MAIN, intent="smalltalk")
    assert (llm.config.model, llm.config.fallback_model) == (BOLT, MAIN)


@pytest.mark.parametrize("intent", ["career", "marriage", "general_astrology", None])
def test_chart_questions_stay_on_the_main_model(monkeypatch, intent) -> None:
    factory, _ = use_nvidia(monkeypatch, [])
    llm = factory.get_llm(ROLE_MAIN, intent=intent)
    assert (llm.config.model, llm.config.fallback_model) == (MAIN, BOLT)


def test_image_turns_go_to_the_vision_model(monkeypatch) -> None:
    """Lightning (primary) is text-only, verified live; Bolt accepts images."""
    factory, _ = use_nvidia(monkeypatch, [])
    llm = factory.get_llm(ROLE_MAIN, intent="smalltalk", has_image=True)
    assert llm.config.model == BOLT


def test_image_fallback_is_dropped_when_fallback_is_known_text_only(monkeypatch) -> None:
    factory, _ = use_nvidia(monkeypatch, [])
    learn_unsupported(BOLT, "vision")
    assert factory.get_llm(ROLE_MAIN, has_image=True).config.fallback_model is None


def test_image_turns_skip_a_text_only_main_model(monkeypatch) -> None:
    factory, _ = use_nvidia(monkeypatch, [], nvidia={"main": LIGHTNING, "fallback": BOLT})
    llm = factory.get_llm(ROLE_MAIN, has_image=True)
    assert (llm.config.model, llm.config.fallback_model) == (BOLT, None)


def test_light_route_can_be_disabled(monkeypatch) -> None:
    factory, _ = use_nvidia(monkeypatch, [], nvidia={"light_model": ""})
    assert factory.get_llm(ROLE_MAIN, intent="smalltalk").config.model == MAIN


# ── error normalisation & backoff ─────────────────────────────────────
@pytest.mark.parametrize("exc, code, retryable, fallback_ok", [
    (http_error(429), reliability.RATE_LIMITED, True, True),
    (http_error(503), reliability.UPSTREAM, True, True),
    # A timeout goes straight to the fallback: retrying a stalled model
    # doubles the user's worst-case wait.
    (timeout_error(), reliability.TIMEOUT, False, True),
    (http_error(401), reliability.AUTH, False, False),
    (http_error(400), reliability.BAD_REQUEST, False, False),
    (http_error(404), reliability.NOT_FOUND, False, True),
    (RuntimeError("?"), reliability.UNKNOWN, False, True),
])
def test_errors_are_normalised(exc, code, retryable, fallback_ok) -> None:
    info = reliability.classify_error(exc)
    assert (info.code, info.retryable, info.fallback_ok) == (code, retryable, fallback_ok)


def test_retry_after_is_honoured_but_capped() -> None:
    info = reliability.classify_error(http_error(429, headers={"retry-after": "3"}))
    assert info.retry_after == 3.0
    assert reliability.backoff_delay(1, base=0.5, cap=8, retry_after=3) == 3
    assert reliability.backoff_delay(1, base=0.5, cap=2, retry_after=300) == 2


def test_backoff_grows_exponentially_and_stays_bounded() -> None:
    import random

    rng = random.Random(0)
    for n in range(1, 12):
        assert 0 <= reliability.backoff_delay(n, base=0.5, cap=8, rng=rng) <= min(8, 0.5 * 2 ** (n - 1))


# ── retries, fallback, circuit breaker ────────────────────────────────
def test_rate_limit_is_retried_then_succeeds_without_fallback(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [http_error(429), reply("answer")])
    result = factory.get_llm(ROLE_MAIN).complete(MSG, request_id="r-429")
    assert result.text == "answer"
    assert client.models == [MAIN, MAIN]
    rows = _rows("r-429")
    assert [(r.success, r.attempt, r.fallback_used) for r in rows] == [(False, 0, False), (True, 1, False)]
    assert rows[0].error_code == "rate_limited:429"


def test_retries_are_limited(monkeypatch) -> None:
    factory, client = use_nvidia(
        monkeypatch, [http_error(503)] * 3 + [reply("from fallback")],
        reliability={"max_retries": 2},
    )
    result = factory.get_llm(ROLE_MAIN).complete(MSG, request_id="r-lim")
    assert client.models == [MAIN, MAIN, MAIN, BOLT], "3 tries on main, then fallback"
    assert result.usage.fallback_used and result.usage.model == BOLT
    assert result.usage.error_code == "upstream_error:503", "fallback row says why"


def test_auth_failure_does_not_fall_back_on_the_same_provider(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [http_error(401), reply("never")])
    with pytest.raises(factory.LLMCallFailed) as excinfo:
        factory.get_llm(ROLE_MAIN).complete(MSG)
    assert client.models == [MAIN]
    assert excinfo.value.info.code == reliability.AUTH


def test_bad_request_does_not_fall_back(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [http_error(400, "prompt too long"), reply("x")])
    with pytest.raises(factory.LLMCallFailed):
        factory.get_llm(ROLE_MAIN).complete(MSG)
    assert client.models == [MAIN]


def test_empty_reply_falls_back(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [reply(""), reply("real answer")])
    result = factory.get_llm(ROLE_MAIN).complete(MSG, request_id="r-empty")
    assert result.text == "real answer" and client.models == [MAIN, BOLT]
    assert _rows("r-empty")[0].error_code == reliability.EMPTY


@pytest.mark.parametrize("broken", ['{"status": "succ', "```json\n{}", "Aapki Kundli mein <|im_end|>"])
def test_malformed_main_answer_falls_back(monkeypatch, broken) -> None:
    factory, client = use_nvidia(monkeypatch, [reply(broken), reply("real answer")])
    result = factory.get_llm(ROLE_MAIN).complete(MSG, request_id="r-malformed")
    assert result.text == "real answer" and client.models == [MAIN, BOLT]
    assert _rows("r-malformed")[0].error_code == reliability.MALFORMED


def test_short_prose_and_json_roles_are_not_malformed(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [reply("Zaroor!")])
    assert factory.get_llm(ROLE_MAIN).complete(MSG).text == "Zaroor!"
    factory, client = use_nvidia(monkeypatch, [reply('{"intent": "career", "confidence": 0.9}')])
    assert factory.get_llm(factory.ROLE_CLASSIFY).complete(MSG).text.startswith("{")
    assert client.models == [BOLT]


def test_timeout_falls_back_immediately_without_retrying_the_stalled_model(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [timeout_error(), reply("from bolt")],
                                 reliability={"max_retries": 2})
    result = factory.get_llm(ROLE_MAIN).complete(MSG)
    assert client.models == [MAIN, BOLT] and result.text == "from bolt"


def test_circuit_opens_after_threshold_and_skips_the_model(monkeypatch) -> None:
    script = [timeout_error(), reply("fb1"), timeout_error(), reply("fb2"), reply("fb3")]
    factory, client = use_nvidia(
        monkeypatch, script,
        reliability={"max_retries": 0, "circuit_failure_threshold": 2},
    )
    for _ in range(3):
        factory.get_llm(ROLE_MAIN).complete(MSG)
    assert client.models == [MAIN, BOLT, MAIN, BOLT, BOLT], "3rd turn skips the open circuit"
    assert reliability.breaker.state(("nvidia", MAIN), 30) == "open"


def test_circuit_half_opens_after_cooldown_and_closes_on_success() -> None:
    now = [0.0]
    cb = reliability.CircuitBreaker(clock=lambda: now[0])
    key = ("nvidia", MAIN)
    assert cb.record_failure(key, threshold=1)
    assert not cb.allow(key, cooldown=10)
    now[0] = 11
    assert cb.allow(key, cooldown=10), "one trial"
    assert not cb.allow(key, cooldown=10), "only one trial at a time"
    cb.record_success(key)
    assert cb.state(key, 10) == "closed"


def test_deadline_stops_retrying(monkeypatch) -> None:
    factory, client = use_nvidia(
        monkeypatch, [http_error(503), reply("x")],
        reliability={"deadline_seconds": 0.0, "max_retries": 5},
    )
    with pytest.raises(factory.LLMCallFailed):
        factory.get_llm(ROLE_MAIN).complete(MSG)
    assert client.models == [MAIN], "no retry and no fallback once the budget is spent"


# ── capabilities: structured output, tools, images, reasoning ─────────
def test_rejected_response_format_is_learned_and_dropped(monkeypatch) -> None:
    factory, client = use_nvidia(
        monkeypatch, [http_error(400, "response_format json_object is not supported"),
                      reply('{"intent": "career", "confidence": 0.9}')],
    )
    result = factory.get_llm(ROLE_CLASSIFY).complete(MSG)
    assert json.loads(result.text)["intent"] == "career"
    assert "response_format" in client.requests[0]
    assert "response_format" not in client.requests[1]
    assert get_spec(BOLT).json_mode is False


def test_json_schema_downgrades_to_json_mode_when_unsupported(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [reply("{}")])
    learn_unsupported(MAIN, "json_schema")
    schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
    factory.llm_for_model(MAIN).complete(MSG, response_format=schema)
    assert client.requests[0]["response_format"] == {"type": "json_object"}


def test_tool_calls_are_parsed(monkeypatch) -> None:
    call = {"name": "get_panchang", "arguments": '{"date": "2026-03-14", "city": "Varanasi"}'}
    factory, client = use_nvidia(monkeypatch, [reply(None, tool_calls=[call])])
    tools = [{"type": "function", "function": {"name": "get_panchang", "parameters": {}}}]
    result = factory.llm_for_model(MAIN).complete(MSG, tools=tools, tool_choice="auto")
    assert client.requests[0]["tools"] == tools
    assert result.tool_calls[0]["name"] == "get_panchang"
    assert result.tool_calls[0]["parsed_arguments"] == {"date": "2026-03-14", "city": "Varanasi"}


def test_tools_are_refused_not_silently_dropped_for_a_model_without_them(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [])
    learn_unsupported(LIGHTNING, "tools")
    with pytest.raises(factory.LLMCallFailed) as excinfo:
        factory.llm_for_model(LIGHTNING).complete(MSG, tools=[{"type": "function"}])
    assert excinfo.value.info.code == reliability.UNSUPPORTED
    assert client.requests == [], "refused before any upstream call"


def test_think_blocks_never_reach_the_user(monkeypatch) -> None:
    factory, _ = use_nvidia(monkeypatch, [reply("<think>secret plan</think>\nNamaste!")])
    assert factory.llm_for_model(BOLT).complete(MSG).text == "Namaste!"


# ── token accounting ──────────────────────────────────────────────────
def test_provider_usage_is_recorded_as_reported(monkeypatch) -> None:
    factory, _ = use_nvidia(monkeypatch, [reply("hi")])
    usage = factory.get_llm(ROLE_MAIN).complete(MSG).usage
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (12, 5, 17)
    assert usage.token_source == "provider" and usage.provider == "nvidia"
    assert usage.estimated_input_tokens is None


def test_missing_usage_is_estimated_separately_never_as_fact(monkeypatch) -> None:
    factory, _ = use_nvidia(monkeypatch, [reply("hello there", usage=None)])
    usage = factory.get_llm(ROLE_MAIN).complete(MSG).usage
    assert usage.input_tokens is None and usage.total_tokens is None
    assert usage.token_source == "estimated"
    assert usage.estimated_input_tokens > 0 and usage.estimated_output_tokens > 0


def test_free_endpoint_usage_is_not_reported_as_spend(monkeypatch) -> None:
    from app.llm.usage_store import to_row

    factory, _ = use_nvidia(monkeypatch, [reply("hi")])
    usage = factory.get_llm(ROLE_MAIN).complete(MSG).usage
    row = to_row(usage.to_dict())
    assert row["total_cost_usd"] == 0.0, "no NVIDIA price configured => no estimate"
    assert row["actual_cost_usd"] is None, "provider reported no charge, not a $0 charge"


def test_every_attempt_reaches_the_jsonl_ledger(monkeypatch, _ledger_to_tmp) -> None:
    from app.llm.usage_ledger import LEDGER_FIELDS, read_ledger

    factory, _ = use_nvidia(monkeypatch, [http_error(429), reply("ok")])
    factory.get_llm(ROLE_MAIN).complete(MSG, request_id="r-ledger")
    rows = [r for r in read_ledger(_ledger_to_tmp) if r["request_id"] == "r-ledger"]
    assert [r["success"] for r in rows] == [False, True]
    assert set(rows[0]) == set(LEDGER_FIELDS)
    assert rows[1]["token_source"] == "provider" and rows[1]["attempt"] == 1


def test_gemini_to_anthropic_fallback_still_works_even_on_auth_errors(monkeypatch) -> None:
    """Cross-provider fallback has its own key, so a 401 on Gemini may fall back."""
    import dataclasses
    from types import SimpleNamespace

    from app.llm import llm_factory
    from app.llm.providers import anthropic_fallback
    from tests.llm_fakes import FakeClient

    patched = dataclasses.replace(
        llm_factory.settings, llm_provider="gemini",
        gemini=dataclasses.replace(llm_factory.settings.gemini, api_key="g"),
        anthropic=dataclasses.replace(llm_factory.settings.anthropic, api_key="a"),
    )
    monkeypatch.setattr(llm_factory, "settings", patched)
    monkeypatch.setattr(llm_factory, "_openai_client", lambda: FakeClient([http_error(401)]))
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(model=kwargs["model"], usage={"input_tokens": 3, "output_tokens": 2},
                               content=[SimpleNamespace(type="text", text="from claude")])

    monkeypatch.setattr(anthropic_fallback, "_client",
                        SimpleNamespace(messages=SimpleNamespace(create=create)))
    result = llm_factory.get_llm(ROLE_MAIN).complete(MSG)
    assert result.text == "from claude"
    assert result.usage.provider == "anthropic" and result.usage.fallback_used
    assert sent["system"] == "be brief", "system prompt moved to Anthropic's top-level param"


def test_fallback_success_row_keeps_the_primary_error_code() -> None:
    from app.llm.usage_store import to_row

    row = to_row({"request_id": "r", "role": ROLE_MAIN, "provider": "nvidia", "model": BOLT,
                  "success": True, "fallback_used": True, "error_code": "timeout"})
    assert row["status"] == "success" and row["fallback_used"] is True
    assert row["error_code"] == "timeout"
    assert row["error_message"] is None
