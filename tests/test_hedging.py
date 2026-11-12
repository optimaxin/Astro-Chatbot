"""Hedged main-turn requests: bound the latency tail without extra calls on
normal turns."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from tests.llm_fakes import timeout_error, use_nvidia

MAIN = "nvidia/nemotron-3.5-lightning-30b-a3b"
BOLT = "nvidia/nemotron-3.5-bolt-e4b"
MSG = [{"role": "user", "content": "career?"}]


@pytest.fixture(autouse=True)
def _ledger_to_tmp(tmp_path, monkeypatch):
    from app.llm import usage_ledger

    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: tmp_path / "usage.jsonl")
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)


class ModelClient:
    """Thread-safe fake: behaviour chosen per model."""

    def __init__(self, behaviour):
        self.behaviour, self.calls, self.lock = behaviour, [], threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs["model"])
        delay, text = self.behaviour[kwargs["model"]]
        time.sleep(delay)
        if isinstance(text, BaseException):
            raise text
        msg = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(model=kwargs["model"], usage={"prompt_tokens": 5, "completion_tokens": 3},
                               choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def _setup(monkeypatch, behaviour, hedge=1.0):
    factory, _ = use_nvidia(monkeypatch, [], reliability={"hedge_after_seconds": hedge})
    client = ModelClient(behaviour)
    monkeypatch.setattr(factory, "_openai_client", lambda: client)
    return factory, client


def test_fast_primary_makes_exactly_one_call(monkeypatch) -> None:
    factory, client = _setup(monkeypatch, {MAIN: (0.0, "primary"), BOLT: (0.0, "fallback")})
    assert factory.get_llm(factory.ROLE_MAIN).complete(MSG).text == "primary"
    assert client.calls == [MAIN]


def test_slow_primary_is_hedged_and_the_first_answer_wins(monkeypatch) -> None:
    factory, client = _setup(monkeypatch, {MAIN: (4.0, "primary"), BOLT: (0.0, "fallback")})
    started = time.perf_counter()
    result = factory.get_llm(factory.ROLE_MAIN).complete(MSG, request_id="hedge-1")
    elapsed = time.perf_counter() - started
    assert result.text == "fallback" and elapsed < 3.0, "did not wait for the stalled primary"
    assert result.usage.fallback_used and result.usage.error_code == "slow_primary"
    assert sorted(client.calls) == sorted([MAIN, BOLT])


def test_primary_that_answers_after_the_hedge_starts_can_still_win(monkeypatch) -> None:
    factory, _ = _setup(monkeypatch, {MAIN: (1.5, "primary"), BOLT: (4.0, "fallback")})
    assert factory.get_llm(factory.ROLE_MAIN).complete(MSG).text == "primary"


def test_fast_primary_failure_still_falls_back(monkeypatch) -> None:
    factory, client = _setup(monkeypatch, {MAIN: (0.0, timeout_error()), BOLT: (0.0, "fallback")})
    assert factory.get_llm(factory.ROLE_MAIN).complete(MSG).text == "fallback"
    assert client.calls == [MAIN, BOLT]


def test_both_legs_failing_reports_llm_call_failed(monkeypatch) -> None:
    factory, _ = _setup(monkeypatch, {MAIN: (1.5, timeout_error()), BOLT: (0.0, timeout_error())})
    with pytest.raises(factory.LLMCallFailed):
        factory.get_llm(factory.ROLE_MAIN).complete(MSG)


def test_hedging_off_restores_sequential_behaviour(monkeypatch) -> None:
    factory, client = _setup(monkeypatch, {MAIN: (0.4, "primary"), BOLT: (0.0, "fallback")}, hedge=0)
    assert factory.get_llm(factory.ROLE_MAIN).complete(MSG).text == "primary"
    assert client.calls == [MAIN]


def test_classifier_role_is_never_hedged(monkeypatch) -> None:
    factory, client = _setup(monkeypatch, {BOLT: (0.5, '{"intent":"career","confidence":0.9}'),
                                           MAIN: (0.0, "x")})
    factory.get_llm(factory.ROLE_CLASSIFY).complete(MSG)
    assert client.calls == [BOLT]
