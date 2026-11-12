"""NVIDIA_MAIN_REASONING_MODELS: reasoning for the main answer only, opt-in."""

from __future__ import annotations

import pytest

from app.llm.llm_factory import ROLE_CLASSIFY, ROLE_MAIN
from tests.llm_fakes import reply, stream, use_nvidia

BOLT = "nvidia/nemotron-3.5-bolt-e4b"
LIGHTNING = "nvidia/nemotron-3.5-lightning-30b-a3b"
MSG = [{"role": "user", "content": "hello"}]


@pytest.fixture(autouse=True)
def _ledger_to_tmp(tmp_path, monkeypatch):
    from app.llm import usage_ledger

    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: tmp_path / "usage.jsonl")
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)


def _thinking(request: dict) -> bool:
    return request["extra_body"]["chat_template_kwargs"]["enable_thinking"]


def test_off_by_default_everywhere(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [reply("a"), reply("b")])
    factory.get_llm(ROLE_MAIN).complete(MSG)
    factory.get_llm(ROLE_CLASSIFY).complete(MSG)
    assert [_thinking(r) for r in client.requests] == [False, False]


def test_main_turn_reasons_only_for_listed_models(monkeypatch) -> None:
    nvidia = {"main": BOLT, "fallback": LIGHTNING, "main_reasoning_models": (BOLT,),
              "reasoning_token_budget": 1500}
    factory, client = use_nvidia(monkeypatch, [reply("a"), reply("b")], nvidia=nvidia,
                                 reliability={"hedge_after_seconds": 0})
    main = factory.get_llm(ROLE_MAIN)
    main.complete(MSG)
    factory.get_llm(ROLE_CLASSIFY).complete(MSG)        # classifier is Bolt too
    first, second = client.requests
    assert _thinking(first) is True and first["max_tokens"] == main.config.max_tokens + 1500
    assert _thinking(second) is False, "classifier/evaluator calls never reason"


def test_unlisted_main_model_is_unchanged(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [reply("a")], nvidia={"main_reasoning_models": (BOLT,)},
                                 reliability={"hedge_after_seconds": 0})
    main = factory.get_llm(ROLE_MAIN)                   # Lightning, not listed
    main.complete(MSG)
    assert _thinking(client.requests[0]) is False
    assert client.requests[0]["max_tokens"] == main.config.max_tokens


def test_streamed_main_turn_reasons_too(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [stream(["Hel", "lo"])],
                                 nvidia={"main": BOLT, "main_reasoning_models": (BOLT,)})
    assert "".join(factory.get_llm(ROLE_MAIN).stream(MSG)) == "Hello"
    assert _thinking(client.requests[0]) is True
