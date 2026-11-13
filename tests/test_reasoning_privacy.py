"""Private reasoning never reaches the answer, the ledger or the logs; and the
stream/fallback paths behave on malformed, empty, throttled and interrupted
provider responses. Provider objects are built by hand here so fields the
shared fakes don't model (NIM's separate `reasoning_content`) are covered."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from app.llm.llm_factory import ROLE_MAIN
from tests.llm_fakes import http_error, reply, stream, timeout_error, use_nvidia

BOLT = "nvidia/nemotron-3.5-bolt-e4b"
LIGHTNING = "nvidia/nemotron-3.5-lightning-30b-a3b"
SECRET = "SECRET-REASONING-7781"
MSG = [{"role": "user", "content": "Is Saturn strong in my chart?"}]
REASONING_ON = {"main": BOLT, "fallback": LIGHTNING, "main_reasoning_models": (BOLT,)}
NO_HEDGE = {"hedge_after_seconds": 0}


@pytest.fixture(autouse=True)
def _ledger(tmp_path, monkeypatch):
    from app.llm import usage_ledger

    path = tmp_path / "usage.jsonl"
    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: path)
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)
    return path


def _completion(content, reasoning=None, model=BOLT, choices=True):
    message = SimpleNamespace(content=content, tool_calls=None, reasoning_content=reasoning)
    return SimpleNamespace(model=model, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                           choices=[SimpleNamespace(message=message, finish_reason="stop")] if choices else [])


class _Stream:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    def __iter__(self):
        yield from self.chunks

    def close(self):
        self.closed = True


def _chunk(content=None, reasoning=None, usage=None, model=BOLT):
    choices = [] if usage else [SimpleNamespace(delta=SimpleNamespace(content=content, reasoning_content=reasoning))]
    return SimpleNamespace(model=model, usage=usage, choices=choices)


class _Client:
    """Returns the given objects in order; records requests."""

    def __init__(self, *steps):
        self.steps, self.requests = list(steps), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def _install(monkeypatch, client, **nvidia):
    factory, _ = use_nvidia(monkeypatch, [], nvidia={**REASONING_ON, **nvidia}, reliability=NO_HEDGE)
    monkeypatch.setattr(factory, "_openai_client", lambda: client)
    return factory


@pytest.fixture
def caplog(monkeypatch):
    """The app's 'aicore' logger does not propagate to root, so pytest's own
    caplog never sees it. Capture it directly, at DEBUG, with extras."""
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(f"{record.getMessage()} {record.__dict__}")

    from app.logging_setup import get_logger

    get_logger("test")                         # ensure the 'aicore' tree is set up
    root = logging.getLogger("aicore")
    handler = _Capture(level=logging.DEBUG)
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    out = SimpleNamespace(records=records, set_level=lambda *_: None)
    out.text_fn = lambda: "\n".join(records)
    yield out
    root.removeHandler(handler)
    root.setLevel(old_level)


def _assert_private(ledger, caplog, *texts):
    for text in texts:
        assert SECRET not in text
    if ledger.exists():
        assert SECRET not in ledger.read_text(encoding="utf-8")
    assert SECRET not in caplog.text_fn()


# ── reasoning privacy ─────────────────────────────────────────────────
def test_reasoning_field_and_think_tags_never_reach_the_answer(monkeypatch, caplog, _ledger) -> None:
    client = _Client(_completion(f"<think>{SECRET}</think>\nSaturn is in your 6th house.", reasoning=SECRET))
    result = _install(monkeypatch, client).get_llm(ROLE_MAIN).complete(MSG)
    assert result.text == "Saturn is in your 6th house."
    _assert_private(_ledger, caplog, result.text)


def test_streamed_reasoning_is_never_yielded(monkeypatch, caplog, _ledger) -> None:
    chunks = [_chunk(reasoning=SECRET), _chunk(reasoning=" more " + SECRET), _chunk(content="<think>"),
              _chunk(content=SECRET), _chunk(content="</think>"), _chunk(content="Saturn is "),
              _chunk(content="in the 6th."), _chunk(usage={"prompt_tokens": 9, "completion_tokens": 7,
                                                           "total_tokens": 16})]
    out = "".join(_install(monkeypatch, _Client(_Stream(chunks))).get_llm(ROLE_MAIN).stream(MSG))
    assert out == "Saturn is in the 6th."
    _assert_private(_ledger, caplog, out)


def test_an_unterminated_think_block_yields_nothing_rather_than_reasoning(monkeypatch, caplog, _ledger) -> None:
    chunks = [_chunk(content="<think>"), _chunk(content=SECRET)]
    factory = _install(monkeypatch, _Client(_Stream(chunks), _Stream([_chunk(content="fallback answer")])))
    out = "".join(factory.get_llm(ROLE_MAIN).stream(MSG))
    assert SECRET not in out
    _assert_private(_ledger, caplog, out)


# ── provider failure modes ────────────────────────────────────────────
@pytest.mark.parametrize("bad", [
    _completion(None),                 # empty content
    _completion("   "),                # whitespace only
    _completion(None, choices=False),  # malformed: no choices
    _completion(f"<think>{SECRET}</think>"),  # only reasoning, no answer
])
def test_empty_or_malformed_primary_falls_back(monkeypatch, caplog, _ledger, bad) -> None:
    client = _Client(bad, _completion("Fallback answer.", model=LIGHTNING))
    result = _install(monkeypatch, client).get_llm(ROLE_MAIN).complete(MSG)
    assert result.text == "Fallback answer." and result.usage.fallback_used
    assert caplog.records, "the failed primary must have been logged (capture is live)"
    _assert_private(_ledger, caplog, result.text)


@pytest.mark.parametrize("error", [http_error(429, "rate limited"), timeout_error()])
def test_429_and_timeout_before_first_token_fall_back(monkeypatch, error) -> None:
    factory, client = use_nvidia(monkeypatch, [error, stream(["Fallback ", "answer."])],
                                 nvidia={**REASONING_ON}, reliability={**NO_HEDGE, "max_retries": 0})
    out = "".join(factory.get_llm(ROLE_MAIN).stream(MSG))
    assert out == "Fallback answer." and client.models == [BOLT, LIGHTNING]


def test_interrupted_stream_keeps_what_was_sent_and_never_splices(monkeypatch) -> None:
    factory, client = use_nvidia(monkeypatch, [stream(["Saturn ", "is ", "strong"], fail_after=2),
                                               stream(["other model"])],
                                 nvidia={**REASONING_ON}, reliability=NO_HEDGE)
    parts = []
    with pytest.raises(Exception):
        for part in factory.get_llm(ROLE_MAIN).stream(MSG):
            parts.append(part)
    assert "".join(parts) == "Saturn is " and client.models == [BOLT], "no second model after the first token"


def test_duplicate_usage_chunks_are_recorded_once(monkeypatch, _ledger) -> None:
    usage = {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}
    client = _Client(_Stream([_chunk(content="Hi."), _chunk(usage=usage), _chunk(usage=usage)]))
    out = "".join(_install(monkeypatch, client).get_llm(ROLE_MAIN).stream(MSG))
    assert out == "Hi."
    lines = [line for line in _ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1 and '"total_tokens": 11' in lines[0]


def test_client_disconnect_closes_upstream(monkeypatch) -> None:
    upstream = _Stream([_chunk(content="Saturn "), _chunk(content="is "), _chunk(content="strong.")])
    gen = _install(monkeypatch, _Client(upstream)).get_llm(ROLE_MAIN).stream(MSG)
    assert next(gen) == "Saturn "
    gen.close()                        # the client went away mid-answer
    assert upstream.closed
