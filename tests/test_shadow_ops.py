"""Humanizer clean-reply shadow sample, and the operational behaviour of the shadow queue
(saturation, dropped samples, failures, memory). Scripted endpoint, no network."""

from __future__ import annotations

import json
import random
import threading
import time

import pytest

from app.observability import background
from tests.llm_fakes import http_error, reply, use_nvidia
from tests.test_five_roles import (  # noqa: F401 — _isolated is an autouse fixture, imported to activate it
    HING_GOOD, MAIN, ULTRA, _flags, _hum_state, _isolated, _nvidia, _rows)

# A reply with no known language issue, and a harmless rewrite of it.
CLEAN = HING_GOOD
REWRITE = HING_GOOD.replace("isliye dhyan rakhiye", "isliye apna dhyan rakhiye")


def _cmp(a_scores, b_scores, preferred="A", facts=True) -> str:
    d = lambda s: dict(zip(("grammar", "naturalness", "readability", "conversational"), s))  # noqa: E731
    return json.dumps({"A": d(a_scores), "B": d(b_scores), "facts_preserved": facts, "preferred": preferred})


def _on(monkeypatch, script, *, rate=1.0, mode="shadow", **flags):
    _flags(humanizer_mode=mode, humanizer_clean_sample_rate=rate, evaluation_async=False, **flags)
    _nvidia(quality_checker=ULTRA)
    return use_nvidia(monkeypatch, script, nvidia={"quality_checker": ULTRA})


def _log(tmp_path) -> list[dict]:
    path = tmp_path / "hm.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


# ═════════════ humanizer clean-reply sample ═════════════
def test_the_shipped_clean_sample_rate_is_five_percent(monkeypatch) -> None:
    from app.config.settings import PipelineFlags

    monkeypatch.delenv("HUMANIZER_CLEAN_SAMPLE_RATE", raising=False)
    assert PipelineFlags().humanizer_clean_sample_rate == 0.05


def test_a_sampled_clean_reply_is_rewritten_and_judged_blind_and_the_reply_is_untouched(monkeypatch, kundli, tmp_path) -> None:
    from app.nodes.humanize import humanize_node

    monkeypatch.setattr(random, "random", lambda: 0.9)          # no swap: A = original, B = rewrite
    _, client = _on(monkeypatch, [reply(REWRITE), reply(_cmp((4, 3, 4, 3), (5, 5, 5, 4), preferred="B"))])
    state = _hum_state(kundli, text=CLEAN)
    out = humanize_node(state)
    assert "response" not in out                                       # NEVER applied to the user's reply
    assert out["validation"]["humanizer"]["status"] == "shadow_recorded"
    line = _log(tmp_path)[0]
    assert line["sample"] == "clean" and line["status"] == "shadow_clean_compared" and line["changed"] is True
    cmp = line["comparison"]
    assert cmp["status"] == "JUDGED" and cmp["preferred"] == "rewrite" and cmp["facts_preserved_by_judge"] is True
    assert cmp["delta"] == {"grammar": 1, "naturalness": 2, "readability": 1, "conversational": 1}
    assert client.models == [MAIN, ULTRA]                              # humanizer on Lightning, judge on Ultra
    roles = {r.purpose: r.role for r in _rows(state["request_id"])}
    assert roles == {"humanize": "job-humanize", "humanize_compare": "job-humanize-compare"}
    assert "dhyan" not in json.dumps(line) and "Venus" not in json.dumps(line)       # scores only, no text


def test_the_judge_is_blind_the_rewrite_is_not_always_version_b(monkeypatch, kundli, tmp_path) -> None:
    from app.nodes.humanize import humanize_node

    monkeypatch.setattr(random, "random", lambda: 0.1)          # swap: A = rewrite, B = original
    _, client = _on(monkeypatch, [reply(REWRITE), reply(_cmp((5, 5, 5, 5), (3, 3, 3, 3), preferred="A"))])
    humanize_node(_hum_state(kundli, text=CLEAN))
    prompt = client.requests[1]["messages"][1]["content"]
    assert prompt.index("apna dhyan") < prompt.index("VERSION B")       # the rewrite was shown as A
    assert "original" not in prompt.lower() and "rewrite" not in prompt.lower()
    cmp = _log(tmp_path)[0]["comparison"]
    assert cmp["rewrite_was"] == "A" and cmp["preferred"] == "rewrite" and cmp["delta"]["grammar"] == 2


def test_the_judge_preferring_the_original_is_recorded_as_such(monkeypatch, kundli, tmp_path) -> None:
    from app.nodes.humanize import humanize_node

    monkeypatch.setattr(random, "random", lambda: 0.9)
    _on(monkeypatch, [reply(REWRITE), reply(_cmp((5, 5, 5, 5), (4, 3, 4, 3), preferred="A"))])
    humanize_node(_hum_state(kundli, text=CLEAN))
    cmp = _log(tmp_path)[0]["comparison"]
    assert cmp["preferred"] == "original" and cmp["delta"]["naturalness"] == -2


def test_a_zero_sample_rate_sends_nothing_to_the_humanizer(monkeypatch, kundli, tmp_path) -> None:
    from app.nodes.humanize import humanize_node

    _, client = _on(monkeypatch, [reply(REWRITE)], rate=0.0)
    out = humanize_node(_hum_state(kundli, text=CLEAN))
    assert out["validation"]["humanizer"] == {"status": "skipped", "reason": "already_natural"}
    assert client.requests == [] and _log(tmp_path) == []


def test_the_clean_sample_is_shadow_only_never_in_on_mode(monkeypatch, kundli) -> None:
    from app.nodes.humanize import humanize_node

    _, client = _on(monkeypatch, [reply(REWRITE)], mode="on")
    out = humanize_node(_hum_state(kundli, text=CLEAN))
    assert out["validation"]["humanizer"]["reason"] == "already_natural" and client.requests == []
    assert "response" not in out


@pytest.mark.parametrize("override,reason", [
    ({"language": {"language": "english"}}, "language_not_supported"),
    ({"intent": "smalltalk"}, "simple_or_nonastro"),
    ({"validation": {"action": "safe_fallback"}}, "safe_reply"),
])
def test_only_eligible_clean_replies_are_sampled(monkeypatch, kundli, override, reason) -> None:
    from app.nodes.humanize import humanize_node

    _, client = _on(monkeypatch, [reply(REWRITE)])
    out = humanize_node(_hum_state(kundli, text=CLEAN, **override))
    assert out["validation"]["humanizer"]["reason"] == reason and client.requests == []


def test_a_clean_rewrite_that_changes_a_fact_is_discarded_and_not_judged(monkeypatch, kundli, tmp_path) -> None:
    from app.nodes.humanize import humanize_node

    _, client = _on(monkeypatch, [reply(REWRITE.replace("Venus", "Mercury"))])
    humanize_node(_hum_state(kundli, text=CLEAN))
    line = _log(tmp_path)[0]
    assert line["status"] == "discarded" and line["reason"].startswith("facts_changed") and "comparison" not in line
    assert len(client.requests) == 1                                   # no judge call for a rejected rewrite


def test_a_clean_rewrite_that_introduces_a_language_error_is_discarded(monkeypatch, kundli, tmp_path) -> None:
    from app.nodes.humanize import humanize_node

    broken = CLEAN.replace("Saturn ke kaaran health-related dikkat aa sakti hai", "Aapko health issues la sakta hai")
    _on(monkeypatch, [reply(broken)])
    humanize_node(_hum_state(kundli, text=CLEAN))
    assert _log(tmp_path)[0]["status"] == "discarded"


def test_an_unchanged_clean_rewrite_is_recorded_without_a_judge_call(monkeypatch, kundli, tmp_path) -> None:
    from app.nodes.humanize import humanize_node

    _, client = _on(monkeypatch, [reply(CLEAN)])
    humanize_node(_hum_state(kundli, text=CLEAN))
    line = _log(tmp_path)[0]
    assert line["status"] == "shadow_clean_unchanged" and line["changed"] is False and len(client.requests) == 1


@pytest.mark.parametrize("judge", [[http_error(500)] * 4, [reply("not json")], [reply(_cmp((9, 9, 9, 9), (1, 1, 1, 1)))]])
def test_an_unavailable_or_invalid_judge_is_recorded_as_unavailable_and_never_raises(monkeypatch, kundli, tmp_path, judge) -> None:
    from app.nodes.humanize import humanize_node

    _on(monkeypatch, [reply(REWRITE), *judge])
    out = humanize_node(_hum_state(kundli, text=CLEAN))
    assert "response" not in out
    cmp = _log(tmp_path)[0]["comparison"]
    assert cmp["status"] == "UNAVAILABLE" and cmp["reason"] and "delta" not in cmp


def test_parse_comparison_rejects_out_of_range_and_malformed_verdicts() -> None:
    from app.prompts.roles.humanizer import parse_comparison

    assert parse_comparison(_cmp((4, 4, 4, 4), (5, 5, 5, 5), "B"))["preferred"] == "B"
    for raw in ("", "{}", _cmp((0, 4, 4, 4), (5, 5, 5, 5)), _cmp((4, 4, 4, 6), (5, 5, 5, 5)), _cmp((4,) * 4, (5,) * 4, "C"),
                '{"A": {}, "B": {}, "preferred": "A", "facts_preserved": true}'):
        assert parse_comparison(raw) is None


def test_the_clean_sample_is_deferred_to_the_background_when_async(monkeypatch, kundli) -> None:
    from app.nodes.humanize import humanize_node

    queued = []
    monkeypatch.setattr(background, "try_submit", lambda fn, *a, **k: queued.append(fn) or True)
    _, client = _on(monkeypatch, [reply(REWRITE)])
    _flags(evaluation_async=True)
    out = humanize_node(_hum_state(kundli, text=CLEAN))
    assert out["validation"]["humanizer"]["status"] == "shadow_pending" and client.requests == [] and len(queued) == 1


# ═════════════ shadow queue: saturation, drops, failures, memory ═════════════
def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_a_saturated_shadow_queue_drops_samples_instead_of_running_them_inline(monkeypatch) -> None:
    monkeypatch.setattr(background, "SHADOW_MAX_PENDING", 3)
    gate, ran_on = threading.Event(), []
    before = background.stats()
    caller = threading.get_ident()
    job = lambda: (ran_on.append(threading.get_ident()), gate.wait(5))      # noqa: E731
    accepted = [background.try_submit(job) for _ in range(10)]
    try:
        assert accepted.count(True) == 3 and accepted.count(False) == 7
        stats = background.stats()
        assert stats["shadow_dropped"] - before["shadow_dropped"] == 7
        assert stats["shadow_pending"] == 3 and stats["shadow_high_water"] >= 3
        assert caller not in ran_on                                       # never on the request thread
    finally:
        gate.set()
        assert background.drain(5)
    assert background.stats()["shadow_pending"] == 0


def test_a_saturated_shadow_pool_does_not_delay_metering_jobs(monkeypatch) -> None:
    gate, done = threading.Event(), threading.Event()
    for _ in range(4):
        background.try_submit(lambda: gate.wait(5))                      # both shadow workers are now busy
    try:
        background.submit(done.set)                                      # a metering write goes to the other pool
        assert done.wait(2), "metering job was held up behind shadow work"
    finally:
        gate.set()
        background.drain(5)


def test_a_failing_shadow_job_is_counted_and_the_worker_survives() -> None:
    before = background.stats()
    ok = threading.Event()

    def boom():
        raise RuntimeError("shadow job failed")

    assert background.try_submit(boom)
    assert background.try_submit(ok.set)                                 # the next job still runs
    assert background.drain(5) and ok.is_set()
    after = background.stats()
    assert after["shadow_failed"] - before["shadow_failed"] == 1
    assert after["shadow_completed"] - before["shadow_completed"] == 2


def test_the_queue_empties_and_nothing_is_retained_after_a_burst() -> None:
    from app.observability.process import rss_mb

    before = background.stats()
    payload = []
    for _ in range(40):
        background.try_submit(lambda: payload.append(None))
        if background.stats()["shadow_pending"] >= background.SHADOW_MAX_PENDING:
            background.drain(5)
    assert background.drain(10)
    stats = background.stats()
    assert stats["shadow_pending"] == 0 and stats["pending"] == 0
    assert stats["shadow_submitted"] - before["shadow_submitted"] == stats["shadow_completed"] - before["shadow_completed"]
    assert background._shadow_pending == set() and background._pending == set()      # noqa: SLF001
    assert rss_mb() is None or rss_mb() > 0


def test_health_reports_queue_depth_drops_failures_and_memory() -> None:
    from fastapi.testclient import TestClient

    from app.api.server import app

    bg = TestClient(app).get("/health").json()["dependencies"]["background"]
    assert {"pending", "shadow_pending", "shadow_dropped", "shadow_failed", "shadow_high_water", "inline_overflow",
            "usage_records_held", "usage_records_evicted", "rss_mb"} <= set(bg)
