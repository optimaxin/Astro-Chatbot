from __future__ import annotations

import json
import threading
from dataclasses import replace
from typing import Any

import httpx
import pytest

from app import auth, providers, storage
from app import main as main_module
from app.catalog import Pricing, calculate_cost, get_catalog, load_catalog
from app.evaluator import parse_judge_output
from app.providers import GenerationResult, ProviderError, birth_from_context, filter_params

from .conftest import CLIENT, OTHER


class FakeProvider:
    def __init__(self, text: str = "Hello there!", fail: ProviderError | None = None) -> None:
        self.text, self.fail, self.calls = text, fail, []
        self._lock = threading.Lock()

    def generate(self, model, messages, params, **request: Any) -> GenerationResult:
        with self._lock:
            self.calls.append({"model": model.id, "messages": messages, "params": params, **request})
        if self.fail:
            raise self.fail
        return GenerationResult(text=self.text, model=model.id, input_tokens=1000,
                                output_tokens=500, total_tokens=1500, provider_latency_ms=42,
                                finish_reason="stop")


@pytest.fixture
def fake(monkeypatch) -> FakeProvider:
    provider = FakeProvider()
    monkeypatch.setattr(main_module, "get_provider", lambda config: provider)
    return provider


def _chat(client, headers=CLIENT, **overrides: Any):
    body = {"provider": "gemini", "model": "gemini-2.5-flash", "message": "Tell me about my career.",
            **overrides}
    return client.post("/api/playground/chat", json=body, headers=headers)


# ── catalog and cost ─────────────────────────────────────────────────
def test_catalog_loads_every_provider_and_model() -> None:
    catalog = get_catalog()
    assert {"gemini", "openrouter", "nvidia", "aicore"} <= set(catalog.providers)
    assert catalog.model("gemini", "gemini-2.5-flash") is not None
    assert catalog.model("aicore", "chatbot-english").request_overrides == {"response_style": "english"}
    assert catalog.evaluator == ("nvidia", "nvidia/nemotron-3.5-bolt-e4b")
    assert next(iter(catalog.providers)) == "aicore"


def test_cost_uses_actual_tokens_and_configured_price() -> None:
    cost = calculate_cost(Pricing(0.30, 2.50), 1000, 500)
    assert cost["input_cost"] == pytest.approx(0.0003)
    assert cost["output_cost"] == pytest.approx(0.00125)
    assert cost["total_cost"] == pytest.approx(0.00155)


def test_cost_is_none_rather_than_invented() -> None:
    assert calculate_cost(None, 1000, 500)["total_cost"] is None
    assert calculate_cost(Pricing(0.3, 2.5), None, 500)["total_cost"] is None


def test_a_model_pointing_at_an_unknown_provider_is_rejected(tmp_path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": [], "models": [{"id": "x", "provider": "nope"}]}))
    with pytest.raises(ValueError):
        load_catalog(path)


def test_unsupported_params_are_not_sent() -> None:
    gemini = get_catalog().providers["gemini"]
    sent = filter_params(gemini, {"temperature": 0.5, "top_k": 40, "top_p": None})
    assert sent == {"temperature": 0.5}


# ── public access, isolation and validation ──────────────────────────
def test_tester_endpoints_are_public_without_a_token(client) -> None:
    assert client.get("/api/playground/models").status_code == 200
    assert client.get("/api/health").json() == {"status": "ok"}


def test_shared_token_mode_still_works(client, monkeypatch) -> None:
    monkeypatch.setattr(auth, "settings", replace(auth.settings, access_token="secret"))
    assert client.get("/api/playground/models").status_code == 401
    assert client.get("/api/playground/models", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_client_id_is_required_for_generation(client, fake) -> None:
    assert _chat(client, headers={}).status_code == 400
    assert _chat(client, headers={"X-Playground-Client": "bad id!"}).status_code == 400


def test_testers_cannot_see_each_others_experiments(client, fake) -> None:
    mine = _chat(client).json()["id"]
    assert client.get(f"/api/playground/experiments/{mine}", headers=OTHER).status_code == 404
    assert client.get("/api/playground/experiments", headers=OTHER).json() == []
    assert client.post("/api/playground/feedback", headers=OTHER,
                       json={"experiment_id": mine, "rating": "bad"}).status_code == 404


def test_models_endpoint_never_exposes_keys_or_urls(client) -> None:
    body = client.get("/api/playground/models").json()
    text = json.dumps(body)
    assert "fake-key-for-tests" not in text and "api_key" not in text and "ai-core.test" not in text
    gemini = next(p for p in body["providers"] if p["id"] == "gemini")
    assert gemini["configured"] is True


def test_unknown_model_is_a_400(client, fake) -> None:
    assert _chat(client, model="not-a-model").status_code == 400


def test_unconfigured_provider_is_a_400(client, fake) -> None:
    assert _chat(client, provider="nvidia", model="nvidia/nemotron-3.5-bolt-e4b").status_code == 400


def test_model_extra_body_is_sent(monkeypatch) -> None:
    from app.providers import OpenAICompatibleProvider

    captured: dict[str, Any] = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop")

    class Client:
        chat = type("C", (), {"completions": Completions()})()

    catalog = get_catalog()
    provider = OpenAICompatibleProvider(catalog.providers["nvidia"])
    monkeypatch.setattr(provider, "_client", lambda: Client())
    with pytest.raises(ProviderError):
        provider.generate(catalog.model("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"),
                          [{"role": "user", "content": "hi"}], {"max_tokens": 20})
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_blank_message_is_rejected(client, fake) -> None:
    assert _chat(client, message="   ").status_code == 422


def test_oversized_body_is_rejected(client, fake) -> None:
    res = client.post("/api/playground/chat", content=b"x" * 2_100_000,
                      headers={**CLIENT, "Content-Type": "application/json"})
    assert res.status_code == 413


def test_security_headers_are_set(client) -> None:
    res = client.get("/api/health")
    assert res.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in res.headers["Content-Security-Policy"]


# ── chat → one experiment ────────────────────────────────────────────
def test_chat_returns_response_tokens_cost_and_latency_in_one_record(client, fake) -> None:
    exp = _chat(client, params={"temperature": 0.4, "top_k": 20}, tester_name="Asha").json()

    assert exp["status"] == "success" and exp["response"] == "Hello there!"
    assert (exp["input_tokens"], exp["output_tokens"], exp["total_tokens"]) == (1000, 500, 1500)
    assert exp["total_cost"] == pytest.approx(0.00155)
    assert exp["pricing_snapshot"]["input_per_1m"] == 0.30
    assert exp["usage_source"] == "provider" and exp["tester_name"] == "Asha"
    assert exp["latency_ms"] >= 0 and exp["provider_latency_ms"] == 42
    assert exp["effective_request"]["params_ignored"] == ["top_k"]
    stored = client.get(f"/api/playground/experiments/{exp['id']}", headers=CLIENT).json()
    assert stored["response"] == exp["response"]


def test_retry_with_same_request_id_creates_no_duplicate(client, fake) -> None:
    first = _chat(client, request_id="req-12345678").json()
    second = _chat(client, request_id="req-12345678").json()
    assert first["id"] == second["id"] and len(fake.calls) == 1
    assert _chat(client, headers=OTHER, request_id="req-12345678").status_code == 409


def test_system_prompt_context_and_history_reach_the_model_in_order(client, fake) -> None:
    _chat(client, system_prompt="Be brief.", context="Lagna: Virgo",
          history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    messages = fake.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "Be brief." in messages[0]["content"] and "Lagna: Virgo" in messages[0]["content"]
    assert messages[-1]["content"] == "Tell me about my career."


def test_provider_failure_is_saved_and_shown_not_raised(client, monkeypatch) -> None:
    failing = FakeProvider(fail=ProviderError("Google Gemini", "Rate limit exceeded.", 429))
    monkeypatch.setattr(main_module, "get_provider", lambda config: failing)
    response = _chat(client)
    assert response.status_code == 200
    exp = response.json()
    assert exp["status"] == "failed" and exp["error"] == "Rate limit exceeded."
    assert exp["input_tokens"] is None and exp["total_cost"] is None
    assert storage.get_experiment(exp["id"])["status"] == "failed"


def test_history_lists_newest_first(client, fake) -> None:
    first = _chat(client).json()["id"]
    second = _chat(client, message="second").json()["id"]
    rows = client.get("/api/playground/experiments", headers=CLIENT).json()
    assert [r["id"] for r in rows[:2]] == [second, first]


def test_rate_limit_applies_to_generation(client, fake, monkeypatch) -> None:
    monkeypatch.setattr(auth, "settings", replace(auth.settings, rate_limit_per_minute=2))
    assert _chat(client).status_code == 200
    assert _chat(client).status_code == 200
    assert _chat(client).status_code == 429


def test_daily_spend_guard_blocks_generation(client, fake, monkeypatch) -> None:
    monkeypatch.setattr(auth, "settings", replace(auth.settings, daily_cost_limit_usd=0.002))
    assert _chat(client).status_code == 200  # spends 0.00155
    assert _chat(client).status_code == 200  # 0.0031 ≥ cap after this one
    assert _chat(client).status_code == 429


def test_daily_request_guard_blocks_generation(client, fake, monkeypatch) -> None:
    monkeypatch.setattr(auth, "settings", replace(auth.settings, daily_request_limit=1))
    assert _chat(client).status_code == 200
    assert _chat(client).status_code == 429


def test_ten_concurrent_testers_each_get_a_record(client, fake) -> None:
    results: list[int] = []

    def run(i: int) -> None:
        res = _chat(client, headers={"X-Playground-Client": f"client-concurrent-{i:02d}"},
                    request_id=f"concurrent-{i:04d}")
        results.append(res.status_code)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [200] * 10
    assert storage.usage_since(0)["requests"] == 10


# ── ai-core chatbot provider ─────────────────────────────────────────
def test_birth_details_are_read_from_json_context() -> None:
    birth = {"date": "1995-04-12", "time": "06:30", "timezoneOffsetMinutes": 330,
             "latitude": 28.6, "longitude": 77.2}
    assert birth_from_context(json.dumps(birth)) == birth
    assert birth_from_context(json.dumps({"birth": birth})) == birth
    assert birth_from_context("Lagna: Virgo") is None
    assert birth_from_context('{"date": "1995-04-12"}') is None


def test_aicore_provider_calls_chat_and_keeps_the_session_token(client, monkeypatch) -> None:
    sent: list[dict[str, Any]] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append({"url": url, "json": json, "headers": headers})
        return httpx.Response(200, json={
            "answer": "Aapka Saturn 10th house mein hai.", "session_token": "tok-1",
            "request_id": "r1", "intent": "career", "degraded": False, "citations": [],
            "usage": {"input_tokens": 900, "output_tokens": 120, "total_tokens": 1020, "model": "nv-lightning",
                      "tokens_available": True, "token_source": "provider", "turn_cost_usd": 0.0004}})

    monkeypatch.setattr(providers.httpx, "post", fake_post)
    providers._aicore_tokens.clear()
    body = {"provider": "aicore", "model": "chatbot-english", "session_id": "s-abc", "system_prompt": "Ignored.",
            "history": [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]}
    first = _chat(client, **body).json()
    # Only what ai-core actually receives is recorded.
    assert first["system_prompt"] is None
    assert [m["role"] for m in first["effective_request"]["messages"]] == ["user"]
    _chat(client, **body, message="And my marriage?")

    assert sent[0]["url"] == "http://ai-core.test/chat"
    assert sent[0]["json"]["response_style"] == "english" and sent[0]["json"]["session_id"] == "playground-s-abc"
    # No chart in Context: ai-core still gets one, so anonymous chat is allowed.
    assert sent[0]["json"]["birth"] == providers.DEFAULT_BIRTH
    assert first["metadata"]["ai_core"]["chart"].startswith("default sample chart")
    assert sent[0]["headers"] == {} and sent[1]["headers"] == {"X-Chat-Session-Token": "tok-1"}
    assert first["status"] == "success" and first["total_tokens"] == 1020
    assert first["total_cost"] == pytest.approx(0.0004) and first["currency"] == "USD"
    assert first["metadata"]["cost_source"] == "ai-core reported (whole turn)"
    assert first["metadata"]["ai_core"]["intent"] == "career"


def test_aicore_degraded_reply_is_recorded_as_failed(client, monkeypatch) -> None:
    monkeypatch.setattr(providers.httpx, "post", lambda *a, **k: httpx.Response(
        200, json={"answer": "Sorry, I can't answer right now.", "degraded": True, "usage": {}}))
    exp = _chat(client, provider="aicore", model="chatbot-hindi").json()
    assert exp["status"] == "failed" and "fallback" in exp["error"]
    assert exp["response"] == "Sorry, I can't answer right now."


def test_aicore_refusal_detail_reaches_the_tester(client, monkeypatch) -> None:
    monkeypatch.setattr(providers.httpx, "post", lambda *a, **k: httpx.Response(
        401, json={"error": "auth_required", "detail": "Please sign in."}))
    exp = _chat(client, provider="aicore", model="chatbot-english").json()
    assert exp["status"] == "failed" and exp["error"] == "ai-core refused the request (HTTP 401): Please sign in."


def test_aicore_unreachable_is_a_recorded_failure(client, monkeypatch) -> None:
    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(providers.httpx, "post", boom)
    exp = _chat(client, provider="aicore", model="chatbot-english").json()
    assert exp["status"] == "failed" and exp["error"] == "Could not connect to the provider."


# ── evaluation and feedback ──────────────────────────────────────────
def test_judge_output_is_parsed_and_clamped() -> None:
    raw = '```json\n{"scores": {"relevance": 9, "accuracy": 12, "human_likeness": 8, "tone": 7, ' \
          '"conciseness": 8, "context_awareness": 9, "emoji_usage": null}, "feedback": "Good.", ' \
          '"strengths": ["Natural"], "improvements": ["Shorter"]}\n```'
    verdict = parse_judge_output(raw)
    assert verdict["scores"]["accuracy"] == 10.0
    assert verdict["scores"]["emoji_usage"] is None
    assert verdict["overall"] == pytest.approx(8.5)
    assert verdict["strengths"] == ["Natural"]


def test_evaluate_attaches_verdict_to_the_same_experiment(client, fake, monkeypatch) -> None:
    exp = _chat(client).json()
    fake.text = ('{"scores": {"relevance": 9, "accuracy": 8, "human_likeness": 9, "tone": 8, '
                 '"conciseness": 9, "context_awareness": 8, "emoji_usage": null}, '
                 '"feedback": "Clear.", "strengths": [], "improvements": []}')
    monkeypatch.setattr("app.evaluator.get_provider", lambda config: fake)
    updated = client.post("/api/playground/evaluate", json={"experiment_id": exp["id"]}, headers=CLIENT).json()
    assert updated["id"] == exp["id"]
    assert updated["evaluation"]["label"] == "AI Evaluation"
    assert updated["evaluation_overall"] == pytest.approx(8.5)
    assert updated["total_tokens"] == 1500
    assert updated["evaluation"]["evaluator_usage"]["total_tokens"] == 1500


def test_failed_experiment_cannot_be_evaluated(client, monkeypatch) -> None:
    failing = FakeProvider(fail=ProviderError("Google Gemini", "boom"))
    monkeypatch.setattr(main_module, "get_provider", lambda config: failing)
    exp = _chat(client).json()
    assert client.post("/api/playground/evaluate", json={"experiment_id": exp["id"]},
                       headers=CLIENT).status_code == 400


def test_human_feedback_is_stored_and_the_original_response_kept(client, fake) -> None:
    exp = _chat(client).json()
    updated = client.post("/api/playground/feedback", headers=CLIENT, json={
        "experiment_id": exp["id"], "rating": "bad", "feedback": "Too long",
        "edited_response": "Shorter reply."}).json()
    assert (updated["human_rating"], updated["human_feedback"], updated["edited_response"]) == \
        ("bad", "Too long", "Shorter reply.")
    assert updated["response"] == "Hello there!"


# ── one service: frontend served alongside the API ───────────────────
def test_frontend_routes_serve_the_app_shell(client) -> None:
    for path in ("/", "/admin", "/admin/experiments"):
        res = client.get(path)
        assert res.status_code == 200 and "id=root" in res.text
    assert client.get("/assets/app.js").text == "console.log('app')"
    assert client.get("/api/does-not-exist").status_code == 404
    assert client.get("/..%2f..%2fetc/passwd").status_code in (200, 404)
    assert "root:" not in client.get("/..%2f..%2fetc/passwd").text
