"""/chat, /chart and /feedback identity: server-derived, never body-supplied."""

from __future__ import annotations

import dataclasses

import pytest

from app.api import auth
from app.config import settings as settings_module
from tests.test_chatbot_mvp import FakeRedis

BIRTH = {"date": "1995-08-14", "time": "10:30", "timezoneOffsetMinutes": 330,
         "latitude": 28.6139, "longitude": 77.209}
TOKENS = {"tok-alice": "alice-id", "tok-bob": "bob-id"}


@pytest.fixture
def env(monkeypatch, kundli):
    """Auth in a given mode, a fake backend verifier, fake Redis, and a
    recording run_turn so tests see exactly what identity a turn ran as."""
    from app.api import server
    from app.memory.redis_memory import RedisMemory, memory

    calls = {"verify": 0, "turns": []}

    def fake_verify(token):
        calls["verify"] += 1
        if token == "tok-down":
            raise auth.AuthUnavailable("backend down")
        return TOKENS.get(token)

    monkeypatch.setattr(auth, "_verify_with_backend", fake_verify)
    auth.clear_cache()
    monkeypatch.setattr(memory, "_client", FakeRedis())
    monkeypatch.setattr(RedisMemory, "configured", property(lambda self: True))
    monkeypatch.setattr(server, "kundli_from_birth", lambda birth, key: kundli)

    def fake_run_turn(question, session_id, user_id, persona_id, **kwargs):
        calls["turns"].append({"session": session_id, "user": user_id,
                               "chart_preset": kwargs.get("kundli") is not None})
        return {"response": "ok", "metrics": {"request_id": f"req-{len(calls['turns'])}",
                                              "generation_failed": False, "errors": []}}

    monkeypatch.setattr(server, "run_turn", fake_run_turn)

    def set_mode(mode):
        s = settings_module.settings
        object.__setattr__(s, "auth", dataclasses.replace(s.auth, mode=mode))

    testclient = pytest.importorskip("fastapi.testclient")
    return testclient.TestClient(server.app), calls, set_mode


def _chat(client, body=None, token=None, session_token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if session_token:
        headers["X-Chat-Session-Token"] = session_token
    payload = {"user_id": "someone-else", "session_id": "kundli-abc", "question": "Career?"}
    payload.update(body or {})
    return client.post("/chat", json=payload, headers=headers)


# ── identity ──────────────────────────────────────────────────────────
def test_signed_in_identity_comes_from_the_token_not_the_body(env) -> None:
    client, calls, set_mode = env
    set_mode("optional")
    assert _chat(client, token="tok-alice").status_code == 200
    turn = calls["turns"][-1]
    assert turn["user"] == "alice-id", "body user_id 'someone-else' was ignored"
    assert turn["session"] == "user:alice-id|kundli-abc"


def test_anonymous_without_birth_cannot_read_a_stored_chart(env) -> None:
    client, _, set_mode = env
    set_mode("optional")
    r = _chat(client)                    # body says user_id=someone-else
    assert r.status_code == 401 and r.json()["error"] == "auth_required"


def test_anonymous_visitor_gets_a_signed_session_and_keeps_it(env) -> None:
    client, calls, set_mode = env
    set_mode("optional")
    first = _chat(client, {"birth": BIRTH}).json()
    token = first["session_token"]
    assert token and calls["turns"][-1]["chart_preset"]
    second = _chat(client, {"birth": BIRTH}, session_token=token).json()
    assert second["session_token"] is None, "no new identity when the token is valid"
    assert calls["turns"][0]["session"] == calls["turns"][1]["session"]


def test_forged_anonymous_token_gets_a_fresh_identity(env) -> None:
    client, calls, set_mode = env
    set_mode("optional")
    real = _chat(client, {"birth": BIRTH}).json()["session_token"]
    anon_id = real.split(".")[0]
    _chat(client, {"birth": BIRTH}, session_token=f"{anon_id}.forgedsignature")
    assert calls["turns"][0]["session"] != calls["turns"][1]["session"]


def test_required_mode_rejects_anonymous_and_expired_tokens(env) -> None:
    client, _, set_mode = env
    set_mode("required")
    assert _chat(client, {"birth": BIRTH}).status_code == 401
    assert _chat(client, token="tok-expired").status_code == 401
    assert _chat(client, token="tok-bob").status_code == 200


def test_unverifiable_token_fails_closed(env) -> None:
    client, _, set_mode = env
    set_mode("optional")
    r = _chat(client, token="tok-down")
    assert r.status_code == 503 and "tok-down" not in r.text


def test_token_verification_is_cached(env) -> None:
    client, calls, set_mode = env
    set_mode("optional")
    for _ in range(3):
        _chat(client, token="tok-alice")
    assert calls["verify"] == 1


def test_off_mode_restores_the_legacy_contract(env) -> None:
    client, calls, set_mode = env
    set_mode("off")
    assert _chat(client).status_code == 200
    assert calls["turns"][-1] == {"session": "kundli-abc", "user": "someone-else", "chart_preset": False}


# ── session hijacking / cross-user isolation ──────────────────────────
def test_reusing_another_users_session_id_reaches_only_your_own_history(env) -> None:
    """The widget's session ids derive from birth details, so they are
    guessable. Bob sending Alice's session id must not land in her history."""
    client, calls, set_mode = env
    set_mode("optional")
    _chat(client, {"session_id": "kundli-alice"}, token="tok-alice")
    _chat(client, {"session_id": "kundli-alice"}, token="tok-bob")
    alice, bob = calls["turns"][0]["session"], calls["turns"][1]["session"]
    assert alice != bob and alice.startswith("user:alice-id|") and bob.startswith("user:bob-id|")


def test_same_birth_details_do_not_share_a_conversation_between_visitors(env) -> None:
    client, calls, set_mode = env
    set_mode("optional")
    _chat(client, {"birth": BIRTH})
    _chat(client, {"birth": BIRTH})           # a different browser, no token
    assert calls["turns"][0]["session"] != calls["turns"][1]["session"]


# ── chart access ──────────────────────────────────────────────────────
def test_stored_chart_is_readable_only_by_its_owner(env, monkeypatch, kundli) -> None:
    from app.api import server

    client, _, set_mode = env
    set_mode("optional")
    monkeypatch.setattr(server, "get_kundli", lambda user_id: kundli)
    assert client.get("/chart/alice-id").status_code == 401
    assert client.get("/chart/alice-id", headers={"Authorization": "Bearer tok-bob"}).status_code == 403
    assert client.get("/chart/alice-id", headers={"Authorization": "Bearer tok-alice"}).status_code == 200


# ── feedback ownership ────────────────────────────────────────────────
def test_only_the_recipient_can_rate_an_answer(env, monkeypatch) -> None:
    from app.evaluation import evaluation_store

    client, _, set_mode = env
    set_mode("optional")
    monkeypatch.setattr(evaluation_store, "write_feedback", lambda **kw: True)
    request_id = _chat(client, token="tok-alice").json()["request_id"]
    vote = {"request_id": request_id, "user_id": "x", "feedback_type": "helpful"}
    assert client.post("/feedback", json=vote, headers={"Authorization": "Bearer tok-bob"}).status_code == 403
    assert client.post("/feedback", json=vote, headers={"Authorization": "Bearer tok-alice"}).status_code == 200
    assert client.post("/feedback", json=vote).status_code == 401


# ── rate limits ───────────────────────────────────────────────────────
def test_rotating_body_user_id_does_not_dodge_the_per_user_limit(env) -> None:
    from app.api import server

    client, _, set_mode = env
    set_mode("optional")
    s = settings_module.settings
    original = s.api
    object.__setattr__(s, "api", dataclasses.replace(original, rate_limit_enabled=True,
                                                     rate_limit_per_minute=2))
    try:
        codes = [_chat(client, {"user_id": f"rotating-{i}"}, token="tok-alice").status_code
                 for i in range(3)]
    finally:
        object.__setattr__(s, "api", original)
    assert codes == [200, 200, 429]
    assert server  # imported for its side effect of registering routes


# ── shared signing key across workers ─────────────────────────────────
def test_startup_refuses_without_a_shared_session_secret() -> None:
    for secret in ("", "too-short"):
        object.__setattr__(settings_module.settings, "auth", dataclasses.replace(
            settings_module.settings.auth, mode="optional", session_secret=secret, require_session_secret=True))
        with pytest.raises(RuntimeError, match="CHAT_SESSION_SECRET"):
            auth.check_session_secret()


def test_startup_allows_an_explicit_single_process_opt_out() -> None:
    object.__setattr__(settings_module.settings, "auth", dataclasses.replace(
        settings_module.settings.auth, mode="optional", session_secret="", require_session_secret=False))
    auth.check_session_secret()                      # warns, does not raise
    assert auth.secret_fingerprint() == "ephemeral"


def test_fingerprint_identifies_the_key_without_revealing_it() -> None:
    key = "k" * 40
    object.__setattr__(settings_module.settings, "auth", dataclasses.replace(
        settings_module.settings.auth, session_secret=key))
    fp = auth.secret_fingerprint()
    assert len(fp) == 12 and key not in fp and fp == auth.secret_fingerprint()
    auth.check_session_secret()


_WORKER = ("import sys; from app.api import auth; "
           "print(auth.issue_anon_token()[1] if sys.argv[1] == 'issue' else auth.parse_anon_token(sys.argv[2]))")


def _worker(env_secret, *args):
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = {**os.environ, "CHAT_SESSION_SECRET": env_secret}
    out = subprocess.run([sys.executable, "-c", _WORKER, *args], env=env, capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1], timeout=120, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_token_from_one_worker_process_is_accepted_by_another() -> None:
    """Two separate interpreter processes stand in for two uvicorn workers."""
    shared = "s" * 48
    token = _worker(shared, "issue")
    anon_id = token.rpartition(".")[0]
    assert _worker(shared, "parse", token) == anon_id
    # A worker with a different key rejects it: no cross-deployment reuse.
    assert _worker("d" * 48, "parse", token) == "None"
    # And without a configured key every process has its own: this is the
    # failure mode the startup check exists to prevent.
    assert _worker("", "parse", _worker("", "issue")) == "None"
