"""Live reliability and isolation probe (real NVIDIA, Redis, Pinecone).

    python -m scripts.reliability_probe

Part A drives a real API process over HTTP (AUTH_MODE=optional, a low rate
limit, an identity backend that is deliberately unreachable):
  session hijacking, forged session tokens, cross-visitor history isolation,
  per-identity rate limiting, fail-closed auth, stored-chart authorisation.
Part B calls the gateway in-process against the real endpoint:
  a real provider failure (non-existent model -> 404) and its fallback, the
  circuit opening after repeated failures, and hedging under real latency.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.usage_tracker import set_traffic_source  # noqa: E402

set_traffic_source("security")

import httpx  # noqa: E402

PORT = 8766
BASE = f"http://127.0.0.1:{PORT}"
BIRTH = {"date": "1995-08-14", "time": "10:30", "timezoneOffsetMinutes": 330,
         "latitude": 28.6139, "longitude": 77.209}
results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def chat(question: str, session_id: str, token: str | None = None, bearer: str | None = None,
         birth: dict | None = BIRTH) -> httpx.Response:
    headers = {}
    if token:
        headers["X-Chat-Session-Token"] = token
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    body = {"user_id": "attacker-chosen-id", "session_id": session_id, "question": question}
    if birth:
        body["birth"] = birth
    return httpx.post(f"{BASE}/chat", json=body, headers=headers, timeout=90)


def part_a() -> None:
    env = {**os.environ, "AUTH_MODE": "optional", "RATE_LIMIT_PER_MINUTE": "4",
           "AUTH_BACKEND_URL": "http://127.0.0.1:9",   # nothing listens: identity provider down
           "AI_TRAFFIC_SOURCE": "security", "PYTHONIOENCODING": "utf-8"}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api.server:app", "--port", str(PORT)],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(1)
        session = f"kundli-probe-{int(time.time())}"

        r1 = chat("Remember this for later: my secret word is PEACOCK-7731.", session)
        token_a = r1.json().get("session_token")
        check(r1.status_code == 200 and bool(token_a), "anonymous visitor receives a signed session token")

        r2 = chat("What secret word did I tell you earlier in this conversation?", session, token=token_a)
        check("PEACOCK" in r2.text.upper(), "same visitor keeps their own conversation", r2.json().get("answer", "")[:80])

        r3 = chat("What secret word did I tell you earlier in this conversation?", session)
        check("PEACOCK" not in r3.text.upper(), "hijack: same session_id, different visitor -> no access",
              r3.json().get("answer", "")[:80])

        forged = token_a.split(".")[0] + ".00000000000000000000000000000000"
        r4 = chat("What secret word did I tell you earlier in this conversation?", session, token=forged)
        check("PEACOCK" not in r4.text.upper() and bool(r4.json().get("session_token")),
              "forged session token -> treated as a new visitor")

        r5 = chat("Career?", session, birth=None)
        check(r5.status_code == 401, "anonymous request without birth details cannot read a stored chart",
              f"HTTP {r5.status_code}")

        r6 = chat("Career?", session, bearer="any-token")
        check(r6.status_code == 503 and "any-token" not in r6.text,
              "bearer token with identity provider down -> fails closed (503)", f"HTTP {r6.status_code}")

        r7 = httpx.get(f"{BASE}/chart/some-user-id", timeout=10)
        check(r7.status_code == 401, "stored chart requires sign-in", f"HTTP {r7.status_code}")

        # The limiter is a fixed one-minute window, so start the burst at the
        # beginning of a window; a burst straddling a boundary can legitimately
        # be split across two windows.
        while time.time() % 60 > 3:
            time.sleep(0.5)
        burst_token = chat("hi", f"{session}-rl").json().get("session_token")
        codes = [chat("hi", f"{session}-rl", token=burst_token).status_code for _ in range(5)]
        check(429 in codes, "per-identity rate limit enforced (limit 4/min)", f"codes {codes}")
    finally:
        server.terminate()
        server.wait(10)


def part_b() -> None:
    from app.llm import llm_factory, reliability

    base = llm_factory.settings
    msg = [{"role": "user", "content": "In one sentence: what is a nakshatra?"}]

    # Real provider failure: a model id the endpoint does not have -> 404.
    broken = dataclasses.replace(base, nvidia=dataclasses.replace(base.nvidia, main="nvidia/does-not-exist-404"),
                                 reliability=dataclasses.replace(base.reliability, hedge_after_seconds=0))
    llm_factory.settings = broken
    try:
        started = time.perf_counter()
        result = llm_factory.get_llm(llm_factory.ROLE_MAIN).complete(msg)
        ms = int((time.perf_counter() - started) * 1000)
        check(result.usage.fallback_used and result.usage.model.endswith("bolt-e4b") and bool(result.text),
              "real 404 on the primary falls back to Bolt", f"{ms} ms, error_code={result.usage.error_code}")

        for _ in range(base.reliability.circuit_failure_threshold):
            time.sleep(3)   # pace: the free tier 429s rapid bursts (observed)
            try:
                llm_factory.get_llm(llm_factory.ROLE_MAIN).complete(msg)
            except Exception as exc:  # noqa: BLE001 — a 429 here is the provider limiting us
                print(f"   note: {type(exc).__name__}: {str(exc)[:90]}")
        state = reliability.breaker.state(("nvidia", "nvidia/does-not-exist-404"),
                                          base.reliability.circuit_cooldown_seconds)
        time.sleep(3)
        started = time.perf_counter()
        try:
            llm_factory.get_llm(llm_factory.ROLE_MAIN).complete(msg)
        except Exception as exc:  # noqa: BLE001
            print(f"   note: {type(exc).__name__}: {str(exc)[:90]}")
        skip_ms = int((time.perf_counter() - started) * 1000)
        check(state == "open", "circuit opens after repeated failures; primary skipped",
              f"state={state}, next call {skip_ms} ms")
    finally:
        llm_factory.settings = base
        reliability.breaker.reset()

    # Hedging under real network latency: hedge almost immediately so the
    # race is exercised for real; the answer must still arrive and be valid.
    hedgy = dataclasses.replace(base, reliability=dataclasses.replace(base.reliability, hedge_after_seconds=0.3))
    llm_factory.settings = hedgy
    try:
        started = time.perf_counter()
        result = llm_factory.get_llm(llm_factory.ROLE_MAIN).complete(msg)
        check(bool(result.text), "hedged request returns a valid answer (real endpoint)",
              f"{int((time.perf_counter() - started) * 1000)} ms, winner={result.usage.model.split('/')[-1]}")
    finally:
        llm_factory.settings = base


def main() -> int:
    part_a()
    part_b()
    from app.observability import background

    background.drain(30)
    failed = [r for r in results if not r[0]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
