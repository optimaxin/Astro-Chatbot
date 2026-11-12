"""Live check that anonymous sessions survive across API instances.

    python -m scripts.multiworker_probe

Starts TWO separate API processes (ports 8771 and 8772) with the same
CHAT_SESSION_SECRET (taken from the environment / .env), as two uvicorn
workers or two instances behind a load balancer would run. A visitor whose
token was issued by instance A continues the same conversation on instance
B, both report the same /health secret fingerprint, and a third instance with
a different key rejects the token (new visitor, no history). Real NVIDIA,
Redis and Pinecone; traffic is tagged 'security'.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.config.settings import settings  # noqa: E402

BIRTH = {"date": "1995-08-14", "time": "10:30", "timezoneOffsetMinutes": 330,
         "latitude": 28.6139, "longitude": 77.209}
results: list[bool] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def start(port: int, secret: str) -> subprocess.Popen:
    env = {**os.environ, "AUTH_MODE": "optional", "CHAT_SESSION_SECRET": secret,
           "AI_TRAFFIC_SOURCE": "security", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api.server:app", "--port", str(port)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(90):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(1)
    raise RuntimeError(f"instance on {port} did not start")


def chat(port: int, question: str, session: str, token: str | None = None) -> httpx.Response:
    headers = {"X-Chat-Session-Token": token} if token else {}
    return httpx.post(f"http://127.0.0.1:{port}/chat", headers=headers, timeout=90,
                      json={"session_id": session, "question": question, "birth": BIRTH})


def main() -> int:
    shared = settings.auth.session_secret
    if len(shared) < 32:
        print("Set CHAT_SESSION_SECRET (>=32 chars) in ai-core/.env first.")
        return 1
    procs = [start(8771, shared), start(8772, shared), start(8773, secrets.token_urlsafe(40))]
    try:
        fps = [httpx.get(f"http://127.0.0.1:{p}/health", timeout=10).json()["dependencies"]["auth"]["session_secret"]
               for p in (8771, 8772, 8773)]
        check(fps[0] == fps[1] != "ephemeral", "instances A and B report the same key fingerprint", f"{fps[0]} / {fps[1]}")
        check(fps[2] != fps[0], "instance C (different key) reports a different fingerprint", fps[2])

        session = f"kundli-mw-{int(time.time())}"
        r1 = chat(8771, "Remember this for later: my secret word is HERON-5520.", session)
        token = r1.json().get("session_token")
        check(r1.status_code == 200 and bool(token), "instance A issues a session token")

        r2 = chat(8772, "What secret word did I tell you earlier in this conversation?", session, token)
        body = r2.json()
        check("HERON" in r2.text.upper() and not body.get("session_token"),
              "instance B accepts A's token and continues the conversation", body.get("answer", "")[:80])

        r3 = chat(8773, "What secret word did I tell you earlier in this conversation?", session, token)
        check("HERON" not in r3.text.upper() and bool(r3.json().get("session_token")),
              "instance C (other key) rejects the token: new visitor, no history", r3.json().get("answer", "")[:80])
    finally:
        for p in procs:
            p.terminate()
            p.wait(10)
    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
