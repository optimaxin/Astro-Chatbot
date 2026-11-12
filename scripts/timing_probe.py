"""Deterministic check of the hedge (6 s) and timeout (25 s) behaviour.

    python -m scripts.timing_probe

Runs the real gateway (llm_factory → hedging → OpenAI client → HTTP) against
a local OpenAI-compatible stub whose per-model response delay is controlled,
so the timings are the gateway's own and not the network's. Uses the
production defaults from settings (LLM_HEDGE_AFTER_SECONDS, NVIDIA timeout);
only the base URL and key point at the stub. Nothing is metered.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm import llm_factory, reliability, usage_tracker  # noqa: E402

DELAYS: dict[str, float] = {}
RESULTS: list[bool] = []


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body["model"]
        time.sleep(DELAYS.get(model, 0.2))
        payload = {"id": "x", "object": "chat.completion", "created": 0, "model": model,
                   "choices": [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant", "content": f"answer from {model}"}}],
                   "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}}
        data = json.dumps(payload).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass  # the client gave up (timeout) — expected in some scenarios


def check(ok: bool, name: str, detail: str) -> None:
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def scenario(name, main_delay, fallback_delay, hedge_after, expect_winner, lo, hi):
    base = llm_factory.settings
    n = base.nvidia
    DELAYS.clear()
    DELAYS.update({n.main: main_delay, n.fallback: fallback_delay})
    llm_factory.settings = dataclasses.replace(
        base, reliability=dataclasses.replace(base.reliability, hedge_after_seconds=hedge_after))
    reliability.breaker.reset()
    started = time.perf_counter()
    try:
        result = llm_factory.get_llm(llm_factory.ROLE_MAIN).complete([{"role": "user", "content": "hi"}])
        winner, err = result.usage.model, result.usage.error_code
    except Exception as exc:  # noqa: BLE001
        winner, err = None, type(exc).__name__
    secs = time.perf_counter() - started
    llm_factory.settings = base
    ok = (winner == expect_winner) and lo <= secs <= hi
    check(ok, name, f"{secs:.1f}s, winner={winner and winner.split('/')[-1]}, primary_error={err}")


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = llm_factory.settings
    usage_tracker.persist = False
    llm_factory.settings = dataclasses.replace(base, nvidia=dataclasses.replace(
        base.nvidia, api_key="stub", base_url=f"http://127.0.0.1:{server.server_port}/v1"))
    llm_factory._client = None
    s = llm_factory.settings
    hedge, timeout = s.reliability.hedge_after_seconds, s.nvidia.timeout_seconds
    main_m, fb = s.nvidia.main, s.nvidia.fallback
    print(f"settings under test: hedge_after={hedge}s nvidia_timeout={timeout}s "
          f"main={main_m.split('/')[-1]} fallback={fb.split('/')[-1]}\n")
    try:
        # Warm-up: the first call in a process pays ~2 s of client start-up
        # (measured); production workers pay it once, not per request.
        scenario("warm-up call (cold start, not judged)", 0.2, 0.2, hedge, main_m, 0, 30)
        RESULTS.pop()
        scenario("fast primary: no hedge, primary answers", 1.0, 0.2, hedge, main_m, 0.9, 2.5)
        scenario(f"slow primary: hedge fires at {hedge:g}s and the fallback wins", 60, 0.5, hedge, fb,
                 hedge + 0.4, hedge + 2.5)
        scenario(f"primary just over the hedge: whichever finishes first wins", hedge + 0.5, 3.0, hedge, main_m,
                 hedge + 0.4, hedge + 2.0)
        scenario(f"hedging off: primary times out at {timeout:g}s, then fallback answers", 60, 0.5, 0, fb,
                 timeout - 0.5, timeout + 3.0)
        # Worst case: the hedge starts the fallback at `hedge`, and the
        # fallback then runs into its own timeout -> hedge + timeout.
        scenario(f"both stuck: error by hedge + timeout ({hedge + timeout:g}s), never hangs",
                 60, 60, hedge, None, hedge + timeout - 0.5, hedge + timeout + 2.0)
    finally:
        llm_factory.settings = base
        llm_factory._client = None
        server.shutdown()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
