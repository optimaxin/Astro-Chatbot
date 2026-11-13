"""Time to first visible token (TTFT) and total time for the main answer, streamed.

    python -m scripts.ttft_probe --label lightning-off [--rounds 2]

Model configuration comes from the environment (NVIDIA_MODEL,
NVIDIA_MAIN_REASONING_MODELS, ...), so run once per configuration. Streams
the main-turn role on the fixed profiling questions (tests/test_questions.py)
with the same system preamble for every configuration. Hedging is not used on
streams (fallback only before the first token). Nothing is metered.

Note: /chat returns a finished answer (no streaming to the browser today), so
for users TTFT == total turn time; this measures what streaming would give.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.usage_tracker import set_traffic_source, usage_tracker  # noqa: E402

usage_tracker.persist = False
set_traffic_source("debug")

from app.config.settings import settings  # noqa: E402
from app.llm import llm_factory  # noqa: E402
from tests.test_questions import QUESTIONS  # noqa: E402

SYSTEM = ("You are a warm, concise Vedic astrology assistant. Reply in the user's language and script. "
          "Chart: Lagna Virgo; Moon Pisces (7th); Sun and Venus Cancer (11th); Mars Virgo (1st); "
          "Mercury Leo (12th); Jupiter Scorpio (3rd); Saturn Aquarius (6th, R); Rahu Libra (2nd); "
          "Ketu Aries (8th); Mahadasha Venus.")


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    llm = llm_factory.get_llm(llm_factory.ROLE_MAIN)
    rows = []
    for _ in range(args.rounds):
        for q in QUESTIONS:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q.question}]
            start, first, parts, error = time.perf_counter(), None, [], None
            try:
                for part in llm.stream(msgs):
                    if first is None and part.strip():
                        first = time.perf_counter() - start
                    parts.append(part)
            except Exception as exc:  # noqa: BLE001
                error = type(exc).__name__
            total = time.perf_counter() - start
            rows.append({"id": q.id, "ttft_s": first, "total_s": round(total, 2), "error": error,
                         "chars": len("".join(parts))})
            print(f"  {q.id:<24} ttft={first if first is None else round(first, 2)}s total={total:.2f}s "
                  f"{error or ''}", flush=True)
    ok = [r for r in rows if r["ttft_s"] is not None]
    ttft, total = [r["ttft_s"] for r in ok], [r["total_s"] for r in ok]
    summary = {"label": args.label, "model": settings.nvidia.main,
               "reasoning_models": list(settings.nvidia.main_reasoning_models),
               "n": len(rows), "errors": len(rows) - len(ok),
               "ttft_p50": pct(ttft, .5) if ttft else None, "ttft_p95": pct(ttft, .95) if ttft else None,
               "total_p50": pct(total, .5) if total else None, "total_p95": pct(total, .95) if total else None,
               "total_mean": round(statistics.fmean(total), 2) if total else None}
    out = settings.base_dir / "logs" / "ttft" / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
