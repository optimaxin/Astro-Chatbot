"""Real-endpoint smoke test and capability probe for the NVIDIA models.

    python -m scripts.smoke_nvidia                 # the 3 configured models
    python -m scripts.smoke_nvidia --models a,b

Talks to the live endpoint directly (not through the gateway's adaptive
fallbacks), so every capability is observed rather than assumed. For each
model it checks: plain completion, reported usage, streaming and stream
usage, JSON mode, tool calling, image input, and whether hidden reasoning
leaks into the visible reply. Paste the printed verdicts into
`app/llm/model_registry.py`.

Never part of CI: it needs NVIDIA_API_KEY and spends free-tier quota.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.llm.providers import get_provider  # noqa: E402
from app.llm.reliability import classify_error  # noqa: E402

TOOL = {"type": "function", "function": {
    "name": "get_panchang", "description": "Panchang for a date and city",
    "parameters": {"type": "object", "properties": {
        "date": {"type": "string", "description": "YYYY-MM-DD"}, "city": {"type": "string"}},
        "required": ["date", "city"]}}}
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def _image_url() -> str:
    from benchmarks.images import to_data_url

    return to_data_url({"kind": "text_card", "lines": ["Saturn is in house 10"]})


def probe(client: Any, model: str) -> dict[str, Any]:
    out: dict[str, Any] = {"model": model}

    def attempt(name: str, fn):
        started = time.perf_counter()
        try:
            value = fn()
            out[name] = {"ok": True, "ms": int((time.perf_counter() - started) * 1000), **value}
        except Exception as exc:  # noqa: BLE001 — recording, not handling
            info = classify_error(exc)
            out[name] = {"ok": False, "error": info.label(), "detail": info.message[:200]}

    base = {"model": model, "max_tokens": 300, "temperature": 0.2}
    ask = [{"role": "user", "content": "In one sentence, what is a nakshatra?"}]

    def basic(extra: dict | None = None):
        r = client.chat.completions.create(messages=ask, **base, **({"extra_body": extra} if extra else {}))
        msg = r.choices[0].message
        text = msg.content or ""
        return {"text": text[:120], "usage": r.usage.model_dump() if r.usage else None,
                "finish": r.choices[0].finish_reason,
                "think_leak": "<think>" in text.lower(),
                "reasoning_field": bool(getattr(msg, "reasoning_content", None))}

    attempt("basic", basic)
    attempt("no_think_kwargs", lambda: basic(NO_THINK))

    def streaming():
        chunks, usage = [], None
        s = client.chat.completions.create(messages=ask, stream=True,
                                           stream_options={"include_usage": True}, **base)
        for c in s:
            if c.usage:
                usage = c.usage.model_dump()
            for ch in c.choices or []:
                if ch.delta and ch.delta.content:
                    chunks.append(ch.delta.content)
        return {"chunks": len(chunks), "stream_usage": usage is not None}

    attempt("streaming", streaming)

    def json_mode():
        r = client.chat.completions.create(
            messages=[{"role": "user", "content": 'Return JSON {"intent": "career", "confidence": 0.9}'}],
            response_format={"type": "json_object"}, **base)
        json.loads(r.choices[0].message.content)
        return {}

    attempt("json_mode", json_mode)

    def tools():
        r = client.chat.completions.create(
            messages=[{"role": "user", "content": "Panchang for Varanasi on 2026-03-14?"}],
            tools=[TOOL], tool_choice="auto", **base)
        calls = r.choices[0].message.tool_calls or []
        return {"called": [c.function.name for c in calls],
                "args": [c.function.arguments for c in calls]}

    attempt("tools", tools)

    def vision():
        r = client.chat.completions.create(messages=[{"role": "user", "content": [
            {"type": "text", "text": "Which house is Saturn in, per the image? Reply with the number."},
            {"type": "image_url", "image_url": {"url": _image_url()}}]}], **base)
        text = r.choices[0].message.content or ""
        return {"text": text[:80], "correct": "10" in text}

    attempt("vision", vision)
    return out


def main() -> int:
    n = settings.nvidia
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join([n.main, n.fallback, n.benchmark]))
    parser.add_argument("--json", default="", help="also write raw results here")
    args = parser.parse_args()
    if not n.configured:
        print("NVIDIA_API_KEY is not set in ai-core/.env")
        return 1

    client = get_provider("nvidia").build_client(settings)
    results = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n== {model}")
        r = probe(client, model)
        results.append(r)
        for key, value in r.items():
            if key != "model":
                verdict = "PASS" if value.get("ok") else "FAIL"
                print(f"  {key:<16} {verdict:<5} {json.dumps({k: v for k, v in value.items() if k != 'ok'}, ensure_ascii=False)[:220]}")
        usage_ok = bool(r["basic"].get("ok") and r["basic"].get("usage"))
        print(f"  -> registry: json_mode={r['json_mode']['ok']} tools={r['tools'].get('ok') and bool(r['tools'].get('called'))} "
              f"vision={r['vision'].get('ok') and r['vision'].get('correct')} stream_usage={r['streaming'].get('stream_usage')} "
              f"usage_reported={usage_ok} accepts_no_think={r['no_think_kwargs']['ok']}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(r["basic"].get("ok") for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
