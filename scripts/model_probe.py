"""Live availability and capability probe for candidate models (NVIDIA NIM).

    python -m scripts.model_probe MODEL_ID [MODEL_ID ...] [--out logs/probe.json]

For each exact model id: is it in the catalog, does the configured key reach
it, what id does the provider report serving, latency, token usage, whether it
reasons by default (a `reasoning_content` field, or a budget exhausted before
any visible text), whether JSON mode and streaming (with a usage block) work,
and the time to first token. An unavailable model is reported as such: this
script never substitutes another id. The API key is read from settings and is
never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config.settings import settings  # noqa: E402

N = settings.nvidia
URL = f"{N.base_url}/chat/completions"
HEADERS = {"Authorization": f"Bearer {N.api_key}", "Content-Type": "application/json"}
PROMPT = [{"role": "user", "content": "Reply with one short sentence: what is a Lagna in Vedic astrology?"}]
JSON_PROMPT = [{"role": "system", "content": "Return only JSON."},
               {"role": "user", "content": 'Return {"ok": true, "word": "lagna"} exactly.'}]
THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def catalog() -> dict[str, dict]:
    r = httpx.get(f"{N.base_url}/models", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return {m["id"]: m for m in r.json().get("data", [])}


def chat(model: str, messages: list[dict], *, extra: dict | None = None, json_mode: bool = False,
         max_tokens: int = 200, timeout: float = 60.0) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, **(extra or {})}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    t0 = time.perf_counter()
    out: dict = {}
    try:
        r = httpx.post(URL, headers=HEADERS, json=body, timeout=timeout)
        out["http"] = r.status_code
        if r.status_code == 200:
            d = r.json()
            msg, usage = d["choices"][0]["message"], d.get("usage") or {}
            out.update(served=d.get("model"), finish=d["choices"][0].get("finish_reason"),
                       text=(msg.get("content") or "")[:200], in_tokens=usage.get("prompt_tokens"),
                       out_tokens=usage.get("completion_tokens"),
                       reasoning=bool(msg.get("reasoning_content") or msg.get("reasoning")))
        else:
            out["error"] = r.text[:200]
    except httpx.TimeoutException:
        out["error"] = f"timeout>{timeout}s"
    except httpx.HTTPError as exc:                      # disconnects, resets: an availability fact, not a crash
        out["error"] = f"transport:{type(exc).__name__}"
    out["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return out


def stream_ttft(model: str, extra: dict | None = None, timeout: float = 60.0) -> dict:
    body = {"model": model, "messages": PROMPT, "max_tokens": 80, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}, **(extra or {})}
    t0, first, usage = time.perf_counter(), None, None
    try:
        with httpx.stream("POST", URL, headers=HEADERS, json=body, timeout=timeout) as r:
            if r.status_code != 200:
                return {"http": r.status_code, "error": r.read().decode()[:160]}
            for line in r.iter_lines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    d = json.loads(line[6:])
                    usage = d.get("usage") or usage
                    if first is None and any((c.get("delta") or {}).get("content") for c in d.get("choices") or []):
                        first = time.perf_counter()
    except httpx.TimeoutException:
        return {"error": f"timeout>{timeout}s"}
    except httpx.HTTPError as exc:
        return {"error": f"transport:{type(exc).__name__}"}
    return {"http": 200, "ttft_ms": int((first - t0) * 1000) if first else None, "stream_usage": bool(usage)}


def probe(model: str, cat: dict[str, dict]) -> dict:
    res: dict = {"model": model, "in_catalog": model in cat, "catalog_meta": {
        k: v for k, v in (cat.get(model) or {}).items() if k not in ("id", "object", "created", "owned_by")}}
    first = chat(model, PROMPT, max_tokens=60, timeout=30)
    res["reachable"] = first.get("http") == 200
    res["first_call"] = first
    if not res["reachable"]:
        return res
    res["served_matches_requested"] = first.get("served") == model
    # Reasoning default: a tight budget that a reasoning model spends before any visible text.
    plain = chat(model, PROMPT, max_tokens=120)
    off = chat(model, PROMPT, extra=THINK_OFF, max_tokens=120)
    res["plain"], res["thinking_off_switch"] = plain, off
    res["reasons_by_default"] = bool(plain.get("reasoning") or (plain.get("finish") == "length" and not plain.get("text")))
    res["switch_accepted"] = off.get("http") == 200
    js = chat(model, JSON_PROMPT, extra=THINK_OFF if res["switch_accepted"] else None, json_mode=True, max_tokens=100)
    res["json_mode"] = {"http": js.get("http"), "valid_json": _is_json(js.get("text", "")), "error": js.get("error")}
    res["stream"] = stream_ttft(model, THINK_OFF if res["switch_accepted"] else None)
    return res


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default="logs/model_probe.json")
    args = ap.parse_args()
    cat = catalog()
    print(f"catalog: {len(cat)} models")
    results = []
    for m in args.models:
        r = probe(m, cat)
        results.append(r)
        s = r.get("stream") or {}
        print(f"{m:46} catalog={r['in_catalog']!s:5} reachable={r['reachable']!s:5} "
              f"served_ok={r.get('served_matches_requested')} reasons_default={r.get('reasons_by_default')} "
              f"switch_ok={r.get('switch_accepted')} json={(r.get('json_mode') or {}).get('valid_json')} "
              f"ttft={s.get('ttft_ms')}ms stream_usage={s.get('stream_usage')} "
              f"err={str((r.get('first_call') or {}).get('error', ''))[:50]}", flush=True)
        time.sleep(2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
