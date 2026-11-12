"""Deterministic scoring for benchmark cases, plus the judge prompt.

Every case is scored by rules first (substring / regex / tool-call / language
/ citation / abstention checks). The optional LLM judge adds a 1-5 quality
grade on top; it never overrides a rule result.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

ABSTAIN = ("not mentioned", "does not contain", "doesn't contain", "do not contain",
           "not provided", "no information", "not specified", "not available in",
           "don't have information", "do not have information", "cannot find",
           "can't find", "not in the provided", "sources do not", "sources don't",
           "isn't mentioned", "is not mentioned", "not stated", "doesn't say", "does not say")

HINGLISH_MARKERS = ("hai", "hain", "aap", "aapki", "aapka", "aapke", "mein", "hoga", "hogi",
                    "kya", "nahi", "karein", "kijiye", "ke liye", "lekin", "aur", "bhi", "ki ")

_SCRIPTS = {
    "hindi_devanagari": (0x0900, 0x097F), "marathi": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF), "gujarati": (0x0A80, 0x0AFF),
    "tamil": (0x0B80, 0x0BFF), "telugu": (0x0C00, 0x0C7F),
}


def _script_share(text: str, lo: int, hi: int) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(lo <= ord(ch) <= hi for ch in letters) / len(letters)


def language_ok(text: str, language: str) -> bool:
    if language in _SCRIPTS:
        return _script_share(text, *_SCRIPTS[language]) >= 0.5
    latin = _script_share(text, 0x0041, 0x024F)
    lowered = f" {text.lower()} "
    markers = sum(1 for m in HINGLISH_MARKERS if f" {m.strip()} " in lowered)
    if language == "hinglish":
        return latin >= 0.7 and markers >= 2
    if language == "english":
        return latin >= 0.9 and markers <= 2
    return True


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def tool_ok(expected: dict | None, calls: list[dict[str, Any]]) -> tuple[bool, str]:
    if expected is None:
        return (not calls), ("ok" if not calls else f"unexpected tool call {calls[0]['name']}")
    if not calls:
        return False, "no tool call"
    call = calls[0]
    if call["name"] != expected["name"]:
        return False, f"called {call['name']} instead of {expected['name']}"
    args = call.get("parsed_arguments")
    if not isinstance(args, dict):
        return False, "arguments were not valid JSON"
    for key, want in (expected.get("args") or {}).items():
        got = args.get(key)
        if isinstance(want, (int, float)) and not isinstance(want, bool):
            if got != want:
                return False, f"{key}={got!r}, expected {want!r}"
        elif _norm(got) != _norm(want):
            return False, f"{key}={got!r}, expected {want!r}"
    return True, "ok"


def score_case(case: dict[str, Any], text: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Rule-based result: {"passed", "failures": [...], "grounded": bool|None,
    "tool_correct": bool|None, "language_ok": bool|None}."""
    checks = case.get("checks") or {}
    lowered = text.lower()
    failures: list[str] = []

    if checks.get("any") and not any(s.lower() in lowered for s in checks["any"]):
        failures.append("none of the expected terms present")
    for term in checks.get("all") or []:
        if term.lower() not in lowered:
            failures.append(f"missing {term!r}")
    for term in checks.get("none") or []:
        if term.lower() in lowered:
            failures.append(f"contains forbidden {term!r}")
    if checks.get("regex") and not re.search(checks["regex"], text, re.I):
        failures.append("answer regex did not match")

    language = None
    if checks.get("language"):
        language = language_ok(text, checks["language"])
        if not language:
            failures.append(f"reply not in {checks['language']}")

    grounded = None
    if case["category"] == "rag":
        if checks.get("answerable", True):
            cited = bool(re.search(r"\[S\d+\]", text))
            if checks.get("cite") and not cited:
                failures.append("no [S#] citation")
            grounded = not failures
        else:
            abstained = any(p in lowered for p in ABSTAIN)
            if not abstained and not checks.get("any"):
                failures.append("did not say the sources lack the answer")
            grounded = not failures

    tool_correct = None
    if case["category"] == "function_calling":
        tool_correct, why = tool_ok(case.get("expected_tool"), tool_calls)
        if not tool_correct:
            failures.append(f"tool: {why}")

    return {"passed": not failures, "failures": failures, "grounded": grounded,
            "tool_correct": tool_correct, "language_ok": language}


def similarity(a: str, b: str) -> float:
    return round(difflib.SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio(), 4)


JUDGE_SYSTEM = (
    "You grade an AI assistant's answer for a Vedic astrology app. Be strict and "
    "consistent. Reply with JSON only: "
    '{"quality": 1-5, "factual": 1-5, "grounded": 1-5 or null, "reason": "<15 words>"}. '
    "quality = helpful, clear, appropriate tone and language. factual = no false "
    "statements (astrology interpretations phrased as tradition are fine). grounded = "
    "only when SOURCES are given: uses only the sources and admits when they lack the answer."
)


def judge_messages(case: dict[str, Any], text: str, tool_calls: list[dict]) -> list[dict]:
    sources = "\n".join(f"[{c['id']}] {c['text']}" for c in case.get("context") or [])
    shown = text or f"(no text; tool calls: {json.dumps(tool_calls)[:400]})"
    hints = json.dumps(case.get("checks") or {}, ensure_ascii=False)[:600]
    body = (f"CATEGORY: {case['category']}\nUSER: {case['prompt']}\n"
            + (f"SOURCES:\n{sources}\n" if sources else "")
            + f"GRADER HINTS (expected properties): {hints}\nANSWER:\n{shown[:4000]}")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": body}]


def parse_judge(raw: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    out: dict[str, Any] = {}
    for key in ("quality", "factual", "grounded"):
        value = data.get(key)
        out[key] = float(value) if isinstance(value, (int, float)) and 1 <= value <= 5 else None
    out["reason"] = str(data.get("reason", ""))[:200]
    return out


__all__ = ["score_case", "tool_ok", "language_ok", "similarity", "judge_messages", "parse_judge"]
