"""LLM naturalness judge — one of three signals, never the only one.

Verdicts are cached on disk by (judge version, language, question, text,
reference), so re-running the benchmark on unchanged replies gives identical
scores and costs nothing; only new or changed replies are judged.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

JUDGE_VERSION = "hqb-judge-1.0"
_CACHE_PATH = Path(__file__).resolve().parents[2] / "logs" / "hinglish-bench" / "judge-cache.json"
_LOCK = threading.Lock()
_cache: dict[str, Any] | None = None
# Pacing: the NVIDIA endpoint rate-limits (429) and the client then opens a
# 30 s circuit; a judge that retries inside that window only fails faster.
_MIN_INTERVAL = float(os.environ.get("HQB_JUDGE_MIN_INTERVAL", "1.5"))
_BACKOFF = float(os.environ.get("HQB_JUDGE_BACKOFF", "35"))
_PACE = threading.Lock()
_last_call = 0.0


def _pace() -> None:
    global _last_call
    with _PACE:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()

JUDGE_PROMPT = """You are a fluent native speaker of Hindi, English and Roman-script Hinglish, reviewing ONE reply
from a Vedic astrology chat assistant. Judge the LANGUAGE, not the astrology.

Score each dimension 1-5 (5 = flawless, 4 = minor slip, 3 = noticeable error, 2 = several errors, 1 = broken):
- grammar: gender and number agreement, verb forms and tense, postpositions (mein/se/ka/ki/ke/par), sentence structure.
- naturalness: would a fluent speaker say it this way in a friendly chat? Penalise word-for-word translation,
  heavy Sanskrit/bureaucratic Hindi in casual Hinglish, English word order, filler.
- vocabulary: correct Roman-Hindi spelling (aapko, not aapako), English words in English spelling (partner, career,
  not paartanar, kareer), real words (no "badhyaari"), natural Hindi-English combinations ("health-related dikkat
  aa sakti hai", not "health issues la sakta hai").
- meaning: if a REFERENCE is given, does the reply keep its meaning and every chart fact (planets, houses, signs,
  dates)? Without a reference: is it free of guarantees ("pakka", "definitely") and of invented specifics?
- consistency: one language and script throughout as expected (Hinglish = Roman script only; Hindi = Devanagari;
  English = English), respectful "aap" throughout, no "tum", no assuming the user's gender.

Expected language: {language}. Natural Hinglish KEEPS common English words (career, partnership, business,
relationship, health, property, Mahadasha, growth) — do not penalise them.

Return strict JSON only:
{{"grammar": n, "naturalness": n, "vocabulary": n, "meaning": n, "consistency": n,
  "errors": [{{"category": "<one of: gender_agreement, number_agreement, postposition, verb_form, sentence_structure,
  spelling, transliteration, non_word, word_choice, collocation, literal_translation, heavy_register, word_order,
  verbose, filler, fact_change, certainty, unsafe_claim, machinery, register_mix, gender_assumption, script_mixing,
  language_switch>", "span": "<the wrong words>", "fix": "<better words>"}}]}}"""


def _key(language: str, question: str, text: str, reference: str | None) -> str:
    blob = json.dumps([JUDGE_VERSION, language, question, text, reference or ""], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load() -> dict[str, Any]:
    global _cache
    if _cache is None:
        _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8")) if _CACHE_PATH.exists() else {}
    return _cache


def save_cache() -> None:
    with _LOCK:
        if _cache is not None:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")


def judge(language: str, question: str, text: str, reference: str | None = None,
          attempts: int = 6) -> dict[str, Any] | None:
    """Cached verdict, or None when the judge model is unavailable (the
    benchmark then scores on rules and facts alone and says so)."""
    return judge_with_status(language, question, text, reference, attempts)[0]


def judge_with_status(language: str, question: str, text: str, reference: str | None = None,
                      attempts: int = 6) -> tuple[dict[str, Any] | None, str]:
    """(verdict, status): status is "cached", "judged" or "failed:<reason>".
    A rate limit or open circuit is waited out (35 s + jitter); other errors
    back off exponentially. Every failure is reported, never dropped."""
    import random

    from benchmarks.judge_runner import JudgeRunner

    key = _key(language, question, text, reference)
    with _LOCK:
        hit = _load().get(key)
    if hit is not None:
        return hit, "cached"
    from app.llm.llm_factory import ROLE_EVALUATE, get_llm

    user = f"User question: {question}\n\nReply to judge:\n{text}"
    if reference:
        user += f"\n\nREFERENCE (the meaning and facts the reply must keep):\n{reference}"
    prompt = JUDGE_PROMPT.format(language=language)
    reason = "no_attempt"
    for attempt in range(attempts):
        _pace()
        try:
            raw = get_llm(role=ROLE_EVALUATE).complete(
                [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
                session_id="hqb-judge", user_id="hqb-bench", request_id=f"hqb-{uuid.uuid4().hex[:8]}",
                temperature=0.0).text
            verdict = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            if not all(isinstance(verdict.get(d), (int, float)) for d in
                       ("grammar", "naturalness", "vocabulary", "meaning", "consistency")):
                raise ValueError("verdict missing dimensions")
            with _LOCK:
                _load()[key] = verdict
                checkpoint = len(_load()) % 20 == 0
            if checkpoint:
                save_cache()          # a stopped run keeps what it judged
            return verdict, "judged"
        except Exception as exc:  # noqa: BLE001 — classified, retried, then reported
            reason = JudgeRunner._reason(exc)
        if attempt + 1 < attempts:
            time.sleep(_BACKOFF + random.uniform(0, 4) if reason in ("rate_limited", "circuit_open")
                       else random.uniform(2, 4) * (2 ** attempt))
    return None, f"failed:{reason}"


def judge_model() -> str:
    try:
        from app.llm.llm_factory import ROLE_EVALUATE, _role_config

        return _role_config(ROLE_EVALUATE).model
    except Exception:  # noqa: BLE001
        return "unavailable"


__all__ = ["judge", "judge_with_status", "save_cache", "judge_model", "JUDGE_VERSION"]
