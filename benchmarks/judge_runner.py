"""Reliable LLM-judge calls for the benchmarks: bounded concurrency, pacing,
retry with exponential backoff and jitter, an on-disk cache, and an explicit
status for every verdict.

Root cause it fixes (final quality gate): the conciseness benchmark judged 2
of 112 turns. Each judge call was tried twice with no wait; the NVIDIA free
tier answered 429, the client opened its 30 s circuit, and every later call
failed instantly inside that window — the failures were then silently
missing from the averages. Here a rate limit or an open circuit is waited
out, each failure is recorded with its reason, and coverage is reported.

    runner = JudgeRunner("kq-judge", "kq-judge-1.0")
    verdict, status = runner.run(cache_key_parts, messages, valid=lambda v: "relevance" in v)
    # status: "cached" | "judged" | "failed:<reason>"
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

_CACHE_DIR = Path(__file__).resolve().parents[1] / "logs" / "judge-cache"


class JudgeRunner:
    def __init__(self, name: str, version: str, *, max_concurrency: int | None = None,
                 min_interval: float | None = None, attempts: int | None = None,
                 base_backoff: float = 4.0, max_backoff: float = 60.0, circuit_wait: float = 32.0,
                 cache_path: Path | None = None) -> None:
        self.name, self.version = name, version
        self.max_concurrency = max_concurrency or int(os.environ.get("JUDGE_CONCURRENCY", "2"))
        self.min_interval = min_interval if min_interval is not None else float(os.environ.get("JUDGE_MIN_INTERVAL",
                                                                                               "1.5"))
        self.attempts = attempts or int(os.environ.get("JUDGE_ATTEMPTS", "6"))
        self.base_backoff, self.max_backoff, self.circuit_wait = base_backoff, max_backoff, circuit_wait
        self.cache_path = cache_path or _CACHE_DIR / f"{name}.json"
        self._slots = threading.BoundedSemaphore(self.max_concurrency)
        self._pace_lock, self._cache_lock = threading.Lock(), threading.Lock()
        self._last = 0.0
        self._cache: dict[str, Any] | None = None
        self.stats: Counter[str] = Counter()

    # ── cache ──────────────────────────────────────────────────────────
    def _key(self, parts: list[Any]) -> str:
        blob = json.dumps([self.name, self.version, *parts], ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _load(self) -> dict[str, Any]:
        if self._cache is None:
            self._cache = (json.loads(self.cache_path.read_text(encoding="utf-8"))
                           if self.cache_path.exists() else {})
        return self._cache

    def save(self) -> None:
        with self._cache_lock:
            if self._cache is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")

    # ── pacing and backoff ─────────────────────────────────────────────
    def _pace(self) -> None:
        with self._pace_lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _delay(self, attempt: int, reason: str) -> float:
        if reason in ("circuit_open", "rate_limited"):
            return self.circuit_wait + random.uniform(0, 4)
        ceiling = min(self.max_backoff, self.base_backoff * (2 ** attempt))
        return random.uniform(ceiling / 2, ceiling)

    @staticmethod
    def _reason(exc: BaseException) -> str:
        text = f"{type(exc).__name__} {exc}".lower()
        if "circuit" in text:
            return "circuit_open"
        if "429" in text or "rate" in text:
            return "rate_limited"
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if isinstance(exc, (ValueError, json.JSONDecodeError)):
            return "bad_json"
        return type(exc).__name__

    # ── one verdict ────────────────────────────────────────────────────
    def run(self, key_parts: list[Any], messages: list[dict[str, str]],
            valid: Callable[[dict[str, Any]], bool] = lambda v: True,
            call: Callable[[list[dict[str, str]]], str] | None = None) -> tuple[dict[str, Any] | None, str]:
        key = self._key(key_parts)
        with self._cache_lock:
            hit = self._load().get(key)
        if hit is not None:
            self.stats["cached"] += 1
            return hit, "cached"
        call = call or _default_call(self.name)
        reason = "no_attempt"
        for attempt in range(self.attempts):
            with self._slots:
                self._pace()
                try:
                    raw = call(messages)
                    verdict = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                    if not isinstance(verdict, dict) or not valid(verdict):
                        raise ValueError("verdict missing fields")
                except Exception as exc:  # noqa: BLE001 — classified, retried, then reported
                    reason = self._reason(exc)
                    self.stats[f"retry:{reason}"] += 1
                else:
                    with self._cache_lock:
                        self._load()[key] = verdict
                        checkpoint = len(self._cache or {}) % 20 == 0
                    if checkpoint:
                        self.save()
                    self.stats["judged"] += 1
                    return verdict, "judged"
            if attempt + 1 < self.attempts:
                time.sleep(self._delay(attempt, reason))
        self.stats[f"failed:{reason}"] += 1
        return None, f"failed:{reason}"


def _default_call(name: str) -> Callable[[list[dict[str, str]]], str]:
    def call(messages: list[dict[str, str]]) -> str:
        from app.llm.llm_factory import ROLE_EVALUATE, get_llm

        return get_llm(role=ROLE_EVALUATE).complete(
            messages, session_id=name, user_id="benchmark", request_id=f"{name}-{uuid.uuid4().hex[:8]}",
            temperature=0.0).text
    return call


_REGISTRY: dict[str, JudgeRunner] = {}
_REGISTRY_LOCK = threading.Lock()


def get_runner(name: str, version: str) -> JudgeRunner:
    """One shared runner per judge prompt, so pacing and caching are global."""
    with _REGISTRY_LOCK:
        if name not in _REGISTRY:
            _REGISTRY[name] = JudgeRunner(name, version)
        return _REGISTRY[name]


def save_all() -> dict[str, dict[str, int]]:
    for r in _REGISTRY.values():
        r.save()
    return {name: dict(r.stats) for name, r in _REGISTRY.items()}


def coverage(statuses: list[str | None], excluded: dict[str, int] | None = None) -> dict[str, Any]:
    """How many verdicts exist, how many failed and why, and how many samples
    were not sent to the judge at all (with the reason) — never hidden."""
    counts = Counter(s or "not_requested" for s in statuses)
    judged = counts["judged"] + counts["cached"]
    eligible = len(statuses)
    return {"eligible": eligible, "with_verdict": judged,
            "coverage_%": round(100.0 * judged / eligible, 1) if eligible else None,
            "failed": {k: v for k, v in counts.items() if k.startswith("failed")},
            "excluded_from_judge": excluded or {}}


__all__ = ["JudgeRunner", "coverage", "get_runner", "save_all"]
