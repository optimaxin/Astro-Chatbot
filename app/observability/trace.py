"""Per-request latency traces.

    with trace.request(request_id) as t:      # once per /chat turn
        with trace.span("redis.get_history"):
            ...
    t.summary()  -> {"total_ms": .., "spans": [{"name", "ms", "ok"}, ...]}

Spans are collected in a contextvar, so nested code records into the trace of
the request it runs for without passing anything around. Code running with no
active trace (scripts, tests, background workers) records nothing.

Each finished trace is appended to `logs/latency.jsonl` — span names and
timings plus the request id, intent and model. Never birth details, question
text or answers: a latency log has no reason to hold user content.
`scripts/latency_report.py` turns that file into avg/p50/p95/p99 per span.
"""

from __future__ import annotations

import contextvars
import functools
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from app.config.settings import BASE_DIR, settings

_current: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar("trace", default=None)
_write_lock = threading.Lock()


@dataclass
class Trace:
    request_id: str
    started: float = field(default_factory=time.perf_counter)
    spans: list[dict[str, Any]] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    total_ms: int | None = None

    def add(self, name: str, ms: float, ok: bool) -> None:
        self.spans.append({"name": name, "ms": round(ms, 1), "ok": ok})

    def summary(self) -> dict[str, Any]:
        by_name: dict[str, float] = {}
        for s in self.spans:
            by_name[s["name"]] = round(by_name.get(s["name"], 0.0) + s["ms"], 1)
        return {"request_id": self.request_id, "total_ms": self.total_ms,
                "spans": self.spans, "by_span_ms": by_name, **self.attrs}


def current() -> Trace | None:
    return _current.get()


def annotate(**attrs: Any) -> None:
    """Attach non-sensitive attributes (intent, model, flags) to the trace."""
    t = _current.get()
    if t is not None:
        t.attrs.update(attrs)


@contextmanager
def span(name: str) -> Iterator[None]:
    t = _current.get()
    if t is None:
        yield
        return
    started = time.perf_counter()
    ok = True
    try:
        yield
    except BaseException:
        ok = False
        raise
    finally:
        t.add(name, (time.perf_counter() - started) * 1000, ok)


def traced(name: str) -> Callable:
    """Decorator form of `span`."""
    def wrap(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            with span(name):
                return fn(*args, **kwargs)
        return inner
    return wrap


def latency_log_path() -> Path:
    path = Path(settings.log_file).parent / "latency.jsonl"
    return path if path.is_absolute() else BASE_DIR / path


@contextmanager
def request(request_id: str, persist: bool = True) -> Iterator[Trace]:
    t = Trace(request_id=request_id)
    token = _current.set(t)
    try:
        yield t
    finally:
        t.total_ms = int((time.perf_counter() - t.started) * 1000)
        _current.reset(token)
        if persist:
            _append(t.summary())


def _append(record: dict[str, Any]) -> None:
    path = latency_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": time.time(), **record}, default=str)
        with _write_lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass  # a latency log must never cost a user their answer


__all__ = ["Trace", "request", "span", "traced", "annotate", "current", "latency_log_path"]
