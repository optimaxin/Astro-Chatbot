"""Work that must happen, but not while the user waits.

Usage metering rows (Redis counters, Postgres) and the sampled LLM judge run
here after the reply is on its way. Both were already best-effort — a failure
never cost a user their answer — so moving them off the request path changes
when they happen, not whether.

* Bounded: at most `MAX_PENDING` jobs queue up; beyond that new jobs run
  inline rather than being dropped, so metering data is never silently lost.
* Context-preserving: each job runs in a copy of the submitting request's
  contextvars, so log lines still carry that request's id.
* Drained at exit (`atexit`), and `drain()` lets scripts and tests wait.
"""

from __future__ import annotations

import atexit
import contextvars
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable

from app.logging_setup import get_logger

log = get_logger("background")

MAX_PENDING = 500
# Shadow roles (quality checker, humanizer) have their OWN small pool and queue: a slow or rate-limited
# Ultra call must never hold up metering writes, and a burst of shadow samples is dropped, not queued
# without bound.
SHADOW_MAX_PENDING = 50
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aicore-bg")
_shadow_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aicore-shadow")
_pending: set[Future] = set()
_shadow_pending: set[Future] = set()
_lock = threading.Lock()
# Process-lifetime counters, for /health and the shadow report (saturation, drops, failures).
_counters = {"submitted": 0, "completed": 0, "failed": 0, "inline_overflow": 0,
             "shadow_submitted": 0, "shadow_completed": 0, "shadow_failed": 0, "shadow_dropped": 0,
             "high_water": 0, "shadow_high_water": 0}


def _bump(key: str, n: int = 1) -> None:
    with _lock:
        _counters[key] += n


def _run(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    _run_counted(fn, args, kwargs, failed_key="failed")


def _run_shadow(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    _run_counted(fn, args, kwargs, failed_key="shadow_failed")


def _run_counted(fn: Callable[..., Any], args: tuple, kwargs: dict, *, failed_key: str) -> None:
    # The copied context carries the submitting request's trace; background
    # work happens after the reply and must not be recorded as part of it.
    from app.observability import trace

    trace._current.set(None)  # noqa: SLF001 — same package
    try:
        fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 — background work must never crash the worker
        _bump(failed_key)
        log.exception("background job failed", extra={"job": getattr(fn, "__name__", str(fn))})


def submit(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    ctx = contextvars.copy_context()
    future: Future | None = None
    with _lock:
        overflow = len(_pending) >= MAX_PENDING
        if not overflow:
            future = _executor.submit(ctx.run, _run, fn, *args, **kwargs)
            _pending.add(future)
            _counters["submitted"] += 1
            _counters["high_water"] = max(_counters["high_water"], len(_pending))
        else:
            _counters["inline_overflow"] += 1
    # Registered OUTSIDE the lock: if the job has already finished,
    # add_done_callback runs `_discard` immediately on this thread, and
    # `_discard` takes `_lock` — holding it here would deadlock.
    if future is not None:
        future.add_done_callback(_discard)
    if overflow:
        log.warning("background queue full; running job inline")
        ctx.run(_run, fn, *args, **kwargs)


def try_submit(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    """Like `submit`, but for advisory shadow work: runs on a separate small pool and is DROPPED (returns
    False) when that queue is full, never run on the caller's thread, so it cannot delay a reply."""
    ctx = contextvars.copy_context()
    future: Future | None = None
    with _lock:
        if len(_shadow_pending) >= SHADOW_MAX_PENDING:
            _counters["shadow_dropped"] += 1
        else:
            future = _shadow_executor.submit(ctx.run, _run_shadow, fn, *args, **kwargs)
            _shadow_pending.add(future)
            _counters["shadow_submitted"] += 1
            _counters["shadow_high_water"] = max(_counters["shadow_high_water"], len(_shadow_pending))
    if future is None:
        log.warning("shadow queue full; sample dropped")
        return False
    future.add_done_callback(_discard_shadow)
    return True


def _discard(future: Future) -> None:
    with _lock:
        _pending.discard(future)
        _counters["completed"] += 1


def _discard_shadow(future: Future) -> None:
    with _lock:
        _shadow_pending.discard(future)
        _counters["shadow_completed"] += 1


def stats() -> dict[str, int]:
    """Queue depth and lifetime counters: saturation (high_water vs the limits), drops, failures."""
    with _lock:
        return {**_counters, "pending": len(_pending), "shadow_pending": len(_shadow_pending),
                "max_pending": MAX_PENDING, "shadow_max_pending": SHADOW_MAX_PENDING}


def drain(timeout: float | None = 30.0) -> bool:
    """Wait for queued jobs. True if everything finished."""
    with _lock:
        snapshot = list(_pending) + list(_shadow_pending)
    done, not_done = wait(snapshot, timeout=timeout)
    return not not_done


def pending() -> int:
    with _lock:
        return len(_pending) + len(_shadow_pending)


atexit.register(drain, 10.0)

__all__ = ["submit", "try_submit", "drain", "pending", "stats"]
