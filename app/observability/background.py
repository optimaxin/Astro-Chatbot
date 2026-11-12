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
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aicore-bg")
_pending: set[Future] = set()
_lock = threading.Lock()


def _run(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    # The copied context carries the submitting request's trace; background
    # work happens after the reply and must not be recorded as part of it.
    from app.observability import trace

    trace._current.set(None)  # noqa: SLF001 — same package
    try:
        fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 — background work must never crash the worker
        log.exception("background job failed", extra={"job": getattr(fn, "__name__", str(fn))})


def submit(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    ctx = contextvars.copy_context()
    future: Future | None = None
    with _lock:
        overflow = len(_pending) >= MAX_PENDING
        if not overflow:
            future = _executor.submit(ctx.run, _run, fn, *args, **kwargs)
            _pending.add(future)
    # Registered OUTSIDE the lock: if the job has already finished,
    # add_done_callback runs `_discard` immediately on this thread, and
    # `_discard` takes `_lock` — holding it here would deadlock.
    if future is not None:
        future.add_done_callback(_discard)
    if overflow:
        log.warning("background queue full; running job inline")
        ctx.run(_run, fn, *args, **kwargs)


def _discard(future: Future) -> None:
    with _lock:
        _pending.discard(future)


def drain(timeout: float | None = 30.0) -> bool:
    """Wait for queued jobs. True if everything finished."""
    with _lock:
        snapshot = list(_pending)
    done, not_done = wait(snapshot, timeout=timeout)
    return not not_done


def pending() -> int:
    with _lock:
        return len(_pending)


atexit.register(drain, 10.0)

__all__ = ["submit", "drain", "pending"]
