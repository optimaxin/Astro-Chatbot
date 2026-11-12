"""Hedged requests for the main answer: bound the latency tail.

Measured live (2026-11-12): Lightning normally answers in 1–3 s, but single
calls stalled for 10–60 s on the free tier. A timeout cannot fix that tail —
a 25 s timeout still means a 25 s answer. So:

    t=0      ask the primary model
    t=H      no answer yet -> ALSO ask the fallback model
    first successful answer wins; the slower call finishes in the background
    and is still metered (its usage row is real traffic).

Only the main-turn role is hedged, only when its fallback is on the same
provider, and only once the primary is already slow — a normal turn makes
exactly one call. The winning text goes through the same chart, language and
citation validation as any other answer. `LLM_HEDGE_AFTER_SECONDS=0` turns
hedging off.
"""

from __future__ import annotations

import contextvars
import dataclasses
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from app.logging_setup import get_logger

log = get_logger("llm.hedge")

_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="aicore-hedge")


def hedged_complete(llm: Any, *, hedge_after: float, deadline: float, common: dict[str, Any]) -> Any:
    """Run llm's primary; add its fallback after `hedge_after` seconds.

    Returns the first successful LLMResult, or raises the primary's error if
    both legs fail (the caller then reports LLMCallFailed as usual)."""
    import time

    from app.llm.llm_factory import LLM

    primary_llm = LLM(llm.config)
    fallback_cfg = dataclasses.replace(llm.config, model=llm.config.fallback_model, fallback_model=None)
    fallback_llm = LLM(fallback_cfg)

    def run(leg: Any, model: str, is_fallback: bool, primary_error: str | None) -> Any:
        return leg._with_retries(model=model, is_fallback=is_fallback, provider_override="same",
                                 deadline=deadline, primary_error=primary_error, **common)

    primary: Future = _pool.submit(contextvars.copy_context().run, run, primary_llm,
                                   llm.config.model, False, None)
    done, _ = wait([primary], timeout=hedge_after)
    if done:
        try:
            return primary.result()
        except Exception as exc:  # noqa: BLE001 — failed fast: same rules as the sequential path
            from app.llm.reliability import classify_error

            info = classify_error(exc)
            if not llm.may_fall_back(info, "same") or time.monotonic() >= deadline:
                raise
            log.warning("falling back to secondary model",
                        extra={"role": llm.config.role, "to_model": llm.config.fallback_model,
                               "reason": info.label()})
            return run(fallback_llm, llm.config.fallback_model, True, info.label())

    log.warning("primary slow; hedging with fallback",
                extra={"role": llm.config.role, "primary": llm.config.model,
                       "fallback": llm.config.fallback_model, "after_s": hedge_after})
    hedge: Future = _pool.submit(contextvars.copy_context().run, run, fallback_llm,
                                 llm.config.fallback_model, True, "slow_primary")
    pending = {primary, hedge}
    first_error: BaseException | None = None
    while pending:
        remaining = max(0.0, deadline - time.monotonic())
        done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
        if not done:
            break
        for future in done:
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 — try the other leg
                first_error = first_error or exc
                continue
            log.info("hedge resolved", extra={"winner": "fallback" if future is hedge else "primary"})
            return result
    if first_error is not None:
        raise first_error
    raise TimeoutError("hedged call exceeded the deadline")


__all__ = ["hedged_complete"]
