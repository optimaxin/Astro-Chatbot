"""Error normalisation, retry/backoff and circuit breaking for LLM calls.

Every provider raises its SDK's own exception types. The gateway never
branches on those directly: `classify_error` turns any exception into an
`ErrorInfo` with three decisions already made —

* `retryable`        try the SAME model again after a backoff
* `fallback_ok`      switching to the fallback model could plausibly help
* `trips_circuit`    counts toward opening this model's circuit

so the policy lives here, in one table, instead of being re-derived at each
call site.

The rule for fallback is "technical failure of this model", not "anything
went wrong". A 401 means the key is wrong — the fallback model on the same
provider uses the same key and would fail identically, so falling back only
doubles the latency of an error. A 400 means the request itself is malformed.
Both are reported straight away instead.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

# ── normalised error codes ────────────────────────────────────────────
TIMEOUT = "timeout"
CONNECTION = "connection_error"
RATE_LIMITED = "rate_limited"
UPSTREAM = "upstream_error"          # 5xx
AUTH = "auth_error"                  # 401 / 403
NOT_FOUND = "model_not_found"        # 404
BAD_REQUEST = "bad_request"          # 400 / 422
PAYMENT = "payment_required"         # 402: credits / quota exhausted
EMPTY = "empty_response"
MALFORMED = "malformed_response"     # main answer that is not prose (JSON fragment, template tokens)
CIRCUIT_OPEN = "circuit_open"
CANCELLED = "cancelled"
UNSUPPORTED = "unsupported_capability"  # e.g. image sent to a text-only model
UNKNOWN = "unknown_error"


@dataclass(frozen=True)
class ErrorInfo:
    code: str
    retryable: bool
    fallback_ok: bool
    trips_circuit: bool
    status: int | None = None
    retry_after: float | None = None
    message: str = ""

    def label(self) -> str:
        """`code` plus the HTTP status, e.g. `rate_limited:429`."""
        return f"{self.code}:{self.status}" if self.status else self.code


# code -> (retryable, fallback_ok, trips_circuit)
_POLICY: dict[str, tuple[bool, bool, bool]] = {
    # Not retried on the same model: a call that already used its whole
    # timeout is stalled upstream, and a second identical wait is the worst
    # case for a user (measured live: a 59.6 s Lightning call). The fallback
    # model answers instead.
    TIMEOUT: (False, True, True),
    CONNECTION: (True, True, True),
    RATE_LIMITED: (True, True, True),
    UPSTREAM: (True, True, True),
    NOT_FOUND: (False, True, True),
    PAYMENT: (False, True, False),
    EMPTY: (False, True, False),
    MALFORMED: (False, True, False),
    CIRCUIT_OPEN: (False, True, False),
    UNSUPPORTED: (False, True, False),
    AUTH: (False, False, False),
    BAD_REQUEST: (False, False, False),
    CANCELLED: (False, False, False),
    # An exception we cannot classify keeps the historical behaviour of this
    # service — try the fallback once — but is never retried blindly.
    UNKNOWN: (False, True, False),
}


class GatewayError(RuntimeError):
    """Raised by the gateway itself (empty reply, open circuit)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _status_of(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except (TypeError, ValueError):
        return None  # an HTTP-date form is ignored; normal backoff applies


def _code_for(exc: BaseException, status: int | None) -> str:
    if isinstance(exc, GatewayError):
        return exc.code
    name = type(exc).__name__
    # Checked by name as well as by type so the gateway does not import every
    # SDK, and so a provider SDK we do not know still classifies sensibly.
    if "Timeout" in name or isinstance(exc, TimeoutError):
        return TIMEOUT
    if "Connection" in name or isinstance(exc, ConnectionError):
        return CONNECTION
    if status is None:
        return UNKNOWN
    if status == 429:
        return RATE_LIMITED
    if status == 402:
        return PAYMENT
    if status in (401, 403):
        return AUTH
    if status == 404:
        return NOT_FOUND
    if status in (400, 413, 422):
        return BAD_REQUEST
    if status == 408:
        return TIMEOUT
    if status >= 500:
        return UPSTREAM
    return UNKNOWN


def classify_error(exc: BaseException) -> ErrorInfo:
    status = _status_of(exc)
    code = _code_for(exc, status)
    retryable, fallback_ok, trips = _POLICY.get(code, _POLICY[UNKNOWN])
    return ErrorInfo(
        code=code,
        retryable=retryable,
        fallback_ok=fallback_ok,
        trips_circuit=trips,
        status=status,
        retry_after=_retry_after(exc) if code == RATE_LIMITED else None,
        message=str(exc)[:500],
    )


def mentions_param(info: ErrorInfo, *names: str) -> bool:
    """True when a 400 complains about one of the named request parameters."""
    if info.code != BAD_REQUEST:
        return False
    text = info.message.lower()
    return any(name in text for name in names)


# ── backoff ───────────────────────────────────────────────────────────
# Indirection so tests can make backoff instant without monkeypatching
# `time.sleep` for the whole process.
sleep: Callable[[float], None] = time.sleep


def backoff_delay(
    retry_number: int,
    *,
    base: float,
    cap: float,
    retry_after: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff with full jitter, bounded by `cap`.

    `retry_number` is 1 for the first retry. A server-sent Retry-After is a
    floor, still clamped to `cap` so a hostile or buggy header cannot park a
    user's request for minutes.
    """
    rng = rng or random
    ceiling = min(cap, base * (2 ** max(0, retry_number - 1)))
    delay = rng.uniform(0, ceiling)
    if retry_after is not None:
        delay = max(delay, retry_after)
    return max(0.0, min(cap, delay))


# ── circuit breaker ───────────────────────────────────────────────────
@dataclass
class _Circuit:
    failures: int = 0
    opened_at: float | None = None
    trial_in_flight: bool = False


class CircuitBreaker:
    """Per-(provider, model) breaker: closed -> open -> half-open -> closed.

    In-process only. Each worker learns about an outage on its own, which
    costs at most `threshold` failed calls per worker — acceptable, and it
    avoids making Redis a dependency of every LLM call.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._circuits: dict[tuple[str, str], _Circuit] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def allow(self, key: tuple[str, str], cooldown: float) -> bool:
        with self._lock:
            circuit = self._circuits.get(key)
            if circuit is None or circuit.opened_at is None:
                return True
            if self._clock() - circuit.opened_at < cooldown:
                return False
            # Half-open: exactly one trial request at a time.
            if circuit.trial_in_flight:
                return False
            circuit.trial_in_flight = True
            return True

    def record_success(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._circuits.pop(key, None)

    def record_failure(self, key: tuple[str, str], threshold: int) -> bool:
        """Returns True if this failure opened (or re-opened) the circuit."""
        with self._lock:
            circuit = self._circuits.setdefault(key, _Circuit())
            circuit.failures += 1
            reopened = circuit.trial_in_flight
            circuit.trial_in_flight = False
            if reopened or circuit.failures >= threshold:
                circuit.opened_at = self._clock()
                return True
            return False

    def release_trial(self, key: tuple[str, str]) -> None:
        """A half-open trial ended without a verdict (e.g. a 400)."""
        with self._lock:
            circuit = self._circuits.get(key)
            if circuit is not None:
                circuit.trial_in_flight = False

    def state(self, key: tuple[str, str], cooldown: float) -> str:
        with self._lock:
            circuit = self._circuits.get(key)
            if circuit is None or circuit.opened_at is None:
                return "closed"
            return "open" if self._clock() - circuit.opened_at < cooldown else "half_open"

    def snapshot(self, cooldown: float) -> dict[str, dict[str, Any]]:
        with self._lock:
            keys = list(self._circuits)
        return {f"{p}/{m}": {"state": self.state((p, m), cooldown),
                             "failures": self._circuits.get((p, m), _Circuit()).failures}
                for p, m in keys}

    def reset(self) -> None:
        with self._lock:
            self._circuits.clear()


breaker = CircuitBreaker()

__all__ = [
    "ErrorInfo",
    "GatewayError",
    "CircuitBreaker",
    "breaker",
    "classify_error",
    "mentions_param",
    "backoff_delay",
    "TIMEOUT",
    "CONNECTION",
    "RATE_LIMITED",
    "UPSTREAM",
    "AUTH",
    "NOT_FOUND",
    "BAD_REQUEST",
    "PAYMENT",
    "EMPTY",
    "MALFORMED",
    "CIRCUIT_OPEN",
    "CANCELLED",
    "UNSUPPORTED",
    "UNKNOWN",
]
