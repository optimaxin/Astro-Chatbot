"""Streaming completions through the gateway.

    stream = get_llm(ROLE_MAIN).stream(messages, request_id=...)
    for delta in stream:          # str fragments as they arrive
        send(delta)
    stream.result                 # LLMResult once the stream finished
    stream.close()                # cancel: closes the upstream HTTP response

Rules that differ from `complete()` and why:

* Retry and fallback happen only BEFORE the first fragment is yielded. Once
  text has reached the caller, switching model would splice two different
  answers into one message, so a mid-stream failure ends the stream with
  `LLMCallFailed` and the caller decides what to show.
* Cancellation (`close()`, or the consumer abandoning the generator) closes
  the upstream response so the provider stops generating, and is recorded as
  a `cancelled` usage row — not as a model failure, and it never trips the
  circuit breaker.
* Usage comes from the provider's final usage chunk (requested via
  `stream_options.include_usage` where the model accepts it). When absent the
  row is marked `token_source="estimated"`; estimates never fill the real
  token columns.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

from app.config.settings import settings
from app.llm import reliability
from app.llm.model_registry import get_spec
from app.llm.reliability import ErrorInfo, GatewayError, breaker, classify_error
from app.llm.usage_tracker import (
    LLMUsage,
    Stopwatch,
    calculate_cost,
    extract_usage,
    new_request_id,
    token_accounting,
    usage_tracker,
)
from app.logging_setup import get_logger

log = get_logger("llm.stream")


class ThinkFilter:
    """Drops a leading <think>...</think> block from a token stream."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.buffer = ""
        self.state = "start" if enabled else "pass"

    def feed(self, text: str) -> str:
        if self.state == "pass":
            return text
        self.buffer += text
        if self.state == "start":
            stripped = self.buffer.lstrip()
            if len(stripped) < len(self.OPEN) and self.OPEN.startswith(stripped.lower()):
                return ""  # could still become "<think>"
            if not stripped.lower().startswith(self.OPEN):
                self.state, out, self.buffer = "pass", self.buffer, ""
                return out
            self.state = "thinking"
        end = self.buffer.lower().find(self.CLOSE)
        if end < 0:
            return ""
        out = self.buffer[end + len(self.CLOSE):].lstrip()
        self.state, self.buffer = "pass", ""
        return out

    def flush(self) -> str:
        # An unterminated think block is reasoning, not answer.
        out = self.buffer if self.state in ("start", "pass") else ""
        self.buffer = ""
        return out


class LLMStream:
    def __init__(self, llm: Any, messages: list[dict[str, Any]], *, session_id: str = "-",
                 user_id: str = "-", request_id: str | None = None,
                 temperature: float | None = None, max_tokens: int | None = None) -> None:
        self.llm = llm
        self.messages = messages
        self.session_id = session_id
        self.user_id = user_id
        self.request_id = request_id or new_request_id()
        self.temperature = llm.config.temperature if temperature is None else temperature
        self.max_tokens = llm.config.max_tokens if max_tokens is None else max_tokens
        self.result: Any = None
        self._response: Any = None
        self._last_kwargs: dict[str, Any] = {}
        self._gen = self._run()

    def __iter__(self) -> Iterator[str]:
        return self._gen

    def __next__(self) -> str:
        return next(self._gen)

    def close(self) -> None:
        self._gen.close()

    def __enter__(self) -> "LLMStream":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── internals ─────────────────────────────────────────────────────
    def _run(self) -> Iterator[str]:
        from app.llm.llm_factory import LLMCallFailed

        deadline = time.monotonic() + settings.reliability.deadline_seconds
        last_error: Exception | None = None
        last_info: ErrorInfo | None = None
        for model, is_fallback, provider_override in self.llm.attempts():
            if provider_override != "same":
                continue  # the Anthropic cross-provider path is non-streaming
            if is_fallback and (not self.llm.may_fall_back(last_info, provider_override)
                                or time.monotonic() >= deadline):
                break
            attempt = 0
            key = (self.llm.provider_for("same"), model)
            cfg = settings.reliability
            while True:
                emitted = False
                try:
                    if not breaker.allow(key, cfg.circuit_cooldown_seconds):
                        raise GatewayError(reliability.CIRCUIT_OPEN, f"circuit open for {key}")
                    for delta in self._one_attempt(model, is_fallback, attempt,
                                                   last_info.label() if last_info else None):
                        emitted = True
                        yield delta
                    breaker.record_success(key)
                    return
                except GeneratorExit:
                    raise
                except Exception as exc:  # noqa: BLE001 — classified below
                    from app.llm.llm_factory import learn_from_bad_request

                    last_error, last_info = exc, classify_error(exc)
                    if (not emitted and last_info.code == reliability.BAD_REQUEST
                            and learn_from_bad_request(model, last_info, self._last_kwargs)):
                        continue  # retry without the rejected parameter
                    if last_info.trips_circuit:
                        breaker.record_failure(key, cfg.circuit_failure_threshold)
                    else:
                        breaker.release_trial(key)
                    if emitted:
                        raise LLMCallFailed(
                            f"stream from {model} failed after output began: {exc}", last_info
                        ) from exc
                    if not last_info.retryable or attempt >= cfg.max_retries:
                        break
                    delay = reliability.backoff_delay(
                        attempt + 1, base=cfg.backoff_base_seconds,
                        cap=cfg.backoff_max_seconds, retry_after=last_info.retry_after)
                    if time.monotonic() + delay >= deadline:
                        break
                    reliability.sleep(delay)
                    attempt += 1
        raise LLMCallFailed(
            f"all models failed to stream for role {self.llm.config.role!r}: {last_error}",
            last_info,
        ) from last_error

    def _one_attempt(self, model: str, is_fallback: bool, attempt: int,
                     primary_error: str | None) -> Iterator[str]:
        from app.llm.llm_factory import LLMResult, _openai_client, active_provider, build_request

        provider = active_provider()
        spec = get_spec(model)
        kwargs = build_request(
            provider=provider, spec=spec, messages=self.messages,
            temperature=self.temperature, max_tokens=self.max_tokens,
            response_format=None, tools=None, tool_choice=None, streaming=True,
        )
        self._last_kwargs = kwargs
        client = _openai_client()
        watch = Stopwatch()
        base = dict(request_id=self.request_id, session_id=self.session_id,
                    user_id=self.user_id, role=self.llm.config.role, provider=provider,
                    fallback_used=is_fallback, attempt=attempt, streamed=True)
        parts: list[str] = []
        raw_usage: Any = None
        served_model = model
        think = ThinkFilter(spec.strips_think_tags)
        first_token_ms: int | None = None
        try:
            self._response = client.chat.completions.create(**kwargs)
            for chunk in self._response:
                served_model = getattr(chunk, "model", None) or served_model
                if getattr(chunk, "usage", None) is not None:
                    raw_usage = chunk.usage
                for choice in getattr(chunk, "choices", None) or []:
                    delta = getattr(getattr(choice, "delta", None), "content", None)
                    if delta:
                        visible = think.feed(delta)
                        if visible:
                            if first_token_ms is None:
                                first_token_ms = watch.elapsed_ms()
                            parts.append(visible)
                            yield visible
            tail = think.flush()
            if tail:
                parts.append(tail)
                yield tail
        except GeneratorExit:
            self._close_response()
            usage_tracker.record(LLMUsage(
                model=model, latency_ms=watch.elapsed_ms(), success=False,
                timestamp=time.time(), error="cancelled by client",
                error_code=reliability.CANCELLED, **base,
                **token_accounting({}, self.messages, "".join(parts)),
            ))
            log.info("stream cancelled", extra={"model": model, "chars": len("".join(parts))})
            raise
        except Exception as exc:
            self._close_response()
            usage_tracker.record(LLMUsage(
                model=model, latency_ms=watch.elapsed_ms(), success=False,
                timestamp=time.time(), error=f"{type(exc).__name__}: {exc}"[:500],
                error_code=classify_error(exc).label(), **base,
            ))
            raise

        text = "".join(parts).strip()
        tokens = extract_usage(raw_usage)
        costs = calculate_cost(model, tokens["input_tokens"], tokens["output_tokens"])
        usage = usage_tracker.record(LLMUsage(
            model=served_model, latency_ms=watch.elapsed_ms(), success=bool(text),
            timestamp=time.time(), input_cost_usd=costs[0], output_cost_usd=costs[1],
            total_cost_usd=costs[2],
            error=None if text else "GatewayError: empty response",
            error_code=primary_error if text else reliability.EMPTY,
            **base, **tokens, **token_accounting(tokens, self.messages, text),
        ))
        log.info("llm stream complete", extra={
            "model": served_model, "first_token_ms": first_token_ms,
            "latency_ms": usage.latency_ms, "token_source": usage.token_source,
        })
        if not text:
            raise GatewayError(reliability.EMPTY, f"model {model} streamed no text")
        self.result = LLMResult(text=text, usage=usage)

    def _close_response(self) -> None:
        response, self._response = self._response, None
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 — closing is best-effort
                pass


__all__ = ["LLMStream", "ThinkFilter"]
