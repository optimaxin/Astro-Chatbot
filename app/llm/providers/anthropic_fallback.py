"""Gemini -> Anthropic cross-provider fallback for the main answer.

Anthropic's Messages API is not OpenAI-wire-compatible (the system prompt is a
top-level parameter, and only user/assistant roles are allowed), so it cannot
go through the shared OpenAI-protocol path and lives here instead. Only used
when `LLM_PROVIDER=gemini` and `ANTHROPIC_API_KEY` is set.
"""

from __future__ import annotations

import time
from typing import Any

from app.llm import reliability
from app.llm.reliability import GatewayError
from app.llm.usage_tracker import (
    LLMUsage,
    Stopwatch,
    calculate_cost,
    extract_usage,
    token_accounting,
    usage_tracker,
)
from app.logging_setup import get_logger

log = get_logger("llm")

PROVIDER_ANTHROPIC = "anthropic"
_client: Any | None = None


def anthropic_client(settings: Any) -> Any:
    global _client
    if _client is not None:
        return _client
    if not settings.anthropic.configured:
        from app.llm.llm_factory import LLMNotConfigured

        raise LLMNotConfigured(
            "ANTHROPIC_API_KEY is not set — required for the Gemini->Anthropic fallback"
        )
    import anthropic

    _client = anthropic.Anthropic(
        api_key=settings.anthropic.api_key,
        base_url=settings.anthropic.base_url or None,
        timeout=settings.anthropic.timeout_seconds,
        max_retries=1,
    )
    return _client


def call_anthropic(
    llm: Any, settings: Any, *, model: str, messages: list[dict[str, Any]], session_id: str,
    user_id: str, request_id: str, is_fallback: bool, temperature: float, max_tokens: int,
    primary_error: str | None = None,
) -> Any:
    from app.llm.llm_factory import LLMResult

    client = anthropic_client(settings)
    watch = Stopwatch()
    system_text = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    conversation = [m for m in messages if m.get("role") in ("user", "assistant")]
    kwargs: dict[str, Any] = {"model": model, "messages": conversation,
                              "temperature": temperature, "max_tokens": max_tokens}
    if system_text:
        kwargs["system"] = system_text
    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:
        llm._record_failure(provider=PROVIDER_ANTHROPIC, model=model, watch=watch, exc=exc,
                            request_id=request_id, session_id=session_id, user_id=user_id,
                            is_fallback=is_fallback, attempt=0)
        raise

    raw_usage = getattr(response, "usage", None)
    tokens = extract_usage(raw_usage)
    input_cost, output_cost, total_cost = calculate_cost(
        model, tokens["input_tokens"], tokens["output_tokens"]
    )
    blocks = getattr(response, "content", None) or []
    text = "".join(
        getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text"
    ).strip()
    usage = usage_tracker.record(LLMUsage(
        request_id=request_id, session_id=session_id, user_id=user_id,
        role=llm.config.role, provider=PROVIDER_ANTHROPIC,
        model=getattr(response, "model", model) or model, latency_ms=watch.elapsed_ms(),
        success=True, timestamp=time.time(), input_cost_usd=input_cost,
        output_cost_usd=output_cost, total_cost_usd=total_cost, fallback_used=is_fallback,
        error_code=primary_error, **tokens, **token_accounting(tokens, messages, text),
    ))
    log.info("llm call complete", extra={
        "role": llm.config.role, "provider": PROVIDER_ANTHROPIC, "model": usage.model,
        "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "latency_ms": usage.latency_ms, "fallback_used": is_fallback,
    })
    if not text:
        raise GatewayError(reliability.EMPTY, f"model {model} returned an empty message")
    return LLMResult(text=text, usage=usage, raw=response)


__all__ = ["anthropic_client", "call_anthropic", "PROVIDER_ANTHROPIC"]
