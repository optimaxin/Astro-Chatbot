"""The only place in the application that talks to an LLM.

Graph nodes ask for a *role*, never a model:

    llm = get_llm(role="job-main-turn")
    result = llm.complete(messages, session_id=..., user_id=..., request_id=...)

Roles map to models through `settings.models`, which today reads env vars.
Moving that mapping into Postgres later means changing `ROLE_CONFIG` and
nothing else — no node, prompt or test refers to a model id.

Every call returns an `LLMResult` carrying the text *and* a fully populated
`LLMUsage` record, so token/latency/cost tracking cannot be forgotten at a
call site.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.config.settings import settings
from app.llm.usage_tracker import (
    LLMUsage,
    Stopwatch,
    calculate_cost,
    extract_native_cost,
    extract_usage,
    new_request_id,
    usage_tracker,
)
from app.logging_setup import get_logger

log = get_logger("llm")

PROVIDER_OPENROUTER = "openrouter"
PROVIDER_GEMINI = "gemini"


def active_provider() -> str:
    """Which upstream the roles resolve against, per `LLM_PROVIDER`.

    Both providers speak the OpenAI chat-completions protocol — Gemini via its
    `/v1beta/openai/` compatibility endpoint — so switching changes the client's
    base URL, credential and model names, and nothing else in the pipeline.
    An unrecognised value falls back to OpenRouter rather than failing at
    import, so a typo degrades to the production provider instead of taking
    the service down.
    """
    provider = settings.llm_provider
    return provider if provider in {PROVIDER_OPENROUTER, PROVIDER_GEMINI} else PROVIDER_OPENROUTER


# Role names are the public API of this module.
ROLE_MAIN = "job-main-turn"
ROLE_CLASSIFY = "job-classify-intent"
ROLE_EVALUATE = "job-evaluate-response"


class LLMNotConfigured(RuntimeError):
    pass


class LLMCallFailed(RuntimeError):
    """Raised only when the primary *and* the fallback model both failed."""


@dataclass
class RoleConfig:
    role: str
    model: str
    temperature: float
    max_tokens: int
    fallback_model: str | None = None
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class _RoleModels:
    """The four model ids a provider supplies, normalised across providers."""

    main: str
    fallback: str
    classifier: str
    evaluator: str


def _models_for_provider() -> _RoleModels:
    """Model ids for the active provider.

    The two providers name the same underlying models differently
    (`google/gemini-2.5-flash` on OpenRouter, `gemini-2.5-flash` direct), so
    the mapping lives here rather than in every caller. No node, prompt or
    test refers to a model id.
    """
    if active_provider() == PROVIDER_GEMINI:
        gemini = settings.gemini
        return _RoleModels(
            main=gemini.main,
            fallback=gemini.fallback,
            classifier=gemini.classifier,
            evaluator=gemini.evaluator,
        )
    models = settings.models
    return _RoleModels(
        main=models.main,
        fallback=models.fallback,
        classifier=models.classifier,
        evaluator=models.evaluator,
    )


def _role_config(role: str) -> RoleConfig:
    # Which model, per provider; how to run it, shared across providers.
    models = _models_for_provider()
    tuning = settings.models
    if role == ROLE_MAIN:
        return RoleConfig(
            role=role,
            model=models.main,
            temperature=tuning.main_temperature,
            max_tokens=tuning.main_max_tokens,
            fallback_model=models.fallback or None,
        )
    if role == ROLE_CLASSIFY:
        return RoleConfig(
            role=role,
            model=models.classifier,
            temperature=tuning.classifier_temperature,
            max_tokens=200,
            fallback_model=models.main or None,
            response_format={"type": "json_object"},
        )
    if role == ROLE_EVALUATE:
        return RoleConfig(
            role=role,
            model=models.evaluator,
            temperature=tuning.evaluator_temperature,
            max_tokens=400,
            fallback_model=models.classifier or None,
            response_format={"type": "json_object"},
        )
    raise ValueError(f"unknown LLM role: {role!r}")


@dataclass
class LLMResult:
    text: str
    usage: LLMUsage
    raw: Any = field(default=None, repr=False)


_client: Any | None = None
# Which provider `_client` was built for, so a provider change between calls
# cannot silently keep using a client pointed at the previous upstream.
_client_provider: str | None = None


def _openai_client() -> Any:
    """One shared OpenAI-protocol client, pointed at the active provider."""
    global _client, _client_provider
    provider = active_provider()
    if _client is not None and _client_provider == provider:
        return _client

    from openai import OpenAI

    if provider == PROVIDER_GEMINI:
        if not settings.gemini.configured:
            raise LLMNotConfigured(
                "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set — "
                "get a key at https://aistudio.google.com/apikey and put it in ai-core/.env"
            )
        client = OpenAI(
            api_key=settings.gemini.api_key,
            base_url=settings.gemini.base_url,
            timeout=settings.gemini.timeout_seconds,
            max_retries=1,
        )
    else:
        if not settings.openrouter.configured:
            raise LLMNotConfigured(
                "OPENROUTER_API_KEY is not set — copy ai-core/.env.example to ai-core/.env and fill it in"
            )
        # OpenRouter attributes traffic to an app through these headers; they
        # are meaningless to Google and are not sent to it.
        default_headers: dict[str, str] = {}
        if settings.openrouter.app_url:
            default_headers["HTTP-Referer"] = settings.openrouter.app_url
        if settings.openrouter.app_title:
            default_headers["X-Title"] = settings.openrouter.app_title

        client = OpenAI(
            api_key=settings.openrouter.api_key,
            base_url=settings.openrouter.base_url,
            timeout=settings.openrouter.timeout_seconds,
            max_retries=1,
            default_headers=default_headers or None,
        )

    _client, _client_provider = client, provider
    return _client


class LLM:
    """A model bound to a role, plus the accounting that must accompany it."""

    def __init__(self, config: RoleConfig) -> None:
        self.config = config

    @property
    def model(self) -> str:
        return self.config.model

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        session_id: str = "-",
        user_id: str = "-",
        request_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        request_id = request_id or new_request_id()
        attempts: list[tuple[str, bool]] = [(self.config.model, False)]
        if self.config.fallback_model and self.config.fallback_model != self.config.model:
            attempts.append((self.config.fallback_model, True))

        last_error: Exception | None = None
        for model, is_fallback in attempts:
            if is_fallback:
                log.warning(
                    "falling back to secondary model",
                    extra={"role": self.config.role, "from_model": self.config.model, "to_model": model},
                )
            try:
                return self._call(
                    model=model,
                    messages=messages,
                    session_id=session_id,
                    user_id=user_id,
                    request_id=request_id,
                    is_fallback=is_fallback,
                    temperature=self.config.temperature if temperature is None else temperature,
                    max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 — retried via fallback below
                last_error = exc
                log.warning(
                    "llm call failed",
                    extra={
                        "role": self.config.role,
                        "model": model,
                        "is_fallback": is_fallback,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:400],
                    },
                )

        raise LLMCallFailed(
            f"all models failed for role {self.config.role!r}: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _call(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str,
        request_id: str,
        is_fallback: bool,
        temperature: float,
        max_tokens: int,
    ) -> LLMResult:
        client = _openai_client()
        watch = Stopwatch()

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        provider = active_provider()
        if provider == PROVIDER_OPENROUTER:
            # Asks OpenRouter to include its own authoritative cost/usage.
            # Vendor-specific: Google's compatibility endpoint rejects unknown
            # body fields, so it must not be sent there.
            kwargs["extra_body"] = {"usage": {"include": True}}
        elif provider == PROVIDER_GEMINI and settings.gemini.reasoning_effort:
            # Without this, Gemini 2.5's hidden thinking tokens eat max_tokens
            # and the visible reply is truncated mid-sentence. See the comment
            # on GeminiSettings.reasoning_effort.
            kwargs["reasoning_effort"] = settings.gemini.reasoning_effort
        if self.config.response_format is not None:
            kwargs["response_format"] = self.config.response_format

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            usage_tracker.record(
                LLMUsage(
                    request_id=request_id,
                    session_id=session_id,
                    user_id=user_id,
                    role=self.config.role,
                    provider=active_provider(),
                    model=model,
                    latency_ms=watch.elapsed_ms(),
                    success=False,
                    timestamp=time.time(),
                    fallback_used=is_fallback,
                    error=f"{type(exc).__name__}: {exc}"[:500],
                )
            )
            raise

        latency_ms = watch.elapsed_ms()
        raw_usage = getattr(response, "usage", None)
        tokens = extract_usage(raw_usage)
        input_cost, output_cost, total_cost = calculate_cost(
            model, tokens["input_tokens"], tokens["output_tokens"]
        )

        choices = getattr(response, "choices", None) or []
        text = (choices[0].message.content or "").strip() if choices else ""

        usage = usage_tracker.record(
            LLMUsage(
                request_id=request_id,
                session_id=session_id,
                user_id=user_id,
                role=self.config.role,
                provider=active_provider(),
                model=getattr(response, "model", model) or model,
                latency_ms=latency_ms,
                success=True,
                timestamp=time.time(),
                input_cost_usd=input_cost,
                output_cost_usd=output_cost,
                total_cost_usd=total_cost,
                native_cost_usd=extract_native_cost(raw_usage),
                fallback_used=is_fallback,
                **tokens,
            )
        )

        log.info(
            "llm call complete",
            extra={
                "role": self.config.role,
                "model": usage.model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "latency_ms": latency_ms,
                "estimated_cost_usd": round(total_cost, 8),
                "fallback_used": is_fallback,
            },
        )

        if not text:
            raise LLMCallFailed(f"model {model} returned an empty message")

        return LLMResult(text=text, usage=usage, raw=response)


def get_llm(role: str) -> LLM:
    """The single entry point for obtaining a model. Never import a model id."""
    return LLM(_role_config(role))


def describe_roles() -> dict[str, dict[str, Any]]:
    """Used by test_connections.py and benchmarking to show active routing."""
    out: dict[str, dict[str, Any]] = {}
    for role in (ROLE_MAIN, ROLE_CLASSIFY, ROLE_EVALUATE):
        cfg = _role_config(role)
        in_price, out_price = settings.pricing.for_model(cfg.model)
        out[role] = {
            "provider": active_provider(),
            "model": cfg.model,
            "fallback": cfg.fallback_model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "price_per_1m": {"input": in_price, "output": out_price},
        }
    return out


__all__ = [
    "get_llm",
    "describe_roles",
    "LLM",
    "LLMResult",
    "LLMNotConfigured",
    "active_provider",
    "PROVIDER_OPENROUTER",
    "PROVIDER_GEMINI",
    "LLMCallFailed",
    "ROLE_MAIN",
    "ROLE_CLASSIFY",
    "ROLE_EVALUATE",
]
