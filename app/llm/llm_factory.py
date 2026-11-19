"""The LLM gateway — the only place in the application that talks to an LLM.

Graph nodes ask for a *role*, never a model:

    llm = get_llm(role="job-main-turn", intent=state["intent"])
    result = llm.complete(messages, session_id=..., user_id=..., request_id=...)

    Layer                    Module
    ─────────────────────    ──────────────────────────────
    gateway (this file)      retries, fallback, accounting
    routing                  model_router.py
    capabilities             model_registry.py
    reliability              reliability.py (errors, backoff, circuit)
    providers                providers/{nvidia,openrouter,gemini}.py
    token tracking           usage_tracker.py -> usage_ledger.py
    streaming                streaming.py

`LLM_PROVIDER` picks the provider (default `nvidia`); OpenRouter and Gemini
remain fully wired and are switched to by changing that one variable.

Every call returns an `LLMResult` carrying the text *and* a fully populated
`LLMUsage` record, so token/latency tracking cannot be forgotten at a call
site. Every attempt — including a failed retry — writes its own usage row.
"""

from __future__ import annotations

import dataclasses
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.config.settings import settings
from app.llm import reliability
from app.llm.model_registry import get_spec, spec_for_call
from app.llm.model_router import route_main
from app.llm.providers import DEFAULT_PROVIDER, PROVIDERS, get_provider
from app.llm.providers.base import RoleModels
from app.llm.reliability import ErrorInfo, GatewayError, breaker, classify_error
from app.llm.request_builder import (
    build_request as _build_request,
    learn_from_bad_request,
    parse_tool_calls as _parse_tool_calls,
    strip_think,
)
from app.llm.usage_tracker import (
    LLMUsage,
    Stopwatch,
    calculate_cost,
    extract_native_cost,
    extract_usage,
    new_request_id,
    token_accounting,
    usage_tracker,
)
from app.logging_setup import get_logger
from app.observability.trace import span

log = get_logger("llm")

PROVIDER_NVIDIA = "nvidia"
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_GEMINI = "gemini"
# Not a value of LLM_PROVIDER — only ever used as a role's fallback_provider.
PROVIDER_ANTHROPIC = "anthropic"


def active_provider() -> str:
    """Which upstream the roles resolve against, per `LLM_PROVIDER`.

    An unrecognised value falls back to the default provider rather than
    failing at import, so a typo degrades instead of taking the service down.
    """
    provider = settings.llm_provider
    return provider if provider in PROVIDERS else DEFAULT_PROVIDER


def provider_configured(provider: str | None = None) -> bool:
    return get_provider(provider or active_provider()).configured(settings)


# Role names are the public API of this module.
ROLE_MAIN = "job-main-turn"
ROLE_CLASSIFY = "job-classify-intent"
ROLE_EVALUATE = "job-evaluate-response"
ROLE_QUALITY = "job-quality-check"
ROLE_HUMANIZE = "job-humanize"
# Blind A/B comparison of a humanizer rewrite against its original (shadow clean-reply sample). Same
# independent model as the quality checker, billed under its own role.
ROLE_HUMANIZE_COMPARE = "job-humanize-compare"


class LLMNotConfigured(RuntimeError):
    pass


class LLMCallFailed(RuntimeError):
    """Raised only when the primary *and* any permitted fallback failed."""

    def __init__(self, message: str, info: ErrorInfo | None = None) -> None:
        super().__init__(message)
        self.info = info


@dataclass
class RoleConfig:
    role: str
    model: str
    temperature: float
    max_tokens: int
    fallback_model: str | None = None
    response_format: dict[str, Any] | None = None
    # "same" (default) retries on the active provider's own client.
    fallback_provider: str = "same"
    route_reason: str = "default"
    # This LLM IS the role's fallback model, called directly (see `fallback_llm`):
    # its single attempt is metered and logged as a fallback call.
    pinned_fallback: bool = False
    # A call whose failure costs nothing (the concise rewrite keeps the original):
    # ONE attempt, no retries, and its failures never count toward opening the
    # model's circuit — background rewrite 429s were able to open Lightning's
    # circuit and push users' main answers to the fallback (2026-11-19). An open
    # circuit is still respected.
    optional: bool = False


def _models_for_provider() -> RoleModels:
    """Model ids for the active provider. No node, prompt or test refers to a
    model id; providers name the same model differently."""
    return get_provider(active_provider()).role_models(settings)


def _role_config(role: str, *, intent: str | None = None, image: bool = False) -> RoleConfig:
    models = _models_for_provider()
    tuning = settings.models
    if role == ROLE_MAIN:
        if active_provider() == PROVIDER_GEMINI and settings.anthropic.configured:
            return RoleConfig(
                role=role,
                model=models.main,
                temperature=tuning.main_temperature,
                max_tokens=tuning.main_max_tokens,
                fallback_model=settings.anthropic.model,
                fallback_provider=PROVIDER_ANTHROPIC,
            )
        route = route_main(models, intent=intent, image=image)
        return RoleConfig(
            role=role,
            model=route.model,
            temperature=tuning.main_temperature,
            max_tokens=tuning.main_max_tokens,
            fallback_model=route.fallback_model,
            route_reason=route.reason,
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
            # The judge's own model is also the classifier's (Bolt on NVIDIA), so the
            # old fallback (`models.classifier`) was the SAME model: the gateway
            # skipped it and a 429 on Bolt meant no second chance (36% of judge calls
            # failed historically). The second model is the main one.
            fallback_model=(models.main if models.main != models.evaluator else models.classifier) or None,
            response_format={"type": "json_object"},
        )
    if role == ROLE_QUALITY:
        # Independence rule: the checker must not be the model that wrote the reply, so it has
        # NO fallback. If it fails the verdict is UNAVAILABLE, never "the author approves itself".
        if not models.quality:
            raise LLMNotConfigured("the quality-checker role has no model: set NVIDIA_QUALITY_CHECKER_MODEL")
        from app.prompts.roles.quality_checker import QUALITY_MAX_TOKENS, QUALITY_TEMPERATURE

        return RoleConfig(role=role, model=models.quality, temperature=QUALITY_TEMPERATURE,
                          max_tokens=QUALITY_MAX_TOKENS, fallback_model=None,
                          response_format={"type": "json_object"})
    if role == ROLE_HUMANIZE:
        # Failure keeps the original reply, so there is nothing to fall back to.
        from app.prompts.roles.humanizer import HUMANIZER_MAX_TOKENS, HUMANIZER_TEMPERATURE

        return RoleConfig(role=role, model=models.humanizer or models.main, temperature=HUMANIZER_TEMPERATURE,
                          max_tokens=HUMANIZER_MAX_TOKENS, fallback_model=None)
    if role == ROLE_HUMANIZE_COMPARE:
        if not models.quality:
            raise LLMNotConfigured("the humanizer comparison needs a judge model: set NVIDIA_QUALITY_CHECKER_MODEL")
        from app.prompts.roles.humanizer import COMPARE_MAX_TOKENS

        return RoleConfig(role=role, model=models.quality, temperature=0.0, max_tokens=COMPARE_MAX_TOKENS,
                          fallback_model=None, response_format={"type": "json_object"})
    raise ValueError(f"unknown LLM role: {role!r}")


@dataclass
class LLMResult:
    text: str
    usage: LLMUsage
    raw: Any = field(default=None, repr=False)
    # [{"id", "name", "arguments" (raw JSON string), "parsed_arguments"}]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None


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
    adapter = get_provider(provider)
    if not adapter.configured(settings):
        raise LLMNotConfigured(adapter.missing_key_message())
    _client, _client_provider = adapter.build_client(settings), provider
    return _client



def build_request(**kwargs: Any) -> dict[str, Any]:
    """`request_builder.build_request` bound to this module's settings."""
    return _build_request(settings=settings, **kwargs)


def _deadline() -> float:
    return time.monotonic() + settings.reliability.deadline_seconds


class LLM:
    """A model bound to a role, plus the accounting that must accompany it."""

    def __init__(self, config: RoleConfig) -> None:
        self.config = config
        # kwargs of the most recent OpenAI-protocol request, so a 400 can be
        # matched to the optional parameter that caused it.
        self._last_request: dict[str, Any] = {}

    @property
    def model(self) -> str:
        return self.config.model

    def attempts(self) -> list[tuple[str, bool, str]]:
        out = [(self.config.model, self.config.pinned_fallback, "same")]
        if self.config.fallback_model and self.config.fallback_model != self.config.model:
            out.append((self.config.fallback_model, True, self.config.fallback_provider))
        return out

    def provider_for(self, provider_override: str) -> str:
        return PROVIDER_ANTHROPIC if provider_override == PROVIDER_ANTHROPIC else active_provider()

    def may_fall_back(self, info: ErrorInfo | None, provider_override: str) -> bool:
        if info is None:
            return True
        if info.code == reliability.CANCELLED:
            return False
        # A different provider has its own key and quota, so even an auth or
        # request error on the primary is worth one try there.
        return info.fallback_ok or provider_override != "same"

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResult:
        """See `_complete`. Every usage row written during the call carries this
        role's routing decision (`route`)."""
        from app.llm.usage_tracker import reset_route, set_route

        token = set_route(self.config.route_reason)
        try:
            return self._complete(messages, **kwargs)
        finally:
            reset_route(token)

    def _complete(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str = "-",
        user_id: str = "-",
        request_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResult:
        request_id = request_id or new_request_id()
        deadline = _deadline()
        common = dict(
            messages=messages, session_id=session_id, user_id=user_id, request_id=request_id,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            tools=tools, tool_choice=tool_choice,
            response_format=response_format if response_format is not None else self.config.response_format,
        )

        hedge_after = settings.reliability.hedge_after_seconds
        # Hedging races the fallback against a slow PRIMARY ANSWER. It is not for
        # repair, language rewrite, grammar correction, the validation fallback, the
        # classifier or the judge: racing those doubled their calls and tokens, and a
        # hedged repair could be won by the light model that wrote the problem
        # (role audit 2026-11-18). Unlabelled main-role calls are not hedged.
        if (hedge_after > 0 and self.config.role == ROLE_MAIN and hedgeable(self.config)
                and self.config.fallback_model and self.config.fallback_provider == "same"
                and self.config.fallback_model != self.config.model):
            from app.llm.hedging import hedged_complete

            try:
                return hedged_complete(self, hedge_after=hedge_after, deadline=deadline, common=common)
            except Exception as exc:  # noqa: BLE001 — both legs failed: report as usual
                info = classify_error(exc)
                raise LLMCallFailed(
                    f"all models failed for role {self.config.role!r}: {type(exc).__name__}: {exc}",
                    info) from exc

        last_error: Exception | None = None
        last_info: ErrorInfo | None = None
        for model, is_fallback, provider_override in self.attempts():
            if is_fallback and not self.config.pinned_fallback:
                if not self.may_fall_back(last_info, provider_override):
                    break
                if time.monotonic() >= deadline:
                    break
                log.warning("falling back to secondary model", extra={
                    "role": self.config.role, "from_model": self.config.model, "to_model": model,
                    "to_provider": provider_override,
                    "reason": last_info.label() if last_info else None,
                })
            try:
                return self._with_retries(
                    model=model, is_fallback=is_fallback, provider_override=provider_override,
                    deadline=deadline, primary_error=last_info.label() if last_info else None,
                    **common,
                )
            except Exception as exc:  # noqa: BLE001 — classified, then fallback
                last_error, last_info = exc, classify_error(exc)
                log.warning("llm call failed", extra={
                    "role": self.config.role, "model": model, "is_fallback": is_fallback,
                    "error_code": last_info.label(), "error_type": type(exc).__name__,
                    "error_message": str(exc)[:400],
                })

        raise LLMCallFailed(
            f"all models failed for role {self.config.role!r}: "
            f"{type(last_error).__name__}: {last_error}",
            last_info,
        ) from last_error

    def stream(self, messages: list[dict[str, Any]], **kwargs: Any):
        """Token stream with the same routing, retry and accounting rules.
        See `app.llm.streaming.LLMStream`."""
        from app.llm.streaming import LLMStream

        return LLMStream(self, messages, **kwargs)

    def _with_retries(self, *, model: str, provider_override: str, deadline: float,
                      **kwargs: Any) -> LLMResult:
        provider = self.provider_for(provider_override)
        key = (provider, model)
        cfg = settings.reliability
        if not breaker.allow(key, cfg.circuit_cooldown_seconds):
            raise GatewayError(reliability.CIRCUIT_OPEN, f"circuit open for {provider}/{model}")

        attempt = 0
        downgrades = 0
        while True:
            try:
                result = self._call(model=model, provider_override=provider_override,
                                    attempt=attempt, **kwargs)
                breaker.record_success(key)
                return result
            except Exception as exc:
                info = classify_error(exc)
                sent = getattr(self, "_last_request", None) or {}
                if info.code == reliability.BAD_REQUEST and downgrades < 4 and \
                        learn_from_bad_request(model, info, sent):
                    downgrades += 1
                    continue
                if info.trips_circuit and not self.config.optional:
                    if breaker.record_failure(key, cfg.circuit_failure_threshold):
                        log.error("circuit opened", extra={"provider": provider, "model": model,
                                                           "error_code": info.label()})
                else:
                    breaker.release_trial(key)
                if not info.retryable or attempt >= (0 if self.config.optional else cfg.max_retries):
                    raise
                delay = reliability.backoff_delay(
                    attempt + 1, base=cfg.backoff_base_seconds, cap=cfg.backoff_max_seconds,
                    retry_after=info.retry_after,
                )
                if time.monotonic() + delay >= deadline:
                    raise
                log.info("retrying llm call", extra={"model": model, "attempt": attempt + 1,
                                                     "delay_s": round(delay, 2),
                                                     "error_code": info.label()})
                reliability.sleep(delay)
                attempt += 1

    def _record_failure(self, *, provider: str, model: str, watch: Stopwatch, exc: Exception,
                        request_id: str, session_id: str, user_id: str, is_fallback: bool,
                        attempt: int, streamed: bool = False) -> None:
        usage_tracker.record(LLMUsage(
            request_id=request_id, session_id=session_id, user_id=user_id,
            role=self.config.role, provider=provider, model=model, requested_model=model,
            latency_ms=watch.elapsed_ms(), success=False, timestamp=time.time(),
            fallback_used=is_fallback, error=f"{type(exc).__name__}: {exc}"[:500],
            error_code=classify_error(exc).label(), attempt=attempt, streamed=streamed,
        ))

    def _call(
        self, *, model: str, messages: list[dict[str, Any]], session_id: str, user_id: str,
        request_id: str, is_fallback: bool, provider_override: str, temperature: float,
        max_tokens: int, attempt: int = 0, primary_error: str | None = None,
        tools: list[dict[str, Any]] | None = None, tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResult:
        if provider_override == PROVIDER_ANTHROPIC:
            from app.llm.providers.anthropic_fallback import call_anthropic

            return call_anthropic(
                self, settings, model=model, messages=messages, session_id=session_id, user_id=user_id,
                request_id=request_id, is_fallback=is_fallback, temperature=temperature,
                max_tokens=max_tokens, primary_error=primary_error,
            )

        provider = active_provider()
        if provider == "nvidia":
            # Reasoning only for the main answer of NVIDIA_MAIN_REASONING_MODELS.
            spec, extra_tokens = spec_for_call(model, main_turn=self.config.role == ROLE_MAIN, settings=settings)
            max_tokens += extra_tokens
        else:
            spec = get_spec(model)
        # Refusals here (unsupported capability, missing key) happen before
        # any upstream call, so they write no usage row.
        kwargs = build_request(
            provider=provider, spec=spec, messages=messages, temperature=temperature,
            max_tokens=max_tokens, response_format=response_format, tools=tools,
            tool_choice=tool_choice, streaming=False,
        )
        self._last_request = kwargs
        client = _openai_client()
        watch = Stopwatch()
        try:
            with span(f"llm.{self.config.role}"):
                response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            self._record_failure(provider=provider, model=model, watch=watch, exc=exc,
                                 request_id=request_id, session_id=session_id, user_id=user_id,
                                 is_fallback=is_fallback, attempt=attempt)
            raise

        latency_ms = watch.elapsed_ms()
        choices = getattr(response, "choices", None) or []
        message = choices[0].message if choices else None
        text = (getattr(message, "content", None) or "") if message is not None else ""
        text = strip_think(text).strip() if spec.strips_think_tags else text.strip()
        tool_calls = _parse_tool_calls(message) if message is not None else []
        finish_reason = getattr(choices[0], "finish_reason", None) if choices else None

        raw_usage = getattr(response, "usage", None)
        tokens = extract_usage(raw_usage)
        input_cost, output_cost, total_cost = calculate_cost(
            model, tokens["input_tokens"], tokens["output_tokens"]
        )
        from app.llm.usage_tracker import current_purpose

        # Only prose answers are checked: a main-role call that is ASKED for JSON
        # (the grammar correction) was rejected as malformed on both models
        # (role audit 2026-11-18), silently disabling that layer.
        malformed = (self.config.role == ROLE_MAIN and not response_format and not tool_calls
                     and current_purpose() not in JSON_PURPOSES and looks_malformed(text))
        ok = bool(text or tool_calls) and not malformed
        failure_code = reliability.MALFORMED if malformed else reliability.EMPTY
        usage = usage_tracker.record(LLMUsage(
            request_id=request_id, session_id=session_id, user_id=user_id,
            role=self.config.role, provider=provider,
            model=getattr(response, "model", model) or model,
            requested_model=model,
            latency_ms=latency_ms, success=ok, timestamp=time.time(),
            input_cost_usd=input_cost, output_cost_usd=output_cost, total_cost_usd=total_cost,
            native_cost_usd=extract_native_cost(raw_usage), fallback_used=is_fallback,
            error=None if ok else f"GatewayError: {failure_code}",
            error_code=primary_error if ok else failure_code,
            attempt=attempt, **tokens, **token_accounting(tokens, messages, text),
        ))

        log.info("llm call complete", extra={
            "role": self.config.role, "provider": provider, "model": usage.model,
            "route": self.config.route_reason, "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens, "token_source": usage.token_source,
            "latency_ms": latency_ms, "estimated_cost_usd": round(total_cost, 8),
            "fallback_used": is_fallback, "attempt": attempt, "tool_calls": len(tool_calls),
        })

        if malformed:
            raise GatewayError(reliability.MALFORMED, f"model {model} returned a non-prose answer "
                                                      f"(finish_reason={finish_reason})")
        if not text and not tool_calls:
            raise GatewayError(reliability.EMPTY, f"model {model} returned an empty message "
                                                  f"(finish_reason={finish_reason})")
        return LLMResult(text=text, usage=usage, raw=response, tool_calls=tool_calls,
                         finish_reason=finish_reason)


# Purposes that may be hedged against the fallback model: the primary answer only.
HEDGED_PURPOSES = frozenset({"answer"})


def hedgeable(config: "RoleConfig") -> bool:
    from app.llm.usage_tracker import current_purpose

    return current_purpose() in HEDGED_PURPOSES and not config.pinned_fallback


# Main-role purposes whose expected output IS JSON, so the prose check does not apply.
JSON_PURPOSES = frozenset({"grammar_correction"})
_NON_PROSE = re.compile(r"^\s*(?:[{\[]|```)|<\|(?:im_end|im_start|eot_id|end_of_text)\|>|</s>\s*$")


def looks_malformed(text: str) -> bool:
    """A main-turn answer is prose. A JSON/code fragment or leaked chat-template
    tokens is a broken generation and goes to the fallback model instead of the
    user (live E2E 2026-11-18: a truncated `{"status": "succ` was delivered).
    Length alone is never a reason: a greeting reply can be two words."""
    return bool(text) and bool(_NON_PROSE.search(text))


def get_llm(role: str, *, intent: str | None = None, has_image: bool = False) -> LLM:
    """The single entry point for obtaining a model. Never import a model id.

    `intent` and `has_image` only affect ROLE_MAIN, see `model_router`."""
    return LLM(_role_config(role, intent=intent, image=has_image))


def fallback_llm(role: str = ROLE_MAIN) -> LLM | None:
    """The role's fallback model (Bolt for the main answer), called directly:
    one model, no routing, no further fallback. None when the role has no
    fallback, or its fallback lives on another provider."""
    cfg = _role_config(role)
    if not cfg.fallback_model or cfg.fallback_model == cfg.model or cfg.fallback_provider != "same":
        return None
    return LLM(dataclasses.replace(cfg, model=cfg.fallback_model, fallback_model=None,
                                   route_reason="validation-fallback", pinned_fallback=True))


def llm_for_model(model: str, *, role: str = ROLE_MAIN, fallback_model: str | None = None,
                  temperature: float | None = None, max_tokens: int | None = None) -> LLM:
    """A specific model with no routing — for benchmarks and smoke tests,
    where the model under test must not be silently replaced."""
    tuning = settings.models
    return LLM(RoleConfig(
        role=role, model=model,
        temperature=tuning.main_temperature if temperature is None else temperature,
        max_tokens=tuning.main_max_tokens if max_tokens is None else max_tokens,
        fallback_model=fallback_model, route_reason="pinned",
    ))


def describe_roles() -> dict[str, dict[str, Any]]:
    """Used by test_connections.py, /health and benchmarking."""
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
    models = _models_for_provider()
    if models.light:
        out[ROLE_MAIN]["light_model"] = models.light
        out[ROLE_MAIN]["light_intents"] = list(models.light_intents)
    return out


__all__ = [
    "get_llm", "llm_for_model", "fallback_llm", "describe_roles", "LLM", "LLMResult", "LLMNotConfigured",
    "LLMCallFailed", "active_provider", "provider_configured", "build_request", "strip_think",
    "PROVIDER_NVIDIA", "PROVIDER_OPENROUTER", "PROVIDER_GEMINI", "PROVIDER_ANTHROPIC",
    "ROLE_MAIN", "ROLE_CLASSIFY", "ROLE_EVALUATE", "ROLE_QUALITY", "ROLE_HUMANIZE", "ROLE_HUMANIZE_COMPARE",
]
