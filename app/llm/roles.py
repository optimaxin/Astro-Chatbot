"""The five model roles, described from the live configuration.

One place that answers "which model does what, with which prompt and limits":
role name, provider, exact model id, fallback, prompt and version, temperature,
output-token limit, timeout and retry policy, and whether the role is switched on.
Everything is read from the same settings the gateway uses (nothing is
duplicated here), so this can never disagree with what actually runs; it is
reported by `/health` and used by the role audit tests.

    intent_classifier   Bolt        structured intent, scope and follow-up detection
    primary_response    Lightning   the user-facing astrology answer
    fallback_response   Bolt        independent second model when the primary fails
    quality_checker     Ultra       independent advisory evaluator, shadow mode, 10% sample
    humanizer           Lightning   language polish, shadow mode (records, never applies)
"""

from __future__ import annotations

from typing import Any

from app.config.settings import settings


_UNSET: Any = object()


def _cfg(role: str) -> Any:
    from app.llm.llm_factory import LLMNotConfigured, _role_config

    try:
        return _role_config(role)
    except LLMNotConfigured:
        return None


def describe_five_roles() -> list[dict[str, Any]]:
    from app.llm.llm_factory import (ROLE_CLASSIFY, ROLE_EVALUATE, ROLE_HUMANIZE, ROLE_MAIN, ROLE_QUALITY,
                                     active_provider)
    from app.llm.usage_tracker import prompt_version_of
    from app.nodes.classifier import CLASSIFIER_SYSTEM_PROMPT
    from app.prompts.builder import PROMPT_VERSION
    from app.prompts.roles.humanizer import HUMANIZER_PROMPT_VERSION
    from app.prompts.roles.quality_checker import QUALITY_MAX_TOKENS, QUALITY_PROMPT_VERSION, QUALITY_TEMPERATURE

    provider, rel, flags = active_provider(), settings.reliability, settings.flags
    timeout = getattr(settings, provider, None) and getattr(getattr(settings, provider), "timeout_seconds", None)
    policy = {"timeout_seconds": timeout, "max_retries": rel.max_retries, "backoff_base_seconds": rel.backoff_base_seconds,
              "deadline_seconds": rel.deadline_seconds}
    main, cls = _cfg(ROLE_MAIN), _cfg(ROLE_CLASSIFY)
    quality, human, judge = _cfg(ROLE_QUALITY), _cfg(ROLE_HUMANIZE), _cfg(ROLE_EVALUATE)

    def row(name: str, gateway_role: str, cfg: Any, prompt: tuple[str, str], *, model: str | None = None,
            fallback: Any = _UNSET, temperature: float | None = None, max_tokens: int | None = None,
            mode: str, enabled: bool, purposes: tuple[str, ...], notes: str = "",
            affects_reply: bool = True) -> dict:
        return {"role": name, "gateway_role": gateway_role, "provider": provider,
                "model": model if model is not None else (cfg.model if cfg else None),
                "fallback_model": (cfg.fallback_model if cfg else None) if fallback is _UNSET else fallback,
                "prompt": {"name": prompt[0], "version": prompt[1]},
                "temperature": cfg.temperature if cfg else temperature,
                "max_output_tokens": cfg.max_tokens if cfg else max_tokens,
                **policy, "mode": mode, "enabled": enabled, "affects_reply": affects_reply,
                "purposes": list(purposes), "notes": notes}

    return [
        row("intent_classifier", ROLE_CLASSIFY, cls, ("classifier", prompt_version_of(CLASSIFIER_SYSTEM_PROMPT)),
            mode=flags.classifier_mode, enabled=True, purposes=("classify",),
            notes="deterministic keyword and off-topic rules run first; the model is called only when they cannot decide"),
        row("primary_response", ROLE_MAIN, main, ("main-system-prompt", PROMPT_VERSION), mode="answer", enabled=True,
            purposes=("answer", "repair", "language_rewrite", "grammar_correction"),
            notes="smalltalk routes to the light model; hedging applies to the answer call only"),
        row("fallback_response", ROLE_MAIN, main, ("main-system-prompt", PROMPT_VERSION),
            model=main.fallback_model if main else None, fallback=None, mode="on technical failure or failed validation",
            enabled=bool(main and main.fallback_model), purposes=("answer", "validation_fallback"),
            notes="same prompt and verified chart facts as the primary; its answer passes the same validation"),
        row("quality_checker", ROLE_QUALITY, quality, ("quality-checker", QUALITY_PROMPT_VERSION),
            model=settings.nvidia.quality_checker or None, fallback=None, temperature=QUALITY_TEMPERATURE,
            max_tokens=QUALITY_MAX_TOKENS, mode=flags.quality_checker_mode, affects_reply=False,
            enabled=flags.quality_checker_mode != "off" and bool(settings.nvidia.quality_checker),
            purposes=("quality_check",),
            notes=("advisory shadow mode, sampled, after delivery; no fallback by design (independence); "
                   "NOT configured" if not settings.nvidia.quality_checker else
                   f"advisory shadow mode, {flags.quality_checker_sample_rate:.0%} sample, background, after delivery; no fallback by design (independence)")),
        row("humanizer", ROLE_HUMANIZE, human, ("humanizer", HUMANIZER_PROMPT_VERSION), mode=flags.humanizer_mode,
            enabled=flags.humanizer_mode != "off", purposes=("humanize",), affects_reply=flags.humanizer_mode == "on",
            notes="runs only on replies that still have language issues; a rewrite that changes a fact is discarded"),
        row("legacy_judge", ROLE_EVALUATE, judge, ("judge", "evaluator-1"), mode="sampled, async, advisory",
            enabled=settings.enable_evaluation, purposes=("evaluate",), affects_reply=False,
            notes="the original sampled judge; superseded by quality_checker once that is promoted"),
    ]


__all__ = ["describe_five_roles"]
