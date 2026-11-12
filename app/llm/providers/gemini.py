"""Google Gemini through its OpenAI-compatible `/v1beta/openai/` endpoint."""

from __future__ import annotations

from typing import Any

from app.llm.model_registry import ModelSpec
from app.llm.providers.base import ProviderAdapter, RoleModels


class GeminiProvider(ProviderAdapter):
    name = "gemini"

    def configured(self, settings: Any) -> bool:
        return settings.gemini.configured

    def missing_key_message(self) -> str:
        return (
            "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set — "
            "get a key at https://aistudio.google.com/apikey and put it in ai-core/.env"
        )

    def client_kwargs(self, settings: Any) -> dict[str, Any]:
        g = settings.gemini
        return {"api_key": g.api_key, "base_url": g.base_url, "timeout": g.timeout_seconds}

    def role_models(self, settings: Any) -> RoleModels:
        g = settings.gemini
        return RoleModels(
            main=g.main, fallback=g.fallback, classifier=g.classifier, evaluator=g.evaluator
        )

    def prepare(
        self, kwargs: dict[str, Any], spec: ModelSpec, settings: Any, *, streaming: bool
    ) -> dict[str, Any]:
        # Without this, Gemini 2.5's hidden thinking tokens eat max_tokens and
        # the visible reply is truncated mid-sentence. See
        # GeminiSettings.reasoning_effort.
        if settings.gemini.reasoning_effort:
            kwargs["reasoning_effort"] = settings.gemini.reasoning_effort
        return kwargs


__all__ = ["GeminiProvider"]
