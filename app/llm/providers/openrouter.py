"""OpenRouter. Not the default during the NVIDIA beta, but fully wired:
set `LLM_PROVIDER=openrouter` to switch every role back to it."""

from __future__ import annotations

from typing import Any

from app.llm.model_registry import ModelSpec
from app.llm.providers.base import ProviderAdapter, RoleModels, merge_extra_body


class OpenRouterProvider(ProviderAdapter):
    name = "openrouter"

    def configured(self, settings: Any) -> bool:
        return settings.openrouter.configured

    def missing_key_message(self) -> str:
        return "OPENROUTER_API_KEY is not set — copy ai-core/.env.example to ai-core/.env and fill it in"

    def client_kwargs(self, settings: Any) -> dict[str, Any]:
        o = settings.openrouter
        # OpenRouter attributes traffic to an app through these headers; they
        # are meaningless to other providers and are not sent to them.
        headers: dict[str, str] = {}
        if o.app_url:
            headers["HTTP-Referer"] = o.app_url
        if o.app_title:
            headers["X-Title"] = o.app_title
        return {
            "api_key": o.api_key,
            "base_url": o.base_url,
            "timeout": o.timeout_seconds,
            "default_headers": headers or None,
        }

    def role_models(self, settings: Any) -> RoleModels:
        m = settings.models
        return RoleModels(
            main=m.main, fallback=m.fallback, classifier=m.classifier, evaluator=m.evaluator
        )

    def prepare(
        self, kwargs: dict[str, Any], spec: ModelSpec, settings: Any, *, streaming: bool
    ) -> dict[str, Any]:
        # Asks OpenRouter to include its own authoritative cost/usage.
        # Vendor-specific: other endpoints reject unknown body fields.
        return merge_extra_body(kwargs, {"usage": {"include": True}})


__all__ = ["OpenRouterProvider"]
