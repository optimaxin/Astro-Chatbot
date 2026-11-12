"""NVIDIA hosted inference (integrate.api.nvidia.com), OpenAI-compatible."""

from __future__ import annotations

from typing import Any

from app.llm.model_registry import ModelSpec
from app.llm.providers.base import ProviderAdapter, RoleModels, merge_extra_body


class NvidiaProvider(ProviderAdapter):
    name = "nvidia"

    def configured(self, settings: Any) -> bool:
        return settings.nvidia.configured

    def missing_key_message(self) -> str:
        return (
            "LLM_PROVIDER=nvidia but NVIDIA_API_KEY is not set — create a key at "
            "https://build.nvidia.com (API key, starts with 'nvapi-') and put it in ai-core/.env"
        )

    def client_kwargs(self, settings: Any) -> dict[str, Any]:
        return {
            "api_key": settings.nvidia.api_key,
            "base_url": settings.nvidia.base_url,
            "timeout": settings.nvidia.timeout_seconds,
        }

    def role_models(self, settings: Any) -> RoleModels:
        n = settings.nvidia
        return RoleModels(
            main=n.main,
            fallback=n.fallback,
            classifier=n.classifier,
            evaluator=n.evaluator,
            light=n.light_model,
            light_intents=n.light_intents,
            benchmark=n.benchmark,
        )

    def prepare(
        self, kwargs: dict[str, Any], spec: ModelSpec, settings: Any, *, streaming: bool
    ) -> dict[str, Any]:
        merge_extra_body(kwargs, spec.extra_body)
        if streaming and spec.stream_usage is not False:
            # Without this the stream carries no usage block and every streamed
            # call would be recorded as "tokens unavailable".
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs


__all__ = ["NvidiaProvider"]
