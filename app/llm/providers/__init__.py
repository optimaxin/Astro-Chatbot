"""Provider adapters, looked up by the `LLM_PROVIDER` value."""

from __future__ import annotations

from app.llm.providers.base import ProviderAdapter, RoleModels
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.nvidia import NvidiaProvider
from app.llm.providers.openrouter import OpenRouterProvider

PROVIDERS: dict[str, ProviderAdapter] = {
    p.name: p for p in (NvidiaProvider(), OpenRouterProvider(), GeminiProvider())
}

# What an empty or unrecognised LLM_PROVIDER resolves to.
DEFAULT_PROVIDER = "nvidia"


def get_provider(name: str) -> ProviderAdapter:
    return PROVIDERS.get(name) or PROVIDERS[DEFAULT_PROVIDER]


__all__ = ["PROVIDERS", "DEFAULT_PROVIDER", "get_provider", "ProviderAdapter", "RoleModels"]
