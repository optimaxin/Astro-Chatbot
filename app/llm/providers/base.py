"""The contract every OpenAI-protocol provider implements.

A provider adapter owns exactly the things that differ between upstreams:

* which credential and base URL to use, and the error when they are missing
* which model id each role resolves to
* vendor-only request fields (OpenRouter's usage accounting, Gemini's
  reasoning switch, NVIDIA's chat-template kwargs)

Everything else — retries, fallback, circuit breaking, usage accounting,
streaming, tool calls, structured output — lives once in the gateway
(`app.llm.llm_factory`) and is identical for every provider. That is the
property that lets `LLM_PROVIDER` switch upstream without touching a node,
a prompt, RAG, the token meter or the frontend.

Adapters take `settings` as an argument rather than importing it, so a test
(or the benchmark) can hand one a modified copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.llm.model_registry import ModelSpec


@dataclass(frozen=True)
class RoleModels:
    """The model ids a provider supplies, normalised across providers."""

    main: str
    fallback: str
    classifier: str
    evaluator: str
    # Optional: answers "simple" intents instead of `main`. Empty = disabled.
    light: str = ""
    light_intents: tuple[str, ...] = ()
    benchmark: str = ""
    # Independent quality checker (shadow evaluator). Empty = the role is not configured.
    quality: str = ""
    # Humanizer. Empty = use `main` with the humanizer's own prompt and settings.
    humanizer: str = ""


class ProviderAdapter:
    name: str = "base"

    def configured(self, settings: Any) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def missing_key_message(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def client_kwargs(self, settings: Any) -> dict[str, Any]:  # pragma: no cover - interface
        """api_key / base_url / timeout / headers for `openai.OpenAI(...)`."""
        raise NotImplementedError

    def role_models(self, settings: Any) -> RoleModels:  # pragma: no cover - interface
        raise NotImplementedError

    def prepare(
        self, kwargs: dict[str, Any], spec: ModelSpec, settings: Any, *, streaming: bool
    ) -> dict[str, Any]:
        """Add vendor-specific fields to a chat.completions request. Pure."""
        return kwargs

    def build_client(self, settings: Any) -> Any:
        from openai import OpenAI

        # max_retries=0: the gateway owns retries (see ReliabilitySettings),
        # so every attempt is visible in the usage ledger and none is doubled.
        return OpenAI(max_retries=0, **self.client_kwargs(settings))


def merge_extra_body(kwargs: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    if not extra:
        return kwargs
    body = dict(kwargs.get("extra_body") or {})
    body.update(extra)
    kwargs["extra_body"] = body
    return kwargs


__all__ = ["ProviderAdapter", "RoleModels", "merge_extra_body"]
