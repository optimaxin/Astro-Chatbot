"""Providers, models and pricing, loaded from config/models.json.

Pricing lives only here. A model with no pricing reports cost as None
("N/A" in the UI) — a cost is never invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.settings import settings

KNOWN_PARAMS = ("temperature", "top_p", "top_k", "max_tokens")


@dataclass(frozen=True)
class Pricing:
    input_per_1m: float
    output_per_1m: float
    currency: str = "USD"
    effective_from: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {"input_per_1m": self.input_per_1m, "output_per_1m": self.output_per_1m,
                "currency": self.currency, "effective_from": self.effective_from}


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    name: str
    base_url: str
    api_key_env: str
    supported_params: tuple[str, ...]
    extra_body: dict[str, Any] = field(default_factory=dict)
    # "openai" (OpenAI-compatible chat completions) or "aicore" (the Tredev chatbot's /chat).
    kind: str = "openai"
    # When set, the upstream URL is read from this env var instead of `base_url`.
    base_url_env: str = ""

    @property
    def resolved_base_url(self) -> str:
        return settings.api_key(self.base_url_env).rstrip("/") if self.base_url_env else self.base_url

    @property
    def configured(self) -> bool:
        key_ok = not self.api_key_env or bool(settings.api_key(self.api_key_env))
        return key_ok and bool(self.resolved_base_url)


@dataclass(frozen=True)
class ModelConfig:
    id: str
    provider: str
    label: str
    pricing: Pricing | None
    request_overrides: dict[str, Any] = field(default_factory=dict)
    # Non-OpenAI body fields this model needs (e.g. NVIDIA's chat_template_kwargs).
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Catalog:
    providers: dict[str, ProviderConfig]
    models: dict[tuple[str, str], ModelConfig]
    evaluator: tuple[str, str] | None

    def model(self, provider: str, model: str) -> ModelConfig | None:
        return self.models.get((provider, model))


def load_catalog(path: Path | None = None) -> Catalog:
    raw = json.loads((path or settings.models_file).read_text(encoding="utf-8"))
    providers = {
        p["id"]: ProviderConfig(
            id=p["id"], name=p["name"], base_url=p.get("base_url", ""), api_key_env=p.get("api_key_env", ""),
            supported_params=tuple(x for x in p.get("supported_params", []) if x in KNOWN_PARAMS),
            extra_body=p.get("extra_body") or {},
            kind=p.get("kind", "openai"), base_url_env=p.get("base_url_env", ""),
        )
        for p in raw["providers"]
    }
    models: dict[tuple[str, str], ModelConfig] = {}
    for m in raw["models"]:
        if m["provider"] not in providers:
            raise ValueError(f"model {m['id']!r} references unknown provider {m['provider']!r}")
        price = m.get("pricing")
        models[(m["provider"], m["id"])] = ModelConfig(
            id=m["id"], provider=m["provider"], label=m.get("label") or m["id"],
            pricing=Pricing(float(price["input_per_1m"]), float(price["output_per_1m"]),
                            price.get("currency", "USD"), price.get("effective_from")) if price else None,
            request_overrides=m.get("request_overrides") or {},
            extra_body=m.get("extra_body") or {},
        )
    ev = raw.get("evaluator")
    return Catalog(providers, models, (ev["provider"], ev["model"]) if ev else None)


def calculate_cost(pricing: Pricing | None, input_tokens: int | None,
                   output_tokens: int | None) -> dict[str, Any]:
    """Cost from actual token counts. Any unknown input yields None, not 0."""
    if pricing is None or input_tokens is None or output_tokens is None:
        return {"input_cost": None, "output_cost": None, "total_cost": None,
                "currency": pricing.currency if pricing else None}
    input_cost = input_tokens * pricing.input_per_1m / 1_000_000
    output_cost = output_tokens * pricing.output_per_1m / 1_000_000
    return {"input_cost": round(input_cost, 10), "output_cost": round(output_cost, 10),
            "total_cost": round(input_cost + output_cost, 10), "currency": pricing.currency}


_catalog: Catalog | None = None


def get_catalog() -> Catalog:
    global _catalog
    if _catalog is None:
        _catalog = load_catalog()
    return _catalog
