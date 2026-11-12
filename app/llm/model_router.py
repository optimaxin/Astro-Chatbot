"""Which model answers a role, for this turn.

Deliberately a rule table, not a classifier. The intent classifier already
runs on every turn; routing reuses its label instead of adding a second model
call to decide which model to call.

Rules for the main answer, in order:

1. The turn carries an image      -> `main` (the vision model). The fallback
                                     is kept only if it is not known to be
                                     text-only, so a fallback can never turn
                                     an image question into a hallucination.
2. The intent is a "light" intent -> `light` (e.g. Nemotron Bolt), with
   and a light model is set          `main` as its fallback.
3. Everything else                -> `main`, with `fallback` behind it.

Classifier and judge roles keep their existing fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.llm.model_registry import get_spec
from app.llm.providers.base import RoleModels


@dataclass(frozen=True)
class Route:
    model: str
    fallback_model: str | None
    reason: str


def has_image(messages: list[dict[str, Any]] | None) -> bool:
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        ):
            return True
    return False


def route_main(models: RoleModels, *, intent: str | None = None, image: bool = False) -> Route:
    if image:
        # Prefer the main model; if it is known text-only, promote the first
        # candidate that is not, rather than sending pixels to a model that
        # will reject them (a 400) or ignore them (a hallucination).
        candidates = [m for m in (models.main, models.fallback, models.light) if m]
        capable = [m for m in dict.fromkeys(candidates) if get_spec(m).vision is not False]
        if not capable:
            return Route(models.main, None, "image:no-vision-model")
        return Route(capable[0], capable[1] if len(capable) > 1 else None, "image")

    if (
        intent
        and models.light
        and intent.lower() in models.light_intents
        and models.light != models.main
    ):
        return Route(models.light, models.main or None, f"light-intent:{intent.lower()}")

    return Route(models.main, models.fallback or None, "default")


__all__ = ["Route", "route_main", "has_image"]
