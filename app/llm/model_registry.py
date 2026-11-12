"""What each model can be asked to do.

Models behind the same OpenAI-compatible endpoint do not accept the same
request: one rejects `response_format`, another has no tool calling, a third
"thinks" before answering unless told not to. Sending an unsupported
parameter is a 400, so the gateway consults this registry *before* building
a request and drops or adapts what a model cannot take.

Two layers:

1. `_KNOWN` — static entries for the models this service routes to. Values
   marked `None` are not yet verified against the live endpoint; the gateway
   treats them as "try it, and learn from a 400" (see `learn_unsupported`).
2. Runtime learning — when a provider rejects a parameter the gateway
   records it here, so the next call to that model omits it instead of
   failing again. Process-local and reset on restart, on purpose: a provider
   adding support should not need a code change to be picked up.

`scripts/probe_models.py` verifies every entry against the real endpoint and
prints the table to paste back here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model: str
    # None = unverified: the request is attempted and a 400 teaches us.
    json_mode: bool | None = None          # response_format={"type": "json_object"}
    json_schema: bool | None = None        # response_format={"type": "json_schema", ...}
    tools: bool | None = None              # tools / tool_choice
    vision: bool | None = None             # image_url content parts
    streaming: bool = True
    stream_usage: bool | None = None       # stream_options={"include_usage": True}
    temperature: bool = True
    # Provider-specific body fields always sent for this model, e.g.
    # chat-template switches that turn hidden reasoning off.
    extra_body: dict[str, Any] = field(default_factory=dict)
    # A model that emits <think>...</think> inline even when asked not to;
    # the gateway strips that block from the visible reply.
    strips_think_tags: bool = False
    notes: str = ""


# Verified against integrate.api.nvidia.com on 2026-11-12 with
# scripts/smoke_nvidia.py plus targeted probes. `None` = still unverified.
_KNOWN: dict[str, ModelSpec] = {
    "mistralai/mistral-large-4-1t-a52b": ModelSpec(
        model="mistralai/mistral-large-4-1t-a52b",
        # Reasons by default: with no switch, 399 of 400 completion tokens were
        # hidden reasoning and the visible reply was empty. chat_template
        # `enable_thinking=false` has no effect and reasoning_effort="none" is
        # a 400; "low" is the lowest accepted value.
        extra_body={"reasoning_effort": "low"},
        notes="Live 2026-11-12: ~60s queue + ~2 tok/s; one sentence took 139s "
              "even at reasoning_effort=low. json/tools/vision unverified (timeouts).",
    ),
    "nvidia/nemotron-3.5-bolt-e4b": ModelSpec(
        model="nvidia/nemotron-3.5-bolt-e4b",
        json_mode=True, tools=True, vision=True, stream_usage=True,
        # Accepted (verified); keeps any template reasoning out of max_tokens.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        strips_think_tags=True,
        notes="Live: <1s. Tool call args correct. Replied in Devanagari to a "
              "Hinglish prompt.",
    ),
    "nvidia/nemotron-3.5-lightning-30b-a3b": ModelSpec(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        json_mode=True, tools=True, stream_usage=True,
        # 400: "multimodal processing is not enabled".
        vision=False,
        # REQUIRED: without it all 300 completion tokens were reasoning and the
        # visible reply was empty; with it ~1s and a normal answer.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        strips_think_tags=True,
        notes="Live: ~1-2s with thinking off. Text only.",
    ),
}

_learned: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def get_spec(model: str) -> ModelSpec:
    """The spec for `model`, with anything learned at runtime applied.

    Unknown models get a permissive default (every optional capability
    unverified), which reproduces exactly how this service behaved before the
    registry existed: send the request as asked and let the provider decide.
    """
    base = _KNOWN.get(model)
    if base is None and model.startswith("nvidia/nemotron-3"):
        # Every Nemotron 3.x probed so far reasons by default and spends the
        # whole max_tokens budget on it (verified for Bolt, Lightning and the
        # super-vl judge: empty reply, finish_reason=length). Off unless an
        # explicit entry above says otherwise.
        base = ModelSpec(model=model, extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                         strips_think_tags=True, notes="nemotron-3 default: thinking off")
    base = base or ModelSpec(model=model)
    with _lock:
        learned = dict(_learned.get(model, {}))
    return replace(base, **learned) if learned else base


def learn_unsupported(model: str, capability: str) -> None:
    """Record that `model` rejected `capability` (a ModelSpec field name)."""
    if capability not in ModelSpec.__dataclass_fields__:
        raise ValueError(f"unknown capability {capability!r}")
    with _lock:
        _learned.setdefault(model, {})[capability] = (
            {} if capability == "extra_body" else False
        )


def reset_learned() -> None:
    with _lock:
        _learned.clear()


def known_models() -> list[str]:
    return list(_KNOWN)


__all__ = ["ModelSpec", "get_spec", "learn_unsupported", "reset_learned", "known_models"]
