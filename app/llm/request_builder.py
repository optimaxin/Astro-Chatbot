"""Request construction for one OpenAI-protocol attempt.

Split out of the gateway so `llm_factory` stays about control flow (retries,
fallback, accounting) and this module stays about *what is sent*: adapting a
request to what a model accepts, learning from 400s, and reading tool calls
and reasoning blocks out of a reply.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.llm import reliability
from app.llm.model_registry import ModelSpec, learn_unsupported
from app.llm.model_router import has_image
from app.llm.providers import get_provider
from app.llm.reliability import ErrorInfo, GatewayError
from app.logging_setup import get_logger

log = get_logger("llm.request")


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove inline reasoning blocks. An unterminated block (reasoning cut
    off by max_tokens) leaves nothing visible, which is reported as empty."""
    text = _THINK_BLOCK.sub("", text)
    if text.lstrip().lower().startswith("<think>"):
        return ""
    return text


def parse_tool_calls(message: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for call in getattr(message, "tool_calls", None) or []:
        fn = getattr(call, "function", None)
        raw_args = getattr(fn, "arguments", "") or ""
        try:
            parsed = json.loads(raw_args) if raw_args else {}
        except (TypeError, ValueError):
            parsed = None  # malformed JSON from the model; kept raw for the caller
        out.append({
            "id": getattr(call, "id", None),
            "name": getattr(fn, "name", None),
            "arguments": raw_args,
            "parsed_arguments": parsed,
        })
    return out


def build_request(
    *,
    provider: str,
    spec: ModelSpec,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    streaming: bool,
    settings: Any = None,
) -> dict[str, Any]:
    """The chat.completions kwargs for one attempt, adapted to the model.

    Optional features the model is known not to support are dropped (JSON
    mode — every caller parses tolerantly) or refused (tools, images — silently
    dropping those would change what the answer means).
    """
    kwargs: dict[str, Any] = {"model": spec.model, "messages": messages, "max_tokens": max_tokens}
    if spec.temperature:
        kwargs["temperature"] = temperature

    rf = response_format
    if rf and rf.get("type") == "json_schema" and spec.json_schema is False:
        rf = {"type": "json_object"} if spec.json_mode is not False else None
    if rf and rf.get("type") == "json_object" and spec.json_mode is False:
        rf = None
    if rf:
        kwargs["response_format"] = rf

    if tools:
        if spec.tools is False:
            raise GatewayError(reliability.UNSUPPORTED, f"{spec.model} does not support tool calling")
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

    if spec.vision is False and has_image(messages):
        raise GatewayError(reliability.UNSUPPORTED, f"{spec.model} does not accept images")

    if streaming:
        kwargs["stream"] = True
    if settings is None:
        from app.config.settings import settings
    return get_provider(provider).prepare(kwargs, spec, settings, streaming=streaming)


def learn_from_bad_request(model: str, info: ErrorInfo, kwargs: dict[str, Any]) -> bool:
    """On a 400 caused by an optional parameter, remember the model rejects it
    and report True so the attempt is repeated without it."""
    body = kwargs.get("extra_body") or {}
    checks = (
        ("response_format" in kwargs, ("response_format", "json_object", "json_schema", "json mode"),
         ("json_mode", "json_schema")),
        ("stream_options" in kwargs, ("stream_options", "include_usage"), ("stream_usage",)),
        ("chat_template_kwargs" in body, ("chat_template_kwargs", "enable_thinking"), ("extra_body",)),
        ("temperature" in kwargs, ("temperature",), ("temperature",)),
    )
    for present, markers, capabilities in checks:
        if present and reliability.mentions_param(info, *markers):
            for capability in capabilities:
                learn_unsupported(model, capability)
            log.warning("model rejected a parameter; retrying without it",
                        extra={"model": model, "capabilities": capabilities})
            return True
    return False


__all__ = ["build_request", "learn_from_bad_request", "strip_think", "parse_tool_calls"]
