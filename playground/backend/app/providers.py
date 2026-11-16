"""Provider adapter layer.

`LLMProvider` is the contract; `OpenAICompatibleProvider` implements it for
every upstream that speaks the OpenAI chat-completions protocol (Gemini's
/v1beta/openai endpoint, OpenRouter, NVIDIA). `AiCoreProvider` calls the
Tredev chatbot's own `/chat`, so testers exercise the full pipeline (RAG,
guardrails, reply validation) rather than a bare model.
"""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.catalog import ModelConfig, ProviderConfig
from app.settings import settings


class ProviderError(RuntimeError):
    """A failed generation, carrying a reason that is safe to show the tester."""

    def __init__(self, provider: str, reason: str, status_code: int | None = None) -> None:
        super().__init__(reason)
        self.provider = provider
        self.reason = reason
        self.status_code = status_code


@dataclass
class GenerationResult:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    provider_latency_ms: int
    finish_reason: str | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    provider_cost: float | None = None
    raw_usage: dict[str, Any] = field(default_factory=dict)
    usage_source: str = "provider"
    extra: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, model: ModelConfig, messages: list[dict[str, str]],
                 params: dict[str, Any], **request: Any) -> GenerationResult: ...


def filter_params(config: ProviderConfig, params: dict[str, Any]) -> dict[str, Any]:
    """Drop parameters the provider does not support, and unset ones."""
    return {k: v for k, v in params.items() if v is not None and k in config.supported_params}


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return dict(usage) if isinstance(usage, dict) else {}


def _nested(d: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def _friendly_reason(exc: Exception) -> tuple[str, int | None]:
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if status == 429 or name == "RateLimitError":
        return "Rate limit exceeded.", status
    if status in (401, 403) or name in {"AuthenticationError", "PermissionDeniedError"}:
        return "The provider rejected the API key.", status
    if status == 402:
        return "The provider account has insufficient credits.", status
    if status == 404 or name == "NotFoundError":
        return "The provider does not recognise this model id.", status
    if status == 410:
        return "The provider has retired this model; it is no longer available.", status
    if name in {"APITimeoutError", "TimeoutError"}:
        return "The provider did not respond in time.", status
    if name == "APIConnectionError":
        return "Could not connect to the provider.", status
    message = getattr(exc, "message", None) or str(exc)
    return f"{name}: {message[:300]}", status


class OpenAICompatibleProvider(LLMProvider):
    def _client(self) -> Any:
        from openai import OpenAI

        key = settings.api_key(self.config.api_key_env)
        if not key:
            raise ProviderError(self.config.name,
                                f"{self.config.api_key_env} is not set on the playground backend.")
        return OpenAI(api_key=key, base_url=self.config.base_url,
                      timeout=settings.timeout_seconds, max_retries=0)

    def generate(self, model: ModelConfig, messages: list[dict[str, str]],
                 params: dict[str, Any], **request: Any) -> GenerationResult:
        client = self._client()
        kwargs: dict[str, Any] = {"model": model.id, "messages": messages}
        sent = filter_params(self.config, params)
        extra_body = {**self.config.extra_body, **model.extra_body}
        if "top_k" in sent:
            # top_k is not part of the OpenAI schema, so it travels in the body.
            extra_body["top_k"] = sent.pop("top_k")
        kwargs.update(sent)
        kwargs.update(model.request_overrides)
        if extra_body:
            kwargs["extra_body"] = extra_body

        started = time.perf_counter()
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — converted to a tester-facing error
            reason, status = _friendly_reason(exc)
            raise ProviderError(self.config.name, reason, status) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        choices = getattr(response, "choices", None) or []
        text = (choices[0].message.content or "").strip() if choices else ""
        usage = _usage_dict(getattr(response, "usage", None))
        cost = usage.get("cost")
        return GenerationResult(
            text=text,
            model=getattr(response, "model", None) or model.id,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            provider_latency_ms=latency_ms,
            finish_reason=getattr(choices[0], "finish_reason", None) if choices else None,
            reasoning_tokens=_nested(usage, "completion_tokens_details", "reasoning_tokens"),
            cached_tokens=_nested(usage, "prompt_tokens_details", "cached_tokens"),
            provider_cost=float(cost) if isinstance(cost, (int, float)) else None,
            raw_usage=usage,
            usage_source="provider" if usage.get("prompt_tokens") is not None else "unavailable",
        )


_BIRTH_KEYS = {"date", "time", "timezoneOffsetMinutes", "latitude", "longitude"}
# ai-core only chats with anonymous visitors who have a chart, so when the
# tester gives none, this sample chart is sent (and flagged in the record).
DEFAULT_BIRTH = {"date": "1995-04-12", "time": "06:30", "timezoneOffsetMinutes": 330,
                 "latitude": 28.6139, "longitude": 77.2090, "placeLabel": "New Delhi (sample chart)"}
# ai-core issues an anonymous session token on the first turn; it must be
# echoed for the conversation to continue. Kept per playground session.
_aicore_tokens: dict[str, str] = {}
_aicore_lock = threading.Lock()


def birth_from_context(context: str | None) -> dict[str, Any] | None:
    """Birth details for ai-core, when the tester pasted them as JSON in Context."""
    if not context or not context.strip().startswith("{"):
        return None
    try:
        data = json.loads(context)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and isinstance(data.get("birth"), dict):
        data = data["birth"]
    return data if isinstance(data, dict) and _BIRTH_KEYS <= set(data) else None


class AiCoreProvider(LLMProvider):
    """The deployed Tredev chatbot. It keeps its own conversation memory, so
    only the latest user message is sent; the system prompt does not apply."""

    def generate(self, model: ModelConfig, messages: list[dict[str, str]],
                 params: dict[str, Any], **request: Any) -> GenerationResult:
        base = self.config.resolved_base_url
        if not base:
            raise ProviderError(self.config.name, f"{self.config.base_url_env} is not set on the playground backend.")
        session = "playground-" + str(request.get("session_id") or "default")
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        body: dict[str, Any] = {"user_id": session, "session_id": session, "question": question,
                                "chat_bubbles": False, **model.request_overrides}
        birth = birth_from_context(request.get("context"))
        body["birth"] = birth or DEFAULT_BIRTH
        with _aicore_lock:
            token = _aicore_tokens.get(session)
        headers = {"X-Chat-Session-Token": token} if token else {}

        started = time.perf_counter()
        try:
            res = httpx.post(f"{base}/chat", json=body, headers=headers, timeout=settings.timeout_seconds)
        except httpx.TimeoutException as exc:
            raise ProviderError(self.config.name, "The provider did not respond in time.") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.config.name, "Could not connect to the provider.") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        if res.status_code == 429:
            raise ProviderError(self.config.name, "Rate limit exceeded.", 429)
        if res.status_code >= 400:
            try:
                detail = str(res.json().get("detail") or "")[:200]
            except ValueError:
                detail = ""
            reason = "ai-core refused the request" if res.status_code in (401, 403) else "ai-core returned an error"
            raise ProviderError(self.config.name, f"{reason} (HTTP {res.status_code}){': ' + detail if detail else '.'}",
                                res.status_code)
        data = res.json()
        if data.get("session_token"):
            with _aicore_lock:
                _aicore_tokens[session] = data["session_token"]

        usage = data.get("usage") or {}
        cost = usage.get("turn_cost_usd", usage.get("estimated_cost_usd"))
        return GenerationResult(
            text=(data.get("answer") or "").strip(),
            model=usage.get("model") or model.id,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            provider_latency_ms=latency_ms,
            provider_cost=float(cost) if isinstance(cost, (int, float)) else None,
            raw_usage=usage,
            usage_source=usage.get("token_source") or ("provider" if usage.get("tokens_available") else "unavailable"),
            extra={**{k: data.get(k) for k in ("request_id", "intent", "persona", "response_style", "degraded",
                                               "notices", "citations", "sources", "chart_facts_used", "latency_ms")},
                   "chart": "from Context" if birth else "default sample chart (1995-04-12 06:30, New Delhi)"},
        )


_ADAPTERS: dict[str, type[LLMProvider]] = {"aicore": AiCoreProvider}


def get_provider(config: ProviderConfig) -> LLMProvider:
    return _ADAPTERS.get(config.kind, OpenAICompatibleProvider)(config)
