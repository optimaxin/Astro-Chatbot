"""Deterministic stand-ins for an OpenAI-protocol endpoint.

`FakeClient` replays a script: each entry is either a response (dict) or an
exception instance, consumed one per `chat.completions.create` call. Every
request's kwargs are captured, so tests assert on what was actually sent.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import httpx
import openai

URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def http_error(status: int, message: str = "error", headers: dict[str, str] | None = None):
    response = httpx.Response(status, request=httpx.Request("POST", URL), headers=headers or {})
    cls = {
        400: openai.BadRequestError, 401: openai.AuthenticationError,
        403: openai.PermissionDeniedError, 404: openai.NotFoundError,
        429: openai.RateLimitError, 500: openai.InternalServerError,
        502: openai.InternalServerError, 503: openai.InternalServerError,
    }.get(status, openai.APIStatusError)
    return cls(message, response=response, body=None)


def timeout_error():
    return openai.APITimeoutError(request=httpx.Request("POST", URL))


def reply(text: str | None = "ok", *, model: str = "m", usage: dict | None = "default",
          tool_calls: list[dict] | None = None, finish_reason: str = "stop") -> dict:
    return {"text": text, "model": model, "usage": usage, "tool_calls": tool_calls,
            "finish_reason": finish_reason}


def _completion(spec: dict, requested_model: str) -> Any:
    usage = spec["usage"]
    if usage == "default":
        usage = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    calls = [
        SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=c["name"],
                                                                  arguments=c["arguments"]))
        for i, c in enumerate(spec["tool_calls"] or [])
    ]
    message = SimpleNamespace(content=spec["text"], tool_calls=calls or None)
    return SimpleNamespace(
        model=requested_model if spec["model"] == "m" else spec["model"],
        choices=[SimpleNamespace(message=message, finish_reason=spec["finish_reason"])],
        usage=usage,
    )


class FakeStream:
    def __init__(self, deltas: list[str], usage: dict | None, fail_after: int | None,
                 model: str) -> None:
        self.deltas, self.usage, self.fail_after, self.model = deltas, usage, fail_after, model
        self.closed = False

    def __iter__(self):
        for i, delta in enumerate(self.deltas):
            if self.fail_after is not None and i == self.fail_after:
                raise http_error(502, "stream broke")
            yield SimpleNamespace(model=self.model, usage=None,
                                  choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))])
        if self.usage is not None:
            yield SimpleNamespace(model=self.model, usage=self.usage, choices=[])

    def close(self) -> None:
        self.closed = True


def stream(deltas: list[str], *, usage: dict | None = "default", fail_after: int | None = None):
    if usage == "default":
        usage = {"prompt_tokens": 20, "completion_tokens": len(deltas),
                 "total_tokens": 20 + len(deltas)}
    return {"stream": deltas, "usage": usage, "fail_after": fail_after}


class FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if not self.script:
            raise AssertionError("FakeClient script exhausted")
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        if "stream" in step:
            s = FakeStream(step["stream"], step["usage"], step["fail_after"], kwargs["model"])
            self.streams.append(s)
            return s
        return _completion(step, kwargs["model"])

    @property
    def models(self) -> list[str]:
        return [r["model"] for r in self.requests]


def use_nvidia(monkeypatch, script: list[Any], **overrides: Any) -> tuple[Any, FakeClient]:
    """Point the gateway at NVIDIA with a fake client. `overrides` may hold
    `nvidia={...}` and/or `reliability={...}` field overrides."""
    from app.config import settings as settings_module
    from app.llm import llm_factory, streaming

    base = settings_module.settings
    nvidia = dataclasses.replace(base.nvidia, api_key="nvapi-test", **overrides.get("nvidia", {}))
    rel = dataclasses.replace(base.reliability, **overrides.get("reliability", {}))
    patched = dataclasses.replace(base, llm_provider="nvidia", nvidia=nvidia, reliability=rel)
    for module in (llm_factory, streaming):
        monkeypatch.setattr(module, "settings", patched)
    client = FakeClient(script)
    monkeypatch.setattr(llm_factory, "_openai_client", lambda: client)
    return llm_factory, client


__all__ = ["FakeClient", "http_error", "timeout_error", "reply", "stream", "use_nvidia"]
