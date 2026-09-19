"""Request and response models for the HTTP layer.

These are the *only* place the service validates untrusted input. Everything
below the API — the graph, the nodes, the services — assumes it is being
called with sane values, which is safe precisely because nothing reaches them
without passing through here first.

The response shape is deliberately close to what the terminal already prints,
so a frontend and a developer at a terminal are looking at the same facts.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.config.settings import settings


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    question: str = Field(..., min_length=1)
    # Optional: defaults to the user's id, so a caller that does not manage
    # sessions still gets a coherent per-user conversation.
    session_id: str | None = Field(default=None, max_length=128)
    persona_id: str | None = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def _bounded_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        limit = settings.api.max_question_chars
        if len(value) > limit:
            raise ValueError(f"question must be at most {limit} characters")
        return value

    @field_validator("user_id", "session_id", "persona_id")
    @classmethod
    def _clean_id(cls, value: str | None) -> str | None:
        return value.strip() if value else value

    def resolved_session_id(self) -> str:
        return self.session_id or f"user-{self.user_id}"


class Usage(BaseModel):
    """Token accounting for the user-visible answer.

    Every field is nullable on purpose: when the provider does not return a
    usage block these stay `None` rather than being filled with a plausible
    estimate. `tokens_available` says which case you are looking at.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    tokens_available: bool = False
    model: str | None = None
    estimated_cost_usd: float | None = None
    # OpenRouter's own figure, when it reports one — used to cross-check the
    # estimate rather than to replace it.
    provider_cost_usd: float | None = None
    # All calls in the turn, including intent classification and the judge.
    turn_total_tokens: int | None = None
    turn_cost_usd: float | None = None
    llm_calls: int = 0


class Source(BaseModel):
    id: str
    document: str
    domain: str
    topic: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    persona: str
    session_id: str
    request_id: str
    intent: str
    sources: list[Source] = []
    chart_facts_used: list[str] = []
    usage: Usage
    latency_ms: int = 0
    # True when the model could not be reached and `answer` is the honest
    # fallback rather than a reading. A client should not render it as advice.
    degraded: bool = False
    # Human-readable notes about which dependencies were unavailable. Never
    # contains a stack trace, a credential or an internal path.
    notices: list[str] = []


class PersonaOut(BaseModel):
    id: str
    name: str
    specialization: str
    tone: str
    system: str
    topicTags: list[str] = []
    topK: int | None = None


class HealthResponse(BaseModel):
    status: str
    dependencies: dict[str, Any]


class ChartResponse(BaseModel):
    user_id: str
    available: bool
    facts: dict[str, Any] = {}
    error: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "Usage",
    "Source",
    "PersonaOut",
    "HealthResponse",
    "ChartResponse",
    "ErrorResponse",
]
