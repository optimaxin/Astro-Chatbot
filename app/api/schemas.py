"""Request and response models for the HTTP layer.

These are the *only* place the service validates untrusted input. Everything
below the API — the graph, the nodes, the services — assumes it is being
called with sane values, which is safe precisely because nothing reaches them
without passing through here first.

The response shape is deliberately close to what the terminal already prints,
so a frontend and a developer at a terminal are looking at the same facts.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.config.settings import settings


class BirthDetails(BaseModel):
    """What the web client may send in place of a stored user profile.

    These five scalars are the *only* chart-related input accepted from a
    client. The chart itself is then recomputed by the Tredev calculators —
    see `kundli_service.kundli_from_birth` for why a ready-made chart is not
    accepted here. Every field is bounded, so nothing free-form from a browser
    can reach the system prompt.
    """

    date: str = Field(..., description="YYYY-MM-DD")
    time: str = Field(..., description="HH:MM, 24h")
    timezoneOffsetMinutes: int = Field(..., ge=-900, le=900)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    # Display-only, never interpreted astrologically. Bounded and stripped of
    # newlines so a long or multi-line value cannot restructure the prompt.
    name: str | None = Field(default=None, max_length=80)
    placeLabel: str | None = Field(default=None, max_length=120)

    @field_validator("date")
    @classmethod
    def _iso_date(cls, value: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
            raise ValueError("date must be YYYY-MM-DD")
        return value.strip()

    @field_validator("time")
    @classmethod
    def _iso_time(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", value):
            raise ValueError("time must be HH:MM in 24-hour form")
        return value

    @field_validator("name", "placeLabel")
    @classmethod
    def _single_line(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return " ".join(value.split()) or None


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    question: str = Field(..., min_length=1)
    # Optional: defaults to the user's id, so a caller that does not manage
    # sessions still gets a coherent per-user conversation.
    session_id: str | None = Field(default=None, max_length=128)
    persona_id: str | None = Field(default=None, max_length=64)
    # Supplied by the Kundli page for a visitor with no stored profile. When
    # present the chart is computed from these rather than looked up by
    # `user_id`, which is what lets a logged-out user chat about the chart
    # they just generated.
    birth: BirthDetails | None = None
    # How the reply should read: whatsapp_hinglish (default) | english |
    # hindi | formal | detailed. An unknown value falls back to the default
    # rather than erroring — a bad style should change how an answer sounds,
    # never cost the user the answer.
    response_style: str | None = Field(default=None, max_length=40)
    # Ask for the answer as several short chat bubbles instead of one block.
    # Defaults to True because every current caller is a chat UI; the terminal
    # REPL does not go through this schema.
    chat_bubbles: bool = True

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

    @field_validator("user_id", "session_id", "persona_id", "response_style")
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
    # The whole reply as one string, kept for any client that wants it
    # (and for logs). `messages` is the same text split for the bubble UI.
    answer: str
    messages: list[str] = []
    persona: str
    response_style: str
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



# Structured reasons a 👎 can carry. A closed set rather than free text so the
# admin dashboard can rank them; `other` plus the optional comment covers
# everything else without turning this into an open vocabulary.
NEGATIVE_REASONS = {
    "incorrect_information", "too_generic", "too_long", "did_not_answer",
    "not_personalized", "poor_hinglish", "too_formal", "not_conversational",
    "wrong_context_usage", "other",
}
POSITIVE_REASONS = {
    "clear", "personalized", "relevant", "helpful_advice",
    "natural_conversation", "good_explanation",
}


class FeedbackRequest(BaseModel):
    """What a user says about one reply.

    Keyed by the turn's `request_id`, which the chat response already returns,
    so the client never has to invent an id or hold extra state.
    """

    request_id: str = Field(..., min_length=1, max_length=64)
    user_id: str = Field(..., min_length=1, max_length=128)
    feedback_type: str = Field(..., description="helpful | not_helpful")
    conversation_id: str | None = Field(default=None, max_length=128)
    # Optional throughout: the thumb is the primary signal and the UI must not
    # nag for the rest.
    rating: int | None = Field(default=None, ge=1, le=5)
    feedback_reason: str | None = Field(default=None, max_length=64)
    feedback_comment: str | None = Field(default=None, max_length=1000)

    @field_validator("feedback_type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"helpful", "not_helpful"}:
            raise ValueError("feedback_type must be 'helpful' or 'not_helpful'")
        return value

    @field_validator("feedback_reason")
    @classmethod
    def _known_reason(cls, value: str | None) -> str | None:
        if not value:
            return None
        value = value.strip().lower()
        if value not in NEGATIVE_REASONS | POSITIVE_REASONS:
            raise ValueError("unknown feedback_reason")
        return value

    @field_validator("feedback_comment")
    @classmethod
    def _clean_comment(cls, value: str | None) -> str | None:
        if not value:
            return None
        return " ".join(value.split()) or None


class FeedbackResponse(BaseModel):
    recorded: bool
    request_id: str
    feedback_type: str
    # False when the store is unreachable. The UI still shows the thumb as
    # selected — losing a vote must not look like a broken button.
    persisted: bool = True


class ResponseStyleOut(BaseModel):
    id: str
    name: str
    language: str
    tone: str
    format: str
    length: str
    emojiLevel: str


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
    "ResponseStyleOut",
    "FeedbackRequest",
    "FeedbackResponse",
    "NEGATIVE_REASONS",
    "POSITIVE_REASONS",
    "HealthResponse",
    "ChartResponse",
    "ErrorResponse",
]
