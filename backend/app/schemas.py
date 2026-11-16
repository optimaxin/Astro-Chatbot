from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.settings import settings

_MAX = settings.max_prompt_chars


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=_MAX)


class GenerationParams(BaseModel):
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=500)
    max_tokens: int | None = Field(default=None, ge=1, le=32000)


class ChatRequest(BaseModel):
    provider: str = Field(..., min_length=1, max_length=64)
    model: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=_MAX)
    system_prompt: str | None = Field(default=None, max_length=_MAX)
    context: str | None = Field(default=None, max_length=_MAX)
    history: list[ChatTurn] = Field(default_factory=list, max_length=60)
    params: GenerationParams = Field(default_factory=GenerationParams)
    session_id: str | None = Field(default=None, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    # Client-generated per send; a retry with the same id returns the first record.
    request_id: str | None = Field(default=None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    tester_name: str | None = Field(default=None, max_length=60)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class EvaluateRequest(BaseModel):
    experiment_id: int = Field(..., ge=1)
    provider: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=200)


class FeedbackRequest(BaseModel):
    experiment_id: int = Field(..., ge=1)
    rating: Literal["good", "bad"] | None = None
    feedback: str | None = Field(default=None, max_length=4000)
    edited_response: str | None = Field(default=None, max_length=_MAX)
