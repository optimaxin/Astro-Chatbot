"""Shared fixtures.

Tests are split into two groups. Everything that can be checked without a
network call runs always; anything needing OpenRouter, Pinecone or Upstash is
skipped with a clear reason when that service is not configured, so
`pytest` is useful both on a laptop with no keys and in a fully configured
environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.kundli.kundli_service import KundliContext, SampleFileKundliProvider  # noqa: E402
from app.memory.redis_memory import Message  # noqa: E402
from app.rag.retriever import RetrievalResult, RetrievedChunk  # noqa: E402

needs_openrouter = pytest.mark.skipif(
    not settings.openrouter.configured, reason="OPENROUTER_API_KEY is not set"
)
needs_pinecone = pytest.mark.skipif(
    not settings.pinecone.configured, reason="PINECONE_API_KEY is not set"
)
needs_redis = pytest.mark.skipif(
    not settings.redis.configured, reason="Upstash Redis is not configured"
)


@pytest.fixture(scope="session")
def kundli() -> KundliContext:
    context = SampleFileKundliProvider().get_kundli("test-user-001")
    assert context.available, f"sample kundli did not load: {context.error}"
    return context


@pytest.fixture
def empty_kundli() -> KundliContext:
    return KundliContext.unavailable("nobody", "test", "no chart on file")


@pytest.fixture
def retrieval() -> RetrievalResult:
    return RetrievalResult(
        chunks=[
            RetrievedChunk(
                text="The 10th house governs profession, reputation and public standing.",
                score=0.81,
                domain="houses",
                topic="the_10th_house",
                document="houses.txt",
                source="astrology_knowledge",
            )
        ],
        used_filter=True,
        widened=False,
        available=True,
    )


@pytest.fixture
def history() -> list[Message]:
    return [
        Message(role="user", content="I want to know about my career.", timestamp=1.0),
        Message(role="assistant", content="Are you working or studying right now?", timestamp=2.0),
        Message(role="user", content="I am working as a software developer.", timestamp=3.0),
    ]
