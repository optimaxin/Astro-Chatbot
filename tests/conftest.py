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


@pytest.fixture(autouse=True)
def _isolated_llm_reliability_state(monkeypatch):
    """Circuit-breaker and learned-capability state is process-global; a test
    that fails a fake model five times must not open that model's circuit for
    the next test. Backoff sleeps are made instant."""
    from app.llm import model_registry, reliability

    reliability.breaker.reset()
    model_registry.reset_learned()
    monkeypatch.setattr(reliability, "sleep", lambda seconds: None)
    yield
    reliability.breaker.reset()
    model_registry.reset_learned()


@pytest.fixture(autouse=True)
def _synchronous_side_effects():
    """Background metering/evaluation would outlive a test's monkeypatches and
    could then reach the REAL Redis/Postgres. Tests run them synchronously by
    default; tests of the async paths opt in and drain inside the test. Any
    job still queued is drained here, before patches are undone."""
    import dataclasses

    from app.config import settings as settings_module
    from app.observability import background

    s = settings_module.settings
    original = s.flags
    original_auth = s.auth
    object.__setattr__(s, "flags", dataclasses.replace(original, metering_async=False,
                                                       evaluation_async=False))
    # The pre-existing API tests check response shape, not identity, so they
    # run with the legacy contract. tests/test_auth.py covers optional and
    # required modes explicitly.
    object.__setattr__(s, "auth", dataclasses.replace(original_auth, mode="off"))
    yield
    background.drain(10)
    object.__setattr__(s, "flags", original)
    object.__setattr__(s, "auth", original_auth)


@pytest.fixture(autouse=True)
def _no_usage_writes_to_postgres(monkeypatch, tmp_path_factory):
    """Keep the test suite out of the real `ai_usage` table.

    `usage_ledger.write` fans out to Postgres, and several tests drive a real
    `LLM.complete()` against a fake client. Without this the suite inserts
    fabricated rows — including failures — into whatever database
    AI_USAGE_DATABASE_URL points at, which is the same one the admin Token
    Meter reports from. Metering data has to reflect real traffic only.
    """
    from app.evaluation import evaluation_store
    from app.llm import usage_ledger, usage_store

    monkeypatch.setattr(usage_store, "write", lambda record: False)
    monkeypatch.setattr(evaluation_store, "write_grounding", lambda *a, **k: False)
    # The live integration tests used to write judge rows (conversation ids
    # 'pytest-…') into the real ai_evaluations table through these two.
    monkeypatch.setattr(evaluation_store, "write_evaluation", lambda *a, **k: False)
    monkeypatch.setattr(evaluation_store, "write_feedback", lambda *a, **k: False)
    # Same reasoning for the other two sinks: tests must not increment the
    # real per-user counters in Upstash or append to the real JSONL ledger.
    # Tests of those sinks re-patch them explicitly.
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)
    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: tmp_path_factory.mktemp("ledger") / "usage.jsonl")
