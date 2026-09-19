"""Retrieval over the astrology knowledge base, with intent-aware filtering.

Retrieval is deliberately *not* a blind similarity sweep over everything.
Each chunk is ingested with a `domain` (career, marriage, dasha, …), and a
classified intent narrows the search to the domains that can plausibly answer
it before ranking by similarity. A career question therefore ranks career and
house knowledge above nakshatra trivia even when the wording happens to be
lexically closer to the latter.

The filter is a preference, not a hard gate: if the filtered search comes back
thin, the retriever widens to an unfiltered search rather than answering from
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config.settings import settings
from app.logging_setup import get_logger
from app.rag.embeddings import EmbeddingError, embed_query
from app.rag.pinecone_client import pinecone_client

log = get_logger("rag.retriever")

# Which knowledge domains are worth searching for each classified intent.
# Ordered most- to least- specific; `foundation` domains describe the general
# machinery (houses, planets, signs, dashas) and are useful for every intent.
FOUNDATION_DOMAINS = ("houses", "planets", "rashis", "dashas", "nakshatras")

INTENT_DOMAINS: dict[str, tuple[str, ...]] = {
    "career": ("career", "houses", "planets", "dashas"),
    "finance": ("finance", "houses", "planets", "dashas"),
    "love": ("relationships", "houses", "planets"),
    "relationship": ("relationships", "houses", "planets"),
    "marriage": ("relationships", "houses", "planets", "nakshatras"),
    "health": ("health", "houses", "planets", "dashas"),
    "personal_growth": ("personal_growth", "houses", "planets", "dashas"),
    "general_astrology": FOUNDATION_DOMAINS,
    "smalltalk": FOUNDATION_DOMAINS,
    "unknown": FOUNDATION_DOMAINS,
}


@dataclass
class RetrievedChunk:
    text: str
    score: float
    domain: str
    topic: str
    document: str
    source: str

    def citation(self) -> str:
        return f"{self.domain}/{self.topic}"


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    used_filter: bool
    widened: bool
    available: bool
    error: str | None = None

    @property
    def count(self) -> int:
        return len(self.chunks)

    def to_prompt_block(self) -> str:
        if not self.chunks:
            return (
                "RETRIEVED ASTROLOGY KNOWLEDGE: none available for this question. "
                "Answer from general principles and say plainly when you are not certain."
            )
        parts = [
            "RETRIEVED ASTROLOGY KNOWLEDGE (ground your reasoning in this; "
            "do not invent classical rules that are not here):"
        ]
        for i, chunk in enumerate(self.chunks, start=1):
            parts.append(f"[{i}] ({chunk.citation()}) {chunk.text}")
        return "\n\n".join(parts)

    @staticmethod
    def unavailable(error: str) -> "RetrievalResult":
        return RetrievalResult(chunks=[], used_filter=False, widened=False, available=False, error=error)


def _to_chunks(matches: list[Any], min_score: float) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for match in matches:
        data = match if isinstance(match, dict) else match.to_dict()
        score = float(data.get("score", 0.0))
        if score < min_score:
            continue
        metadata = data.get("metadata") or {}
        text = str(metadata.get("text", "")).strip()
        if not text:
            continue
        chunks.append(
            RetrievedChunk(
                text=text,
                score=score,
                domain=str(metadata.get("domain", "unknown")),
                topic=str(metadata.get("topic", "unknown")),
                document=str(metadata.get("document", "unknown")),
                source=str(metadata.get("source", "astrology_knowledge")),
            )
        )
    return chunks


def _query(vector: list[float], top_k: int, domains: tuple[str, ...] | None) -> list[Any]:
    index = pinecone_client.index()
    if index is None:
        raise RuntimeError("Pinecone index is not available")
    kwargs: dict[str, Any] = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True,
        "namespace": settings.pinecone.namespace,
    }
    if domains:
        kwargs["filter"] = {"domain": {"$in": list(domains)}}
    response = index.query(**kwargs)
    data = response if isinstance(response, dict) else response.to_dict()
    return list(data.get("matches") or [])


def retrieve(
    query: str,
    intent: str = "general_astrology",
    top_k: int | None = None,
    extra_domains: tuple[str, ...] | None = None,
) -> RetrievalResult:
    """Retrieve knowledge for a question. Never raises — RAG is optional.

    `extra_domains` is the persona's own specialization, unioned with (never
    replacing) the intent's domains. A union rather than an override, because
    a persona narrows *emphasis*, and a Career persona asked a health question
    should still be able to reach health knowledge rather than answer from the
    wrong shelf.
    """
    if not settings.enable_rag:
        return RetrievalResult.unavailable("RAG disabled via ENABLE_RAG=false")
    if not pinecone_client.configured:
        return RetrievalResult.unavailable("PINECONE_API_KEY is not set")

    top_k = top_k or settings.pinecone.top_k
    domains = INTENT_DOMAINS.get(intent, FOUNDATION_DOMAINS)
    if extra_domains:
        domains = tuple(dict.fromkeys([*domains, *extra_domains]))

    try:
        vector = embed_query(query)
    except EmbeddingError as exc:
        log.warning("embedding failed, skipping retrieval", extra={"error_message": str(exc)[:300]})
        return RetrievalResult.unavailable(str(exc))

    try:
        matches = _query(vector, top_k, domains)
        chunks = _to_chunks(matches, settings.pinecone.min_score)
        widened = False

        # A narrow filter that returns almost nothing is worse than no filter.
        if len(chunks) < max(1, top_k // 2):
            widened = True
            wider = _query(vector, top_k, None)
            seen = {c.text for c in chunks}
            chunks.extend(c for c in _to_chunks(wider, settings.pinecone.min_score) if c.text not in seen)
    except Exception as exc:  # noqa: BLE001 — a RAG outage must not end the turn
        log.warning(
            "pinecone query failed",
            extra={"intent": intent, "error_type": type(exc).__name__, "error_message": str(exc)[:300]},
        )
        return RetrievalResult.unavailable(f"{type(exc).__name__}: {exc}")

    chunks.sort(key=lambda c: c.score, reverse=True)
    # Relative cutoff: drop anything far weaker than the best hit, so a single
    # strong match is not padded out with near-noise.
    if chunks:
        floor = chunks[0].score * settings.pinecone.relative_min
        chunks = [c for c in chunks if c.score >= floor]
    chunks = chunks[:top_k]

    log.info(
        "retrieval complete",
        extra={
            "intent": intent,
            "domains": list(domains),
            "chunks": len(chunks),
            "widened": widened,
            "top_score": round(chunks[0].score, 4) if chunks else None,
        },
    )
    return RetrievalResult(chunks=chunks, used_filter=True, widened=widened, available=True)


__all__ = ["retrieve", "RetrievalResult", "RetrievedChunk", "INTENT_DOMAINS", "FOUNDATION_DOMAINS"]
