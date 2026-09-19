"""Text → vector, via Pinecone's hosted inference API.

Using Pinecone's own embedding service keeps this to a single vendor and a
single API key for the whole RAG path — OpenRouter is a chat-completions
gateway and does not expose a dependable embeddings endpoint, so pairing it
with a separate embedding provider would add a third credential for no gain.

`documents` and `queries` are embedded with different input types, which is
what asymmetric retrieval models such as `llama-text-embed-v2` and
`multilingual-e5-large` expect.
"""

from __future__ import annotations

from app.config.settings import settings
from app.logging_setup import get_logger
from app.rag.pinecone_client import pinecone_client

log = get_logger("rag.embeddings")

# Pinecone's inference API caps how many inputs one call may carry.
_BATCH_SIZE = 90


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced at all."""


def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    if not texts:
        return []
    pc = pinecone_client.client()
    if pc is None:
        raise EmbeddingError("Pinecone is not configured — set PINECONE_API_KEY")

    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        try:
            response = pc.inference.embed(
                model=settings.pinecone.embedding_model,
                inputs=batch,
                parameters={"input_type": input_type, "truncate": "END"},
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"embedding call failed ({settings.pinecone.embedding_model}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        for item in response.data:
            values = item["values"] if isinstance(item, dict) else item.values
            vectors.append(list(values))

    if len(vectors) != len(texts):
        raise EmbeddingError(f"expected {len(texts)} embeddings, provider returned {len(vectors)}")
    return vectors


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed knowledge chunks for storage."""
    return _embed(texts, "passage")


def embed_query(text: str) -> list[float]:
    """Embed a single user question for retrieval."""
    vectors = _embed([text], "query")
    return vectors[0]


def embedding_dimension() -> int:
    return settings.pinecone.embedding_dimension


__all__ = ["embed_documents", "embed_query", "embedding_dimension", "EmbeddingError"]
