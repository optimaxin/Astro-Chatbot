"""Pinecone connection management for the astrology KNOWLEDGE base.

Scope note, because it is the single most important data-design decision in
this service: Pinecone holds *general astrology knowledge* only — what the
10th house signifies, what a Saturn Mahadasha classically means, and so on.
It never holds a user's kundli. A kundli is structured, user-specific,
exactly-retrievable data and belongs in the Tredev backend (see
`app/kundli/kundli_service.py`); putting it in a vector store would turn an
exact lookup into a fuzzy one for no benefit.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import settings
from app.logging_setup import get_logger

log = get_logger("rag.pinecone")


class PineconeClient:
    def __init__(self) -> None:
        self._pc: Any | None = None
        self._index: Any | None = None

    @property
    def configured(self) -> bool:
        return settings.pinecone.configured

    def client(self) -> Any | None:
        if self._pc is not None:
            return self._pc
        if not self.configured:
            return None
        try:
            from pinecone import Pinecone

            self._pc = Pinecone(api_key=settings.pinecone.api_key)
            return self._pc
        except Exception:
            log.exception("pinecone client construction failed")
            return None

    def index_exists(self) -> bool:
        pc = self.client()
        if pc is None:
            return False
        try:
            return pc.has_index(settings.pinecone.index_name)
        except Exception:
            log.exception("has_index failed")
            return False

    def ensure_index(self) -> bool:
        """Create the serverless index if it is missing. Used by ingestion."""
        pc = self.client()
        if pc is None:
            return False
        name = settings.pinecone.index_name
        try:
            if pc.has_index(name):
                return True
            from pinecone import ServerlessSpec

            log.info(
                "creating pinecone index",
                extra={"index": name, "dimension": settings.pinecone.embedding_dimension},
            )
            pc.create_index(
                name=name,
                dimension=settings.pinecone.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone.cloud,
                    region=settings.pinecone.region,
                ),
            )
            # Serverless indexes come up quickly but not instantly.
            import time

            for _ in range(60):
                if pc.describe_index(name).status.get("ready"):
                    return True
                time.sleep(1)
            log.warning("index created but not ready within 60s", extra={"index": name})
            return False
        except Exception:
            log.exception("ensure_index failed", extra={"index": name})
            return False

    def index(self) -> Any | None:
        if self._index is not None:
            return self._index
        pc = self.client()
        if pc is None:
            return None
        try:
            self._index = pc.Index(settings.pinecone.index_name)
            return self._index
        except Exception:
            log.exception("could not open index", extra={"index": settings.pinecone.index_name})
            return None

    def stats(self) -> dict[str, Any] | None:
        index = self.index()
        if index is None:
            return None
        try:
            raw = index.describe_index_stats()
            return raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
        except Exception:
            log.exception("describe_index_stats failed")
            return None

    def ping(self) -> tuple[bool, str]:
        """Used by scripts/test_connections.py."""
        if not self.configured:
            return False, "PINECONE_API_KEY is not set"
        pc = self.client()
        if pc is None:
            return False, "could not construct the Pinecone client"
        name = settings.pinecone.index_name
        try:
            if not pc.has_index(name):
                return (
                    False,
                    f"index '{name}' does not exist yet — run: python -m scripts.ingest_knowledge",
                )
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

        stats = self.stats()
        if stats is None:
            return False, f"index '{name}' exists but stats could not be read"
        namespaces = stats.get("namespaces") or {}
        vectors = namespaces.get(settings.pinecone.namespace, {}).get("vector_count", 0)
        if not vectors:
            return (
                True,
                f"index '{name}' reachable but namespace '{settings.pinecone.namespace}' is empty "
                "— run: python -m scripts.ingest_knowledge",
            )
        return True, f"index '{name}' reachable, {vectors} vectors in '{settings.pinecone.namespace}'"


pinecone_client = PineconeClient()

__all__ = ["PineconeClient", "pinecone_client"]
