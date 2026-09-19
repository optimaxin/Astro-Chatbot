"""Ingest the astrology knowledge base into Pinecone.

    python -m scripts.ingest_knowledge [--dry-run] [--recreate] [--pdf-domain DOMAIN]

Pipeline: load -> clean -> chunk -> embed -> upsert.

Two kinds of source are handled:

* `data/knowledge/*.txt` — the curated files, split on their `## ` headings,
  so every chunk is one self-contained topic rather than an arbitrary window.
* `data/knowledge/**/*.pdf` — books, extracted with pypdf and split by
  chapter/heading first, falling back to an overlapping sentence window
  (CHUNK_SIZE / CHUNK_OVERLAP). Each PDF needs a retrieval domain, declared
  in `data/knowledge/manifest.json` or passed with `--pdf-domain`.

Every chunk carries metadata used for filtered retrieval:

    {"domain": "career", "topic": "saturn_and_career",
     "source": "astrology_knowledge", "document": "career.txt", "language": "en"}

Chunk ids are deterministic (`document::topic::n`), so re-running the script
updates existing vectors in place instead of creating duplicates.

Only general astrology knowledge belongs here. A user's kundli must never be
ingested — it is exact structured data and is fetched by key from the Tredev
backend (see `app/kundli/kundli_service.py`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.console import LINE, fail_mark, ok_mark, safe_print  # noqa: E402
from app.logging_setup import get_logger  # noqa: E402
from app.rag.chunking import (  # noqa: E402
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    Chunk,
    PdfExtractionError,
    chunk_pdf,
    clean_markdown,
    clean_text,
    slugify,
)
from app.rag.embeddings import EmbeddingError, embed_documents  # noqa: E402
from app.rag.pinecone_client import pinecone_client  # noqa: E402

log = get_logger("ingest")

# Filename -> retrieval domain. The domain is what `retriever.INTENT_DOMAINS`
# filters on, so it is declared here rather than inferred from the filename.
DOCUMENT_DOMAINS: dict[str, str] = {
    "planets.txt": "planets",
    "houses.txt": "houses",
    "rashis.txt": "rashis",
    "nakshatras.txt": "nakshatras",
    "dashas.txt": "dashas",
    "career.txt": "career",
    "relationships.txt": "relationships",
    "finance.txt": "finance",
    "health.txt": "health",
    "personal_growth.txt": "personal_growth",
}

# The domains `retriever.INTENT_DOMAINS` can actually filter on. A chunk
# ingested outside this set is unreachable by filtered retrieval.
KNOWN_DOMAINS = set(DOCUMENT_DOMAINS.values())

UPSERT_BATCH = 50


def split_long(heading: str, body: str) -> list[str]:
    """Split an over-long section on sentence boundaries, keeping its heading."""
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    sentences = re.split(r"(?<=[.!?])\s+", body)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > MAX_CHUNK_CHARS and current:
            parts.append(current)
            # Repeat the heading so every part stays self-contained.
            current = f"{heading}: {sentence}"
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def chunk_document(path: Path, domain: str) -> list[Chunk]:
    raw = clean_markdown(path.read_text(encoding="utf-8"))
    # Keep the heading attached to its body — it carries the topic name and
    # makes each chunk readable on its own once retrieved.
    sections = re.split(r"(?m)^##\s+", raw)
    chunks: list[Chunk] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading, _, body = section.partition("\n")
        heading = heading.strip()
        body = clean_markdown(body)
        if len(body) < MIN_CHUNK_CHARS:
            continue

        topic = slugify(heading)
        for i, part in enumerate(split_long(heading, body)):
            chunks.append(
                Chunk(
                    id=f"{path.stem}::{topic}::{i}",
                    text=f"{heading}. {part}",
                    metadata={
                        "domain": domain,
                        "topic": topic,
                        "heading": heading[:200],
                        "source": "astrology_knowledge",
                        "document": path.name,
                        "language": "en",
                        "text": f"{heading}. {part}",
                    },
                )
            )
    return chunks


def load_manifest(knowledge_dir: Path) -> dict[str, dict[str, str]]:
    """Optional `data/knowledge/manifest.json`, describing the PDFs.

        {"brihat_parashara.pdf": {"domain": "houses", "book": "BPHS"}}

    A PDF's retrieval domain cannot be guessed from its filename, and a chunk
    filed under the wrong domain is unreachable by filtered retrieval. So the
    mapping is declared rather than inferred, exactly as `DOCUMENT_DOMAINS`
    does for the curated text files.
    """
    path = knowledge_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        safe_print(f"  {fail_mark()} manifest.json could not be read ({type(exc).__name__}) — ignoring")
        return {}
    if not isinstance(data, dict):
        safe_print(f"  {fail_mark()} manifest.json must be an object keyed by filename — ignoring")
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def load_pdf_chunks(knowledge_dir: Path, default_domain: str | None) -> list[Chunk]:
    manifest = load_manifest(knowledge_dir)
    chunks: list[Chunk] = []

    for path in sorted(knowledge_dir.rglob("*.pdf")):
        entry = manifest.get(path.name, {})
        domain = entry.get("domain") or default_domain
        if not domain:
            safe_print(
                f"  {fail_mark()} {path.name}: no domain — add it to manifest.json "
                "or pass --pdf-domain; skipped"
            )
            continue
        if domain not in KNOWN_DOMAINS:
            safe_print(
                f"  {fail_mark()} {path.name}: domain '{domain}' is not one retrieval filters on "
                f"({', '.join(sorted(KNOWN_DOMAINS))}); skipped"
            )
            continue
        try:
            document_chunks = chunk_pdf(
                path,
                domain=domain,
                book=entry.get("book"),
                system=entry.get("system", "vedic"),
                language=entry.get("language", "en"),
            )
        except PdfExtractionError as exc:
            safe_print(f"  {fail_mark()} {exc}")
            continue
        safe_print(f"  {ok_mark()} {path.name:<22} domain={domain:<16} chunks={len(document_chunks)}")
        chunks.extend(document_chunks)

    return chunks


def load_chunks(knowledge_dir: Path, pdf_domain: str | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.glob("*.txt")):
        domain = DOCUMENT_DOMAINS.get(path.name)
        if domain is None:
            # An unregistered file would be unreachable by filtered retrieval,
            # so say so rather than silently ingesting it into a dead domain.
            safe_print(
                f"  {fail_mark()} {path.name}: no domain registered in DOCUMENT_DOMAINS — skipped"
            )
            continue
        document_chunks = chunk_document(path, domain)
        safe_print(f"  {ok_mark()} {path.name:<22} domain={domain:<16} chunks={len(document_chunks)}")
        chunks.extend(document_chunks)

    chunks.extend(load_pdf_chunks(knowledge_dir, pdf_domain))
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest astrology knowledge into Pinecone")
    parser.add_argument("--dry-run", action="store_true", help="chunk only, do not embed or upsert")
    parser.add_argument(
        "--pdf-domain",
        default=None,
        help="retrieval domain for PDFs not listed in data/knowledge/manifest.json",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="delete everything in the namespace before upserting",
    )
    args = parser.parse_args()

    safe_print(LINE)
    safe_print("TREDEV ASTRO AI")
    safe_print("KNOWLEDGE INGESTION")
    safe_print(LINE)
    safe_print()
    safe_print(f"Source     : {settings.knowledge_dir}")
    safe_print(f"Index      : {settings.pinecone.index_name}")
    safe_print(f"Namespace  : {settings.pinecone.namespace}")
    safe_print(f"Embeddings : {settings.pinecone.embedding_model} ({settings.pinecone.embedding_dimension}d)")
    safe_print()

    if not settings.knowledge_dir.exists():
        safe_print(f"{fail_mark()} knowledge directory not found: {settings.knowledge_dir}")
        return 1

    safe_print("Chunking documents:")
    chunks = load_chunks(settings.knowledge_dir, args.pdf_domain)
    safe_print()
    safe_print(f"Total chunks: {len(chunks)}")

    if not chunks:
        safe_print(f"{fail_mark()} nothing to ingest")
        return 1

    if args.dry_run:
        safe_print()
        safe_print("Dry run — sample chunk:")
        sample = chunks[0]
        safe_print(f"  id       : {sample.id}")
        safe_print(f"  metadata : {({k: v for k, v in sample.metadata.items() if k != 'text'})}")
        safe_print(f"  text     : {sample.text[:200]}...")
        safe_print()
        safe_print("No embedding or upsert performed.")
        return 0

    if not pinecone_client.configured:
        safe_print(f"{fail_mark()} PINECONE_API_KEY is not set (see ai-core/.env.example)")
        return 1

    safe_print()
    safe_print("Ensuring index exists...")
    if not pinecone_client.ensure_index():
        safe_print(f"{fail_mark()} could not create or reach index '{settings.pinecone.index_name}'")
        return 1
    index = pinecone_client.index()
    if index is None:
        safe_print(f"{fail_mark()} could not open index '{settings.pinecone.index_name}'")
        return 1
    safe_print(f"{ok_mark()} index ready")

    if args.recreate:
        safe_print(f"Clearing namespace '{settings.pinecone.namespace}'...")
        try:
            index.delete(delete_all=True, namespace=settings.pinecone.namespace)
            safe_print(f"{ok_mark()} namespace cleared")
        except Exception as exc:  # noqa: BLE001 — an empty namespace also raises
            safe_print(f"  (nothing to clear: {type(exc).__name__})")

    safe_print(f"Embedding {len(chunks)} chunks...")
    try:
        vectors = embed_documents([c.text for c in chunks])
    except EmbeddingError as exc:
        safe_print(f"{fail_mark()} embedding failed: {exc}")
        return 1
    safe_print(f"{ok_mark()} embedded ({len(vectors[0])} dimensions)")

    if len(vectors[0]) != settings.pinecone.embedding_dimension:
        safe_print(
            f"{fail_mark()} dimension mismatch: model returned {len(vectors[0])}, "
            f"EMBEDDING_DIMENSION is {settings.pinecone.embedding_dimension}. "
            "Fix EMBEDDING_DIMENSION in .env and recreate the index."
        )
        return 1

    safe_print("Upserting to Pinecone...")
    payload = [
        {"id": chunk.id, "values": vector, "metadata": chunk.metadata}
        for chunk, vector in zip(chunks, vectors)
    ]
    upserted = 0
    for start in range(0, len(payload), UPSERT_BATCH):
        batch = payload[start : start + UPSERT_BATCH]
        try:
            index.upsert(vectors=batch, namespace=settings.pinecone.namespace)
            upserted += len(batch)
            safe_print(f"  {upserted}/{len(payload)}")
        except Exception as exc:  # noqa: BLE001
            safe_print(f"{fail_mark()} upsert failed at batch starting {start}: {type(exc).__name__}: {exc}")
            return 1

    stats = pinecone_client.stats() or {}
    namespace_stats = (stats.get("namespaces") or {}).get(settings.pinecone.namespace, {})
    safe_print()
    safe_print(f"{ok_mark()} ingested {upserted} chunks")
    safe_print(f"    namespace '{settings.pinecone.namespace}' now reports "
               f"{namespace_stats.get('vector_count', 'unknown')} vectors")
    safe_print()
    safe_print("Verify with:  python -m scripts.test_connections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
