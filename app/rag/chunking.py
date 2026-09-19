"""Text extraction and chunking for offline knowledge ingestion.

Lives in `app/` rather than in `scripts/` so it can be unit-tested without a
Pinecone key and without running the ingestion CLI. Nothing here touches the
network — extraction, cleaning and splitting are pure functions over text.

The splitting strategy is structure-first, in the order the brief asks for:

    chapter / heading  ->  topic  ->  verse or rule  ->  fixed window

A fixed-size window is the last resort, not the default, because a chunk that
begins mid-sentence and ends mid-rule retrieves badly no matter how good the
embedding model is. When the fallback is used it carries an overlap so a rule
split across a boundary still appears whole in one of the two chunks.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.config.settings import settings
from app.logging_setup import get_logger

log = get_logger("rag.chunking")

# Pinecone caps the size of a metadata payload, and the text is stored in it.
MAX_CHUNK_CHARS = 1800
MIN_CHUNK_CHARS = 80

# The nine bodies the rest of the service reads. Used to tag a chunk with the
# planet it is about, so retrieval can filter on it later.
PLANETS = (
    "sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn", "rahu", "ketu",
)

_HOUSE_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}

# A line is treated as a heading when it is short, not a sentence, and either
# a markdown heading, a numbered section, or written in title/upper case.
_MD_HEADING = re.compile(r"^#{1,6}\s+(?P<text>.+?)\s*$")
_NUMBERED_HEADING = re.compile(
    r"^(?:chapter|section|part|adhyaya)\s+[\dIVXLC]+\s*[.:\-—]?\s*(?P<text>.*)$", re.IGNORECASE
)
_DECIMAL_HEADING = re.compile(r"^\d{1,2}(?:\.\d{1,2})*\s+(?P<text>[A-Z].{2,80})$")


@dataclass
class Chunk:
    """One unit of knowledge, ready to embed and upsert.

    `metadata["text"]` duplicates `text` on purpose: Pinecone returns only
    metadata on a query, so the text has to live there for retrieval to have
    anything to send to the model.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── cleaning ──────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.replace(" ", " ").replace("﻿", "")


def clean_markdown(text: str) -> str:
    """Clean a hand-authored document, preserving its line structure.

    Used for the curated `data/knowledge/*.txt` files, whose `## ` headings are
    the chunk boundaries — joining their wrapped lines would erase exactly the
    structure that makes them chunk well.
    """
    text = _normalise(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Clean extracted PDF text, repairing the artefacts extraction leaves.

    Unlike `clean_markdown` this rejoins wrapped lines, because in a PDF a
    single newline is a line wrap rather than a paragraph break. Lines that
    look like headings are left on their own line so `split_by_headings` can
    still see them.
    """
    text = _normalise(text)
    # PDFs hyphenate across line breaks; rejoin those before collapsing lines.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Join wrapped lines, but never across a heading boundary.
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if (
            out
            and out[-1]
            and not looks_like_heading(stripped)
            and not looks_like_heading(out[-1])
            and not out[-1].rstrip().endswith((".", "!", "?", ":"))
            and not stripped.startswith(("•", "-", "*"))
        ):
            out[-1] = f"{out[-1].rstrip()} {stripped}"
        else:
            out.append(stripped)
    text = "\n".join(out)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Page furniture: a line that is only a number is a page number.
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$\n?", "", text)
    return text.strip()


def slugify(value: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:limit] or "untitled"


# ── heading detection ─────────────────────────────────────────────────
def looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not (3 <= len(stripped) <= 90):
        return False
    if _MD_HEADING.match(stripped) or _NUMBERED_HEADING.match(stripped):
        return True
    if _DECIMAL_HEADING.match(stripped):
        return True
    if stripped.endswith((".", ",", ";")):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    # ALL CAPS, or Title Case with no sentence punctuation.
    if all(c.isupper() for c in letters):
        return True
    words = stripped.split()
    return 1 < len(words) <= 8 and all(w[:1].isupper() or not w[:1].isalpha() for w in words)


def heading_text(line: str) -> str:
    stripped = line.strip()
    for pattern in (_MD_HEADING, _NUMBERED_HEADING, _DECIMAL_HEADING):
        match = pattern.match(stripped)
        if match:
            return (match.group("text") or stripped).strip() or stripped
    return stripped


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split into (heading, body). Returns [("", text)] when there are none."""
    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in lines:
        if looks_like_heading(line):
            if current_body or current_heading:
                sections.append((current_heading, current_body))
            current_heading = heading_text(line)
            current_body = []
        else:
            current_body.append(line)
    if current_body or current_heading:
        sections.append((current_heading, current_body))

    result = [(h, "\n".join(b).strip()) for h, b in sections]
    result = [(h, b) for h, b in result if b or h]
    return result or [("", text)]


# ── fixed-window fallback ─────────────────────────────────────────────
def chunk_with_overlap(
    text: str,
    chunk_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[str]:
    """Split on sentence boundaries into windows of roughly `chunk_chars`.

    Boundaries are chosen between sentences, never mid-sentence, and each
    window repeats the tail of the previous one so a rule that straddles a
    boundary survives intact in at least one chunk.
    """
    chunk_chars = chunk_chars or settings.ingestion.chunk_chars
    overlap_chars = overlap_chars if overlap_chars is not None else settings.ingestion.overlap_chars
    # An overlap at or above the window size would never advance.
    overlap_chars = min(overlap_chars, max(0, chunk_chars // 2))

    text = text.strip()
    if len(text) <= chunk_chars:
        return [text] if text else []

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > chunk_chars and current:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            # Start the overlap at a sentence boundary where one exists.
            if tail:
                cut = tail.find(". ")
                tail = tail[cut + 2 :] if cut != -1 else tail
            current = f"{tail} {sentence}".strip()
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ── metadata enrichment ───────────────────────────────────────────────
def detect_planet(text: str) -> str | None:
    """The planet a chunk is chiefly about, or None when it is not about one.

    Returns a value only on a clear majority, because a wrong `planet` tag is
    worse than an absent one: retrieval can filter on it.
    """
    lowered = text.lower()
    counts = {p: len(re.findall(rf"\b{p}\b", lowered)) for p in PLANETS}
    counts = {p: n for p, n in counts.items() if n}
    if not counts:
        return None
    best, best_count = max(counts.items(), key=lambda kv: kv[1])
    total = sum(counts.values())
    return best if best_count >= 2 and best_count / total >= 0.5 else None


def detect_house(text: str) -> int | None:
    lowered = text.lower()
    found: list[int] = [int(m) for m in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)\s+house", lowered)]
    found += [n for word, n in _HOUSE_WORDS.items() if re.search(rf"\b{word}\s+house\b", lowered)]
    found = [n for n in found if 1 <= n <= 12]
    if not found:
        return None
    counts: dict[int, int] = {}
    for n in found:
        counts[n] = counts.get(n, 0) + 1
    best, best_count = max(counts.items(), key=lambda kv: kv[1])
    return best if best_count >= 2 or len(counts) == 1 else None


def build_metadata(
    *,
    text: str,
    domain: str,
    topic: str,
    document: str,
    book: str,
    chapter: str = "",
    system: str = "vedic",
    language: str = "en",
) -> dict[str, Any]:
    """Assemble a chunk's Pinecone metadata.

    Keys with no value are omitted rather than set to "" or null: Pinecone
    metadata filters treat a missing key and an empty one differently, and an
    empty `planet` on every chunk would make the field useless as a filter.
    """
    metadata: dict[str, Any] = {
        "text": text,
        "domain": domain,
        "topic": topic,
        "document": document,
        "book": book,
        "system": system,
        "language": language,
        "source": "astrology_knowledge",
    }
    if chapter:
        metadata["chapter"] = chapter[:200]
    planet = detect_planet(text)
    if planet:
        metadata["planet"] = planet
    house = detect_house(text)
    if house is not None:
        metadata["house"] = house
    return metadata


# ── PDF ────────────────────────────────────────────────────────────────
class PdfExtractionError(RuntimeError):
    pass


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF. Raises rather than returning empty silently."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise PdfExtractionError("pypdf is not installed — pip install -r requirements.txt") from exc

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 — surface the library's message
        raise PdfExtractionError(f"{path.name}: {type(exc).__name__}: {exc}") from exc

    text = "\n\n".join(p for p in pages if p.strip())
    if not text.strip():
        raise PdfExtractionError(
            f"{path.name}: no extractable text — it is probably a scanned image. "
            "OCR it before ingesting."
        )
    return text


def chunk_pdf(
    path: Path,
    domain: str,
    book: str | None = None,
    system: str = "vedic",
    language: str = "en",
) -> list[Chunk]:
    """PDF -> extract -> clean -> structure-first chunks with metadata.

    Chunk ids are deterministic (`stem::topic::n`), so re-running ingestion
    updates vectors in place instead of creating duplicates.
    """
    raw = clean_text(extract_pdf_text(path))
    book = book or path.stem.replace("_", " ").title()
    chunks: list[Chunk] = []
    seen_ids: set[str] = set()

    for heading, body in split_by_headings(raw):
        body = body.strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue
        topic = slugify(heading) if heading else "body"
        for i, part in enumerate(chunk_with_overlap(body)):
            if len(part) < MIN_CHUNK_CHARS:
                continue
            # Repeat the heading in the text so the chunk reads standalone
            # once it comes back from a vector search stripped of its context.
            text = f"{heading}. {part}" if heading else part
            text = text[:MAX_CHUNK_CHARS]
            chunk_id = f"{path.stem}::{topic}::{i}"
            # Two different sections can slugify to the same topic; keep ids
            # unique or the second would overwrite the first on upsert.
            suffix = 0
            while chunk_id in seen_ids:
                suffix += 1
                chunk_id = f"{path.stem}::{topic}::{i}-{suffix}"
            seen_ids.add(chunk_id)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    metadata=build_metadata(
                        text=text,
                        domain=domain,
                        topic=topic,
                        document=path.name,
                        book=book,
                        chapter=heading,
                        system=system,
                        language=language,
                    ),
                )
            )

    log.info("pdf chunked", extra={"document": path.name, "domain": domain, "chunks": len(chunks)})
    return chunks


def total_chars(chunks: Iterable[Chunk]) -> int:
    return sum(len(c.text) for c in chunks)


__all__ = [
    "Chunk",
    "MAX_CHUNK_CHARS",
    "MIN_CHUNK_CHARS",
    "PLANETS",
    "PdfExtractionError",
    "build_metadata",
    "chunk_pdf",
    "chunk_with_overlap",
    "clean_markdown",
    "clean_text",
    "detect_house",
    "detect_planet",
    "extract_pdf_text",
    "heading_text",
    "looks_like_heading",
    "slugify",
    "split_by_headings",
    "total_chars",
]
