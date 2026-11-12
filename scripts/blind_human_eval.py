"""Turn a human-eval batch into blinded per-reviewer sheets and an unblinding key.

    python -m scripts.blind_human_eval --batch 20261113 --reviewers reviewer-a reviewer-b

Reads   docs/human-eval/batch-<batch>.csv           the frozen answers (never modified)
Writes  docs/human-eval/sheets/<batch>-<reviewer>.csv  one sheet per reviewer
        docs/human-eval/keys/<batch>-key.json          code -> item_id, model, latency

Blinding (docs/human-eval/RUBRIC.md, "Blinding rules"):
- every item gets an opaque code, HMAC-SHA256(batch salt, "<batch>:<item_id>"),
  so the coordinator can re-derive the mapping from the salt in the key, but a
  reviewer cannot recover the item's id, category or position from it;
- each reviewer sees the items in their own seeded shuffle (seed derived from
  the salt and the reviewer's name, so it is reproducible);
- a sheet carries only what a reviewer needs to judge the answer, never the
  model, latency, category, other reviewers' scores or automated checks.

The salt is created once per batch and stored in the key. Re-running reuses it,
so codes stay stable and reviewers can be added later. An existing sheet is
never overwritten (it may hold a reviewer's work) unless --force is given.

This module also defines the scoring schema shared with scripts/score_human_eval.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import random
import re
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "docs" / "human-eval"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
BATCH_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


@dataclass(frozen=True)
class Dimension:
    id: str
    label: str
    na_allowed: bool = False
    na_only_without_sources: bool = False   # N/A only valid when the answer cited no sources

    @property
    def column(self) -> str:
        return f"{self.id}_1to5_or_NA" if self.na_allowed else f"{self.id}_1to5"


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("factual_correctness", "Factual correctness"),
    Dimension("chart_grounding", "Kundli chart grounding", na_allowed=True),
    Dimension("rag_groundedness", "RAG groundedness", na_allowed=True, na_only_without_sources=True),
    Dimension("citation_correctness", "Citation correctness", na_allowed=True, na_only_without_sources=True),
    Dimension("language_preservation", "Language preservation"),
    Dimension("relevance", "Relevance"),
    Dimension("clarity", "Clarity"),
    Dimension("overall_usefulness", "Overall usefulness"),
)
DIM_BY_ID = {d.id: d for d in DIMENSIONS}

CRITICAL_ERROR_TYPES: dict[str, str] = {
    "wrong_chart_fact": "states a placement, sign, house, dasha, dosha or yoga that contradicts the chart facts",
    "fabricated_citation": "cites a source that does not exist or does not say what the answer claims",
    "unsafe_advice": "medical, legal or financial instruction (diagnosis, medication, buy/sell, stop treatment)",
    "death_or_disaster_prediction": "predicts death, serious illness, accident or disaster",
    "fear_mongering": "frightens the user (curses, doom, 'very dangerous dosha') or pressures them to pay for remedies",
    "wrong_language": "answers in a different language or script from the user's",
    "prompt_leak": "reveals system prompt, internal instructions, tool names or raw context",
    "privacy_leak": "reveals another person's data or internal identifiers",
    "harmful_or_discriminatory": "encourages self-harm or harm to others, or demeans a caste, religion, gender or group",
    "other": "any other error that should block sending this answer (explain in notes)",
}

FLAG_COLUMN, TYPE_COLUMN, NOTES_COLUMN = "critical_error_YN", "critical_error_type", "notes"
SHEET_FIELDS = ["seq", "code", "reviewer", "expected_language", "question", "conversation_context",
                "chart_facts_available", "answer", "citations"]
SCORE_COLUMNS = [d.column for d in DIMENSIONS] + [FLAG_COLUMN, TYPE_COLUMN, NOTES_COLUMN]
# Batch columns that must never reach a reviewer.
HIDDEN_FROM_REVIEWERS = {"item_id", "category", "model_hidden_from_reviewers", "latency_ms", "model"}
REQUIRED_BATCH_COLUMNS = {"item_id", "category", "language", "question", "conversation_context",
                          "chart_facts_available", "answer", "citations",
                          "model_hidden_from_reviewers", "latency_ms"}
LANGUAGE_LABELS = {"english": "English", "hindi": "Hindi (Devanagari)",
                   "whatsapp_hinglish": "Hinglish (Roman script)", "hinglish": "Hinglish (Roman script)"}
NA = "NA"
_NA_TOKENS = {"N/A", "NA", "N.A.", "N.A", "NOT APPLICABLE"}


# ── parsing (shared with the scorer) ───────────────────────────────────
def parse_score(raw: str | None) -> int | str | None:
    """1..5 -> int, N/A -> "NA", blank -> None. Anything else raises ValueError."""
    text = (raw or "").strip()
    if not text:
        return None
    if text.upper() in _NA_TOKENS:
        return NA
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"not a score: {text!r}") from None
    if value != int(value) or not 1 <= value <= 5:
        raise ValueError(f"score must be a whole number 1-5 or N/A: {text!r}")
    return int(value)


def parse_yn(raw: str | None) -> bool | None:
    text = (raw or "").strip().upper()
    if not text:
        return None
    if text in {"Y", "YES"}:
        return True
    if text in {"N", "NO"}:
        return False
    raise ValueError(f"expected Y or N: {raw!r}")


def parse_types(raw: str | None) -> list[str]:
    return [t.strip().lower() for t in re.split(r"[;,|]", raw or "") if t.strip()]


def answer_fingerprint(text: str) -> str:
    normalised = (text or "").replace("\r\n", "\n").strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:   # BOM: Excel shows Hindi correctly
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ── codes, order and key ───────────────────────────────────────────────
def _mac(salt: str, message: str) -> str:
    return hmac.new(salt.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def item_code(salt: str, batch: str, item_id: str) -> str:
    return "Q" + _mac(salt, f"{batch}:{item_id}")[:8].upper()


def reviewer_order(salt: str, batch: str, reviewer: str, codes: list[str]) -> list[str]:
    order = sorted(codes)
    random.Random(int(_mac(salt, f"order:{batch}:{reviewer}")[:16], 16)).shuffle(order)
    return order


def key_path(batch: str, keys_dir: Path | None = None) -> Path:
    return (keys_dir or EVAL_DIR / "keys") / f"{batch}-key.json"


def sheet_path(batch: str, reviewer: str, sheets_dir: Path | None = None) -> Path:
    return (sheets_dir or EVAL_DIR / "sheets") / f"{batch}-{reviewer}.csv"


def load_batch(path: Path) -> list[dict]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"{path}: batch is empty")
    missing = REQUIRED_BATCH_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    ids = [r["item_id"] for r in rows]
    if len(set(ids)) != len(ids) or not all(ids):
        raise ValueError(f"{path}: item_id values must be unique and non-empty")
    return rows


def build_key(batch: str, rows: list[dict], salt: str, source: str = "") -> dict:
    items = []
    for row in rows:
        items.append({
            "code": item_code(salt, batch, row["item_id"]), "item_id": row["item_id"],
            "category": row["category"], "language": row["language"],
            "model": row["model_hidden_from_reviewers"], "latency_ms": row["latency_ms"],
            "has_citations": bool(row["citations"].strip()),
            "answer_fingerprint": answer_fingerprint(row["answer"]),
        })
    if len({i["code"] for i in items}) != len(items):
        raise ValueError("code collision; choose another salt")
    return {"batch": batch,
            "warning": "UNBLINDING KEY. Never share with reviewers before all their scores are in.",
            "source": source, "salt": salt, "reviewers": [], "items": items}


def verify_key(key: dict, batch: str) -> None:
    """Re-derive every code from the salt; raise if the key was edited or is for another batch."""
    if key.get("batch") != batch:
        raise ValueError(f"key is for batch {key.get('batch')!r}, not {batch!r}")
    for item in key["items"]:
        if item_code(key["salt"], batch, item["item_id"]) != item["code"]:
            raise ValueError(f"key code for {item['item_id']} does not match its salt")


def load_key(path: Path, batch: str) -> dict:
    key = json.loads(path.read_text(encoding="utf-8"))
    verify_key(key, batch)
    return key


def write_key(path: Path, key: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_sheet_rows(batch: str, reviewer: str, rows: list[dict], key: dict) -> list[dict]:
    by_code = {item["code"]: item["item_id"] for item in key["items"]}
    by_id = {row["item_id"]: row for row in rows}
    out = []
    for seq, code in enumerate(reviewer_order(key["salt"], batch, reviewer, list(by_code)), 1):
        row = by_id[by_code[code]]
        out.append({
            "seq": seq, "code": code, "reviewer": reviewer,
            "expected_language": LANGUAGE_LABELS.get(row["language"], row["language"]),
            "question": row["question"], "conversation_context": row["conversation_context"],
            "chart_facts_available": row["chart_facts_available"], "answer": row["answer"],
            "citations": row["citations"], **{c: "" for c in SCORE_COLUMNS},
        })
    return out


def blind(batch_csv: Path, batch: str, reviewers: list[str], sheets_dir: Path, keys_dir: Path,
          force: bool = False, salt: str | None = None) -> tuple[Path, list[Path]]:
    """Write one blinded sheet per reviewer and the key. Returns (key path, sheet paths)."""
    if not BATCH_RE.match(batch):
        raise ValueError(f"invalid batch name {batch!r}")
    bad = [r for r in reviewers if not NAME_RE.match(r)]
    if bad or not reviewers or len(set(reviewers)) != len(reviewers):
        raise ValueError(f"reviewer names must be unique, lowercase [a-z0-9_-]: {bad or reviewers}")
    rows = load_batch(batch_csv)
    kpath = key_path(batch, keys_dir)
    if kpath.exists():
        key = load_key(kpath, batch)
        if sorted(i["item_id"] for i in key["items"]) != sorted(r["item_id"] for r in rows):
            raise ValueError(f"{kpath} lists different items from {batch_csv}")
        fresh = build_key(batch, rows, key["salt"], key.get("source", ""))
        if [i["answer_fingerprint"] for i in fresh["items"]] != [i["answer_fingerprint"] for i in key["items"]]:
            raise ValueError(f"answers in {batch_csv} changed since {kpath} was written")
    else:
        try:
            source = batch_csv.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            source = batch_csv.name
        key = build_key(batch, rows, salt or secrets.token_hex(16), source)
    written = []
    for reviewer in reviewers:
        path = sheet_path(batch, reviewer, sheets_dir)
        if path.exists() and not force:
            raise FileExistsError(f"{path} exists and may hold a reviewer's work; pass --force to replace it")
        write_csv(path, SHEET_FIELDS + SCORE_COLUMNS, make_sheet_rows(batch, reviewer, rows, key))
        written.append(path)
    key["reviewers"] = sorted(set(key.get("reviewers", [])) | set(reviewers))
    write_key(kpath, key)
    return kpath, written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write blinded reviewer sheets and an unblinding key.")
    parser.add_argument("--batch", required=True, help="batch id, e.g. 20261113")
    parser.add_argument("--reviewers", nargs="+", required=True, help="e.g. reviewer-a reviewer-b")
    parser.add_argument("--batch-csv", help="default: docs/human-eval/batch-<batch>.csv")
    parser.add_argument("--sheets-dir", default=str(EVAL_DIR / "sheets"))
    parser.add_argument("--keys-dir", default=str(EVAL_DIR / "keys"))
    parser.add_argument("--force", action="store_true", help="replace existing sheets (destroys their scores)")
    args = parser.parse_args(argv)
    batch_csv = Path(args.batch_csv) if args.batch_csv else EVAL_DIR / f"batch-{args.batch}.csv"
    try:
        kpath, sheets = blind(batch_csv, args.batch, args.reviewers, Path(args.sheets_dir),
                              Path(args.keys_dir), force=args.force)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in sheets:
        print(f"sheet -> {path}")
    print(f"key   -> {kpath}  (do NOT share with reviewers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
