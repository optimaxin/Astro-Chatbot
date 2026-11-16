"""TradevaAstro Hinglish Quality Benchmark (HQB).

An independent, repeatable evaluation of how natural, grammatical and correct
the chatbot's Hinglish (and Hindi / English) is. It never changes chatbot
behaviour: it reads replies, it does not produce them.

    dataset   250 examples (schema.py): 150 Hinglish (100 astrology + 50
              conversational), 50 Hindi, 50 English; gold replies plus
              negatives with annotated errors and corrected alternatives.
    rules     deterministic grammar / spelling / consistency checks (rules.py)
    facts     Kundli-fact preservation against the benchmark chart (facts.py)
    judge     LLM naturalness judge, cached for repeatability (judge.py)
    scoring   five 1–5 dimensions, weighted, with a fact gate (scoring.py)
    runner    scripts/hinglish_quality_bench.py — modes, storage, regression
              comparison and the HTML dashboard (report.py)

Methodology: docs/hinglish-quality-benchmark.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.hinglish_quality.schema import Example, assign_split

BENCHMARK_VERSION = "hqb-1.0"
_HERE = Path(__file__).resolve().parent


def load_examples() -> list[Example]:
    from benchmarks.hinglish_quality import data_english, data_hindi, data_hinglish_astro, data_hinglish_chat

    examples = [assign_split(e) for module in (data_hinglish_astro, data_hinglish_chat, data_hindi, data_english)
                for e in module.EXAMPLES]
    reviews = load_reviews()
    if reviews:
        examples = [Example(**{**e.__dict__, "review": reviews[e.id]["status"]}) if e.id in reviews else e
                    for e in examples]
    ids = [e.id for e in examples]
    assert len(ids) == len(set(ids)), "duplicate example id"
    return examples


def load_reviews() -> dict[str, dict]:
    """Native-speaker sign-offs, imported from reviewer sheets
    (scripts/hinglish_quality_bench.py import-review)."""
    path = _HERE / "reviews.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def dataset_hash(examples: list[Example]) -> str:
    """Fingerprint of the dataset, stored with every result so two runs are
    only compared on the same data."""
    blob = json.dumps([e.__dict__ | {"errors": [x.__dict__ for x in e.errors]} for e in examples],
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
