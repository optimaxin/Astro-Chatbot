"""The Hinglish Quality Benchmark's own regression tests (offline, no judge).

They pin the dataset's composition and the evaluator's behaviour, so a change
to the rules or the scoring that makes the benchmark blind (gold text
flagged, a negative missed, a correction ranked below its original, a wrong
fact passing) fails here before it can mislead a release comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.hinglish_quality import BENCHMARK_VERSION, dataset_hash, load_examples
from benchmarks.hinglish_quality.facts import check_facts
from benchmarks.hinglish_quality.rules import check
from benchmarks.hinglish_quality.schema import CATEGORIES
from benchmarks.hinglish_quality.scoring import GATE_CAP, WEIGHTS, score

EXAMPLES = load_examples()


def _score(text: str, language: str, reference: str | None = None):
    return score(check(text, language), check_facts(text, reference), None)


def test_dataset_composition() -> None:
    count = lambda **kw: sum(1 for e in EXAMPLES if all(getattr(e, k) == v for k, v in kw.items()))  # noqa: E731
    assert len(EXAMPLES) >= 200
    assert count(language="hinglish", domain="astrology") >= 100
    assert count(language="hinglish", domain="conversational") >= 50
    assert count(language="hindi") >= 25 and count(language="english") >= 25
    assert count(kind="negative") >= 80
    assert all(e.corrected and e.errors for e in EXAMPLES if e.kind == "negative")
    topics = {e.topic for e in EXAMPLES}
    assert {"career", "marriage", "health", "property", "dasha", "placement", "chat"} <= topics
    assert sum(1 for e in EXAMPLES if e.source == "screenshot") >= 5


def test_every_annotated_category_is_known_and_has_a_dimension() -> None:
    for e in EXAMPLES:
        for error in e.errors:
            assert error.category in CATEGORIES, (e.id, error.category)


def test_the_dataset_fingerprint_is_stable() -> None:
    assert dataset_hash(EXAMPLES) == dataset_hash(load_examples())
    assert BENCHMARK_VERSION.startswith("hqb-")


def test_weights_favour_meaning_and_grammar() -> None:
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
    assert WEIGHTS["meaning"] > WEIGHTS["grammar"] > max(WEIGHTS["vocabulary"], WEIGHTS["naturalness"],
                                                         WEIGHTS["consistency"])


@pytest.mark.parametrize("example", [e for e in EXAMPLES if e.kind == "gold"], ids=lambda e: e.id)
def test_gold_text_passes(example) -> None:
    result = _score(example.text, example.language, example.text)
    assert result.passed and not result.errors, result.errors


@pytest.mark.parametrize("example", [e for e in EXAMPLES if e.kind == "negative"], ids=lambda e: e.id)
def test_negative_is_caught_and_its_correction_ranks_higher(example) -> None:
    bad = _score(example.text, example.language, example.reference)
    good = _score(example.corrected, example.language, example.corrected)
    assert bad.errors and not bad.passed
    assert good.passed and good.overall > bad.overall


def test_a_natural_reply_cannot_hide_a_wrong_fact() -> None:
    wrong = _score("Saturn aapke 7th house mein hai, aur yeh rishton mein dheeraj sikhata hai.", "hinglish")
    assert wrong.gated and wrong.overall <= GATE_CAP and not wrong.passed
    assert wrong.dimensions["naturalness"] == 5.0          # the language itself is fine


def test_an_invented_year_is_a_fact_error() -> None:
    assert check_facts("Aapki shaadi 2025 mein ho sakti hai.").fact_change
    assert not check_facts("Venus Antardasha June 2024 se April 2027 tak hai.").fact_change


def test_a_wrong_house_lord_is_a_fact_error() -> None:
    assert check_facts("Your 10th lord Mercury sits in your 6th house.").contradictions
    assert not check_facts("Your 10th lord Jupiter sits in your 6th house.").contradictions


def test_the_judge_cannot_fail_a_clean_reply_alone() -> None:
    text = "Aapka Lagna Pisces hai, aur iska lord Jupiter 6th house mein hai."
    harsh = {d: 1 for d in WEIGHTS} | {"errors": [{"category": "word_choice", "span": "x", "fix": "y"}]}
    result = score(check(text, "hinglish"), check_facts(text), harsh)
    assert min(result.dimensions.values()) == 3.0 and not result.gated


def test_compare_flags_a_regression(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("hqb", Path(__file__).parents[1] / "scripts" /
                                                  "hinglish_quality_bench.py")
    hqb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hqb)
    summary = {"overall": 4.6, "grammar": 4.5, "naturalness": 4.5, "vocabulary": 4.6, "meaning": 4.9,
               "consistency": 4.9, "pass_rate_%": 80.0, "fact_gate_%": 1.0, "errors_total": 10}
    base = {"meta": {"label": "base", "dataset_hash": "x"}, "summary": {"all": summary}}
    worse = {"meta": {"label": "new", "dataset_hash": "x"},
             "summary": {"all": {**summary, "overall": 4.3, "grammar": 4.1}}}
    (tmp_path / "a.json").write_text(json.dumps(base))
    (tmp_path / "b.json").write_text(json.dumps(worse))
    assert hqb.compare(str(tmp_path / "a.json"), str(tmp_path / "b.json"), fail=True) == 1
    assert hqb.compare(str(tmp_path / "a.json"), str(tmp_path / "a.json"), fail=True) == 0


def test_the_report_renders(tmp_path: Path) -> None:
    from benchmarks.hinglish_quality.report import write_report

    example = EXAMPLES[0]
    row = {"id": example.id, "variant": "text", "language": example.language, "kind": example.kind,
           "question": example.question, "text": example.text,
           "score": _score(example.text, example.language, example.text).to_dict()}
    run = {"meta": {"label": "t", "mode": "dataset", "created": "now", "benchmark_version": BENCHMARK_VERSION,
                    "dataset_hash": "x", "judge": {"used": False}, "git_commit": "abc", "git_dirty": False},
           "rows": [row]}
    (tmp_path / "run.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    html = write_report([tmp_path / "run.json"], tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Hinglish Quality Benchmark" in html and example.id in html and "<\\/" not in html.split("RUNS =")[0]
