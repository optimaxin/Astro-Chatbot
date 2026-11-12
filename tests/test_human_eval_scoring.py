"""The blinded human-eval workflow, on synthetic fixtures only.

Every score in this file is made up to exercise arithmetic and validation.
None of it is an evaluation result, and none of it is written under
docs/human-eval/ (everything goes to pytest's tmp_path).
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from scripts.blind_human_eval import (
    DIMENSIONS, EVAL_DIR, HIDDEN_FROM_REVIEWERS, SCORE_COLUMNS, blind, item_code, key_path,
    load_key, parse_score, read_csv,
)
from scripts.build_human_eval import FOLLOW_UPS, SINGLE
from scripts.score_human_eval import (
    NO_REVIEWS, krippendorff_alpha_ordinal, quadratic_kappa, score,
)

BATCH = "29990101"
SALT = "test-salt-not-secret"
CITED = {"T003", "T004"}          # items whose answers cited sources


def _batch_csv(path: Path, n: int = 8) -> Path:
    fields = ["item_id", "category", "language", "question", "conversation_context",
              "chart_facts_available", "answer", "citations", "model_hidden_from_reviewers", "latency_ms"]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(1, n + 1):
            item = f"T{i:03d}"
            writer.writerow({
                "item_id": item, "category": "career" if i % 2 else "hindi",
                "language": "english" if i % 2 else "hindi", "question": f"question {i}",
                "conversation_context": "", "chart_facts_available": "lagna=Virgo",
                "answer": f"synthetic answer {i}\nsecond line",
                "citations": "Houses guide" if item in CITED else "",
                "model_hidden_from_reviewers": "secret-model-x", "latency_ms": str(1000 + i)})
    return path


@pytest.fixture
def setup(tmp_path):
    batch_csv = _batch_csv(tmp_path / f"batch-{BATCH}.csv")
    kpath, sheets = blind(batch_csv, BATCH, ["reviewer-a", "reviewer-b"], tmp_path / "sheets",
                          tmp_path / "keys", salt=SALT)
    return tmp_path, batch_csv, kpath, sheets


def _fill(sheet: Path, kpath: Path, scores: dict[str, int | str] | None = None,
          overrides: dict[str, dict[str, str]] | None = None) -> None:
    """Fill every row with synthetic values; `overrides` is keyed by item_id."""
    key = load_key(kpath, BATCH)
    item_of = {i["code"]: i["item_id"] for i in key["items"]}
    rows = read_csv(sheet)
    for row in rows:
        item = item_of[row["code"]]
        for dim in DIMENSIONS:
            value = (scores or {}).get(dim.id, 4)
            if dim.na_only_without_sources and item not in CITED:
                value = "N/A"
            row[dim.column] = str(value)
        row["critical_error_YN"] = "N"
        row.update((overrides or {}).get(item, {}))
    with open(sheet, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ── statistics ─────────────────────────────────────────────────────────
def test_kappa_is_one_for_perfect_agreement() -> None:
    assert quadratic_kappa([1, 2, 3, 4, 5, 3], [1, 2, 3, 4, 5, 3]) == pytest.approx(1.0)


def test_kappa_hand_computed_case() -> None:
    # observed sum of squared differences = 1; expected = Σ (i-j)² r_i c_j / n = 9/3 = 3 → κ = 1 - 1/3
    assert quadratic_kappa([1, 2, 3], [1, 2, 2]) == pytest.approx(2 / 3)


def test_kappa_undefined_without_variation() -> None:
    assert quadratic_kappa([4, 4, 4], [4, 4, 4]) is None
    assert quadratic_kappa([], []) is None


def test_krippendorff_alpha_known_values() -> None:
    assert krippendorff_alpha_ordinal([[1, 1], [2, 2], [3, 3]]) == pytest.approx(1.0)
    # coincidences o23=o32=1, n=(2,3,1); D_o = 8, D_e = 180 → α = 1 - 5·8/180 = 7/9
    assert krippendorff_alpha_ordinal([[1, 1], [2, 2], [3, 2]]) == pytest.approx(7 / 9)
    assert krippendorff_alpha_ordinal([[3], [4]]) is None     # no pairable values


def test_score_parsing() -> None:
    assert parse_score("4") == 4 and parse_score(" 5.0 ") == 5
    assert parse_score("n/a") == "NA" and parse_score("NA") == "NA"
    assert parse_score("") is None
    for bad in ("0", "6", "3.5", "good"):
        with pytest.raises(ValueError):
            parse_score(bad)


# ── blinding ───────────────────────────────────────────────────────────
def test_sheets_are_blinded_and_shuffled_per_reviewer(setup) -> None:
    tmp_path, batch_csv, kpath, (sheet_a, sheet_b) = setup
    rows_a, rows_b = read_csv(sheet_a), read_csv(sheet_b)
    assert not HIDDEN_FROM_REVIEWERS & set(rows_a[0])
    text = sheet_a.read_text(encoding="utf-8-sig")
    assert "secret-model-x" not in text and "1001" not in text and "T001" not in text
    codes_a = [r["code"] for r in rows_a]
    assert sorted(codes_a) == sorted(r["code"] for r in rows_b)
    assert codes_a != [r["code"] for r in rows_b]                # different order per reviewer
    assert all(not any(r[c] for c in SCORE_COLUMNS) for r in rows_a + rows_b)   # empty score columns
    key = load_key(kpath, BATCH)
    assert all(i["code"] != i["item_id"] and i["item_id"] not in i["code"] for i in key["items"])
    batch_order = [item_code(SALT, BATCH, f"T{i:03d}") for i in range(1, 9)]
    assert codes_a != batch_order


def test_key_round_trips_and_codes_are_stable(setup) -> None:
    tmp_path, batch_csv, kpath, (sheet_a, _) = setup
    key = load_key(kpath, BATCH)
    assert {i["code"]: i["item_id"] for i in key["items"]} == {
        item_code(key["salt"], BATCH, f"T{i:03d}"): f"T{i:03d}" for i in range(1, 9)}
    assert {i["model"] for i in key["items"]} == {"secret-model-x"}
    by_code = {i["code"]: i for i in key["items"]}
    for row in read_csv(sheet_a):
        assert row["question"] == f"question {int(by_code[row['code']]['item_id'][1:])}"
    # adding a reviewer later reuses the salt; an existing sheet is never overwritten
    blind(batch_csv, BATCH, ["reviewer-c"], tmp_path / "sheets", tmp_path / "keys")
    assert load_key(kpath, BATCH)["salt"] == SALT
    assert load_key(kpath, BATCH)["reviewers"] == ["reviewer-a", "reviewer-b", "reviewer-c"]
    with pytest.raises(FileExistsError):
        blind(batch_csv, BATCH, ["reviewer-a"], tmp_path / "sheets", tmp_path / "keys")


def test_blinding_never_modifies_the_batch(tmp_path) -> None:
    batch_csv = _batch_csv(tmp_path / "batch.csv")
    before = hashlib.sha256(batch_csv.read_bytes()).hexdigest()
    blind(batch_csv, BATCH, ["reviewer-a"], tmp_path / "s", tmp_path / "k", salt=SALT)
    assert hashlib.sha256(batch_csv.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError):
        blind(batch_csv, BATCH, ["../evil"], tmp_path / "s", tmp_path / "k")


# ── scoring and the gate ───────────────────────────────────────────────
def test_no_reviews_report(setup) -> None:
    tmp_path, _, kpath, sheets = setup
    text = score(BATCH, sheets, kpath)
    assert NO_REVIEWS in text and "NO HUMAN REVIEWS RECORDED — beta gate NOT PASSED" in text
    assert "Mean" not in text and "Beta gate (proposal)" not in text and "| Factual correctness |" not in text


def test_refuses_gate_with_one_complete_reviewer(setup) -> None:
    tmp_path, _, kpath, (sheet_a, sheet_b) = setup
    _fill(sheet_a, kpath)
    text = score(BATCH, [sheet_a, sheet_b], kpath)
    assert "Beta gate NOT EVALUATED" in text and "1 complete reviewer sheet" in text
    assert "| Factual correctness |" not in text and "Krippendorff" not in text
    assert "| reviewer-a |" in text and "complete" in text


def test_na_rules_and_missing_cells_are_reported(setup) -> None:
    tmp_path, _, kpath, (sheet_a, sheet_b) = setup
    _fill(sheet_a, kpath, overrides={
        "T003": {"citation_correctness_1to5_or_NA": "N/A"},       # cited sources: N/A not allowed
        "T005": {"factual_correctness_1to5": "N/A"},               # never N/A
        "T006": {"clarity_1to5": ""},                              # blank: missing, not imputed
        "T007": {"relevance_1to5": "7"}})
    _fill(sheet_b, kpath)
    text = score(BATCH, [sheet_a, sheet_b], kpath)
    assert "T003 (" in text and "N/A not allowed, the answer cited sources" in text
    assert "factual_correctness: N/A is not allowed" in text
    assert "1 blank cell(s): T006" in text and "relevance: score must be a whole number" in text
    assert "Beta gate NOT EVALUATED" in text


def test_two_reviewers_full_report(setup) -> None:
    tmp_path, _, kpath, (sheet_a, sheet_b) = setup
    _fill(sheet_a, kpath, {"overall_usefulness": 5, "factual_correctness": 5},
          overrides={"T002": {"overall_usefulness_1to5": "2"}})
    _fill(sheet_b, kpath, {"overall_usefulness": 5, "factual_correctness": 4}, overrides={
        "T004": {"critical_error_YN": "Y", "critical_error_type": "fabricated_citation",
                 "notes": "cites 'Houses guide' for a claim it does not make"}})
    text = score(BATCH, [sheet_a, sheet_b], kpath)
    assert "| Overall usefulness | 4.81 |" in text                 # (7*5 + 2 + 8*5) / 16
    assert "| Factual correctness | 4.50 |" in text
    assert "| RAG groundedness | 4.00 |" in text and "| 4 | 12 |" in text   # 4 scored, 12 N/A
    assert "QW Cohen's κ" in text and "Krippendorff α (ordinal)" in text
    assert "| T002 | Overall usefulness | reviewer-a=2, reviewer-b=5 |" in text
    assert "| T004 | reviewer-b | fabricated_citation |" in text and "| unresolved |" in text
    assert "### By category" in text and "### By language" in text
    assert "Beta gate (proposal): **NOT PASSED**" in text
    assert "| Unresolved critical errors = 0 | 1 | **FAIL** |" in text

    adjudication = tmp_path / "adj.csv"
    adjudication.write_text("item_id,resolution,adjudicator,note\nT004,not_an_error,lead,source does say it\n",
                            encoding="utf-8")
    assert "Beta gate (proposal): **PASSED**" in score(BATCH, [sheet_a, sheet_b], kpath, adjudication)


def test_critical_error_requires_type_and_note(setup) -> None:
    tmp_path, _, kpath, (sheet_a, sheet_b) = setup
    _fill(sheet_a, kpath, overrides={"T001": {"critical_error_YN": "Y", "critical_error_type": "scary"}})
    _fill(sheet_b, kpath)
    text = score(BATCH, [sheet_a, sheet_b], kpath)
    assert "critical error needs a type from the taxonomy" in text
    assert "critical error needs a note" in text and "Beta gate NOT EVALUATED" in text


# ── the committed artefacts ────────────────────────────────────────────
def test_batch_has_at_least_fifty_items_across_required_categories() -> None:
    items = len(SINGLE) + len(FOLLOW_UPS)
    assert items >= 50
    cats = {c for c, _ in SINGLE} | {c for c, *_ in FOLLOW_UPS}
    assert {"career", "marriage", "finance", "general", "hindi", "hinglish", "follow-up"} <= cats


def test_committed_sheets_are_blinded() -> None:
    sheets = sorted((EVAL_DIR / "sheets").glob("20261113-*.csv"))
    if not sheets:
        pytest.skip("no committed reviewer sheets")
    key = load_key(key_path("20261113"), "20261113")
    codes = {i["code"] for i in key["items"]}
    for sheet in sheets:
        rows = read_csv(sheet)
        assert not HIDDEN_FROM_REVIEWERS & set(rows[0])
        assert {r["code"] for r in rows} == codes and len(rows) == 60
