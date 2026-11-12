"""Aggregate blinded human review sheets. Never fills in or estimates a score.

    python -m scripts.score_human_eval --batch 20261113
    python -m scripts.score_human_eval --batch 20261113 --sheets a.csv b.csv --out -

Input: the reviewer sheets written by scripts/blind_human_eval.py and filled in
by people (rubric: docs/human-eval/RUBRIC.md), plus the unblinding key, plus an
optional adjudication file (docs/human-eval/adjudication-<batch>.csv) that
resolves flagged critical errors.

Output (docs/human-eval/report-<batch>.md by default): validation problems,
per-dimension means with 95% confidence intervals, a per-reviewer summary,
means by category / language / model, inter-reviewer agreement (exact %,
within-1 %, quadratic-weighted Cohen's kappa, Krippendorff's alpha), every
flagged critical error, items to adjudicate, and the beta-gate verdict.

Blank cells are reported as missing, never imputed. No score is computed
unless at least two reviewers have completed every item. Every number in the
report comes from a person; automated checks are deliberately not included.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import fmean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.blind_human_eval import (  # noqa: E402
    CRITICAL_ERROR_TYPES, DIMENSIONS, EVAL_DIR, FLAG_COLUMN, NA, NOTES_COLUMN, ROOT, TYPE_COLUMN,
    answer_fingerprint, key_path, load_key, parse_score, parse_types, parse_yn, read_csv,
)

# Beta gate — a proposal the team can tune (docs/human-eval/RUBRIC.md, "Beta gate").
GATE = {"min_complete_reviewers": 2, "overall_usefulness": 4.0, "factual_correctness": 4.0,
        "max_unresolved_critical": 0}
LOW_AGREEMENT = 0.40           # kappa / alpha below this is flagged (not gating)
RESOLVED = {"not_an_error", "fixed"}
RESOLUTIONS = RESOLVED | {"confirmed"}
NO_REVIEWS = "NO HUMAN REVIEWS RECORDED — beta gate NOT PASSED"
SHORT = {"factual_correctness": "fact", "chart_grounding": "chart", "rag_groundedness": "rag",
         "citation_correctness": "cite", "language_preservation": "lang", "relevance": "rel",
         "clarity": "clar", "overall_usefulness": "useful"}
MAX_LISTED = 25


@dataclass
class Review:
    reviewer: str
    file: str
    scores: dict[str, dict[str, int | str]] = field(default_factory=dict)   # item_id -> dim -> 1..5 | NA
    flags: dict[str, tuple[bool, list[str], str]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    filled: int = 0
    n_items: int = 0

    @property
    def complete(self) -> bool:
        return not self.problems and self.filled > 0 and len(self.scores) == self.n_items


# ── statistics ─────────────────────────────────────────────────────────
def _ci(values: list[float]) -> str:
    if len(values) < 2:
        return "n/a"
    return f"±{1.96 * stdev(values) / math.sqrt(len(values)):.2f}"


def quadratic_kappa(a: list[int], b: list[int]) -> float | None:
    """Quadratic-weighted Cohen's kappa for two raters. None when undefined (no variation)."""
    if len(a) != len(b) or not a:
        return None
    ra, rb, n = Counter(a), Counter(b), len(a)
    observed = sum((x - y) ** 2 for x, y in zip(a, b))
    expected = sum((i - j) ** 2 * ra[i] * rb[j] for i in ra for j in rb) / n
    return None if expected == 0 else 1 - observed / expected


def krippendorff_alpha_ordinal(units: list[list[int]], values: range = range(1, 6)) -> float | None:
    """Krippendorff's alpha with the ordinal metric; units = scores per item (any number of raters)."""
    units = [u for u in units if len(u) >= 2]
    coincidence: dict[tuple[int, int], float] = defaultdict(float)
    for unit in units:
        for i, c in enumerate(unit):
            for j, k in enumerate(unit):
                if i != j:
                    coincidence[(c, k)] += 1 / (len(unit) - 1)
    n_c = {c: sum(coincidence[(c, k)] for k in values) for c in values}
    n = sum(n_c.values())
    if n <= 1:
        return None

    def delta(c: int, k: int) -> float:
        lo, hi = min(c, k), max(c, k)
        return (sum(n_c[g] for g in range(lo, hi + 1)) - (n_c[c] + n_c[k]) / 2) ** 2

    d_obs = sum(coincidence[(c, k)] * delta(c, k) for c in values for k in values)
    d_exp = sum(n_c[c] * n_c[k] * delta(c, k) for c in values for k in values)
    return None if d_exp == 0 else 1 - (n - 1) * d_obs / d_exp


# ── loading and validation ─────────────────────────────────────────────
def load_review(path: Path, key: dict) -> Review:
    rows = read_csv(path)
    items = {i["code"]: i for i in key["items"]}
    named = {(r.get("reviewer") or "").strip() for r in rows} - {""}
    stem = path.stem[len(key["batch"]) + 1:] if path.stem.startswith(key["batch"] + "-") else path.stem
    review = Review(reviewer=named.pop() if len(named) == 1 else stem, file=path.name, n_items=len(items))
    if len(named) > 0:
        review.problems.append("more than one name in the reviewer column")
    seen: set[str] = set()
    missing: list[str] = []
    for row in rows:
        code = (row.get("code") or "").strip()
        item = items.get(code)
        if item is None or code in seen:
            review.problems.append(f"row seq={row.get('seq')}: {'duplicate' if code in seen else 'unknown'} code {code!r}")
            continue
        seen.add(code)
        where = f"{item['item_id']} ({code})"
        if answer_fingerprint(row.get("answer", "")) != item["answer_fingerprint"]:
            review.warnings.append(f"{where}: answer text differs from the frozen batch")
        scores: dict[str, int | str] = {}
        for dim in DIMENSIONS:
            raw = row.get(dim.column)
            review.filled += bool((raw or "").strip())
            try:
                value = parse_score(raw)
            except ValueError as exc:
                review.problems.append(f"{where} {dim.id}: {exc}")
                continue
            if value is None:
                missing.append(f"{where} {dim.id}")
            elif value == NA and not dim.na_allowed:
                review.problems.append(f"{where} {dim.id}: N/A is not allowed for this dimension")
            elif value == NA and dim.na_only_without_sources and item["has_citations"]:
                review.problems.append(f"{where} {dim.id}: N/A not allowed, the answer cited sources")
            else:
                scores[dim.id] = value
        flag, types, note = _read_flag(row, where, review, missing)
        if len(scores) == len(DIMENSIONS) and flag is not None:
            review.scores[item["item_id"]] = scores
        if flag is not None:
            review.flags[item["item_id"]] = (flag, types, note)
    absent = [f"{i['item_id']} ({c})" for c, i in items.items() if c not in seen]
    if review.filled == 0:
        review.problems.insert(0, "not started (no scores entered)")
        return review
    if absent:
        review.problems.append(f"{len(absent)} item(s) missing from the sheet: {', '.join(absent[:10])}")
    if missing:
        review.problems.append(f"{len(missing)} blank cell(s): " + "; ".join(missing[:MAX_LISTED])
                               + (" …" if len(missing) > MAX_LISTED else ""))
    return review


def _read_flag(row: dict, where: str, review: Review, missing: list[str]):
    raw = row.get(FLAG_COLUMN)
    review.filled += bool((raw or "").strip())
    try:
        flag = parse_yn(raw)
    except ValueError as exc:
        review.problems.append(f"{where} {FLAG_COLUMN}: {exc}")
        return None, [], ""
    types, note = parse_types(row.get(TYPE_COLUMN)), (row.get(NOTES_COLUMN) or "").strip()
    if flag is None:
        missing.append(f"{where} {FLAG_COLUMN}")
    elif flag:
        unknown = [t for t in types if t not in CRITICAL_ERROR_TYPES]
        if not types or unknown:
            review.problems.append(f"{where}: critical error needs a type from the taxonomy (got {unknown or 'none'})")
        if not note:
            review.problems.append(f"{where}: critical error needs a note quoting the phrase")
    elif types:
        review.problems.append(f"{where}: critical_error_type given but critical_error_YN is N")
    return flag, types, note


def load_adjudication(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    out = {}
    for row in read_csv(path):
        item = (row.get("item_id") or "").strip()
        resolution = (row.get("resolution") or "").strip().lower()
        if item:
            out[item] = {"resolution": resolution if resolution in RESOLUTIONS else f"invalid:{resolution}",
                         "adjudicator": (row.get("adjudicator") or "").strip(),
                         "note": (row.get("note") or "").strip()}
    return out


# ── report ─────────────────────────────────────────────────────────────
def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _numeric(reviews: list[Review], dim: str, items: set[str] | None = None) -> list[int]:
    return [s[dim] for r in reviews for i, s in r.scores.items()
            if (items is None or i in items) and isinstance(s[dim], int)]


def _critical(reviews: list[Review], adjudication: dict) -> tuple[list[str], int]:
    lines, flagged = [], set()
    for r in reviews:
        for item, (flag, types, note) in sorted(r.flags.items()):
            if flag:
                flagged.add(item)
                res = adjudication.get(item, {}).get("resolution", "unresolved")
                lines.append(f"| {item} | {r.reviewer} | {', '.join(types) or '—'} | {_cell(note) or '—'} | {res} |")
    unresolved = sum(1 for i in flagged if adjudication.get(i, {}).get("resolution") not in RESOLVED)
    head = ["| Item | Reviewer | Type | Note | Resolution |", "|---|---|---|---|---|"]
    return (head + lines if lines else ["No reviewer flagged a critical error."]), unresolved


def _status_section(reviews: list[Review]) -> list[str]:
    out = ["## Reviewer sheets", "", "| Reviewer | File | Status | Items scored |", "|---|---|---|---|"]
    for r in reviews:
        status = "complete" if r.complete else ("not started" if r.filled == 0 else "INCOMPLETE / INVALID")
        out.append(f"| {r.reviewer} | {r.file} | {status} | {len(r.scores)}/{r.n_items} |")
    for r in reviews:
        if r.filled and (r.problems or r.warnings):
            out += ["", f"**{r.reviewer}**: {len(r.problems)} problem(s), {len(r.warnings)} warning(s)"]
            out += [f"- {p}" for p in r.problems[:MAX_LISTED]] + [f"- warning: {w}" for w in r.warnings[:10]]
    return out


def _group_table(reviews: list[Review], key: dict, attribute: str) -> list[str]:
    groups: dict[str, set[str]] = defaultdict(set)
    for item in key["items"]:
        groups[item[attribute]].add(item["item_id"])
    out = [f"### By {attribute}", "", "| " + attribute + " | items | " + " | ".join(SHORT[d.id] for d in DIMENSIONS) + " |",
           "|---|---|" + "---|" * len(DIMENSIONS)]
    for name, ids in sorted(groups.items()):
        means = [_fmt(fmean(v)) if (v := _numeric(reviews, d.id, ids)) else "N/A" for d in DIMENSIONS]
        out.append(f"| {name} | {len(ids)} | " + " | ".join(means) + " |")
    return out + [""]


def _agreement(reviews: list[Review]) -> tuple[list[str], list[str], dict[str, float | None]]:
    rows = ["| Dimension | Pairs | Exact | Within 1 | " + ("QW Cohen's κ" if len(reviews) == 2 else "Mean pairwise QW κ")
            + " | Krippendorff α (ordinal) | N/A vs score |", "|---|---|---|---|---|---|---|"]
    disagreements, kappas = [], {}
    items = sorted(set.intersection(*(set(r.scores) for r in reviews)))
    for dim in DIMENSIONS:
        exact = within = pairs = na_mismatch = 0
        pair_kappas = []
        for ra, rb in combinations(reviews, 2):
            xs, ys = [], []
            for item in items:
                a, b = ra.scores[item][dim.id], rb.scores[item][dim.id]
                if isinstance(a, int) and isinstance(b, int):
                    xs.append(a)
                    ys.append(b)
                    exact += a == b
                    within += abs(a - b) <= 1
                elif (a == NA) != (b == NA):
                    na_mismatch += 1
            pairs += len(xs)
            pair_kappas.append(quadratic_kappa(xs, ys))
        defined = [k for k in pair_kappas if k is not None]
        kappa = fmean(defined) if defined else None
        units = [[r.scores[i][dim.id] for r in reviews if isinstance(r.scores[i][dim.id], int)] for i in items]
        alpha = krippendorff_alpha_ordinal(units)
        kappas[dim.id] = kappa
        pct = (lambda x: f"{100 * x / pairs:.0f}%") if pairs else (lambda x: "—")
        rows.append(f"| {dim.label} | {pairs} | {pct(exact)} | {pct(within)} | {_fmt(kappa)} | {_fmt(alpha)} | {na_mismatch} |")
        for item in items:
            vals = [r.scores[item][dim.id] for r in reviews]
            nums = [v for v in vals if isinstance(v, int)]
            if (nums and max(nums) - min(nums) >= 2) or (NA in vals and nums):
                shown = ", ".join(f"{r.reviewer}={r.scores[item][dim.id]}" for r in reviews)
                disagreements.append(f"| {item} | {dim.label} | {shown} |")
    return rows, disagreements, kappas


def build_report(batch: str, reviews: list[Review], key: dict, adjudication: dict | None = None,
                 sources: list[str] | None = None) -> str:
    adjudication = adjudication or {}
    head = [f"# Human evaluation report — batch {batch}", ""]
    files = ", ".join(sources or [r.file for r in reviews]) or "none"
    critical, unresolved = _critical(reviews, adjudication)
    tail = ["", "## Automated checks", "",
            "Not included. Every number in this report was entered by a human reviewer; automated and "
            "model-judge results live in separate reports and must never be presented as human scores."]
    if not any(r.filled for r in reviews):
        return "\n".join(head + [f"**{NO_REVIEWS}**", "",
                                 f"Sheets read: {files}. No reviewer has entered any score, so no score "
                                 "is shown. Fill the reviewer sheets (docs/human-eval/RUBRIC.md) and re-run.",
                                 ""] + _status_section(reviews) + tail) + "\n"
    complete = [r for r in reviews if r.complete]
    names = [r.reviewer for r in complete]
    if len(set(names)) != len(names):   # the same person twice is not two independent reviewers
        complete = []
        reviews[0].warnings.append("the same reviewer name appears on more than one complete sheet")
    if len(complete) < GATE["min_complete_reviewers"]:
        return "\n".join(head + [
            f"**Beta gate NOT EVALUATED (therefore NOT PASSED): {len(complete)} complete reviewer sheet(s); "
            f"at least {GATE['min_complete_reviewers']} distinct reviewers must complete all {len(key['items'])} items.**",
            "", "No scores are shown until then.", ""] + _status_section(reviews)
            + ["", "## Critical errors flagged so far", ""] + critical + tail) + "\n"

    out = head + ["Scores from " + ", ".join(names) + f" ({len(key['items'])} items each). Sheets read: {files}.", ""]
    out += ["## Scores (all complete reviewers pooled)", "",
            "| Dimension | Mean | 95% CI | Scored | N/A |", "|---|---|---|---|---|"]
    means = {}
    for dim in DIMENSIONS:
        vals = _numeric(complete, dim.id)
        n_na = sum(1 for r in complete for s in r.scores.values() if s[dim.id] == NA)
        means[dim.id] = fmean(vals) if vals else None
        out.append(f"| {dim.label} | {_fmt(means[dim.id])} | {_ci(vals)} | {len(vals)} | {n_na} |")
    out += ["", "## Per-reviewer summary", "",
            "| Reviewer | " + " | ".join(SHORT[d.id] for d in DIMENSIONS) + " | N/A cells | critical flags |",
            "|---|" + "---|" * (len(DIMENSIONS) + 2)]
    for r in complete:
        cells = [_fmt(fmean(v)) if (v := _numeric([r], d.id)) else "N/A" for d in DIMENSIONS]
        n_na = sum(1 for s in r.scores.values() for v in s.values() if v == NA)
        out.append(f"| {r.reviewer} | " + " | ".join(cells) + f" | {n_na} | {sum(f[0] for f in r.flags.values())} |")
    out += ["", "## Mean scores by group", "",
            "Column key: " + ", ".join(f"{SHORT[d.id]} = {d.label}" for d in DIMENSIONS) + ".", ""]
    for attribute in ("category", "language", "model"):
        out += _group_table(complete, key, attribute)
    agreement, disagreements, kappas = _agreement(complete)
    out += ["## Inter-reviewer agreement", ""] + agreement
    low = [d.label for d in DIMENSIONS if kappas[d.id] is not None and kappas[d.id] < LOW_AGREEMENT]
    if low:
        out += ["", f"Warning: agreement below {LOW_AGREEMENT} on {', '.join(low)}; calibrate reviewers before trusting these means."]
    out += ["", "## Items to adjudicate (difference ≥ 2, or N/A vs a score)", ""]
    out += (["| Item | Dimension | Scores |", "|---|---|---|"] + disagreements) if disagreements else ["None."]
    out += ["", "## Critical errors (every item flagged by any reviewer)", ""] + critical
    checks = [
        (f"≥ {GATE['min_complete_reviewers']} reviewers completed all items", f"{len(complete)}", True),
        (f"Mean overall usefulness ≥ {GATE['overall_usefulness']}", _fmt(means["overall_usefulness"]),
         (means["overall_usefulness"] or 0) >= GATE["overall_usefulness"]),
        (f"Mean factual correctness ≥ {GATE['factual_correctness']}", _fmt(means["factual_correctness"]),
         (means["factual_correctness"] or 0) >= GATE["factual_correctness"]),
        ("Unresolved critical errors = 0", str(unresolved), unresolved <= GATE["max_unresolved_critical"]),
        ("Agreement reported", "see table above", True),
    ]
    passed = all(ok for *_, ok in checks)
    out += ["", f"## Beta gate (proposal): **{'PASSED' if passed else 'NOT PASSED'}**", "",
            "| Criterion | Value | Result |", "|---|---|---|"]
    out += [f"| {name} | {value} | {'pass' if ok else '**FAIL**'} |" for name, value, ok in checks]
    return "\n".join(out + tail) + "\n"


def score(batch: str, sheets: list[Path], key_file: Path, adjudication: Path | None = None) -> str:
    key = load_key(key_file, batch)
    reviews = [load_review(p, key) for p in sorted(sheets)]
    rel = []
    for p in sorted(sheets):
        try:
            rel.append(p.resolve().relative_to(ROOT).as_posix())
        except ValueError:
            rel.append(p.name)
    return build_report(batch, reviews, key, load_adjudication(adjudication), rel)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score blinded human review sheets.")
    parser.add_argument("--batch", required=True, help="batch id, e.g. 20261113")
    parser.add_argument("--sheets", nargs="*", help="default: docs/human-eval/sheets/<batch>-*.csv")
    parser.add_argument("--key", help="default: docs/human-eval/keys/<batch>-key.json")
    parser.add_argument("--adjudication", help="default: docs/human-eval/adjudication-<batch>.csv if present")
    parser.add_argument("--out", help="default: docs/human-eval/report-<batch>.md; '-' for stdout")
    args = parser.parse_args(argv)
    sheets = [Path(p) for p in args.sheets] if args.sheets else sorted((EVAL_DIR / "sheets").glob(f"{args.batch}-*.csv"))
    key_file = Path(args.key) if args.key else key_path(args.batch)
    if not key_file.exists():
        print(f"error: key not found: {key_file} (run scripts.blind_human_eval first)", file=sys.stderr)
        return 1
    adjudication = Path(args.adjudication) if args.adjudication else EVAL_DIR / f"adjudication-{args.batch}.csv"
    try:
        text = score(args.batch, sheets, key_file, adjudication)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.out == "-":
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
        return 0
    out = Path(args.out) if args.out else EVAL_DIR / f"report-{args.batch}.md"
    out.write_text(text, encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
