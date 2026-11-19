"""Did a rewrite keep the meaning of the answer it rewrote?

Any model pass that only reworks wording (a language rewrite, a grammar
correction) is checked against the already-validated answer before its output
is used. The check is deterministic and compares *claims*, not wording: both
texts go through the same extractors, so it is language independent.

A rewrite is rejected when it:

* contradicts the chart, or states a placement the chart cannot confirm;
* drops a verified chart fact (placement, house sign, lordship, Dasha, ...);
* introduces a claim the original did not make (Sade Sati, a dosha verdict, a
  transit, an aspect, a yoga, a Dasha statement, a sign lord);
* flips a Dasha's temporal status (past <-> current <-> future);
* adds, drops or changes a year or a month;
* raises the certainty of a prediction, or removes its hedging.

The caller then keeps the original answer. There is no second attempt: a
rewrite that fails is discarded, never retried until it passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.evaluation.chart_claims import _YEAR, Claim, _mentions, split_sentences, validate_reply
from app.evaluation.fact_claims import _SADE, HARD_KINDS, claim_signature
from app.evaluation.text_normalize import normalize_numerals
from app.kundli.chart_facts import ChartFacts

# Verified facts a rewrite must keep. "Keep" means the same entities are still
# mentioned together in one sentence (see `_lost`): comparing extracted claims
# across two phrasings or languages is brittle, because an extractor that fires
# on one wording can miss the other ("its lord Jupiter" vs "iska lord Jupiter").
PRESERVED = frozenset({"placement_house", "placement_sign", "house_sign", "lordship", "lagna", "moon_sign",
                       "mahadasha", "antardasha", "sign_lord", "sign_lord_not", "dasha_status", "sade_sati"})
# Claim kinds a rewrite may never introduce. A VERIFIED Dasha status or sign
# lord a rewrite happens to make explicit is harmless, and a wrong one is a
# contradiction, which is checked separately.
NEVER_ADDED = frozenset({"sade_sati", "sade_sati_phase", "dosha_verdict", "dosha", "transit", "aspect",
                         "influence", "yoga", "conjunction"})
# What a *repair* (which is allowed to change a wrong claim) may never introduce.
NEVER_ADDED_BY_REPAIR = frozenset({"sade_sati", "sade_sati_phase", "dosha_verdict", "dosha", "transit"})

_MONTHS = {
    1: ("january", "jan", "जनवरी"), 2: ("february", "feb", "फरवरी", "फ़रवरी"), 3: ("march", "mar", "मार्च"),
    4: ("april", "apr", "अप्रैल"), 5: ("may", "मई"), 6: ("june", "jun", "जून"), 7: ("july", "jul", "जुलाई"),
    8: ("august", "aug", "अगस्त"), 9: ("september", "sep", "sept", "सितंबर", "सितम्बर"),
    10: ("october", "oct", "अक्टूबर"), 11: ("november", "nov", "नवंबर", "नवम्बर"), 12: ("december", "dec", "दिसंबर", "दिसम्बर"),
}
_MONTH_RE = [(n, re.compile(rf"(?<![\wऀ-ॿ])(?:{'|'.join(map(re.escape, ws))})(?![\wऀ-ॿ])", re.I))
             for n, ws in _MONTHS.items() if n != 5]
# English "may" is mostly the modal verb ("you may enjoy…"): it is the month only
# next to a number ("May 2026", "20 May"). Counting the modal as a month rejected
# rewrites that merely dropped a "may" (concise-rewrite tests, 2026-11-19).
_MONTH_RE.insert(4, (5, re.compile(r"(?<![\wऀ-ॿ])मई(?![\wऀ-ॿ])|\bmay\s+\d|\b\d{1,2}(?:st|nd|rd|th)?\s+may\b",
                                   re.I)))

_ABSOLUTE = re.compile(
    r"\b(?:will|shall)\s+(?:definitely|certainly|surely|absolutely)\b|\b100\s?%|\bguarantee\w*|\bfor\s+sure\b|"
    r"\bpakk[aie]\b|\bzaroor\b|\bzarur\b|\bnishchit\b|\bdefinitely\b|\bcertainly\b|"
    r"निश्चित(?:\s+रूप\s+से)?|पक्का|पक्की|ज़रूर|जरूर|अवश्य", re.I)
_NEGATION = re.compile(r"\b(?:not|no|never|cannot|can't|nahi|nahin|mat)\b|नहीं|मत\b", re.I)
_HEDGE = re.compile(
    r"\b(?:may|might|could|can|possible|possibly|likely|tends?|suggests?|indicates?|chances?|possibility|"
    r"probably|perhaps|shayad|sakta|sakti|sakte|sambhav|sambhavna|lagta|lagti|dikhta|dikhti)\b|"
    r"सकता|सकती|सकते|संभव|संभावना|शायद|लगता|लगती|दिखता|दिखती", re.I)


@dataclass
class RewriteVerdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def _sigs(claims: list[Claim], kinds, status: str | None = None) -> set[tuple]:
    return {claim_signature(c) for c in claims if c.kind in kinds and (status is None or c.status == status)}


def _needs(claim: Claim) -> set[tuple[str, object]] | None:
    """The entities that must still appear together for `claim` to be kept."""
    k = claim.kind
    if k == "placement_house" or k == "lordship":
        return {("graha", claim.subject), ("house", claim.value)}
    if k == "placement_sign":
        return {("graha", claim.subject), ("sign", claim.value)}
    if k == "house_sign":
        return {("house", claim.value[0]), ("sign", claim.value[1])}
    if k in ("lagna", "moon_sign"):
        return {("sign", claim.value)}
    if k in ("mahadasha", "antardasha"):
        return {("graha", claim.value)}
    if k in ("sign_lord", "sign_lord_not"):
        return {("sign", claim.subject), ("graha", claim.value)}
    if k == "dasha_status":
        return {("graha", str(claim.subject).split()[0])}
    return None


def _entities(text: str) -> list[set[tuple[str, object]]]:
    return [{(m.kind, m.value) for m in _mentions(s)} for s in split_sentences(text)]


def _lost(old: list[Claim], old_text: str, new_text: str) -> list[str]:
    """Verified facts of the original that the rewrite no longer states. A fact
    whose entities the scanner cannot see together even in the ORIGINAL ("7th
    lord Jupiter" names the 7th without a house noun) is not measurable this
    way and is left to the contradiction check, never reported as lost."""
    before, after = _entities(old_text), _entities(new_text)
    lost = []
    for c in old:
        if c.status != "verified" or c.kind not in PRESERVED:
            continue
        if c.kind == "sade_sati":
            if c.value and not _SADE.search(new_text):
                lost.append("sade_sati")
            continue
        need = _needs(c)
        if need and any(need <= e for e in before) and not any(need <= e for e in after):
            lost.append(f"{c.kind}: {c.subject} = {c.value}")
    return sorted(set(lost))


def _tenses(claims: list[Claim]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for c in claims:
        if c.kind == "dasha_status":
            out.setdefault(str(c.subject), set()).add(str(c.value))
    return out


def _months(text: str) -> set[int]:
    return {n for n, rx in _MONTH_RE if rx.search(text)}


def _years(text: str) -> set[int]:
    return {int(y) for y in _YEAR.findall(text)}


def _absolute(text: str) -> bool:
    return any(_ABSOLUTE.search(s) and not _NEGATION.search(s)
               for s in re.split(r"(?<=[.!?।])\s+|\n+", text))


def _factual_failures(claims: list[Claim]) -> list[str]:
    bad = [c for c in claims if c.status == "contradicted" or (c.status == "unsupported" and c.kind in HARD_KINDS)]
    return [c.describe() for c in bad]


def compare_rewrite(original: str, rewritten: str, facts: ChartFacts | None) -> RewriteVerdict:
    """Strict comparison for a rewrite that may only change wording."""
    if not rewritten or not rewritten.strip():
        return RewriteVerdict(False, ["empty rewrite"])
    old_text, new_text = normalize_numerals(original), normalize_numerals(rewritten)
    old, new = validate_reply(old_text, facts).claims, validate_reply(new_text, facts).claims
    reasons: list[str] = []

    old_failures = set(_factual_failures(old))
    reasons += [f"rewrite {f}" for f in _factual_failures(new) if f not in old_failures]

    reasons += [f"dropped verified fact {fact}" for fact in _lost(old, old_text, new_text)]
    old_tense, new_tense = _tenses(old), _tenses(new)
    reasons += [f"Dasha status changed for {key}: {sorted(old_tense[key])} -> {sorted(new_tense[key])}"
                for key in sorted(old_tense.keys() & new_tense.keys()) if old_tense[key] != new_tense[key]]

    added = _sigs(new, NEVER_ADDED) - _sigs(old, NEVER_ADDED)
    reasons += [f"introduced claim {sig[0]}: {sig[1]} = {sig[2]}" for sig in sorted(added)]

    old_years, new_years = _years(old_text), _years(new_text)
    if old_years != new_years:
        reasons.append(f"years changed: {sorted(old_years)} -> {sorted(new_years)}")
    old_months, new_months = _months(old_text), _months(new_text)
    if old_months != new_months:
        reasons.append(f"months changed: {sorted(old_months)} -> {sorted(new_months)}")

    if _absolute(new_text) and not _absolute(old_text):
        reasons.append("added certainty")
    if _HEDGE.search(old_text) and not _HEDGE.search(new_text):
        reasons.append("removed hedging")
    return RewriteVerdict(not reasons, reasons)


def check_repair(draft: str, repaired: str, facts: ChartFacts | None) -> RewriteVerdict:
    """A repair may change a wrong claim, but not bring in a new risky one or
    make a prediction more certain than the draft."""
    if not repaired or not repaired.strip():
        return RewriteVerdict(False, ["empty repair"])
    old_text, new_text = normalize_numerals(draft), normalize_numerals(repaired)
    old, new = validate_reply(old_text, facts).claims, validate_reply(new_text, facts).claims
    added = _sigs(new, NEVER_ADDED_BY_REPAIR) - _sigs(old, NEVER_ADDED_BY_REPAIR)
    reasons = [f"introduced claim {sig[0]}: {sig[1]} = {sig[2]}" for sig in sorted(added)]
    if _absolute(new_text) and not _absolute(old_text):
        reasons.append("added certainty")
    return RewriteVerdict(not reasons, reasons)


__all__ = ["RewriteVerdict", "compare_rewrite", "check_repair", "PRESERVED", "NEVER_ADDED"]
