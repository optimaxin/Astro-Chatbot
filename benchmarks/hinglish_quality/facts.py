"""Meaning preservation: Kundli facts and dates.

Two independent checks, neither of which is an LLM:

1. chart contradictions — app.evaluation.chart_claims against the benchmark
   chart (a planet put in the wrong house or sign, a wrong lord, wrong Dasha
   dates). This is the same fact checker production uses; here it is used as
   a measuring instrument only.
2. reference comparison — chart tokens (planets, houses, signs, years) of the
   gold reference that the candidate drops or replaces, and years the
   candidate invents that neither the chart's Dasha table nor the reference
   contains.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field

_PLANET_ALIASES = {
    "sun": ("sun", "surya", "सूर्य"), "moon": ("moon", "chandra", "चंद्रमा", "चंद्र", "चन्द्र"),
    "mars": ("mars", "mangal", "मंगल"), "mercury": ("mercury", "budh", "बुध"),
    "jupiter": ("jupiter", "guru", "brihaspati", "बृहस्पति", "गुरु"), "venus": ("venus", "shukra", "शुक्र"),
    "saturn": ("saturn", "shani", "शनि"), "rahu": ("rahu", "राहु"), "ketu": ("ketu", "केतु"),
}
_SIGN_ALIASES = {
    "aries": ("aries", "mesh", "मेष"), "taurus": ("taurus", "vrishabh", "वृषभ"), "gemini": ("gemini", "mithun", "मिथुन"),
    "cancer": ("cancer", "kark", "कर्क"), "leo": ("leo", "simha", "सिंह"), "virgo": ("virgo", "kanya", "कन्या"),
    "libra": ("libra", "tula", "तुला"), "scorpio": ("scorpio", "vrishchik", "वृश्चिक"),
    "sagittarius": ("sagittarius", "dhanu", "धनु"), "capricorn": ("capricorn", "makar", "मकर"),
    "aquarius": ("aquarius", "kumbh", "कुंभ"), "pisces": ("pisces", "meen", "मीन"),
}
_HINDI_ORDINALS = {"पहले": 1, "पहला": 1, "दूसरे": 2, "दूसरा": 2, "तीसरे": 3, "तीसरा": 3, "चौथे": 4, "चौथा": 4,
                   "पंचम": 5, "पाँचवें": 5, "छठे": 6, "छठा": 6, "सप्तम": 7, "सातवें": 7, "आठवें": 8, "नवम": 9,
                   "नौवें": 9, "दशम": 10, "दसवें": 10, "ग्यारहवें": 11, "ग्यारहवाँ": 11, "बारहवें": 12, "बारहवाँ": 12}


def _token_patterns() -> list[tuple[str, re.Pattern[str]]]:
    out = []
    for key, aliases in {**_PLANET_ALIASES, **_SIGN_ALIASES}.items():
        latin = [a for a in aliases if a.isascii()]
        deva = [a for a in aliases if not a.isascii()]
        parts = []
        if latin:
            parts.append(r"\b(?:" + "|".join(latin) + r")\b")
        if deva:
            parts.append("(?:" + "|".join(deva) + ")")
        out.append((key, re.compile("|".join(parts), re.I)))
    return out


_TOKENS = _token_patterns()
_HOUSE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b|(\d{1,2})(?:वें|वाँ|ले)", re.I)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d\d)(?!\d)")


def chart_tokens(text: str) -> set[str]:
    text = text or ""
    found = {key for key, pattern in _TOKENS if pattern.search(text)}
    found |= {f"h{m.group(1) or m.group(2)}" for m in _HOUSE.finditer(text)}
    found |= {f"h{n}" for word, n in _HINDI_ORDINALS.items() if word in text}
    found |= {f"y{y}" for y in _YEAR.findall(text)}
    return found


@functools.lru_cache(maxsize=1)
def _bench_chart():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("_kq_bench", root / "scripts" / "kundli_quality_bench.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root / "scripts"))
    spec.loader.exec_module(module)
    from app.kundli.chart_facts import build_chart_facts
    from app.kundli.kundli_service import KundliContext

    chart = module.screenshot_chart(KundliContext)
    return build_chart_facts(chart), chart


@functools.lru_cache(maxsize=1)
def chart_years() -> frozenset[str]:
    _, chart = _bench_chart()
    return frozenset(_YEAR.findall(chart.to_prompt_block("general_astrology")))


_PLANET_WORD = "|".join(a for aliases in _PLANET_ALIASES.values() for a in aliases)
_ORDINAL_WORD = "|".join(_HINDI_ORDINALS)
# "10th lord Mercury", "lord of the 10th is Mercury", "10th house ka lord Jupiter", "दशम भाव का स्वामी बृहस्पति"
_LORD_CLAIMS = [
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)\s+(?:house\s+)?lord\s+(?:is\s+)?(?:the\s+)?({_PLANET_WORD})\b", re.I),
    re.compile(rf"\blord\s+of\s+(?:your\s+|the\s+)?(\d{{1,2}})(?:st|nd|rd|th)(?:\s+house)?\s+is\s+({_PLANET_WORD})\b", re.I),
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)\s+house\s+ka\s+(?:lord|swami)\s+({_PLANET_WORD})\b", re.I),
    re.compile(rf"({_ORDINAL_WORD})\s+भाव\s+का\s+स्वामी\s+({_PLANET_WORD})"),
]


def _planet_key(word: str) -> str | None:
    word = word.lower()
    return next((k for k, aliases in _PLANET_ALIASES.items() if word in (a.lower() for a in aliases)), None)


def lord_errors(text: str, facts) -> list[str]:
    """House-lord statements that contradict the chart (the production
    validator does not check "Nth lord X"; found by this benchmark)."""
    out = []
    for pattern in _LORD_CLAIMS:
        for match in pattern.finditer(text or ""):
            raw_house, planet = match.group(1), _planet_key(match.group(2))
            house = int(raw_house) if raw_house.isdigit() else _HINDI_ORDINALS.get(raw_house)
            fact = facts.houses.get(house) if house else None
            if fact and planet and fact.lord.lower() != planet:
                out.append(f"{match.group(0)} (chart: {fact.lord})")
    return out


@dataclass
class FactReport:
    contradictions: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)       # in the reference, missing from the candidate
    invented_years: list[str] = field(default_factory=list)

    @property
    def fact_change(self) -> bool:
        return bool(self.contradictions or self.invented_years)

    def to_dict(self) -> dict:
        return {"contradictions": self.contradictions, "dropped": self.dropped, "invented_years": self.invented_years}


def check_facts(candidate: str, reference: str | None = None, source: str | None = None) -> FactReport:
    """`source` is the text the candidate was produced FROM (correction mode):
    only facts present in both the source and the reference can be expected
    to survive — a corrector cannot add what the original never said."""
    from app.evaluation.chart_claims import validate_reply

    facts, _ = _bench_chart()
    report = FactReport()
    result = validate_reply(candidate or "", facts)
    report.contradictions = [getattr(c, "text", str(c))[:80] for c in result.contradictions]
    report.contradictions += lord_errors(candidate, facts)
    allowed_years = set(chart_years()) | set(_YEAR.findall(reference or ""))
    report.invented_years = sorted({y for y in _YEAR.findall(candidate or "") if y not in allowed_years})
    if reference:
        expected = chart_tokens(reference)
        if source is not None:
            expected &= chart_tokens(source)
        report.dropped = sorted(expected - chart_tokens(candidate))
    return report


__all__ = ["FactReport", "check_facts", "chart_tokens", "chart_years"]
