"""Deterministic validation of chart claims in a generated reply.

Extracts what a reply asserts about THIS user's chart — which house or sign a
graha occupies, which graha rules a house, the Lagna, Moon sign, nakshatra,
running dasha, retrograde status, yogas and doshas — and checks each claim
against `ChartFacts` (the Kundli engine's output). English, Roman-script
Hinglish and Devanagari Hindi phrasings are covered, including indirect forms:

    "Your career house is strong. Jupiter sits there."      (coreference)
    "Jupiter is positioned in your professional house."     (house by theme)
    "Your tenth house contains Jupiter."                    (reversed)
    "Guru aapke dasve ghar mein baithe hain."               (Hinglish)

Every extracted claim gets one status:

* `verified`      — matches the chart
* `contradicted`  — the chart says otherwise. Must not reach the user.
* `unsupported`   — chart-specific, but the engine does not compute it
                    (aspects/"influence", Manglik, a yoga it never evaluated).
                    Rewritten in repair mode; never called false.
* `generic`       — phrased as general astrology ("Jupiter in the 10th
                    generally gives…"), not about this chart. Recorded only.

Negated statements ("Jupiter is not in your 10th house") are not claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.kundli.chart_facts import ORD, SIGNS, ChartFacts

# ── vocabulary ────────────────────────────────────────────────────────
GRAHA_ALIASES: dict[str, tuple[str, ...]] = {
    "sun": ("sun", "surya", "suraj", "ravi", "सूर्य", "रवि"),
    "moon": ("moon", "chandra", "chandrama", "चंद्र", "चन्द्र", "चंद्रमा", "चन्द्रमा"),
    "mars": ("mars", "mangal", "kuja", "मंगल"),
    "mercury": ("mercury", "budh", "budha", "बुध"),
    "jupiter": ("jupiter", "guru", "brihaspati", "brahaspati", "गुरु", "गुरू", "बृहस्पति"),
    "venus": ("venus", "shukra", "sukra", "शुक्र"),
    "saturn": ("saturn", "shani", "shanidev", "शनि"),
    "rahu": ("rahu", "राहु"),
    "ketu": ("ketu", "केतु"),
}
SIGN_ALIASES: dict[str, tuple[str, ...]] = {
    "Aries": ("aries", "mesh", "mesha", "मेष"), "Taurus": ("taurus", "vrishabh", "vrishabha", "वृषभ"),
    "Gemini": ("gemini", "mithun", "mithuna", "मिथुन"), "Cancer": ("cancer", "kark", "karka", "कर्क"),
    "Leo": ("leo", "simha", "सिंह"), "Virgo": ("virgo", "kanya", "कन्या"),
    "Libra": ("libra", "tula", "तुला"), "Scorpio": ("scorpio", "vrishchik", "vrischika", "वृश्चिक"),
    "Sagittarius": ("sagittarius", "dhanu", "dhanus", "धनु"),
    "Capricorn": ("capricorn", "makar", "makara", "मकर"),
    "Aquarius": ("aquarius", "kumbh", "kumbha", "कुंभ", "कुम्भ"), "Pisces": ("pisces", "meen", "meena", "मीन"),
}
ORDINAL_WORDS = {
    1: ("first", "pehle", "pehla", "pahle", "पहले"), 2: ("second", "doosre", "dusre", "doosra", "दूसरे"),
    3: ("third", "teesre", "tisre", "teesra", "तीसरे", "तीसरा"), 4: ("fourth", "chauthe", "chautha", "चौथे", "चौथा"),
    5: ("fifth", "paanchve", "panchve", "paanchva", "पांचवें", "पाँचवें", "पांचवां", "पाँचवाँ"),
    6: ("sixth", "chhathe", "chhate", "छठे", "छठा"),
    7: ("seventh", "saatve", "satve", "saatva", "सातवें", "सातवां", "सातवाँ", "सप्तम"),
    8: ("eighth", "aathve", "athve", "आठवें", "आठवां"),
    9: ("ninth", "nauve", "nauva", "नौवें", "नौवां"), 10: ("tenth", "dasve", "dasven", "daswe", "dasva", "दसवें", "दसवां", "दशम"),
    11: ("eleventh", "gyarahve", "gyarve", "ग्यारहवें", "ग्यारहवां", "एकादश"),
    12: ("twelfth", "barahve", "barhve", "बारहवें", "बारहवां", "द्वादश"),
}
# Only unambiguous house themes ("money" could be 2nd or 11th, so it is left out).
THEME_HOUSES = {
    10: ("career", "profession", "professional", "karma", "status", "work"), 7: ("marriage", "spouse", "partnership", "kalatra"),
    3: ("siblings", "courage", "communication"), 4: ("home", "mother", "property"),
    5: ("children", "progeny", "education", "romance"), 6: ("enemies", "debts", "disease"),
    8: ("longevity", "transformation"), 9: ("luck", "fortune", "dharma", "father", "bhagya"),
    11: ("gains", "income"), 12: ("losses", "expenses", "foreign"),
}
HOUSE_NOUN = r"(?:house|houses|bhava|bhav|bhaav|ghar|khana|sthan|sthana|भाव|घर)"
# End of a word in Latin OR Devanagari. Python's `\b` does not fire after a
# Devanagari vowel sign (U+0947 is not `\w`), so "शुक्र 7वें घर में है" used to
# produce no claim at all.
_WORD_END = r"(?![\wऀ-ॿ])"
PLACE_VERBS = (r"\b(?:is|are|was|sits?|sitting|sat|placed|positioned|posited|located|occupies|occupying|"
               r"resides|residing|stationed|lies|found|in|baitha|baithe|baithi|sthit|hai|hain)\b|में|बैठ|स्थित")
CONTAIN_VERBS = r"\b(?:contains?|has|have|hosts?|holds?|houses|is occupied by|occupied by|mein\s+hai)\b"
INFLUENCE_VERBS = (r"\b(?:influences?|influencing|aspects?|aspecting|casts?|looks? at|affects?|impacts?|"
                   r"blesses|strengthens|activates|energi[sz]es|drishti|nazar)\b")
LORD_FWD = r"\b(?:rules?|ruling|owns?|lords? over|is the lord of|is lord of|is the ruler of|swami)\b"
LORD_BACK = (r"\b(?:ruled by|owned by|lord is|ruler is|lord,|ka swami|ke swami|ki swami)\b"
             r"|(?:का|के|की)\s+(?:स्वामी|भावेश)")
# The Hindi "न" counts only as a standalone word: as a suffix it ended
# "विराजमान", "स्थान", "जीवन", and every such sentence was skipped as negated.
NEGATION = re.compile(r"\b(?:not|isn't|aren't|never|no|nahi|nahin|na)\b|n't\b|नहीं|(?<![ऀ-ॿ])न\s", re.I)
GENERIC = re.compile(r"\b(?:if|when|whenever|generally|typically|traditionally|usually|often|in general|"
                     r"a person|someone|people|those|anyone|agar|jab|yadi|aam taur)\b|अगर|जब|यदि", re.I)
SECOND_PERSON = re.compile(r"\b(?:your|you|you're|aap|aapke|aapki|aapka|apke|apki|tumhare|tumhari|tumhara|"
                           r"tera|tere|this chart|the chart)\b|आपके|आपकी|आपका|तुम्हारे|तुम्हारी", re.I)
COREF = re.compile(r"\b(?:there|in it|that house|wahan|wahin|usme|usmein)\b|वहां|वहाँ|उसमें", re.I)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")


NAKSHATRAS = ("Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu", "Pushya",
              "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati",
              "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana",
              "Dhanishta", "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati")
_NAK_KEY = {re.sub(r"[^a-z]", "", n.lower()): n for n in NAKSHATRAS}
# Longest first so "Uttara Bhadrapada" wins over a bare "Bhadrapada".
_NAKSHATRA_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(n).replace(r"\ ", r"[\s-]?") for n in sorted(NAKSHATRAS, key=len, reverse=True))
    + r")\b", re.I)


def _canonical_nakshatra(text: str) -> str:
    return _NAK_KEY.get(re.sub(r"[^a-z]", "", text.lower()), text)


def _alias_pattern(aliases: dict[Any, tuple[str, ...]]) -> list[tuple[Any, re.Pattern]]:
    out = []
    for key, words in aliases.items():
        alts = "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))
        out.append((key, re.compile(rf"(?<![\wऀ-ॿ])(?:{alts})(?![\wऀ-ॿ])", re.I)))
    return out


_GRAHA_RE = _alias_pattern(GRAHA_ALIASES)
_SIGN_RE = _alias_pattern(SIGN_ALIASES)
_HOUSE_RES: list[tuple[int, re.Pattern]] = []
for _n in range(1, 13):
    words = "|".join(ORDINAL_WORDS[_n])
    _HOUSE_RES.append((_n, re.compile(
        # A bare ordinal after "in"/"mein"-style context ("Mercury in 11th",
        # "11th mein") — missed before 2026-11-13 and found in review (H001).
        rf"(?:(?<=\bin\s)(?:the\s)?{_n}(?:st|nd|rd|th)\b(?!\s*(?:lord|ka|ke|ki)\b)|"
        rf"\b{_n}(?:st|nd|rd|th)(?=\s+(?:mein|me|में){_WORD_END})|"
        rf"\b{_n}(?:st|nd|rd|th|वें|वे|वां|वा|वीं|वी)?\s*{HOUSE_NOUN}|\bhouse\s+{_n}\b|(?:{words})\s*{HOUSE_NOUN}|"
        rf"(?:{ '|'.join(THEME_HOUSES.get(_n, ())) or 'x^'})\s+{HOUSE_NOUN}|"
        rf"{HOUSE_NOUN}\s+of\s+(?:{ '|'.join(THEME_HOUSES.get(_n, ())) or 'x^'}))", re.I)))


@dataclass
class Mention:
    kind: str           # graha | house | sign
    value: Any
    start: int
    end: int


@dataclass
class Claim:
    kind: str           # placement_house | placement_sign | lordship | influence | lagna | moon_sign |
                        # nakshatra | mahadasha | antardasha | retrograde | yoga | dosha
    subject: str
    value: Any
    status: str = "unknown"
    truth: Any = None
    sentence: str = ""
    sentence_index: int = -1

    def describe(self) -> str:
        if self.kind == "placement_house":
            return f"says {self.subject} is in the {ORD.get(self.value, self.value)} house; chart: {ORD.get(self.truth, self.truth)} house"
        if self.kind == "placement_sign":
            return f"says {self.subject} is in {self.value}; chart: {self.truth}"
        if self.kind == "lordship":
            return f"says {self.subject} rules the {ORD.get(self.value, self.value)} house; chart: its lord is {self.truth}"
        if self.kind == "house_sign":
            return f"says the {ORD.get(self.value[0], self.value[0])} house is {self.value[1]}; chart: {self.truth}"
        if self.kind == "conjunction":
            return f"says {self.subject} are together (conjunct); chart: {self.truth}"
        if self.kind == "dasha_dates":
            return f"gives the years {self.value} for the {self.subject}; chart dates: {self.truth}"
        return f"says {self.subject} {self.kind} = {self.value}; chart: {self.truth}"


@dataclass
class ValidationResult:
    claims: list[Claim] = field(default_factory=list)

    @property
    def contradictions(self) -> list[Claim]:
        return [c for c in self.claims if c.status == "contradicted"]

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if c.status == "unsupported"]

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for c in self.claims:
            counts[c.status] = counts.get(c.status, 0) + 1
        return {"claims": len(self.claims), "by_status": counts,
                "contradictions": [c.describe() for c in self.contradictions],
                # The model's own sentence, kept on the turn state for review;
                # never written to logs.
                "contradiction_sentences": [c.sentence[:200] for c in self.contradictions],
                "unsupported": [f"{c.kind}: {c.subject} -> {c.value}" for c in self.unsupported]}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text or "") if s.strip()]


def _mentions(sentence: str) -> list[Mention]:
    found: list[Mention] = []
    for graha, rx in _GRAHA_RE:
        for m in rx.finditer(sentence):
            # "moon sign" / "chandra rashi" talk about the Rashi, not the Moon's house.
            if graha == "moon" and re.match(r"\s*(?:sign|rashi)", sentence[m.end():], re.I):
                continue
            found.append(Mention("graha", graha, m.start(), m.end()))
    for n, rx in _HOUSE_RES:
        for m in rx.finditer(sentence):
            found.append(Mention("house", n, m.start(), m.end()))
    for sign, rx in _SIGN_RE:
        for m in rx.finditer(sentence):
            found.append(Mention("sign", sign, m.start(), m.end()))
    found.sort(key=lambda x: x.start)
    # A house phrase can match two patterns; keep the first per span.
    deduped: list[Mention] = []
    for m in found:
        if deduped and m.kind == deduped[-1].kind and m.start < deduped[-1].end:
            continue
        deduped.append(m)
    return deduped


def _in_enumeration(sentence: str, house: Mention) -> bool:
    """True when this house mention ends a list like "1st, 4th, 7th or 12th house"."""
    window = sentence[max(0, house.start - 40):house.start]
    return bool(re.search(r"\d{1,2}(?:st|nd|rd|th)?(?:\s*(?:,|/|\bor\b|\bya\b|\band\b|\baur\b))+\s*$",
                          window, re.I))


def _between(sentence: str, a: Mention, b: Mention) -> str:
    return sentence[min(a.end, b.end):max(a.start, b.start)]


def _no_graha_between(mentions: list[Mention], a: Mention, b: Mention) -> bool:
    lo, hi = min(a.start, b.start), max(a.start, b.start)
    return not any(m.kind == "graha" and lo < m.start < hi for m in mentions)


def _no_house_between(mentions: list[Mention], a: Mention, b: Mention) -> bool:
    """A placement pairs a planet with its NEAREST house mention: in "Jupiter
    3rd mein hai, aur 7th mein Moon hai" the 7th belongs to the Moon, not to
    Jupiter (a false positive found in the final live evaluation)."""
    lo, hi = min(a.start, b.start), max(a.start, b.start)
    return not any(m.kind == "house" and lo < m.start < hi for m in mentions)


_GRAHA_WORDS = "|".join(sorted({re.escape(w) for ws in GRAHA_ALIASES.values() for w in ws if w.isascii()},
                               key=len, reverse=True))
_LORD_OF_N = re.compile(rf"\b({_GRAHA_WORDS})\b,?\s+(?:is|being|as)\s+(?:the\s+|your\s+)?(\d{{1,2}})(?:st|nd|rd|th)"
                        rf"(?:\s*(?:and|&|,)\s*(\d{{1,2}})(?:st|nd|rd|th))?\s+(?:house\s+)?lord\b", re.I)
_AS_ITS_LORD = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)\s+(?:house|bhav\w*)\b[^.;]{{0,40}}?"
                          rf"(?:\bwith\s+({_GRAHA_WORDS})\s+as\s+its\s+lord\b|\b(?:its|whose)\s+lord\s+(?:is\s+)?"
                          rf"({_GRAHA_WORDS})\b)", re.I)
# "6th lord Mercury", "4th house lord Saturn", "10th house ka lord Jupiter" —
# the Hinglish benchmark found wrong lords in this form passing unchecked.
# Devanagari planet names too: a Hindi reply's "4th house lord शनि 4th house में है"
# (final quality gate; the 4th lord is Mercury) passed unchecked.
_DEVA_GRAHA = "|".join(sorted({w for ws in GRAHA_ALIASES.values() for w in ws if not w.isascii()}, key=len,
                              reverse=True))
_NTH_LORD_GRAHA = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th|वें|वाँ|वां)?\s+(?:house\s+|bhav\s+|भाव\s+)?"
                             rf"(?:ka\s+|ke\s+|का\s+|के\s+)?(?:lord|swami|स्वामी)\s+(?:is\s+)?"
                             rf"({_GRAHA_WORDS}|{_DEVA_GRAHA})(?![\wऀ-ॿ])", re.I)


def _graha_key(word: str) -> str | None:
    lowered = word.lower()
    return next((k for k, ws in GRAHA_ALIASES.items() if lowered in ws), None)


def _extract(sentence: str, index: int, last_house: int | None) -> tuple[list[Claim], int | None]:
    """Claims in one sentence, plus the last house it mentioned (for coreference)."""
    claims: list[Claim] = []
    if NEGATION.search(sentence):
        houses = [m.value for m in _mentions(sentence) if m.kind == "house"]
        return [], (houses[-1] if houses else last_house)
    mentions = _mentions(sentence)
    grahas = [m for m in mentions if m.kind == "graha"]
    houses = [m for m in mentions if m.kind == "house"]
    signs = [m for m in mentions if m.kind == "sign"]
    lowered = sentence.lower()
    used: set[tuple[int, int]] = set()
    # A planet described as transiting is not making a birth-chart claim in
    # that clause ("Saturn is transiting Pisces" ≠ natal Saturn in Pisces).
    transiting = [g for g in grahas if re.match(r"[^.,;]{0,20}?\b(?:transit\w*|gochar)\b",
                                                sentence[g.end:], re.I)]
    grahas = [g for g in grahas if g not in transiting]

    def add(kind: str, subject: str, value: Any) -> None:
        claims.append(Claim(kind, subject, value, sentence=sentence, sentence_index=index))

    # Lordship first: "Venus rules your 9th", "9th house is ruled by Venus",
    # "the lord of your 7th house, Jupiter, ..." — these are NOT placements.
    # "1st, 2nd, 4th, 7th, 8th ya 12th house" is an enumeration — a rule
    # being explained — not a placement in this chart.
    if re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\s*(?:,|/|\bor\b|\bya\b|\band\b|\baur\b)\s*\d{1,2}(?:st|nd|rd|th)?\b",
                 sentence, re.I):
        houses = [h for h in houses if not re.match(
            r"\d", sentence[h.start:h.end]) or not _in_enumeration(sentence, h)]

    for h in houses:
        for g in grahas:
            mid = _between(sentence, g, h)
            if len(mid) > 45 or not _no_graha_between(mentions, g, h):
                continue
            before_h = sentence[max(0, h.start - 25):h.start].lower()
            after_h_lord = re.match(r"\s*(?:'s\s+)?(?:lord|ruler|swami|ka swami|ke swami)\b",
                                    sentence[h.end:h.end + 14], re.I)
            if (g.start < h.start and after_h_lord) or \
               (g.start < h.start and re.search(LORD_FWD, mid, re.I)) or \
               (h.start < g.start and re.search(LORD_BACK, mid, re.I)) or \
               (g.start > h.start and re.search(r"\b(?:lord|ruler|swami)\b", before_h + mid, re.I)
                and re.search(r"\blord of\b|\bruler of\b", before_h, re.I)):
                add("lordship", g.value, h.value)
                used.add((g.start, h.start))

    for h in houses:
        for g in grahas:
            if (g.start, h.start) in used or not _no_graha_between(mentions, g, h) \
                    or not _no_house_between(mentions, g, h):
                continue
            mid = _between(sentence, g, h)
            if len(mid) > 60:
                continue
            after_h = sentence[h.end:h.end + 12]
            if g.start < h.start and (re.search(PLACE_VERBS, mid, re.I) or
                                      re.match(rf"\s*(?:mein|me|में){_WORD_END}", after_h, re.I)):
                add("placement_house", g.value, h.value)
            elif h.start < g.start and not re.search(r"[,;]|\b(?:aur|and|but|lekin|par)\b|और", mid, re.I) \
                    and (re.search(CONTAIN_VERBS, mid, re.I) or re.match(r"^\s*(?:mein|me|में)(?:\s|$)", mid, re.I)):
                # Reversed form ("10th mein Guru", "your 10th house contains
                # Jupiter") must not cross a clause boundary: in "…12th mein
                # hai, aur Venus-Sun 11th mein" the 12th is not Venus's.
                add("placement_house", g.value, h.value)
            elif re.search(INFLUENCE_VERBS, mid, re.I):
                add("influence", g.value, h.value)

    for s in signs:
        for g in grahas:
            if not _no_graha_between(mentions, g, s):
                continue
            mid = _between(sentence, g, s)
            # "The Moon is exalted in Taurus" / "Mars rules Aries" state a
            # general rule about the sign, not where it sits in this chart.
            if re.search(r"\b(?:exalt\w*|debilitat\w*|uchch\w*|neech\w*|own sign|rules?|ruled|ruler|"
                         r"lord|swami|friend\w*|enem\w*|moolatrikona)\b|उच्च|नीच", mid, re.I):
                continue
            after_s = sentence[s.end:s.end + 18]
            if g.start < s.start and len(mid) <= 40 and (
                    re.search(r"\b(?:in|mein|me|is|hai)\b|में|राशि", mid, re.I)
                    or re.match(rf"\s*(?:rashi|raashi|राशि)?\s*(?:mein|me|में){_WORD_END}", after_s, re.I)):
                add("placement_sign", g.value, s.value)
            elif s.start < g.start and not mid.strip():               # "Scorpio Jupiter"
                add("placement_sign", g.value, s.value)

    if not houses and len(grahas) == 1 and last_house and COREF.search(sentence) \
            and re.search(PLACE_VERBS, sentence, re.I):
        add("placement_house", grahas[0].value, last_house)          # "Jupiter sits there."

    def nearest_sign(keyword: str) -> str | None:
        best = None
        for k in re.finditer(keyword, sentence, re.I):
            ks, ke = k.start(), k.end()
            for s in signs:
                gap = s.start - ke if s.start >= ke else ks - s.end
                if 0 <= gap <= 25 and not any(g.start in range(min(ks, s.start), max(ke, s.end))
                                              for g in grahas):
                    if best is None or gap < best[0]:
                        best = (gap, s.value)
        return best[1] if best else None

    # "Your 7th house is Virgo" / "7वां भाव कुंभ राशि है": the sign ON a house.
    # Production replies stated the wrong one and nothing checked it.
    for h in houses:
        for s in signs:
            mid = sentence[h.end:s.start]
            if s.start >= h.end and len(mid) <= 14 and not any(h.end <= g.start < s.start for g in grahas) \
                    and not re.search(r"[,;]|\b(?:and|aur|lord|swami)\b|और|स्वामी", mid, re.I) \
                    and not (sentence[s.end:s.end + 12].strip().lower().startswith(("lord", "ka swami"))):
                add("house_sign", f"house {h.value}", (h.value, s.value))
                break

    lagna = nearest_sign(r"\b(?:ascendant|lagna|lagan|rising)\b|लग्न")
    if lagna:
        add("lagna", "lagna", lagna)
    moon_sign = nearest_sign(r"\b(?:moon sign|chandra rashi|rashi|raashi)\b|राशि")
    # "10th house (Sagittarius rashi)" names the sign ON a house, not the Moon
    # sign — a false contradiction that cost a repair (round-4 probe).
    house_signs = {c.value[1] for c in claims if c.kind == "house_sign"}
    explicit_moon = re.search(r"\b(?:moon sign|chandra rashi|janma rashi|moon)\b|चंद्र|जन्म राशि", sentence, re.I)
    if moon_sign and moon_sign in house_signs and not explicit_moon:
        moon_sign = None
    # "पंचम भाव की राशि कर्क है" is the 5th house's sign (Hinglish benchmark
    # found it read as a Moon-sign contradiction).
    if moon_sign and not explicit_moon and re.search(
            r"भाव\s+की\s+राशि|\bhouse\s+(?:ka|ki)\s+(?:sign|rashi)\b|\bhouse'?s\s+sign\b", sentence, re.I):
        moon_sign = None
    if moon_sign and not any(g.value != "moon" for g in grahas):
        add("moon_sign", "moon", moon_sign)

    md = re.search(r"\b(sun|moon|mars|mercury|jupiter|venus|saturn|rahu|ketu|shani|guru|shukra|budh|mangal)"
                   r"(?:'s)?\s+(maha\s*dasha|mahadasha|antar\s*dasha|antardasha|dasha|dasa|period)", lowered)
    if md and re.search(r"\b(?:running|current|currently|ongoing|are in|you're in|under|chal rah[ia])\b", lowered) \
            and not re.search(r"\b(?:will|next|upcoming|after|later|begins?|starts?)\b", lowered):
        graha = next(k for k, words in GRAHA_ALIASES.items() if md.group(1) in words)
        add("antardasha" if "antar" in md.group(2) else "mahadasha", "dasha", graha)

    for g in grahas:
        if re.search(rf"{re.escape(sentence[g.start:g.end].lower())}[^.]{{0,25}}\b(?:retrograde|vakri)\b|वक्री", lowered):
            add("retrograde", g.value, True)
    # Transits: only Saturn's current sign is computed (Sade Sati data) and is
    # checked; any other transit statement about this chart is unsupported.
    for g in transiting:
        add("transit", g.value, "transit")

    for nak in _NAKSHATRA_RE.finditer(sentence):
        name = _canonical_nakshatra(nak.group(0))
        before = sentence[max(0, nak.start() - 45):nak.start()]
        owner = [g for g in grahas if g.end <= nak.start() and nak.start() - g.end <= 30]
        if owner and re.search(r"\b(?:in|is in|mein|placed in)\b|में", sentence[owner[-1].end:nak.start()], re.I):
            add("planet_nakshatra", owner[-1].value, name)
        elif re.search(r"\b(?:your|aapka|aapki|birth|janma|janm)\b[^.]{0,20}\bnakshatra\b|\bborn (?:in|under)\b",
                       before + sentence[nak.end():nak.end() + 12], re.I):
            add("nakshatra", "moon", name)

    filler = {"a", "an", "the", "strong", "powerful", "good", "rare", "auspicious", "this", "your", "have", "has"}
    for yoga in re.findall(r"([a-z][a-z\-]+(?:\s[a-z][a-z\-]+)?)[\"'”’]?\s+yoga\b", lowered):
        name = " ".join(w for w in yoga.split() if w not in filler)
        if name and re.search(r"\b(?:you have|your chart has|forms?|formed|present|there is|banta|ban raha|"
                              r"banata|banati|bana raha|ban rahi)\b",
                              lowered):
            add("yoga", "yoga", name)
    if re.search(r"\bkaal\s*sarp", lowered) and re.search(r"\b(?:have|has|present|there is|hai)\b", lowered):
        add("dosha", "kaal sarp", True)
    if re.search(r"\b(?:manglik|mangal dosha|kuja dosha)\b", lowered) and \
            re.search(r"\b(?:you are|you're|you have|hai|ho)\b", lowered):
        add("dosha", "manglik", True)

    # Lordship phrasings the proximity rules above miss (benchmark review):
    # "Mercury is the 6th and 9th lord", "the 7th house … with Venus as its
    # lord", "your 7th house is Virgo, with its lord Mercury …".
    seen_lords = {(c.subject, c.value) for c in claims if c.kind == "lordship"}

    def lord_claim(graha_word: str, house: str) -> None:
        graha = _graha_key(graha_word)
        if graha and (graha, int(house)) not in seen_lords and 1 <= int(house) <= 12:
            seen_lords.add((graha, int(house)))
            add("lordship", graha, int(house))

    for m in _LORD_OF_N.finditer(sentence):
        for house in filter(None, (m.group(2), m.group(3))):
            lord_claim(m.group(1), house)
    for m in _AS_ITS_LORD.finditer(sentence):
        lord_claim(m.group(2) or m.group(3), m.group(1))
    for m in _NTH_LORD_GRAHA.finditer(sentence):
        lord_claim(m.group(2), m.group(1))

    new_last = houses[-1].value if houses else last_house
    return claims, new_last


def _judge(claim: Claim, facts: ChartFacts, generic: bool, personal: bool) -> None:
    k = claim.kind
    if k == "placement_house":
        claim.truth = facts.house_of(claim.subject)
        ok = claim.truth == claim.value
    elif k == "placement_sign":
        claim.truth = facts.sign_of(claim.subject)
        ok = claim.truth == claim.value
    elif k == "lordship":
        claim.truth = facts.lord_of_house(claim.value)
        ok = claim.truth == claim.subject if claim.truth else None
    elif k == "influence":
        house = facts.houses.get(claim.value)
        related = house and (claim.subject in house.occupants or house.lord == claim.subject)
        claim.status = "verified" if related else ("generic" if generic else "unsupported")
        return
    elif k == "nakshatra":
        claim.truth = facts.moon_nakshatra
        ok = _canonical_nakshatra(str(claim.truth or "")) == claim.value if claim.truth else None
    elif k == "planet_nakshatra":
        claim.truth = (facts.planets.get(claim.subject) or {}).get("nakshatra")
        ok = _canonical_nakshatra(str(claim.truth or "")) == claim.value if claim.truth else None
    elif k == "house_sign":
        house, sign = claim.value
        fact = facts.houses.get(house)
        claim.truth = fact.sign if fact else None
        ok = (fact.sign == sign) if fact else None
    elif k == "lagna":
        claim.truth, ok = facts.lagna, facts.lagna == claim.value
    elif k == "moon_sign":
        claim.truth, ok = facts.moon_sign, facts.moon_sign == claim.value
    elif k in ("mahadasha", "antardasha"):
        claim.truth = facts.mahadasha if k == "mahadasha" else facts.antardasha
        ok = claim.truth == claim.value if claim.truth else None
    elif k == "retrograde":
        claim.truth = (facts.planets.get(claim.subject) or {}).get("retrograde")
        ok = claim.truth == claim.value if claim.truth is not None else None
    elif k == "yoga":
        name = str(claim.value)
        hit = lambda pool: any(name in y or y.startswith(name) for y in pool)  # noqa: E731
        ok = True if hit(facts.yogas_present) else (False if hit(facts.yogas_absent) else None)
        claim.truth = "present" if ok else ("absent" if ok is False else "not evaluated")
    elif k == "transit":
        # Saturn's current transit sign IS computed (Sade Sati data); every
        # other transit is not.
        if claim.subject == "saturn" and facts.saturn_transit_sign:
            claim.truth = facts.saturn_transit_sign
            named = [s for s, rx in _SIGN_RE if rx.search(claim.sentence)]
            ok = (facts.saturn_transit_sign in named) if named else True
        else:
            claim.truth, ok = "not computed", None
    elif k == "dosha":
        if claim.subject == "kaal sarp" and facts.kaal_sarp is not None:
            claim.truth, ok = facts.kaal_sarp, facts.kaal_sarp
        else:
            claim.truth, ok = "not computed", None
    else:
        ok = None

    if ok is None or claim.truth is None:
        claim.status = "generic" if generic else "unsupported"
    elif ok:
        claim.status = "verified"
    elif generic and not personal:
        claim.status = "generic"
    else:
        claim.status = "contradicted"


_DASHA_WORD = re.compile(r"maha\s*dasha|mahadasha|antar\s*dasha|antardasha|\bdasha|\bdasa\b|\bperiod\b|"
                         r"महादशा|अंतर्दशा|अन्तर्दशा|दशा", re.I)
_ANTAR_WORD = re.compile(r"antar|अंतर|अन्तर", re.I)
_MAHA_WORD = re.compile(r"maha|महा", re.I)
_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20\d\d|21\d\d)(?!\d)")


def _span_text(start: str, end: str) -> str:
    return f"{start[:7]} to {end[:7]}"


def _dasha_date_claim(sentence: str, index: int, facts: ChartFacts) -> Claim | None:
    """Years stated for a dasha period, checked against the engine's dates.

    The allowed years are those of the period(s) the sentence names (e.g. the
    Venus Antardasha); a year inside a period counts as consistent. Found in
    production: "Venus Antardasha … January 2021 to January 2028" when the
    engine had never given the model that period's dates at all.
    """
    years = sorted({int(y) for y in _YEAR.findall(sentence)})
    if not years or not _DASHA_WORD.search(sentence) or not facts.mahadasha_period:
        return None
    named = [g for g, rx in _GRAHA_RE if rx.search(sentence)]
    spans: list[tuple[str, str, str]] = []          # (label, start, end)
    antar, maha = bool(_ANTAR_WORD.search(sentence)), bool(_MAHA_WORD.search(sentence))
    for graha in named:
        if antar:
            spans += [(f"{graha} antardasha", s, e) for lord, s, e in facts.antardashas if lord == graha]
        if (maha or not antar) and graha == facts.mahadasha:
            spans.append((f"{graha} mahadasha", *facts.mahadasha_period))
    resolved = bool(spans)
    if not resolved:
        spans = [(f"{facts.mahadasha} mahadasha", *facts.mahadasha_period),
                 *((f"{lord} antardasha", s, e) for lord, s, e in facts.antardashas)]

    def inside(year: int) -> bool:
        return any(int(s[:4]) <= year <= int(e[:4]) for _, s, e in spans)

    label = ", ".join(dict.fromkeys(lbl for lbl, _, _ in spans)) if resolved else "dasha period"
    claim = Claim("dasha_dates", label, years, sentence=sentence, sentence_index=index,
                  truth="; ".join(f"{lbl} {_span_text(s, e)}" for lbl, s, e in spans[:4]))
    wrong_clause = _clause_dasha_error(sentence, facts, antar, maha) if resolved and len(named) > 1 else None
    if wrong_clause:
        claim.status, claim.truth = "contradicted", wrong_clause
    elif all(inside(y) for y in years):
        claim.status = "verified"
    elif resolved:
        claim.status = "contradicted"
    else:
        claim.status = "generic" if GENERIC.search(sentence) else "unsupported"
    return claim


_CLAUSE_BREAK = re.compile(r"\s*(?:;|,\s*(?:and|aur|but|lekin|then|phir|while|jabki|whereas)\b|\s+(?:and\s+then|"
                           r"and\s+after|aur\s+uske\s+baad|uske\s+baad)\b)", re.I)


def _clause_dasha_error(sentence: str, facts: ChartFacts, antar: bool, maha: bool) -> str | None:
    """A sentence naming two periods is checked clause by clause: "Rahu
    Antardasha (until July 2027) may pause it, and after July 2027 Saturn's
    antardasha can help" pooled both periods and passed (live browser turn;
    the engine's Saturn Antardasha starts in 2028)."""
    for clause in _CLAUSE_BREAK.split(sentence):
        years = [int(y) for y in _YEAR.findall(clause)]
        grahas = [g for g, rx in _GRAHA_RE if rx.search(clause)]
        if not years or not grahas:
            continue
        own: list[tuple[str, str, str]] = []
        for graha in grahas:
            if antar:
                own += [(f"{graha} antardasha", s, e) for lord, s, e in facts.antardashas if lord == graha]
            if (maha or not antar) and graha == facts.mahadasha and facts.mahadasha_period:
                own.append((f"{graha} mahadasha", *facts.mahadasha_period))
        if own and not all(any(int(s[:4]) <= y <= int(e[:4]) for _, s, e in own) for y in years):
            return "; ".join(f"{lbl} {_span_text(s, e)}" for lbl, s, e in own[:3])
    return None


def validate_reply(text: str, facts: ChartFacts) -> ValidationResult:
    result = ValidationResult()
    if not facts.available or not text:
        return result
    last_house: int | None = None
    for i, sentence in enumerate(split_sentences(text)):
        dated = _dasha_date_claim(sentence, i, facts)
        if dated is not None:
            result.claims.append(dated)
        claims, last_house = _extract(sentence, i, last_house)
        generic = bool(GENERIC.search(sentence))
        personal = bool(SECOND_PERSON.search(sentence))
        for claim in claims:
            # A bare "Jupiter in the 10th house brings…" with no verb and no
            # "your" reads as general astrology, not a statement about this chart.
            # …but only when it states a general effect. Verb-less chart
            # shorthand ("Mercury in 11th, lord of the 10th") is a claim
            # about this chart (found in human-eval batch review, H001).
            bare = claim.kind == "placement_house" and not personal and not re.search(
                r"\b(?:is|are|sits?|sitting|placed|positioned|posited|located|occupies|resides|hai|hain|baith\w*)\b"
                # "Mercury's placement in the 7th …" asserts where Mercury IS
                # (production reply; it was misread as general astrology).
                r"|'s\s+(?:placement|position)\b|बैठ|स्थित", sentence, re.I) and bool(re.search(
                r"\b(?:gives?|brings?|makes?|indicates?|creates?|denotes?|signifies|tends? to|is read as|"
                r"can|may|often|usually|deta|deti|dete)\b", sentence, re.I))
            if claim.kind == "lordship":
                # A house's lord depends on THIS chart's Lagna: "Traditionally 7th
                # house ka lord Venus yahan…" is a chart claim (HQB final run, wrong
                # lord passed as "generic"). Only a conditional or the natural
                # zodiac ("natural lord", karaka) is general astrology.
                _judge(claim, facts, bool(_LORD_GENERAL.search(sentence)), personal)
            else:
                _judge(claim, facts, generic or bare, personal)
            result.claims.append(claim)
        from app.evaluation.chart_relations import aspect_claims, conjunction_claims

        result.claims += conjunction_claims(sentence, i, facts)
        if not any(c.kind == "influence" and c.sentence_index == i for c in result.claims):
            result.claims += aspect_claims(sentence, i, facts)
    return result


_LORD_GENERAL = re.compile(r"\b(?:if|when|whenever|agar|jab|yadi|a person|someone|people|anyone|natural(?:ly)?|"
                           r"karaka|karak|kalpurush\w*|kaal\s*purush\w*)\b|अगर|जब|यदि|कारक", re.I)


def strip_sentences(text: str, indices: set[int]) -> str:
    """Remove the given sentences, keeping the paragraph breaks around the
    rest (chat bubbles are split on blank lines downstream)."""
    out = text
    for i, sentence in enumerate(split_sentences(text)):
        if i in indices:
            out = out.replace(sentence, "", 1)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n[ \t]+", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


__all__ = ["Claim", "ValidationResult", "validate_reply", "split_sentences", "strip_sentences"]
