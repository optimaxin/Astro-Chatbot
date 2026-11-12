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
    3: ("third", "teesre", "tisre", "teesra", "तीसरे"), 4: ("fourth", "chauthe", "chautha", "चौथे"),
    5: ("fifth", "paanchve", "panchve", "paanchva", "पांचवें", "पाँचवें"), 6: ("sixth", "chhathe", "chhate", "छठे"),
    7: ("seventh", "saatve", "satve", "saatva", "सातवें"), 8: ("eighth", "aathve", "athve", "आठवें"),
    9: ("ninth", "nauve", "nauva", "नौवें"), 10: ("tenth", "dasve", "dasven", "daswe", "dasva", "दसवें"),
    11: ("eleventh", "gyarahve", "gyarve", "ग्यारहवें"), 12: ("twelfth", "barahve", "barhve", "बारहवें"),
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
PLACE_VERBS = (r"\b(?:is|are|was|sits?|sitting|sat|placed|positioned|posited|located|occupies|occupying|"
               r"resides|residing|stationed|lies|found|in|baitha|baithe|baithi|sthit|hai|hain)\b|में|बैठ|स्थित")
CONTAIN_VERBS = r"\b(?:contains?|has|have|hosts?|holds?|houses|is occupied by|occupied by|mein\s+hai)\b"
INFLUENCE_VERBS = (r"\b(?:influences?|influencing|aspects?|aspecting|casts?|looks? at|affects?|impacts?|"
                   r"blesses|strengthens|activates|energi[sz]es|drishti|nazar)\b")
LORD_FWD = r"\b(?:rules?|ruling|owns?|lords? over|is the lord of|is lord of|is the ruler of|swami)\b"
LORD_BACK = r"\b(?:ruled by|owned by|lord is|ruler is|lord,|ka swami|ke swami|ki swami)\b"
NEGATION = re.compile(r"\b(?:not|isn't|aren't|never|no|nahi|nahin|na)\b|n't\b|नहीं|न\s", re.I)
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
        rf"(?:\b{_n}(?:st|nd|rd|th|वें|वे)?\s*{HOUSE_NOUN}|\bhouse\s+{_n}\b|(?:{words})\s*{HOUSE_NOUN}|"
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
            if (g.start, h.start) in used or not _no_graha_between(mentions, g, h):
                continue
            mid = _between(sentence, g, h)
            if len(mid) > 60:
                continue
            after_h = sentence[h.end:h.end + 12]
            if g.start < h.start and (re.search(PLACE_VERBS, mid, re.I) or
                                      re.match(r"\s*(?:mein|me|में)\b", after_h, re.I)):
                add("placement_house", g.value, h.value)
            elif h.start < g.start and (re.search(CONTAIN_VERBS, mid, re.I) or
                                        re.match(r"^\s*(?:mein|me|में)\s", mid, re.I)):
                add("placement_house", g.value, h.value)
            elif re.search(INFLUENCE_VERBS, mid, re.I):
                add("influence", g.value, h.value)

    for s in signs:
        for g in grahas:
            if not _no_graha_between(mentions, g, s):
                continue
            mid = _between(sentence, g, s)
            after_s = sentence[s.end:s.end + 18]
            if g.start < s.start and len(mid) <= 40 and (
                    re.search(r"\b(?:in|mein|me|is|hai)\b|में|राशि", mid, re.I)
                    or re.match(r"\s*(?:rashi|raashi|राशि)?\s*(?:mein|me|में)\b", after_s, re.I)):
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

    lagna = nearest_sign(r"\b(?:ascendant|lagna|lagan|rising)\b|लग्न")
    if lagna:
        add("lagna", "lagna", lagna)
    moon_sign = nearest_sign(r"\b(?:moon sign|chandra rashi|rashi|raashi)\b|राशि")
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
    for yoga in re.findall(r"([a-z][a-z\-]+(?:\s[a-z][a-z\-]+)?)\s+yoga\b", lowered):
        name = " ".join(w for w in yoga.split() if w not in filler)
        if name and re.search(r"\b(?:you have|your chart has|forms?|formed|present|there is|banta|ban raha)\b",
                              lowered):
            add("yoga", "yoga", name)
    if re.search(r"\bkaal\s*sarp", lowered) and re.search(r"\b(?:have|has|present|there is|hai)\b", lowered):
        add("dosha", "kaal sarp", True)
    if re.search(r"\b(?:manglik|mangal dosha|kuja dosha)\b", lowered) and \
            re.search(r"\b(?:you are|you're|you have|hai|ho)\b", lowered):
        add("dosha", "manglik", True)

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


def validate_reply(text: str, facts: ChartFacts) -> ValidationResult:
    result = ValidationResult()
    if not facts.available or not text:
        return result
    last_house: int | None = None
    for i, sentence in enumerate(split_sentences(text)):
        claims, last_house = _extract(sentence, i, last_house)
        generic = bool(GENERIC.search(sentence))
        personal = bool(SECOND_PERSON.search(sentence))
        for claim in claims:
            # A bare "Jupiter in the 10th house brings…" with no verb and no
            # "your" reads as general astrology, not a statement about this chart.
            bare = claim.kind == "placement_house" and not personal and not re.search(
                r"\b(?:is|are|sits?|sitting|placed|positioned|posited|located|occupies|resides|hai|hain|baith\w*)\b"
                r"|बैठ|स्थित", sentence, re.I)
            _judge(claim, facts, generic or bare, personal)
            result.claims.append(claim)
    return result


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
