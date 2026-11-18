"""Structured factual claims for the high-risk statements about a chart.

`chart_claims` extracts placements and houses with proximity rules. This module
covers the claim types where a wrong sentence does the most damage, each as a
typed `Claim` with a table-driven verifier against the Kundli engine's facts:

    sign_lord      "Kanya ka swami Mangal hai"        canonical sign -> lord table
    dasha_status   "Mars Antardasha chal rahi hai"    current / past / future, from the
                                                      engine's dated period sequence
    sade_sati      "Sade Sati chal rahi hai"          engine flag and phase
    dosha_verdict  "aapko Manglik dosha hai / nahi"   engine computes none of these

Extraction is lexicon and template based (English, Hinglish, Devanagari) and
runs on text whose Devanagari numerals were normalised first. No model is
involved anywhere. `structured_claims` is called from
`chart_claims.validate_reply`, so every caller gets these checks, and the older
regex checks stay as a second line of defence.

Two claim kinds are special:

* `sign_lord` needs no chart: a wrong lord is wrong for everyone, so it is
  checked even when no chart is available.
* `dosha_verdict` is never "verified". The engine computes only Kaal Sarp and
  Sade Sati; for every other dosha a verdict either way, including "you do not
  have it", is unsupported, because missing data is not evidence of absence.
"""

from __future__ import annotations

import re
from typing import Any

from app.evaluation.chart_claims import (
    GENERIC,
    GRAHA_ALIASES,
    NEGATION,
    SECOND_PERSON,
    SIGN_ALIASES,
    Claim,
    split_sentences,
)
from app.kundli.chart_facts import CO_LORDS, SIGN_LORDS, ChartFacts
from app.safety.notices import is_system_sentence

PLACEMENT_KINDS = frozenset({"placement_house", "placement_sign", "house_sign", "lagna"})
# An unsupported claim of one of these kinds is treated like a contradiction
# (the chart cannot confirm it, so it was invented), unlike an unsupported
# aspect or yoga, which is only recorded.
HARD_KINDS = PLACEMENT_KINDS | {"sade_sati", "dasha_status", "sign_lord", "sign_lord_not", "dasha_period",
                                 "dasha_boundary"}

_B = r"(?<![\wऀ-ॿ])"
_E = r"(?![\wऀ-ॿ])"
_GRAHA_BY_WORD = {w.lower(): g for g, ws in GRAHA_ALIASES.items() for w in ws}
_SIGN_BY_WORD = {w.lower(): s for s, ws in SIGN_ALIASES.items() for w in ws}


def _alt(words) -> str:
    return "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))


_G = rf"(?:{_alt(_GRAHA_BY_WORD)})"
_S = rf"(?:{_alt(_SIGN_BY_WORD)})"
_SIGN_ANY = re.compile(rf"{_B}{_S}{_E}", re.I)

# ── sign lordship ─────────────────────────────────────────────────────
_LORD = (r"(?:swami|swaami|lord|ruler|adhipati|adhipathi|maalik|malik|owner|"
         r"स्वामी|अधिपति|मालिक|राशिपति|राशीश)")
_RASHI = r"(?:\s+(?:rashi|raashi|rasi|sign|lagna|lagan|ascendant|rising|राशि|लग्न))?"
_POSS = r"(?:\s*'s|\s+(?:ka|ke|ki|का|के|की))?"
_PART = r"(?:\s+(?:hi|bhi|toh|to|hee|ही|भी))?"
_COP = r"(?:\s+(?:is|are|hai|hain|ho|being|है|हैं))?"
_SIGN_LIST = rf"{_S}(?:\s*(?:,|and|&|aur|और)\s*{_S})*"

# Tried in this order; a sign already used by an earlier template is not reused.
_LORD_TEMPLATES = tuple(re.compile(rx, re.I) for rx in (
    # "Virgo ka swami hi Mangal hai", "कन्या राशि का स्वामी बुध है", "Virgo's ruler is Mars"
    rf"{_B}(?P<ss>{_S}){_E}{_RASHI}{_POSS}\s+{_LORD}{_E}{_PART}{_COP}\s+(?P<g>{_G}){_E}",
    # "Virgo is ruled by Mercury"
    rf"{_B}(?P<ss>{_S}){_E}{_RASHI}\s+(?:is\s+|are\s+)?(?:ruled|owned|governed|lorded)\s+by\s+(?P<g>{_G}){_E}",
    # "Mars rules Aries and Scorpio", "Mercury is the lord of Virgo"
    rf"{_B}(?P<g>{_G}){_E}\s+(?:is\s+the\s+(?:lord|ruler|owner)\s+of|is\s+lord\s+of|rules|owns|governs|lords)"
    rf"\s+(?:the\s+)?(?:signs?\s+)?(?P<ss>{_SIGN_LIST}){_E}",
    # "the lord of Virgo is Mercury"
    rf"(?:lord|ruler|owner)\s+of\s+(?:the\s+)?(?:sign\s+)?(?P<ss>{_S}){_E}(?:\s*,\s*|\s+(?:is|being)\s+|\s+)"
    rf"(?P<g>{_G}){_E}",
    # "Mars is Virgo's lord"
    rf"{_B}(?P<g>{_G}){_E}\s+(?:is|hai)\s+(?P<ss>{_S})(?:'s|\s+ka|\s+ke|\s+ki)\s+{_LORD}{_E}",
    # "Mangal Kanya ka swami hai", "बुध कन्या राशि का स्वामी है"
    rf"{_B}(?P<g>{_G}){_E}\s+(?P<ss>{_SIGN_LIST}){_RASHI}{_POSS}\s+{_LORD}{_E}",
    # "कन्या पर बुध का आधिपत्य"
    rf"{_B}(?P<ss>{_S}){_E}{_RASHI}\s+(?:par|पर)\s+(?P<g>{_G}){_E}\s+(?:ka|का)\s+(?:aadhipatya|adhipatya|आधिपत्य)",
))


def sign_lord_claims(sentence: str, index: int) -> list[Claim]:
    claims: list[Claim] = []
    used: list[tuple[int, int]] = []
    for rx in _LORD_TEMPLATES:
        for m in rx.finditer(sentence):
            signs = [_SIGN_BY_WORD[s.lower()] for s in _SIGN_ANY.findall(m.group("ss"))]
            span = (m.start("ss"), m.end("ss"))
            if not signs or any(a < span[1] and span[0] < b for a, b in used):
                continue
            used.append(span)
            graha = _GRAHA_BY_WORD[m.group("g").lower()]
            # "…ka swami Mangal nahi hai" says Mars is NOT the lord: that is
            # false when it is. Negation counts only right at the claim.
            negative = bool(NEGATION.search(sentence[m.start():m.end() + 16]))
            for sign in signs:
                is_lord = graha in {SIGN_LORDS[sign], *CO_LORDS.get(sign, ())}
                claims.append(Claim("sign_lord_not" if negative else "sign_lord", sign, graha,
                                    status="verified" if is_lord != negative else "contradicted",
                                    truth=SIGN_LORDS[sign], sentence=sentence, sentence_index=index))
    return claims


# ── Dasha status: current / past / future ─────────────────────────────
_LEVEL = re.compile(
    r"(?P<maha>maha\s*-?\s*dasha|mahadasha|महादशा)|"
    r"(?P<antar>antar\s*-?\s*dasha|antardasha|अंतर्दशा|अन्तर्दशा|bhukti|भुक्ति)|"
    r"(?P<plain>(?<![a-z])dasha|(?<![a-z])dasa|दशा)", re.I)
_PAST = re.compile(
    r"chal(?:i|a|e)?\s+(?:rahi|raha|rahe)\s+(?:thi|tha|the)|chali\s+thi|chala\s+tha|chal\s+chuk[ia]|"
    r"khatam\s+ho\s+(?:chuk[ia]|gay[ia]|gai)|ho\s+chuk[ia]|beet\s+(?:chuk[ia]|gay[ia]|gai)|samapt|"
    r"\bended\b|\bhas\s+ended\b|\bwas\s+running\b|\bhad\s+been\b|\bpreviously\b|\bearlier\b|\bin\s+the\s+past\b|"
    r"\balready\s+(?:ended|over|finished|passed)\b|\bwas\b|\bwere\b|\bover\s+now\b|\bfinished\b|"
    r"चली\s+थी|चला\s+था|चल\s+चुक[ीा]|समाप्त|खत्म\s+हो|बीत\s+(?:चुक|गई|गया)|पिछली|पिछला|पहले\s+की|"
    # Perfective "ran": "Mercury Antardasha 2025 se 2027 tak chali" (live E2E: shipped as
    # past for the CURRENT period). Not "chali aa rahi hai", which is still going.
    r"\bchal(?:i|a)\b(?!\s+(?:rahi|raha|aa|ja))|\bran\b|\blasted\b|shuru\s+hu(?:i|a)\s+th[iae]\b|"
    rf"थी{_E}|था{_E}|चली{_E}(?!\s+(?:रही|आ|जा))|चला{_E}(?!\s+(?:रहा|आ|जा))", re.I)
_CURRENT = re.compile(
    r"chal\s+rah[iae]|chalti\s+hai|\babhi\b|filh?aal|\bcurrently\b|\bcurrent\b|\bpresently\b|\bright\s+now\b|"
    r"\bat\s+present\b|\bongoing\b|\bis\s+running\b|\bare\s+running\b|\brunning\b|\bunder\s+(?:the\s+)?(?:influence\s+of\s+)?"
    r"|\byou\s+are\s+in\b|\byou're\s+in\b|\bnow\b|"
    # "Ab aap Ketu Antardasha mein hain" (live E2E: said of a FUTURE period), and
    # "Ketu antardasha November 2027 se shuru hui hai" (has begun; live, same period).
    r"\bab\b|shuru\s+(?:hui|hua|ho\s+(?:gayi|gai|gaya|chuki|chuka))\b(?!\s+th[iae]\b)|"
    r"\b(?:has|have)\s+(?:already\s+)?(?:started|begun)\b|शुरू\s+(?:हुई|हुआ|हो\s+(?:गई|गया|चुकी))(?!\s+थ)|"
    r"\baap\b[^.,;]{0,40}?\b(?:mein|me)\s+(?:hain|hai|ho)\b|"
    rf"अब{_E}|आप[^।,;.]{{0,40}}?में\s+(?:हैं|है){_E}|"
    r"चल\s+रह[ीाे]|चलती\s+है|अभी|फिलहाल|वर्तमान|इस\s+समय|चल\s+रही", re.I)
_FUTURE = re.compile(
    r"aane\s+wal[ia]|aayeg[ia]|agl[ie]\b|shuru\s+hog[ia]|start\s+hog[ia]|\bwill\s+(?:start|begin|come|commence)\b|"
    r"\bupcoming\b|\bnext\b|\bbegins\b|\bstarts\s+from\b|\bwill\s+be\s+(?:the\s+)?next\b|"
    r"आने\s+वाल[ीा]|आएगी|आएगा|अगली|अगला|शुरू\s+होगी|शुरू\s+होगा", re.I)
_TENSES = (("past", _PAST), ("future", _FUTURE), ("current", _CURRENT))
_CONDITIONAL = re.compile(r"\b(?:if|when|whenever|agar|jab|yadi)\b|अगर|जब|यदि", re.I)
_PAIR_GAP = 25      # characters between a Dasha word and its planet
_MARKER_GAP = 45    # characters between a tense word and the period it describes
_STRONG_BREAK = re.compile(
    r"\s*(?:;|\s+(?:uske\s+baad|iske\s+baad|baad\s+mein|after\s+that|and\s+then|then|phir|fir|but|lekin|jabki|while|"
    r"whereas|however)\b|\s+(?:उसके|इसके)\s+बाद|\s+फिर\s|\s+लेकिन\s|\s+जबकि\s)\s*", re.I)
_GRAHA_FINDERS = tuple((g, re.compile(rf"{_B}(?:{_alt(ws)}){_E}", re.I)) for g, ws in GRAHA_ALIASES.items())


def _pairs(chunk: str) -> list[tuple[str, str, int, int]]:
    """(graha, level, start, end) for each Dasha word and the planet it names.
    Greedy by distance, one planet per Dasha word, so "महादशा शुक्र की है और
    अंतर्दशा बुध की" pairs each word with the planet right after it."""
    levels = [(m, "maha" if m.group("maha") else "antar" if m.group("antar") else "any")
              for m in _LEVEL.finditer(chunk)]
    grahas = [(g, m) for g, rx in _GRAHA_FINDERS for m in rx.finditer(chunk)]
    options = []
    for li, (lm, _) in enumerate(levels):
        for gi, (_, gm) in enumerate(grahas):
            gap = lm.start() - gm.end() if gm.end() <= lm.start() else gm.start() - lm.end()
            if 0 <= gap <= _PAIR_GAP:
                options.append((gap, li, gi))
    out, used_l, used_g = [], set(), set()
    for gap, li, gi in sorted(options):
        if li in used_l or gi in used_g:
            continue
        used_l.add(li)
        used_g.add(gi)
        lm, level = levels[li]
        graha, gm = grahas[gi]
        out.append((graha, level, min(lm.start(), gm.start()), max(lm.end(), gm.end())))
    return sorted(out, key=lambda p: p[2])


# Phrases that carry their own polarity: "has not ended" denies the past,
# "has not started yet" ASSERTS the future. Matched first so the plain tense
# words inside them ("ended", "started") are not read a second time.
_POLAR = (
    ("past", True, re.compile(
        r"khatam\s+(?:nahi|nahin)\s+hu[ie]|has(?:n't|\s+not)\s+ended|not\s+(?:yet\s+)?(?:ended|over|finished)|"
        r"समाप्त\s+नहीं|खत्म\s+नहीं", re.I)),
    ("future", False, re.compile(
        r"(?:shuru|start)\s+(?:nahi|nahin)\s+hu[ie]|has(?:n't|\s+not)\s+(?:yet\s+)?(?:started|begun)|"
        r"not\s+(?:yet\s+)?(?:started|begun)|yet\s+to\s+(?:start|begin)|शुरू\s+नहीं\s+हुई?", re.I)),
)


def _tense_markers(chunk: str) -> list[tuple[str, int, int, bool | None]]:
    """(tense, start, end, polarity) for each tense word; polarity is True for
    "denied", False for "asserted", None when the caller should look for a
    negation next to it. A present-tense match inside a past-tense phrase
    ("chal rahi thi") is dropped."""
    found: list[tuple[str, int, int, bool | None]] = []
    for tense, negated, rx in _POLAR:
        found += [(tense, m.start(), m.end(), negated) for m in rx.finditer(chunk)]
    specials = [(a, b) for _, a, b, _ in found]
    for tense, rx in _TENSES:
        for m in rx.finditer(chunk):
            if any(a < m.end() and m.start() < b for a, b in specials):
                continue
            if tense == "current" and any(t == "past" and a <= m.start() < b for t, a, b, _ in found):
                continue
            found.append((tense, m.start(), m.end(), None))
    return found


def _verify_dasha(tense: str, level: str, graha: str, facts: ChartFacts,
                  negated: bool = False) -> tuple[str, str]:
    maha = facts.dasha_status(graha, "maha")
    antar = facts.dasha_status(graha, "antar")
    known = [s for s in ((maha if level in ("maha", "any") else None), (antar if level in ("antar", "any") else None))
             if s and s[0]]
    if not known:
        return "generic", ""
    statuses = {s for s, _ in known}
    truth = "; ".join(t for _, t in known)
    if negated:
        # "is NOT running", "has NOT ended": wrong exactly when the thing it denies is true.
        return ("contradicted" if tense in statuses else "verified"), truth
    if tense == "current":
        return ("verified" if "current" in statuses else "contradicted"), truth
    if "current" in statuses:
        return "contradicted", truth              # claims it is over / not yet begun
    wanted = {"past": {"past"}, "future": {"future"}}[tense]
    if statuses & wanted:
        return "verified", truth
    # A non-running Mahadasha could be past or future: not decidable.
    return ("contradicted" if statuses - {"not_current"} else "generic"), truth


# ── Dasha boundaries: which year a period STARTS or ENDS ──────────────
_YEAR_TOKEN = re.compile(r"(?<!\d)(19[5-9]\d|20\d\d|21\d\d)(?!\d)")
_START_AFTER = re.compile(rf"^\s*(?:ke\s+\w+\s+)?(?:se|से){_E}|^\s*(?:mein|me|में)\s+(?:shuru|start|शुरू)", re.I)
_END_AFTER = re.compile(rf"^\s*(?:ke\s+\w+\s+)?(?:tak|तक){_E}|^\s*(?:mein|me|में|ko)\s+(?:khatam|samapt|खत्म|समाप्त)", re.I)
_START_BEFORE = re.compile(r"(?:\bfrom|\bsince|\bstarting|\bstarts?\s+(?:in|from)|\bbegins?\s+in|\bbegan\s+in|"
                           r"\bstarted\s+in)\s+(?:[a-z]+\s+)?$", re.I)
_END_BEFORE = re.compile(r"(?:\buntil|\btill|\bthrough|\bto|\bends?\s+in|\bended\s+in|\bending\s+in)\s+(?:[a-z]+\s+)?$",
                         re.I)


def _period(graha: str, level: str, facts: ChartFacts) -> tuple[str, str] | None:
    if level in ("maha", "any") and graha == facts.mahadasha and facts.mahadasha_period:
        return facts.mahadasha_period
    if level in ("antar", "any"):
        for lord, start, end in facts.antardashas:
            if lord == graha:
                return start, end
    return None


def dasha_boundary_claims(sentence: str, index: int, facts: ChartFacts) -> list[Claim]:
    """"Ketu Antardasha January 2029 se shuru hogi" when Ketu's Antardasha STARTS
    in 2027 and ENDS in 2029 (live E2E 2026-11-18): every year sits inside the
    period, so a year-in-range check passed it. Each year that a boundary word
    marks as a start ("se", "from", "starts in") or an end ("tak", "until",
    "ends in") is compared with that end of the engine's period. A date for a
    Mahadasha that is not the running one cannot be checked (only the running
    one is dated) and is unsupported."""
    claims: list[Claim] = []
    for chunk in _STRONG_BREAK.split(sentence):
        pairs = _pairs(chunk)
        if not pairs:
            continue
        for m in _YEAR_TOKEN.finditer(chunk):
            after, before = chunk[m.end():m.end() + 24], chunk[max(0, m.start() - 24):m.start()]
            edge = ("start" if _START_AFTER.search(after) or _START_BEFORE.search(before) else
                    "end" if _END_AFTER.search(after) or _END_BEFORE.search(before) else None)
            if edge is None:
                continue
            preceding = [p for p in pairs if p[3] <= m.start() and m.start() - p[3] <= 70]
            following = [p for p in pairs if p[2] >= m.end() and p[2] - m.end() <= 40]
            owner = preceding[-1] if preceding else (following[0] if following else None)
            if owner is None:
                continue
            graha, level = owner[0], owner[1]
            name = f"{graha} {'antardasha' if level == 'antar' else 'mahadasha' if level == 'maha' else 'dasha'}"
            year = int(m.group(1))
            if level == "maha" and facts.mahadasha and graha != facts.mahadasha:
                claims.append(Claim("dasha_period", name, f"{edge}s {year}", status="unsupported",
                                    truth=f"only the running {facts.mahadasha} Mahadasha is dated",
                                    sentence=sentence, sentence_index=index))
                continue
            period = _period(graha, level, facts)
            if period is None:
                continue
            actual = int((period[0] if edge == "start" else period[1])[:4])
            claims.append(Claim("dasha_boundary", name, f"{edge}s {year}",
                                status="verified" if actual == year else "contradicted",
                                truth=f"{period[0][:7]} to {period[1][:7]}", sentence=sentence, sentence_index=index))
    return claims


def dasha_status_claims(sentence: str, index: int, facts: ChartFacts) -> list[Claim]:
    claims: list[Claim] = []
    for chunk in _STRONG_BREAK.split(sentence):
        pairs = _pairs(chunk)
        if not pairs:
            continue
        owned: dict[int, set[tuple[str, bool]]] = {}
        for tense, a, b, polarity in _tense_markers(chunk):
            gaps = [(max(0, a - pe, ps - b), i) for i, (_, _, ps, pe) in enumerate(pairs)]
            gap, nearest = min(gaps)
            if gap <= _MARKER_GAP:
                # "is not running", "nahi chal rahi", "नहीं चल रही": a negation right
                # next to the tense word denies it (unless the phrase fixed its own polarity).
                negated = polarity if polarity is not None else bool(
                    NEGATION.search(chunk[max(0, a - 16):b + 12]))
                owned.setdefault(nearest, set()).add((tense, negated))
        for i, (graha, level, _, _) in enumerate(pairs):
            readings = owned.get(i, set())
            if len(readings) != 1:
                continue                          # no tense word, or contradictory ones: nothing to judge
            tense, negated = next(iter(readings))
            status, truth = _verify_dasha(tense, level, graha, facts, negated)
            name = "antardasha" if level == "antar" else "mahadasha" if level == "maha" else "dasha"
            claims.append(Claim("dasha_status", f"{graha} {name}", f"not {tense}" if negated else tense,
                                status=status, truth=truth, sentence=sentence, sentence_index=index))
    return claims


# ── Sade Sati ─────────────────────────────────────────────────────────
_SADE = re.compile(r"sade\s*-?\s*sati|saadhe\s*-?\s*saati|saade\s*-?\s*saati|sadesati|साढ़े\s*साती|साढे\s*साती|"
                   r"साढ़ेसाती|साढेसाती|साड़े\s*साती", re.I)
_PHASES = (
    ("rising", re.compile(r"\brising\b|first\s+phase|1st\s+phase|\bpehla\b|\bpehle\s+phase\b|पहला|पहले\s+चरण", re.I)),
    ("peak", re.compile(r"\bpeak\b|second\s+phase|2nd\s+phase|\bmiddle\b|\bdoosra\b|\bdusra\b|दूसरा|चरम", re.I)),
    ("setting", re.compile(r"\bsetting\b|third\s+phase|3rd\s+phase|last\s+phase|final\s+phase|\bteesra\b|\btisra\b|"
                           r"तीसरा|अंतिम", re.I)),
)


def sade_sati_claims(sentence: str, index: int, facts: ChartFacts, generic: bool) -> list[Claim]:
    m = _SADE.search(sentence)
    if not m:
        return []
    negative = bool(NEGATION.search(sentence[max(0, m.start() - 12):m.end() + 30]))
    claims = [Claim("sade_sati", "sade sati", not negative, sentence=sentence, sentence_index=index)]
    claim = claims[0]
    if generic:
        claim.status = "generic"
    elif facts.sade_sati is None:
        claim.status, claim.truth = "unsupported", "not computed"
    else:
        claim.truth = facts.sade_sati
        claim.status = "verified" if claim.value == facts.sade_sati else "contradicted"
    phases = [p for p, rx in _PHASES if rx.search(sentence)]
    if len(phases) == 1 and not negative and facts.sade_sati and facts.sade_sati_phase and not generic:
        claims.append(Claim("sade_sati_phase", "sade sati", phases[0], truth=facts.sade_sati_phase,
                            status="verified" if phases[0] == facts.sade_sati_phase else "contradicted",
                            sentence=sentence, sentence_index=index))
    return claims


# ── dosha verdicts ────────────────────────────────────────────────────
_DOSHA = re.compile(
    r"manglik|mangalik|mangal\s+dosh\w*|kuja?\s+dosh\w*|angarak|pitra\s+dosh\w*|pitru\s+dosh\w*|shrapit|"
    r"kemadrum\w*|guru\s+chandal|grahan\s+dosh\w*|nadi\s+dosh\w*|bhakoot\s+dosh\w*|gand\s*mool|\bdosh(?:a|as|am)?\b|"
    r"मांगलिक|मांगलीक|मंगल\s+दोष|पितृ\s+दोष|काल\s*सर्प|दोष", re.I)
_COMPUTED = re.compile(r"kaal\s*sarp\w*(?:\s+dosh\w*)?|kalsarp\w*(?:\s+dosh\w*)?|काल\s*सर्प\w*(?:\s+दोष)?", re.I)
_SADE_ANY = re.compile(_SADE.pattern + r"(?:\s+dosh\w*|\s+दोष)?", re.I)


def mentions_uncomputed_dosha(sentence: str, facts: ChartFacts | None) -> bool:
    """A sentence about a dosha the engine does not compute. Kaal Sarp and Sade
    Sati are computed (when the engine returned them) and are judged against
    its values instead. The assistant's own fixed notices are exempt."""
    if is_system_sentence(sentence):
        return False
    text = sentence
    if facts is not None and facts.available:
        if facts.kaal_sarp is not None:
            text = _COMPUTED.sub(" ", text)
        if facts.sade_sati is not None:
            text = _SADE_ANY.sub(" ", text)
    return bool(_DOSHA.search(text))


def mentions_computed_dosha(sentence: str, facts: ChartFacts | None) -> bool:
    """A sentence about Kaal Sarp or Sade Sati, the doshas the engine computes
    (and only when it returned them), that says nothing about any other dosha."""
    if facts is None or not facts.available or mentions_uncomputed_dosha(sentence, facts):
        return False
    return bool((facts.kaal_sarp is not None and _COMPUTED.search(sentence))
                or (facts.sade_sati is not None and _SADE.search(sentence)))


def dosha_claims(sentence: str, index: int, facts: ChartFacts) -> list[Claim]:
    if not mentions_uncomputed_dosha(sentence, facts):
        return []
    return [Claim("dosha_verdict", "dosha", True, status="unsupported", truth="not computed",
                  sentence=sentence, sentence_index=index)]


def structured_claims(text: str, facts: ChartFacts) -> list[Claim]:
    """All structured claims in `text` (already numeral-normalised)."""
    out: list[Claim] = []
    for i, sentence in enumerate(split_sentences(text)):
        out += sign_lord_claims(sentence, i)
        out += dosha_claims(sentence, i, facts)
        if facts.available:
            personal = bool(SECOND_PERSON.search(sentence))
            generic = bool(GENERIC.search(sentence)) and not personal
            # "Traditionally"/"usually" do not make "…chal rahi hai" a general
            # statement; only an explicit condition does.
            if personal or not _CONDITIONAL.search(sentence):
                out += dasha_status_claims(sentence, i, facts)
                out += dasha_boundary_claims(sentence, i, facts)
            out += sade_sati_claims(sentence, i, facts, generic)
    return out


def claim_signature(claim: Claim) -> tuple[Any, ...]:
    """Identity of a claim independent of wording and language, for comparing a
    rewrite with the answer it rewrote."""
    return (claim.kind, str(claim.subject), str(claim.value))


__all__ = ["HARD_KINDS", "PLACEMENT_KINDS", "structured_claims", "sign_lord_claims",
           "dasha_status_claims", "sade_sati_claims", "dosha_claims", "mentions_uncomputed_dosha",
           "mentions_computed_dosha", "claim_signature"]
