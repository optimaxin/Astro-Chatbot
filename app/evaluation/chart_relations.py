"""Conjunction claims, checked against the Kundli engine's placements.

Root cause (final quality gate): `chart_claims` checks where ONE planet is,
but a sentence that puts two planets together was never checked. A live
browser reply said "Rahu in Gemini (5th house) and Saturn in Taurus (4th
house), both conjunct your Moon and Jupiter" — the house claims were right,
the conjunction was invented, and it reached the user.

A conjunction claim is any explicit "together" relation between named grahas:

    "Rahu … conjunct your Moon and Jupiter", "Mars-Venus conjunction",
    "Shani aur Chandra ki yuti", "Jupiter ke saath Sun aur Mercury bhi 6th mein
    hai", "सूर्य और बुध की युति", "Jupiter with Venus in the same house"

It is TRUE only when every named graha sits in the same house (the chart is
whole-sign, so same house = same sign; `ChartFacts.conjunctions` is derived
the same way). Engine output is the only source; nothing is computed here
beyond comparing house numbers. Aspects are not checked: the engine does not
compute them, and `chart_claims` already marks aspect statements unsupported.
"""

from __future__ import annotations

import re

from app.evaluation.chart_claims import GENERIC, NEGATION, SECOND_PERSON, Claim, _GRAHA_RE
from app.kundli.chart_facts import ChartFacts

# The relation words. "with"/"saath" alone are too common to count unless the
# sentence also says the planets share a house.
_KEYWORD = re.compile(
    r"\bconjunct(?:ion|s|ed)?\b|\bconjoin\w*|\bjoined\s+by\b|\byuti\b|\byog\s+bana\w*|युति|"
    r"\b(?:together|ek\s+saath|saath\s+mein|saath\s+baith\w*|same\s+(?:house|sign)|ek\s+hi\s+(?:house|ghar|bhav|rashi))\b|"
    r"एक\s+साथ|एक\s+ही\s+भाव|साथ\s+बैठ",
    re.I)
_SAATH = re.compile(r"\b(?:ke\s+saath|with|along\s+with|alongside)\b|के\s+साथ", re.I)
# Words that may sit between grahas of one group, or between a group and the keyword.
_GLUE = re.compile(
    r"^(?:\s|,|&|/|-|–|\band\b|\baur\b|\bboth\b|\bdono\b|\byour\b|\baapke\b|\baapki\b|\bnatal\b|\bthe\b|\bis\b|"
    r"\bare\b|\bwith\b|\bke\b|\bsaath\b|\bki\b|\bka\b|\bbhi\b|\bclosely\b|\bin\b|\bhai\b|\bhain\b|और|की|का|के|"
    r"साथ|भी|दोनों)*$", re.I)


def _mentions(sentence: str) -> list[tuple[int, int, str]]:
    found = []
    for graha, rx in _GRAHA_RE:
        for m in rx.finditer(sentence):
            if graha == "moon" and re.match(r"\s*(?:sign|rashi)", sentence[m.end():], re.I):
                continue
            found.append((m.start(), m.end(), graha))
    found.sort()
    return found


_BOTH = re.compile(r"(?:\bboth\b|\bdono\b|दोनों)[\s,]*(?:\b(?:are|is|hai|hain)\b\s*)?$", re.I)


def _group(sentence: str, mentions: list[tuple[int, int, str]], start: int, end: int) -> set[str]:
    """Grahas joined to the span [start, end) by glue words only, both ways.

    "X and Y, both conjunct Z" names its left side with "both": the two
    nearest grahas before it. A one-sided relation ("…, jo Sun ke saath yuti
    banata hai") takes the nearest graha before it as the other side."""
    group: set[str] = set()
    before = [m for m in mentions if m[1] <= start]
    left = start
    for s, e, graha in reversed(before):
        if not _GLUE.match(sentence[e:left]):
            break
        group.add(graha)
        left = s
    if not group and _BOTH.search(sentence[max(0, start - 25):start]):
        for _, _, graha in reversed(before):
            if len(group) == 2:
                break
            group.add(graha)
        left = min((s for s, _, g in before if g in group), default=start)
    right = end
    for s, e, graha in [m for m in mentions if m[0] >= end]:
        if not _GLUE.match(sentence[right:s]):
            break
        group.add(graha)
        right = e
    if len(group) == 1:
        prior = [m for m in mentions if m[1] <= left and left - m[1] <= 80 and m[2] not in group]
        if prior:
            group.add(prior[-1][2])
    return group


def conjunction_claims(sentence: str, index: int, facts: ChartFacts) -> list[Claim]:
    if not facts.available or NEGATION.search(sentence):
        return []
    mentions = _mentions(sentence)
    if len({g for _, _, g in mentions}) < 2:
        return []
    groups: list[set[str]] = []
    for m in _KEYWORD.finditer(sentence):
        groups.append(_group(sentence, mentions, m.start(), m.end()))
    if re.search(r"\bsame\s+house\b|\bek\s+hi\s+(?:house|ghar|bhav)\b|एक\s+ही\s+भाव", sentence, re.I):
        for m in _SAATH.finditer(sentence):
            groups.append(_group(sentence, mentions, m.start(), m.end()))
    claims: list[Claim] = []
    seen: set[frozenset[str]] = set()
    personal = bool(SECOND_PERSON.search(sentence))
    # "Venus conjunct Mars gives passion" with no "your" and no "is/hai" reads
    # as general astrology, like a bare placement in chart_claims.
    stated = bool(re.search(r"\b(?:is|are|sits?|placed|hai|hain|baith\w*)\b|बैठ|स्थित|है", sentence, re.I))
    generic = not personal and (bool(GENERIC.search(sentence)) or not stated)
    for group in groups:
        key = frozenset(group)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        houses = {g: facts.house_of(g) for g in sorted(key)}
        if any(h is None for h in houses.values()):
            continue
        claim = Claim("conjunction", " + ".join(sorted(key)), sorted(key), sentence=sentence, sentence_index=index,
                      truth=", ".join(f"{g} in house {h}" for g, h in houses.items()))
        claim.status = "verified" if len(set(houses.values())) == 1 else ("generic" if generic else "contradicted")
        claims.append(claim)
    return claims


_ASPECT = re.compile(
    r"(?:\b(?:aspects?|aspecting|drishti|dekhta|dekhti|dekhte)\b|दृष्टि|देखता|देखती)", re.I)
_ORDINAL_HOUSE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b|(?:पहले|दूसरे|तीसरे|चौथे|पांचवें|पाँचवें|छठे|सातवें|आठवें|"
                            r"नौवें|दसवें|ग्यारहवें|बारहवें)\s+भाव"
                            # Sanskrit ordinals ("शनि की सप्तम पर दृष्टि"), but not
                            # the house-lord form ("सप्तमेश").
                            r"|(?:प्रथम|द्वितीय|तृतीय|चतुर्थ|पंचम|षष्ठ|सप्तम|अष्टम|नवम|दशम|एकादश|द्वादश)(?![ऀ-ॿ])"
                            # Digit + Hindi ordinal suffix: "शनि की 7वें भाव पर दृष्टि".
                            r"|(?<!\d)\d{1,2}\s*(?:वें|वां|वाँ|वीं)(?![ऀ-ॿ])", re.I)
# Negation only counts in the clause that states the aspect: "शनि की सप्तम पर
# दृष्टि … है, लेकिन विवाह से इंकार नहीं करती" negates the second clause, not the aspect.
_CLAUSE = re.compile(r"[,;:]|\b(?:but|lekin|however|and|aur)\b|लेकिन|परंतु|किंतु", re.I)


def _negated_aspect(sentence: str) -> bool:
    return any(_ASPECT.search(c) and NEGATION.search(c) for c in _CLAUSE.split(sentence))


def aspect_claims(sentence: str, index: int, facts: ChartFacts) -> list[Claim]:
    """"Shani 7th ko aspect karta hai" / "शनि की दृष्टि सातवें भाव पर": the engine
    computes no aspects (chart_facts.NOT_COMPUTED), so a planet-to-house
    aspect stated about this chart is unsupported — never verified, never
    "corrected" by a rule computed here. Generic statements are left alone."""
    if not facts.available or not _ASPECT.search(sentence) or _negated_aspect(sentence):
        return []
    mentions = _mentions(sentence)
    house = _ORDINAL_HOUSE.search(sentence)
    if not mentions or not house:
        return []
    if GENERIC.search(sentence) and not SECOND_PERSON.search(sentence):
        return []
    return [Claim("aspect", mentions[0][2], house.group(0), status="unsupported", sentence=sentence,
                  sentence_index=index, truth="aspects are not computed for this chart")]


__all__ = ["conjunction_claims", "aspect_claims"]
