"""Authoritative chart facts, structured for prompting and validation.

The Kundli engine (backend Swiss-Ephemeris calculators) is the source of
truth. This module only *reads* what the engine computed and adds facts that
follow from it by fixed rule, never by judgement:

* house signs and house lords — only when the engine's own house numbers
  prove the chart uses whole-sign houses (every planet's house equals its
  sign's distance from the Lagna sign). If they do not, no house table is
  derived, because the derivation would then be wrong.
* conjunctions — two grahas in the same sign (same house, whole-sign).

Everything else (aspects, transits, strength, yogas the engine did not
evaluate) is NOT produced here; the prompt lists it as unavailable so the
model says "I'd need to check" instead of inventing it.

Every fact carries its provenance: `engine` (computed by the calculators) or
`derived:<rule>` (computed here from engine output by the named rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SIGNS = ("Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra",
         "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces")
SIGN_LORDS = {
    "Aries": "mars", "Taurus": "venus", "Gemini": "mercury", "Cancer": "moon",
    "Leo": "sun", "Virgo": "mercury", "Libra": "venus", "Scorpio": "mars",
    "Sagittarius": "jupiter", "Capricorn": "saturn", "Aquarius": "saturn", "Pisces": "jupiter",
}
GRAHAS = ("sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn", "rahu", "ketu")
HOUSE_MEANINGS = {
    1: "self, body, personality", 2: "wealth, family, speech", 3: "courage, siblings, communication",
    4: "home, mother, property", 5: "children, education, romance", 6: "health, enemies, service",
    7: "marriage, partnership", 8: "longevity, transformation", 9: "luck, dharma, father",
    10: "career, status, karma", 11: "gains, income, friends", 12: "losses, expenses, foreign lands",
}
ORD = {n: f"{n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'}" for n in range(1, 13)}

# Which houses a question of each intent is classically read from.
_FOCUS_HOUSES = {
    "career": (1, 2, 6, 10, 11), "finance": (2, 5, 8, 9, 11), "love": (5, 7, 11),
    "relationship": (5, 7, 11), "marriage": (2, 7, 8, 12), "health": (1, 6, 8, 12),
    "personal_growth": (1, 4, 5, 9), "general": (1, 4, 7, 10),
}

# What the engine does not compute, stated to the model explicitly.
NOT_COMPUTED = ("planetary aspects (drishti)", "current transits (other than Sade Sati status)",
                "planetary strength (shadbala)", "Manglik/Mangal dosha", "divisional charts other than D9")


def _sign(value: Any) -> str | None:
    text = str(value or "").strip().capitalize()
    return text if text in SIGNS else None


@dataclass
class HouseFact:
    number: int
    sign: str
    lord: str
    lord_house: int | None
    occupants: list[str]
    provenance: str = "derived:whole-sign-from-engine-lagna"


@dataclass
class ChartFacts:
    available: bool
    lagna: str | None = None
    lagna_degree: float | None = None
    planets: dict[str, dict[str, Any]] = field(default_factory=dict)   # graha -> sign/house/...
    houses: dict[int, HouseFact] = field(default_factory=dict)
    whole_sign: bool = False
    conjunctions: list[tuple[str, ...]] = field(default_factory=list)
    moon_sign: str | None = None
    moon_nakshatra: str | None = None
    mahadasha: str | None = None
    antardasha: str | None = None
    yogas_present: set[str] = field(default_factory=set)
    yogas_absent: set[str] = field(default_factory=set)
    kaal_sarp: bool | None = None
    sade_sati: bool | None = None
    saturn_transit_sign: str | None = None      # engine: doshas.sadeSati.saturnTransitRashi
    navamsa: dict[str, str] = field(default_factory=dict)              # graha -> D9 sign
    # Dasha periods as the engine dated them (ISO strings). The model used to
    # be told the Antardasha lord without its dates and invented them ("Venus
    # Antardasha, January 2021 to January 2028" — production, 2026-11).
    mahadasha_period: tuple[str, str] | None = None
    antardasha_period: tuple[str, str] | None = None
    antardashas: list[tuple[str, str, str]] = field(default_factory=list)  # (lord, start, end), in order

    def house_of(self, graha: str) -> int | None:
        return (self.planets.get(graha) or {}).get("house")

    def sign_of(self, graha: str) -> str | None:
        return (self.planets.get(graha) or {}).get("sign")

    def lord_of_house(self, house: int) -> str | None:
        fact = self.houses.get(house)
        return fact.lord if fact else None


def build_chart_facts(kundli: Any) -> ChartFacts:
    """`kundli` is a KundliContext. Pure; never raises on partial data."""
    if not getattr(kundli, "available", False):
        return ChartFacts(available=False)

    facts = ChartFacts(available=True)
    facts.lagna = _sign((kundli.ascendant or {}).get("rashi"))
    degree = (kundli.ascendant or {}).get("degreeInSign")
    facts.lagna_degree = round(float(degree), 2) if isinstance(degree, (int, float)) else None

    for p in kundli.planets or []:
        graha = str(p.get("id", "")).lower()
        if graha not in GRAHAS:
            continue
        deg = p.get("degreeInSign")
        facts.planets[graha] = {
            "sign": _sign(p.get("rashi")),
            "house": p.get("house") if isinstance(p.get("house"), int) else None,
            "degree": round(float(deg), 2) if isinstance(deg, (int, float)) else None,
            "nakshatra": p.get("nakshatra"),
            "pada": p.get("nakshatraPada"),
            "retrograde": bool(p.get("retrograde")),
        }

    moon = facts.planets.get("moon") or {}
    facts.moon_sign = moon.get("sign")
    facts.moon_nakshatra = (kundli.moon_nakshatra or {}).get("name") or moon.get("nakshatra")

    # Whole-sign is proved from the engine's own numbers, never assumed.
    if facts.lagna and facts.planets:
        base = SIGNS.index(facts.lagna)
        checks = [(SIGNS.index(p["sign"]) - base) % 12 + 1 == p["house"]
                  for p in facts.planets.values() if p.get("sign") and p.get("house")]
        facts.whole_sign = bool(checks) and all(checks)
    if facts.whole_sign and facts.lagna:
        base = SIGNS.index(facts.lagna)
        for n in range(1, 13):
            sign = SIGNS[(base + n - 1) % 12]
            lord = SIGN_LORDS[sign]
            facts.houses[n] = HouseFact(
                number=n, sign=sign, lord=lord, lord_house=facts.house_of(lord),
                occupants=[g for g, p in facts.planets.items() if p.get("house") == n],
            )

    by_sign: dict[str, list[str]] = {}
    for graha, p in facts.planets.items():
        if p.get("sign"):
            by_sign.setdefault(p["sign"], []).append(graha)
    facts.conjunctions = [tuple(g) for g in by_sign.values() if len(g) > 1]

    md = kundli.current_mahadasha or {}
    if md.get("lord"):
        facts.mahadasha = str(md["lord"]).lower()
        from app.kundli.kundli_service import _active_period

        if md.get("startsAt") and md.get("endsAt"):
            facts.mahadasha_period = (str(md["startsAt"]), str(md["endsAt"]))
        facts.antardashas = [(str(a["lord"]).lower(), str(a["startsAt"]), str(a["endsAt"]))
                             for a in md.get("antardashas") or []
                             if a.get("lord") and a.get("startsAt") and a.get("endsAt")]
        ad = _active_period(md.get("antardashas") or [])
        if ad and ad.get("lord"):
            facts.antardasha = str(ad["lord"]).lower()
            if ad.get("startsAt") and ad.get("endsAt"):
                facts.antardasha_period = (str(ad["startsAt"]), str(ad["endsAt"]))

    for yoga in kundli.yogas or []:
        name = str(yoga.get("name", "")).strip()
        if name:
            (facts.yogas_present if yoga.get("present") else facts.yogas_absent).add(name.lower())
    kaal = (kundli.doshas or {}).get("kaalSarp")
    if isinstance(kaal, dict) and "isKaalSarp" in kaal:
        facts.kaal_sarp = bool(kaal.get("isKaalSarp"))
    sade = (kundli.doshas or {}).get("sadeSati")
    if isinstance(sade, dict) and "active" in sade:
        facts.sade_sati = bool(sade.get("active"))
    if isinstance(sade, dict):
        facts.saturn_transit_sign = _sign(sade.get("saturnTransitRashi"))

    for p in (getattr(kundli, "navamsa", None) or {}).get("planets") or []:
        graha, sign = str(p.get("id", "")).lower(), _sign(p.get("rashi"))
        if graha in GRAHAS and sign:
            facts.navamsa[graha] = sign
    return facts


_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December")


def _month(iso: Any) -> str:
    """"2026-05-20" -> "May 2026". Unparseable input is returned unchanged."""
    text = str(iso or "")
    try:
        year, month = int(text[:4]), int(text[5:7])
        return f"{_MONTHS[month - 1]} {year}"
    except (ValueError, IndexError):
        return text or "?"


def render_prompt_block(kundli: Any, intent: str = "general") -> str:
    """The chart as the model sees it: facts first, labelled, no interpretation."""
    facts = build_chart_facts(kundli)
    if not facts.available:
        return ("USER KUNDLI: not available for this user. Do not state or guess any chart "
                "placement, dasha, yoga or dosha; offer to read the chart once birth details are given.")

    lines = ["USER KUNDLI — computed by the Tredev Kundli engine (Swiss Ephemeris) from real birth data; "
             "treat as fact. Quote it exactly; never move a planet, change a house or sign, or add one "
             "that is not listed."]
    birth = kundli.birth or {}
    born = " ".join(str(x) for x in (birth.get("date"), birth.get("time")) if x)
    if born:
        place = birth.get("placeLabel")
        lines.append(f"Birth: {born}{f', {place}' if place else ''}")
    if facts.lagna:
        deg = f" {facts.lagna_degree}°" if facts.lagna_degree is not None else ""
        lines.append(f"Lagna (Ascendant, = 1st house): {facts.lagna}{deg}")
    if facts.moon_sign:
        lines.append(f"Moon sign (Rashi): {facts.moon_sign}; birth Nakshatra: {facts.moon_nakshatra or 'n/a'}")

    lines.append("PLANETS — graha: sign (rashi) | house (bhava) | degree | nakshatra-pada:")
    for graha in GRAHAS:
        p = facts.planets.get(graha)
        if not p:
            continue
        house = ORD.get(p["house"], "?") if p.get("house") else "?"
        deg = f"{p['degree']}°" if p.get("degree") is not None else "n/a"
        nak = f"{p['nakshatra']}-{p['pada']}" if p.get("nakshatra") else "n/a"
        retro = " | retrograde" if p.get("retrograde") else ""
        lines.append(f"- {graha.capitalize()}: {p.get('sign')} | {house} house | {deg} | {nak}{retro}")

    if facts.houses:
        lines.append("HOUSES (whole-sign, derived from the Lagna) — house: sign, lord (where the lord sits), occupants:")
        for n, h in facts.houses.items():
            occ = ", ".join(o.capitalize() for o in h.occupants) or "none"
            lord_at = ORD.get(h.lord_house, "?") if h.lord_house else "?"
            lines.append(f"- {ORD[n]} ({HOUSE_MEANINGS[n]}): {h.sign}, lord {h.lord.capitalize()} "
                         f"(in {lord_at}); occupants: {occ}")
    focus = _FOCUS_HOUSES.get(intent, _FOCUS_HOUSES["general"])
    lines.append(f"Houses that matter for a '{intent}' question: " + ", ".join(ORD[h] for h in focus))
    if facts.conjunctions:
        lines.append("Conjunctions (same sign): " + "; ".join(
            " + ".join(g.capitalize() for g in c) for c in facts.conjunctions))

    if facts.mahadasha:
        md = kundli.current_mahadasha or {}
        line = (f"Current Vimshottari Mahadasha: {facts.mahadasha.capitalize()} "
                f"({_month(md.get('startsAt'))} to {_month(md.get('endsAt'))})")
        if facts.antardasha:
            span = (f" ({_month(facts.antardasha_period[0])} to {_month(facts.antardasha_period[1])})"
                    if facts.antardasha_period else "")
            line += f"; current Antardasha: {facts.antardasha.capitalize()}{span}"
        lines.append(line)
        if facts.antardashas:
            lines.append(f"Antardasha sequence within the {facts.mahadasha.capitalize()} Mahadasha "
                         "(use these exact dates for any timing; no other periods are known): " + "; ".join(
                             f"{lord.capitalize()} {_month(start)} to {_month(end)}"
                             for lord, start, end in facts.antardashas))
    status = []
    if facts.sade_sati is not None:
        phase = ((kundli.doshas or {}).get("sadeSati") or {}).get("phase")
        status.append(f"Sade Sati {'active' + (f', {phase} phase' if phase else '') if facts.sade_sati else 'not active'}")
    if facts.kaal_sarp is not None:
        status.append(f"Kaal Sarp Dosha {'present' if facts.kaal_sarp else 'not present'}")
    if facts.yogas_present:
        status.append("Yogas present: " + ", ".join(sorted(y.title() for y in facts.yogas_present)))
    if facts.yogas_absent:
        status.append("Yogas checked and NOT present: " + ", ".join(sorted(y.title() for y in facts.yogas_absent)))
    if status:
        lines.append("; ".join(status))
    if facts.navamsa and intent in {"marriage", "relationship", "love"}:
        lines.append("Navamsa (D9) signs: " + ", ".join(
            f"{g.capitalize()} {s}" for g, s in facts.navamsa.items()))

    lines.append("NOT computed for this chart (say you'd need to check; never guess): "
                 + ", ".join(NOT_COMPUTED) + ".")
    return "\n".join(lines)


__all__ = ["ChartFacts", "HouseFact", "build_chart_facts", "render_prompt_block",
           "SIGNS", "SIGN_LORDS", "GRAHAS", "HOUSE_MEANINGS", "ORD"]
