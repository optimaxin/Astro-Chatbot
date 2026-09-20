"""Kundli access, behind an interface the rest of the pipeline cannot see past.

A kundli is *structured, user-specific, exactly-retrievable* data, so it is
fetched by key — never searched for in a vector store. The Tredev backend
already computes charts from real Swiss-Ephemeris positions
(`backend/app/services/astrology/kundli.ts`, exposed as
`POST /api/calculators/kundli-full`) and persists only birth details, so this
module deliberately mirrors that shape rather than inventing its own.

Two providers implement the same `KundliProvider` protocol:

* `SampleFileKundliProvider` (default, `KUNDLI_SOURCE=sample`) reads
  `data/sample_kundli.json`, which was produced by *running the real backend
  calculators* — not hand-written — so terminal testing never exercises
  invented astrology.
* `TredevBackendKundliProvider` (`KUNDLI_SOURCE=backend`) POSTs birth details
  to the live backend and uses the chart it returns.

Swapping providers changes nothing anywhere else in the pipeline.
"""

from __future__ import annotations

import json
from datetime import date
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from app.config.settings import settings
from app.logging_setup import get_logger

log = get_logger("kundli")

# The 7 classical grahas plus the lunar nodes. Uranus/Neptune/Pluto are
# present in the backend's output but carry no standard Vedic reading, so
# they are excluded from the prompt context rather than invited to be
# over-interpreted by the model.
VEDIC_BODIES = ("sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn", "rahu", "ketu")

ORDINALS = {
    1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th",
    7: "7th", 8: "8th", 9: "9th", 10: "10th", 11: "11th", 12: "12th",
}

# What each house classically signifies — used to tell the model which
# houses matter for the *current* question instead of dumping all twelve.
HOUSE_THEMES: dict[str, tuple[int, ...]] = {
    "career": (1, 2, 6, 10, 11),
    "finance": (2, 5, 8, 9, 11),
    "love": (5, 7, 11),
    "relationship": (5, 7, 11),
    "marriage": (2, 7, 8, 12),
    "health": (1, 6, 8, 12),
    "personal_growth": (1, 4, 5, 9),
    "general": (1, 4, 7, 10),
}


def _active_period(periods: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The period whose [startsAt, endsAt) range contains today.

    The backend marks Mahadashas active but not Antardashas, so the sub-period
    is derived here from its own ISO dates rather than assumed.
    """
    today = date.today().isoformat()
    for period in periods:
        starts, ends = period.get("startsAt"), period.get("endsAt")
        if starts and ends and starts <= today < ends:
            return period
    return None


@dataclass
class KundliContext:
    """A chart reduced to exactly what a language model should be told."""

    user_id: str
    source: str
    available: bool
    birth: dict[str, Any] = field(default_factory=dict)
    ascendant: dict[str, Any] = field(default_factory=dict)
    planets: list[dict[str, Any]] = field(default_factory=list)
    moon_nakshatra: dict[str, Any] = field(default_factory=dict)
    current_mahadasha: dict[str, Any] = field(default_factory=dict)
    doshas: dict[str, Any] = field(default_factory=dict)
    yogas: list[dict[str, Any]] = field(default_factory=list)
    avakhada: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    # ── views ─────────────────────────────────────────────────────────
    def planet(self, name: str) -> dict[str, Any] | None:
        return next((p for p in self.planets if p.get("id") == name), None)

    def planets_in_houses(self, houses: tuple[int, ...]) -> list[dict[str, Any]]:
        return [p for p in self.planets if p.get("house") in houses]

    def to_prompt_block(self, intent: str = "general") -> str:
        """Compact, factual chart summary for the system prompt.

        Only what is actually in the chart goes in. Nothing here is phrased
        as an interpretation — interpretation is the model's job, grounded in
        the retrieved knowledge, not something smuggled in as "data".
        """
        if not self.available:
            return "USER KUNDLI: not available for this user. Do not state or guess any chart placement."

        lines: list[str] = ["USER KUNDLI (computed from real birth data — treat as fact):"]

        name = self.birth.get("name")
        place = self.birth.get("placeLabel")
        born = " ".join(str(x) for x in (self.birth.get("date"), self.birth.get("time")) if x)
        if born:
            lines.append(f"- Born: {born}{f' in {place}' if place else ''}{f' ({name})' if name else ''}")

        if self.ascendant.get("rashi"):
            lines.append(f"- Ascendant (Lagna): {self.ascendant['rashi']}")

        for body in VEDIC_BODIES:
            p = self.planet(body)
            if not p:
                continue
            house = ORDINALS.get(p.get("house", 0), str(p.get("house")))
            retro = " (retrograde)" if p.get("retrograde") else ""
            lines.append(
                f"- {body.capitalize()}: {p.get('rashi')}, {house} house, "
                f"nakshatra {p.get('nakshatra')} pada {p.get('nakshatraPada')}{retro}"
            )

        if self.moon_nakshatra.get("name"):
            lines.append(
                f"- Birth (Moon) nakshatra: {self.moon_nakshatra['name']} "
                f"pada {self.moon_nakshatra.get('pada')}, lord {self.moon_nakshatra.get('lord')}"
            )

        if self.current_mahadasha.get("lord"):
            md = self.current_mahadasha
            lines.append(
                f"- Current Vimshottari Mahadasha: {str(md['lord']).capitalize()} "
                f"({md.get('startsAt')} to {md.get('endsAt')})"
            )
            active_ad = _active_period(md.get("antardashas") or [])
            if active_ad:
                lines.append(
                    f"- Current Antardasha: {str(active_ad['lord']).capitalize()} "
                    f"({active_ad.get('startsAt')} to {active_ad.get('endsAt')})"
                )

        sade_sati = self.doshas.get("sadeSati") or {}
        if sade_sati.get("active"):
            lines.append(f"- Sade Sati: active, {sade_sati.get('phase')} phase")
        kaal_sarp = self.doshas.get("kaalSarp") or {}
        if kaal_sarp:
            lines.append(f"- Kaal Sarp Dosha: {'present' if kaal_sarp.get('isKaalSarp') else 'not present'}")

        present_yogas = [y.get("name") for y in self.yogas if y.get("present")]
        if present_yogas:
            lines.append(f"- Yogas present: {', '.join(present_yogas)}")

        houses = HOUSE_THEMES.get(intent, HOUSE_THEMES["general"])
        relevant = self.planets_in_houses(houses)
        house_list = ", ".join(ORDINALS.get(h, str(h)) for h in houses)
        if relevant:
            occupants = "; ".join(
                f"{p['id'].capitalize()} in {ORDINALS.get(p['house'], p['house'])}" for p in relevant
            )
            lines.append(f"- Houses that matter for a '{intent}' question ({house_list}): {occupants}")
        else:
            lines.append(f"- Houses that matter for a '{intent}' question ({house_list}): no planets occupy them")

        return "\n".join(lines)

    def fact_index(self) -> dict[str, set[str]]:
        """Ground-truth lookup used by the deterministic hallucination check.

        Maps each body to the set of sign/house strings that are actually
        true for this chart, so a claim like "your Moon is in Leo" can be
        contradicted without asking an LLM.
        """
        index: dict[str, set[str]] = {}
        for p in self.planets:
            body = str(p.get("id", "")).lower()
            if body not in VEDIC_BODIES:
                continue
            index[body] = {str(p.get("rashi", "")).lower(), str(p.get("house", ""))}
        if self.ascendant.get("rashi"):
            index["ascendant"] = {str(self.ascendant["rashi"]).lower()}
        return index

    def to_facts(self) -> dict[str, Any]:
        """Compact, JSON-safe chart facts for API clients and caching.

        This is the flat view of the chart — the same data `to_prompt_block`
        renders as prose, but keyed so a frontend can display it and so the
        whole thing round-trips through Redis as JSON. It is a *view*, not a
        second source of truth: every value is read straight off the fields
        the Kundli engine populated.

        Fields the engine does not compute are returned empty rather than
        filled with plausible values. `transits` is always `[]` today because
        nothing in this service computes transits; inventing them here is
        precisely the failure mode the rest of the pipeline guards against.
        """
        if not self.available:
            return {"available": False, "error": self.error}

        moon = self.planet("moon") or {}

        planets: dict[str, dict[str, Any]] = {}
        for body in VEDIC_BODIES:
            p = self.planet(body)
            if not p:
                continue
            planets[body.capitalize()] = {
                "house": p.get("house"),
                "sign": p.get("rashi"),
                "nakshatra": p.get("nakshatra"),
                "pada": p.get("nakshatraPada"),
                "retrograde": bool(p.get("retrograde")),
            }

        doshas: list[dict[str, Any]] = []
        sade_sati = self.doshas.get("sadeSati") or {}
        if sade_sati.get("active"):
            doshas.append({"name": "Sade Sati", "active": True, "phase": sade_sati.get("phase")})
        kaal_sarp = self.doshas.get("kaalSarp") or {}
        if kaal_sarp.get("isKaalSarp"):
            doshas.append({"name": "Kaal Sarp", "active": True, "type": kaal_sarp.get("type")})

        dashas: dict[str, Any] = {}
        if self.current_mahadasha.get("lord"):
            md = self.current_mahadasha
            dashas["mahadasha"] = {
                "lord": str(md["lord"]).capitalize(),
                "startsAt": md.get("startsAt"),
                "endsAt": md.get("endsAt"),
            }
            active_ad = _active_period(md.get("antardashas") or [])
            if active_ad:
                dashas["antardasha"] = {
                    "lord": str(active_ad["lord"]).capitalize(),
                    "startsAt": active_ad.get("startsAt"),
                    "endsAt": active_ad.get("endsAt"),
                }

        return {
            "available": True,
            "source": self.source,
            "lagna": self.ascendant.get("rashi"),
            "moonSign": moon.get("rashi"),
            "nakshatra": self.moon_nakshatra.get("name") or moon.get("nakshatra"),
            "nakshatraPada": self.moon_nakshatra.get("pada") or moon.get("nakshatraPada"),
            "planets": planets,
            "doshas": doshas,
            "dashas": dashas,
            # Not computed by this engine. Empty is the honest answer.
            "transits": [],
            "yogas": [y.get("name") for y in self.yogas if y.get("present")],
        }

    def facts_used(self) -> list[str]:
        """The chart facts that were actually available to the model this turn.

        Returned on the API response as `chart_facts_used` so a client — or a
        reviewer reading a log — can see exactly which ground truth the answer
        had access to, without shipping the whole chart.
        """
        if not self.available:
            return []
        facts = self.to_facts()
        used: list[str] = []
        if facts.get("lagna"):
            used.append(f"lagna={facts['lagna']}")
        if facts.get("moonSign"):
            used.append(f"moonSign={facts['moonSign']}")
        if facts.get("nakshatra"):
            used.append(f"nakshatra={facts['nakshatra']}")
        for name, p in (facts.get("planets") or {}).items():
            used.append(f"{name}={p.get('sign')}/H{p.get('house')}")
        mahadasha = (facts.get("dashas") or {}).get("mahadasha") or {}
        if mahadasha.get("lord"):
            used.append(f"mahadasha={mahadasha['lord']}")
        for dosha in facts.get("doshas") or []:
            used.append(f"dosha={dosha.get('name')}")
        return used

    @staticmethod
    def from_facts(user_id: str, facts: dict[str, Any]) -> "KundliContext | None":
        """Rebuild a context from a cached `to_facts()` payload.

        Only the fields `to_facts` actually carries are restored, so a cache
        hit can never smuggle in data the original chart did not have.
        """
        if not facts or not facts.get("available"):
            return None
        planets = [
            {
                "id": name.lower(),
                "house": p.get("house"),
                "rashi": p.get("sign"),
                "nakshatra": p.get("nakshatra"),
                "nakshatraPada": p.get("pada"),
                "retrograde": p.get("retrograde", False),
            }
            for name, p in (facts.get("planets") or {}).items()
        ]
        mahadasha = (facts.get("dashas") or {}).get("mahadasha") or {}
        antardasha = (facts.get("dashas") or {}).get("antardasha") or {}
        current_mahadasha: dict[str, Any] = {}
        if mahadasha.get("lord"):
            current_mahadasha = {
                "lord": mahadasha["lord"],
                "startsAt": mahadasha.get("startsAt"),
                "endsAt": mahadasha.get("endsAt"),
                "antardashas": [antardasha] if antardasha.get("lord") else [],
            }
        doshas: dict[str, Any] = {}
        for dosha in facts.get("doshas") or []:
            if dosha.get("name") == "Sade Sati":
                doshas["sadeSati"] = {"active": True, "phase": dosha.get("phase")}
            elif dosha.get("name") == "Kaal Sarp":
                doshas["kaalSarp"] = {"isKaalSarp": True, "type": dosha.get("type")}
        return KundliContext(
            user_id=user_id,
            source=f"{facts.get('source', 'unknown')}-cached",
            available=True,
            ascendant={"rashi": facts.get("lagna")} if facts.get("lagna") else {},
            planets=planets,
            moon_nakshatra=(
                {"name": facts["nakshatra"], "pada": facts.get("nakshatraPada")}
                if facts.get("nakshatra")
                else {}
            ),
            current_mahadasha=current_mahadasha,
            doshas=doshas,
            yogas=[{"name": y, "present": True} for y in facts.get("yogas") or []],
        )

    @staticmethod
    def unavailable(user_id: str, source: str, error: str) -> "KundliContext":
        return KundliContext(user_id=user_id, source=source, available=False, error=error)


def _build_context(user_id: str, source: str, payload: dict[str, Any]) -> KundliContext:
    """Normalise a `/kundli-full`-shaped payload into a `KundliContext`."""
    kundli = payload.get("kundli") or {}
    return KundliContext(
        user_id=user_id,
        source=source,
        available=bool(kundli.get("planets")),
        birth=payload.get("birth") or {},
        ascendant=kundli.get("ascendant") or {},
        planets=kundli.get("planets") or [],
        moon_nakshatra=kundli.get("moonNakshatra") or {},
        current_mahadasha=payload.get("currentMahadasha")
        or next((m for m in payload.get("mahadashaTimeline", []) if m.get("active")), {}),
        doshas=payload.get("doshas") or {},
        yogas=payload.get("yogas") or [],
        avakhada=payload.get("avakhada") or {},
    )


class KundliProvider(Protocol):
    name: str

    def get_kundli(self, user_id: str) -> KundliContext: ...


class SampleFileKundliProvider:
    """Terminal-testing source: a chart produced by the real backend calculators."""

    name = "sample-file"

    def __init__(self, path: Any | None = None) -> None:
        self.path = path or settings.kundli.sample_path

    @lru_cache(maxsize=8)  # noqa: B019 — provider is a process-lifetime singleton
    def _load(self) -> dict[str, Any]:
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def get_kundli(self, user_id: str) -> KundliContext:
        try:
            payload = self._load()
        except FileNotFoundError:
            return KundliContext.unavailable(user_id, self.name, f"{self.path} not found")
        except json.JSONDecodeError as exc:
            return KundliContext.unavailable(user_id, self.name, f"{self.path} is not valid JSON: {exc}")
        return _build_context(user_id, self.name, payload)


class TredevBackendKundliProvider:
    """Live source: POSTs birth details to the existing Tredev calculators API.

    Birth details are what the Tredev backend actually stores per user
    (`users.birth_*`, `kundli_history`); the chart itself is recomputed on
    demand. Mapping `user_id` -> birth details needs the authenticated user
    endpoint, which is out of scope for terminal testing, so birth details are
    read from the sample file while the *chart* comes from the live backend.
    """

    name = "tredev-backend"

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.kundli.backend_url).rstrip("/")
        self._fallback = SampleFileKundliProvider()

    def _birth_details(self, user_id: str) -> dict[str, Any] | None:
        context = self._fallback.get_kundli(user_id)
        birth = context.birth
        required = {"date", "time", "timezoneOffsetMinutes", "latitude", "longitude"}
        return birth if required.issubset(birth.keys()) else None

    def get_kundli(self, user_id: str) -> KundliContext:
        birth = self._birth_details(user_id)
        if birth is None:
            return KundliContext.unavailable(
                user_id, self.name, "no birth details available to request a chart with"
            )
        try:
            import httpx

            response = httpx.post(
                f"{self.base_url}/api/calculators/kundli-full",
                json={
                    "date": birth["date"],
                    "time": birth["time"],
                    "timezoneOffsetMinutes": birth["timezoneOffsetMinutes"],
                    "latitude": birth["latitude"],
                    "longitude": birth["longitude"],
                },
                timeout=20.0,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "backend kundli fetch failed, falling back to sample file",
                extra={"user_id": user_id, "error_type": type(exc).__name__, "error_message": str(exc)[:300]},
            )
            return self._fallback.get_kundli(user_id)

        if not body.get("success"):
            log.warning("backend returned an error payload", extra={"user_id": user_id})
            return self._fallback.get_kundli(user_id)

        payload = dict(body.get("data") or {})
        payload["birth"] = birth
        return _build_context(user_id, self.name, payload)


def kundli_from_birth(birth: dict[str, Any], user_id: str = "-") -> KundliContext:
    """Compute a chart from explicit birth details via the Tredev calculators.

    This is the path a *logged-out* web visitor takes: they have just filled in
    the birth form on the Kundli page, so there is no stored profile to look
    up, but there is no reason to trust a chart the browser computed either.
    The browser therefore sends only the birth details and the chart is
    recomputed here, by the same engine that drew the chart on the page.

    That is deliberate. Accepting a ready-made chart from a client would put
    attacker-controlled strings ("lagna", "nakshatra", …) directly into the
    system prompt, which is the one place this service must stay
    uncontaminated. Birth details are five bounded scalars and are validated
    at the API boundary before they reach this function.

    Never raises: a backend outage returns an unavailable context so the turn
    can still run and say plainly that the chart could not be loaded.
    """
    required = {"date", "time", "timezoneOffsetMinutes", "latitude", "longitude"}
    missing = required - birth.keys()
    if missing:
        return KundliContext.unavailable(
            user_id, "birth-details", f"missing birth details: {', '.join(sorted(missing))}"
        )

    base_url = settings.kundli.backend_url.rstrip("/")
    try:
        import httpx

        response = httpx.post(
            f"{base_url}/api/calculators/kundli-full",
            json={
                "date": birth["date"],
                "time": birth["time"],
                "timezoneOffsetMinutes": birth["timezoneOffsetMinutes"],
                "latitude": birth["latitude"],
                "longitude": birth["longitude"],
            },
            timeout=20.0,
        )
        response.raise_for_status()
        body = response.json()
    except Exception as exc:  # noqa: BLE001 — a calculator outage must not end the turn
        log.warning(
            "kundli computation failed",
            extra={"error_type": type(exc).__name__, "error_message": str(exc)[:300]},
        )
        return KundliContext.unavailable(
            user_id, "birth-details", f"could not reach the chart calculator: {type(exc).__name__}"
        )

    if not body.get("success"):
        log.warning("calculator returned an error payload")
        return KundliContext.unavailable(
            user_id, "birth-details", "the chart calculator rejected these birth details"
        )

    payload = dict(body.get("data") or {})
    payload["birth"] = birth
    context = _build_context(user_id, "birth-details", payload)
    log.info(
        "kundli computed from birth details",
        extra={"available": context.available, "planet_count": len(context.planets)},
    )
    return context


def get_provider() -> KundliProvider:
    if settings.kundli.source == "backend":
        return TredevBackendKundliProvider()
    return SampleFileKundliProvider()


_provider: KundliProvider | None = None


def get_kundli(user_id: str) -> KundliContext:
    """The one function the rest of the pipeline calls."""
    global _provider
    if _provider is None:
        _provider = get_provider()
    context = _provider.get_kundli(user_id)
    log.info(
        "kundli loaded",
        extra={
            "user_id": user_id,
            "source": context.source,
            "available": context.available,
            "planet_count": len(context.planets),
        },
    )
    return context


__all__ = [
    "KundliContext",
    "KundliProvider",
    "SampleFileKundliProvider",
    "TredevBackendKundliProvider",
    "get_kundli",
    "get_provider",
    "HOUSE_THEMES",
    "VEDIC_BODIES",
]
