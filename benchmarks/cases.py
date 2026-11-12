"""Benchmark dataset for comparing LLMs on the Tredev Astro Kundli chatbot.

Exports:
    TOOLS  -- OpenAI function-calling tool definitions keyed by tool name.
    CASES  -- 220 deterministic benchmark cases across seven categories.

The case data lives in ``cases_<category>.py`` modules next to this file and is
concatenated here in a fixed order. Nothing here is random; importing the module
always yields the identical dataset.
"""

from __future__ import annotations

import re
from collections import Counter

from benchmarks.cases_coding import CODING_CASES
from benchmarks.cases_function_calling import FUNCTION_CALLING_CASES
from benchmarks.cases_general import GENERAL_CASES
from benchmarks.cases_image import IMAGE_CASES
from benchmarks.cases_multilingual import MULTILINGUAL_CASES
from benchmarks.cases_rag import RAG_CASES
from benchmarks.cases_reasoning import REASONING_CASES


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
RASHIS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
ASTROLOGER_TYPES = ["vedic", "kp", "numerology", "tarot", "vastu"]

_DATE = {"type": "string", "description": "Calendar date in YYYY-MM-DD format, e.g. 2026-03-14."}
_CHART_ID = {
    "type": "string",
    "description": "Saved chart identifier such as 'KND-1007'. Omit to use the signed-in user's own default chart.",
}

TOOLS: dict[str, dict] = {
    "get_planet_placement": _tool(
        "get_planet_placement",
        "Look up the sign, house and degree of one planet in a saved birth chart (Kundli).",
        {
            "planet": {"type": "string", "enum": PLANETS, "description": "The graha to look up."},
            "chart_id": _CHART_ID,
        },
        ["planet"],
    ),
    "get_panchang": _tool(
        "get_panchang",
        "Get the Hindu Panchang (tithi, vara, nakshatra, yoga, karana, sunrise, sunset, Rahu Kaal) "
        "for a given date and city.",
        {
            "date": _DATE,
            "city": {"type": "string", "description": "City name, e.g. 'Varanasi'."},
        },
        ["date", "city"],
    ),
    "get_current_dasha": _tool(
        "get_current_dasha",
        "Get the currently running Vimshottari Mahadasha and Antardasha, with start and end dates, "
        "for a saved birth chart.",
        {"chart_id": _CHART_ID},
        [],
    ),
    "match_kundli": _tool(
        "match_kundli",
        "Run Ashtakoota Guna Milan (36-point) compatibility matching between two saved birth charts.",
        {
            "person_a_chart_id": {"type": "string", "description": "Chart ID of the first person, e.g. 'KND-1001'."},
            "person_b_chart_id": {"type": "string", "description": "Chart ID of the second person, e.g. 'KND-1002'."},
        },
        ["person_a_chart_id", "person_b_chart_id"],
    ),
    "convert_timezone": _tool(
        "convert_timezone",
        "Convert a clock time from one IANA time zone to another (today's date is assumed).",
        {
            "time": {"type": "string", "description": "24-hour clock time in HH:MM format, e.g. 14:30."},
            "from_tz": {"type": "string", "description": "Source IANA time zone, e.g. 'Asia/Kolkata'."},
            "to_tz": {"type": "string", "description": "Target IANA time zone, e.g. 'Europe/London'."},
        },
        ["time", "from_tz", "to_tz"],
    ),
    "book_consultation": _tool(
        "book_consultation",
        "Book a paid one-on-one consultation with a Tredev Astro astrologer.",
        {
            "astrologer_type": {
                "type": "string",
                "enum": ASTROLOGER_TYPES,
                "description": "Kind of astrologer to book.",
            },
            "date": _DATE,
            "slot": {"type": "string", "description": "Start time of the slot in 24-hour HH:MM format, IST, e.g. 18:00."},
        },
        ["astrologer_type", "date", "slot"],
    ),
    "get_daily_horoscope": _tool(
        "get_daily_horoscope",
        "Get the daily horoscope (rashifal) for a Moon sign (rashi) on a given date.",
        {
            "rashi": {
                "type": "string",
                "enum": RASHIS,
                "description": "Moon sign in English, e.g. 'Aries' for Mesha, 'Pisces' for Meena.",
            },
            "date": _DATE,
        },
        ["rashi", "date"],
    ),
    "calculate_kundli": _tool(
        "calculate_kundli",
        "Create and save a new birth chart (Kundli) from birth details. Returns a chart_id.",
        {
            "name": {"type": "string", "description": "Full name of the person."},
            "birth_date": _DATE,
            "birth_time": {"type": "string", "description": "Local birth time in 24-hour HH:MM format, e.g. 23:15."},
            "birth_place": {"type": "string", "description": "Birth city, e.g. 'Jaipur'."},
        },
        ["name", "birth_date", "birth_time", "birth_place"],
    ),
}


CASES: list[dict] = [
    *GENERAL_CASES,
    *RAG_CASES,
    *REASONING_CASES,
    *CODING_CASES,
    *MULTILINGUAL_CASES,
    *FUNCTION_CALLING_CASES,
    *IMAGE_CASES,
]


# --------------------------------------------------------------------------- #
# Integrity checks (run at import so a broken dataset never reaches the harness)
# --------------------------------------------------------------------------- #

EXPECTED_COUNTS = {
    "general": 35,
    "rag": 40,
    "reasoning": 30,
    "coding": 25,
    "multilingual": 35,
    "function_calling": 30,
    "image": 25,
}
_PREFIX = {
    "general": "gen", "rag": "rag", "reasoning": "rsn", "coding": "code",
    "multilingual": "ml", "function_calling": "fn", "image": "img",
}
_ALLOWED_KEYS = {
    "id", "category", "prompt", "system", "history", "context", "tools",
    "expected_tool", "image", "checks", "consistency",
}
_ALLOWED_CHECKS = {"any", "all", "none", "regex", "answerable", "cite", "language"}

assert len(CASES) == 220, len(CASES)
assert len({c["id"] for c in CASES}) == len(CASES), "duplicate case ids"
assert dict(Counter(c["category"] for c in CASES)) == EXPECTED_COUNTS, Counter(c["category"] for c in CASES)
for _c in CASES:
    assert set(_c) <= _ALLOWED_KEYS, (_c["id"], set(_c) - _ALLOWED_KEYS)
    assert set(_c["checks"]) <= _ALLOWED_CHECKS, (_c["id"], set(_c["checks"]) - _ALLOWED_CHECKS)
    assert re.fullmatch(_PREFIX[_c["category"]] + r"-\d{3}", _c["id"]), _c["id"]
    if "regex" in _c["checks"]:
        re.compile(_c["checks"]["regex"], re.I)
    if _c["category"] == "function_calling":
        assert all(t in TOOLS for t in _c["tools"]), _c["id"]
        if _c["expected_tool"] is not None:
            assert _c["expected_tool"]["name"] in _c["tools"], _c["id"]
del _c
