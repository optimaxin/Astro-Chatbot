"""Coding cases: small Python / TypeScript / SQL tasks (25)."""

from __future__ import annotations

_RASHI_BUG = '''RASHIS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
          "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

def rashi_of(longitude: float) -> str:
    return RASHIS[int(longitude / 30) + 1]

print(rashi_of(10))   # expected Aries
print(rashi_of(345))  # crashes
'''

_MUTABLE_BUG = '''def add_reading(reading, history=[]):
    history.append(reading)
    return history

print(add_reading("Sun in Leo"))
print(add_reading("Moon in Pisces"))  # why does this print both readings?
'''

_ROTATE = '''ORDER = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]

def sequence_from(start, n):
    i = ORDER.index(start)
    return [ORDER[(i + k) % len(ORDER)] for k in range(n)]

print(sequence_from("Moon", 3))
'''

_SUM = '''YEARS = {"Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7,
         "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Mercury": 17}
print(sum(YEARS.values()))
'''

_DEDUP = '''def unique_users(events):
    seen = []
    for e in events:
        if e["user_id"] not in seen:
            seen.append(e["user_id"])
    return seen
'''

_SQL_BUG = '''SELECT model, COUNT(*) AS calls
FROM llm_calls
WHERE COUNT(*) > 100
GROUP BY model;
'''

_SCHEMA = ("Table llm_calls(id INTEGER PRIMARY KEY, model TEXT, user_id TEXT, created_at TIMESTAMP, "
           "latency_ms INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, success BOOLEAN)")


def _c(n: int, prompt: str, checks: dict, **extra) -> dict:
    return {"id": f"code-{n:03d}", "category": "coding", "prompt": prompt, **extra, "checks": checks}


CODING_CASES: list[dict] = [
    _c(1, "Write a Python function `longitude_to_rashi(longitude: float) -> str` that maps a sidereal longitude "
          "(0 to <360 degrees) to its rashi name using English names Aries ... Pisces, 30 degrees per sign. "
          "Handle values >= 360 or negative by normalising with modulo.",
       {"all": ["def longitude_to_rashi", "return", "Aries", "Pisces"], "any": ["% 360", "%360", "% 360.0"]},
       consistency=True),
    _c(2, "Write a Python function `parse_ddmmyyyy(s: str) -> datetime.date` that parses dates like '23-07-1998' "
          "and raises ValueError for invalid input.",
       {"all": ["def parse_ddmmyyyy"], "any": ["strptime", "split(", "ValueError"]}),
    _c(3, f"Given {_SCHEMA}, write a SQL query that returns the number of calls and total tokens "
          "(prompt + completion) per model, sorted by number of calls descending.",
       {"all": ["GROUP BY", "ORDER BY"], "any": ["COUNT(", "count("]}, consistency=True),
    _c(4, "This Python code has a bug. Explain it and give the fixed function.\n\n```python\n"
          + _RASHI_BUG + "```",
       {"any": ["off-by-one", "off by one", "+ 1", "+1", "IndexError", "out of range"],
        "regex": r"RASHIS\[\s*int\(\s*longitude\s*/\s*30\s*\)\s*\]|RASHIS\[\s*int\(\s*longitude\s*//\s*30\s*\)\s*\]"
                 r"|RASHIS\[\s*int\(\s*\(?\s*longitude\s*%\s*360\s*\)?\s*(//|/)\s*30\s*\)\s*\]|//\s*30"}),
    _c(5, "Write a TypeScript function `nakshatraIndex(longitude: number): number` returning the 0-based "
          "nakshatra index (27 nakshatras of 13°20' each) for a longitude in degrees.",
       {"all": ["nakshatraIndex"], "any": ["13.333", "40 / 3", "40/3", "360 / 27", "360/27", "13 + 1/3", "800"]}),
    _c(6, "What does this Python code print? Answer with the exact output.\n\n```python\n" + _ROTATE + "```",
       {"all": ["Moon", "Mars", "Rahu"]}),
    _c(7, "Write a Python regular expression and a small function `is_valid_time(s)` that accepts 24-hour times "
          "in HH:MM format (00:00 to 23:59) and rejects everything else.",
       {"all": ["def is_valid_time"], "any": ["[01]", "2[0-3]", "[0-5]"]}),
    _c(8, f"Given {_SCHEMA}, write a SQL query for the top 5 users by number of calls in the last 7 days.",
       {"all": ["GROUP BY", "LIMIT"], "any": ["ORDER BY", "order by"]}),
    _c(9, "Write a Python function `age_on(dob: date, on: date) -> int` that returns completed years of age, "
          "correctly handling birthdays later in the year.",
       {"all": ["def age_on", "return"], "any": ["month", "(on.month, on.day)", "<"]}),
    _c(10, "Write a minimal FastAPI endpoint `GET /horoscope/{rashi}` that validates `rashi` against the 12 "
           "English rashi names and returns JSON `{\"rashi\": ..., \"text\": ...}` with a placeholder text; "
           "return 404 for an unknown rashi.",
       {"any": ["@app.get", "@router.get", "APIRouter"], "all": ["404"]}),
    _c(11, "Write a TypeScript interface `Kundli` with fields: chartId (string), name (string), birthDate "
           "(ISO string), lagna (one of the 12 rashi names as a string-literal union), and planets (a map from "
           "planet name to {sign: string; house: number; degree: number}).",
       {"all": ["interface Kundli", "chartId"], "any": ["Record<", "[planet", "[key"]}),
    _c(12, "Why does the second call print both readings? Fix it.\n\n```python\n" + _MUTABLE_BUG + "```",
       {"any": ["mutable default", "default argument", "default list", "shared", "evaluated once"],
        "all": ["None"]}),
    _c(13, "Write a Python function `mahadasha_timeline(start_lord: str, start_year: float, n: int)` that returns "
           "a list of (lord, start_year, end_year) tuples for the next n Vimshottari mahadashas, using the order "
           "Ketu 7, Venus 20, Sun 6, Moon 10, Mars 7, Rahu 18, Jupiter 16, Saturn 19, Mercury 17.",
       {"all": ["def mahadasha_timeline", "Ketu", "Mercury"], "any": ["% 9", "%9", "% len", "cycle"]}),
    _c(14, "What is the time complexity of this function, and how would you make it faster?\n\n```python\n"
           + _DEDUP + "```",
       {"any": ["O(n^2)", "O(n²)", "O(n**2)", "quadratic", "O(n * m)", "O(n*m)"], "all": ["set"]}),
    _c(15, f"Given {_SCHEMA}, write a SQL query that returns, per model, the average latency in ms and the "
           "success rate as a percentage, only for calls created in 2026.",
       {"all": ["AVG", "GROUP BY"], "any": ["2026", "BETWEEN", ">="]}),
    _c(16, "Write pytest unit tests for a function `longitude_to_rashi(lon)` that returns 'Aries' for 0-30, "
           "'Taurus' for 30-60 and so on. Cover a boundary value and 359.9.",
       {"all": ["def test_", "assert"], "any": ["359.9", "Pisces"]}),
    _c(17, "An LLM returned tool-call arguments as a JSON string that may be malformed. Write a Python helper "
           "`parse_args(raw: str) -> dict` that returns {} on invalid JSON or when the JSON is not an object.",
       {"all": ["json.loads"], "any": ["JSONDecodeError", "except", "ValueError"]}),
    _c(18, "Write a TypeScript `debounce` function with generics that delays calling `fn` until `waitMs` "
           "milliseconds have passed since the last call.",
       {"all": ["setTimeout", "clearTimeout"]}),
    _c(19, "Write Python async code that sends the same prompt to three model clients concurrently and returns "
           "their results in a dict keyed by model name. Assume each client has `async def complete(prompt)`.",
       {"any": ["asyncio.gather", "TaskGroup", "as_completed"], "all": ["async def"]}),
    _c(20, "This SQL fails. Why, and what is the corrected query?\n\n```sql\n" + _SQL_BUG + "```",
       {"all": ["HAVING"]}, consistency=True),
    _c(21, "Explain what this regular expression matches, with one example that matches and one that does "
           "not: `^\\d{2}-\\d{2}-\\d{4}$`",
       {"any": ["DD-MM-YYYY", "dd-mm-yyyy", "two digits", "2 digits", "MM-DD-YYYY"]}),
    _c(22, "Write a Python function that converts a naive datetime in IST (Asia/Kolkata) to a timezone-aware "
           "UTC datetime using the standard library.",
       {"any": ["zoneinfo", "ZoneInfo", "timezone(timedelta"], "all": ["def "]}),
    _c(23, "Given a list of user rashi names like ['Leo', 'Aries', 'Leo', 'Pisces'], write Python that returns "
           "the counts of each rashi sorted from most to least common.",
       {"any": ["Counter", "most_common", "sorted("]}),
    _c(24, "What does this Python code print?\n\n```python\n" + _SUM + "```",
       {"regex": r"\b120\b"}),
    _c(25, "Write a Python function `tithi_number(sun_lon: float, moon_lon: float) -> int` returning 1-30, where "
           "tithi = floor(((moon - sun) mod 360) / 12) + 1.",
       {"all": ["def tithi_number", "12"], "any": ["% 360", "%360", "fmod", "mod"]}),
]
