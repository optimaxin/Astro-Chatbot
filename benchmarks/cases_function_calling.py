"""Function-calling cases (30): 22 must call exactly one tool, 8 must answer directly.

`expected_tool.args` lists only the arguments that must match (case-insensitive
string compare). Dates are always given unambiguously in the prompt.
"""

from __future__ import annotations

SYSTEM = ("You can call the provided tools. Call a tool only when live chart data, calendar data or an action "
          "(such as a booking) is needed; answer general astrology knowledge questions directly without tools. "
          "Today's date is 2026-11-12.")


# Tool-call cases are scored primarily via `expected_tool`. Their text check only guards against the model
# claiming it has no tool access (passes on an empty text reply). No-tool cases check the answer text.
TOOL_CASE_CHECKS = {"none": ["i don't have access", "i do not have access", "i can't access", "i cannot access",
                             "i don't have the ability", "unable to access"]}


def _f(n: int, prompt: str, tools: list[str], expected: dict | None, checks: dict | None = None, **extra) -> dict:
    return {"id": f"fn-{n:03d}", "category": "function_calling", "prompt": prompt, "system": SYSTEM,
            "tools": tools, "expected_tool": expected, **extra,
            "checks": checks or {k: list(v) for k, v in TOOL_CASE_CHECKS.items()}}


def _call(tool_name: str, /, **args) -> dict:
    return {"name": tool_name, "args": args}


FUNCTION_CALLING_CASES: list[dict] = [
    _f(1, "What's the panchang for Varanasi on 2026-03-14?",
       ["get_panchang", "get_daily_horoscope", "convert_timezone"],
       _call("get_panchang", date="2026-03-14", city="Varanasi"), consistency=True),
    _f(2, "Show me the Leo horoscope for 2026-11-13.",
       ["get_daily_horoscope", "get_panchang"],
       _call("get_daily_horoscope", rashi="Leo", date="2026-11-13")),
    _f(3, "Mesh rashi ka rashifal batao 2026-11-15 ke liye.",
       ["get_daily_horoscope", "get_panchang", "get_current_dasha"],
       _call("get_daily_horoscope", rashi="Aries", date="2026-11-15")),
    _f(4, "Where is Jupiter placed in chart KND-1007?",
       ["get_planet_placement", "get_current_dasha", "match_kundli"],
       _call("get_planet_placement", planet="Jupiter", chart_id="KND-1007"), consistency=True),
    _f(5, "Where is my Saturn placed?",
       ["get_planet_placement", "get_current_dasha"],
       _call("get_planet_placement", planet="Saturn", chart_id="KND-4821"),
       history=[
           {"role": "user", "content": "I just created my Kundli on the app."},
           {"role": "assistant", "content": "Great! Your chart is saved with chart_id KND-4821. "
                                            "What would you like to know about it?"},
       ]),
    _f(6, "Which dasha is running right now for chart KND-2290?",
       ["get_current_dasha", "get_planet_placement", "get_panchang"],
       _call("get_current_dasha", chart_id="KND-2290")),
    _f(7, "Please match the kundlis KND-1001 and KND-1002 for marriage.",
       ["match_kundli", "get_planet_placement", "calculate_kundli"],
       _call("match_kundli", person_a_chart_id="KND-1001", person_b_chart_id="KND-1002")),
    _f(8, "Convert 14:30 from Asia/Kolkata to America/New_York.",
       ["convert_timezone", "get_panchang"],
       _call("convert_timezone", time="14:30", from_tz="Asia/Kolkata", to_tz="America/New_York")),
    _f(9, "Book me a Vedic astrologer on 2026-11-20 at 6 PM.",
       ["book_consultation", "get_current_dasha", "get_daily_horoscope"],
       _call("book_consultation", astrologer_type="vedic", date="2026-11-20", slot="18:00")),
    _f(10, "Make a kundli for Rohit Sharma, born 1992-03-12 at 06:40 in Jaipur.",
       ["calculate_kundli", "get_planet_placement", "match_kundli"],
       _call("calculate_kundli", name="Rohit Sharma", birth_date="1992-03-12", birth_time="06:40",
             birth_place="Jaipur")),
    _f(11, "I need the Ujjain panchang for 2026-12-01, especially Rahu Kaal.",
       ["get_panchang", "convert_timezone", "book_consultation"],
       _call("get_panchang", date="2026-12-01", city="Ujjain")),
    _f(12, "What does today's horoscope say for Pisces?",
       ["get_daily_horoscope", "get_panchang"],
       _call("get_daily_horoscope", rashi="Pisces", date="2026-11-12")),
    _f(13, "Tell me the Moon's sign and house in chart KND-5531.",
       ["get_planet_placement", "get_current_dasha"],
       _call("get_planet_placement", planet="Moon", chart_id="KND-5531")),
    _f(14, "I'd like a tarot reading session on 2026-11-25 at 10:30.",
       ["book_consultation", "get_daily_horoscope"],
       _call("book_consultation", astrologer_type="tarot", date="2026-11-25", slot="10:30")),
    _f(15, "My cousin's puja starts at 09:00 Europe/London time. What is that in Asia/Kolkata?",
       ["convert_timezone", "get_panchang", "book_consultation"],
       _call("convert_timezone", time="09:00", from_tz="Europe/London", to_tz="Asia/Kolkata")),
    _f(16, "Can you check our compatibility now?",
       ["match_kundli", "calculate_kundli", "get_planet_placement"],
       _call("match_kundli", person_a_chart_id="KND-7710", person_b_chart_id="KND-7711"),
       history=[
           {"role": "user", "content": "My chart ID is KND-7710 and my fiancee's chart ID is KND-7711."},
           {"role": "assistant", "content": "Thanks, I have both chart IDs: KND-7710 (you) and KND-7711 "
                                            "(your fiancee)."},
       ]),
    _f(17, "What is my current mahadasha and antardasha?",
       ["get_current_dasha", "get_planet_placement", "get_panchang"],
       _call("get_current_dasha")),
    _f(18, "In which house is Rahu in my chart?",
       ["get_planet_placement", "get_current_dasha"],
       _call("get_planet_placement", planet="Rahu")),
    _f(19, "Give me the panchang for Kolkata on 14th March 2026.",
       ["get_panchang", "get_daily_horoscope"],
       _call("get_panchang", date="2026-03-14", city="Kolkata")),
    _f(20, "What's the Scorpio rashifal for 1 January 2027?",
       ["get_daily_horoscope", "get_panchang", "convert_timezone"],
       _call("get_daily_horoscope", rashi="Scorpio", date="2027-01-01")),
    _f(21, "Book a numerology consultation for 2026-12-05 at 11:00.",
       ["book_consultation", "calculate_kundli"],
       _call("book_consultation", astrologer_type="numerology", date="2026-12-05", slot="11:00"),
       consistency=True),
    _f(22, "Please create a chart for Meera Nair. She was born on 5 August 2001 at 11:15 PM in Kochi.",
       ["calculate_kundli", "match_kundli", "get_planet_placement", "get_current_dasha"],
       _call("calculate_kundli", name="Meera Nair", birth_date="2001-08-05", birth_time="23:15",
             birth_place="Kochi")),
    # ---- No tool should be called -------------------------------------------
    _f(23, "What does the 7th house represent in a birth chart?",
       ["get_planet_placement", "get_current_dasha", "get_panchang"], None,
       {"any": ["marriage", "partner", "spouse", "partnership"]}),
    _f(24, "Thanks a lot, that's all for today!",
       ["get_panchang", "book_consultation"], None,
       {"any": ["welcome", "glad", "happy", "pleasure", "take care", "anytime", "any time"]}),
    _f(25, "In general, what is Rahu in Vedic astrology?",
       ["get_planet_placement", "get_current_dasha"], None,
       {"any": ["node", "shadow", "chhaya", "north"]}),
    _f(26, "What's the difference between a panchang and a kundli?",
       ["get_panchang", "calculate_kundli", "get_daily_horoscope"], None,
       {"any": ["calendar", "almanac", "birth chart", "tithi"]}),
    _f(27, "Which planet rules the sign Pisces in Vedic astrology?",
       ["get_planet_placement", "get_daily_horoscope"], None,
       {"any": ["Jupiter", "Guru", "Brihaspati"]}),
    _f(28, "What is a tithi?",
       ["get_panchang", "convert_timezone"], None,
       {"any": ["lunar day", "12 degrees", "12°", "Moon", "30"]}),
    _f(29, "How does the 36-point Guna Milan scoring work in general? I don't have any charts saved yet.",
       ["match_kundli", "calculate_kundli"], None,
       {"any": ["36", "koota", "kuta", "nadi", "bhakoot"]}),
    _f(30, "Hi! What kinds of things can you help me with?",
       ["get_panchang", "get_daily_horoscope", "book_consultation", "get_current_dasha", "match_kundli"], None,
       {"any": ["panchang", "horoscope", "consultation", "kundli", "dasha", "match"]}),
]
