"""Reasoning cases with a single verifiable answer (30).

Every prompt asks the model to finish with a line "Answer: <value>"; the regex
anchors on that line (tolerating markdown bold / punctuation after "Answer").
All expected values were computed and double-checked by hand and with Python.
"""

from __future__ import annotations

_SIGNS_NOTE = ("Use the sidereal zodiac where each sign spans exactly 30 degrees starting from Aries at 0 degrees: "
               "Aries, Taurus, Gemini, Cancer, Leo, Virgo, Libra, Scorpio, Sagittarius, Capricorn, Aquarius, Pisces.")
_VIM = ("Vimshottari Mahadasha order and lengths: Ketu 7, Venus 20, Sun 6, Moon 10, Mars 7, Rahu 18, "
        "Jupiter 16, Saturn 19, Mercury 17 years (then the cycle repeats).")
_END = " Show brief working, then end with a final line in the form 'Answer: {fmt}'."


def _ans(pattern: str) -> str:
    """Regex for the final 'Answer: ...' line, tolerant of bold markers and spacing."""
    return r"answer\W{0,6}\s*(?:" + pattern + ")"


def _q(n: int, prompt: str, fmt: str, pattern: str, **extra) -> dict:
    return {"id": f"rsn-{n:03d}", "category": "reasoning", "prompt": prompt + _END.format(fmt=fmt),
            **extra, "checks": {"regex": _ans(pattern)}}


REASONING_CASES: list[dict] = [
    _q(1, f"{_SIGNS_NOTE} A planet is at an absolute longitude of 247 degrees. Which sign is it in, and at what "
          "degree within that sign?", "<Sign> <degree>", r"Sagittarius\W*7\b", consistency=True),
    _q(2, f"{_SIGNS_NOTE} The Moon is at 95.5 degrees absolute longitude. Which sign and degree within the sign?",
       "<Sign> <degree>", r"Cancer\W*(5\.5|5\s*°?\s*30)"),
    _q(3, f"{_SIGNS_NOTE} Saturn is at 329 degrees. Which sign and degree within the sign?",
       "<Sign> <degree>", r"Aquarius\W*29\b"),
    _q(4, f"{_SIGNS_NOTE} Venus is at 183 degrees. Which sign and degree within the sign?",
       "<Sign> <degree>", r"Libra\W*3\b"),
    _q(5, "The 27 nakshatras each span 13 degrees 20 minutes, starting at 0 degrees with Ashwini, then Bharani, "
          "Krittika, Rohini, Mrigashira, Ardra, Punarvasu, Pushya, Ashlesha, and so on. The Moon is at exactly "
          "100 degrees. Which nakshatra is it in?", "<nakshatra>", r"Pushya"),
    _q(6, f"{_VIM} A person was born at the exact start of their Venus Mahadasha on 1 January 1990. Which "
          "Mahadasha will be running on 1 January 2030?", "<planet>", r"Mars\b", consistency=True),
    _q(7, f"{_VIM} What is the total length of one complete Vimshottari cycle in years?",
       "<number>", r"120\b"),
    _q(8, f"{_VIM} Priya was born on 1 January 2000 with exactly 4 years of her Moon Mahadasha remaining. "
          "How old will she be (in completed years) when her Jupiter Mahadasha begins?",
       "<age>", r"29\b"),
    _q(9, f"{_VIM} An Antardasha's length = (Mahadasha years x Antardasha planet years) / 120. How many months "
          "long is the Venus Antardasha within the Venus Mahadasha?", "<months>", r"40\b"),
    _q(10, f"{_VIM} An Antardasha's length = (Mahadasha years x Antardasha planet years) / 120. How many months "
           "long is the Moon Antardasha within the Sun Mahadasha?", "<months>", r"6\b"),
    _q(11, f"{_VIM} An Antardasha's length = (Mahadasha years x Antardasha planet years) / 120. How many years "
           "long is the Rahu Antardasha within the Rahu Mahadasha? Give a decimal.", "<years>", r"2\.7\b"),
    _q(12, "India Standard Time is UTC+5:30. A user books a reading at 10:30 IST. What is that time in UTC "
           "(24-hour HH:MM)?", "<HH:MM>", r"0?5:00\b"),
    _q(13, "India Standard Time is UTC+5:30 and London in January is on GMT (UTC+0). A live session starts at "
           "23:45 IST. What time is that in London (24-hour HH:MM)?", "<HH:MM>", r"18:15\b"),
    _q(14, "India Standard Time is UTC+5:30. Assume New York is at UTC-4. A webinar starts at 09:00 IST. What "
           "clock time is it in New York at that moment (24-hour HH:MM)?", "<HH:MM>", r"23:30\b"),
    _q(15, "Someone was born on 15 August 1994. How old are they, in completed years, on 1 March 2026?",
       "<age>", r"31\b"),
    _q(16, "How many days are there from 14 January 2026 to 4 March 2026? Count 14 Jan as day 0 (i.e. the "
           "difference between the two dates).", "<days>", r"49\b"),
    _q(17, "1 January 2026 was a Thursday. What day of the week is 26 January 2026?",
       "<weekday>", r"Monday"),
    _q(18, "Saturn spends about 2.5 years in each sign. Sade Sati lasts while Saturn transits the 12th, 1st and "
           "2nd signs from the natal Moon. How many years does Sade Sati last in total?",
       "<years>", r"7\.5\b"),
    _q(19, "Counting signs inclusively (the sign itself is the 1st), which sign is 7th from Taurus? Order: Aries, "
           "Taurus, Gemini, Cancer, Leo, Virgo, Libra, Scorpio, Sagittarius, Capricorn, Aquarius, Pisces.",
       "<sign>", r"Scorpio"),
    _q(20, "In whole-sign houses the lagna sign is house 1, the next sign is house 2, and so on. The lagna is "
           "Capricorn and Mars is in Aries. Which house is Mars in?", "<house number>", r"4\b|4th\b"),
    _q(21, "Rule for this question: a chart has Mangal Dosha if Mars is in house 1, 2, 4, 7, 8 or 12 counted from "
           "the lagna (whole-sign houses, lagna = house 1). The lagna is Virgo and Mars is in Sagittarius. Does "
           "the chart have Mangal Dosha by this rule?", "Yes or No", r"Yes\b"),
    _q(22, "Rohan, Priya and Aman each had one consultation on Monday, Tuesday and Wednesday, one person per day. "
           "Priya did not go on Monday. Aman went on the day right after Priya. Who went on Monday?",
       "<name>", r"Rohan"),
    _q(23, "Tithi number = floor((Moon longitude - Sun longitude) / 12) + 1, using the difference modulo 360. "
           "The Moon is at 130 degrees and the Sun is at 40 degrees. What is the tithi number?",
       "<number>", r"8\b"),
    _q(24, "A consultation costs Rs 499 before tax and 18% GST is added. What is the final amount in rupees, "
           "to two decimal places?", "<amount>", r"(Rs\.?\s*|₹\s*|INR\s*)?588\.82\b"),
    _q(25, "Tredev Plus costs Rs 299 per month or Rs 2,999 per year. How many rupees does a user save over 12 "
           "months by choosing the yearly plan?", "<amount>", r"(Rs\.?\s*|₹\s*|INR\s*)?589\b"),
    _q(26, f"{_SIGNS_NOTE} A planet's tropical longitude is 100 degrees and the ayanamsa is 24 degrees. "
           "Sidereal longitude = tropical - ayanamsa. Which sidereal sign is the planet in?",
       "<sign>", r"Gemini", consistency=True),
    _q(27, f"{_SIGNS_NOTE} Ketu is always exactly 180 degrees from Rahu. Rahu is at 15 degrees Gemini. Which sign "
           "is Ketu in, and at what degree?", "<Sign> <degree>", r"Sagittarius\W*15\b"),
    _q(28, "There are two Ekadashis in every lunar month. How many Ekadashis occur in a lunar year of exactly "
           "12 lunar months?", "<number>", r"24\b"),
    _q(29, "Rule: divide the daytime (sunrise to sunset) into 8 equal parts; on Mondays Rahu Kaal is the 2nd part. "
           "On a Monday, sunrise is 06:00 and sunset is 18:00. At what time does Rahu Kaal start (24-hour HH:MM)?",
       "<HH:MM>", r"0?7:30\b"),
    _q(30, "A nakshatra spans 13 degrees 20 minutes. If the Moon moves at a constant 12 degrees per day, how long "
           "does it take to cross one full nakshatra? Give hours and minutes.",
       "<H> hours <M> minutes", r"26\s*(hours|hrs?|h)\W*(and\s*)?40\s*(minutes|mins?|m)\b"),
]
