"""Image-understanding cases (25).

Images are rendered by the harness with Pillow from the `image` spec (ASCII text,
simple tables, or a left-to-right row of coloured shapes). Questions can only be
answered by reading the image.
"""

from __future__ import annotations

_END = " End your reply with a final line in the form 'Answer: {fmt}'."


def _ans(pattern: str) -> str:
    return r"answer\W{0,6}\s*(?:" + pattern + ")"


def _i(n: int, prompt: str, image: dict, checks: dict, **extra) -> dict:
    return {"id": f"img-{n:03d}", "category": "image", "prompt": prompt, "image": image, **extra,
            "checks": checks}


def _s(*pairs: str) -> dict:
    """Shapes spec from 'color type' strings, e.g. _s('red circle', 'blue square')."""
    return {"kind": "shapes",
            "shapes": [{"type": p.split()[1], "color": p.split()[0]} for p in pairs]}


PLANET_TABLE = {
    "kind": "table",
    "header": ["Planet", "House", "Sign"],
    "rows": [
        ["Sun", "10", "Taurus"], ["Moon", "4", "Scorpio"], ["Mars", "7", "Aquarius"],
        ["Mercury", "9", "Aries"], ["Jupiter", "2", "Virgo"], ["Venus", "11", "Gemini"],
        ["Saturn", "11", "Gemini"], ["Rahu", "5", "Capricorn"], ["Ketu", "11", "Cancer"],
    ],
}
BIRTH_CARD = {
    "kind": "text_card",
    "lines": ["TREDEV ASTRO - BIRTH DETAILS", "Name: Kavya Rao", "Date of birth: 23-07-1998",
              "Time of birth: 04:17 AM", "Place of birth: Mysuru, Karnataka", "Chart ID: KND-6604"],
}
PANCHANG_TABLE = {
    "kind": "table",
    "header": ["Date", "Tithi", "Nakshatra", "Sunrise"],
    "rows": [
        ["2026-03-14", "Krishna Ekadashi", "Uttara Ashadha", "06:21"],
        ["2026-03-15", "Krishna Dwadashi", "Shravana", "06:20"],
        ["2026-03-16", "Krishna Trayodashi", "Dhanishta", "06:19"],
    ],
}
DASHA_TABLE = {
    "kind": "table",
    "header": ["Mahadasha", "Start", "End"],
    "rows": [["Moon", "2003-05-11", "2013-05-11"], ["Mars", "2013-05-11", "2020-05-11"],
             ["Rahu", "2020-05-11", "2038-05-11"], ["Jupiter", "2038-05-11", "2054-05-11"]],
}
KOOTA_TABLE = {
    "kind": "table",
    "header": ["Koota", "Max", "Score"],
    "rows": [["Varna", "1", "1"], ["Vashya", "2", "2"], ["Tara", "3", "3"], ["Yoni", "4", "4"],
             ["Graha Maitri", "5", "5"], ["Gana", "6", "6"], ["Bhakoot", "7", "0"], ["Nadi", "8", "8"]],
}

IMAGE_CASES: list[dict] = [
    _i(1, "This table is from a user's birth chart. Which house is Saturn in?" + _END.format(fmt="<house number>"),
       PLANET_TABLE, {"regex": _ans(r"11\b|11th|eleven")}, consistency=True),
    _i(2, "According to this table, which planet is in the 7th house?" + _END.format(fmt="<planet>"),
       PLANET_TABLE, {"regex": _ans(r"Mars")}),
    _i(3, "How many red circles are in this image?" + _END.format(fmt="<number>"),
       _s("red circle", "blue square", "red circle", "green triangle", "red square"),
       {"regex": _ans(r"2\b|two")}, consistency=True),
    _i(4, "How many shapes are there in total in this image?" + _END.format(fmt="<number>"),
       _s("yellow circle", "purple square", "orange triangle", "black circle", "blue triangle", "green square"),
       {"regex": _ans(r"6\b|six")}),
    _i(5, "What colour is the triangle in this image?" + _END.format(fmt="<colour>"),
       _s("blue circle", "green triangle", "red square"),
       {"regex": _ans(r"green")}),
    _i(6, "What is the time of birth written on this card?",
       BIRTH_CARD, {"any": ["04:17", "4:17"]}),
    _i(7, "Which city was this person born in, according to the card?",
       BIRTH_CARD, {"any": ["Mysuru", "Mysore"]}),
    _i(8, "From this panchang table, what is the nakshatra on 2026-03-15?" + _END.format(fmt="<nakshatra>"),
       PANCHANG_TABLE, {"regex": _ans(r"Shravana")}),
    _i(9, "From this dasha table, on what date does the Rahu Mahadasha end?",
       DASHA_TABLE, {"any": ["2038-05-11", "11 May 2038", "May 11, 2038", "11-05-2038", "11/05/2038"]}),
    _i(10, "What is the amount paid on this receipt?",
       {"kind": "text_card",
        "lines": ["TREDEV ASTRO - PAYMENT RECEIPT", "Order ID: TRD-88231", "Service: Kundli Matching Report",
                  "Amount paid: INR 749", "Status: SUCCESS"]},
       {"regex": r"749"}),
    _i(11, "How many blue squares are in this image?" + _END.format(fmt="<number>"),
       _s("blue circle", "red square", "blue triangle", "green square"),
       {"regex": _ans(r"0\b|zero|none|no blue")}),
    _i(12, "Counting from the left, what is the third shape in this image (colour and shape)?"
       + _END.format(fmt="<colour> <shape>"),
       _s("red square", "yellow circle", "purple triangle", "blue circle"),
       {"regex": _ans(r"purple\W+triangle")}),
    _i(13, "This is a Guna Milan breakdown. What is the total score (sum of the Score column)?"
       + _END.format(fmt="<number>"),
       KOOTA_TABLE, {"regex": _ans(r"29\b")}, consistency=True),
    _i(14, "What lucky number is shown on this card?",
       {"kind": "text_card",
        "lines": ["DAILY GUIDANCE - 12 NOV 2026", "Rashi: Vrishabha (Taurus)", "Lucky colour: Green",
                  "Lucky number: 7", "Mood: Calm and focused"]},
       {"regex": r"\b(7|seven)\b"}),
    _i(15, "Which rashi has the highest Career score in this table?" + _END.format(fmt="<rashi>"),
       {"kind": "table", "header": ["Rashi", "Love", "Career", "Health"],
        "rows": [["Aries", "7", "6", "8"], ["Cancer", "9", "5", "6"], ["Virgo", "6", "9", "7"],
                 ["Capricorn", "5", "8", "9"]]},
       {"regex": _ans(r"Virgo")}),
    _i(16, "How many purple shapes are in this image?" + _END.format(fmt="<number>"),
       _s("purple circle", "purple square", "orange circle", "purple triangle", "black square"),
       {"regex": _ans(r"3\b|three")}),
    _i(17, "What error code is shown in this screenshot?",
       {"kind": "text_card",
        "lines": ["Something went wrong", "Error 504: Gateway Timeout", "Request ID: 7f3a-91c2",
                  "Please try again in a few minutes."]},
       {"regex": r"\b504\b"}),
    _i(18, "Which astrologer in this table has the highest rating?" + _END.format(fmt="<name>"),
       {"kind": "table", "header": ["Astrologer", "Rating", "Reviews"],
        "rows": [["Ramesh Pandey", "4.5", "812"], ["Meera Iyer", "4.9", "1204"],
                 ["Devraj Trivedi", "4.7", "2316"], ["Sunita Joshi", "4.6", "530"]]},
       {"regex": _ans(r"Meera")}),
    _i(19, "Using the longitude on this card and 30 degrees per sign starting at Aries = 0, which sign is the "
           "Moon in?" + _END.format(fmt="<sign>"),
       {"kind": "text_card", "lines": ["PLANET LONGITUDES (SIDEREAL)", "Moon: 247 deg", "Sun: 12 deg"]},
       {"regex": _ans(r"Sagittarius")}),
    _i(20, "Are there more circles than squares in this image?" + _END.format(fmt="Yes or No"),
       _s("red circle", "blue square", "green square", "yellow circle", "black square"),
       {"regex": _ans(r"No\b")}),
    _i(21, "What date is listed for Mokshada Ekadashi in this table?",
       {"kind": "table", "header": ["Ekadashi", "Date"],
        "rows": [["Nirjala", "2026-06-25"], ["Devshayani", "2026-07-25"], ["Dev Uthani", "2026-11-20"],
                 ["Mokshada", "2026-12-20"]]},
       {"any": ["2026-12-20", "20 December 2026", "December 20, 2026", "20-12-2026", "20/12/2026", "20 Dec"]}),
    _i(22, "How many remedies are listed on this card?" + _END.format(fmt="<number>"),
       {"kind": "text_card",
        "lines": ["SUGGESTED REMEDIES", "1. Chant Hanuman Chalisa on Tuesdays", "2. Donate red lentils",
                  "3. Offer water to the Sun at sunrise", "4. Feed birds on Saturdays",
                  "5. Light a sesame-oil lamp"]},
       {"regex": _ans(r"5\b|five")}),
    _i(23, "Describe the shapes in this image: what shape are they and what colour?",
       _s("yellow triangle", "yellow triangle", "yellow triangle", "yellow triangle"),
       {"all": ["yellow", "triangle"]}),
    _i(24, "How many rows in this table have status 'active'?" + _END.format(fmt="<number>"),
       {"kind": "table", "header": ["User", "Plan", "Status"],
        "rows": [["u101", "Plus", "active"], ["u102", "Free", "inactive"], ["u103", "Plus", "active"],
                 ["u104", "Free", "active"], ["u105", "Plus", "cancelled"]]},
       {"regex": _ans(r"3\b|three")}),
    _i(25, "Read the lagna from this card. Which planet is the lord of that lagna sign?"
       + _END.format(fmt="<planet>"),
       {"kind": "text_card", "lines": ["CHART SNAPSHOT", "Lagna: Leo 12 deg 40 min", "Moon sign: Libra"]},
       {"regex": _ans(r"Sun|Surya")}),
]
