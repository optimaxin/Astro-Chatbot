"""Deterministic off-topic detection: things that are plainly not about the user's
chart or life — phone troubleshooting, cricket scores, general trivia, recipes.

Runs before any model is called, in English, Roman-script Hinglish and
Devanagari Hindi. Tuned for PRECISION, like `scope.py`: a missed off-topic
message only costs one model call, but a wrongly blocked astrology question
costs the user their answer. Two layers keep it precise:

1. An IN-SCOPE GUARD. A message with astrology vocabulary ("kundli", "dasha",
   "7th house", a planet name), or a first-person question about the user's own
   life ("will my career…", "meri shaadi…"), is never off-topic, whatever else
   it says. "Is my match compatible?", "my score in guna milan", "career period",
   "my house in the 10th" all pass through here untouched.
2. Each rule needs an explicit off-topic ENTITY plus a request word. "Cricket"
   alone fires nothing; "cricket score" does.

What the rules cannot decide (a bare "Who is the best player?") is left to the
classifier, which has an `off_topic` label and sees the conversation.
"""

from __future__ import annotations

import re

_I = re.IGNORECASE

# ── the in-scope guard ────────────────────────────────────────────────
_ASTRO = re.compile(
    r"\b(?:kundli|kundali|janam\s*patri|janma\s*kundli|horoscope|rashi|raashi|lagna|lagan|graha|grah|nakshatra|"
    r"dasha|mahadasha|antardasha|sade\s*sati|saadhe\s*saati|manglik|dosha?|yog|yoga|astrolog\w*|jyotish\w*|"
    r"ascendant|gochar|transit|muhurat|muhurta|panchang|guna\s*milan|gun\s*milan|kundli\s*milan|"
    r"saturn|jupiter|venus|mars|mercury|rahu|ketu|shani|guru|shukra|mangal|budh|surya|chandra|"
    r"\d{1,2}(?:st|nd|rd|th)\s+(?:house|bhav|bhava|ghar)|house\s+(?:lord|of\s+my)|bhav|bhava)\b"
    r"|कुंडली|कुण्डली|जन्मपत्री|राशि|लग्न|ग्रह|नक्षत्र|दशा|ज्योतिष|भाव|मांगलिक|साढ़ेसाती|गोचर|शनि|गुरु|शुक्र|मंगल",
    _I)
# A first-person question about one's own life.
_PERSONAL = re.compile(
    r"\b(?:my|mine|mera|meri|mere|mujhe|mujhko|main|mai|hamara|hamari|i|i'm|i\s+am|will\s+i|can\s+i|should\s+i)\b"
    r"[^.?!]{0,60}\b(?:career|job|naukri|marriage|married|shaadi|shadi|love|relationship|health|money|paisa|"
    r"business|future|life|salary|promotion|luck|kismat|children|baby|studies|exam|foreign|abroad|wealth|"
    r"finance|financial|partner|wife|husband|family)\b"
    r"|(?:मेरा|मेरी|मेरे|मुझे|मैं)[^।?!.]{0,60}(?:करियर|नौकरी|शादी|विवाह|प्यार|रिश्ता|स्वास्थ्य|सेहत|पैसा|धन|"
    r"व्यापार|बिज़नेस|भविष्य|जीवन|तनख्वाह|प्रमोशन|किस्मत|बच्चे|पढ़ाई|परीक्षा|विदेश|परिवार|पति|पत्नी)",
    _I)


def in_scope_signal(text: str) -> bool:
    """Astrology vocabulary or a personal-life framing: never off-topic."""
    return bool(_ASTRO.search(text) or _PERSONAL.search(text))


# ── the rules: an explicit entity plus a request word ─────────────────
_DEVICE = (r"(?:phone|mobile|smartphone|laptop|computer|pc|desktop|wifi|wi-?fi|internet|router|printer|tv|"
           r"television|ac|fridge|refrigerator|washing\s+machine|bluetooth|charger|battery|screen|keyboard|"
           r"whatsapp|facebook|instagram|youtube|gmail|email|windows|android|iphone|app|software|printer|"
           r"फोन|मोबाइल|लैपटॉप|कंप्यूटर|वाईफाई|इंटरनेट|चार्जर|बैटरी)")
_BROKEN = (r"(?:not\s+(?:working|charging|turning\s+on|connecting|opening)|won'?t\s+(?:turn|start|charge|open|connect)|"
           r"doesn'?t\s+(?:work|charge|start)|stopped\s+working|hang(?:ing|s)?|slow|crash(?:ing|ed|es)?|broken|"
           r"dead|overheat\w*|virus|error|install|uninstall|reset|restart|reboot|fix|repair|update|"
           r"nahi\s+(?:chal|chalta|chalti|chal\s+raha|chal\s+rahi|kaam|ho\s+raha|ho\s+rahi|on|khul|khulta|khulti|"
           r"charge|start)\w*|charge\s+nahi|kharab|band\s+(?:ho|hai)|hang\s+ho|slow\s+ho|theek\s+(?:nahi|kaise)|"
           r"नहीं\s+(?:चल|चार्ज|खुल)\w*|चार्ज\s+नहीं|खराब|बंद\s+हो)")

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tech_support", re.compile(rf"\b{_DEVICE}\b[^.?!]{{0,50}}{_BROKEN}|{_BROKEN}[^.?!]{{0,40}}\b{_DEVICE}\b"
                                rf"|{_DEVICE}[^.?!]{{0,30}}{_BROKEN}", _I)),
    ("sports", re.compile(
        r"\b(?:cricket|ipl|t20|odi|test\s+match|world\s+cup|football|fifa|premier\s+league|la\s+liga|tennis|"
        r"wimbledon|hockey|kabaddi|nba|nfl|formula\s*1|f1|olympics?|badminton|kohli|dhoni|messi|ronaldo|"
        r"क्रिकेट|आईपीएल|फुटबॉल)\b[^.?!]{0,50}\b(?:score|scores|scorecard|won|win|wins|winner|result|results|live|"
        r"schedule|fixtures?|points\s+table|squad|playing\s+xi|captain|wickets?|innings|goals?|"
        r"jeeta|jeeti|jita|haara|kaun\s+jeet\w*|ka\s+score|ka\s+match|स्कोर|जीता|हारा|मैच)\b"
        r"|\b(?:score|result|winner)\s+of\s+(?:yesterday'?s?|today'?s?|the|last\s+night'?s?)\s+\w*\s*(?:match|game)\b"
        r"|\bwho\s+won\b[^.?!]{0,40}\b(?:match|game|cup|series|final|tournament|election)\b"
        r"|\b(?:best|greatest|top)\s+(?:cricketer|footballer|batsman|batter|bowler|player|striker|goalkeeper)\b"
        r"|\b(?:kal|aaj)\s+ka\s+(?:match|score)\b|मैच\s+का\s+स्कोर", _I)),
    ("trivia", re.compile(
        r"\bcapital\s+of\s+\w+|\bpopulation\s+of\s+\w+|\bcurrency\s+of\s+\w+|\bwho\s+is\s+the\s+"
        r"(?:president|prime\s+minister|pm|ceo|chief\s+minister|cm)\s+of\b|\bwho\s+(?:invented|discovered|wrote|"
        r"painted|founded)\b|\bhow\s+many\s+(?:days|states|countries|continents|planets\s+are\s+there)\b|"
        r"\bspeed\s+of\s+(?:light|sound)\b|\bwhat\s+is\s+the\s+(?:tallest|longest|largest|smallest|deepest)\b|"
        r"\b\w+\s+(?:ki|ka)\s+(?:rajdhani|rajdhaani)\b|\brajdhani\s+kya\b|"
        r"\b(?:pradhan\s*mantri|rashtrapati|mukhyamantri)\s+kaun\b|\bkaun\s+(?:hai|hain)\s+\w+\s+ka\s+"
        r"(?:pradhan\s*mantri|president|ceo)\b|की\s+राजधानी|कौन\s+है\s+\S+\s+का\s+(?:प्रधानमंत्री|राष्ट्रपति)", _I)),
    ("food", re.compile(
        r"\b(?:best|top|famous)\b[^.?!]{0,25}\b(?:biryani|pizza|burger|restaurant|restaurants|cafe|dhaba|food|"
        r"street\s+food|hotel|momos|chaat)\b|\b(?:recipe|recipes)\b|\b(?:kaise\s+(?:banta|banti|banate|banaye|"
        r"banayen)|banane\s+ki\s+(?:vidhi|recipe))\b|\bhow\s+to\s+(?:cook|bake|make\s+(?:biryani|pasta|cake|"
        r"tea|coffee|pizza))\b|रेसिपी|कैसे\s+बनाएं", _I)),
    ("entertainment", re.compile(
        r"\b(?:best|new|latest|top|upcoming)\b[^.?!]{0,25}\b(?:movie|movies|film|films|song|songs|web\s*series|"
        r"show|shows|album|bollywood|netflix)\b|\b(?:movie|film|song|series)\s+(?:review|release|download|lyrics|"
        r"trailer|cast)\b|\blyrics\s+of\b|\b(?:watch|stream)\s+\w+\s+(?:online|free)\b", _I)),
    ("price_lookup", re.compile(
        r"\b(?:price|rate|cost)\s+of\s+(?:petrol|diesel|gold|silver|bitcoin|ethereum|iphone|laptop|tomato|onion|"
        r"dollar|usd)\b|\b(?:petrol|diesel|gold|silver|bitcoin|dollar)\s+(?:price|rate)\b|"
        r"\b(?:sone|chandi)\s+ka\s+(?:bhav|rate)\b|\bstock\s+price\b", _I)),
)


def offtopic_rule(text: str) -> str | None:
    """The name of the off-topic rule `text` matches, or None. Never fires on a
    message the in-scope guard recognises."""
    if not text or in_scope_signal(text):
        return None
    for name, pattern in _RULES:
        if pattern.search(text):
            return name
    return None


__all__ = ["offtopic_rule", "in_scope_signal"]
