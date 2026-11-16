"""Deterministic language checks for the benchmark.

Deliberately SEPARATE from the chatbot's own validator (app/evaluation,
app/prompts/hinglish.py's normaliser): a benchmark that only re-ran the
production rules would grade the chatbot with its own answer key. Shared
pieces are limited to plain word lists (English misspellings, machinery
phrases), not the correction logic.

Each check returns Finding(category, span, message). A rule only fires on a
pattern that is wrong in every context it matches; anything that depends on
meaning is left to the judge. Precision matters more than recall here — the
gold examples are the false-positive test (tests/test_hinglish_benchmark.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    category: str
    span: str
    message: str


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


# ── Hinglish ──────────────────────────────────────────────────────────
# Roman-Hindi spellings no fluent writer uses, and English words re-spelt.
_MISSPELT = {
    "aapako": "aapko", "aapaki": "aapki", "aapake": "aapke", "kee": "ki", "bataate": "batate", "hogaa": "hoga",
    "antardash": "Antardasha", "mahadash": "Mahadasha", "kareer": "career", "kariyar": "career",
    "promoshan": "promotion", "foren": "foreign", "imoshanal": "emotional", "paartanar": "partner",
    "committment": "commitment", "rileshanship": "relationship", "kanekshan": "connection",
    "oportunity": "opportunity", "opportunites": "opportunities", "carrer": "career", "finanse": "finance",
    "personaliti": "personality", "kammunikeshan": "communication",
}
_NON_WORDS = {"badhyaari": "behtar growth", "badhyari": "behtar growth", "bhayankar karta": "",
              "sambhalti": ""}
_FEM_NOUNS = ("dikkat|samasya|pareshani|takleef|naukri|shaadi|mahadasha|antardasha|dasha|boli|ruchi|zarurat|"
              "jaankari|information|kundli|growth|life|zindagi|sehat|health|energy|baat|soch|sambhavna|deri|"
              "unnati|safalta|aamdani|kamai|yatra|umar|chinta|tension|history|recognition|opportunity|stability")
_FEM_PLURAL = "khushiyan|dikkatein|samasyayein|samasyaen|problems|opportunities|zimmedariyan|chances|details|baatein"
_MASC_NOUNS = ("career|business|ghar|partner|sawaal|swabhaav|swabhav|lagna|confidence|mann|kaam|samay|time|"
               "swasthya|field|period|yog|asar|fayda|nuksan|rishta|mauka|faisla|future|bhavishya|chart|plan|"
               "result|support|focus|energy level|sign|mahaul|sambandh")
_MASC_VERB = r"(?:sakta|raha|rahega|hoga|aayega|aaega|milega|banega|badhega|chalega|jayega|jaega|deta|hota|karta|laata)"
_FEM_VERB = r"(?:sakti|rahi|rahegi|hogi|aayegi|milegi|banegi|badhegi|chalegi|jayegi|deti|hoti|karti)"

HINGLISH_RULES: list[tuple[str, re.Pattern[str], str]] = [
    # gender: feminine noun + masculine verb, within a short clause.
    ("gender_agreement", _rx(rf"\b(?:{_FEM_NOUNS})\b(?!\s+(?:mein|me|se|par|ko|ka|ki|ke)\b)(?:\s+\w+){{0,2}}?\s+"
                             rf"(?:aa|ho|mil|ban|badh|chal|reh|rah|la|de)\s+{_MASC_VERB}\b"),
     "feminine noun with a masculine verb"),
    ("gender_agreement", _rx(rf"\b(?:{_FEM_NOUNS})\s+(?:milega|hoga|rahega|aayega|banega|badhega|chalega|jayega)\b"),
     "feminine noun with a masculine verb"),
    ("gender_agreement", _rx(rf"\b(?:aapka|mera|uska|iska|tumhara)\s+(?:{_FEM_NOUNS})\b(?!\s+(?:ke|ki|ka)\b)"),
     "masculine possessive before a feminine noun"),
    ("gender_agreement", _rx(r"\baap\s+ka\s+(?:kundli|life|zindagi|shaadi|naukri)\b"), "'aap ka' before a feminine noun"),
    ("gender_agreement", _rx(rf"\b(?:aapki|meri|uski|iski|tumhari|apni)\s+(?:{_MASC_NOUNS})\b(?!\s+(?:ki|ke)\b)"),
     "feminine possessive before a masculine noun"),
    ("gender_agreement", _rx(r"\b(?:ka|ke)\s+zarurat\b"), "'zarurat' is feminine: 'ki zarurat'"),
    ("gender_agreement", _rx(rf"\b(?:{_FEM_PLURAL})\b(?:\s+\w+){{0,2}}?\s+(?:aayenge|honge|milenge|banenge)\b"),
     "feminine plural noun with a masculine plural verb"),
    ("gender_agreement", _rx(r"\b(?:sawaal|swagat|kaam|lagna)\s+(?:bahut\s+)?achhi\b"), "masculine noun with 'achhi'"),
    ("gender_agreement", _rx(r"\bboli\s+kadwa\b"), "'boli' is feminine: 'kadvi'"),
    # number: singular participle + "hain", plural participle + "hai".
    # (feminine plural is "sakti hain", so only masculine singular forms count)
    ("number_agreement", _rx(r"\b(?:raha|deta|hota|karta|sakta|lagta|jata|aata)\s+hain\b"),
     "singular verb form with 'hain'"),
    ("number_agreement", _rx(r"\b(?:rahe|dete|hote|karte|sakte|poochte|lagte|jaate|jate|chahte)\s+hai\b"),
     "plural verb form with 'hai'"),
    ("number_agreement", _rx(r"\b(?:aapka|aapki|mera|meri|uska|uski|iska|iski)\s+(?!\w+s\b)\w+(?:\s+\w+)?\s+hain\b"
                             r"(?!\s+jo)"),
     "singular subject with 'hain'"),
    ("number_agreement", _rx(r"\b(?:problems|opportunities|dikkatein|samasyayein|chances)\s+(?:aa|ho|ban|mil)\s+sakta\b"),
     "plural noun with a singular verb"),
    ("number_agreement", _rx(r"\basar\s+(?:\w+\s+){0,3}padenge\b"), "'asar' is singular"),
    # postpositions / case.
    ("postposition", _rx(r"\b(?:aapka|mera|uska|iska|tumhara)\s+\w+\s+(?:mein|me|par|se|ke\s+liye)\b"),
     "oblique case: 'aapke … mein'"),
    ("postposition", _rx(r"\b\w+ko\s+madad\b|\baapko\s+madad\b"), "'madad' takes 'ki': 'aapki madad'"),
    ("postposition", _rx(r"\bto\s+(?:[A-Z][a-z]+\s+)?\d{4}\s+tak\b"), "range is 'se … tak', not 'to … tak'"),
    ("postposition", _rx(r"\b(?:sun|moon|mars|mercury|jupiter|venus|saturn|rahu|ketu)\s+ka\s+\d+\w*\s+house\s+mein\s+hone\b"),
     "'X ka … mein hone' — use 'X … mein hone se'"),
    ("postposition", _rx(r"\bharmony\s+ke\s+nuksan\b"), "'harmony ke nuksan' — use 'harmony mein dikkat'"),
    # verb forms.
    ("verb_form", _rx(r"\bjoda\s+hota\s+hai\b"), "'joda hota hai' — 'juda hota hai' or 'joda jata hai'"),
    ("verb_form", _rx(r"\bchahiye\s+hai\b|\bhogi\s+hai\b|\bhoga\s+hai\b"), "double auxiliary"),
    # collocations: English plural noun + "la sakta", "nuksan de sakta".
    ("collocation", _rx(r"\b(?:issues?|problems?|delays?|challenges?|kathinaiyon\s+ko)\s+la\s+sakt[aie]\b"),
     "unnatural 'X la sakta hai' — use 'X aa sakti hai' / 'X ho sakti hai'"),
    ("collocation", _rx(r"\bka\s+effect\s+\w+(?:\s+\w+)?\s+la\s+sakta\b"), "'ka effect … la sakta' — use 'ke kaaran … aa sakti'"),
    ("collocation", _rx(r"\bnuksan\s+de\s+sakt[aie]\b"), "'nuksan de sakta' — use 'dikkat aa sakti' / 'nuksan ho sakta'"),
    ("collocation", _rx(r"\bsamay-samay\s+par\s+delays\b"), "Hindi idiom with an English plural — 'samay-samay par deri'"),
    ("word_choice", _rx(r"\bsituate\s+(?:hai|hain|ho)\b"), "'situate hai' — just 'hai'"),
    # register: heavy Sanskrit / bureaucratic Hindi inside Hinglish.
    ("heavy_register", _rx(r"\b(?:prapt|praapt|pradaan|avashya|kartavyanishth|shatruon|vijay\s+dilata|upyog|ityadi|"
                           r"vishleshan|kathinaiyon|avsar)\b"), "formal/literary word in casual Hinglish"),
    ("literal_translation", _rx(r"\bchat\s+ka\s+itihaas\b|\bmain\s+khush\s+hoon\s+\w+\s+dene\s+mein\b|"
                                r"\bhone\s+ke\s+chances\s+hain\b"), "word-for-word translation"),
    ("word_order", _rx(r"\bmain\s+samjha\s+nahi\s+aapka\b|\bcheck\s+kar\s+raha\s+hoon\s+aapki\s+\w+\s+ko\b"),
     "English word order"),
    # address and respect.
    ("register_mix", _rx(r"\baap\b[^.?!]{0,40}\b(?:karo|batao|socho|chalo|dekho|suno|jao)\b"), "'aap' with a 'tum' imperative"),
    ("register_mix", _rx(r"\baap\b[^.?!]{0,30}\b(?:chahte|sakte|karte|rahe|jaante)\s+ho\b"), "'aap … ho' — use 'hain'"),
    ("gender_assumption", _rx(r"\baap\b[^.?!]{0,40}\b(?:hongi|sakti\s+hain|karti\s+hain|rahi\s+hain|chahti\s+hain)\b"),
     "assumes the user is female"),
    # certainty (meaning).
    ("certainty", _rx(r"\b(?:pakka|pakki|definitely|avashya|guaranteed?|zaroor\s+(?:hoga|hogi|milega|milegi)|"
                      r"nishchit(?:\s+roop\s+se)?)\b(?![^.?!]{0,30}\b(?:nahi|nahin)\b)"), "guaranteed outcome"),
    ("unsafe_claim", _rx(r"\b(?:bimari|rog|disease)\s+(?:hogi|hoga|ho\s+jayegi|ho\s+jaegi)\b"), "predicts an illness"),
    # masculine noun (not followed by a postposition) with a feminine verb.
    ("gender_agreement", _rx(r"\b(?:mann|career|business|partner|kaam|ghar|confidence|swabhaav|lagna)\b"
                             # at most one word in between and not a plural noun — that is the real
                             # subject ("career opportunities mil sakti hain" is correct; live run)
                             r"(?!\s+(?:mein|me|se|par|ko|ka|ki|ke)\b)(?:\s+(?!\w+s\b)\w+)?\s+(?:aa|ho|mil|ban|badh|chal|"
                             r"reh|rah)\s+(?:sakti|rahi|rahegi|hogi)\b"), "masculine noun with a feminine verb"),
    # feminine noun ... masculine future, with no new subject in between.
    ("gender_agreement", _rx(rf"\b(?:{_FEM_NOUNS})\b(?!\s+(?:mein|me|se|par|ko|ka|ki|ke|\w+n[ae])\b)"
                             r"(?:\s+(?!(?:mein|me|se|par|ko|ka|ki|ke|aur|ya|jo|jisse|isliye|\w+n[ae])\b)\w+){1,3}?\s+"
                             r"(?:rahega|hoga|milega|banega|chalega|aayega)\b"), "feminine noun with a masculine verb"),
    ("sentence_structure", _rx(r"\bmein\s+hona\s+(?:\w+\s+){0,6}?(?:deti|karti|banati|deta|karta|banata)\s+hai\b"),
     "'X mein hona … deti hai' — use 'X mein hone se … mil sakta hai'"),
    ("gender_agreement", _rx(r"\b(?:aapka|mera|uska)\s+(?:birth\s+)?details\b"), "'details' is feminine plural"),
]
_TUM = _rx(r"\b(?:tum|tumhara|tumhari|tumhare|tumhe|tumko)\b")
_AAP = _rx(r"\b(?:aap|aapka|aapki|aapke|aapko)\b")
_HINDI_FUNCTION = _rx(r"\b(?:hai|hain|mein|aap|aapka|aapki|aapke|aapko|ka|ki|ke|ko|se|aur|nahi|kya|ho|hoga|hogi)\b")

# ── Hindi (Devanagari) ────────────────────────────────────────────────
_PLANETS_HI = "मंगल|चंद्रमा|चंद्र|सूर्य|शनि|बृहस्पति|गुरु|बुध|शुक्र|राहु|केतु"
HINDI_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("gender_agreement", re.compile(r"(?:महादशा|अंतर्दशा|दशा)\s+(?:(?!में|से|पर|को)\S+\s+){0,3}?(?:चल\s+रहा|रहेगा|होगा)"),
     "दशा is feminine"),
    ("gender_agreement", re.compile(rf"(?:{_PLANETS_HI})[^।,]{{0,40}}?(?:बनाती|देती|करती)\s+है"), "planets take masculine verbs"),
    ("gender_agreement", re.compile(r"(?:आपकी|मेरी)\s+जीवनसाथी"), "जीवनसाथी takes a masculine possessive"),
    ("gender_agreement", re.compile(r"उन्नति\s+होगा|समस्या\s+(?:हो|आ)\s+सकता"), "feminine noun with a masculine verb"),
    ("gender_agreement", re.compile(r"(?:माता|माँ)\s+\S+\s+होंगे"), "माँ takes feminine forms"),
    ("number_agreement", re.compile(r"(?:आता|जाता|होता|रहता|करता|देता)\s+हैं"), "singular verb with हैं"),
    ("number_agreement", re.compile(r"(?:लग्न|स्वामी)\s+\S+\s+हैं(?!\s*,)"), "singular subject with हैं"),
    ("number_agreement", re.compile(r"समस्याएं\s+(?:हो|आ)\s+सकता"), "plural noun with a singular verb"),
    ("postposition", re.compile(r"(?:पहला|दूसरा|तीसरा|चौथा|पाँचवाँ|छठा|सातवाँ|आठवाँ|नौवाँ|दसवाँ)\s+भाव\s+(?:में|का|से)"),
     "oblique case: चौथे भाव में"),
    ("postposition", re.compile(r"(?:आपका|मेरा)\s+\S+\s+में"), "oblique case: आपके … में"),
    ("gender_agreement", re.compile(r"मंगल\s+\S+\s+भाव\s+में\s+\S+\s+\S+\s+को\s+\S+\s+बनाती"), "मंगल is masculine"),
    ("register_mix", re.compile(r"तुम्हारा|तुम्हारी|तुम्हें"), "तुम in a respectful reply"),
    ("register_mix", re.compile(r"आप\s+(?:\S+\s+){0,2}हो[।\s]*$"), "आप … हो — use हैं"),
    ("certainty", re.compile(r"निश्चित(?:\s+रूप\s+से)?|पक्की\s+होगी|पक्का\s+होगा|अवश्य\s+होगा"), "guaranteed outcome"),
]

# ── English ───────────────────────────────────────────────────────────
_PLANETS_EN = "Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu"
ENGLISH_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("number_agreement", _rx(r"\byou\s+has\b|\b(?:lagna|ascendant|chart)\s+are\b"), "subject–verb agreement"),
    ("number_agreement", re.compile(rf"\b(?:{_PLANETS_EN})\b(?:\s+in\s+(?:its\s+own\s+sign|your\s+\d+\w*\s+house|"
                                    rf"\d+\w*\s+house))\s+(?:give|make|bring|show|create)\b"),
     "singular planet with a plural verb"),
    ("postposition", _rx(r"\bsince\s+\w+\s+\d{4}\s+to\b|\bfrom\s+\w+\s+\d{4}\s+till\b"), "'from … to'"),
    ("verbose", _rx(r"\b(?:might|may)\s+possibly\b|\bis\s+indicating\s+that\b"), "hedging stacked on hedging"),
    ("verbose", _rx(r"\band\s+(?:she|he|it|they)\s+will\s+be\b.*\band\s+(?:she|he|it|they)\s+will\s+be\b"),
     "repeated clause"),
    ("filler", _rx(r"\blooks?\s+promising\b|\bpositive\s+energy\b|\blooking\s+promising\b"), "filler phrase"),
    ("certainty", _rx(r"\b(?:definitely|certainly|surely|guaranteed)\b|\bwill\s+face\s+many\b"), "guaranteed outcome"),
    ("unsafe_claim", _rx(r"\byou\s+should\s+invest\b|\bmany\s+diseases\b"), "advice or diagnosis"),
    ("word_choice", _rx(r"\bin\s+foreign\b|\bcome\s+slow\b"), "non-standard English"),
    ("sentence_structure", _rx(r"\bbe\s+(?:very\s+|quite\s+)?\w+ing\s+person\b|\bbe\s+very\s+\w+\s+person\b"),
     "missing article"),
]
_HINGLISH_MARKERS = _rx(r"\b(?:aapka|aapki|aapke|aapko|achha|accha|rahega|rahegi|hoga|hogi|hai|hain|mein|kya|nahi|"
                        r"lekin|isliye)\b")

_MACHINERY = _rx(r"retrieved\s+knowledge|knowledge\s+(?:base|passages?)|language\s+model|bhasha\s+model|"
                 r"created\s+by\s+nvidia|\bnvidia\b|भाषा\s+मॉडल|एनवीआईडीआईए|chart\s+data\s+(?:says|shows)")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _findall(rules, text: str) -> list[Finding]:
    out: list[Finding] = []
    for category, pattern, message in rules:
        for match in pattern.finditer(text):
            out.append(Finding(category, match.group(0), message))
    return out


def check(text: str, language: str) -> list[Finding]:
    """All deterministic findings for one reply in `language`."""
    text = text or ""
    findings: list[Finding] = []
    if _MACHINERY.search(text):
        findings.append(Finding("machinery", _MACHINERY.search(text).group(0), "mentions the system behind the answer"))
    if language == "hinglish":
        lowered = re.findall(r"[A-Za-z]+(?:\s+karta)?", text)
        for word in lowered:
            key = word.lower()
            if key in _MISSPELT:
                findings.append(Finding("transliteration" if key in {"kareer", "kariyar", "promoshan", "foren",
                                                                     "imoshanal", "paartanar", "rileshanship",
                                                                     "kanekshan", "kammunikeshan", "personaliti"}
                                        else "spelling", word, f"write '{_MISSPELT[key]}'"))
            if key in _NON_WORDS:
                findings.append(Finding("non_word", word, "not a Hindi word"))
        findings += _findall(HINGLISH_RULES, text)
        if _TUM.search(text) and _AAP.search(text):
            findings.append(Finding("register_mix", "tum + aap", "mixes 'tum' and 'aap'"))
        if _DEVANAGARI.search(text):
            findings.append(Finding("script_mixing", "".join(_DEVANAGARI.findall(text))[:12], "Devanagari in Hinglish"))
    elif language == "hindi":
        findings += _findall(HINDI_RULES, text)
        latin = re.findall(r"[A-Za-z]+", text)
        roman_hindi = [w for w in latin if _HINDI_FUNCTION.fullmatch(w)]
        deva = len(_DEVANAGARI.findall(text))
        if roman_hindi and deva == 0:
            findings.append(Finding("language_switch", " ".join(latin[:4]), "Roman Hinglish instead of Hindi"))
        elif len(roman_hindi) >= 2 or (roman_hindi and len(latin) >= 3):
            findings.append(Finding("script_mixing", " ".join(roman_hindi[:4]), "Roman Hindi inside a Hindi reply"))
    else:
        findings += _findall(ENGLISH_RULES, text)
        for word in re.findall(r"[A-Za-z]+", text):
            if word.lower() in {"carrer", "finanse", "recieve", "marrige", "oppurtunity", "seperate", "acheive"}:
                findings.append(Finding("spelling", word, "misspelt English"))
        markers = {m.lower() for m in _HINGLISH_MARKERS.findall(text)}
        if len(markers) >= 2:
            findings.append(Finding("language_switch", "Hindi words", "Hinglish inside an English reply"))
        if _DEVANAGARI.search(text):
            findings.append(Finding("script_mixing", "".join(_DEVANAGARI.findall(text))[:12], "Devanagari in English"))
    return _dedupe(findings)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str]] = set()
    out = []
    for f in findings:
        key = (f.category, f.span.lower())
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


__all__ = ["Finding", "check"]
