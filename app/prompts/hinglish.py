"""Hinglish language quality: vocabulary, normalisation and validation.

Hinglish = natural Hindi in Roman script, with the English words people
actually use kept in their correct English spelling ("partner",
"commitment", "relationship"), never re-spelled the Hindi way ("paartanar",
"rileshanship").

Three parts, all deterministic:

* ``DEVANAGARI_LEXICON`` / ``DEVANAGARI_PHRASES`` — Devanagari -> Roman for the
  words that appear in astrology chat, including English loanwords written in
  Devanagari. Used by app/prompts/transliterate.py.
* ``normalize_hinglish(text)`` — repairs forms that are wrong in EVERY context:
  misspelled English ("committment"), letter-by-letter transliterations
  ("aapako", "kanekshan"), and a few fixed grammar errors (masculine
  possessive before a postposition: "aapka career mein" -> "aapke career
  mein"). It never touches a correctly spelled word, a chart fact or a date:
  each rule targets a token that cannot be right where it stands.
* ``hinglish_issues(text)`` — what is still wrong after normalisation, for the
  validator (one repair at most) and the benchmark.

Real-user report (2026-11): "Aapako partner pasand aaega jo soch-samajh ke
bolta hai", "paartanar", "committment", "maanasik kanekshan", "bataate hain kee".
"""

from __future__ import annotations

import re

# ── Devanagari -> Roman ───────────────────────────────────────────────
# English loanwords first: phonetic rules cannot recover their spelling.
_LOANWORDS = {
    "पार्टनर": "partner", "पार्टनरशिप": "partnership", "कनेक्शन": "connection", "कमिटमेंट": "commitment",
    "रिलेशनशिप": "relationship", "रिलेशन": "relation", "करियर": "career", "कैरियर": "career",
    "मैरिज": "marriage", "पर्सनैलिटी": "personality", "पर्सनालिटी": "personality",
    "कम्युनिकेशन": "communication", "ऑपर्च्युनिटी": "opportunity", "ऑपर्च्युनिटीज": "opportunities",
    "अवसर": "avsar", "फाइनेंशियल": "financial", "फाइनेंस": "finance", "ग्रोथ": "growth",
    "प्रैक्टिकल": "practical", "इंटेलिजेंट": "intelligent", "एनालिसिस": "analysis",
    "कॉन्फिडेंस": "confidence", "रिस्पॉन्सिबिलिटी": "responsibility", "कम्पैटिबिलिटी": "compatibility",
    "कंपैटिबिलिटी": "compatibility", "प्रोफेशनल": "professional", "डेवलपमेंट": "development",
    "एजुकेशन": "education", "इमोशनल": "emotional", "जॉब": "job", "बिज़नेस": "business",
    "बिजनेस": "business", "सपोर्टिव": "supportive", "सपोर्ट": "support", "पीरियड": "period",
    "हाउस": "house", "लॉर्ड": "lord", "चार्ट": "chart", "टाइम": "time", "फोकस": "focus",
    "सक्सेस": "success", "स्टेबिलिटी": "stability", "स्टेबल": "stable", "बैलेंस": "balance",
    "नेचर": "nature", "ट्रेडिशनल": "traditional", "ट्रेडिशनली": "traditionally", "पॉजिटिव": "positive",
    "नेगेटिव": "negative", "चैलेंज": "challenge", "चैलेंजेस": "challenges", "प्लेसमेंट": "placement",
    "इंडिकेशन": "indication", "इंडिकेशंस": "indications", "हेल्थ": "health", "फैमिली": "family",
    "लव": "love", "मैच": "match", "सेटल": "settle", "वर्क": "work", "प्रोफेशन": "profession",
    "डिसीजन": "decision", "प्लानिंग": "planning", "एनर्जी": "energy", "स्ट्रॉन्ग": "strong",
    "लाइफ": "life", "पार्टनर्स": "partners",
}
_HINDI_WORDS = {
    "है": "hai", "हैं": "hain", "हो": "ho", "था": "tha", "थी": "thi", "थे": "the", "में": "mein",
    "की": "ki", "कि": "ki", "के": "ke", "का": "ka", "को": "ko", "से": "se", "पर": "par", "और": "aur",
    "भी": "bhi", "यह": "yeh", "ये": "ye", "वह": "woh", "वो": "woh", "नहीं": "nahi", "ना": "na",
    "आप": "aap", "आपका": "aapka", "आपकी": "aapki", "आपके": "aapke", "आपको": "aapko", "आपसे": "aapse",
    "जो": "jo", "तो": "toh", "होगा": "hoga", "होगी": "hogi", "होंगे": "honge", "होना": "hona",
    "सकता": "sakta", "सकती": "sakti", "सकते": "sakte", "रहा": "raha", "रही": "rahi", "रहे": "rahe",
    "लिए": "liye", "साथ": "saath", "बहुत": "bahut", "अच्छा": "accha", "अच्छी": "acchi", "अच्छे": "acche",
    "शादी": "shaadi", "विवाह": "vivah", "कुंडली": "Kundli", "महादशा": "Mahadasha", "अंतर्दशा": "Antardasha",
    "दशा": "Dasha", "भाव": "bhav", "घर": "ghar", "लग्न": "Lagna", "राशि": "rashi", "नक्षत्र": "Nakshatra",
    "बुध": "Budh", "शुक्र": "Shukra", "शनि": "Shani", "गुरु": "Guru", "बृहस्पति": "Brihaspati",
    "सूर्य": "Surya", "चंद्र": "Chandra", "चंद्रमा": "Chandrama", "मंगल": "Mangal", "राहु": "Rahu",
    "केतु": "Ketu", "समय": "samay", "संकेत": "sanket", "पारंपरिक": "paramparik", "ज्योतिष": "jyotish",
    "ज्योतिषी": "jyotishi", "बताते": "batate", "बताता": "batata", "बताती": "batati", "बताया": "bataya",
    "मानसिक": "maansik", "सोच": "soch", "समझ": "samajh", "समझकर": "samajhkar", "बात": "baat",
    "करता": "karta", "करती": "karti", "करते": "karte", "करना": "karna", "करें": "karein", "देता": "deta",
    "देती": "deti", "देते": "dete", "मिल": "mil", "मिलेगा": "milega", "मिलेगी": "milegi", "जीवन": "jeevan",
    "जीवनसाथी": "jeevansathi", "साथी": "saathi", "स्वभाव": "swabhav", "प्यार": "pyaar", "रिश्ता": "rishta",
    "रिश्ते": "rishte", "परिवार": "parivaar", "पैसा": "paisa", "धन": "dhan", "काम": "kaam", "नौकरी": "naukri",
    "स्वामी": "swami", "स्थिति": "sthiti", "अनुसार": "anusaar", "हिसाब": "hisaab", "मतलब": "matlab",
    "अभी": "abhi", "चल": "chal", "जब": "jab", "तक": "tak", "अगर": "agar", "लेकिन": "lekin", "क्योंकि": "kyunki",
    "पसंद": "pasand", "आएगा": "aayega", "आएगी": "aayegi", "आ": "aa", "ऐसा": "aisa", "ऐसी": "aisi",
    "मजबूत": "majboot", "मज़बूत": "mazboot", "महत्व": "mahatva", "संभावना": "sambhavna", "चुनौती": "chunauti",
    "विशेष": "vishesh", "सप्तम": "saptam", "दशम": "dasham",
    "इशारा": "ishara", "घरेलू": "gharelu", "माहौल": "maahaul", "ओर": "or", "दे": "de", "देना": "dena",
    "तरफ": "taraf", "रूप": "roop", "जीवनशैली": "jeevanshaili", "व्यक्तित्व": "vyaktitva", "स्वास्थ्य": "swasthya",
    "शिक्षा": "shiksha", "पढ़ाई": "padhai", "बच्चे": "bacche", "संतान": "santaan", "माता": "maata",
    "पिता": "pita", "भाई": "bhai", "बहन": "behen", "घर-परिवार": "ghar-parivaar", "सुख": "sukh",
}
DEVANAGARI_LEXICON: dict[str, str] = {**_HINDI_WORDS, **_LOANWORDS}
# Multi-word phrases mapped as a unit (checked before single words).
DEVANAGARI_PHRASES: dict[str, str] = {
    "मानसिक कनेक्शन": "emotional connection",
    "भावनात्मक जुड़ाव": "emotional connection",
    "सोच-समझकर": "soch-samajhkar",
}

# ── Roman Hinglish vocabulary: wrong form -> correct form ─────────────
# Only forms that are wrong wherever they appear: misspelled English, and
# letter-by-letter transliterations of English or of common Hindi words.
ENGLISH_VOCAB: dict[str, str] = {
    "paartanar": "partner", "paartnar": "partner", "partnar": "partner", "paatnar": "partner",
    "committment": "commitment", "commitement": "commitment", "comitment": "commitment",
    "kamitament": "commitment", "kamitment": "commitment",
    "rileshanship": "relationship", "rileshnship": "relationship", "relashionship": "relationship",
    "relationsip": "relationship",
    "imotional": "emotional", "imoshanal": "emotional", "emosanal": "emotional",
    "kanekshan": "connection", "kaneksan": "connection", "conection": "connection", "connnection": "connection",
    "parsnaliti": "personality", "parsanaliti": "personality", "personaliti": "personality",
    "kariyar": "career", "kairiyar": "career", "carrer": "career", "carreer": "career",
    "kamyunikeshan": "communication", "kamyunikesan": "communication", "comunication": "communication",
    "oporchunities": "opportunities", "oparchunitij": "opportunities", "oporchuniti": "opportunity",
    "opportunites": "opportunities", "oppurtunities": "opportunities", "oppurtunity": "opportunity",
    "fainenshiyal": "financial", "phainenshiyal": "financial", "finacial": "financial",
    "mairij": "marriage", "mairej": "marriage", "marrige": "marriage", "mariage": "marriage",
    "praiktikal": "practical", "praktikal": "practical",
    "intellijent": "intelligent", "intelijent": "intelligent", "inteligent": "intelligent",
    "enalisis": "analysis", "analisis": "analysis", "analysys": "analysis",
    "konfidens": "confidence", "confidance": "confidence",
    "responsibiliti": "responsibility", "responsiblity": "responsibility", "risponsibiliti": "responsibility",
    "kompatibiliti": "compatibility", "compatiblity": "compatibility", "compatability": "compatibility",
    "profeshanal": "professional", "profesional": "professional", "proffesional": "professional",
    "divelapment": "development", "devlopment": "development", "developement": "development",
    "ejukeshan": "education", "edukeshan": "education", "eduction": "education",
    "saportiv": "supportive", "saport": "support", "piriyad": "period",
    "traditionaly": "traditionally", "tredishanal": "traditional", "posibility": "possibility",
    "possibilites": "possibilities", "sucess": "success", "succes": "success", "stabilty": "stability",
    # letter-by-letter transliterations of common Hindi words
    "aapako": "aapko", "aapaka": "aapka", "aapakee": "aapki", "aapakii": "aapki", "aapake": "aapke",
    "aapakaa": "aapka", "bataate": "batate", "bataataa": "batata", "karataa": "karta", "karatee": "karti",
    "hotaa": "hota", "hotee": "hoti", "sakataa": "sakta", "sakatee": "sakti", "rahaa": "raha", "rahee": "rahi",
    "jyotishee": "jyotishi", "kundalee": "kundli", "shaadee": "shaadi", "naheen": "nahi",
}
# Phrase-level rewrites (meaning-preserving; context included in the pattern).
_PHRASES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bmaana?sik\s+(?:kanekshan|connection)\b", re.I), "emotional connection"),
    (re.compile(r"\bimotional\s+kanekshan\b", re.I), "emotional connection"),
    # "soch-samajh ke bolta hai" -> "soch-samajhkar baat karta hai"
    (re.compile(r"\bsoch[- ]samajh\s*(?:ke|kar)\s+bol(ta|ti|te)\b", re.I), r"soch-samajhkar baat kar\1"),
    (re.compile(r"\bsoch[- ]samajh\s+ke\b", re.I), "soch-samajhkar"),
    # "bataate hain kee" / "hai kee" -> "... ki": "kee" is never the conjunction's spelling.
    (re.compile(r"\b(hai|hain|ho|tha|thi|the|batate hain|kehte hain|maana jata hai)\s+kee\b", re.I), r"\1 ki"),
)
# Masculine possessive before a postposition takes the oblique form:
# "aapka career mein" -> "aapke career mein". Feminine "aapki" stays.
# Exactly one word in between: in "aapka Lagna Pisces mein hai" the
# postposition belongs to "Pisces", and "aapka" is correct.
_OBLIQUE = re.compile(r"\b(aapk|tumhar|mer|uns?k|ink|isk|usk|hamar)a(\s+[A-Za-z]+\s+)"
                      r"(mein|me|se|par|ko|ke\s+liye|ke\s+saath|tak)\b", re.I)

_AAP_FEMININE = re.compile(r"\b(aap\b[^.!?;]{0,60}?\b)(sakti|rahi|karti|chahti|hoti|jaati|paati|lagti|deti|"
                           r"leti|samajhti|sochti|mehsoos karti)(\s+(?:hain|ho|hongi|thi))\b", re.I)
_FEM_TO_NEUTRAL = {"sakti": "sakte", "rahi": "rahe", "karti": "karte", "chahti": "chahte", "hoti": "hote",
                   "jaati": "jaate", "paati": "paate", "lagti": "lagte", "deti": "dete", "leti": "lete",
                   "samajhti": "samajhte", "sochti": "sochte", "mehsoos karti": "mehsoos karte"}

_WORD = re.compile(r"[A-Za-z]+")
# Spellings that look like an English word re-spelled the Hindi way and are
# not in the vocabulary: reported, never auto-corrected.
_SUSPECT = re.compile(r"\b[a-z]*(?:eshan|ishan|kshan|shanal|iti|aiti|ij|ikal|ent?s?ij)\b", re.I)
_SUSPECT_OK = {"iti", "shaanti", "shanti", "prakriti", "sthiti", "niti", "kriti", "riti", "jyoti", "pragati",
               "sampatti", "sanskriti", "kalpana", "aakarshan", "lakshan", "tikshan", "bhishan", "dhan",
               "parikshan", "chikitsa", "prasidh", "jaati", "gati", "mati", "pati", "sati", "unnati",
               "shakti", "bhakti", "mukti", "vyakti", "aakriti", "samriddhi", "bhumi", "upasthiti",
               "abhivyakti", "uttarayan", "dakshin", "lakshmi", "rashi", "dristi", "drishti", "smriti",
               "anubhuti", "sangati", "sahamati", "kriti", "vritti", "pravritti", "nikal", "kalakriti",
               "sanskriti", "aakriti", "vikal", "nishchit"}


def _match_case(original: str, replacement: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def normalize_hinglish(text: str) -> tuple[str, list[str]]:
    """Fix forms that are wrong in every context. Returns (text, fixes)."""
    fixes: list[str] = []
    out = text or ""
    for rx, repl in _PHRASES:
        out, n = rx.subn(repl, out)
        if n:
            fixes.append(f"phrase:{rx.pattern[:30]}x{n}")

    def word(m: re.Match) -> str:
        token = m.group(0)
        right = ENGLISH_VOCAB.get(token.lower())
        if right is None:
            return token
        fixes.append(f"{token.lower()}->{right}")
        return _match_case(token, right)

    out = _WORD.sub(word, out)

    def oblique(m: re.Match) -> str:
        fixes.append(f"oblique:{m.group(1)}a->e")
        return f"{m.group(1)}e{m.group(2)}{m.group(3)}"

    out = _OBLIQUE.sub(oblique, out)

    def neutral(m: re.Match) -> str:
        fixes.append(f"aap-gender:{m.group(2)}")
        tail = m.group(3).replace("hongi", "honge").replace(" thi", " the")
        return f"{m.group(1)}{_match_case(m.group(2), _FEM_TO_NEUTRAL[m.group(2).lower()])}{tail}"

    out = _AAP_FEMININE.sub(neutral, out)
    return out, fixes


def hinglish_issues(text: str) -> dict[str, list[str]]:
    """What still looks wrong in a Hinglish reply. Pure; changes nothing."""
    words = [w.lower() for w in _WORD.findall(text or "")]
    issues: dict[str, list[str]] = {}
    misspelled = sorted({w for w in words if w in ENGLISH_VOCAB})
    if misspelled:
        issues["misspelled"] = misspelled
    suspect = sorted({m.group(0).lower() for m in _SUSPECT.finditer(text or "")
                      if m.group(0).lower() not in _SUSPECT_OK and m.group(0).lower() not in ENGLISH_VOCAB
                      and len(m.group(0)) > 4})
    if suspect:
        issues["suspect_transliteration"] = suspect
    if re.search(r"[ऀ-ॿ]", text or ""):
        issues["devanagari"] = sorted(set(re.findall(r"[ऀ-ॿ]+", text)))[:5]
    if len(re.findall(r"\bke according\b", text or "", re.I)) >= 3:
        issues["repeated_phrase"] = ["ke according"]
    if _OBLIQUE.search(text or ""):
        issues["grammar"] = [m.group(0) for m in _OBLIQUE.finditer(text)][:3]
    tum = bool(re.search(r"\b(?:tum|tumhara|tumhari|tumhare|tumhe|tumko)\b", text or "", re.I))
    aap = bool(re.search(r"\b(?:aap|aapka|aapki|aapke|aapko)\b", text or "", re.I))
    if tum and aap:
        issues["address_mixed"] = ["tum+aap"]
    return issues


# Roman Hindi -> Devanagari for a HINDI reply that slipped into Hinglish
# mid-sentence ("आपकी कुंडली में 5th house ka sign Cancer है" — round-3 benchmark).
# Only unambiguous Hindi words and astrology proper nouns; words that are also
# English ("the", "to", "me", "main", "do") are never touched.
ROMAN_TO_DEVANAGARI: dict[str, str] = {
    "hai": "है", "hain": "हैं", "mein": "में", "ka": "का", "ki": "की", "ke": "के", "ko": "को", "se": "से",
    "aur": "और", "bhi": "भी", "yeh": "यह", "ye": "ये", "jo": "जो", "toh": "तो", "nahi": "नहीं", "nahin": "नहीं",
    "aap": "आप", "aapka": "आपका", "aapki": "आपकी", "aapke": "आपके", "aapko": "आपको", "hoga": "होगा",
    "hogi": "होगी", "honge": "होंगे", "sakta": "सकता", "sakti": "सकती", "sakte": "सकते", "raha": "रहा",
    "rahi": "रही", "rahe": "रहे", "liye": "लिए", "milega": "मिलेगा", "milegi": "मिलेगी", "deta": "देता",
    "deti": "देती", "dete": "देते", "karta": "करता", "karti": "करती", "karte": "करते", "hota": "होता",
    "hoti": "होती", "hote": "होते", "jata": "जाता", "jaata": "जाता", "jati": "जाती", "jaati": "जाती",
    "mana": "माना", "maana": "माना", "wala": "वाला", "wali": "वाली", "wale": "वाले", "lekin": "लेकिन",
    "kyunki": "क्योंकि", "abhi": "अभी", "isliye": "इसलिए", "jiska": "जिसका", "jiski": "जिसकी",
    "jiske": "जिसके", "uska": "उसका", "uski": "उसकी", "uske": "उसके", "iska": "इसका", "iski": "इसकी",
    "iske": "इसके", "tak": "तक", "saath": "साथ", "kaafi": "काफी", "thoda": "थोड़ा", "bahut": "बहुत",
    "sabse": "सबसे", "ek": "एक", "situate": "स्थित", "sthit": "स्थित", "chal": "चल", "lagti": "लगती",
    "lagta": "लगता", "Kundli": "कुंडली", "kundli": "कुंडली", "Mahadasha": "महादशा", "mahadasha": "महादशा",
    "Antardasha": "अंतर्दशा", "antardasha": "अंतर्दशा", "Dasha": "दशा", "dasha": "दशा", "Lagna": "लग्न",
    "lagna": "लग्न", "rashi": "राशि", "bhav": "भाव", "shaadi": "शादी", "vivah": "विवाह", "swami": "स्वामी",
    "Mercury": "बुध", "Venus": "शुक्र", "Jupiter": "बृहस्पति", "Saturn": "शनि", "Sun": "सूर्य", "Moon": "चंद्र",
    "Mars": "मंगल", "Rahu": "राहु", "Ketu": "केतु", "Aries": "मेष", "Taurus": "वृषभ", "Gemini": "मिथुन",
    "Leo": "सिंह", "Virgo": "कन्या", "Libra": "तुला", "Scorpio": "वृश्चिक", "Sagittarius": "धनु",
    "Capricorn": "मकर", "Aquarius": "कुंभ", "Pisces": "मीन", "Navamsa": "नवांश", "Nakshatra": "नक्षत्र",
}
_ROMAN_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]+(?![A-Za-z0-9])")


def devanagarize_hindi(text: str) -> tuple[str, int]:
    """Write the Roman-script Hindi words of a Hindi reply in Devanagari.
    English terms (career, job, house, lord) stay as they are. Pure."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        word = m.group(0)
        target = ROMAN_TO_DEVANAGARI.get(word) or ROMAN_TO_DEVANAGARI.get(word.lower()) \
            if word.lower() not in ("cancer",) else None
        if target:
            count += 1
            return target
        return word

    return _ROMAN_TOKEN.sub(repl, text or ""), count


__all__ = ["ROMAN_TO_DEVANAGARI", "devanagarize_hindi", "DEVANAGARI_LEXICON", "DEVANAGARI_PHRASES", "ENGLISH_VOCAB", "normalize_hinglish", "hinglish_issues"]
