"""One astrology vocabulary per reply language.

The same table drives (1) the TERMINOLOGY line in the system prompt, so the
model is told which form of each term to use, and (2) a deterministic clean-up
of the half-translated forms models produce when they switch script mid-word
("महाdasha", "अन्तर्दasha" — both seen in production), applied to every reply
whatever model or fallback route wrote it.

Planet and sign *aliases* for chart validation live in
`app.evaluation.chart_claims`; this module is about how a reply should READ.
"""

from __future__ import annotations

import re

# concept -> (English, Hindi, Hinglish)
TERMS: dict[str, tuple[str, str, str]] = {
    "mahadasha": ("Mahadasha", "महादशा", "Mahadasha"),
    "antardasha": ("Antardasha", "अंतर्दशा", "Antardasha"),
    "dasha": ("Dasha", "दशा", "Dasha"),
    "lagna": ("Ascendant (Lagna)", "लग्न", "Lagna"),
    "rashi": ("Moon sign (Rashi)", "राशि", "Rashi"),
    "nakshatra": ("Nakshatra", "नक्षत्र", "Nakshatra"),
    "house": ("house", "भाव", "house"),
    "house_lord": ("house lord", "भावेश / स्वामी", "lord"),
    "1st_house": ("1st house", "प्रथम भाव (लग्न)", "1st house"),
    "2nd_house": ("2nd house", "द्वितीय भाव", "2nd house"),
    "4th_house": ("4th house", "चतुर्थ भाव", "4th house"),
    "5th_house": ("5th house", "पंचम भाव", "5th house"),
    "6th_house": ("6th house", "षष्ठ भाव", "6th house"),
    "7th_house": ("7th house", "सप्तम भाव", "7th house"),
    "10th_house": ("10th house", "दशम भाव", "10th house"),
    "11th_house": ("11th house", "एकादश भाव", "11th house"),
    "12th_house": ("12th house", "द्वादश भाव", "12th house"),
    "navamsa": ("Navamsa (D9)", "नवांश (D9)", "Navamsa (D9)"),
    "yoga": ("Yoga", "योग", "Yoga"),
    "dosha": ("Dosha", "दोष", "Dosha"),
    "sun": ("Sun", "सूर्य", "Surya (Sun)"),
    "moon": ("Moon", "चंद्र", "Chandra (Moon)"),
    "mars": ("Mars", "मंगल", "Mangal"),
    "mercury": ("Mercury", "बुध", "Budh"),
    "jupiter": ("Jupiter", "बृहस्पति (गुरु)", "Jupiter"),
    "venus": ("Venus", "शुक्र", "Shukra"),
    "saturn": ("Saturn", "शनि", "Shani"),
    "rahu": ("Rahu", "राहु", "Rahu"),
    "ketu": ("Ketu", "केतु", "Ketu"),
}
_COLUMN = {"english": 0, "hindi": 1, "hinglish": 2}

# The concepts named in the prompt's terminology line (the rest are obvious).
_PROMPT_CONCEPTS = ("mahadasha", "antardasha", "lagna", "7th_house", "10th_house", "11th_house",
                    "mercury", "venus", "jupiter", "saturn", "mars", "moon", "sun")


def term(concept: str, language: str) -> str:
    return TERMS[concept][_COLUMN.get(language, 0)]


def terminology_line(language: str) -> str:
    """One compact line for the system prompt."""
    col = _COLUMN.get(language, 0)
    pairs = ", ".join(TERMS[c][col] for c in _PROMPT_CONCEPTS)
    if language == "hindi":
        return ("TERMINOLOGY (Hindi): use these Devanagari forms consistently — " + pairs +
                ". House numbers may be written as '7वां भाव'. Never write a word half in Devanagari and "
                "half in Latin (wrong: 'महाdasha'). Do not add English glosses in brackets.")
    if language == "hinglish":
        return ("TERMINOLOGY (Hinglish): use these Roman-script forms consistently — " + pairs +
                ". Keep house numbers as '7th house'. No Devanagari at all, not even in brackets.")
    return ("TERMINOLOGY (English): use these forms consistently — " + pairs +
            ". Sanskrit terms such as Mahadasha, Antardasha and Nakshatra stay as they are; do not add "
            "Hindi glosses.")


# Half-translated tokens -> the clean Devanagari word. Matched case-insensitively.
_HYBRID_FIXES: tuple[tuple[re.Pattern, str], ...] = (
    # "महाdasha" and "महाद"+"asha"; "अन्तर्दasha" (both seen in production).
    (re.compile(r"महा\s?(?:d|द)[aā]sh?a", re.I), "महादशा"),
    (re.compile(r"(?:अन्तर्|अंतर्|अन्तर|अंतर)\s?(?:d|द)[aā]sh?a", re.I), "अंतर्दशा"),
    (re.compile(r"maha\s?दशा", re.I), "महादशा"),
    (re.compile(r"antar\s?दशा", re.I), "अंतर्दशा"),
    (re.compile(r"अन्तर्दशा"), "अंतर्दशा"),
)
# A Devanagari gloss in brackets inside a Latin-script reply: "10th house (करियर)".
_DEVANAGARI_GLOSS = re.compile(r"\s*\((?=[^()]*[ऀ-ॿ])[^()a-zA-Z]{1,40}\)")
# "Sun, Mercury aur Jupiter (सूर्य, बुध बृहस्पति)" — a gloss broken across the
# reply's own sentence split is caught by the generic Devanagari-run rule.
_HYBRID_WORD = re.compile(r"[ऀ-ॿ]+[a-zA-Z]+|[a-zA-Z]+[ऀ-ॿ]+")


def normalize_terms(text: str, language: str) -> tuple[str, list[str]]:
    """Deterministic clean-up. Returns (text, list of fixes applied). Pure.

    Only rewrites forms that are wrong in every language (hybrid tokens) or
    wrong for the target script (Devanagari glosses in a Latin reply). It
    never translates a correct term.
    """
    fixes: list[str] = []
    out = text or ""
    for rx, clean in _HYBRID_FIXES:
        out, n = rx.subn(clean, out)
        if n:
            fixes.append(f"hybrid->{clean}")
    if language in ("english", "hinglish"):
        out, n = _DEVANAGARI_GLOSS.subn("", out)
        if n:
            fixes.append(f"removed_devanagari_glosses:{n}")
    if language == "hinglish" and _DEVANAGARI_ANY.search(out) and (_devanagari_share(out) <= 0.15 or _all_known(out)):
        # Hinglish is Latin-only. Written out in Roman script: a Devanagari word
        # the lexicon knows ("आपको" -> "aapko", "पार्टनर" -> "partner"), and the
        # tail of a mid-word script switch ("vyavसाय" -> "vyavsaay"). A free-
        # standing word the lexicon does not know is left alone, so the
        # language repair rewrites the sentence: guessing its spelling is
        # exactly what produced "paartanar" and "maanasik kanekshan".
        from app.prompts.hinglish import DEVANAGARI_PHRASES
        from app.prompts.transliterate import _DEVANAGARI_WORD, romanize_word

        for phrase, roman in DEVANAGARI_PHRASES.items():
            out = out.replace(phrase, roman)

        def roman(m: re.Match) -> str:
            spelled, known = romanize_word(m.group(0))
            glued = (m.start() > 0 and out[m.start() - 1].isascii() and out[m.start() - 1].isalpha())
            return spelled if known or glued else m.group(0)

        before = out
        out = _DEVANAGARI_WORD.sub(roman, out)
        if out != before:
            fixes.append("transliterated_devanagari")
    if language in ("hinglish", "english"):
        from app.prompts.hinglish import normalize_hinglish

        out, vocab_fixes = normalize_hinglish(out)
        fixes.extend(vocab_fixes)
    if language == "hindi":
        # "लिएharmony", "संभावनाtraditionally": the words are fine, the space
        # between them is missing.
        out, n = _MISSING_SPACE.subn(" ", out)
        if _devanagari_share(out) >= 0.1:
            from app.prompts.hinglish import devanagarize_hindi

            out, converted = devanagarize_hindi(out)
            if converted:
                fixes.append(f"devanagarized_roman_hindi:{converted}")
        if n:
            fixes.append(f"split_script_joins:{n}")
    return out, fixes


_DEVANAGARI_ANY = re.compile(r"[ऀ-ॿ]")


def _all_known(text: str) -> bool:
    from app.prompts.transliterate import unknown_words

    return not unknown_words(text)


def _devanagari_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha() or "ऀ" <= ch <= "ॿ"]
    return sum(1 for ch in letters if "ऀ" <= ch <= "ॿ") / len(letters) if letters else 0.0
_MISSING_SPACE = re.compile(r"(?<=[ऀ-ॿ])(?=[A-Za-z]{3,})|(?<=[A-Za-z]{3})(?=[क-ह])")


def hybrid_tokens(text: str) -> list[str]:
    """Words mixing Devanagari and Latin letters (should be none)."""
    return _HYBRID_WORD.findall(text or "")


__all__ = ["TERMS", "term", "terminology_line", "normalize_terms", "hybrid_tokens"]
