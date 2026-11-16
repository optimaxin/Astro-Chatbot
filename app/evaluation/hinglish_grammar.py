"""Contextual Hinglish grammar and naturalness layer.

Root cause (2026-11 language audit): every existing check was WORD-level —
misspellings ("aapako"), Devanagari in Roman text, the language of the
reply. Nothing looked at agreement, postpositions or word combinations, so
"Mahadasha chal raha hai", "health issues la sakta hai", "domestic harmony ke
nuksan de sakta hai" and "property abroad se joda hota hai" all passed. The
model (a small Nemotron) translates the English knowledge passages word for
word ("can harm domestic harmony" -> "ke nuksan de sakta hai").

This layer, run after the language gate:

1. detect      sentence-level issues with their category (pure);
2. fix         deterministically ONLY where the context fixes the answer —
               a feminine noun directly followed by its masculine verb, the
               "aap … Mahadasha chal rahe hain" construction, "joda hota hai";
               a word list never decides a sentence's meaning;
3. correct     the sentences that still have issues, in ONE targeted model
               call that sees only those sentences (app.nodes.language_quality);
4. verify      each corrected sentence keeps every chart fact, adds no
               certainty, stays about the same length and has fewer issues —
               otherwise the original sentence is kept.

Hindi gets the deterministic agreement fixes only; English is detection-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LAYER_VERSION = "hinglish-grammar-1.0"


@dataclass(frozen=True)
class Issue:
    sentence: int
    category: str
    span: str
    hint: str
    severe: bool = False       # unreadable as written: a non-word, a broken construction


def _rx(p: str) -> re.Pattern[str]:
    return re.compile(p, re.I)


_FEM = ("dasha|mahadasha|antardasha|samasya|dikkat|pareshani|naukri|takleef|unnati|deri|growth|stability|"
        "aamdani|kamai|shaadi|ruchi|boli|sehat|zindagi|life|energy")
_TO_FEM = {"raha": "rahi", "sakta": "sakti", "rahega": "rahegi", "jayega": "jayegi", "hoga": "hogi",
           "milega": "milegi", "aayega": "aayegi", "banega": "banegi", "chalega": "chalegi", "badhega": "badhegi"}

# (category, pattern, hint, severe) — detection. Every pattern is wrong in
# every context it matches.
_DETECT: list[tuple[str, re.Pattern[str], str, bool]] = [
    ("agreement", _rx(rf"\b(?:{_FEM})\s+(?:(?:chal|aa|ho|mil|ban|badh|reh|rah)\s+)?"
                      r"(?:raha|sakta|rahega|jayega|hoga|milega|aayega|banega|chalega|badhega)\b"),
     "feminine noun with a masculine verb", False),
    ("agreement", _rx(r"\baap\s+(?:[A-Z]\w+\s+)?(?:Mahadasha|Antardasha|dasha)\b[^.?!]{0,60}?\s+chal\s+rah[ei]\s+hai"),
     "the Dasha is the subject: 'aapki … Mahadasha chal rahi hai'", False),
    ("agreement", _rx(r"\b(?:Mahadasha|Antardasha|dasha)\s+(?:[A-Z]\w+\s+)?ka\s+chal\s+raha\b"),
     "'Mercury ki Mahadasha chal rahi hai'", False),
    ("spelling", _rx(r"\b(?:antardash|mahadash)\b"), "Antardasha / Mahadasha", False),
    ("verb_form", _rx(r"\bjoda\s+hota\s+hai\b"), "'juda hota hai'", False),
    ("verb_form", _rx(r"\b(?:chahiye|hogi|hoga)\s+hai\b"), "double auxiliary", False),
    ("register", _rx(r"\baap\b(?:\s+\w+){0,2}\s+(?:chahte|sakte|karte|rahe)\s+ho\b"), "'aap … hain', not 'ho'", False),
    ("register", _rx(r"\baap\b[^.?!]{0,40}\b(?:karo|batao|socho|dekho|suno)\b"), "'aap' takes 'kijiye/bataiye'", False),
    ("collocation", _rx(r"\b(?:issues?|problems?|delays?|challenges?|tension|stress)\s+la\s+sakt[aie]\b"),
     "'X la sakta hai' with an English noun — 'X aa sakti hai' / 'X ho sakti hai'", False),
    ("collocation", _rx(r"\bka\s+effect\b"), "'X ka effect …' — 'X ke kaaran …' / 'X ka asar …'", False),
    ("collocation", _rx(r"\bnuksan\s+de\s+sakt[aie]\b|\bke\s+nuksan\b"), "'nuksan de sakta' — 'mein dikkat aa sakti hai'",
     False),
    ("collocation", _rx(r"\bsamay-samay\s+par\s+delays\b"), "'samay-samay par deri'", False),
    ("word_choice", _rx(r"\bsituate\s+(?:hai|hain)\b"), "just 'hai'", False),
    ("register", _rx(r"\b(?:prapt|praapt|pradaan|avashya|kartavyanishth|upyog|ityadi|vishleshan|kathinaiyon)\b"),
     "literary Hindi in casual Hinglish", False),
    ("non_word", _rx(r"\b(?:badhyaari|badhyari|bhayankar\s+karta)\b"), "not a Hindi word", True),
    ("copied_english", _rx(r"[\"“][A-Z][^\"”]{30,}[\"”]"), "quoted English passage inside Hinglish", False),
    # number: a singular participle with "hain", a plural one with "hai".
    ("agreement", _rx(r"\b(?:raha|deta|hota|karta|sakta|lagta|jata|aata)\s+hain\b"), "singular verb with 'hain'", False),
    ("agreement", _rx(r"\b(?:aapka|aapki|mera|meri|uska|uski|iska|iski)\s+\w+(?:\s+\w+)?\s+hain\b(?=\s*[.!?।]|\s*$)"),
     "singular subject with 'hain'", False),
    ("agreement", _rx(r"\b(?:rahe|dete|hote|karte|sakte|poochte|lagte|jaate|chahte)\s+hai\b"),
     "plural verb with 'hai'", False),
    ("agreement", _rx(r"\b(?:khushiyan|dikkatein|samasyayein|problems|issues|opportunities|zimmedariyan)\s+"
                      r"(?:(?:aa|ho|ban|mil)\s+sakta\s+hai|aayenge|honge|milenge)\b"), "feminine plural subject", False),
    ("agreement", _rx(r"\b(?:mann|career|business|partner|kaam|ghar|confidence|swabhaav|nature)\s+(?:\w+\s+)?"
                      r"(?:aa|ho|ban|badh|chal|reh|rah)\s+(?:sakti|rahi|rahegi)\b"), "masculine noun with a feminine verb",
     False),
    ("agreement", _rx(r"\b(?:boli|ruchi|zarurat|information|jaankari)\b(?:\s+\w+){1,3}?\s+(?:ho\s+sakta|hoga|rahega|"
                      r"badh\s+sakta)\b"), "feminine noun with a masculine verb", False),
    ("agreement", _rx(r"\b(?:Mahadasha|Antardasha|dasha)\b(?!\s+(?:mein|me|se|par|ko|ka|ki|ke)\b)(?:\s+\w+){1,3}?\s+"
                      r"(?:rahega|hoga|chalega)\b"), "the Dasha is feminine: 'rahegi'", False),
    ("postposition", _rx(r"\b(?:ka|ke)\s+zarurat\b|\baapko\s+madad\s+kar"), "'ki zarurat' / 'aapki madad'", False),
    ("postposition", _rx(r"\bto\s+[A-Z][a-z]+\s+\d{4}\s+tak\b"), "'… se … tak'", False),
    ("register", _rx(r"\b(?:tum|tumhara|tumhari|tumhare|tumhe|tumko)\b"), "use 'aap' throughout", False),
    ("register", _rx(r"\baap\b[^.?!]{0,40}\bhongi\b"), "does not assume the user's gender", False),
    ("register", _rx(r"\b(?:sthit|shatruon|vijay\s+dila\w*)\b"), "literary Hindi in casual Hinglish", False),
]

# Grammatical gender of nouns as Hinglish speakers use them, for possessive
# agreement ("aapki business" -> "aapke business mein", "aap ka kundli" ->
# "aapki kundli"). The noun decides; the following word decides the oblique.
_NOUN_GENDER = {**{w: "f" for w in (
    "dasha mahadasha antardasha samasya dikkat pareshani naukri takleef unnati deri growth stability aamdani kamai "
    "shaadi ruchi boli sehat zindagi life energy kundli information jaankari zarurat baat soch health personality "
    "family opportunity details history chinta madad").split()},
    **{w: "m" for w in (
        "career business ghar partner sawaal swabhaav swabhav lagna confidence mann kaam samay field yog asar fayda "
        "nuksan rishta mauka faisla chart plan result focus mahaul sambandh swasthya bhavishya future nature "
        "swagat").split()}}
_POSTPOSITION = re.compile(r"^(?:mein|me|se|par|ko|ke|ki|ka|tak|liye)\b", re.I)
_POSSESSIVE = re.compile(r"\b(aap\s+ka|aap\s+ki|aapka|aapki|aapke|mera|meri|mere|uska|uski|uske|iska|iski|iske|apna|"
                         r"apni|apne)(\s+(?:birth\s+|(?:Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu|"
                         r"Surya|Chandra|Mangal|Budh|Guru|Brihaspati|Shukra|Shani)\s+"
                         r"(?=(?:maha|antar)?dasha\b))?)([a-z]+)\b(\s+\S+)?", re.I)
# The planet is skipped only before a Dasha word: "aapka Rahu Mahadasha" (live
# browser test) is decided by Mahadasha, but in "aapka Mars kundli mein" the
# noun after the planet is not what the possessive belongs to.
_STEMS = {"aap ka": "aapk", "aap ki": "aapk", "aapka": "aapk", "aapki": "aapk", "aapke": "aapk", "mera": "mer",
          "meri": "mer", "mere": "mer", "uska": "usk", "uski": "usk", "uske": "usk", "iska": "isk", "iski": "isk",
          "iske": "isk", "apna": "apn", "apni": "apn", "apne": "apn"}


def _possessive_form(match: re.Match[str]) -> str | None:
    """The correct possessive for this noun in this position, or None."""
    gender = _NOUN_GENDER.get(match.group(3).lower())
    if gender is None:
        return None
    current = " ".join(match.group(1).lower().split()).replace(" ", "")
    stem = _STEMS[" ".join(match.group(1).lower().split())]
    nxt = (match.group(4) or "").strip()
    if gender == "f":
        return stem + "i"
    if current.endswith("e"):
        return None          # "aapke sawaal" may be plural: the noun alone cannot decide
    return stem + ("e" if _POSTPOSITION.match(nxt) else "a")


def _possessive_issues(sentence: str) -> list[str]:
    out = []
    for m in _POSSESSIVE.finditer(sentence):
        right = _possessive_form(m)
        if right and right != " ".join(m.group(1).lower().split()).replace(" ", ""):
            out.append(m.group(0).strip())
    return out

_HINDI_DETECT: list[tuple[str, re.Pattern[str], str, bool]] = [
    ("agreement", re.compile(r"(?:महादशा|अंतर्दशा|दशा)\s+चल\s+रहा\s+है"), "दशा is feminine: चल रही है", False),
    ("agreement", re.compile(r"(\d)(?:था|ठा)\s+भाव\s+(?=का|की|के|में|से|पर)"), "oblique: चौथे भाव का", False),
]


def detect(text: str, language: str) -> list[Issue]:
    from app.evaluation.chart_claims import split_sentences

    rules = _DETECT if language == "hinglish" else _HINDI_DETECT if language == "hindi" else []
    issues: list[Issue] = []
    for i, sentence in enumerate(split_sentences(text or "")):
        for category, pattern, hint, severe in rules:
            for match in pattern.finditer(sentence):
                issues.append(Issue(i, category, match.group(0), hint, severe))
        if language == "hinglish" and _english_only(sentence):
            # Severe: if the one correction pass cannot make it Hinglish, the
            # sentence is dropped. A pasted knowledge passage ("Body parts
            # ruled: bones, teeth, …") reached the browser this way.
            issues.append(Issue(i, "copied_english", sentence[:40], "an English sentence inside Hinglish", True))
        if language == "hinglish":
            issues += [Issue(i, "agreement", span, "possessive must agree with its noun") for span in
                       _possessive_issues(sentence)]
            from app.prompts.hinglish_lexicon import suspect_words

            # A slip of a real word ("assar", "aspekt", "Mahadashi"): detected
            # here, rewritten only by the contextual correction call.
            issues += [Issue(i, "non_word", word, f"not a real word as spelled (close to '{anchor}'); write the "
                             "word this sentence means, spelled correctly") for word, anchor in suspect_words(sentence)]
            if re.match(r"\s*[a-z]", sentence):
                issues.append(Issue(i, "capitalization", sentence[:15], "sentence starts in lowercase"))
    return issues


_HINDI_WORDS = _rx(r"\b(?:hai|hain|mein|ka|ki|ke|ko|se|aur|aap|aapka|aapki|aapke|aapko|ho|hota|hoti|sakta|sakti|"
                   r"sakte|yeh|ye|jo|isliye|lekin|par|tak|bhi|nahi|raha|rahi|rahe)\b")


_LABEL_LIST = re.compile(r"^\s*[A-Z][A-Za-z ]{2,30}:\s+\S+(?:\s*,\s*[^,]+){2,}")


def _english_only(sentence: str) -> bool:
    """A real English sentence (12+ words), or a short "Label: a, b, c" list
    copied from a knowledge passage, with no Hindi word in it."""
    if _HINDI_WORDS.search(sentence):
        return False
    words = re.findall(r"[A-Za-z']+", sentence)
    return len(words) >= 12 or (len(words) >= 6 and bool(_LABEL_LIST.match(sentence)))


# ── deterministic, context-bound fixes ────────────────────────────────
def _fem_verb(m: re.Match[str]) -> str:
    return re.sub(r"\b(raha|sakta|rahega|jayega|hoga|milega|aayega|banega|chalega|badhega)\b",
                  lambda v: _match_case(v.group(1), _TO_FEM[v.group(1).lower()]), m.group(0))


def _match_case(original: str, new: str) -> str:
    return new[:1].upper() + new[1:] if original[:1].isupper() else new


def _fix_possessive(m: re.Match[str]) -> str:
    right = _possessive_form(m)
    current = " ".join(m.group(1).lower().split()).replace(" ", "")
    if not right or right == current:
        return m.group(0)
    return f"{_match_case(m.group(1), right)}{m.group(2)}{m.group(3)}{m.group(4) or ''}"


_LOWER_START = re.compile(r"(\A\s*|[.!?]\s+|\n\s*)([a-z])")
_HINDI_OBLIQUE = {"1": "पहले", "2": "दूसरे", "3": "तीसरे", "4": "चौथे", "6": "छठे"}

_FIXES: dict[str, list[tuple[re.Pattern[str], object]]] = {
    "hinglish": [
        (_rx(r"\bantardash\b"), "Antardasha"),
        (_rx(r"\bmahadash\b"), "Mahadasha"),
        # "Abhi aap Mercury Mahadasha chal rahi hai" -> "Abhi aapki Mercury Mahadasha chal rahi hai"
        (_rx(r"\baap\s+((?:[A-Z]\w+\s+)?(?:Mahadasha|Antardasha|dasha)\b[^.?!]{0,60}?)\s+chal\s+rah[ei]\s+hain?\b"),
         lambda m: f"{'Aapki' if m.group(0)[0] == 'A' else 'aapki'} {m.group(1)} chal rahi hai"),
        # feminine noun DIRECTLY followed by its verb: the noun decides the form.
        (_rx(rf"\b(?:{_FEM})\s+(?:(?:chal|aa|ho|mil|ban|badh|reh|rah)\s+)?"
             r"(?:raha|sakta|rahega|jayega|hoga|milega|aayega|banega|chalega|badhega)\b"), _fem_verb),
        (_rx(r"\bjoda\s+hota\s+hai\b"), "juda hota hai"),
        (_rx(r"\bchahiye\s+hai\b"), "chahiye"),
        (_rx(r"\b(hogi|hoga)\s+hai\b"), lambda m: m.group(1)),
        (_rx(r"\bsituate\s+(hai|hain)\b"), lambda m: m.group(1)),
        # "samay-samay par delays ho sakte hain" -> "samay-samay par deri ho sakti hai"
        (_rx(r"\bsamay-samay\s+par\s+delays\s+ho\s+sakte\s+hain\b"), "samay-samay par deri ho sakti hai"),
        (_rx(r"\b(aap\b(?:\s+\w+){0,2}\s+(?:chahte|sakte|karte|rahe))\s+ho\b"), lambda m: f"{m.group(1)} hain"),
        # a plural participle never takes "hai": "log … poochte hai" -> "poochte hain"
        (_rx(r"\b(rahe|dete|hote|karte|sakte|poochte|lagte|jaate|chahte)\s+hai\b"), lambda m: f"{m.group(1)} hain"),
        # a singular possessive subject ("aapka Lagna Pisces hain") takes "hai"
        (_rx(r"\b((?:aapka|aapki|mera|meri|uska|uski|iska|iski)\s+\w+(?:\s+\w+)?)\s+hain\b(?=\s*[.!?।]|\s*$)"),
         lambda m: f"{m.group(1)} hai"),
        # feminine plural subject directly before its verb
        (_rx(r"\b(khushiyan|dikkatein|samasyayein|problems|issues|opportunities|zimmedariyan)\s+"
             r"(aa|ho|ban|mil)\s+sakta\s+hai\b"), lambda m: f"{m.group(1)} {m.group(2)} sakti hain"),
        (_rx(r"\b(khushiyan|dikkatein|samasyayein|opportunities|zimmedariyan)\s+(aayenge|honge|milenge)\b"),
         lambda m: f"{m.group(1)} {m.group(2)[:-1]}i"),
        (_rx(r"\b(?:ka|ke)\s+zarurat\b"), "ki zarurat"),
        (_rx(r"\baapko\s+madad\s+kar"), "aapki madad kar"),
        (_rx(r"\bto\s+([A-Z][a-z]+\s+\d{4}\s+tak)\b"), lambda m: f"se {m.group(1)}"),
        (_rx(r"\b(aap\b[^.?!]{0,40}\b)hongi\b"), lambda m: f"{m.group(1)}honge"),
        (_POSSESSIVE, lambda m: _fix_possessive(m)),
        # a bubble or sentence that starts in lowercase ("aur Shani bones …",
        # live browser test). Roman script, so the first letter is capitalised.
        (_LOWER_START, lambda m: f"{m.group(1)}{m.group(2).upper()}"),
    ],
    "hindi": [
        (re.compile(r"((?:महादशा|अंतर्दशा|दशा)\s+चल\s+)रहा(\s+है)"), lambda m: f"{m.group(1)}रही{m.group(2)}"),
        (re.compile(r"(\d)(?:था|ठा)(\s+भाव\s+)(?=का|की|के|में|से|पर)"),
         lambda m: f"{_HINDI_OBLIQUE.get(m.group(1), m.group(0)[:-len(m.group(2))])}{m.group(2)}"),
    ],
}


def correct_deterministic(text: str, language: str) -> tuple[str, list[str]]:
    """Apply only the fixes whose context makes the answer unambiguous."""
    out, fixes = text or "", []
    if language == "hinglish":
        from app.prompts.hinglish import normalize_hinglish

        out, word_fixes = normalize_hinglish(out)
        fixes += word_fixes
    for pattern, repl in _FIXES.get(language, []):
        out, n = pattern.subn(repl, out)
        if n:
            fixes.append(f"{pattern.pattern[:28]}x{n}")
    return out, fixes


# ── verification of a model-corrected sentence ────────────────────────
_CERTAIN = _rx(r"\b(?:pakka|pakki|definitely|zaroor|avashya|nishchit|guarantee)\b|निश्चित|पक्का|पक्की|अवश्य")


@dataclass
class Verdict:
    accepted: bool
    reason: str = ""
    details: dict = field(default_factory=dict)


def verify_sentence(original: str, corrected: str, language: str) -> Verdict:
    from app.evaluation.conciseness import _chart_tokens

    if not corrected or not corrected.strip():
        return Verdict(False, "empty")
    if _chart_tokens(original) != _chart_tokens(corrected):
        return Verdict(False, "chart facts changed",
                       {"before": sorted(_chart_tokens(original)), "after": sorted(_chart_tokens(corrected))})
    if _CERTAIN.search(corrected) and not _CERTAIN.search(original):
        return Verdict(False, "added certainty")
    ratio = len(corrected.split()) / max(1, len(original.split()))
    if not 0.5 <= ratio <= 1.6:
        return Verdict(False, f"length ratio {ratio:.2f}")
    if language == "hinglish" and re.search(r"[ऀ-ॿ]", corrected):
        return Verdict(False, "Devanagari in Hinglish")
    if len(detect(corrected, language)) >= max(1, len(detect(original, language))):
        return Verdict(False, "not better")
    return Verdict(True)


__all__ = ["Issue", "detect", "correct_deterministic", "verify_sentence", "Verdict", "LAYER_VERSION"]
