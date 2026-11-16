"""Deterministic reply-quality checks and the safe fixes for them.

Runs on every reply, after chart/language validation, inside
`validate_reply_node` — so every route (primary, fallback, repair) passes
through it before a user sees anything. No model calls here: each check is a
pure function, cheap enough to run on every turn.

| metric            | check                                              | safe fix              |
|-------------------|----------------------------------------------------|-----------------------|
| follow_up         | reply ends with a question it did not need          | trailing question cut |
| repetition        | a sentence repeats an earlier one                   | later copy removed    |
| terminology       | half-Devanagari/half-Latin tokens, Devanagari glosses | normalised          |
| machinery         | "chart data", "retrieved knowledge", "context" …    | repair (one regen)    |
| filler            | vague verdicts standing in for interpretation       | reported              |
| fragmentation     | more than one message for one reply                 | n/a (single message)  |

Every production screenshot issue from 2026-11 maps to a row above; see
tests/test_screenshot_regressions.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.evaluation.chart_claims import split_sentences
from app.prompts.terminology import hybrid_tokens, normalize_terms

VALIDATOR_VERSION = "kq-validator-1.1"

# Intents where a clarifying question can be the right reply.
_CLARIFY_INTENTS = {"unknown", "smalltalk"}

# A question that probes the user's life/plans/preferences instead of
# answering. These are the endings seen in production.
_PROBE = re.compile(
    r"\b(?:what kind of|are you (?:looking|considering|planning|thinking)|do you have (?:any|a) (?:specific|"
    r"particular)|would you like|anything else|any other question|what are your|what do you (?:do|want)|"
    r"which (?:role|field|company)|is there (?:anything|something)|your (?:current )?plans|"
    r"kya (?:aap|tum)|koi (?:dusra|doosra|aur|specific|vishesh)|kaunsa|soch rahe|dekh rahe|chahte (?:ho|hain)|"
    r"poochna chah)\b"
    r"|क्या (?:आप|तुम)|कोई (?:विशेष|और|दूसरा)|योजना|सूझ रहा",
    re.I)

# "Koi aur chart fact poochni ho to pooch sakti hain!", "Feel free to ask
# anything else." — an invitation that ends the reply instead of the answer.
_CLOSING_OFFER = re.compile(
    # Not "If you have any symptoms, please see a doctor." — the benchmark
    # found that safety line removed as an "offer".
    r"\b(?:feel free to ask|let me know if|if you (?:have|want|would like) (?:any (?:other |more |further )?"
    r"(?:questions?|doubts?|queries)|to know)|happy to (?:help|explain)|"
    r"koi (?:aur|dusra|doosra)[^.!?।]{0,40}pooch|pooch (?:sakte|sakti) (?:hain|ho)|batana chahein to|"
    r"aur kuch (?:jaanna|poochna)|please share your (?:exact )?(?:birth|date of birth)|"
    r"if you have a specific (?:timing|future|question))\b|कुछ और (?:जानना|पूछना)|पूछ सकते हैं",
    re.I)


_PROPER = {
    "sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn", "rahu", "ketu", "aries", "taurus", "gemini",
    "cancer", "leo", "virgo", "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces", "mahadasha",
    "antardasha", "dasha", "lagna", "kundli", "navamsa", "vedic", "vimshottari", "tredev", "astro", "yoga", "dosha",
    "nakshatra", "rashi", "surya", "chandra", "mangal", "budh", "guru", "shukra", "shani", "ascendant", "d9",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "i", "sade", "sati", "manglik", "kaal", "sarp", "gajakesari", "budhaditya", "raj",
    "guna", "milan", "panchang", "muhurat", "ashtakoot",
}
_TITLE_RUN = re.compile(r"(?<=[a-z,;:] )((?:[A-Z][a-z]+)(?:\s+(?:[A-Z][a-z]+)){1,3})")


def lowercase_copied_headings(text: str) -> tuple[str, int]:
    """'…aap Charitable Inclination aur Beautiful Clothing…' -> lower case,
    when no word in the run is a proper noun of the chart (planets, signs,
    Mahadasha, months…). Only runs that start mid-sentence are touched."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        words = m.group(1).split()
        if any(w.lower() in _PROPER for w in words):
            return m.group(1)
        count += 1
        return m.group(1).lower()

    return _TITLE_RUN.sub(repl, text or ""), count


def _devanagari_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha() or "ऀ" <= ch <= "ॿ"]
    return sum(1 for ch in letters if "ऀ" <= ch <= "ॿ") / len(letters) if letters else 0.0


_CAPITALISED = ("Mahadasha", "Antardasha", "Dasha", "Lagna", "Kundli", "Navamsa", "Nakshatra", "Vedic", "Sun",
                "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu", "Shani", "Shukra", "Budh",
                "Guru", "Surya", "Chandra", "Mangal", "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
                "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces", "January", "February",
                "March", "April", "June", "July", "August", "September", "October", "November", "December", "I")
_CAP_RE = re.compile(r"\b(" + "|".join(w.lower() for w in _CAPITALISED) + r")\b")


def restore_capitalization(text: str) -> tuple[str, bool]:
    """A Latin-script reply written entirely in lower case ("your lagna is
    aquarius…", live browser test round 5): capitalise sentence starts and the
    proper nouns of the chart. Untouched when the reply already has capitals."""
    latin = [ch for ch in text or "" if ch.isascii() and ch.isalpha()]
    if len(latin) < 20 or any(ch.isupper() for ch in latin):
        return text, False
    out = _CAP_RE.sub(lambda m: next(w for w in _CAPITALISED if w.lower() == m.group(1)), text)
    out = re.sub(r"(^|[.!?]\s+|\n\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    return out, True


def closing_offers(text: str) -> list[int]:
    """Trailing sentences that only invite another question."""
    sentences = split_sentences(text)
    out: list[int] = []
    for i in range(len(sentences) - 1, 0, -1):
        if _CLOSING_OFFER.search(sentences[i]) and len(sentences[i]) < 160:
            out.append(i)
        else:
            break
    return sorted(out)


# A reply the model stopped mid-sentence (output limit, a dropped stream):
# the unfinished tail is cut back to the last complete sentence.
_SENTENCE_END = re.compile(r"[.!?।…)\]\"'”’](?:\s*[\U0001F300-\U0001FAFF\u2600-\u27BF])*\s*$")


def trim_unfinished_tail(text: str) -> tuple[str, bool]:
    stripped = (text or "").rstrip()
    if not stripped or _SENTENCE_END.search(stripped):
        return text, False
    cut = max(stripped.rfind(p) for p in (". ", "। ", "! ", "? ", ".\n", "।\n", "!\n", "?\n"))
    if cut < 0 or cut < len(stripped) * 0.4:
        return text, False        # too little would remain: keep it as it is
    return stripped[:cut + 1].rstrip(), True


_MACHINERY = re.compile(
    r"\b(?:chart data|retrieved knowledge|retrieved (?:passages|documents)|knowledge base|provided context|"
    r"the context (?:provided|given)|the documents?|my (?:instructions|system prompt)|the data (?:provided|given)|"
    r"data mein|knowledge (?:nahi|bhi) |available data|chart data diya|data diya gaya|data provided|"
    r"the passages?|passages? (?:note|say|says|mention|state|kehta|kehte|batata)|passage yeh|the retrieved|"
    # Self-description instead of an answer (round-4 benchmark: "क्या मेरी शादी हो
    # चुकी है?" -> "मैं एनवीआईडीआईए के शोधकर्ताओं द्वारा बनाया गया एक भाषा मॉडल हूं").
    r"(?:a|an) (?:large )?language model|created by nvidia|trained by|bhasha model|"
    r"information (?:provided|given))|चार्ट डेटा|डेटा में|भाषा मॉडल|एनवीआईडीआईए|" + r"(?:" + r"|".join((
        # The model naming itself or its maker (HQB final run: "मैं नेमोट्रॉन हूं…",
        # "नहीं, मेरा विकास NVIDIA के शोधकर्ताओं द्वारा किया गया है।").
        r"\bnvidia\b", r"\bnemotron\b", r"नेमोट्रॉन", r"शोधकर्ताओं\s+द्वारा", r"मॉडल\s+हूँ", r"मॉडल\s+हूं",
        r"\blanguage\s+model\b", r"\bai\s+model\b",
        # A knowledge source's breadcrumb title quoted into the answer (HQB final
        # run: "10TH HOUSE LORD › Tenth house lord in the sixth house › Effects").
        r"›")) + r")",
    re.I)
_IDENTITY = re.compile(r"\bnvidia\b|\bnemotron\b|नेमोट्रॉन|एनवीआईडीआईए|शोधकर्ताओं\s+द्वारा|भाषा\s+मॉडल|"
                       r"\blanguage\s+model\b|\bbhasha\s+model\b", re.I)
# Sent only when every sentence of a reply is the model describing itself.
IDENTITY_REPLY = {
    "english": "I'm Tredev Astro's AI astrology assistant, and I answer from your own Kundli. Ask me anything about "
               "your chart.",
    "hinglish": "Main Tredev Astro ka AI astrology assistant hoon, aur aapki Kundli ke hisaab se jawab deta hoon. "
                "Aap apni Kundli ke baare mein kuch bhi pooch sakte hain.",
    "hindi": "मैं Tredev Astro का AI ज्योतिष सहायक हूँ और आपकी कुंडली के आधार पर जवाब देता हूँ। आप अपनी कुंडली "
             "के बारे में कुछ भी पूछ सकते हैं।",
}

# Vague verdicts. Reported (not removed): removing them could take real
# content with them; the prompt is what prevents them.
_FILLER = re.compile(
    r"\b(?:chances (?:look|are|seem) (?:average|okay|ok|decent|good)|theek[- ]?(?:thaak|thenek|thak)|"
    r"looks? (?:very )?promising|positive energy|everything will be (?:fine|ok)|good opportunities ahead|"
    r"bright future|sab (?:theek|accha) hoga)\b",
    re.I)


# A sentence about the assistant's own language ability ("Yes, I can answer in
# Hindi"). Right only when the user just asked to switch; in production one
# leaked into the NEXT answer as "नहीं, मैं हिंदी में जवाब दे सकता हूँ".
_LANGUAGE_META = re.compile(
    r"\b(?:i can|i'?ll|i will|main|mai)\b[^.?!।]{0,30}\b(?:reply|answer|respond|jawab|baat)\b[^.?!।]{0,20}"
    r"\b(?:hindi|english|hinglish)\b"
    r"|\b(?:hindi|english|hinglish)\s+(?:mein|me)\s+(?:jawab|reply|baat)\b[^.?!।]{0,25}"
    r"\b(?:sakta|sakti|karunga|karungi|dunga|dungi)\b"
    r"|(?:हिंदी|हिन्दी|अंग्रेज़ी|अंग्रेजी|इंग्लिश)\s+में\s+(?:जवाब|बात)[^.?!।]{0,25}(?:सकता|सकती|दूँगा|दूंगा|दूँगी|करूँगा)",
    re.I)


# "[K1] ke hisaab se, …" / "[K2] के अनुसार, …" lose their subject when the
# citation tag is stripped; the leftover phrase is removed.
_DANGLING_ATTRIBUTION = re.compile(
    r"(^|(?<=[.!?।]\s)|(?<=\n))\s*(?:(?:ke\s+hisaab\s+se|ke\s+anusaar|ke\s+mutabik|के\s+अनुसार|के\s+हिसाब\s+से)"
    # English "According to …" is only dangling when nothing follows it but a
    # comma ("According to , …"): "According to your chart, …" is a sentence
    # (the Hinglish benchmark caught "As per the …" being cut to "The …").
    r"\s*[,:]?|(?:according\s+to|as\s+per)\s*[,:])\s*(?=\S)", re.I)
# "As per the retrieved knowledge passages, X" -> "X": the attribution is machinery.
_MACHINERY_ATTRIBUTION = re.compile(
    r"(^|(?<=[.!?।]\s)|(?<=\n))\s*(?:as\s+per|according\s+to|based\s+on)\s+(?:the\s+)?(?:retrieved\s+|available\s+)?"
    r"(?:knowledge(?:\s+base)?|passages?|documents?|context|sources?|data)(?:\s+\w+){0,2}\s*,\s*", re.I)


def fix_dangling_attribution(text: str) -> tuple[str, int]:
    out, m = _MACHINERY_ATTRIBUTION.subn(r"\1", text)
    out, n = _DANGLING_ATTRIBUTION.subn(r"\1", out)
    n += m
    if n:
        out = re.sub(r"(^|[.!?।]\s+|\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    return out, n


def machinery_sentences(text: str) -> list[int]:
    return [i for i, s in enumerate(split_sentences(text)) if _MACHINERY.search(s)]


def language_meta_sentences(text: str) -> list[int]:
    return [i for i, s in enumerate(split_sentences(text)) if _LANGUAGE_META.search(s)]


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith(("?", "？"))


def _norm(sentence: str) -> str:
    return " ".join(re.sub(r"[^\w\sऀ-ॿ]", " ", sentence.lower()).split())


def trailing_questions(text: str) -> list[int]:
    """Indices of the consecutive question sentences that end the reply."""
    sentences = split_sentences(text)
    out: list[int] = []
    for i in range(len(sentences) - 1, -1, -1):
        if _is_question(sentences[i]):
            out.append(i)
        else:
            break
    return sorted(out)


def unneeded_follow_up(text: str, intent: str | None) -> list[int]:
    """Trailing questions that should not be there.

    A reply that is ONLY a question (a genuine clarification) is left alone,
    and so is any reply to an unclassifiable message. Otherwise a closing
    question after a real answer is removed — every production example of it
    was a probe about the user's plans, job or preferences.
    """
    sentences = split_sentences(text)
    tail = trailing_questions(text)
    if not tail or len(tail) == len(sentences) or (intent or "") in _CLARIFY_INTENTS:
        return []
    answered = len(sentences) - len(tail)
    if answered < 1:
        return []
    # A short clarification after a single short statement ("I need your
    # birth time for that. Could you share it?") is legitimate.
    probes = [i for i in tail if _PROBE.search(sentences[i])]
    if answered == 1 and not probes and len(sentences[0]) < 80:
        return []
    return tail


def repeated_sentences(text: str) -> list[int]:
    """Later sentences that repeat an earlier one (same normalised words)."""
    sentences = split_sentences(text)
    keys = [_norm(s) for s in sentences]
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    seen: set[str] = set()
    dupes: list[int] = []
    for i, key in enumerate(keys):
        # Short sentences only count when they loop ("मुझे खेद है।" ×40 in the
        # benchmark); a repeated "Yes." is not worth touching.
        if not key or (len(key.split()) < 4 and counts[key] < 3):
            continue
        if key in seen:
            dupes.append(i)
        seen.add(key)
    return dupes


def mentions_machinery(text: str) -> bool:
    return bool(_MACHINERY.search(text or ""))


def filler_phrases(text: str) -> list[str]:
    return [m.group(0) for m in _FILLER.finditer(text or "")]


def _drop(text: str, indices: set[int]) -> str:
    """Remove whole sentences while keeping the paragraph layout."""
    if not indices:
        return text
    sentences = split_sentences(text)
    targets = [sentences[i] for i in sorted(indices) if i < len(sentences)]
    out = text
    for sentence in reversed(targets):
        pos = out.rfind(sentence)
        if pos >= 0:
            out = out[:pos] + out[pos + len(sentence):]
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


@dataclass
class QualityReport:
    follow_up_removed: int = 0
    follow_up_kept: int = 0
    repeated_removed: int = 0
    restated_removed: int = 0
    date_ranges_removed: int = 0
    language_meta_removed: int = 0
    machinery_removed: int = 0
    terminology_fixes: list[str] = field(default_factory=list)
    hybrid_tokens: list[str] = field(default_factory=list)
    machinery: bool = False
    filler: list[str] = field(default_factory=list)
    messages: int = 1
    validator_version: str = VALIDATOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"validator_version": self.validator_version,
                "follow_up_removed": self.follow_up_removed,
                "follow_up_kept": self.follow_up_kept,
                "repeated_removed": self.repeated_removed,
                "restated_removed": self.restated_removed,
                "date_ranges_removed": self.date_ranges_removed,
                "language_meta_removed": self.language_meta_removed,
                "machinery_removed": self.machinery_removed,
                "terminology_fixes": self.terminology_fixes,
                "hybrid_tokens": self.hybrid_tokens[:5],
                "machinery": self.machinery,
                "filler": self.filler[:5],
                "messages": self.messages}


def assess(text: str, language: str | None, intent: str | None) -> QualityReport:
    """Measure without changing anything."""
    report = QualityReport()
    report.follow_up_removed = len(unneeded_follow_up(text, intent))
    report.repeated_removed = len(repeated_sentences(text))
    report.hybrid_tokens = hybrid_tokens(text)
    report.machinery = mentions_machinery(text)
    report.filler = filler_phrases(text)
    return report


def apply_safe_fixes(text: str, language: str | None, intent: str | None,
                     question: str | None = None, followup_allowed: bool | None = None,
                     previous_assistant: str | None = None) -> tuple[str, QualityReport]:
    """Fix what can be fixed without a model call. Never empties a reply:
    a fix that would leave less than one real sentence is skipped."""
    from app.prompts.language_policy import explicit_request

    report = QualityReport()
    out, report.terminology_fixes = normalize_terms(text, language or "english")
    if language == "hinglish" and re.search(r"[ऀ-ॿ]", out) and _devanagari_share(out) <= 0.15:
        # Only STRAY Devanagari words are romanised. A reply that is mostly
        # Devanagari is left to the language gate's rewrite: romanising a whole
        # Hindi answer produced "paramparik roop se … lanbe samay" (live
        # browser test, round 5).
        from app.prompts.hinglish import normalize_hinglish
        from app.prompts.transliterate import to_roman

        out, more = normalize_hinglish(to_roman(out))
        report.terminology_fixes += ["romanised_remaining_devanagari", *more]
    if language in ("english", "hinglish"):
        out, recased = restore_capitalization(out)
        if recased:
            report.terminology_fixes.append("restored_capitalization")
    # The chat bubble renders plain text: "**Budh Mahadasha**" would show its
    # asterisks (seen in the benchmark despite the no-markdown rule).
    out, bold = re.subn(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), out)
    if bold:
        report.terminology_fixes.append(f"markdown_bold:{bold}")
    if language in ("english", "hinglish"):
        out, headings = lowercase_copied_headings(out)
        if headings:
            report.terminology_fixes.append(f"lowercased_copied_headings:{headings}")
    out, dangling = fix_dangling_attribution(out)
    if dangling:
        report.terminology_fixes.append(f"dangling_attribution:{dangling}")

    if not explicit_request(question):
        meta = set(language_meta_sentences(out))
        if meta and len(split_sentences(out)) - len(meta) >= 1:
            out = _drop(out, meta)
            report.language_meta_removed = len(meta)

    dupes = set(repeated_sentences(out))
    if dupes and len(split_sentences(out)) - len(dupes) >= 1:
        out = _drop(out, dupes)
        report.repeated_removed = len(dupes)

    # Conciseness (app.evaluation.conciseness): restatements that add no chart
    # fact, and Dasha dates already given.
    from app.evaluation.conciseness import drop_repeated_date_ranges, restated_sentences

    restated = set(restated_sentences(out))
    if restated and len(split_sentences(out)) - len(restated) >= 1:
        out = _drop(out, restated)
        report.restated_removed = len(restated)
    out, report.date_ranges_removed = drop_repeated_date_ranges(out, previous_assistant, question)

    tail = set(unneeded_follow_up(out, intent))
    if (intent or "") not in _CLARIFY_INTENTS:
        tail |= set(closing_offers(out))
        # With a per-turn follow-up decision, EVERY trailing question is under
        # it — also the short "statement + question" a clarification would be
        # ("theek hai" -> "Achha, samajh aaya?"; "…dekh rahe hain?" probes).
        if followup_allowed is not None and len(trailing_questions(out)) < len(split_sentences(out)):
            tail |= set(trailing_questions(out))
    if followup_allowed and tail:
        # One permitted follow-up survives: the LAST trailing question, if it
        # offers a specific next topic and is not a repeat (app.prompts.followup).
        from app.prompts.followup import is_good_followup

        sentences = split_sentences(out)
        last = max(tail)
        answer = " ".join(s for i, s in enumerate(sentences) if i not in tail)
        if _is_question(sentences[last]) and is_good_followup(sentences[last], previous_assistant, answer, question):
            tail.discard(last)
            report.follow_up_kept = 1
    if tail and len(split_sentences(out)) - len(tail) >= 1:
        out = _drop(out, tail)
        report.follow_up_removed = len(tail)

    out, trimmed = trim_unfinished_tail(out)
    if trimmed:
        report.terminology_fixes.append("trimmed_unfinished_tail")

    # Last resort for machinery talk the repair pass did not remove: drop the
    # sentence when a real answer remains without it.
    leaky = set(machinery_sentences(out))
    if leaky and len(split_sentences(out)) - len(leaky) >= 2:
        out = _drop(out, leaky)
        report.machinery_removed = len(leaky)
    # The model's own name or maker never reaches the user: such a sentence goes
    # even from a short reply, and a reply that is nothing else is replaced.
    sentences = split_sentences(out)
    identity = {i for i, s in enumerate(sentences) if _IDENTITY.search(s)}
    if identity:
        out = _drop(out, identity) if len(sentences) > len(identity) else \
            IDENTITY_REPLY.get(language or "english", IDENTITY_REPLY["english"])
        report.machinery_removed += len(identity)

    report.hybrid_tokens = hybrid_tokens(out)
    report.machinery = mentions_machinery(out)
    report.filler = filler_phrases(out)
    return out, report


__all__ = ["VALIDATOR_VERSION", "QualityReport", "assess", "apply_safe_fixes", "unneeded_follow_up",
           "trailing_questions", "repeated_sentences", "mentions_machinery", "filler_phrases"]
