"""Deterministic conciseness fixes and measures — the safe ones only.

The model is told to be brief (app.prompts.segments / answer_depth); this
module removes the redundancy that still gets through when doing so cannot
lose information:

* a later sentence that restates an earlier one — most of its content words
  already said in ONE earlier sentence, and no chart fact (planet, house,
  sign, year) that has not already been stated;
* a Dasha date range repeated inside the same reply ("(January 2021 to
  January 2038)" twice) — the first mention keeps it;
* a Dasha date range the previous assistant turn already gave, when the user
  is not asking about timing — the period's name stays, the dates go.

Paraphrases with different words are left to the prompt and measured by the
benchmark judge; guessing at them here would risk deleting real content.
"""

from __future__ import annotations

import re

from app.evaluation.chart_claims import split_sentences

_WORD = re.compile(r"[a-zऀ-ॿ]+(?:-[a-z]+)?", re.I)
_STOP = set("""
a an the and or but of to in on at for with by from as is are was were be been being it its this that these those
your you my me i we our they their them he she his her can could may might will would should shall do does did
has have had not no also very more most much some any all into than then so such there here which who what when
where how why about over under while during through because if
hai hain ho hota hoti hote hoga hogi honge ka ki ke ko se mein me par aur ya bhi jo ye yeh wo woh is us inke unke
aapka aapki aapke aapko aap koi kuch ek kar karta karti karte sakta sakti sakte raha rahi rahe tha thi the liye
जो है हैं हो का की के को से में पर और या भी यह वह इस उस आप आपका आपकी आपके आपको एक कर सकता सकती सकते लिए था थी थे
""".split())
_PLANETS = ("sun|moon|mars|mercury|jupiter|venus|saturn|rahu|ketu|surya|chandra|mangal|budh|guru|brihaspati|"
            "shukra|shani|सूर्य|चंद्र|चन्द्र|मंगल|बुध|गुरु|बृहस्पति|शुक्र|शनि|राहु|केतु")
_SIGNS = ("aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces|"
          "mesh|vrishabh|mithun|kark|simha|kanya|tula|vrishchik|dhanu|makar|kumbh|meen")
_CHART_TOKEN = re.compile(rf"\b(?:{_PLANETS}|{_SIGNS})\b|\b\d{{1,2}}(?:st|nd|rd|th|वें|वाँ)?\b|\b(?:19|20|21)\d\d\b",
                          re.I)
_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
           "जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर")
# "(January 2021 to January 2038)", "(June 2024 se April 2027 tak)", "(जून 2024 से अप्रैल 2027 तक)"
_DATE_RANGE = re.compile(
    rf"\s*\((?:{_MONTHS})?\s*\d{{4}}\s*(?:to|until|till|se|से|–|-)\s*(?:{_MONTHS})?\s*\d{{4}}\s*(?:tak|तक)?\)",
    re.I)
_TIMING_QUESTION = re.compile(
    r"\b(?:when|timing|time|period|dasha|mahadasha|antardasha|until|kab|kab\s+tak|samay|waqt|date|year|years)\b"
    r"|कब|समय|दशा|अवधि|तारीख|साल",
    re.I)


def _content(sentence: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(sentence) if len(w) > 2 and w.lower() not in _STOP}


def _chart_tokens(sentence: str) -> set[str]:
    return {m.group(0).lower() for m in _CHART_TOKEN.finditer(sentence)}


def restated_sentences(text: str, threshold: float = 0.75) -> list[int]:
    """Later sentences that add nothing: ≥ `threshold` of their content words
    appear in one earlier sentence and every chart fact in them was already
    stated. Questions are never touched (the follow-up has its own rules)."""
    sentences = split_sentences(text)
    seen_facts: set[str] = set()
    earlier: list[set[str]] = []
    out: list[int] = []
    for i, sentence in enumerate(sentences):
        words, facts = _content(sentence), _chart_tokens(sentence)
        if (not sentence.rstrip().endswith("?") and len(words) >= 5 and facts <= seen_facts
                and any(len(words & prev) / len(words) >= threshold for prev in earlier)):
            out.append(i)
            continue
        seen_facts |= facts
        earlier.append(words)
    return out


def _sentence_key(sentence: str) -> str:
    return " ".join(re.findall(r"[\wऀ-ॿ]+", sentence.lower()))


def repeated_from_previous(text: str, previous_assistant: str | None) -> list[int]:
    """Sentences copied word for word from the previous assistant turn (a follow-up
    re-stating the last answer, live 2026-11-19). Returned only when at least two
    new content sentences remain; questions are never counted."""
    if not previous_assistant:
        return []
    said = {_sentence_key(s) for s in split_sentences(previous_assistant)}
    sentences = split_sentences(text)
    copied = [i for i, s in enumerate(sentences)
              if not s.rstrip().endswith(("?", "？")) and len(s.split()) >= 5 and _sentence_key(s) in said]
    fresh = sum(1 for i, s in enumerate(sentences) if i not in copied and not s.rstrip().endswith(("?", "？")))
    return copied if copied and fresh >= 2 else []


def redundant_pairs(text: str, threshold: float = 0.6) -> int:
    """Measure only: sentence pairs where most of the later one's content words
    repeat an earlier one (the benchmark's 'duplicate information' signal)."""
    sets = [_content(s) for s in split_sentences(text)]
    return sum(1 for j, b in enumerate(sets) if len(b) >= 5
               for a in sets[:j] if len(a & b) / len(b) >= threshold)


def drop_repeated_date_ranges(text: str, previous_assistant: str | None = None,
                              question: str | None = None) -> tuple[str, int]:
    """Keep each Dasha date range once per reply; drop ranges the previous
    turn already gave unless the user is asking about timing."""
    known: set[str] = set()
    if previous_assistant and not _TIMING_QUESTION.search(question or ""):
        known = {_key(m.group(0)) for m in _DATE_RANGE.finditer(previous_assistant)}
    seen: set[str] = set()
    removed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal removed
        key = _key(match.group(0))
        if key in seen or key in known:
            removed += 1
            return ""
        seen.add(key)
        return match.group(0)

    out = _DATE_RANGE.sub(repl, text)
    return (out, removed) if removed else (text, 0)


_DASHA = re.compile(r"maha\s*dash\w*|antar\s*dash\w*|\bdash[aā]\b|महादशा|अंतर्दशा|अन्तर्दशा|दशा", re.I)
# A sentence that only continues the Dasha thought ("Ye period …", "This period …").
_PERIOD_FOLLOW_ON = re.compile(
    r"^\s*(?:(?:ye|yeh|is|iss|this|that|the)\s+(?:period|phase|samay|daur)\b|together,?\s+they\b|"
    r"(?:ye|yeh)\s+dono\b|यह\s+(?:अवधि|समय)|इस\s+(?:अवधि|समय)|ये\s+दोनों)", re.I)


def drop_unasked_dasha(text: str, question: str | None, depth: str | None) -> tuple[str, int]:
    """Remove the running-Dasha sentences from a simple or normal answer whose
    question is not about timing or the Dasha.

    Root cause (final quality gate, 112 live turns): normal answers about one
    placement ("Shani meri kundli mein kahan hai?", "11th house kya dikhata
    hai?", personality, health) added "Abhi aapki Mercury Mahadasha chal rahi
    hai (January 2021 se January 2038) …" plus a sentence about the period —
    the style example showed a Dasha paragraph in every answer — and were
    93–152 words against a 30–60 target. Kept whenever the user asks about
    time, a period or the Dasha, for detailed and complex answers, and when
    fewer than two sentences would remain."""
    if depth not in ("simple", "normal") or _TIMING_QUESTION.search(question or "") \
            or _DASHA.search(question or ""):
        return text, 0
    sentences = split_sentences(text)
    drop: set[int] = set()
    for i, sentence in enumerate(sentences):
        if _DASHA.search(sentence) or (i - 1 in drop and _PERIOD_FOLLOW_ON.search(sentence)):
            drop.add(i)
    if not drop or len(sentences) - len(drop) < 2:
        return text, 0
    from app.evaluation.chart_claims import strip_sentences

    return strip_sentences(text, drop), len(drop)


# The soft ceiling of each budget (125% of its upper bound — the benchmark's band).
_CEILING = {"simple": 37, "normal": 100}     # normal target: 2–3 messages, 40–80 words


def trim_to_budget(text: str, depth: str | None) -> tuple[str, int]:
    """A simple or normal answer far over its budget loses its LAST
    elaboration sentences, whole, until it fits.

    Root cause (final quality gate, c5): normal answers ignored the "TWO
    messages" target and ran 76–126 words in four bubbles; the extra
    sentences were restatements ("The 9th house lord in the 2nd creates a
    strong foundation…") and side readings after the answer. The answer comes
    first (prompt rule), so the first two sentences are always kept, a
    closing follow-up question is kept, nothing is cut mid-sentence, and
    detailed or complex answers are never trimmed."""
    limit = _CEILING.get(depth or "")
    if not limit or len(text.split()) <= limit:
        return text, 0
    sentences = split_sentences(text)
    follow_up = 1 if sentences and sentences[-1].rstrip().endswith(("?", "？")) else 0
    body = sentences[:len(sentences) - follow_up]
    budget = limit - sum(len(s.split()) for s in sentences[len(body):])
    kept, words = 0, 0
    for sentence in body:
        if kept >= 2 and words + len(sentence.split()) > budget:
            break
        kept, words = kept + 1, words + len(sentence.split())
    if kept == len(body):
        return text, 0
    from app.evaluation.chart_claims import strip_sentences

    return strip_sentences(text, set(range(kept, len(body)))), len(body) - kept


def join_simple_answer(text: str, depth: str | None) -> str:
    """A simple factual answer is ONE bubble: join the model's paragraphs when
    the whole answer is short (conciseness benchmark: "Meri rashi kya hai?"
    came back as two bubbles). Applied to the stored text, so a history
    reload rebuilds the same single bubble."""
    if depth != "simple" or "\n" not in text.strip():
        return text
    if len(text.split()) > 40 or text.rstrip().endswith("?"):
        return text
    return " ".join(p.strip() for p in text.split("\n") if p.strip())


def _key(range_text: str) -> str:
    return " ".join(re.findall(r"\d{4}", range_text))


__all__ = ["restated_sentences", "redundant_pairs", "drop_repeated_date_ranges", "join_simple_answer",
           "drop_unasked_dasha", "trim_to_budget", "repeated_from_previous"]
