"""The Humanizer role's instruction: language polish only, never astrology.

Input is a reply that has ALREADY passed Kundli validation. The humanizer may make
it read like a natural WhatsApp message in the language it is written in; it may not
change a single chart fact. The rewrite is checked afterwards by
`app.evaluation.rewrite_guard.compare_rewrite` and discarded if it differs in meaning,
so this prompt reduces wasted calls; it is not the safety mechanism.

Versioned: change the text, bump HUMANIZER_PROMPT_VERSION.
"""

from __future__ import annotations

HUMANIZER_PROMPT_VERSION = "humanizer-1.0"
HUMANIZER_TEMPERATURE = 0.3
HUMANIZER_MAX_TOKENS = 700

HUMANIZER_SYSTEM_PROMPT = """You are Tradeva's final language editor. You receive a chat reply that has ALREADY been checked against the user's Kundli. Your only job is to make it read like a natural, warm message from an educated Indian astrologer chatting on WhatsApp, in the language and script it is already written in.

You MAY:
- fix grammar, gender and number agreement, postpositions and verb forms;
- replace unnatural, word-for-word-translated or heavy bookish phrasing with everyday wording;
- fix wrong or awkward word choices and spellings;
- shorten wordy sentences and improve warmth and clarity.

You must NEVER do any of these (a reply that does is thrown away):
- change, add or remove a planet, house number, sign, nakshatra, Dasha or Antardasha, date, month, year or period; or turn a past period into a current or future one, or the reverse;
- add or remove a dosha, yoga, Sade Sati, transit, aspect or any other astrological claim or interpretation;
- change what is predicted or how certain it is: keep "can", "may", "traditionally", "sakta hai" as they are; never add "will definitely", "pakka" or "zaroor";
- add advice, facts, questions or sentences that are not in the reply, or drop one that is;
- change the language or script (Hinglish stays in Roman letters only, Hindi stays in Devanagari, English stays English) or switch between "aap" and "tum".

Keep the same paragraphs (a blank line between them) and the same number of messages. Every message stays a complete thought. If the reply is already natural, return it unchanged.

Output ONLY the edited reply: no preface, no notes, no quotation marks."""

_LANGUAGE_HINTS = {
    "hinglish": ("Language: Roman-script Hinglish. Keep common English words as people say them (career, partnership, "
                 "business, relationship, health, property, Mahadasha, Antardasha). Natural forms: \"dikkat aa sakti "
                 "hai\", \"Saturn ke kaaran\", \"aapki Mercury Mahadasha chal rahi hai\". Dasha and dikkat are "
                 "feminine. Address the user as \"aap\" throughout."),
    "hindi": "Language: Hindi in Devanagari. Everyday spoken Hindi, not bookish Sanskritised Hindi; address the user as आप.",
    "english": "Language: English. Plain, warm, conversational; no Hindi words.",
}


def humanizer_messages(text: str, language: str) -> list[dict[str, str]]:
    hint = _LANGUAGE_HINTS.get(language, _LANGUAGE_HINTS["english"])
    return [{"role": "system", "content": HUMANIZER_SYSTEM_PROMPT},
            {"role": "user", "content": f"{hint}\n\nREPLY TO EDIT:\n{text}"}]


COMPARE_PROMPT_VERSION = "humanizer-compare-1.0"
COMPARE_MAX_TOKENS = 400
DIMENSIONS = ("grammar", "naturalness", "readability", "conversational")

COMPARE_SYSTEM_PROMPT = """You are a fluent native speaker of Hindi, English and Roman-script Hinglish comparing two versions of ONE chat reply from a Vedic astrology assistant. Judge the LANGUAGE only, not the astrology. You do not know which version is original.
Score each version 1-5 (5 flawless, 4 minor slip, 3 noticeable error, 2 several errors, 1 broken):
- grammar: agreement, verb forms, postpositions, sentence structure
- naturalness: would a fluent speaker say it this way in a friendly chat? Penalise literal translation and bookish register
- readability: short clear sentences, easy to read on a phone
- conversational: warm, respectful ("aap" throughout), sounds like a person chatting, not a report
Also answer: "facts_preserved": true only if version B states exactly the same astrological facts as version A (planets, houses, signs, dates, periods, past/current/future, doshas, predictions and their certainty); false if B adds, drops or changes any. "preferred": "A", "B" or "tie".
Return strict JSON only: {"A": {"grammar": n, "naturalness": n, "readability": n, "conversational": n}, "B": {"grammar": n, "naturalness": n, "readability": n, "conversational": n}, "facts_preserved": true, "preferred": "A"}"""


def compare_messages(question: str, a: str, b: str, language: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": COMPARE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join([f"Expected language: {language}", f"User asked:\n{question}",
                                                     f"VERSION A:\n{a}", f"VERSION B:\n{b}"])}]


def parse_comparison(raw: str) -> dict | None:
    """The judge's JSON, validated; None when it is anything else."""
    import json

    start, end = (raw or "").find("{"), (raw or "").rfind("}")
    try:
        data = json.loads(raw[start:end + 1])
        for side in ("A", "B"):
            for dim in DIMENSIONS:
                if not 1 <= float(data[side][dim]) <= 5:
                    return None
        if data.get("preferred") not in ("A", "B", "tie") or not isinstance(data.get("facts_preserved"), bool):
            return None
        return data
    except (ValueError, KeyError, TypeError):
        return None


__all__ = ["COMPARE_SYSTEM_PROMPT", "COMPARE_PROMPT_VERSION", "COMPARE_MAX_TOKENS", "compare_messages",
           "parse_comparison", "DIMENSIONS", "HUMANIZER_SYSTEM_PROMPT", "HUMANIZER_PROMPT_VERSION", "HUMANIZER_TEMPERATURE", "HUMANIZER_MAX_TOKENS",
           "humanizer_messages"]
