"""Central language policy: which language a reply must be written in.

One place decides it, every route obeys it (primary model, fallback models,
the repair pass, canned deflections). Priority, highest first:

1. An explicit request in the CURRENT message ("Reply in English from now
   on", "Can you explain in Hindi?", "Ab English mein answer do"). It is also
   saved as the conversation's preference.
2. The conversation's saved preference (session state, Redis), until the user
   asks for a different language.
3. The language of the current message itself (script + vocabulary).
4. The user's most recent earlier message that had a clear signal.
5. English, only when nothing above gives a signal.

A client-requested concrete style (``response_style="formal"``) or a pinned
deployment default sits above all of this: those are explicit configuration.

Found in production (2026-11): "I want you to reply in english" was obeyed for
one turn and the very next Hinglish message flipped the reply back to
Hinglish, because the request was never remembered. Rule 2 fixes that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ENGLISH = "english"
HINGLISH = "hinglish"
HINDI = "hindi"
LANGUAGES = (ENGLISH, HINGLISH, HINDI)

# Response-style id used for each language, and back.
STYLE_FOR_LANGUAGE = {ENGLISH: "english", HINGLISH: "whatsapp_hinglish", HINDI: "hindi"}
LANGUAGE_OF_STYLE = {"english": ENGLISH, "formal": ENGLISH, "detailed": ENGLISH,
                     "whatsapp_hinglish": HINGLISH, "hindi": HINDI}

# Session-state key holding a persistent explicit preference.
PREFERENCE_KEY = "language_preference"

_LANG_WORD = {
    HINDI: r"(?:hindi|hindī|devanagari|हिंदी|हिन्दी)",
    HINGLISH: r"(?:hinglish|roman\s+hindi)",
    ENGLISH: r"(?:english|angrezi|angreji|अंग्रेज़ी|अंग्रेजी|इंग्लिश)",
}
_VERBS = (r"(?:reply|respond|answer|write|speak|talk|explain|tell|say|chat|continue|baat|bolo|bol|"
          r"likho|likhiye|batao|bataiye|btao|samjhao|samjhaiye|jawab|jawaab|answer\s+do|reply\s+do)")


def _patterns(word: str) -> list[re.Pattern]:
    return [
        # "in Hindi", "using English", "into Hinglish"
        re.compile(rf"(?<![\wऀ-ॿ])(?:in|using|into)\s+{word}(?![\wऀ-ॿ])", re.I),
        # "Hindi mein / me / main", "हिंदी में"
        re.compile(rf"(?<![\wऀ-ॿ]){word}\s*(?:mein|me|main|mai|mei|में)(?![\wऀ-ॿ])", re.I),
        # "speak English", "reply English please", "answer only English"
        re.compile(rf"\b{_VERBS}\s+(?:me\s+|to\s+me\s+)?(?:only\s+|in\s+)?{word}(?![\wऀ-ॿ])", re.I),
        # "English please", "only Hindi", "Hindi only"
        re.compile(rf"(?:^|\b)(?:only\s+{word}|{word}\s+(?:only|please|pls|plz))(?![\wऀ-ॿ])", re.I),
        # a message that is nothing but the language name: "English", "Hindi?"
        re.compile(rf"^\s*{word}\s*[?.!]*\s*$", re.I),
    ]


_REQUEST_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    # Hinglish before Hindi: "in Hinglish" must not be read as a Hindi request.
    (HINGLISH, _patterns(_LANG_WORD[HINGLISH])),
    (HINDI, _patterns(_LANG_WORD[HINDI])),
    (ENGLISH, _patterns(_LANG_WORD[ENGLISH])),
]

# "What is my rashi called in Hindi?" asks for a translation, not a switch.
_TRANSLATION_ASK = re.compile(r"\b(?:called|known as|name|naam|word|mean|means|meaning|matlab|translate|spell)\b", re.I)


def explicit_request(text: str | None) -> str | None:
    """The language the user explicitly asks the assistant to use, if any."""
    if not text:
        return None
    for language, patterns in _REQUEST_PATTERNS:
        if any(p.search(text) for p in patterns):
            if _TRANSLATION_ASK.search(text) and not re.search(_VERBS, text, re.I):
                return None
            return language
    return None


def is_language_switch_only(text: str | None) -> bool:
    """True when the message does nothing but ask for a language change
    ("I want you to reply in english", "Kya aap hindi me jawab de sakte hai")."""
    if not text or explicit_request(text) is None:
        return False
    words = re.findall(r"[\wऀ-ॿ]+", text.lower())
    return len(words) <= 12 and not re.search(
        r"\b(?:career|marriage|shaadi|shadi|dasha|house|planet|kundli|chart|job|money|health|family|"
        r"rashi|lagna|love|finance|child|baby)\b|कुंडली|शादी|करियर|दशा", text, re.I)


_REEXPLAIN = re.compile(r"\b(?:explain|repeat|again|translate|tell|say|batao|bataiye|btao|samjhao|samjhaiye|"
                        r"dobara|phir se|fir se)\b|समझा|बता|फिर से|दोबारा", re.I)

SWITCH_CONFIRMATION = {
    ENGLISH: "Sure — I'll reply in English from now on.",
    HINGLISH: "Zaroor — ab se main aapko Hinglish mein jawab dunga.",
    HINDI: "ज़रूर — अब से मैं आपको हिंदी में जवाब दूँगा।",
}


def wants_reexplain(text: str | None) -> bool:
    """A switch request that also asks for the previous answer again
    ("Can you explain in Hindi?") rather than just a language change
    ("I want you to reply in english")."""
    return bool(text) and bool(_REEXPLAIN.search(text))


def detect_language(text: str | None) -> str | None:
    """Language of the message itself, or None when it carries no signal
    ("ok", "hmm", "5th house?")."""
    if not text:
        return None
    from app.prompts.response_style import language_signal

    style = language_signal(text)
    return LANGUAGE_OF_STYLE.get(style) if style else None


@dataclass(frozen=True)
class LanguageDecision:
    language: str            # english | hinglish | hindi (for a client-pinned style: its language)
    style_id: str            # the response-style id that carries it
    source: str              # client | deployment | explicit | saved | detected | history | default
    persist: str | None = None   # a new preference to save, when the user just stated one
    switch_only: bool = False    # the message only asked for the language change

    def to_dict(self) -> dict[str, object]:
        return {"language": self.language, "style": self.style_id, "source": self.source,
                "persisted": self.persist, "switch_only": self.switch_only}


def resolve_language(question: str | None, saved_preference: str | None = None,
                     earlier_user_messages: list[str] | None = None,
                     requested_style: str | None = None,
                     deployment_default: str | None = None) -> LanguageDecision:
    """Apply the priority order in the module docstring. Pure."""
    from app.prompts.response_style import is_known_style

    if is_known_style(requested_style):
        style = requested_style.strip().lower()
        return LanguageDecision(LANGUAGE_OF_STYLE.get(style, ENGLISH), style, "client")
    if is_known_style(deployment_default):
        style = deployment_default.strip().lower()
        return LanguageDecision(LANGUAGE_OF_STYLE.get(style, ENGLISH), style, "deployment")

    asked = explicit_request(question)
    if asked:
        return LanguageDecision(asked, STYLE_FOR_LANGUAGE[asked], "explicit", persist=asked,
                                switch_only=is_language_switch_only(question))
    saved = (saved_preference or "").strip().lower()
    if saved in LANGUAGES:
        return LanguageDecision(saved, STYLE_FOR_LANGUAGE[saved], "saved")
    detected = detect_language(question)
    if detected:
        return LanguageDecision(detected, STYLE_FOR_LANGUAGE[detected], "detected")
    for earlier in reversed(earlier_user_messages or []):
        found = explicit_request(earlier) or detect_language(earlier)
        if found:
            return LanguageDecision(found, STYLE_FOR_LANGUAGE[found], "history")
    return LanguageDecision(ENGLISH, STYLE_FOR_LANGUAGE[ENGLISH], "default")


__all__ = ["ENGLISH", "HINGLISH", "HINDI", "LANGUAGES", "STYLE_FOR_LANGUAGE", "LANGUAGE_OF_STYLE",
           "PREFERENCE_KEY", "LanguageDecision", "explicit_request", "is_language_switch_only",
           "wants_reexplain", "SWITCH_CONFIRMATION",
           "detect_language", "resolve_language"]
