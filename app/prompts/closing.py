"""Conversation closings: "bye", "thank you", "dhanyawad", "good night", "bas itna hi".

A clear closing is answered with one short, warm line in the conversation's
language — no astrology model, no fallback, no humanizer. Before this, "bye"
and "thank you" each cost a full main-model turn and could come back as a
chart reading (audit 2026-11-20).

Detection is deterministic for clear cases: the message must consist ONLY of
closing phrases, polite filler and, optionally, a "later" clause ("I will ask
about my career tomorrow"). Anything that opens something new — a question
mark, "but", "ek aur sawaal", "mujhe ek baat poochni hai" — is NOT a closing and
goes through the normal pipeline. Ambiguous messages are left to the intent
classifier (intent "closing"). Safety is always checked before this.
"""

from __future__ import annotations

import hashlib
import re

GOODBYE, GRATITUDE, GOOD_WISHES, CONVERSATION_END, COMBINED = (
    "goodbye", "gratitude", "good_wishes", "conversation_end", "combined_closing")

# (kind, pattern). Longest phrases first inside each alternation.
_PHRASES: tuple[tuple[str, str], ...] = (
    (GRATITUDE, r"bahut\s+bahut\s+(?:dhanyawa+d|dhanyava+d|shukriya)|thank\s*(?:you|u)\s+(?:so\s+much|very\s+much)|"
                r"thanks\s+(?:a\s+lot|so\s+much)|many\s+thanks|thank\s*(?:you|u)|thankyou|thanks|thanx|thnx|thx|"
                r"tysm|\bty\b|dhanyawa+d|dhanyava+d|dhanyabad|shukriya|shukria|shukriyaa|"
                r"धन्यवाद|शुक्रिया"),
    (GOODBYE, r"bye[\s-]*bye|good\s*bye|bye+|bbye|tata|ta[\s-]ta|see\s+(?:you|ya)(?:\s+(?:later|soon|tomorrow))?|cya|"
              r"talk\s+(?:to\s+you\s+)?later|alvida|phir\s+(?:milenge|milte\s+hain|baat\s+karenge)|"
              r"chalta\s+(?:hoon|hu)|chalti\s+(?:hoon|hu)|chalo|बाय|अलविदा|फिर\s+मिलेंगे|चलो|चलता\s+हूँ|चलती\s+हूँ"),
    (GOOD_WISHES, r"good\s*night|good\s*morning|good\s*evening|good\s*day|have\s+a\s+(?:nice|good|great)\s+(?:day|night|evening)|"
                  r"take\s+care|\bgn\b|\bgm\b|shubh\s*ratri|shubh\s+(?:din|prabhat|sandhya)|suprabhat|"
                  r"(?:apna\s+)?khayal\s+(?:rakhna|rakhiye|rakhein)|शुभ\s*रात्रि|शुभ\s+प्रभात|सुप्रभात|ख़याल\s+रखिए|ध्यान\s+रखिए"),
    (CONVERSATION_END, r"bas\s+itna\s+hi|bas\s+itna|bas\s+yahi|aur\s+kuch\s+nahi|kuch\s+nahi|aur\s+nahi|that'?s\s+all|"
                       r"that\s+is\s+all|that'?s\s+it|nothing\s+else|no\s+more\s+questions?|ok(?:ay)?\s+done|done\s+for\s+today|"
                       r"बस\s+इतना\s+ही|और\s+कुछ\s+नहीं"),
)
_FILLER = re.compile(
    r"\b(?:ok+|okay|okk+|k|achha|accha|acha|achcha|theek\s+hai|thik\s+hai|thk\s+h|ji|jee|sir|maam|ma'am|madam|so\s+much|"
    r"very\s+much|a\s+lot|bahut|bahot|aapka|aapko|apka|you|again|once\s+again|all|dear|bhai|bhaiya|didi|guru\s*ji|"
    r"pandit\s*ji|for\s+(?:everything|the\s+help|your\s+help|helping|your\s+time)|madad\s+ke\s+liye|for\s+today|now|"
    r"ab|then|toh|to|and|aur|ke\s+liye|mujhe|for|everything|the|help|great|nice|good|very|helpful|"
    r"shubh|ho|hai|hain)\b|[🙏😊❤️👍🌸✨🙂😀!.,~\-]+", re.I)
# "…, I will ask about my career tomorrow" / "kal poochunga" / "baad mein baat karte hain"
_LATER = re.compile(
    r"\b(?:i'?ll|i\s+will|will)\s+(?:ask|come\s+back|talk|check|message|ping)\b[^?]*\b(?:tomorrow|later|next\s+time|soon|again)\b|"
    r"\b(?:kal|baad\s+mein|baad\s+me|phir\s+kabhi|next\s+time)\b[^?]{0,30}\b(?:poochunga|poochungi|puchunga|puchungi|"
    r"baat\s+karenge|baat\s+karunga|baat\s+karungi|aaunga|aaungi|aata\s+hoon|aati\s+hoon)\b", re.I)
# Opens something new: then it is not a closing.
_CONTINUES = re.compile(
    r"\?|？|\b(?:but|however|although|lekin|magar|par\s+(?:ek|mujhe|abhi)|one\s+more|another\s+question|also|"
    r"ek\s+aur|ek\s+baat|aur\s+bhi|(?:pooch|puch|pooc)(?:na|ni|h?na)\s+(?:hai|tha|chahta|chahti)|"
    r"batao|bataiye|batayiye|tell\s+me|explain|what|why|how|when|which|kya|kab|kaise|kyun|kaun)\b|"
    r"लेकिन|पर\s+एक|एक\s+और|बताइए|क्या|कब|कैसे", re.I)


def detect_closing(text: str | None) -> str | None:
    """The closing kind for a message that only closes the conversation, else None."""
    t = (text or "").strip()
    if not t or len(t) > 160 or _CONTINUES.search(t):
        return None
    later = bool(_LATER.search(t))
    rest = _LATER.sub(" ", t)
    kinds: list[str] = []
    for kind, pattern in _PHRASES:
        rx = re.compile(rf"(?<![\wऀ-ॿ])(?:{pattern})(?![\wऀ-ॿ])", re.I)
        if rx.search(rest):
            kinds.append(kind)
            rest = rx.sub(" ", rest)
    rest = _FILLER.sub(" ", rest)
    if re.search(r"[\wऀ-ॿ]", rest):
        return None                      # something else is being said: not only a closing
    if later and CONVERSATION_END not in kinds:
        kinds.append(CONVERSATION_END)
    if not kinds:
        return None
    return COMBINED if len(set(kinds)) > 1 else kinds[0]


def infer_closing_kind(text: str | None) -> str:
    """Best kind for a message the classifier judged a closing (it may carry
    extra words: "thanks, that reading really helped me")."""
    t = text or ""
    kinds = [kind for kind, pattern in _PHRASES
             if re.search(rf"(?<![\wऀ-ॿ])(?:{pattern})(?![\wऀ-ॿ])", t, re.I)]
    if len(set(kinds)) > 1:
        return COMBINED
    return kinds[0] if kinds else (CONVERSATION_END if _LATER.search(t) else GRATITUDE)


_HINDI_ONLY = re.compile(r"dhanyawa|dhanyava|dhanyabad|shukri|alvida|shubh|suprabhat|khayal|bas\s+itna|aur\s+kuch\s+nahi|"
                         r"phir\s+mil|chalta|chalti|kuch\s+nahi|bas\s+yahi", re.I)


def closing_language(text: str | None, conversation_lang: str) -> str:
    """The conversation's language, except a Hindi/Hinglish-only phrase in an
    otherwise unknown/English context ("dhanyawad" as a first message) gets a
    Hinglish (Roman) or Hindi (Devanagari) reply."""
    t = text or ""
    if re.search(r"[ऀ-ॿ]", t):
        return "hi"
    if conversation_lang == "en" and _HINDI_ONLY.search(t):
        return "hinglish"
    return conversation_lang


_REPLIES: dict[str, dict[str, tuple[str, ...]]] = {
    GOODBYE: {
        "en": ("Take care! Have a lovely day. 😊", "Bye for now — take care, and come back whenever you like! 😊"),
        "hinglish": ("Achha ji! Apna khayal rakhiye. Aapka din shubh ho! 😊",
                     "Theek hai ji, phir milenge! Apna khayal rakhiyega. 😊"),
        "hi": ("अच्छा जी! अपना ख़याल रखिए। आपका दिन शुभ हो! 😊", "ठीक है जी, फिर मिलेंगे! अपना ख़याल रखिएगा। 😊"),
    },
    GRATITUDE: {
        "en": ("You're most welcome! 😊 Whenever you want to know something about your Kundli, I'm here.",
               "Happy to help! 😊 Come back anytime you have a question about your chart."),
        "hinglish": ("Aapka swagat hai! 😊 Jab bhi Kundli se juda kuch jaanna ho, main yahin hoon.",
                     "Khushi hui madad karke! 😊 Kundli ke baare mein kabhi bhi poochiye."),
        "hi": ("आपका स्वागत है! 😊 जब भी कुंडली से जुड़ा कुछ जानना हो, मैं यहीं हूँ।",
               "मदद करके ख़ुशी हुई! 😊 कुंडली के बारे में कभी भी पूछिए।"),
    },
    GOOD_WISHES: {
        "en": ("Thank you, same to you! Take good care. 🙏",),
        "hinglish": ("Aapko bhi! Apna khayal rakhiye. 🙏",),
        "hi": ("आपको भी! अपना ख़याल रखिए। 🙏",),
    },
    CONVERSATION_END: {
        "en": ("Of course! It was lovely talking with you. Whenever you want to know more, just come back.",),
        "hinglish": ("Bilkul ji! Aapse baat karke achha laga. Jab bhi kuch aur jaanna ho, dobara zaroor aaiyega.",),
        "hi": ("बिल्कुल जी! आपसे बात करके अच्छा लगा। जब भी कुछ और जानना हो, दोबारा ज़रूर आइएगा।",),
    },
    COMBINED: {
        "en": ("Glad I could help! Take care, and see you again soon. 😊",),
        "hinglish": ("Khushi hui aapki madad karke! Apna khayal rakhiye. Phir milenge! 😊",),
        "hi": ("आपकी मदद करके ख़ुशी हुई! अपना ख़याल रखिए। फिर मिलेंगे! 😊",),
    },
}
# Time-of-day wishes answered in kind.
_WISH = {
    "night": {"en": "Good night! Rest well. 🙏", "hinglish": "Shubh ratri! Achhe se aaram kijiye. 🙏",
              "hi": "शुभ रात्रि! अच्छे से आराम कीजिए। 🙏"},
    "morning": {"en": "Good morning! Wishing you a lovely day. 😊", "hinglish": "Good morning! Aapka din shubh ho. 😊",
                "hi": "सुप्रभात! आपका दिन शुभ हो। 😊"},
}


def closing_reply(kind: str, text: str | None, lang: str, seed: str = "") -> str:
    """One short line in `lang` ("en" | "hinglish" | "hi"); varied, but stable for a given seed."""
    lang = lang if lang in ("en", "hinglish", "hi") else "en"
    t = (text or "").lower()
    if kind == GOOD_WISHES:
        if re.search(r"night|\bgn\b|ratri|रात्रि", t):
            return _WISH["night"][lang]
        if re.search(r"morning|\bgm\b|prabhat|suprabhat|प्रभात", t):
            return _WISH["morning"][lang]
    options = _REPLIES.get(kind, _REPLIES[GRATITUDE])[lang]
    pick = int(hashlib.sha256(f"{seed}|{text}".encode()).hexdigest(), 16) % len(options)
    return options[pick]


__all__ = ["detect_closing", "closing_reply", "closing_language", "GOODBYE", "GRATITUDE", "GOOD_WISHES",
           "CONVERSATION_END", "COMBINED"]
