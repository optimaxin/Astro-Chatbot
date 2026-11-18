"""Localized system notices: what the assistant says when it must NOT answer
from the chart. They are written here, in English, Hindi and Roman Hinglish,
and never by a model, so a notice can neither guess nor drift.

Each notice is a tuple of sentences. The sentences are registered in
`SYSTEM_SENTENCES`, which the factual validator uses to recognise (and leave
alone) the assistant's own explanation: without that, "dosha ki calculation
available nahi hai" would itself be judged a dosha statement.
"""

from __future__ import annotations

from app.safety.block_messages import resolve_lang

# Self-contained on purpose: it must read correctly whether the user asked about
# a dosha or the model brought one up on its own.
DOSHA_UNAVAILABLE: dict[str, tuple[str, ...]] = {
    "en": (
        "I can't give a verdict on doshas such as Manglik, either way: the chart engine doesn't calculate "
        "them yet, and I won't guess from missing data.",
        "I can still read your planetary placements and Dasha periods.",
    ),
    "hi": (
        "मांगलिक जैसे दोषों पर मैं किसी भी तरह का निष्कर्ष नहीं दे सकता: चार्ट इंजन अभी इनकी गणना नहीं करता, "
        "और अधूरी जानकारी के आधार पर मैं अनुमान नहीं लगाऊँगा।",
        "फिर भी मैं आपके ग्रहों की स्थिति और दशा पढ़ सकता हूँ।",
    ),
    "hinglish": (
        "Manglik jaise doshas ke baare mein main na \"hai\" keh sakta hoon na \"nahi hai\": chart engine abhi "
        "inki calculation nahi karta, aur adhoori jaankari par main andaaza nahi lagaunga.",
        "Aapke planetary placements aur Dasha periods main zaroor bata sakta hoon.",
    ),
}

CLARIFY: dict[str, tuple[str, ...]] = {
    "en": ("Could you tell me what you'd like to know about: marriage, career, money, health, or something else?",
           "Then I can look at your chart for exactly that."),
    "hi": ("आप किस बारे में जानना चाहते हैं: विवाह, करियर, धन, स्वास्थ्य या कुछ और?",
           "बताइए, फिर मैं आपकी कुंडली में ठीक वही देखकर बताऊँगा।"),
    "hinglish": ("Aap kis baare mein jaanna chahte hain: shaadi, career, paisa, health ya kuch aur?",
                 "Bataiye, phir main aapki kundli mein bilkul wahi dekhkar bataunga."),
}

_HOUSE_SIGN: dict[str, tuple[str, ...]] = {
    "en": ("Note: the house numbers in your chart data contradict its signs, so I'm withholding every house "
           "placement until the chart is re-checked.",
           "Planet signs, your Lagna and your Dasha periods are not affected."),
    "hi": ("ध्यान दें: आपकी कुंडली के डेटा में भाव संख्याएँ और राशियाँ आपस में मेल नहीं खातीं, इसलिए जाँच होने तक "
           "मैं कोई भी भाव-स्थिति नहीं बता रहा।",
           "ग्रहों की राशियाँ, आपका लग्न और दशाएँ इससे प्रभावित नहीं हैं।"),
    "hinglish": ("Note: aapki chart data mein house numbers aur signs ek doosre se match nahi kar rahe, isliye "
                 "chart dobara check hone tak main koi bhi house placement nahi bata raha.",
                 "Planet signs, aapka Lagna aur Dasha periods par iska asar nahi hai."),
}

_CONFLICT = {
    "en": "Note: your chart data has conflicting records for {planets}, so I'm not stating its position "
          "until the chart is re-checked.",
    "hi": "ध्यान दें: आपकी कुंडली के डेटा में {planets} के रिकॉर्ड आपस में मेल नहीं खाते, इसलिए जाँच होने तक "
          "मैं इसकी स्थिति नहीं बता रहा।",
    "hinglish": "Note: aapki chart data mein {planets} ke records ek doosre se match nahi kar rahe, isliye "
                "chart dobara check hone tak main iski position nahi bata raha.",
}

_HINDI_GRAHA = {"sun": "सूर्य", "moon": "चंद्रमा", "mars": "मंगल", "mercury": "बुध", "jupiter": "बृहस्पति",
                "venus": "शुक्र", "saturn": "शनि", "rahu": "राहु", "ketu": "केतु"}


def _join(sentences: tuple[str, ...]) -> str:
    return " ".join(sentences)


def dosha_unavailable(style_id: str | None) -> str:
    return _join(DOSHA_UNAVAILABLE[resolve_lang(style_id)])


def clarification(style_id: str | None) -> str:
    return _join(CLARIFY[resolve_lang(style_id)])


def integrity_notice(style_id: str | None, mismatch: bool, conflicts: list[str]) -> str:
    lang = resolve_lang(style_id)
    parts: list[str] = []
    if conflicts:
        names = [(_HINDI_GRAHA.get(g, g) if lang == "hi" else g.capitalize()) for g in conflicts]
        parts.append(_CONFLICT[lang].format(planets=", ".join(names)))
    if mismatch:
        parts.append(_join(_HOUSE_SIGN[lang]))
    return " ".join(parts)


SYSTEM_SENTENCES: frozenset[str] = frozenset(
    s for table in (DOSHA_UNAVAILABLE, CLARIFY, _HOUSE_SIGN) for tup in table.values() for s in tup)


def is_system_sentence(sentence: str) -> bool:
    """True for the assistant's own fixed notices. (The conflict notice names
    planets and never mentions a dosha, so it needs no exemption.)"""
    return sentence.strip() in SYSTEM_SENTENCES


__all__ = ["DOSHA_UNAVAILABLE", "CLARIFY", "SYSTEM_SENTENCES", "dosha_unavailable", "clarification",
           "integrity_notice", "is_system_sentence"]
