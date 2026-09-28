"""Requests for a phone number, an address, a photo, or a meeting.

The risk here is fabrication. Asked "what's your number?", a helpful model
will often invent one, or agree to a call it cannot attend. Both are concrete
harms: the user dials a stranger, or waits for someone who never arrives.
Neither is prevented by a prompt line, because the model is not lying on
purpose — it is completing a pattern.

**Deliberately narrower than the service this came from.** That product's
persona was a human astrologer who never disclosed being an AI, so its gate
also caught "are you a real person?" and deflected it. This assistant is the
opposite by design — `BASE_SYSTEM_PROMPT` says "You are an AI assistant, not a
human astrologer. If asked, say so plainly and without apology." Porting the
deflection would have turned an honest product into an evasive one, so
identity questions are left to the model and only the fabrication risk is
gated here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.safety.block_messages import DEFAULT_LANG, resolve_lang
from app.safety.categories import SafetyCategory


@dataclass(frozen=True, slots=True)
class BoundaryProbe:
    blocked: bool
    category: SafetyCategory | None = None
    rule: str | None = None
    reply: str = ""


_CONTACT = re.compile(
    r"\b(your|aapka|tumhara|apka)\s+(phone|number|mobile|whatsapp|telegram|"
    r"insta(?:gram)?|email|e-mail|contact)\b|"
    r"\b(give|send|share|share kar|de do|do)\s+(me\s+)?(your\s+)?"
    r"(phone|number|mobile|whatsapp|email|contact)\b|"
    r"\bcontact\s+(details|info|number)\b|"
    r"\bnumber\s+(do|dedo|de do|bhejo|send)\b",
    re.IGNORECASE,
)

_MEETING = re.compile(
    r"\b(can|could|will)\s+(we|i|you)\s+(meet|talk on (the )?phone|have a call|"
    r"video call|speak on call)\b|"
    r"\b(meet|call)\s+(you|me)\s+(in person|tomorrow|today|somewhere)\b|"
    r"\blet'?s\s+(meet|talk on call|have a call)\b|"
    r"\b(mil|milo|milna|miloge|milenge|mil sakte)\b|"
    r"\bcall\s+(kar|karo|karenge|kariye)\b",
    re.IGNORECASE,
)

_LOCATION = re.compile(
    r"\b(your|aapka|tumhara)\s+(address|location|office|shop|city)\b|"
    r"\bwhere\s+(are\s+you|do\s+you)\s+(from|live|stay|sit|located|based)\b|"
    r"\bwhere\s+is\s+your\s+(office|shop|place)\b|"
    r"\b(aap|tum)\s+kaha[nm]?\s+(rehte|rehti|ho|hain|baithte)\b",
    re.IGNORECASE,
)

_PHOTO = re.compile(
    r"\b(your|aapki|tumhari)\s+(photo|pic|picture|selfie|dp|face)\b|"
    r"\b(send|share|show)\s+(me\s+)?(your\s+)?(photo|pic|picture|selfie|dp)\b",
    re.IGNORECASE,
)

# One reply covers all four: the honest answer is the same in every case, and
# it points at the consultation route BASE_SYSTEM_PROMPT already tells the
# model exists, rather than inventing a channel.
_RESPONSES: dict[str, str] = {
    "en": (
        "I don't have a phone number, an address or a photo — I'm the AI assistant "
        "inside Tredev Astro, so this chat is the only place I exist. If you'd like "
        "to speak with a human astrologer, Tredev Astro has consultations for that. "
        "In the meantime, ask me anything about your chart and I'll help here."
    ),
    "hi": (
        "मेरे पास न फ़ोन नंबर है, न पता, न कोई फ़ोटो — मैं Tredev Astro का AI सहायक हूँ, "
        "इसलिए यह चैट ही मेरी एकमात्र जगह है। अगर आप किसी मानव ज्योतिषी से बात करना "
        "चाहते हैं, तो Tredev Astro पर परामर्श उपलब्ध है। तब तक, अपनी कुंडली के बारे में "
        "कुछ भी पूछिए — यहीं मदद करूँगा।"
    ),
    "hinglish": (
        "Mere paas na phone number hai, na address, na koi photo — main Tredev Astro "
        "ka AI assistant hoon, isliye yeh chat hi meri ek hi jagah hai. Agar aap kisi "
        "human astrologer se baat karna chahte hain, to Tredev Astro par consultation "
        "available hai. Tab tak, apni kundli ke baare mein kuch bhi poochiye — yahin "
        "help karunga."
    ),
}


def detect_boundary_probe(text: str | None, lang: str | None = None) -> BoundaryProbe:
    """Catch requests for contact details, a meeting, a location or a photo."""
    if not text:
        return BoundaryProbe(blocked=False)

    for rule, pattern in (
        ("contact", _CONTACT),
        ("meeting", _MEETING),
        ("location", _LOCATION),
        ("photo", _PHOTO),
    ):
        if pattern.search(text):
            key = resolve_lang(lang)
            return BoundaryProbe(
                blocked=True,
                category=SafetyCategory.boundary,
                rule=rule,
                reply=_RESPONSES.get(key, _RESPONSES[DEFAULT_LANG]),
            )

    return BoundaryProbe(blocked=False)


__all__ = ["BoundaryProbe", "detect_boundary_probe"]
