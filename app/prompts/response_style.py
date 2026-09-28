"""How a reply *sounds* — language, tone, emoji — as configuration.

Deliberately separate from `reply_format.py`, which owns how a reply is
*shaped* (how many messages, the closing question, the splitter). The two
compose:

    reply_format.STRUCTURE_RULES   ->  shared by every style
  + response_style.instruction     ->  language and tone for this user
                                   ->  one Gemini call, one final answer

Keeping them apart is what makes a style swappable. Adding "marathi" later is
a new entry in this file and nothing else: the bubble cap, the closing
question and the anti-hallucination rules are not restated per style, so they
cannot drift between them.

There is no second "humanisation" pass. The style instruction goes into the
same system prompt as everything else, and the model produces the final
user-facing text in a single inference call. A second call would double the
latency and the token cost for something the model already does well in one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Fallback when a style has to be chosen without a question to read, e.g. an
# unknown id passed straight to `get_style`.
DEFAULT_STYLE_ID = "whatsapp_hinglish"

# Not a style: tells `resolve_style` to mirror the language the user wrote in.
AUTO_STYLE_ID = "auto"


@dataclass(frozen=True)
class ResponseStyle:
    """One way of speaking. `instruction` is what reaches the model."""

    id: str
    name: str
    language: str
    tone: str
    format: str
    length: str
    emoji_level: str
    instruction: str

    def to_dict(self) -> dict[str, Any]:
        """Safe to expose over the API — describes the style, not the prompt."""
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "tone": self.tone,
            "format": self.format,
            "length": self.length,
            "emojiLevel": self.emoji_level,
        }


# Shared by every style that is not explicitly formal. Stated once here rather
# than repeated in each instruction, so "don't over-do it" cannot end up
# phrased three different ways.
_EMOJI_MODERATE = (
    "Emojis: at most one or two in a whole reply, and only where they genuinely fit "
    "(😊 👀 ✨ 💼 🔮). Never one per sentence. A reply with no emoji at all is fine."
)

_NO_GUARANTEES = (
    "Never promise an outcome. Prefer \"chances lag rahe hain\", \"possible hai\", "
    "\"indication milti hai\", \"astrology ke according\" over \"pakka hoga\" or "
    "\"definitely ho jayega\"."
)


STYLES: dict[str, ResponseStyle] = {
    "whatsapp_hinglish": ResponseStyle(
        id="whatsapp_hinglish",
        name="WhatsApp Hinglish",
        language="natural Hinglish",
        tone="friendly and conversational",
        format="WhatsApp",
        length="short to medium",
        emoji_level="moderate",
        instruction=(
            "LANGUAGE AND TONE: NATURAL HINGLISH\n"
            "Write the way two Indians actually text each other on WhatsApp — Hindi and "
            "English mixed naturally, in Roman script.\n"
            "\n"
            "- Do NOT translate every English word into Hindi. \"career\", \"growth\", "
            "\"opportunity\", \"job switch\", \"focus\" stay in English because that is how "
            "people actually say them.\n"
            "- Do NOT write formal/literary Hindi. No \"आपकी कुंडली के अनुसार\", no "
            "\"तदनुसार\". This is a chat, not a report.\n"
            "- Natural connectors when they fit: haan, acha, dekho, basically, kaafi, "
            "thoda, lag raha hai, ho sakta hai, try kar sakte ho. Use them where they "
            "read naturally — do not sprinkle them into every line.\n"
            "- Friendly and warm, but still trustworthy. Not a stand-up act: avoid "
            "\"bro\", \"bhai\", \"yaar\", \"lol\" and laughing emoji unless the user is "
            "clearly talking that way first.\n"
            "- Address the user as \"tum\" / \"tumhare\", not the formal \"aap ji\".\n"
            f"- {_EMOJI_MODERATE}\n"
            f"- {_NO_GUARANTEES}\n"
            "\n"
            "WRITE LIKE THIS\n"
            "Haan, career ka next 1 year kaafi interesting lag raha hai 👀\n"
            "\n"
            "Jupiter tumhare 10th house mein hai. Ye growth ka planet hai.\n"
            "\n"
            "Responsibilities thodi badh sakti hain, but recognition ke chances bhi hain.\n"
            "\n"
            "Abhi kya kaam kar rahe ho?\n"
            "\n"
            "NOT LIKE THIS\n"
            "According to your Kundli, the placement of planets in your tenth house "
            "indicates significant career opportunities during the upcoming year.\n"
            "\n"
            "(That is formal, English-only, and reads like a report. Never do that.)"
        ),
    ),
    "english": ResponseStyle(
        id="english",
        name="English",
        language="English",
        tone="friendly and conversational",
        format="chat",
        length="short to medium",
        emoji_level="moderate",
        instruction=(
            "LANGUAGE AND TONE: EVERYDAY ENGLISH\n"
            "Plain, warm, spoken English — the way you would explain something to a "
            "friend, not the way a report reads.\n"
            "\n"
            "- Contractions are good: \"you're\", \"it's\", \"that's\".\n"
            "- No Hindi words. The user has asked for English.\n"
            "- Never open with \"According to your Kundli\" or \"Based on the provided "
            "information\". Just say the thing.\n"
            f"- {_EMOJI_MODERATE}\n"
            "- Never promise an outcome. Prefer \"this suggests\", \"traditionally read "
            "as\", \"there's a good chance\" over \"you will\" or \"this guarantees\".\n"
            "\n"
            "WRITE LIKE THIS\n"
            "Good news — your career house is strong.\n"
            "\n"
            "Jupiter sits there. That's the planet of growth.\n"
            "\n"
            "So knowledge-based work suits you well.\n"
            "\n"
            "What are you studying right now?"
        ),
    ),
    "hindi": ResponseStyle(
        id="hindi",
        name="Hindi",
        language="Hindi (Devanagari)",
        tone="friendly and conversational",
        format="WhatsApp",
        length="short to medium",
        emoji_level="moderate",
        instruction=(
            "LANGUAGE AND TONE: SIMPLE SPOKEN HINDI\n"
            "Devanagari script. Everyday spoken Hindi, the kind used on WhatsApp — not "
            "literary or Sanskritised Hindi.\n"
            "\n"
            "- Common English words that Hindi speakers actually use stay as they are: "
            "career, job, growth, chance. Do not force a Sanskrit equivalent.\n"
            "- Keep sentences short. Avoid long compound sentences.\n"
            "- Address the user as \"तुम\" / \"तुम्हारा\", warm rather than formal.\n"
            f"- {_EMOJI_MODERATE}\n"
            "- Never promise an outcome. Prefer \"संभावना है\", \"ऐसा लग रहा है\", "
            "\"ज्योतिष के अनुसार\" over \"पक्का होगा\".\n"
            "\n"
            "WRITE LIKE THIS\n"
            "हाँ, career के लिए अगला साल अच्छा लग रहा है 👀\n"
            "\n"
            "तुम्हारे 10th house में Jupiter है। ये growth का ग्रह है।\n"
            "\n"
            "Responsibility थोड़ी बढ़ सकती है, पर पहचान भी मिलेगी।\n"
            "\n"
            "अभी क्या काम कर रहे हो?"
        ),
    ),
    "formal": ResponseStyle(
        id="formal",
        name="Formal English",
        language="English",
        tone="professional and measured",
        format="chat",
        length="short to medium",
        emoji_level="none",
        instruction=(
            "LANGUAGE AND TONE: PROFESSIONAL ENGLISH\n"
            "Measured, respectful, precise. Suitable for a user who wants a serious "
            "consultation rather than a chat.\n"
            "\n"
            "- Address the user as \"you\", politely. No slang, no contractions-heavy "
            "casualness, no Hindi.\n"
            "- NO emojis at all in this style.\n"
            "- Still not a textbook: keep sentences short and readable. Formal does not "
            "mean dense.\n"
            "- Never promise an outcome. Use \"the chart suggests\", \"this is "
            "traditionally read as\", \"there are indications of\"."
        ),
    ),
    "detailed": ResponseStyle(
        id="detailed",
        name="Detailed",
        language="English",
        tone="friendly but thorough",
        format="chat with short sections",
        length="longer",
        emoji_level="none",
        instruction=(
            "LANGUAGE AND TONE: THOROUGH BUT STILL READABLE\n"
            "The user has asked for depth. Give the reasoning, not just the conclusion.\n"
            "\n"
            "- Name which house, planet or period each point comes from, so the user can "
            "see where it came from.\n"
            "- Still plain English and still short sentences. Depth means more points, "
            "not longer sentences.\n"
            "- NO emojis at all in this style.\n"
            "- Never promise an outcome.\n"
            "- Only go deeper on what the chart data and retrieved knowledge actually "
            "support. Depth is never an excuse to invent a placement."
        ),
    ),
}


def get_style(style_id: str | None) -> ResponseStyle:
    """Resolve a style id, falling back to the default.

    An unknown id degrades to the default rather than raising: a bad
    `response_style` from a client should change how an answer reads, never
    cost the user their answer.
    """
    if not style_id:
        return STYLES[DEFAULT_STYLE_ID]
    return STYLES.get(style_id.strip().lower(), STYLES[DEFAULT_STYLE_ID])


def is_known_style(style_id: str | None) -> bool:
    return bool(style_id) and style_id.strip().lower() in STYLES


_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_WORD = re.compile(r"[a-z]+")

# Roman-script Hindi words that are not also common English words, so an
# English sentence cannot trip them ("main", "me", "to", "hi" are left out).
_HINGLISH_MARKERS = frozenset("""
    hai hain hoon hu ho hoga hogi honge hua hui tha thi
    kya kyu kyun kyon kaise kaisa kaisi kab kahan kaun kitna kitne kitni
    mera meri mere mujhe mujhko hum humara hamara humein hamare
    tum tumhara tumhari tumhare tujhe aap aapka aapki aapke apna apni apne
    ka ki ke ko se mein bhi aur nahi nahin nhi mat
    yeh ye woh wo yahan wahan abhi kabhi phir fir kuch bahut bohot zyada jyada thoda
    kar karu karun karna karo karke kiya karenge raha rahi rahe rahega rahegi
    sakta sakti sakte chahiye chahta chahti milega milegi batao bataye bataiye bata
    acha accha achha theek thik sahi haan ji
    shaadi shadi naukri paisa paise pyaar pyar rishta ghar saal mahina kaam padhai
    """.split())


def detect_style(text: str | None) -> str:
    """Pick the style that answers in the language the user wrote in."""
    if not text:
        return "english"
    if _DEVANAGARI.search(text):
        return "hindi"
    words = _WORD.findall(text.lower())
    if not words:
        return "english"
    hits = sum(1 for w in words if w in _HINGLISH_MARKERS)
    if hits >= 2 or (hits == 1 and len(words) <= 4):
        return "whatsapp_hinglish"
    return "english"


def resolve_style(requested: str | None, question: str | None,
                  default: str | None = AUTO_STYLE_ID) -> ResponseStyle:
    """An explicitly requested style wins; otherwise mirror the user's language.

    `default` is the deployment's configured default — a concrete style id
    there pins every unrequested turn to it instead of auto-detecting.
    """
    if is_known_style(requested):
        return get_style(requested)
    if is_known_style(default):
        return get_style(default)
    return STYLES[detect_style(question)]


def list_styles() -> list[ResponseStyle]:
    return list(STYLES.values())


__all__ = [
    "ResponseStyle",
    "STYLES",
    "DEFAULT_STYLE_ID",
    "AUTO_STYLE_ID",
    "get_style",
    "is_known_style",
    "list_styles",
    "detect_style",
    "resolve_style",
]
