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
    "Emojis: usually none; at most one in a whole reply, and only where it genuinely fits."
)

_NO_GUARANTEES = (
    "Never promise an outcome. Prefer \"chances lag rahe hain\", \"possible hai\", "
    "\"indication milti hai\", \"astrology ke according\" over \"pakka hoga\" or "
    "\"definitely ho jayega\"."
)


# The examples below teach TONE only. They once contained a concrete
# placement ("Jupiter sits there" / "Jupiter tumhare 10th house mein hai"), and
# the model copied it into a real answer as a fact about the user's chart — a
# live hallucination on 2026-11-12. Slots in <angle brackets> stand for the
# user's real chart facts; tests/test_accuracy.py fails if a concrete
# planet-in-house claim is ever put back into an example.
_EXAMPLE_NOTE = (
    "(The examples show tone and structure ONLY. Text in <angle brackets> marks where the "
    "user's REAL chart facts from the USER KUNDLI block go — never copy a placement from an "
    "example, never output angle brackets, and never reuse an example sentence word for word: "
    "write your own, about this user's actual question.)"
)

STYLES: dict[str, ResponseStyle] = {
    "whatsapp_hinglish": ResponseStyle(
        id="whatsapp_hinglish",
        name="WhatsApp Hinglish",
        language="natural Hinglish",
        tone="friendly, respectful and conversational",
        format="chat",
        length="proportional to the question",
        emoji_level="minimal",
        instruction=(
            "LANGUAGE AND TONE: NATURAL HINGLISH\n"
            "Write the way an educated Indian astrologer chats — Hindi and English mixed naturally, "
            "in Roman script only.\n"
            "\n"
            "- Do NOT translate every English word into Hindi. \"career\", \"growth\", "
            "\"opportunity\", \"7th house\", \"Mahadasha\" stay as people actually say them.\n"
            "- Do NOT write formal/literary Hindi, and NO Devanagari at all — not even a gloss in "
            "brackets like \"(करियर)\".\n"
            "- Respectful: address the user as \"aap\" / \"aapki\" / \"aapke\" consistently. Never "
            "switch to \"tum\".\n"
            "- English words keep their correct ENGLISH spelling: partner, commitment, relationship, "
            "emotional connection, personality, communication, opportunities, financial growth, "
            "practical, professional. Never re-spell them the Hindi way (paartanar, committment, "
            "rileshanship, kanekshan).\n"
            "- Hindi words in common chat spelling: aapko (not aapako), batate hain ki (not bataate hain "
            "kee), soch-samajhkar, shaadi, Kundli.\n"
            "- Natural grammar: \"Aapke career mein growth ki opportunities ban sakti hain\" (not \"Aapka "
            "career mein opportunities hone ke chances hain\"); \"Aapko aisa partner pasand aa sakta hai jo "
            "soch-samajhkar baat karta ho\" (not \"Aapako partner pasand aaega jo soch-samajh ke bolta "
            "hai\"). Do not start every sentence with \"ke according\".\n"
            "- No slang (\"bro\", \"yaar\", \"lol\").\n"
            "- Explain what the knowledge passages say in your own natural words; never paste their English "
            "headings or Title-Case phrases (\"Charitable Inclination, Beautiful Clothing\").\n"
            "- Prefer everyday words over heavy Sanskrit (\"partner\" or \"jeevansathi\", not \"kartavyanishth "
            "vyakti\"), and say \"hai\", not \"situate hai\".\n"
            "- Use only words you are sure of. If you are unsure of a Hindi word, use the English one "
            "(\"timing\", \"impact\", \"aspect\", \"favourable\"); never invent or half-transliterate a word "
            "(not \"mulyaman samay\", \"assar\", \"aspekt\", \"Mahadashi\").\n"
            "- Do not assume the user's gender: with \"aap\" use the plural forms (\"aap kar sakte hain\", "
            "\"aap mehsoos karte hain\"), never \"aap kar sakti hain\".\n"
            f"- {_EMOJI_MODERATE}\n"
            f"- {_NO_GUARANTEES}\n"
            "\n"
            f"{_EXAMPLE_NOTE}\n"
            "WRITE LIKE THIS\n"
            "Aapke <sawaal wale area> mein <chart jo sabse zaroori baat dikhata hai, seedhe aur warm shabdon "
            "mein>.\n"
            "\n"
            "<USER KUNDLI ka ek asli placement>, jise traditionally <aasaan meaning> se joda jaata hai.\n"
            "\n"
            "Rozmarra ki zindagi mein iska matlab hai ki <practical, insaani matlab — koi naya placement "
            "nahi>.\n"
            "\n"
            "Isliye <ek concrete suggestion>, aur <kis baat ka dhyan rakhein>.\n"
            "\n"
            "NOT LIKE THIS\n"
            "Aapki Kundli mein 10th house <sign> hai aur uska lord <house> mein baitha hai. Traditionally "
            "10th house career dikhata hai. 2nd house wealth aur 11th house gains dikhata hai, aur aapki "
            "<Mahadasha> (<dates>) bhi chal rahi hai.\n"
            "\n"
            "(That is a chart report: four placements and nothing the person can use. Never do that.)\n"
            "\n"
            "According to your Kundli, the placement of planets in your tenth house "
            "indicates significant career opportunities during the upcoming year. Aap kaunsa "
            "role dekh rahe ho?\n"
            "\n"
            "(That is formal English mixed with an unnecessary personal question. Never do that.)"
        ),
    ),
    "english": ResponseStyle(
        id="english",
        name="English",
        language="English",
        tone="friendly, respectful and conversational",
        format="chat",
        length="proportional to the question",
        emoji_level="minimal",
        instruction=(
            "LANGUAGE AND TONE: EVERYDAY ENGLISH\n"
            "Plain, warm, spoken English — the way a knowledgeable astrologer explains something "
            "clearly, not the way a report reads.\n"
            "\n"
            "- Contractions are fine: \"you're\", \"it's\", \"that's\".\n"
            "- No Hindi or Hinglish words or sentences.\n"
            "- Never open with \"According to your Kundli\" or \"Based on the provided "
            "information\". Just say the thing.\n"
            f"- {_EMOJI_MODERATE}\n"
            "- Never promise an outcome. Prefer \"this suggests\", \"traditionally read "
            "as\", \"this can be interpreted as\" over \"you will\" or \"this guarantees\".\n"
            "\n"
            f"{_EXAMPLE_NOTE}\n"
            "WRITE LIKE THIS\n"
            "Your <area they asked about> <the main thing the chart shows for it, said plainly and warmly>.\n"
            "\n"
            "<One real placement from the USER KUNDLI block>, which is traditionally read as <its meaning in "
            "plain words>.\n"
            "\n"
            "In everyday terms, <what it means for their life — no new placement>.\n"
            "\n"
            "So <one concrete step they can take>, and <one thing to be careful about>.\n"
            "\n"
            "NOT LIKE THIS\n"
            "Your 10th house is <sign> and its lord sits in the <house>. The 10th house governs career, and "
            "the lord there points to service. Your 2nd house shows wealth, your 11th house shows gains, and "
            "you are running <Mahadasha> (<dates>).\n"
            "\n"
            "(That is a chart report: four placements and nothing the person can use. Never do that.)"
        ),
    ),
    "hindi": ResponseStyle(
        id="hindi",
        name="Hindi",
        language="Hindi (Devanagari)",
        tone="friendly, respectful and conversational",
        format="chat",
        length="proportional to the question",
        emoji_level="minimal",
        instruction=(
            "LANGUAGE AND TONE: SIMPLE SPOKEN HINDI\n"
            "Devanagari script. Clear, everyday Hindi — not literary or heavily Sanskritised, and "
            "not English sentences.\n"
            "\n"
            "- Address the user as \"आप\" / \"आपकी\" / \"आपके\" consistently. Never switch to "
            "\"तुम\" / \"तुम्हारी\".\n"
            "- Astrology terms in Devanagari: महादशा, अंतर्दशा, लग्न, भाव, बुध, शुक्र. Never "
            "half-Latin forms like \"महाdasha\".\n"
            "- Common English words Hindi speakers use (career, job) may stay as they are.\n"
            "- Keep sentences short.\n"
            f"- {_EMOJI_MODERATE}\n"
            "- Never promise an outcome. Prefer \"संभावना है\", \"ऐसा माना जाता है\", "
            "\"ज्योतिष के अनुसार\" over \"पक्का होगा\".\n"
            "\n"
            f"{_EXAMPLE_NOTE}\n"
            "WRITE LIKE THIS\n"
            "आपके <सवाल वाले क्षेत्र> में <चार्ट जो सबसे ज़रूरी बात दिखाता है, सीधे और अपनेपन से>।\n"
            "\n"
            "<USER KUNDLI से एक असली placement>, जिसे पारंपरिक रूप से <आसान शब्दों में अर्थ> से जोड़ा जाता है।\n"
            "\n"
            "रोज़ की ज़िंदगी में इसका मतलब है कि <व्यावहारिक, इंसानी अर्थ — कोई नया placement नहीं>।\n"
            "\n"
            "इसलिए <एक ठोस सुझाव>, और <किस बात का ध्यान रखें>।\n"
            "\n"
            "NOT LIKE THIS\n"
            "आपकी कुंडली में दशम भाव <राशि> में है और इसका स्वामी <भाव> में है। दशम भाव करियर दिखाता है। "
            "दूसरा भाव धन और ग्यारहवाँ भाव लाभ दिखाता है, और आपकी <महादशा> (<तारीखें>) भी चल रही है।\n"
            "\n"
            "(यह चार्ट की रिपोर्ट है: चार placements और व्यक्ति के काम की कोई बात नहीं। ऐसा कभी न लिखें।)"
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
    bhai bhaiya bhaiyya didi yaar yar koi naam kaunsa kaunsi kaunse konsa konsi kon
    bolo batado bataoge bataogi batana btao bta samjhao samjha samajh dekho dekhiye suno
    kyunki lekin magar agar toh jab tab kal aaj hamesha sirf bilkul shayad zaroor jarur pakka
    jaldi kaafi matlab pata maloom malum lagta lagti lagega chalega chalegi
    mai mujhse humko hume unka unki unke uska uski uske iska iski iske wala wali wale
    zindagi jindagi kismat naseeb bhagya vivah biwi pati patni beta beti ladka ladki
    dost parivaar parivar mummy papa sehat tabiyat galat badhiya badiya kharab bekar
    """.split())

# Plainly English words. A message with neither these nor Hinglish markers
# ("ok", "hmm", "5th house?") carries no language signal of its own.
# Deliberately excludes words Hinglish shares with English ("me", "to", "main").
_ENGLISH_MARKERS = frozenset("""
    what when where why how who which whose will would can could should shall
    is are am was were be been do does did the a an my your you i it this that these those
    tell about more please give show explain and or but not yes thanks thank
    """.split())

_ELONGATED = re.compile(r"(.)\1+$")     # "bhaii" -> "bhai", "yaarr" -> "yaar"


def _is_marker(word: str) -> bool:
    return word in _HINGLISH_MARKERS or _ELONGATED.sub(r"\1", word) in _HINGLISH_MARKERS


def language_signal(text: str | None) -> str | None:
    """The style a message itself asks for, or None when it gives no signal
    either way (then the conversation's earlier language should decide)."""
    if not text:
        return None
    from app.prompts.language import requested_style

    asked = requested_style(text)
    if asked:
        return asked
    if _DEVANAGARI.search(text):
        return "hindi"
    words = _WORD.findall(text.lower())
    if not words:
        return None
    hits = sum(1 for w in words if _is_marker(w))
    if hits >= 2 or (hits == 1 and len(words) <= 4):
        return "whatsapp_hinglish"
    if hits or any(w in _ENGLISH_MARKERS for w in words) or len(words) > 4:
        return "english"
    return None


def detect_style(text: str | None, history: list[str] | None = None) -> str:
    """Pick the style that answers in the language the user wrote in.

    An explicit request ("answer in Hindi", "English mein batao") beats script
    detection. A message with no signal of its own ("ok", "hmm") keeps the
    language of the user's most recent earlier message that had one, so a
    Hinglish conversation does not flip to English on a short reply.
    """
    signal = language_signal(text)
    if signal:
        return signal
    for earlier in reversed(history or []):
        signal = language_signal(earlier)
        if signal:
            return signal
    return "english"


def needs_history(requested: str | None, question: str | None, default: str | None = AUTO_STYLE_ID) -> bool:
    """True when the style will be auto-detected and the message alone cannot
    decide it, i.e. only then is it worth reading the conversation history."""
    return not is_known_style(requested) and not is_known_style(default) and language_signal(question) is None


def resolve_style(requested: str | None, question: str | None,
                  default: str | None = AUTO_STYLE_ID, history: list[str] | None = None) -> ResponseStyle:
    """An explicitly requested style wins; otherwise mirror the user's language.

    `default` is the deployment's configured default — a concrete style id
    there pins every unrequested turn to it instead of auto-detecting.
    `history` is the user's earlier messages, oldest first (optional).
    """
    if is_known_style(requested):
        return get_style(requested)
    if is_known_style(default):
        return get_style(default)
    return STYLES[detect_style(question, history)]


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
    "language_signal",
    "needs_history",
    "resolve_style",
]
