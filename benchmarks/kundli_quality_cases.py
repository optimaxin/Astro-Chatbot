"""Kundli chatbot quality benchmark: 100 scored turns.

* 25 English, 25 Hinglish, 25 Hindi single-turn questions
* 25 multi-turn contextual turns (9 conversations), including the language
  switches and follow-ups seen in the 2026-11 production screenshots

Each turn states the language the reply MUST be in, following the language
policy (explicit request > saved preference > message language). Flags mark
questions a chart cannot answer literally. Scored by
scripts/kundli_quality_bench.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    question: str
    language: str                  # english | hinglish | hindi — required reply language
    topic: str = "general"
    life_fact: bool = False        # "am I married?" — chart cannot establish it
    beyond_chart: bool = False     # rebirth, death date … — not computed by a Kundli
    switch: bool = False           # the turn only asks for a language change
    depth: str = ""                # simple | normal | detailed | complex ("" = derive from topic)

    @property
    def expected_depth(self) -> str:
        if self.depth:
            return self.depth
        if self.topic in ("rashi", "lagna", "switch", "smalltalk"):
            return "simple"
        if self.topic in ("career", "marriage", "family", "finance", "partner"):
            return "detailed"
        return "normal"


@dataclass(frozen=True)
class Conversation:
    id: str
    group: str                     # english | hinglish | hindi | multiturn
    turns: tuple[Turn, ...] = field(default_factory=tuple)


def _single(group: str, items: list[tuple[str, str]], **flags_by_index: dict) -> list[Conversation]:
    out = []
    for i, (topic, q) in enumerate(items, 1):
        extra = flags_by_index.get(str(i), {})
        out.append(Conversation(f"{group[:2]}-{i:02d}", group, (Turn(q, group, topic, **extra),)))
    return out


ENGLISH = [
    ("dasha", "What is my current Dasha about?"),
    ("dasha", "Which Mahadasha and Antardasha am I running, and until when?"),
    ("career", "Tell me about my career."),
    ("career", "Is this a good period to change my job?"),
    ("marriage", "When is a good period for marriage?"),
    ("marriage", "What does my 7th house say about my married life?"),
    ("marriage", "Am I married or not?"),
    ("finance", "How will my financial growth be?"),
    ("finance", "Does my chart support savings and wealth?"),
    ("family", "Can you tell me about my family and relationships?"),
    ("family", "What does my chart say about my mother and home life?"),
    ("health", "Anything in my chart about my health and energy?"),
    ("placement", "Where is Saturn in my chart and what does it mean?"),
    ("placement", "What does Venus in my 7th house mean?"),
    ("placement", "Which house is my Jupiter in?"),
    ("lagna2", "What is my ascendant and what does it say about my personality?"),
    ("rashi", "What is my moon sign?"),
    ("house", "Who is the lord of my 10th house and where is it placed?"),
    ("house", "What does my 11th house indicate about gains?"),
    ("yoga", "Do I have any yogas in my chart?"),
    ("d9", "What does my Navamsa say about marriage?"),
    ("children", "What does my chart say about children?"),
    ("beyond", "When will my rebirth happen?"),
    ("education", "Is higher education favourable for me?"),
    ("growth", "What is my life purpose according to my chart?"),
]
HINGLISH = [
    ("dasha", "Meri current dasha ke baare mein batao"),
    ("dasha", "Abhi kaunsi Mahadasha aur Antardasha chal rahi hai?"),
    ("career", "Mera career kaisa rahega?"),
    ("career", "Mere career k bare me batao"),
    ("career", "Kya ye job change karne ka sahi time hai?"),
    ("marriage", "Meri shaadi kab hogi?"),
    ("marriage", "Meri shaadi ka sahi time kya hai"),
    ("marriage", "Kya meri shaadi ho chuki hai?"),
    ("finance", "Paise ke maamle mein meri kundli kya kehti hai?"),
    ("finance", "Meri financial growth kaisi rahegi?"),
    ("family", "Meri family aur relationships ke baare mein batao"),
    ("family", "Ghar aur mummy ke baare mein kundli kya kehti hai?"),
    ("health", "Meri sehat ke baare mein kuch hai kundli mein?"),
    ("placement", "Shani meri kundli mein kahan hai aur uska kya matlab hai?"),
    ("placement", "Mere 7th house mein Venus hone ka kya matlab hai?"),
    ("placement", "Jupiter kis house mein hai?"),
    ("lagna2", "Mera lagna kya hai aur meri personality kaisi hai?"),
    ("rashi", "Meri rashi kya hai?"),
    ("house", "Mere 10th house ka lord kaun hai aur kahan baitha hai?"),
    ("house", "11th house mere liye kya dikhata hai?"),
    ("yoga", "Kya meri kundli mein koi yoga hai?"),
    ("children", "Bachchon ke baare mein kundli kya kehti hai?"),
    ("beyond", "Mera rebirth kb hoga?"),
    ("education", "Padhai ke liye mera time kaisa hai?"),
    ("growth", "Meri zindagi ka purpose kya hai kundli ke hisaab se?"),
]
HINDI = [
    ("dasha", "मेरी वर्तमान दशा के बारे में बताइए।"),
    ("dasha", "अभी कौन सी महादशा और अंतर्दशा चल रही है?"),
    ("career", "मेरा करियर कैसा रहेगा?"),
    ("career", "क्या यह नौकरी बदलने का सही समय है?"),
    ("marriage", "मेरी शादी कब होगी?"),
    ("marriage", "मुझे मेरी शादी के बारे में बताइए।"),
    ("marriage", "क्या मेरी शादी हो चुकी है?"),
    ("finance", "मेरी आर्थिक स्थिति कैसी रहेगी?"),
    ("finance", "क्या मेरी कुंडली में धन योग है?"),
    ("family", "मेरे परिवार और रिश्तों के बारे में बताइए।"),
    ("family", "मेरी माता और घर के बारे में कुंडली क्या कहती है?"),
    ("health", "मेरे स्वास्थ्य के बारे में कुंडली क्या कहती है?"),
    ("placement", "मेरी कुंडली में शनि कहाँ है और उसका क्या अर्थ है?"),
    ("placement", "सप्तम भाव में शुक्र होने का क्या मतलब है?"),
    ("placement", "बृहस्पति किस भाव में है?"),
    ("lagna2", "मेरा लग्न क्या है और मेरा स्वभाव कैसा है?"),
    ("rashi", "मेरी राशि क्या है?"),
    ("house", "मेरे दशम भाव का स्वामी कौन है और वह कहाँ है?"),
    ("house", "एकादश भाव मेरे लिए क्या दर्शाता है?"),
    ("yoga", "मेरी कुंडली में कौन से योग हैं?"),
    ("kundli", "मेरी कुंडली में क्या है?"),
    ("children", "संतान के बारे में मेरी कुंडली क्या कहती है?"),
    ("beyond", "मेरा पुनर्जन्म कब होगा?"),
    ("education", "उच्च शिक्षा के लिए मेरा समय कैसा है?"),
    ("growth", "मेरी कुंडली के अनुसार मेरे जीवन का उद्देश्य क्या है?"),
]

_LIFE = {"life_fact": True}
_BEYOND = {"beyond_chart": True}

SINGLE: list[Conversation] = [
    *_single("english", ENGLISH, **{"7": _LIFE, "23": _BEYOND}),
    *_single("hinglish", HINGLISH, **{"8": _LIFE, "23": _BEYOND}),
    *_single("hindi", HINDI, **{"7": _LIFE, "23": _BEYOND}),
]

E, H, D = "english", "hinglish", "hindi"
MULTI: list[Conversation] = [
    # Screenshot 1-3: marriage, then a switch to Hindi which must persist.
    Conversation("mt-01", "multiturn", (
        Turn("When is a good period for marriage?", E, "marriage"),
        Turn("Kya aap hindi me jawab de sakte hai", D, "switch", switch=True),
        Turn("Meri shaadi ka sahi time kya hai", D, "marriage"),
    )),
    # Screenshot 7-9: Hinglish chat, explicit English request, persistence.
    Conversation("mt-02", "multiturn", (
        Turn("Mere career k bare me batao", H, "career"),
        Turn("I want you to reply in english", E, "switch", switch=True),
        Turn("Mera rebirth kb hoga?", E, "beyond", beyond_chart=True),
    )),
    # Topic continuation: career -> finances should not repeat the career reading.
    Conversation("mt-03", "multiturn", (
        Turn("Tell me about my career.", E, "career"),
        Turn("What about my financial growth?", E, "finance"),
        Turn("And which period is best for it?", E, "dasha"),
    )),
    # Hinglish continuation and a short follow-up with no language signal.
    Conversation("mt-04", "multiturn", (
        Turn("Meri shaadi kab hogi?", H, "marriage"),
        Turn("ok aur partner kaisa hoga?", H, "marriage"),
        Turn("theek hai", H, "smalltalk"),
    )),
    # Hindi throughout, dasha follow-up.
    Conversation("mt-05", "multiturn", (
        Turn("मेरी वर्तमान दशा के बारे में बताइए।", D, "dasha"),
        Turn("इसका मेरे करियर पर क्या असर होगा?", D, "career"),
    )),
    # No persistent preference: each question sets its own language.
    Conversation("mt-06", "multiturn", (
        Turn("Tell me about my 7th house.", E, "marriage"),
        Turn("Mera 10th house kaisa hai?", H, "career"),
        Turn("मेरा लग्न क्या है?", D, "lagna"),
    )),
    # Life-fact question mid-conversation (screenshot 6).
    Conversation("mt-07", "multiturn", (
        Turn("Can you tell me about my family and relationship", E, "family"),
        Turn("I am married or not?", E, "marriage", life_fact=True),
    )),
    # Hindi preference, then English again ("Ab English mein answer do").
    Conversation("mt-08", "multiturn", (
        Turn("Can you explain in Hindi?", D, "switch", switch=True),
        Turn("Mera career kaisa rahega?", D, "career"),
        Turn("Ab English mein answer do", E, "switch", switch=True),
        Turn("What does my 11th house say?", E, "house"),
    )),
    # User states a wrong chart fact; the chart must win.
    Conversation("mt-09", "multiturn", (
        Turn("My Mercury is in the 7th house, right? What does that mean for marriage?", E, "placement"),
        Turn("So is my 7th lord strong?", E, "house"),
    )),
]

# Round 3 extension (multi-message + Hinglish quality): depth-labelled questions,
# partner-personality and full-analysis questions, and the real-user Hinglish
# screenshot case. Kept in their own group so the 100-turn core stays comparable.
EXTENDED: list[Conversation] = [
    Conversation("ex-01", "extended", (Turn("What is my Lagna?", E, "lagna", depth="simple"),)),
    Conversation("ex-02", "extended", (Turn("What kind of partner personality does my chart show?", E,
                                            "partner", depth="detailed"),)),
    Conversation("ex-03", "extended", (Turn("Give me a complete marriage analysis: indications, partner nature, "
                                            "timing and my current Dasha.", E, "marriage", depth="complex"),)),
    Conversation("ex-04", "extended", (Turn("Give me a detailed career analysis with timing.", E, "career",
                                            depth="complex"),)),
    Conversation("ex-05", "extended", (Turn("Mera lagna kya hai?", H, "lagna", depth="simple"),)),
    # real-user screenshot case (Hinglish quality)
    Conversation("ex-06", "extended", (Turn("Mera partner kaisa hoga aur shaadi kab hogi? Detail mein batao", H,
                                            "partner", depth="complex"),)),
    Conversation("ex-07", "extended", (Turn("Mere life partner ki personality kaisi hogi?", H, "partner",
                                            depth="detailed"),)),
    Conversation("ex-08", "extended", (Turn("Meri relationship aur commitment ke baare mein batao", H, "partner",
                                            depth="detailed"),)),
    Conversation("ex-09", "extended", (Turn("मेरे जीवनसाथी का स्वभाव कैसा होगा?", D, "partner",
                                            depth="detailed"),)),
    Conversation("ex-10", "extended", (Turn("मेरी शादी का पूरा विश्लेषण बताइए — योग, जीवनसाथी, समय और दशा।", D,
                                            "marriage", depth="complex"),)),
    Conversation("ex-11", "extended", (
        Turn("Mere career k bare me batao", H, "career"),
        Turn("Aur meri shaadi ke baare mein detail mein batao", H, "marriage", depth="complex"),
    )),
]

CONVERSATIONS: list[Conversation] = [*SINGLE, *MULTI, *EXTENDED]


def turn_count() -> dict[str, int]:
    counts: dict[str, int] = {}
    for conv in CONVERSATIONS:
        counts[conv.group] = counts.get(conv.group, 0) + len(conv.turns)
    return counts


__all__ = ["Turn", "Conversation", "CONVERSATIONS", "SINGLE", "MULTI", "EXTENDED", "turn_count"]
