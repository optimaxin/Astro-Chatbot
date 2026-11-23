"""Dangers beyond self-harm: poisoning, harm to others, dangerous materials,
medical emergencies — detected before any model runs.

Audit 2026-11-20: "maine zeher kha liya hai", "I drank rat poison", "I will
kill my husband", "Priya ke pati ko maar dunga", "how to make a bomb" all went
straight to the astrology model. Like `crisis.py` this layer is deterministic
(it cannot fail open during a provider outage) and tuned for recall on the
dangerous side. Each rule needs an intent or event PLUS a target or a
substance, so ordinary astrology talk ("will Saturn kill my career?", "I took
my pills today") does not fire.

    check_poisoning          swallowed / about to swallow poison, pills, chemicals
    check_violence           intent to kill or hurt another person, weapons
    check_dangerous          how to make explosives / poisons / toxic gas (refuse),
                             or a safety question about chemicals (guide)
    check_medical_emergency  someone is in a medical emergency right now
"""

from __future__ import annotations

import re

from app.safety.block_messages import block_message
from app.safety.categories import SafetyCategory
from app.safety.crisis import CrisisCheck

_I = re.IGNORECASE

# ── poisoning ─────────────────────────────────────────────────────────
_POISON_NOUN = (r"poison(?!\s+ivy)|pesticide|insecticide|rat\s*(?:poison|kill\w*)|phenyl|phenol|bleach|"
                r"acid(?!\s*reflux|ity)|kerosene|toilet\s*cleaner|harpic|(?:whole|entire|full)\s+bottle|"
                r"(?:all|lots\s+of|too\s+many|a\s+handful\s+of|\d{2,})\s+(?:my\s+|the\s+|of\s+(?:my|the)\s+)?"
                r"(?:sleeping\s+)?(?:pills|tablets)|strips?\s+of\s+(?:pills|tablets|medicine)")
# Something swallowed already, or announced: "should I take sleeping pills to
# sleep better?" is a question, not this.
_POISON_EN = re.compile(
    rf"(?:\b(?:drank|drunk|swallowed|took|taken|ate|eaten|consumed|had|overdosed\s+on)\b|"
    rf"\b(?:going\s+to|gonna|will|want\s+to|about\s+to|planning\s+to|i'?ll)\s+(?:drink|swallow|take|eat|consume)\b)"
    rf"[^.?!]{{0,30}}\b(?:{_POISON_NOUN})", _I)
_POISON_HI_NOUN = (r"zeh?e?r|zahar|jahar|keet\s*nashak|kitnashak|chuhe?\s+maar\w*\s+(?:ki\s+|wali\s+)?dawa\w*|"
                   r"phenyl|tez?aab|tejab|neend\s+ki\s+goli\w*|(?:saari|bahut\s+saari|dher\s+saari)\s+goli\w*|"
                   r"(?:goli|goliyan|pills?|tablets?|dawai|dawa)\s+ki\s+(?:poori|puri)\s+(?:shishi|bottle|strip|patti)|"
                   r"(?:poori|puri)\s+(?:shishi|bottle|strip|patti)\s+(?:goli|goliyan|pills?|tablets?|dawai|dawa)|"
                   r"(?:saari|poori|puri|\d{2,})\s+(?:\w+\s+)?(?:goliyan|pills?|tablets?|dawai\w*)")
_POISON_HI_VERB = r"(?:kha|pi|pee|le)\s*(?:liya|li|liye|lunga|lungi|loonga|loongi|lu|lun|loon|rahi|raha|lene|khane|peene)"
_POISON_HI = re.compile(rf"(?:{_POISON_HI_NOUN})[^.?!]{{0,25}}{_POISON_HI_VERB}|{_POISON_HI_VERB}[^.?!]{{0,15}}(?:{_POISON_HI_NOUN})", _I)
# "maine goliyan kha li" (plural pills, no quantity) and "I took 30 already" next to a medicine word:
# an overdose until the user says otherwise. A single "dawai kha li" stays routine medication.
_POISON_HI_PLURAL = re.compile(r"\b(?:maine|mene|maine\s+bahut|mene\s+bahut)\s+(?:saari\s+|bahut\s+si\s+)?goliyan\s+"
                               r"(?:kha|le)\s*(?:li|liya|liye)\b", _I)
_OVERDOSE_COUNT = re.compile(r"\b(?:took|taken|swallowed|ate|had)\s+(?:about\s+|around\s+|like\s+)?\d{2,}\b[^.?!]{0,20}"
                             r"\b(?:already|just\s+now|right\s+now|at\s+once|in\s+one\s+go)\b", _I)
_MEDICINE_WORD = re.compile(r"\b(?:pills?|tablets?|capsules?|medicine|medication|meds|dawai|dawa|goli\w*)\b", _I)
_POISON_DEV = re.compile(r"(?:ज़हर|जहर|ज़ह्र|कीटनाशक|चूहे\s*मार\S*\s*\S*\s*दवा|नींद\s*की\s*गोलि\S*|सारी\s*गोलि\S*|तेज़ाब|तेजाब)"
                         r"[^।?!]{0,25}(?:खा|पी|ले)\s*(?:लिया|ली|लिए|लूंगा|लूँगा|लूंगी|लूँगी|लूं|लूँ|रहा|रही)")


def check_poisoning(text: str | None, lang: str | None = None) -> CrisisCheck:
    """Poison, an overdose or a dangerous chemical swallowed or about to be — by
    anyone (a child who drank bleach is the same emergency)."""
    t = text or ""
    if (_POISON_EN.search(t) or _POISON_HI.search(t) or _POISON_DEV.search(t) or _POISON_HI_PLURAL.search(t)
            or (_OVERDOSE_COUNT.search(t) and _MEDICINE_WORD.search(t))):
        return CrisisCheck(True, SafetyCategory.poisoning_emergency, "poison_ingestion",
                           block_message(SafetyCategory.poisoning_emergency, lang))
    return CrisisCheck(False)


# ── harm to another person ────────────────────────────────────────────
_PEOPLE = (r"husband|wife|partner|boyfriend|girlfriend|bf|gf|ex|fianc\w*|father|dad|mother|mom|mum|brother|"
           r"sister|son|daughter|boss|colleague|neighbou?r|friend|in-?laws?|mother-in-law|father-in-law|family|"
           r"child|kid|uncle|aunt|cousin|teacher|landlord|rival|lover|spouse")
# Case-insensitive everywhere except a capitalised NAME as the target ("kill Rahul").
_TARGET = (rf"(?:(?i:him|her|them|someone|somebody|everyone|people|my\s+(?:{_PEOPLE})|his\s+\w+|"
           rf"her\s+(?:{_PEOPLE})|that\s+(?:guy|girl|man|woman|person))|[A-Z][a-z]{{2,}})")
_HARM_VERB = r"kill|murder|stab|shoot|strangle|poison|burn|hurt|harm|attack|beat\s+up|throw\s+acid\s+on|acid\s+attack"
_VIOLENCE_EN = re.compile(
    rf"(?i:\b(?:i|i'?ll|i\s+will|i\s+am|i'?m|gonna|going\s+to|want\s+to|wanna|planning\s+to|plan\s+to|let\s+me|help\s+me|"
    rf"how\s+(?:can|do|should|could)\s+i|how\s+to|best\s+way\s+to|ways?\s+to|easiest\s+way\s+to)\b"
    rf"[^.?!]{{0,30}}\b(?:{_HARM_VERB})\b)\s+{_TARGET}"
    # a bare command: "kill Rahul", "stab him"
    rf"|^\s*(?i:{_HARM_VERB})\s+{_TARGET}")
_WEAPON = re.compile(r"\b(?:knife|gun|pistol|revolver|rifle|blade|axe|sword|chaku|chhuri|churi|bandook|bandooq|katta|talwar)\b", _I)
_WEAPON_INTENT = re.compile(rf"\b(?:{_HARM_VERB}|maa?r\s*(?:dunga|dungi|du|doon|daalunga|daalungi))\b", _I)
# Hinglish first-person future: "maar dunga" = "I will kill/beat"; "maar dega" (he will) is abuse, not this.
_VIOLENCE_HI = re.compile(
    r"\b(?:jaan\s+se\s+)?maa?r\s*(?:dunga|dungi|doonga|doongi|daalunga|daalungi|dalunga|dalungi|daalu|dalu|du|doon)\b|"
    r"\bjaan\s+(?:le\s*(?:lunga|lungi|loonga|loongi)|se\s+maarna)\b|"
    r"(?<!sab\s)(?<!sab\skuch\s)(?<!sabkuch\s)\b(?:khatam|khatm|qatl|katl|katal)\s+kar\s*(?:dunga|dungi|doonga|doongi|du|doon)\b|"
    r"\b(?:zeher|zahar|jahar)\s+de\s*(?:dunga|dungi|doonga|doongi|du|doon)\b|"
    r"\b(?:goli|chaku|chhuri)\s+maar\s*(?:dunga|dungi|du|doon)\b|"
    r"\b(?:tez?aab|tejab|acid)\s+(?:daal|phenk)\s*(?:dunga|dungi|du|doon)\b|"
    r"\bkaise\s+maa?ru\b|\bmaarne\s+ka\s+(?:plan|tarika)\b", _I)
# Harm aimed at oneself ("khud ko maar dunga") is self-harm, not this; "main khud usko maar
# dunga" ("I myself will kill him") is still harm to another.
_SELF = re.compile(r"\b(?:khud|apne\s+aap)\s+ko\b|\bmyself\b|खुद\s*को", _I)
_VIOLENCE_DEV = re.compile(r"मार\s*(?:दूंगा|दूँगा|दूंगी|दूँगी|डालूंगा|डालूँगा|डालूंगी|डालूँगी)|जान\s*से\s*मार(?:\s*दूं|\s*दूँ|ूंगा|ूँगा)|"
                           r"जान\s*ले\s*(?:लूंगा|लूँगा|लूंगी|लूँगी)|(?:ख़त्म|खत्म|क़त्ल|कत्ल)\s*कर\s*(?:दूंगा|दूँगा|दूंगी|दूँगी)|"
                           r"ज़हर\s*दे\s*(?:दूंगा|दूँगा|दूंगी|दूँगी)")


def check_violence(text: str | None, lang: str | None = None) -> CrisisCheck:
    """Intent to kill or hurt another person, or a weapon with intent. Harm to
    oneself ("khud ko maar dunga") is self-harm and handled by crisis.py first."""
    t = text or ""
    hit = (_VIOLENCE_EN.search(t) or _VIOLENCE_DEV.search(t)
           or (_VIOLENCE_HI.search(t) and not _SELF.search(t))
           or (_WEAPON.search(t) and _WEAPON_INTENT.search(t)))
    if hit:
        return CrisisCheck(True, SafetyCategory.violence_risk, "harm_to_others",
                           block_message(SafetyCategory.violence_risk, lang))
    return CrisisCheck(False)


# ── dangerous materials ───────────────────────────────────────────────
_DANGER_THING = (r"bomb|explosives?|ied|grenade|molotov|detonator|tnt|rdx|c-?4|nitroglycerin|napalm|thermite|"
                 r"barood|visphotak|\bbam\b|poison(?:\s+(?:gas|someone|a\s+person))?|toxic\s+gas|poisonous\s+gas|"
                 r"nerve\s+agent|chlorine\s+gas|mustard\s+gas|ricin|cyanide|arsenic|pistol|katta|gun|weapon|hathiyar")
_MAKE = (r"how\s+(?:to|do\s+i|can\s+i|do\s+you)\s+(?:make|build|create|prepare|assemble|produce|get|buy)|"
         r"(?:make|build|create|prepare|assemble)\s+(?:a|an|some|my\s+own)?|recipe\s+for|ingredients?\s+(?:for|of)|"
         r"steps?\s+to\s+(?:make|build)|instructions?\s+(?:for|to)|formula\s+(?:for|of)|"
         r"kaise\s+(?:banaye|banate|banaun|banau|banta|banega|banaen)|banane\s+(?:ka|ki)\s+(?:tarika|vidhi|formula)|"
         r"banana\s+sikha\w*")
_DANGER_REQUEST = re.compile(rf"(?:{_MAKE})[^.?!]{{0,25}}(?:{_DANGER_THING})|(?:{_DANGER_THING})[^.?!]{{0,15}}(?:{_MAKE})", _I)
_DANGER_DEV = re.compile(r"(?:बम|विस्फोटक|ज़हर|जहर|हथियार)[^।?!]{0,15}(?:कैसे\s*बना|बनाने\s*का\s*तरीका|बनाना\s*सिखा)")
_CHEMICAL = r"bleach|ammonia|chemicals?|acid|cleaners?|phenyl|harpic|toxic\s+gas|poisonous\s+gas|fumes|gas\s+leak"
_SAFETY_Q = re.compile(
    rf"(?:is\s+it\s+safe|safe\s+to|can\s+i\s+mix|what\s+happens\s+if|accidentally|by\s+mistake|what\s+(?:should|to)\s+do\s+if|"
    rf"(?:which|what)\s+(?:household\s+)?chemicals?\s+(?:make|produce|create|form|give|release)|"
    rf"mila\w*\s+(?:sakte|safe)|galti\s+se)[^.?!]{{0,40}}(?:{_CHEMICAL})|(?:{_CHEMICAL})[^.?!]{{0,40}}"
    rf"(?:safe\s+hai|is\s+it\s+safe|mix\s+kar\s+sakte|galti\s+se)", _I)


def check_dangerous(text: str | None, lang: str | None = None) -> CrisisCheck:
    """Never instructions for explosives, weapons, poisons or toxic gas. A safety
    question about chemicals ("is it safe to mix bleach and ammonia?") gets
    non-operational safety guidance instead of an off-topic redirect."""
    t = text or ""
    if _DANGER_REQUEST.search(t) or _DANGER_DEV.search(t):
        return CrisisCheck(True, SafetyCategory.dangerous_request, "dangerous_instructions",
                           block_message(SafetyCategory.dangerous_request, lang))
    if _SAFETY_Q.search(t):
        return CrisisCheck(True, SafetyCategory.dangerous_safety, "chemical_safety",
                           block_message(SafetyCategory.dangerous_safety, lang))
    return CrisisCheck(False)


# ── medical emergency happening now ───────────────────────────────────
# Happening NOW: "he had a heart attack last year, will he be okay?" is a chart
# question, and "my business collapsed" / "my phone is not responding" are not emergencies.
_EMERGENCY_EN = re.compile(
    r"\b(?:is\s+having|i'?m\s+having|i\s+am\s+having|just\s+had)\s+(?:a\s+)?(?:heart\s+attack|stroke(?!\s+of)|seizure)\b|"
    r"\b(?:is\s+not|isn'?t|stopped|can'?t|cannot)\s+breath(?:e|ing)\b|"
    r"\b(?:is|went|has\s+gone|just\s+went|is\s+lying)\s+unconscious\b|"
    r"\b(?:he|she|someone|my\s+\w+|i)\s+(?:just\s+)?fainted\b|\bbleeding\s+(?:a\s+lot|heavily|badly|non-?stop)\b|"
    r"\b(?:bitten\s+by\s+a\s+snake|snake\s+(?:just\s+)?bit)\b|\bchest\s+pain\s+(?:right\s+)?now\b", _I)
_EMERGENCY_HI = re.compile(r"\b(?:saa?np\s+ne\s+kaat|behosh\s+ho\s*(?:gaya|gayi|gaye)|saa?ns\s+nahi\s+(?:le|aa)\s*(?:raha|rahi|pa)|"
                           r"heart\s+attack\s+aa\s*(?:gaya|raha)|daura\s+pad\s*(?:gaya|raha)|seene\s+mein\s+(?:tez\s+)?dard\s+(?:ho\s+raha|abhi))\b", _I)
_EMERGENCY_DEV = re.compile(r"सांप\s*ने\s*काट|साँप\s*ने\s*काट|बेहोश\s*हो\s*(?:गया|गई|गए)|सांस\s*नहीं\s*(?:ले|आ)|"
                            r"दिल\s*का\s*दौरा\s*(?:पड़|आ)|सीने\s*में\s*(?:तेज़\s*)?दर्द\s*हो\s*रहा")


def check_medical_emergency(text: str | None, lang: str | None = None) -> CrisisCheck:
    t = text or ""
    if _EMERGENCY_EN.search(t) or _EMERGENCY_HI.search(t) or _EMERGENCY_DEV.search(t):
        return CrisisCheck(True, SafetyCategory.medical_emergency, "medical_emergency_now",
                           block_message(SafetyCategory.medical_emergency, lang))
    return CrisisCheck(False)


def check_danger(text: str | None, lang: str | None = None) -> CrisisCheck:
    """All four, most urgent first. Self-harm (crisis.py) is checked before this."""
    for check in (check_poisoning, check_violence, check_medical_emergency, check_dangerous):
        result = check(text, lang)
        if result.blocked:
            return result
    return CrisisCheck(False)


__all__ = ["check_danger", "check_poisoning", "check_violence", "check_dangerous", "check_medical_emergency"]
