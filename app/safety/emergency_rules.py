"""Emergency rules added after the safety audit of 2026-11-22.

`crisis.py` (self-harm) and `danger.py` (poisoning, violence, medical emergency, dangerous materials)
catch the explicit statements. The audit screenshots showed what they still missed, all of it sent
to the Kundli classifier and answered "that is outside what I can help with":

    "Bhai mai poison lene ja raha hu"          poison named in English, intention in Hindi
    "Kitne chances hain mai zinda bach jaunga?" a survival question: someone is asking whether they
                                                will live, so something has already happened
    "Maine ek ko maar diya hai"                an admission of killing someone
    "Uski haddiyan toot gayi hain"             a report that a person is injured
    "I might hurt someone", "kuch galat kar lunga"   intent that is not (yet) specific

Same design as the other layers: deterministic, no model, tuned for recall because a helpline shown
to someone who did not need it costs a moment and a missed emergency costs far more. Every rule needs
an event or intent AND a person or substance, and astrology vocabulary exempts the wording that
astrologers use ("will I survive this Saturn period?").

    check_poison_intent       a named poison / pills together with going-to-take-it wording
    check_lethality           asking whether poison is fatal, whether they will survive, how much kills
    check_violence_possible   "I might hurt someone", "mann kar raha hai maar doon", revenge
    check_violence_admission  "maine usko maar diya", "I killed someone"
    check_injury_report       a bone broken, blood, unconscious, not moving, not breathing
    check_uncertain           intent that might be self-harm or harm to others: ask, do not assume
    check_self_harm_variants  misspelled / slang self-harm ("sucide", "kms", "unalive")
"""

from __future__ import annotations

import re

from app.safety.block_messages import block_message
from app.safety.categories import SafetyCategory
from app.safety.crisis import CrisisCheck

_I = re.IGNORECASE

# Words that make a survival / death question a chart question ("will I survive this Saturn period?").
_ASTRO = re.compile(
    r"\b(?:kundli|kundali|dasha|dashaa|mahadasha|antardasha|graha|grah|planet|planets|shani|saturn|rahu|ketu|mangal|"
    r"mars|guru|jupiter|venus|shukra|mercury|budh|moon|chandra|surya|sun|house|bhav|lagna|horoscope|chart|transit|"
    r"gochar|yog|dosh|janam|jyotish|astrology|astrologer|sade\s*sati|saade\s*saati|rashi|nakshatra|maraka|"
    r"aayu|ayu|umar|longevity)\b|कुंडली|दशा|ग्रह|शनि|राहु|केतु|मंगल|ज्योतिष|राशि|साढ़े\s*साती", _I)

# ── a substance ───────────────────────────────────────────────────────
_SUBSTANCE = (r"poi?s[ioea]*n\w*|z[ae]h[ae]r|zehr|zahr|j[ae]h[ae]r|vish\s+(?:kha|pi|pee)|"
              r"rat\s*(?:poison|kill\w*)|chuhe?\s*maar\w*\s*(?:ki\s+)?dawa\w*|keet\s*nashak|kitnashak|pesticide|insecticide|"
              r"phenyl|phenol|tez?aab|tejab|bleach|harpic|toilet\s*cleaner|kerosene|"
              r"(?:saari|saare|bahut\s+saari|dher\s+saari|poori|puri|\d{2,})\s+"
              r"(?:\w+\s+)?(?:goli|goliyan|pills?|tablets?|dawai|dawa|medicine|capsules?)|"
              r"ज़हर|जहर|कीटनाशक|तेज़ाब|तेजाब|नींद\s*की\s*गोलि\S*")
_TAKE_INTENT = (
    # Hinglish: "lene ja raha hu", "kha loonga", "pee lunga", "khane ka mann", "peene wala hu"
    r"(?:lene|khane|peene|pine|pinay|lena|khana|peena)\s+(?:ja\s*|jaa\s*)(?:raha|rahi|rha|rhi)\b|"
    r"(?:lene|khane|peene|pine)\s+(?:wala|wali|waala|waali|ka\s+(?:mann|man|soch|plan|irada|iraada|mood))\b|"
    r"(?:khana|peena|pina|lena)\s+(?:hai|h)\b|\b(?:drank|drunk|swallowed|ate|eaten|consumed|took|taken)\b|"
    r"(?:खाने|पीने|लेने)\s+(?:जा\s*रहा|जा\s*रही)|(?:खा|पी|ले)\s*(?:लूंगा|लूँगा|लूंगी|लूँगी|लिया|ली)|"
    r"(?:le|kha|pi|pee)\s*(?:lu|lun|loon|lunga|lungi|loonga|loongi|raha\s+hu|raha\s+hoon|rahi\s+hu|rahi\s+hoon|"
    r"liya|li|liye)\b|(?:kha|pi|pee)\s*ke\s+(?:mar|so)\b|"
    # English
    r"(?:going\s+to|gonna|about\s+to|want\s+to|wanna|planning\s+to|plan\s+to|thinking\s+(?:of|about)|"
    r"i'?ll|i\s+will|let\s+me|how\s+(?:much|many)\s+(?:should|do|can)\s+i|decided\s+to|"
    r"ready\s+to)\s+(?:\w+\s+){0,2}(?:eat|drink|take|swallow|consume|have|try|use)\b|"
    r"\b(?:eat|drink|swallow|consume)(?:ing)?\b|"
    r"kaise\s+(?:khaun|khau|piyun|piyu|loon|lu)\b")
_NOT_INTENT = re.compile(r"\b(?:poison\s+ivy|food\s+poisoning|zeher\s+ugal|vish\s+(?:yog|dosh|kanya))\b", _I)
_POISON_INTENT = re.compile(rf"(?:{_SUBSTANCE})[^.?!]{{0,40}}(?:{_TAKE_INTENT})|(?:{_TAKE_INTENT})[^.?!]{{0,30}}(?:{_SUBSTANCE})", _I)
_POISON_SOURCE = re.compile(r"\b(?:poison|zeher|zahar|jahar|zehar)\s+(?:kahan|kaha|kidhar|kaise)\s+(?:se\s+)?(?:milega|milegi|"
                            r"milta|mil\s+sakta|mil\s+sakti|laun|lau|lun|banta|banate)\b|"
                            r"\bmujhe\s+(?:poison|zeher|zahar)\s+chahiye\b", _I)


def _reply(category: SafetyCategory, rule: str, lang: str | None) -> CrisisCheck:
    return CrisisCheck(True, category, rule, block_message(category, lang))


def check_poison_intent(text: str | None, lang: str | None = None) -> CrisisCheck:
    """A poison, a toxic chemical or a handful of pills named next to going-to-take-it wording,
    in English, Hindi or both at once ("Bhai mai poison lene ja raha hu")."""
    t = text or ""
    if _NOT_INTENT.search(t):
        return CrisisCheck(False)
    if _POISON_INTENT.search(t) or _POISON_SOURCE.search(t):
        return _reply(SafetyCategory.poisoning_emergency, "poison_intent", lang)
    return CrisisCheck(False)


# ── asking whether it is fatal / whether they will live ───────────────
_SURVIVE = re.compile(
    r"\bz[ie]nda\s+(?:bach|rah|reh)\w*|\bjind[ae]\s+(?:bach|rah|reh)\w*|\bbach\s*(?:paunga|paungi|jaunga|jaungi|jaaunga|jaaungi|"
    r"sakta\s+hu|sakta\s+hoon|sakti\s+hu|sakti\s+hoon|payenge|jayenge)\b|\bbachne\s+ke\s+chances?\b|"
    r"\bjee\s+(?:paunga|paungi|sakunga|sakungi)\b|\bmarne\s+ke\s+chances?\b|"
    r"\bchances?\s+(?:of|that|i)\s+(?:me\s+|i\s+|will\s+)?(?:surviv\w+|dying|living|death|live)\b|\b(?:will|would|can|could)\s+i\s+"
    r"(?:even\s+)?(?:survive|live\s+through|make\s+it\s+through)\b|\bam\s+i\s+going\s+to\s+survive\b|"
    r"\b(?:survival|fatality|mortality)\s+(?:rate|chances?|probability|odds)\b|\bhow\s+likely\s+(?:am\s+i\s+)?to\s+(?:die|survive)\b|"
    r"ज़िंदा\s+बच|जिंदा\s+बच|बच\s+(?:पाऊंगा|पाऊँगा|पाऊंगी|पाऊँगी|जाऊंगा|जाऊँगा|जाऊंगी|जाऊँगी)", _I)
_LETHAL = re.compile(
    r"\b(?:lethal|fatal|deadly|lethality)\s+(?:dose|amount|quantity|level)\b|\bhow\s+(?:many|much)\b[^.?!]{0,50}"
    r"\b(?:to\s+(?:die|kill)|will\s+(?:kill|be\s+fatal)|is\s+(?:fatal|lethal|enough\s+to\s+(?:die|kill)))\b|"
    r"\b(?:enough|sufficient)\s+to\s+(?:die|kill)\b|\bkitn[ei]\s+(?:goli|goliyan|zeher|dawai|poison)\b[^.?!]{0,40}"
    r"\b(?:mar|marr|jaan|khatam)\w*|\b(?:goli|goliyan|zeher|dawai|poison)\b[^.?!]{0,40}\bse\s+(?:mar|marr|maut)\w*|"
    r"\bkitna\s+(?:zeher|poison|dose)\b[^.?!]{0,30}\b(?:mar|maut|jaan|kaafi|enough)\w*|\bhow\s+long\b[^.?!]{0,30}"
    r"\b(?:to\s+die|before\s+(?:i\s+)?die|does\s+it\s+take\s+to\s+die)\b", _I)
# Worry about a medical procedure or an exam is not this.
_PROCEDURE = re.compile(r"\b(?:surgery|operation|cancer|chemo\w*|tumou?r|disease|illness|treatment|exam|interview|flight|journey|"
                        r"trek|war|covid|accident|delivery|pregnan\w+)\b", _I)
_QUESTION_WORD = re.compile(r"\b(?:kitne|kitna|kya|how|what|will|can|chances?|probability|odds)\b|\?", _I)


def check_lethality(text: str | None, lang: str | None = None) -> CrisisCheck:
    """Questions about whether something is fatal, how much kills, or whether the user will live.
    With a substance named it is a poisoning emergency; without one ("Kitne chances hain mai zinda
    bach jaunga?") someone is still asking whether they will survive something, so the reply asks
    whether they are safe and have taken anything. Chart vocabulary exempts "will I survive this
    Saturn period?". The reply never contains a number or a probability."""
    t = text or ""
    if _LETHAL.search(t):
        return _reply(SafetyCategory.poisoning_emergency, "lethality_request", lang)
    if _SURVIVE.search(t) and not _ASTRO.search(t):
        if re.search(rf"(?:{_SUBSTANCE})", t, _I):
            return _reply(SafetyCategory.poisoning_emergency, "survival_after_substance", lang)
        if _QUESTION_WORD.search(t) and not _PROCEDURE.search(t):
            return _reply(SafetyCategory.self_harm_crisis, "survival_question", lang)
    return CrisisCheck(False)


# ── harm to others: possible, and already done ────────────────────────
_PERSON = (r"(?i:someone|somebody|anyone|everyone|people|him|her|them|my\s+(?:husband|wife|boss|father|mother|brother|sister|"
           r"son|daughter|friend|ex|colleague|neighbou?r|in-?laws?|family|partner|boyfriend|girlfriend)|that\s+(?:guy|girl|man|"
           r"woman|person)|usko|use|usse|unko|unhe|kisi\s+ko|sabko|sab\s+ko)")
_VIOLENCE_POSSIBLE = re.compile(
    rf"\b(?:i|we)\s+(?:might|may|could|can'?t\s+(?:stop|control)\s+(?:myself|me\s+from)|feel\s+like|am\s+afraid\s+i'?ll|"
    rf"am\s+scared\s+i'?ll|have\s+an?\s+urge\s+to|will\s+end\s+up)\s+(?:\w+\s+){{0,2}}(?:kill\w*|hurt\w*|harm\w*|stab\w*|"
    rf"shoot\w*|beat\w*|attack\w*|strangl\w*|murder\w*)\s+{_PERSON}|"
    rf"\bfeel(?:ing)?\s+(?:like\s+)?(?:killing|hurting|beating|stabbing)\s+{_PERSON}|"
    rf"\b(?:mann|man|dil)\s+(?:kar\s+raha|kar\s+rha|karta|kehta)\s+hai\s+(?:ki\s+)?(?:\w+\s+){{0,3}}"
    rf"(?:maar|mar|khatam|nuksaan|hurt)\s*(?:doon|du|dun|dalu|daalu|dalun|daalun|dunga)|\bkisi\s+ko\s+(?:nuksaan|chot|hurt)\s+"
    rf"(?:pahuncha|pahunch|de)\w*|\bnuksaan\s+pahuncha\s+(?:doon|dunga|dungi|sakta\s+hu|sakta\s+hoon)\b|"
    rf"\bbadla\s+(?:loonga|lunga|lungi|loongi)\b|\b(?:usko|use|unko)\s+(?:chhodunga|chhodungi|chhodoonga)\s+nahi\b|"
    rf"\b(?:i'?ll|i\s+will|i'?m\s+going\s+to|gonna|going\s+to|want\s+to|wanna)\s+(?:\w+\s+){{0,2}}(?:beat|hit|slap|punch|kill|"
    rf"hurt|stab|shoot|strangle|attack|murder)\s+{_PERSON}\b|\b(?:knife|gun|pistol|blade|bat|rod|hammer|axe)\b[^.?!]{{0,40}}"
    rf"\b(?:use|using)\s+(?:it|that|them)\s+(?:on|against)\s+{_PERSON}", _I)
_ANIMAL_OR_THING = (r"machhar|machchar|makkhi|keeda|keede|kide|cockroach|chhipkali|chuha|chuhe|bug|bugs|mosquito|fly|flies|ant|"
                    r"cheenti|makdi|spider|bichhoo|bichchu|saanp|snake|kutta|kutte|billi|rat|mouse|insect")
_ADMISSION_HI = re.compile(
    rf"\b(?:maine|mene|main\s*ne|mai\s*ne|mainey|hum\s*ne|humne)\s+(?:(?!(?:{_ANIMAL_OR_THING})\b)\w+\s+){{0,3}}(?:ko\s+)?"
    rf"(?:jaan\s+se\s+)?(?:maar|mar|khatam|qatl|katl|katal)\s*(?:diya|dia|dala|daala|di|dee)\b|"
    rf"\b(?:maine|mene|main\s*ne|mai\s*ne)\s+(?:\w+\s+){{0,2}}(?:khoon|murder|qatl|katl)\s+(?:kar\s+diya|kiya|ho\s+gaya)\b|"
    rf"\b(?:maine|mene|main\s*ne|mai\s*ne)\s+(?:\w+\s+){{0,3}}(?:chaku|chhuri|churi|goli|knife|gun|danda|rod|hathoda)\s+"
    rf"(?:\w+\s+){{0,2}}(?:maar|ghop|ghus|chala)\w*|"
    rf"\b(?:usko|use|usse|unko|kisi\s+ko|ek\s+ko|ek\s+aadmi\s+ko|ladke\s+ko|aadmi\s+ko|ladki\s+ko)\s+(?:jaan\s+se\s+)?"
    rf"(?:maar|mar)\s*(?:diya|dala|daala|di)\b|"
    rf"\b(?:usko|use|unko|kisi\s+ko)\s+zinda\s+nahi\s+chhoda\b|"
    rf"मैंने\s+(?:\S+\s+){{0,3}}(?:को\s+)?(?:जान\s+से\s+)?(?:मार|खत्म|कत्ल)\s*(?:दिया|डाला|दी)|"
    rf"(?:उसको|उसे|किसी\s+को|एक\s+को)\s+(?:जान\s+से\s+)?मार\s*(?:दिया|डाला)", _I)
_ADMISSION_EN = re.compile(
    rf"\b(?:i|we)\s+(?:just\s+|think\s+i\s+|have\s+|already\s+)?(?:killed|murdered|stabbed|shot|strangled|choked|poisoned|"
    rf"beat\s+up|pushed|ran\s+over|hurt|injured|hit)\s+(?:{_PERSON}|a\s+(?:man|woman|guy|girl|boy|child|kid|person)|"
    rf"(?-i:[A-Z][a-z]{{2,}}))(?:\s+(?:badly|really\s+badly|to\s+death|with\s+\w+))?|"
    rf"\b(?:i|we)\s+(?:think\s+i\s+)?(?:may\s+have|might\s+have|accidentally)\s+(?:killed|hurt|injured)\s+{_PERSON}|"
    rf"\b(?:he|she|they)\s+(?:is|are|was)\s+(?:dead|not\s+breathing|not\s+moving)\s+because\s+of\s+me\b|"
    rf"\bi\s+(?:am|'m)\s+a\s+(?:murderer|killer)\b", _I)
# "I hit him" / "I hurt her" are only an admission when the message also says how bad it was.
_STRONG_VERB = re.compile(r"\b(?:killed|murdered|stabbed|shot|strangled|choked|poisoned|beat\s+up)\b", _I)
_SERIOUS = re.compile(r"\b(?:badly|bleeding|blood|unconscious|not\s+(?:moving|breathing|responding)|dead|dying|"
                      r"hospital|ambulance|serious\w*|critical|head|skull)\b|khoon|behosh", _I)


def check_violence_possible(text: str | None, lang: str | None = None) -> CrisisCheck:
    """"I might hurt someone", "I feel like killing him", "mann kar raha hai usko maar doon"."""
    if _VIOLENCE_POSSIBLE.search(text or ""):
        return _reply(SafetyCategory.violence_risk, "harm_to_others_possible", lang)
    return CrisisCheck(False)


def check_violence_admission(text: str | None, lang: str | None = None) -> CrisisCheck:
    """The user says they have killed or seriously hurt someone ("Maine ek ko maar diya hai").
    Not assumed to be fiction. The reply puts the injured person first and says no to concealment."""
    t = text or ""
    en = _ADMISSION_EN.search(t)
    if en and not _STRONG_VERB.search(t) and not _SERIOUS.search(t):
        en = None
    if en or _ADMISSION_HI.search(t):
        return _reply(SafetyCategory.violence_aftermath, "violence_admission", lang)
    return CrisisCheck(False)


# ── an injured, bleeding, unconscious or not-breathing person ─────────
_BODY = (r"haddi|haddiyan|haddiyaan|haddiya|hadiyan|hadi|pasli|pasliyan|khopdi|sir|haath|hath|pair|taang|jabda|naak|dant|"
         r"kamar|gardan|ungli|ribs?|arm|leg|skull|jaw|bones?|nose|neck|spine|back|wrist|ankle|hip")
_KIN = (r"(?:father|mother|dad|mom|mum|brother|sister|son|daughter|husband|wife|friend|baby|child|kid|grandmother|grandfather|"
        r"uncle|aunt|colleague|neighbou?r|boyfriend|girlfriend|partner|dog)")
_KIN_HI = r"papa|pita|mummy|maa|mama|bhai|behen|beta|beti|pati|patni|dost|dada|dadi|nana|nani|chacha|chachi|bacha"
_INJURY = re.compile(
    rf"\b(?:{_BODY})\s+(?:\w+\s+)?(?:toot|tut|phat|tuut)\s*(?:gayi|gaya|gaye|gai|chuki|chuka|chuke|gayee|gya|gyi)\b|"
    rf"\b(?:broke|broken|fractured|cracked|smashed)\s+(?:his|her|their|my|the|a)?\s*(?:{_BODY})\b|"
    rf"\b(?:{_BODY})\s+(?:is|are|was|were|got|has\s+been|have\s+been)\s+(?:broken|fractured|cracked|smashed)\b|"
    rf"\b(?:khoon|blood)\s+(?:beh|bah|nikal|aa)\s*(?:raha|rahi|rha|rhi)\b|\bbahut\s+(?:zyada\s+)?khoon\b|"
    rf"\b(?:he|she|they|someone|somebody|my\s+{_KIN}|the\s+(?:man|woman|boy|girl|baby|child|driver|patient|guy))\s+"
    rf"(?:is|are|was)\s+(?:not\s+(?:moving|breathing|responding|waking|conscious)|unresponsive|unconscious|lifeless|"
    rf"bleeding(?!\s+(?:money|cash|profit|funds|losses))|badly\s+(?:hurt|injured)|lying\s+(?:on\s+the\s+)?(?:floor|ground|road))\b|"
    rf"\b(?:he|she|they|baby|child|{_KIN})\s+(?:won'?t|wouldn'?t|doesn'?t|isn'?t|is\s+not)\s+(?:wake|waking|waking\s+up|"
    rf"responding|breathing|moving)\b|"
    rf"\b(?:wo|woh|vo|use|usko|bachcha|baccha|{_KIN_HI})\b[^.?!]{{0,20}}\b(?:hil|hilta|hilti|jaag|hosh\s+mein\s+aa|saans\s+le|"
    rf"sans\s+le|uth)\s*(?:nahi|nhi)\s+(?:raha|rahi|rahe|rha|rhi|pa\s+raha|pa\s+rahi)\b|\b(?:nabz|pulse)\s+(?:nahi|nhi|not|stopped)\b|"
    rf"\b(?:behosh|bedosh)\s+(?:pada|padi|padha)\b|\b(?:zameen|zamin|road|sadak)\s+par\s+(?:pada|padi|gira|giri)\b|"
    rf"\b(?:head|sir)\s+(?:injury|mein\s+chot|par\s+chot|phat)\b|\bchot\s+lagi\s+hai\b[^.?!]{{0,30}}(?:gehri|bahut|serious)|"
    rf"\b(?:laash|dead\s+body)\b|\b(?:mujhe|hume|humein|usko|use)\s+bachao\b|\bkoi\s+(?:to\s+)?bachao\b|"
    rf"(?:हड्डी|हड्डियाँ|हड्डियां)\s*(?:\S+\s+)?टूट\s*(?:गई|गया|गए|गयीं|गईं)|"
    rf"खून\s*(?:बह|निकल)\s*(?:रहा|रही)|बेहोश\s*(?:पड़ा|पड़ी)|हिल\s*नहीं\s*(?:रहा|रही)", _I)
_FUTURE_QUESTION = re.compile(r"\b(?:hogi|hoga|hongi|honge|toot\s*(?:jayegi|jayega|jayenge|sakti|sakta)|will|would|"
                              r"fracture\s+(?:yog|hoga|hogi)|yog)\b", _I)


def check_injury_report(text: str | None, lang: str | None = None) -> CrisisCheck:
    """A person is injured or unresponsive right now ("Uski haddiyan toot gayi hain"): call for help
    first. Future-tense chart questions ("will I have a fracture?") are not this."""
    t = text or ""
    if _INJURY.search(t) and not (_FUTURE_QUESTION.search(t) and _ASTRO.search(t)):
        return _reply(SafetyCategory.medical_emergency, "injury_report", lang)
    return CrisisCheck(False)


# ── intent that is not specific: ask, do not assume ───────────────────
_UNCERTAIN = re.compile(
    r"\bsab\s+(?:kuch\s+)?khatam\s+kar\s*(?:dunga|dungi|doonga|doongi|lunga|lungi|loonga|loongi)\b|"
    r"\b(?:i'?m|i\s+am)\s+(?:going\s+to|gonna|about\s+to)\s+do\s+something\s+(?:stupid|crazy|drastic|terrible|bad|foolish)\b|"
    r"\bi\s+(?:did|have\s+done)\s+something\s+(?:terrible|horrible|awful|unforgivable|very\s+bad|really\s+bad)\b|"
    r"\bi\s+(?:can'?t|cannot)\s+(?:control|stop)\s+(?:my\s+anger|myself|what\s+i\s+might\s+do)\b|"
    r"\bi'?m\s+(?:scared|afraid)\s+(?:of\s+)?what\s+i\s+(?:might|will|could)\s+do\b|\b(?:i\s+)?won'?t\s+be\s+(?:a\s+problem|"
    r"around|here)\s+(?:for\s+anyone\s+)?(?:anymore|any\s*more|after\s+tonight|tomorrow)\b|\btonight\s+is\s+(?:the\s+)?(?:night|"
    r"my\s+last)\b|\bi\s+have\s+(?:made|taken)\s+(?:up\s+)?(?:my\s+)?(?:decision|a\s+decision)\b[^.?!]{0,30}\b(?:end|finish|over)\b|"
    r"\b(?:main|mai|mein)\s+kuch\s+(?:galat|bura|bada|ulta\s+seedha)\s+kar\s*(?:lunga|lungi|loonga|loongi|dunga|dungi|baithunga|"
    r"baithungi|doonga|doongi)\b|\bkuch\s+(?:galat|bura)\s+kar\s*(?:lunga|lungi|loonga|loongi|baithunga|baithungi)\b|"
    r"\bmaine\s+kuch\s+(?:bahut\s+)?(?:galat|bura|bhayanak)\s+kar\s+diya\b|\bmujhe\s+darr\s+hai\s+(?:ki\s+)?main\s+kuch\b|"
    r"\bmain\s+kuch\s+kar\s+(?:baithunga|baithungi)\b|\bmain\s+(?:ab\s+)?kisi\s+ke\s+liye\s+(?:problem|bojh|mushkil)\s+nahi\s+"
    r"(?:banunga|banungi)\b|\baaj\s+raat\s+(?:ke\s+baad|tak)\b[^.?!]{0,25}\b(?:nahi\s+(?:rahunga|rahungi)|khatam)\b", _I)


def check_uncertain(text: str | None, lang: str | None = None) -> CrisisCheck:
    """Something may be about to happen, and the wording does not say what or to whom. The reply
    asks whether they are safe and have taken or done anything, and gives 112 and the helplines."""
    if _UNCERTAIN.search(text or ""):
        return _reply(SafetyCategory.crisis_followup, "uncertain_high_risk", lang)
    return CrisisCheck(False)


# ── misspelled and slang self-harm ────────────────────────────────────
_SELF_HARM_VARIANTS = re.compile(
    r"\bmar\s*ja+[ou]+n+g[ai]\b|\b(?:jeena|jina)\s+(?:nhi|nahi|nai|nahin)\s+(?:hai|h|chahta|chahti)\b|"
    r"\bjeene\s+ka\s+(?:mann|man|dil)\s+(?:nhi|nahi|nai)\b|"
    r"\bsu+i*c+i*d+e*\w*|\bsu+i*s+i+d+e*\w*|\bsu+c+i+d+e*\b|\bsuicde\b|\bkms\b|"
    r"\bunaliv\w*\s+(?:myself|me)\b|\b(?:wanna|want\s+to|going\s+to|gonna)\s+unaliv\w*\b|\boff\s+myself\b|"
    r"\bkil+\s*my\s*self\b|\bkill\s+mysel\b|\bend\s+my\s*self\b|\bhang\s+my\s*self\b|\bjump\s+(?:off|from)\s+(?:the\s+)?"
    r"(?:roof|building|bridge|terrace|balcony)\b|\bslit\s+my\s+(?:wrists?|throat)\b|"
    r"\b(?:atmahtya|aatmahtya|aatmhatya|atmhatya|aatmahatya|atmahatya|aatma\s*hatya)\w*|"
    r"\bmaut\s+(?:chahiye|chahta|chahti|ki\s+dua|aa\s+jaye|aa\s+jaaye|mil\s+jaye)\b|\bmar\s+jaane\s+ka\s+(?:mann|man|dil|soch)\b|"
    r"\bmarne\s+ka\s+(?:mann|man|dil|soch|plan|irada)\b|\b(?:apni|apna)\s+jaan\s+(?:le\s*(?:lunga|lungi|loonga|loongi)|"
    r"de\s*(?:dunga|dungi|doonga|doongi))\b|\bkhud\s+ko\s+(?:khatam|maar|mita|nuksan|hurt)\b|"
    r"\bjeene\s+ki\s+(?:wajah|vajah|ummeed|umeed)\s+nahi\b|\bpankhe\s+(?:se|par)\s+(?:latak|lataka)\w*|"
    r"\bphaansi\s+(?:laga|lagaa)\w*|\bnas\s+kaat\w*", _I)


def check_self_harm_variants(text: str | None, lang: str | None = None) -> CrisisCheck:
    """Misspellings and slang the main rules do not list: "sucide", "kms", "unalive", "aatmhatya",
    "marne ka mann", "phaansi laga loonga". Chart vocabulary does not exempt these."""
    if _SELF_HARM_VARIANTS.search(text or ""):
        return _reply(SafetyCategory.self_harm_crisis, "self_harm_variant", lang)
    return CrisisCheck(False)


__all__ = ["check_injury_report", "check_lethality", "check_poison_intent", "check_self_harm_variants",
           "check_uncertain", "check_violence_admission", "check_violence_possible"]
