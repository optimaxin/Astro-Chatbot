"""Hinglish reference pairs: an awkward sentence and its natural rewrite.

Used to (1) regression-test the deterministic normaliser on the mechanical
cases and (2) calibrate the LLM judge: a judge that cannot tell these apart
cannot be trusted to score Hinglish quality.

REVIEW STATUS: drafted from real chatbot output and the examples in the
2026-11 quality briefs. These have NOT yet been reviewed by a native speaker;
`reviewed_by` stays empty until someone does, and the calibration report
prints the status so nobody mistakes this for a reviewed set.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HinglishPair:
    awkward: str
    natural: str
    issue: str                    # spelling | transliteration | grammar | word_choice | tone | kb_copy
    mechanical: bool = False      # fixable by app.prompts.hinglish.normalize_hinglish
    reviewed_by: str = ""         # native-speaker reviewer, once reviewed


PAIRS: list[HinglishPair] = [
    HinglishPair("Aapako partner pasand aaega jo soch-samajh ke bolta hai.",
                 "Aapko partner pasand aaega jo soch-samajhkar baat karta hai.", "transliteration", True),
    HinglishPair("Jyotishi bataate hain kee yeh period achha hai.",
                 "Jyotishi batate hain ki yeh period achha hai.", "transliteration", True),
    HinglishPair("Aapke paartanar ke saath committment aur maanasik kanekshan zaroori hai.",
                 "Aapke partner ke saath commitment aur emotional connection zaroori hai.", "spelling", True),
    HinglishPair("Aapka career mein growth ki opportunities ban sakti hain.",
                 "Aapke career mein growth ki opportunities ban sakti hain.", "grammar", True),
    HinglishPair("Isse aap emotionally sensitive ho sakti hain.",
                 "Isse aap emotionally sensitive ho sakte hain.", "grammar", True),
    HinglishPair("Yeh period marriage ke liye achha indications deta hai.",
                 "Traditional astrology mein yeh period marriage ke liye supportive maana jata hai.", "grammar"),
    HinglishPair("Aapka career mein opportunities hone ke chances hain.",
                 "Aapke career mein growth ki opportunities ban sakti hain.", "grammar"),
    HinglishPair("Venus Antardasha ke effects ke hisaab se, aap Charitable Inclination, Beautiful Clothing aur "
                 "Artistic Communication ka experience kar sakte hain.",
                 "Venus Antardasha mein aap creativity, achhe kapdon aur dusron ki madad karne ki taraf zyada "
                 "jhukaav mehsoos kar sakte hain.", "kb_copy"),
    HinglishPair("Traditionally, 7th house ke lord ki 6th mein rahna ka matlab yeh hai ki partner working "
                 "professional hoga.",
                 "Traditionally, 7th lord ka 6th house mein hona yeh dikhata hai ki partner working professional "
                 "ho sakta hai.", "grammar"),
    HinglishPair("Aapki Kundli ke according, aapke Kundli ke according 10th house ke according career achha hai.",
                 "Aapki Kundli mein 10th house career ke liye supportive dikhta hai.", "tone"),
    HinglishPair("Sambandh ke sandarbh mein sambhavnayein prabal hain.",
                 "Relationship ke context mein achhi possibilities dikh rahi hain.", "word_choice"),
    HinglishPair("Aapka jeevansathi ek karmath aur kartavyanishth vyakti hoga.",
                 "Aapka partner mehnati aur responsible ho sakta hai.", "word_choice"),
    HinglishPair("Shaadi kab hogi, yeh dikhane ke liye current running Antardasha ke dates ka use karna important "
                 "hai.",
                 "Shaadi ki timing ke liye current Antardasha ki dates dekhi jaati hain.", "word_choice"),
    HinglishPair("Aapko partner ke saath samjh-samajh kar baat karne ka chance deta hai.",
                 "Yeh period partner ke saath samajhdari se baat karne ka mauka deta hai.", "grammar"),
    HinglishPair("Mercury 6th house mein situate hai.", "Mercury 6th house mein hai.", "word_choice"),
    HinglishPair("Abhi aap Mercury Mahadasha chal rahi hai.", "Abhi aapki Mercury Mahadasha chal rahi hai.",
                 "grammar"),
    HinglishPair("Partner ka nature thora tough aur demanding hoga, guaranteed.",
                 "Partner ka nature thoda strict ya demanding ho sakta hai.", "tone"),
    HinglishPair("Kripya dhyan dein ki aapke jeevan mein vivah ka yog prabal hai.",
                 "Aapki Kundli mein marriage ke yog achhe dikh rahe hain.", "tone"),
    HinglishPair("Yeh combination typically spouse ko working professional dikhata hai lekin problems ya "
                 "differences ki samasya dikhata hai.",
                 "Yeh combination traditionally ek working professional partner dikhata hai, saath hi kuch "
                 "differences ki possibility bhi.", "grammar"),
    HinglishPair("Aapke liye kariyar mein fainenshiyal growth ke oporchunities hain.",
                 "Aapke career mein financial growth ki opportunities dikh rahi hain.", "spelling"),
]

REVIEW_STATUS = ("draft — not yet native-speaker reviewed"
                 if not all(p.reviewed_by for p in PAIRS) else "native-speaker reviewed")

__all__ = ["HinglishPair", "PAIRS", "REVIEW_STATUS"]
