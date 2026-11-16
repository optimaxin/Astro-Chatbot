"""Near-miss detection for garbled Roman-Hinglish words.

Root cause (final quality gate): the model (a small Nemotron) sometimes
writes a word that is ONE slip away from a real one — "assar" (asar),
"aspekt" (aspect), "Mahadashi" (Mahadasha), "naahin" (nahin), "mileti"
(milti), "partnershi", "favorabil", "kiundli" — and nothing caught it: the
misspelling list in app/prompts/hinglish.py only knows forms seen before.

This module only DETECTS. A token is suspect when it is not a known word but
is within one or two edits of an anchor (a word the model is known to need:
astrology terms, common English words of Hinglish chat, everyday Hindi). The
sentence is then sent to the existing one-pass contextual correction
(app.nodes.language_quality), which decides the replacement in context and is
verified; a word list never rewrites the sentence itself. Words legitimately
close to an anchor ("disha" next to "dasha", "jodna" next to "judna") are in
KNOWN so they are never flagged.

Invented words that are near nothing ("samansooth") are not caught here; the
prompt tells the model to prefer the English word when unsure, and the
benchmark judge measures what remains.
"""

from __future__ import annotations

import re
from functools import lru_cache

_ANCHORS = """
mahadasha antardasha dasha kundli kundali lagna rashi nakshatra graha grah bhav yoga dosha gochar drishti
shubh ashubh karak swami navamsa planet planets house houses aspect aspects conjunction transit placement
surya chandra mangal budh guru brihaspati shukra shani rahu ketu sun moon mars mercury jupiter venus saturn
mesh vrishabh mithun kark simha kanya tula vrishchik dhanu makar kumbh meen aries taurus gemini cancer virgo
libra scorpio sagittarius capricorn aquarius pisces
career partnership relationship business personality opportunity opportunities commitment communication
professional financial education health family property marriage growth stability traditionally traditional
favourable favorable indication indicates indicator spiritual emotional practical support supportive period
timing impact effect challenges challenge confidence discipline responsibility harmony balance partner
integrate integral favour success successful
asar nahin nahi milti milta milte milegi milega milenge naukri mehnat zindagi shaadi jeevansathi sambandh
judta judti jude juda judi kaaran samay mahatva mahatvapurn kamyabi safalta pareshani dikkat samasya sehat
parivar aapki aapka aapke aapko hamesha zaroor zarurat jaankari matlab mushkil achha achhi bhagya labh
nuksan shanti swabhav vyaktitva kshamta pragati unnati vivah saathi rishta rishte mann dhyan sakta sakti
sakte rahega rahegi rahenge hoga hogi honge hota hoti hote karta karti karte deta deti dete dikhata dikhati
dikhate batata batati batate lagta lagti lagte isliye kyunki shuruaat zyada thoda bahut hisaab mutabik
anusaar samajh mahaul vichar shuruaati
"""

# Real words within an edit or two of an anchor — never flagged.
_NEAR_BUT_REAL = """
disha dasa dash desh daya dasham kundalini rashiyan rashiyon raashi nakshatron grahon grahas bhava bhaav bhavon
bhaav yog yogas yogon doshas dosh shubhh karaka karakatva karakon swamitva swamini navamsha planetary housing
aspected aspecting conjunctions transits placements placed surya chandrama mangalik shukr shanidev ketuke
mesha vrishabha mithuna karka simh kanyaa tulaa vrishchika dhanus makara kumbha meena cancerian careers
partnerships relationships businesses personalities commitments communications professionals finance
financially educational healthy families properties marriages grown growing traditions tradition favourably
favorably indications indicate indicated indicating indicators spiritually emotionally emotion emotions
practically supported supporting supports periods periodic impacts impacted effects effective challenged
challenging confident disciplined responsibilities responsible harmonious balanced partners integrated
favours favor favors successfully succeed
asara asal nahim milta milna milne mila mili mile milkar milan milenge milegi milega judna judne judkar jodta
jodti jodte joda jodi jode jodna jodne jodkar jodo kaarya kaaran karan samaya samayik mahatvapoorn mahatv
kamyaab safal pareshaan pareshaniyan dikkaten dikkatein samasyaen samasyayein samasyaon sehatmand parivaar
pariwar hamesha zaroori zaroorat zarurat zaruri jaankar jaankariyan matlab mushkilen mushkilein achhe achchha
acha accha acchi achhai bhagyashali bhaagya labha laabh nuksaan shaanti shant swabhaav swabhav vyakti kshamata
kshamtaen pragatisheel unnat vivaah vivahit saath saathi rishton rishtey mana mane mano dhyaan dhyaanse
sakta sakti sakte rahe raha rahi rahta rahti rahte rahenge rahegi rahega hogi hoga honge hona hone hoti hota
hote hua hui hue karna karne kiya kiye ki kar karke kare karein karo karenge karegi karega dena dene diya diye
dega degi denge dikhna dikhne dikha dikhi dikhe dikhta dikhti dikhte dikhai dikhaata dikhaati batana batane
bataya batayi bataye batao bataiye lagna lagne laga lagi lage lagega lagegi lagenge lagata lagati isiliye
kyonki kyuki shuru shuruat zyaada zyadatar thodi thode bahot hisab mutaabik anusar samjh samajhna samajhne
samajhkar samajhdaar mahal vichaar vichaaron vichaaro vicharon vichardhara shuruaat
the that this then than they them there these those their what when where which while with also into onto
over under after before about above below again against
mein june agar main mani chances hoon date banata banati banate banaata banaati badh badhata badhati badhate
badhega badhegi sahi shadi wealth chahenge chahega honi thora karni shakti lete leta dedication parivarik
samajik samaj samajhta samajhte samajhni samarth affect karma karam mood mool internal bhag bhagwan bhagna
bhagta dhana dhan dhane bujh lagan lagana lagate lagatar mitti profession professions sati stature aapne
choti chota home sake avsar mahatvapurna duration respect dikhane dikhana dikhakar dikhaya dikhaye dikhayi
dikhwane sambandhi sambandhit sambhav ability mean moti sikhata data deni deri grahan jyada laata laati
prabhav practices practice varta yahin yahi aakar avenues baithta baithte barhata barhati bhay concern
darshan darshak hongi ichha instability jankari jaankari jarurat jaroori judte judai judge kaamyaabi
kamiyaabi kamyab karwa karya mahurat marries mehnati nibhata nibhate nibhati pareshan parinam rakhi
shukriya teen aksar anya arise baate badalte bhai bhogi chota dikhwane hata kaaryon kahta kari julti labs
late loon mahatma makan many marg milke multi nadi nayi prosperity raho rahul rakhega rakhegi rakhenge
rahengi rasi saar sachchi samagra saman sambandho sasti sathi satta seen sidha suitability swabhavik uthenge
vesh vrishti vyaktitv wahi kyunke farak fark kaarak ladna aayenge padenge jatate karmi doon thoda batayen bataayen
"""

_WORD = re.compile(r"[A-Za-z]+")


@lru_cache(maxsize=1)
def _sets() -> tuple[frozenset[str], frozenset[str]]:
    anchors = frozenset(_ANCHORS.split())
    return anchors, anchors | frozenset(_NEAR_BUT_REAL.split())


def _distance(a: str, b: str, limit: int) -> int:
    """Optimal-string-alignment distance, stopping early past `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] | None = None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            if prev2 is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


def _inflection_of_known(word: str, known: frozenset[str]) -> bool:
    """English plural/verb forms and Hindi plural endings of a known word."""
    for suffix in ("s", "es", "ed", "ing", "ly", "on", "en", "ein", "yon", "iyan", "iyon", "wala", "wali", "wale"):
        if word.endswith(suffix) and word[:-len(suffix)] in known:
            return True
    return False


def suspect_words(text: str) -> list[tuple[str, str]]:
    """(word as written, the anchor it is a slip of) for each suspect token."""
    anchors, known = _sets()
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for token in _WORD.findall(text or ""):
        word = token.lower()
        if len(word) < 4 or word in known or word in seen or _inflection_of_known(word, known):
            continue
        seen.add(word)
        limit = 1 if len(word) < 7 else 2
        best = min(((_distance(word, a, limit), a) for a in anchors if abs(len(a) - len(word)) <= limit),
                   default=(limit + 1, ""))
        if 0 < best[0] <= limit:
            out.append((token, best[1]))
    return out


__all__ = ["suspect_words"]
