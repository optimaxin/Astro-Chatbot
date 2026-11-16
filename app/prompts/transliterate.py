"""Devanagari -> Roman, the way Hinglish speakers actually write it.

Used only as a last-resort repair for a Roman-script (Hinglish) reply that
still contains a few Devanagari words. It must produce what a person types
("aapko partner batate hain ki"), not a scholarly transliteration.

The first version of this module transliterated letter by letter and
produced exactly the spellings real users reported as broken: "aapako",
"paartanar", "maanasik kanekshan", "bataate hain kee". Two changes fix it:

1. A lexicon (app/prompts/hinglish.py) maps common words, and every English
   loanword written in Devanagari ("पार्टनर", "कनेक्शन"), to its normal Roman
   spelling ("partner", "connection"). Loanwords cannot be recovered by
   phonetic rules, so they are never guessed: an unknown word that looks like
   a loanword is reported, and the caller regenerates instead.
2. For other words, Hinglish conventions: schwa deletion ("aap-ko", not
   "aa-pa-ko"), long vowels written short after the first syllable ("batate",
   not "bataate"), word-final "ी" as "i" ("ki", not "kee").
"""

from __future__ import annotations

import re

_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ए": "e", "ऐ": "ai",
           "ओ": "o", "औ": "au", "ऋ": "ri", "ऑ": "o", "ॲ": "e"}
_MATRAS = {"ा": "aa", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai", "ो": "o",
           "ौ": "au", "ृ": "ri", "ॉ": "o", "ॅ": "e"}
_CONSONANTS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j",
               "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t",
               "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh",
               "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s",
               "ह": "h", "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f"}
_NUKTA_BASE = {"क": "क़", "ख": "ख़", "ग": "ग़", "ज": "ज़", "ड": "ड़", "ढ": "ढ़", "फ": "फ़"}
_NASAL = {"ं": "n", "ँ": "n"}
_MARKS = {"ः": "h", "।": ".", "॥": ".", "ॐ": "om"}
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}
_VIRAMA, _NUKTA = "्", "़"
# Words exclude the danda (।, ॥), which is punctuation ("है।" is "है" + ".").
_DEVANAGARI_WORD = re.compile(r"[ऀ-ॣ०-९ॱ-ॿ]+(?:-[ऀ-ॣ०-९ॱ-ॿ]+)*")


def _syllables(word: str) -> list[tuple[str, str | None]]:
    """[(consonant-or-vowel roman, vowel roman | None)], None = inherent schwa."""
    out: list[tuple[str, str | None]] = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if i + 1 < len(chars) and chars[i + 1] == _NUKTA and ch in _NUKTA_BASE:
            ch = _NUKTA_BASE[ch]
            i += 1
        if ch in _CONSONANTS:
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt in _MATRAS:
                out.append((_CONSONANTS[ch], _MATRAS[nxt]))
                i += 1
            elif nxt == _VIRAMA:
                out.append((_CONSONANTS[ch], ""))       # explicit: no vowel
                i += 1
            else:
                out.append((_CONSONANTS[ch], None))      # inherent "a", maybe deleted
        elif ch in _VOWELS:
            out.append(("", _VOWELS[ch]))
        elif ch in _NASAL and out:
            onset, vowel = out[-1]
            out[-1] = (onset, (vowel if vowel is not None else "a") + "n")
        elif ch in _MARKS:
            out.append((_MARKS[ch], ""))
        elif ch in _DIGITS:
            out.append((_DIGITS[ch], ""))
        i += 1
    return out


def _phonetic(word: str) -> str:
    syl = _syllables(word)
    parts: list[str] = []
    for k, (onset, vowel) in enumerate(syl):
        if vowel is None:
            last = k == len(syl) - 1
            prev_vowel = k > 0 and (syl[k - 1][1] not in ("",))
            next_has_vowel = k + 1 < len(syl) and syl[k + 1][1] not in (None, "")
            # Hindi schwa deletion: silent at the end of a word, and between a
            # vowel and a consonant that carries its own vowel (आपको -> aapko).
            after_cluster = k > 0 and syl[k - 1][1] == ""
            if last:
                vowel = "a" if after_cluster else ""        # भविष्य -> bhavishya, but घर -> ghar
            else:
                vowel = "" if (k > 0 and prev_vowel and next_has_vowel) else "a"
        elif k > 0 and vowel == "aa":
            vowel = "a"          # chat spelling: "batate", "jaankari" keeps only the first "aa"
        parts.append(onset + vowel)
    return "".join(parts)


def romanize_word(word: str) -> tuple[str, bool]:
    """(Roman spelling, known) — known=False when the word was guessed."""
    from app.prompts.hinglish import DEVANAGARI_LEXICON

    if word in DEVANAGARI_LEXICON:
        return DEVANAGARI_LEXICON[word], True
    if "-" in word:
        pieces = [romanize_word(p) for p in word.split("-")]
        return "-".join(p for p, _ in pieces), all(k for _, k in pieces)
    return _phonetic(word), False


def to_roman(text: str) -> str:
    """Replace every Devanagari word with its Roman (Hinglish) spelling."""
    from app.prompts.hinglish import DEVANAGARI_PHRASES

    out = text or ""
    for phrase, roman in DEVANAGARI_PHRASES.items():
        out = out.replace(phrase, roman)
    out = _DEVANAGARI_WORD.sub(lambda m: romanize_word(m.group(0))[0], out)
    return out.replace("।", ".").replace("॥", ".")


def unknown_words(text: str) -> list[str]:
    """Devanagari words the lexicon does not know (their spelling would be a guess)."""
    from app.prompts.hinglish import DEVANAGARI_LEXICON, DEVANAGARI_PHRASES

    stripped = text or ""
    for phrase in DEVANAGARI_PHRASES:
        stripped = stripped.replace(phrase, " ")
    return [w for w in _DEVANAGARI_WORD.findall(stripped)
            if w not in DEVANAGARI_LEXICON and not all(p in DEVANAGARI_LEXICON for p in w.split("-"))]


__all__ = ["to_roman", "romanize_word", "unknown_words"]
