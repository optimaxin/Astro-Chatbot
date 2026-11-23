"""Spelling variants of a message, for the deterministic safety rules.

A rule written for "suicide" does not see "su1c1de", "s u i c i d e" or "suuuicide". Rather than
widening every pattern, each rule set is run over the raw text AND over these variants. The variants
only ever ADD matches (the raw text is always screened), so a normalisation can never hide a message.

    NFKC + zero-width characters removed     full-width letters, invisible characters
    leet between letters                     p0ison, su1c1de, z3her, k!ll
    spaced / dotted letters joined           s u i c i d e, s.u.i.c.i.d.e, p-o-i-s-o-n
    runs of 3+ identical letters shortened   suuuicide -> suuicide, killlll -> kill
"""

from __future__ import annotations

import re
import unicodedata

_INVISIBLE = re.compile("[​-‏‪-‮⁠﻿­]")
_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"}
_LEET_BETWEEN_LETTERS = re.compile(r"(?<=[A-Za-z])[013457@$!](?=[A-Za-z])")
_SPACED_LETTERS = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][ .\-_*]){3,}[A-Za-z](?![A-Za-z])")
_LONG_RUN = re.compile(r"([A-Za-z])\1{2,}")


def _join_spaced(match: re.Match) -> str:
    return re.sub(r"[ .\-_*]", "", match.group(0))


def variants(text: str | None) -> list[str]:
    """The text itself first, then each distinct normalised form (at most three)."""
    raw = text or ""
    if not raw:
        return [raw]
    base = _INVISIBLE.sub("", unicodedata.normalize("NFKC", raw))
    out = [raw]
    stages = [base,
              _LEET_BETWEEN_LETTERS.sub(lambda m: _LEET[m.group(0)], base)]
    stages.append(_SPACED_LETTERS.sub(_join_spaced, stages[-1]))
    stages.append(_LONG_RUN.sub(lambda m: m.group(1) * 2, stages[-1]))
    for form in stages:
        if form not in out:
            out.append(form)
    return out


__all__ = ["variants"]
