"""Text normalisation applied before any factual validation.

`normalize_numerals` maps Devanagari digits to ASCII. The substitution is one
character for one character, so sentence boundaries and sentence indices of
the normalised copy are identical to the original's: a claim found in the copy
can still be stripped from the original by index.

Found in the 2026-11-18 live run: a Hindi reply stating "७वें भाव" or "२०२७"
produced no claim at all, because every validator pattern is written with
ASCII digits.
"""

from __future__ import annotations

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_numerals(text: str | None) -> str:
    return (text or "").translate(_DEVANAGARI_DIGITS)


__all__ = ["normalize_numerals"]
