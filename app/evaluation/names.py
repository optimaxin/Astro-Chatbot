"""Never address the user by a name nobody gave us.

Live 2026-11-19: "Rahul, aapko pyaar mein patience rakhni chahiye…" was sent to a
user whose profile name is "Test User" and who never mentioned a name. A name
may be used only when the verified profile (the chart's birth record) or the
user's own messages supply it; any other vocative name is removed, with the
sentence kept.

Detection is deliberately narrow — a name used to ADDRESS the user:
  "Rahul, aapko …" / "Rahul ji, …" / "Dear Rahul, …" / "Hi Priya, …" / "राहुल जी, …"
A capitalised word before a comma is only a name candidate when it is followed
by a second-person word (you/aap/आप…) or carries a greeting or "ji"; discourse
words ("So, you…", "Traditionally, aapka…") and chart vocabulary never count.
"""

from __future__ import annotations

import re
from typing import Any

from app.evaluation.chart_claims import GRAHA_ALIASES, SIGN_ALIASES

_SECOND_PERSON = (r"(?i:you|your|you're|aap|aapko|aapka|aapki|aapke|aapne|aapse|tum|tumhe|tumhara|tumhari|"
                  r"आप|आपको|आपका|आपकी|आपके|आपने)\b")
_GREETING = r"(?i:dear|hi|hello|hey|namaste|namaskar)"
# Sentence start (or after a blank line / sentence end).
_START = r"(?:^|(?<=[.!?।]\s)|(?<=\n))"
# The name itself is case-sensitive (a capitalised word); greeting, "ji" and the
# second-person word are not.
_LATIN_VOCATIVE = re.compile(
    rf"{_START}[ \t]*(?:(?P<greet>{_GREETING})[ \t]+)?(?P<name>[A-Z][a-z]{{2,}})(?P<ji>[ \t]+(?i:ji))?[ \t]*,[ \t]*"
    rf"(?=(?P<next>{_SECOND_PERSON})?)", re.M)
_MARK = "\x00"
_DEVANAGARI_VOCATIVE = re.compile(rf"{_START}[ \t]*(?:नमस्ते[ \t]+)?(?P<name>[ऀ-ॿ]{{2,}})[ \t]+जी[ \t]*,[ \t]*", re.M)

_NOT_NAMES = frozenset("""
    so yes no also however overall currently traditionally practically honestly well now today then next first
    second third finally otherwise instead meanwhile therefore additionally remember note okay ok sure together
    again plus still sadly luckily ideally generally basically simply actually besides because since while if
    when where here there this that these those my our their his her its astrology kundli chart career love
    marriage money health life work family friends dosha dasha mahadasha antardasha lagna rashi nakshatra house
    haan ji abhi isliye lekin aur toh ab sach filhaal waise phir dekhiye suniye shayad bilkul agar jab chalo
    achha accha theek zaroor kyunki matlab yaani bas
""".split()) | {w for ws in (*GRAHA_ALIASES.values(), *SIGN_ALIASES.values()) for w in ws if w.isascii()}
_NOT_NAMES_DEVANAGARI = frozenset({"गुरु", "पंडित", "आचार्य", "बाबा", "माता", "पिता", "भाई", "दीदी", "स्वामी"})

_STATED_NAME = re.compile(r"\b(?:my name is|i am|i'm|this is|mera naam|naam hai)\s+([A-Za-z]{2,})"
                          r"|(?:मेरा नाम)\s+([ऀ-ॿ]{2,})", re.I)


def allowed_names(kundli: Any = None, user_messages: list[str] | None = None) -> set[str]:
    """Names the user is verifiably called: the profile name and any name they gave themselves."""
    names: set[str] = set()
    birth = getattr(kundli, "birth", None) or {}
    for part in re.findall(r"[A-Za-zऀ-ॿ]{2,}", str(birth.get("name") or "")):
        names.add(part.lower())
    for text in user_messages or []:
        for m in _STATED_NAME.finditer(text or ""):
            names.add((m.group(1) or m.group(2)).lower())
    return names


def _is_vocative(m: re.Match[str]) -> bool:
    name = m.group("name")
    if name.lower() in _NOT_NAMES or name.lower().endswith("ly"):
        return False
    return bool(m.group("greet") or m.group("ji") or m.group("next"))


def invented_names(text: str, allowed: set[str] | None = None) -> list[str]:
    allowed = {a.lower() for a in allowed or set()}
    found = [m.group("name") for m in _LATIN_VOCATIVE.finditer(text or "") if _is_vocative(m)]
    found += [m.group("name") for m in _DEVANAGARI_VOCATIVE.finditer(text or "")
              if m.group("name") not in _NOT_NAMES_DEVANAGARI]
    return [n for n in found if n.lower() not in allowed]


def strip_invented_names(text: str, allowed: set[str] | None = None) -> tuple[str, list[str]]:
    """Remove each invented vocative ("Rahul, aapko…" -> "Aapko…"); the sentence stays."""
    allowed = {a.lower() for a in allowed or set()}
    removed: list[str] = []

    def drop(m: re.Match[str], devanagari: bool = False) -> str:
        name = m.group("name")
        keep = (name in _NOT_NAMES_DEVANAGARI) if devanagari else not _is_vocative(m)
        if keep or name.lower() in allowed:
            return m.group(0)
        removed.append(name)
        lead = m.group(0)[:len(m.group(0)) - len(m.group(0).lstrip())]
        return lead + _MARK                         # keep leading whitespace; mark where the sentence now starts

    out = _LATIN_VOCATIVE.sub(drop, text or "")
    out = _DEVANAGARI_VOCATIVE.sub(lambda m: drop(m, True), out)
    # Re-capitalise only the word that now starts an edited sentence.
    out = re.sub(f"{_MARK}(.?)", lambda m: m.group(1).upper(), out)
    return out, removed


def apply_name_guard(state: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """validate_reply step: remove names the user never gave."""
    text = out.get("response", state.get("response") or "")
    users = [getattr(m, "content", "") for m in state.get("history") or [] if getattr(m, "role", "") == "user"]
    allowed = allowed_names(state.get("kundli"), [*users, state.get("question") or ""])
    fixed, removed = strip_invented_names(text, allowed)
    if not removed:
        return out
    validation = dict(out.get("validation") or {})
    validation["invented_names_removed"] = removed
    return {**out, "response": fixed, "validation": validation}


__all__ = ["allowed_names", "invented_names", "strip_invented_names", "apply_name_guard"]
