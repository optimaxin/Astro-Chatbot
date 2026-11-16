"""Language-quality step of validate_reply (runs after the language gate).

    reply (language already right)
      -> detect sentence-level grammar/naturalness issues
      -> deterministic fixes where the context decides (no model call)
      -> ONE targeted correction call for the sentences still flagged
      -> per-sentence verification (chart facts, certainty, length, better)
      -> a still-unreadable sentence is dropped if the answer stands without
         it; an answer that cannot be repaired becomes the safe reply
      -> segmentation happens later, on the final text, as before.

Never loops: at most one model call per reply. Paragraph breaks — and so the
message boundaries — are kept, because sentences are replaced in place.
HINGLISH_GRAMMAR=off|detect|deterministic|correct (default correct).
"""

from __future__ import annotations

import json
from typing import Any

from app.config.settings import settings
from app.logging_setup import get_logger

log = get_logger("nodes.language_quality")

_STYLE_LANGUAGE = {"whatsapp_hinglish": "hinglish", "hindi": "hindi"}

CORRECTION_PROMPT = """You are a careful copy-editor for natural Roman-script Hinglish, as an educated Indian
astrologer writes it in chat. You will get numbered sentences, each with the problems found in it.

Fix ONLY those problems. For each sentence:
- keep its meaning, and keep EVERY planet, house number, sign, date and number exactly as written;
- add no new claim, no certainty ("pakka", "definitely"), and no new information;
- keep about the same length; Roman script only; address the user as "aap";
- keep common English words as people say them (career, partnership, business, relationship, health,
  property, Mahadasha, Antardasha, growth) — do not translate them into Hindi;
- natural forms: "dikkat aa sakti hai", "Saturn ke kaaran", "juda hota hai", "aapki Mercury Mahadasha chal rahi hai".

Return strict JSON only: {"1": "corrected sentence", "2": "..."} with the same numbers."""


# Added to the system prompt of a Hinglish turn (app/prompts/builder.py) while
# the layer is on: the generation-side half of the fix.
GRAMMAR_GUIDANCE = (
    "HINGLISH GRAMMAR:\n"
    "- Agreement: Dasha, Mahadasha, Antardasha, dikkat, samasya, naukri, shaadi and growth are feminine, and the "
    "Dasha is the subject: \"Abhi aapki Mercury Mahadasha chal rahi hai\" (not \"Abhi aap Mercury Mahadasha chal "
    "rahe hain\"); \"dikkat aa sakti hai\" (not \"aa sakta hai\").\n"
    "- Never translate an English passage word for word: \"Saturn ke kaaran health-related dikkat aa sakti hai\" "
    "(not \"Saturn ka effect health issues la sakta hai\"); \"domestic harmony mein dikkat aa sakti hai\" (not "
    "\"ke nuksan de sakta hai\"); \"abroad se juda hota hai\" (not \"joda hota hai\"); \"samay-samay par deri\" "
    "(not \"delays\").\n"
    "- Keep English terms as people say them: career, partnership, business, relationship, health, property, "
    "Mahadasha, Antardasha, planetary placement, spiritual growth, communication, professional growth.")


def _mode() -> str:
    return getattr(settings.flags, "hinglish_grammar", "correct")


def grammar_guidance(language: str | None) -> str | None:
    return GRAMMAR_GUIDANCE if language == "hinglish" and _mode() != "off" else None


def apply_language_quality(state: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    from app.evaluation.hinglish_grammar import LAYER_VERSION, correct_deterministic, detect

    mode = _mode()
    language = _STYLE_LANGUAGE.get((state.get("response_style") or "").lower())
    text = out.get("response", state.get("response") or "")
    if mode == "off" or language is None or not text or state.get("generation_failed"):
        return out
    report: dict[str, Any] = {"version": LAYER_VERSION, "language": language, "issues_before": 0}
    issues = detect(text, language)
    report["issues_before"] = len(issues)
    report["categories"] = sorted({i.category for i in issues})
    fixed = text
    if mode in ("deterministic", "correct") and issues:
        fixed, fixes = correct_deterministic(text, language)
        report["deterministic_fixes"] = fixes
        issues = detect(fixed, language)
    if mode == "correct" and issues and language == "hinglish":
        fixed, llm = _targeted_correction(state, fixed, issues, language)
        report.update(llm)
        issues = detect(fixed, language)
    if mode in ("deterministic", "correct") and any(i.severe for i in issues):
        fixed, dropped = _drop_unreadable(fixed, issues, state)
        report["dropped_sentences"] = dropped
        issues = detect(fixed, language)
    report["issues_after"] = len(issues)
    validation = dict(out.get("validation") or {})
    validation["language_quality"] = report
    result = {**out, "validation": validation}
    if mode != "detect" and fixed != text:
        result["response"] = fixed
    return result


def _targeted_correction(state: dict[str, Any], text: str, issues, language: str) -> tuple[str, dict[str, Any]]:
    """One model call for the flagged sentences only; each answer verified."""
    from app.evaluation.chart_claims import split_sentences
    from app.evaluation.hinglish_grammar import verify_sentence
    from app.llm.llm_factory import ROLE_MAIN, get_llm

    sentences = split_sentences(text)
    flagged: dict[int, list[str]] = {}
    for issue in issues:
        if issue.sentence < len(sentences):
            flagged.setdefault(issue.sentence, []).append(f"{issue.category}: \"{issue.span}\" ({issue.hint})")
    numbered = {str(n + 1): idx for n, idx in enumerate(sorted(flagged))}
    user = "\n".join(f"{n}. {sentences[idx]}\n   problems: {'; '.join(flagged[idx])}" for n, idx in numbered.items())
    stats: dict[str, Any] = {"llm_sentences": len(numbered), "llm_accepted": 0, "llm_rejected": []}
    try:
        raw = get_llm(role=ROLE_MAIN).complete(
            [{"role": "system", "content": CORRECTION_PROMPT}, {"role": "user", "content": user}],
            session_id=state.get("session_id", "-"), user_id=state.get("user_id", "-"),
            request_id=state.get("request_id", "-"), temperature=0.0).text
        answers = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception as exc:  # noqa: BLE001 — the reply stands as it was
        log.warning("language quality correction failed", extra={"error_type": type(exc).__name__})
        stats["llm_error"] = type(exc).__name__
        return text, stats
    out = text
    for n, idx in numbered.items():
        original, corrected = sentences[idx], str(answers.get(n) or "").strip()
        verdict = verify_sentence(original, corrected, language)
        if verdict.accepted and original in out:
            out = out.replace(original, corrected, 1)
            stats["llm_accepted"] += 1
        else:
            stats["llm_rejected"].append(verdict.reason)
    return out, stats


def _drop_unreadable(text: str, issues, state: dict[str, Any]) -> tuple[str, int]:
    """A sentence that is still unreadable is removed when the answer stands
    without it; otherwise the honest safe reply replaces the answer."""
    from app.evaluation.chart_claims import split_sentences, strip_sentences

    bad = {i.sentence for i in issues if i.severe}
    remaining = len(split_sentences(text)) - len(bad)
    if remaining >= 1:
        return strip_sentences(text, bad), len(bad)
    from app.nodes.validate import safe_fallback

    return safe_fallback(state.get("response_style")), len(bad)


__all__ = ["apply_language_quality", "CORRECTION_PROMPT"]
