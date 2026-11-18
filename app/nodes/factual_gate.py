"""Policy gates applied to the FINAL reply text, after every rewrite.

* `apply_dosha_policy` — the engine computes only Kaal Sarp and Sade Sati. For
  any other dosha, every sentence the model wrote about it is removed (a
  verdict in either direction, and also the hedged "lag sakta hai"), and the
  assistant's own localized notice says the calculation is unavailable.
  Missing data is never read as "you do not have it".
* `apply_integrity_notice` — when the chart data contradicts itself, say so, in
  the user's language, instead of silently answering around it.
* `final_gate` — one last validation of whatever is about to ship: a sentence
  that still contradicts the chart (or states a placement it cannot confirm) is
  dropped; if nothing usable is left the safe reply replaces the answer.

All deterministic; none of them calls a model.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from app.evaluation.chart_claims import split_sentences, strip_sentences, validate_reply
from app.evaluation.fact_claims import HARD_KINDS, mentions_computed_dosha, mentions_uncomputed_dosha
from app.kundli.chart_facts import ChartFacts
from app.safety.notices import dosha_unavailable, integrity_notice, is_system_sentence

MIN_KEEP_CHARS = 25


ORPHAN_CHARS = 40       # what is left after removal must still be an answer, not a fragment


def with_notice(text: str, notice: str) -> str:
    """The notice as its own paragraph, placed before a closing follow-up
    question so the conversation still ends on the question bubble."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paragraphs) > 1 and paragraphs[-1].rstrip().endswith(("?", "？")):
        return "\n\n".join([*paragraphs[:-1], notice, paragraphs[-1]])
    return "\n\n".join([*paragraphs, notice])


def apply_dosha_policy(text: str, question: str | None, style_id: str | None,
                       facts: ChartFacts | None) -> tuple[str, dict[str, Any]]:
    """Two cases.

    The user ASKED about an uncomputed dosha: the model's whole answer goes, bar
    sentences about the doshas the engine does compute (Kaal Sarp, Sade Sati).
    A Manglik answer that never says "Manglik" ("Mars is in your 1st house,
    classical practice lists several cancellations…") is still a verdict, and a
    placement offered next to the question invites the inference.

    The model brought one up on its own: only the sentences that mention it go.
    """
    sentences = split_sentences(text)
    bad = {i for i, s in enumerate(sentences) if mentions_uncomputed_dosha(s, facts)}
    asked = mentions_uncomputed_dosha(question or "", facts)
    noticed = any(is_system_sentence(s) for s in sentences)
    if not bad and not (asked and not noticed):
        return text, {}
    if asked:
        drop = {i for i, s in enumerate(sentences) if not is_system_sentence(s) and not mentions_computed_dosha(s, facts)}
    else:
        drop = bad
    kept = strip_sentences(text, drop).strip()
    if noticed:
        final = kept
    elif len(kept) < ORPHAN_CHARS:
        final = dosha_unavailable(style_id)
    else:
        final = with_notice(kept, dosha_unavailable(style_id))
    return final, {"action": "dosha_unavailable", "removed_sentences": len(drop), "asked": asked}


def apply_integrity_notice(text: str, style_id: str | None, facts: ChartFacts | None) -> tuple[str, dict[str, Any]]:
    if facts is None or not facts.available or not facts.integrity_problems:
        return text, {}
    notice = integrity_notice(style_id, bool(facts.house_sign_mismatch), list(facts.conflicts))
    if not notice or notice in text:
        return text, {}
    return with_notice(text, notice), {"problems": facts.integrity_problems}


def final_gate(text: str, facts: ChartFacts | None, style_id: str | None,
               fallback: Callable[[str | None], str]) -> tuple[str, dict[str, Any]]:
    """Nothing the validator can still contradict is allowed to ship."""
    if not text or facts is None or not facts.available:
        return text, {}
    result = validate_reply(text, facts)
    bad = ({c.sentence_index for c in result.contradictions}
           | {c.sentence_index for c in result.unsupported if c.kind in HARD_KINDS})
    if not bad:
        return text, {}
    kept = strip_sentences(text, bad)
    report = {"removed_sentences": len(bad)}
    if len(kept) < MIN_KEEP_CHARS:
        return fallback(style_id), {**report, "action": "safe_fallback"}
    return kept, {**report, "action": "stripped"}


def apply_final_policy(state: dict[str, Any], out: dict[str, Any],
                       fallback: Callable[[str | None], str]) -> dict[str, Any]:
    """The three gates, in order, on the text that is about to ship: dosha
    policy, then the last validation, then the chart-integrity notice (last, so
    nothing after it can strip or re-judge it)."""
    from app.kundli.chart_facts import build_chart_facts

    original = out.get("response", state.get("response") or "")
    if not original:
        return out
    style_id = state.get("response_style")
    kundli = state.get("kundli")
    facts = build_chart_facts(kundli) if kundli is not None else ChartFacts(available=False)
    from app.config.settings import settings

    text, dosha = apply_dosha_policy(original, state.get("question"), style_id, facts)
    # CHART_VALIDATION_MODE=log is record-only by design (rollback / measurement):
    # the dosha policy and the integrity notice are policy and stay on, but chart
    # claims are not stripped.
    gate: dict[str, Any] = {}
    if settings.flags.chart_validation_mode != "log":
        text, gate = final_gate(text, facts, style_id, fallback)
    text, integrity = apply_integrity_notice(text, style_id, facts)
    if text == original:
        return out
    validation = dict(out.get("validation") or {})
    for key, note in (("dosha_policy", dosha), ("final_gate", gate), ("integrity_notice", integrity)):
        if note:
            validation[key] = note
    return {**out, "response": text, "validation": validation}


__all__ = ["apply_dosha_policy", "apply_integrity_notice", "apply_final_policy", "final_gate", "MIN_KEEP_CHARS"]
