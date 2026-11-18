"""Validate the draft reply before anyone sees it — synchronously, every turn.

Two deterministic checks run on the generated draft:

1. **Chart claims** (`app.evaluation.chart_claims`) against the Kundli
   engine's facts. A contradicted claim must not reach the user.
2. **Script** (`app.prompts.language`): the reply is in the script the user
   wrote in (e.g. not Devanagari for a Roman-script Hinglish question).

What happens on a failure is `CHART_VALIDATION_MODE`:

* `repair` (default) — ONE corrective regeneration with the main model, told
  exactly what was wrong and what the chart says. The rewrite is validated
  again; any contradiction that survives is stripped sentence by sentence.
* `strip`  — contradicted sentences are removed; no extra model call.
* `log`    — nothing is changed; the result is recorded for evaluation.

Validation itself has no "off" switch. Unsupported (not contradicted)
interpretive claims (aspects, yogas, transits) are rewritten only in repair
mode and never stripped: they are not known to be false, only not
verifiable. An unsupported *placement* (planet's house/sign, a house's sign,
the Lagna) that survives the repair is stripped: the chart cannot confirm it,
so it was invented.
"""

from __future__ import annotations

import re
from typing import Any

from app.config.settings import settings
from app.evaluation.chart_claims import (
    ValidationResult,
    split_sentences,
    strip_sentences,
    validate_reply,
)
from app.evaluation.rewrite_guard import check_repair, compare_rewrite
from app.evaluation.text_normalize import normalize_numerals
from app.kundli.chart_facts import ChartFacts, build_chart_facts
from app.llm.llm_factory import ROLE_MAIN, get_llm
from app.llm.usage_tracker import llm_purpose
from app.prompts.builder import PROMPT_VERSION
from app.logging_setup import get_logger
from app.observability.trace import span
from app.prompts.language import (
    copied_example_indices,
    deflection,
    has_placeholder,
    leaks_prompt,
    reply_matches,
)
from app.rag.grounding import process_citations, source_title, verification_messages
from app.nodes.repair_prompts import _LANGUAGE_NAME, _correction_note, _repair_messages  # noqa: F401
from app.nodes.validate_chain import resolve_after_repair, without_dosha
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.validate")

MIN_KEEP_CHARS = 25
# Shown only if repair and stripping together leave nothing usable.
SAFE_FALLBACK = ("I want to be precise about your chart, and I couldn't phrase that answer reliably "
                 "just now. Could you ask it once more?")


def _language(state: ChatbotState) -> str:
    from app.prompts.language_policy import LANGUAGE_OF_STYLE

    return LANGUAGE_OF_STYLE.get((state.get("response_style") or "").lower(), "english")


def validate_reply_node(state: ChatbotState) -> dict[str, Any]:
    """Chart/language/leak validation, deterministic quality fixes, then
    citation processing of the text that will actually be sent."""
    pre: dict[str, Any] = {}
    if not state.get("generation_failed") and state.get("response"):
        # Terminology clean-up first (hybrid tokens, Devanagari glosses in a
        # Latin reply): deterministic, and it spares a language regeneration
        # for a reply whose only fault was "(करियर)"-style glosses.
        from app.prompts.terminology import normalize_terms

        cleaned, fixes = normalize_terms(state["response"], _language(state))
        if fixes:
            state = {**state, "response": cleaned}
            pre = {"response": cleaned}
    out = {**pre, **_validate(state)}
    kundli = state.get("kundli")
    if not state.get("generation_failed") and (kundli is None or not getattr(kundli, "available", False)):
        # Again on the FINAL text: a language/leak repair can itself write a
        # chart reading (seen live), and there is no chart to check it against.
        guarded = _guard_without_chart(out.get("response", state.get("response") or ""),
                                       state.get("response_style"))
        if guarded is not None:
            out = {**out, "response": guarded["response"],
                   "validation": {**(out.get("validation") or {}), **guarded["validation"]}}
    if not state.get("generation_failed"):
        out = _drop_copied_examples(state, out)
    out = _cite(state, out)
    if not state.get("generation_failed"):
        out = _quality_fixes(state, out)
        out = _language_gate(state, out)
        from app.nodes.language_quality import apply_language_quality

        out = apply_language_quality(state, out)
        from app.prompts.followup import apply_followup

        out = apply_followup(state, out)
        from app.nodes.factual_gate import apply_final_policy

        out = apply_final_policy(state, out, safe_fallback)
    return out


def _segments_ok(style_id: str | None, text: str) -> bool:
    """The whole reply and each bubble it will be shown as are in the
    required language (a reply can pass as a whole while one bubble is in
    another language)."""
    from app.prompts.segments import segment_reply

    if not reply_matches(style_id, text):
        return False
    return all(_segment_language_ok(style_id, seg) for seg in segment_reply(text))


_HINDI_FUNCTION = re.compile(r"\b(?:hai|hain|mein|ka|ki|ke|ko|se|aur|bhi|yeh|ye|jo|aap|aapka|aapki|aapke|aapko|"
                             r"nahi|tak|par|hoga|hogi|sakta|sakti|sakte|raha|rahi|rahe|liye)\b", re.I)


def _segment_language_ok(style_id: str | None, segment: str) -> bool:
    """One bubble in the required language. Hindi and English are judged
    exactly as a whole reply is. A Hinglish bubble may lean on English terms
    ("Venus Antardasha June 2024 se April 2027 tak"), so it fails only when it
    contains Devanagari or is a real English sentence with no Hindi at all —
    the stricter whole-reply rule here sent the round-4 screenshot case to
    the fallback."""
    if (style_id or "") != "whatsapp_hinglish":
        return reply_matches(style_id, segment)
    if re.search(r"[ऀ-ॿ]", segment):
        return False
    words = re.findall(r"[A-Za-z]+", segment)
    return len(words) < 12 or bool(_HINDI_FUNCTION.search(segment))


def _language_gate(state: ChatbotState, out: dict[str, Any]) -> dict[str, Any]:
    """Last check before delivery. A reply still in the wrong language gets ONE
    more language-only rewrite; if that fails too, the user gets the honest
    fallback in their own language rather than a mixed-language answer."""
    style_id = state.get("response_style")
    text = out.get("response", state.get("response") or "")
    if not text or _segments_ok(style_id, text):
        return out
    from app.evaluation.response_quality import apply_safe_fixes

    validation = dict(out.get("validation") or {})
    note = _correction_note(ValidationResult(), True, style_id, False)
    messages = _repair_messages(state, text, note, True)
    try:
        with span("validate.language_gate"), llm_purpose("language_rewrite", f"{PROMPT_VERSION}+language"):
            rewritten = get_llm(role=ROLE_MAIN).complete(
                messages, session_id=state["session_id"], user_id=state["user_id"],
                request_id=state["request_id"]).text
        rewritten, _ = apply_safe_fixes(rewritten, _language(state), state.get("intent"), state.get("question"))
    except Exception as exc:  # noqa: BLE001
        log.warning("language gate rewrite failed", extra={"error_type": type(exc).__name__})
        rewritten = ""
    facts_ok = True
    kundli = state.get("kundli")
    if rewritten:
        # A rewrite may change the language, nothing else: same facts, same
        # Dasha tense, same dates, no new claim, same certainty. If it fails
        # the original is kept; there is no second rewrite.
        verdict = compare_rewrite(text, rewritten, build_chart_facts(kundli) if kundli is not None else None)
        facts_ok = verdict.ok
        if not verdict.ok:
            validation["language_gate_rewrite_rejected"] = verdict.reasons[:4]
    if rewritten and facts_ok and _segments_ok(style_id, rewritten) and not leaks_prompt(rewritten):
        validation["language_gate"] = "rewritten"
        return {**out, "response": rewritten, "validation": validation}
    # Neither candidate passes the per-bubble check. If one still passes the
    # whole-reply language check, deliver it: in the degraded-NVIDIA benchmark
    # run the rewrite call itself timed out and 7 correct-language Hindi
    # answers were replaced by "please ask again" for a single weak bubble.
    for label, candidate in (("rewritten", rewritten if facts_ok else ""), ("original", text)):
        if candidate and reply_matches(style_id, candidate) and not leaks_prompt(candidate):
            validation["language_gate"] = f"{label}_whole_reply_ok"
            return {**out, "response": candidate, "validation": validation}
    validation["language_gate"] = "fallback"
    log.warning("reply failed the language gate twice; sending the fallback")
    return {**out, "response": safe_fallback(style_id), "validation": validation}


_TAG_ATTRIBUTION = re.compile(
    r"(?:\b(?:according to|as per)\s+)((?:\[\s*K\s*\d+\s*\])+)\s*,?\s*"
    r"|((?:\[\s*K\s*\d+\s*\])+)\s*(?:के\s+अनुसार|के\s+हिसाब\s+से|ke\s+anusaar|ke\s+hisaab\s+se|ke\s+according)\s*,?",
    re.I)


def _cite(state: ChatbotState, out: dict[str, Any]) -> dict[str, Any]:
    if state.get("generation_failed") or not settings.flags.grounded_citations:
        return out
    text = out.get("response", state.get("response") or "")
    retrieval = state.get("retrieval")
    chunks = list(getattr(retrieval, "chunks", []) or [])
    display = settings.flags.citation_display
    report_extra: dict[str, Any] = {}
    text, report_extra = _verify_grounding(state, text, chunks)
    if report_extra.get("changed"):
        out = {**out, "response": text}
    if display != "inline":
        # An attribution phrase attached to a tag loses its subject when the tag
        # is removed: "इस स्थिति को [K1] के अनुसार …" became "इस स्थिति को के अनुसार"
        # (live browser test). The phrase goes with the tag.
        text = _TAG_ATTRIBUTION.sub("", text)
    result = process_citations(text, len(chunks), "K", keep_inline=display == "inline")
    citations = [
        {"id": f"{chunks[i - 1].document}::{chunks[i - 1].topic}",
         "title": source_title(chunks[i - 1].document, chunks[i - 1].topic)}
        for i in result.cited
    ]
    # The footer is added by the API layer, not here: this text is what goes
    # into conversation history, and a "Source:" line in history would teach
    # the model to imitate it.
    final = result.text
    report = dict(out.get("validation") or {})
    report["citations"] = {"cited": len(citations), "invalid": result.invalid,
                           "retrieved": len(chunks), "had_tags": result.had_tags}
    if report_extra:
        report["grounding_verifier"] = report_extra
    if result.invalid:
        log.warning("removed citations that pointed at no retrieved passage",
                    extra={"invalid": result.invalid, "retrieved": len(chunks)})
    update: dict[str, Any] = {**out, "citations": citations, "validation": report}
    if final != (state.get("response") or "") or "response" in out:
        update["response"] = final
    return update


NO_CHART_MESSAGE = {
    "english": ("I couldn't load your birth chart just now, so I can't read your placements or Dasha "
                "periods. Please try again in a moment."),
    "hinglish": ("Abhi aapki Kundli load nahi ho paayi, isliye main aapke placements ya Dasha nahi padh sakta. "
                 "Kripya thodi der baad dobara try kijiye."),
    "hindi": ("अभी आपकी कुंडली लोड नहीं हो पाई, इसलिए मैं आपके ग्रहों की स्थिति या दशा नहीं पढ़ सकता। "
              "कृपया थोड़ी देर बाद फिर से कोशिश करें।"),
}


def _guard_without_chart(draft: str, style_id: str | None) -> dict[str, Any] | None:
    """No chart was loaded, so ANY statement about this user's chart is invented.

    Found in a live run with the calculator unreachable: the model wrote a full
    fictional chart ("lagna Vrishabha … Shani Mahadasha 2023 to 2048") despite
    the prompt saying the chart is unavailable, and the claim checker had no
    facts to check against. Chart-specific sentences are removed; if little is
    left, an honest "couldn't load your chart" reply is sent instead.
    """
    from app.evaluation.chart_claims import (GENERIC, SECOND_PERSON, _DASHA_WORD, _YEAR, _extract,
                                             split_sentences, strip_sentences)
    from app.prompts.language_policy import LANGUAGE_OF_STYLE

    sentences = split_sentences(draft)
    bad: set[int] = set()
    last_house = None
    for i, sentence in enumerate(sentences):
        sentence = normalize_numerals(sentence)          # "२०२७", "७वें" must count as numbers
        claims, last_house = _extract(sentence, i, last_house)
        generic = bool(GENERIC.search(sentence)) and not SECOND_PERSON.search(sentence)
        dated = bool(_DASHA_WORD.search(sentence) and _YEAR.search(sentence))
        if (claims and not generic) or dated:
            bad.add(i)
    if not bad:
        return None
    kept = strip_sentences(draft, bad)
    language = LANGUAGE_OF_STYLE.get((style_id or "").lower(), "english")
    if len(kept) < 60 or len(bad) * 2 >= len(sentences):
        kept = NO_CHART_MESSAGE[language]
        action = "no_chart_fallback"
    else:
        action = "no_chart_claims_stripped"
    log.warning("chart unavailable; chart claims removed from reply", extra={"removed": len(bad)})
    return {"response": kept, "validation": {"mode": settings.flags.chart_validation_mode, "action": action,
                                             "removed_sentences": len(bad), "language_ok": True}}


def _quality_fixes(state: ChatbotState, out: dict[str, Any]) -> dict[str, Any]:
    """Deterministic quality pass (app.evaluation.response_quality): unneeded
    closing questions, repeated sentences, terminology. Always measured;
    fixed unless REPLY_QUALITY_FIXES=false."""
    from app.evaluation.response_quality import apply_safe_fixes, assess

    text = out.get("response", state.get("response") or "")
    if not text:
        return out
    language, intent = _language(state), state.get("intent")
    if settings.flags.quality_fixes:
        previous_assistant = next((m.content for m in reversed(state.get("history") or [])
                                   if getattr(m, "role", "") == "assistant"), None)
        fixed, report = apply_safe_fixes(text, language, intent, state.get("question"),
                                         followup_allowed=bool((state.get("followup") or {}).get("allowed")),
                                         previous_assistant=previous_assistant)
    else:
        fixed, report = text, assess(text, language, intent)
    from app.evaluation.conciseness import drop_unasked_dasha, join_simple_answer, trim_to_budget

    if settings.flags.quality_fixes:
        fixed, unasked = drop_unasked_dasha(fixed, state.get("question"), state.get("answer_depth"))
        fixed, trimmed = trim_to_budget(fixed, state.get("answer_depth"))
    fixed = join_simple_answer(fixed, state.get("answer_depth"))
    validation = dict(out.get("validation") or {})
    validation["quality"] = {**report.to_dict(), "unasked_dasha_removed": unasked, "over_budget_trimmed": trimmed} \
        if settings.flags.quality_fixes else report.to_dict()
    result = {**out, "validation": validation}
    if fixed != text:
        result["response"] = fixed
    return result


def _drop_copied_examples(state: ChatbotState, out: dict[str, Any]) -> dict[str, Any]:
    """Remove sentences copied verbatim from the prompt's style examples.

    Found in the human-eval batch (2026-11-13): replies opened with the
    example's "Good news — the career side of your chart looks steady." and
    asked a working professional the example's "What are you studying right
    now?". Removed only when a real answer remains."""
    text = out.get("response", state.get("response") or "")
    copied = copied_example_indices(text)
    if not copied:
        return out
    sentences = split_sentences(text)
    kept = strip_sentences(text, copied)
    if len(sentences) - len(copied) < 2 or len(kept) < MIN_KEEP_CHARS:
        return out
    report = dict(out.get("validation") or {})
    report["copied_example_sentences_removed"] = len(copied)
    log.info("removed sentences copied from style examples", extra={"count": len(copied)})
    return {**out, "response": kept, "validation": report}


def _verify_grounding(state: ChatbotState, text: str, chunks: list[Any]) -> tuple[str, dict[str, Any]]:
    """GROUNDING_VERIFIER: one extra pass of the main model that checks the
    answer against the retrieved passages and removes what they do not state.
    Its output is validated against the chart again before use, so it can
    never re-introduce a placement error."""
    mode = settings.flags.grounding_verifier
    if mode == "off" or not chunks or not text or state.get("generation_failed"):
        return text, {}
    if mode == "cited" and not re.search(r"\[\s*K\s*\d", text):
        return text, {}
    sources = [(f"K{i}", f"{source_title(c.document, c.topic)}\n{c.text}") for i, c in enumerate(chunks, 1)]
    try:
        with span("validate.grounding_verifier"), llm_purpose("grounding_verify"):
            checked = get_llm(role=ROLE_MAIN).complete(
                verification_messages(state["question"], sources, text, tag="K"),
                session_id=state["session_id"], user_id=state["user_id"],
                request_id=state["request_id"], temperature=0.0).text
    except Exception as exc:  # noqa: BLE001 — keep the (already chart-validated) answer
        log.warning("grounding verifier unavailable", extra={"error_type": type(exc).__name__})
        return text, {"ran": False, "error": type(exc).__name__}
    if not checked.strip() or leaks_prompt(checked):
        return text, {"ran": True, "used": False, "reason": "empty or leaked"}
    kundli = state.get("kundli")
    facts = build_chart_facts(kundli) if kundli is not None else None
    if kundli is not None and validate_reply(checked, facts).contradictions:
        return text, {"ran": True, "used": False, "reason": "verifier output contradicted the chart"}
    if not check_repair(text, checked, facts).ok:
        return text, {"ran": True, "used": False, "reason": "verifier output added a risky claim"}
    return checked, {"ran": True, "used": True, "changed": checked.strip() != text.strip()}


def _validate(state: ChatbotState) -> dict[str, Any]:
    if state.get("generation_failed"):
        return {}
    draft = state.get("response") or ""
    style_id = state.get("response_style")

    # Prompt/instruction leakage is never repaired by asking the same model
    # again — the reply is replaced outright with a polite deflection.
    if leaks_prompt(draft):
        log.warning("reply recited internal instructions; replaced")
        return {"response": deflection(style_id),
                "validation": {"mode": settings.flags.chart_validation_mode, "action": "prompt_leak_blocked"}}
    mode = settings.flags.chart_validation_mode
    kundli = state.get("kundli")
    if kundli is None or not getattr(kundli, "available", False):
        guarded = _guard_without_chart(draft, style_id)
        if guarded is not None:
            return guarded
    facts = build_chart_facts(kundli) if kundli is not None else ChartFacts(available=False)

    with span("validate.chart_claims"):
        result = validate_reply(draft, facts)
    # Dosha verdicts the engine cannot back are removed by the dosha policy at
    # the end (factual_gate), so they neither trigger a repair nor get stripped here.
    flow = without_dosha(result)
    language_ok = reply_matches(style_id, draft)
    report: dict[str, Any] = {"mode": mode, "draft": result.to_dict(), "language_ok": language_ok,
                              "action": "none"}

    fix_language = settings.flags.language_guard and not language_ok
    # Unsupported chart-specific claims (transits, aspects, Manglik, …) are not
    # known to be false but are not facts either: removed or rewritten.
    fix_unsupported = mode in ("repair", "auto") and bool(flow.unsupported)
    fix_placeholder = has_placeholder(draft)
    report["placeholder"] = fix_placeholder
    from app.evaluation.response_quality import mentions_machinery

    fix_machinery = mentions_machinery(draft)
    report["machinery"] = fix_machinery
    # Hinglish quality (app.prompts.hinglish): known misspellings were already
    # corrected deterministically; what is left and worth one repair is a run
    # of Hindi-style spellings of English words, or mixed tum/aap.
    hinglish: dict = {}
    if (style_id or "") == "whatsapp_hinglish":
        from app.prompts.hinglish import hinglish_issues

        issues = hinglish_issues(draft)
        if len(issues.get("suspect_transliteration", [])) >= 2 or issues.get("address_mixed"):
            hinglish = issues
        report["hinglish_issues"] = issues
    fix_machinery = fix_machinery or bool(hinglish)
    needs_fix = (bool(result.contradictions) or fix_language or fix_unsupported or fix_placeholder
                 or fix_machinery)
    if not needs_fix or mode == "log" and not (fix_language or fix_placeholder or fix_machinery):
        return {"validation": report}

    final = draft
    if mode == "auto" and (flow.contradictions or flow.unsupported) \
            and not (fix_language or fix_placeholder or fix_machinery):
        # Strip instead of regenerating when that still leaves a real answer:
        # same chart accuracy, ~2.5 s less latency, and live data showed the
        # regeneration often had to be stripped anyway.
        bad = {c.sentence_index for c in [*flow.contradictions, *flow.unsupported]}
        kept = strip_sentences(draft, bad)
        total = len(split_sentences(draft))
        if total - len(bad) >= 2 and len(kept) >= 0.6 * len(draft):
            final = kept
            report["action"] = "stripped"
            report["after"] = validate_reply(final, facts).to_dict()
            report["language_ok_after"] = reply_matches(style_id, final)
            log.warning("reply validation intervened", extra={
                "mode": mode, "action": "stripped", "contradictions": len(result.contradictions)})
            return {"response": final, "validation": report}
    if mode in ("repair", "auto") or fix_language or fix_placeholder or fix_machinery:
        note = _correction_note(flow, fix_language, style_id,
                                fix_unsupported, fix_placeholder, fix_machinery and mentions_machinery(draft),
                                hinglish=hinglish)
        messages = _repair_messages(state, draft, note, fix_language)
        try:
            with span("validate.repair_llm"), llm_purpose("repair", f"{PROMPT_VERSION}+repair"):
                # The main model (not the light route) does repairs: the light
                # model is the one most likely to have produced the problem.
                repaired = get_llm(role=ROLE_MAIN).complete(
                    messages, session_id=state["session_id"], user_id=state["user_id"],
                    request_id=state["request_id"]).text
            final = repaired
            report["action"] = "regenerated"
        except Exception as exc:  # noqa: BLE001 — fall through to stripping
            log.warning("repair regeneration failed", extra={"error_type": type(exc).__name__})
            report["repair_error"] = type(exc).__name__

    # Still wrong on the facts after the one primary repair -> one fresh answer
    # from Bolt -> the localized safe reply (validate_chain.py). Never retried.
    final, report["action"], notes = resolve_after_repair(state, draft, final, facts, mode, report["action"],
                                                          safe_fallback)
    report.update(notes)
    report["after"] = validate_reply(final, facts).to_dict()
    report["language_ok_after"] = reply_matches(style_id, final)

    log.warning("reply validation intervened", extra={
        "mode": mode, "action": report["action"],
        "contradictions": len(result.contradictions), "unsupported": len(result.unsupported),
        "language_ok": language_ok,
        "remaining_contradictions": len((report.get("after") or {}).get("contradictions") or []),
    })
    return {"response": final, "validation": report}


_SAFE_FALLBACK_BY_STYLE = {
    "whatsapp_hinglish": ("Main aapke chart ke baare mein bilkul sahi jawab dena chahta hoon, lekin abhi yeh jawab "
                          "theek se nahi ban paaya. Kripya ek baar phir poochiye."),
    "hindi": "मैं आपकी कुंडली के बारे में सटीक जवाब देना चाहता हूँ, लेकिन अभी यह जवाब ठीक से नहीं बन पाया। "
             "कृपया एक बार फिर पूछिए।",
}


def safe_fallback(style_id: str | None) -> str:
    """The last-resort reply, in the user's language (the benchmark found the
    English one sent to a Hinglish user)."""
    return _SAFE_FALLBACK_BY_STYLE.get((style_id or "").lower(), SAFE_FALLBACK)


__all__ = ["validate_reply_node", "SAFE_FALLBACK", "safe_fallback"]
