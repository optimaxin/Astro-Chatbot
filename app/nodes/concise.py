"""Guarded concise rewrite: shorten only the over-long bubbles of a validated reply.

Why (2026-11-19 audit): the primary model follows the bubble ORDER it is asked
for but not its size — advice answers stayed at ~150 words with a third of
bubbles over 35 words, whatever the prompt said. Compression has to happen
after generation, without being allowed to change what the answer says.

    reply (already chart/language validated)
      -> plan: content messages over LONG_WORDS words (not a simple answer,
         not a fallback / deterministic reply). None -> nothing happens.
      -> ONE model call for the flagged messages only (JSON in, JSON out),
         bounded by CONCISE_REWRITE_TIMEOUT_SECONDS
      -> each rewritten message is accepted on its own only if it is shorter,
         complete, in the same language, keeps every verified chart fact, date,
         Dasha status, hedge and certainty (rewrite_guard.compare_rewrite), keeps
         its advice and any dosha mention, and adds no name, question or claim
      -> the whole reply is checked again; on any failure the original stays.

Never retried. A failure, timeout or invalid output keeps the original reply.

CONCISE_REWRITE_MODE: off (default) | shadow (runs after delivery, records the
outcome without text, never changes the reply) | on (ships accepted rewrites).
"""

from __future__ import annotations

import contextvars
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from app.config.settings import BASE_DIR, settings
from app.evaluation.chart_claims import validate_reply
from app.evaluation.names import allowed_names, invented_names
from app.evaluation.rewrite_guard import compare_rewrite
from app.kundli.chart_facts import ChartFacts, build_chart_facts
from app.llm.llm_factory import ROLE_HUMANIZE, get_llm
from app.llm.usage_tracker import llm_purpose, usage_tracker
from app.logging_setup import get_logger
from app.prompts.language import has_placeholder, leaks_prompt
from app.prompts.segments import is_complete
from app.safety.notices import is_system_sentence

log = get_logger("nodes.concise")

PROMPT_VERSION = "concise-1.1"
LONG_WORDS = 35          # a content message above this is rewritten
TARGET_WORDS = 25        # what each rewritten message aims for
MAX_WORDS_AFTER = 38     # a rewritten message longer than this is not a compression
MIN_SAVING = 0.15        # a rewrite into ONE message must remove at least 15% of its words
SPLIT_MAX_WORDS = 32     # each part of a message split in two
SHADOW_LOG = Path("logs") / "concise_shadow.jsonl"

_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="aicore-concise")

SYSTEM_PROMPT = """You tighten chat messages written by a Vedic astrology assistant: same points, fewer words.

You get the whole reply for context, then numbered messages to tighten. For each numbered message:
- COMPRESS, DO NOT DELETE. Keep the point of EVERY sentence — shorten its wording instead: cut filler words, merge two sentences that share a subject, use a shorter phrase for a long one. Drop a sentence ONLY if another message of the reply already says the same thing.
- Always keep: practical advice and recommendations; what kind of partner, work, field or approach suits the user; relationship guidance; every prediction WITH its exact certainty word ("can", "may", "traditionally", "sakta hai", "ho sakta hai", "संभावना", "सकता").
- Keep every chart fact not already stated in another message — planet, house number, sign, lord, Mahadasha/Antardasha, date, month and year — written the same way. Keep any dosha or Sade Sati statement.
- Output the number of short chat messages stated next to it, each ONE or TWO complete sentences of at most 25 words, with a blank line between messages. Use the second message when one cannot hold every point.
- Same language and script as the message. Address the user as "aap" / "आप" in Hinglish and Hindi.
- Never add a fact, a prediction, a name, a question or certainty ("definitely", "pakka", "ज़रूर").

Example (English):
In:  "In everyday terms, you may enjoy roles where you can focus deeply, and you should try to build a quiet routine that lets you research and write regularly, because steady practice can bring recognition over time."
Out: "You may enjoy roles that need deep focus. Try a quiet routine for research and writing; steady practice can bring recognition."

Return strict JSON only: {"1": "shortened message", "2": "..."} with the same numbers."""

# A recommendation, not a description: an imperative opening a sentence or
# clause ("Try…", "so focus on…"), a "you should / make sure / be careful", or
# the polite Hindi/Hinglish request forms. "where you can plan carefully" is not advice.
_IMPERATIVE = (r"try|focus|consider|avoid|keep|make sure|plan|build|take|start|stay|talk|save|invest|practi[cs]e|"
               r"be careful|watch|prioriti[sz]e|choose|explore|seek|limit|nurture|update|give|look for|spend|set")
_ADVICE = re.compile(
    rf"(?:^|[.!?;:—–-]\s*|\bso,?\s+)(?:{_IMPERATIVE})\b|"
    r"\b(?:you\s+(?:should|could\s+try|might\s+want|need\s+to)|it\s+(?:helps|would\s+help)\s+to|"
    r"make\s+sure|be\s+careful|it'?s\s+(?:wise|worth|best))\b|"
    r"\b(?:kijiye|kijie|karein|karen|rakhein|rakhiye|rakhen|dijiye|bachiye|bachein|chahiye|sochiye|apnaiye|"
    r"badhaiye|dhyan\s+(?:dein|rakhein|rakhiye|dijiye))\b|कीजिए|करें|रखें|बचें|चाहिए|सोचें|अपनाएं|बढ़ाएं|ध्यान दें",
    re.I | re.M)
_DOSHA = re.compile(r"manglik|mangal\s*dosh|kaal\s*sarp|kal\s*sarp|sade\s*sati|sadhe\s*sati|\bdosh[a]?\b|"
                    r"मांगलिक|मंगल\s*दोष|काल\s*सर्प|साढ़े\s*साती|साढ़ेसाती|दोष", re.I)


# "Traditionally, this points to…" is how a reading stays a reading; the global
# hedge list (rewrite_guard._HEDGE) does not count it, the rewrite must keep it.
_SOFTENER = re.compile(r"\btraditionally\b|\bparamparik\w*|\bparampara\w*|पारंपरिक|परंपरा|माना\s+जाता", re.I)


def _hedged(text: str) -> bool:
    """A real hedge. The bare noun "chance(s)" is not one here: "…ka chance
    milega" ("you WILL get the chance") passed as hedged after the rewrite
    turned "chance de sakta hai" into it (replay, 2026-11-19)."""
    from app.evaluation.rewrite_guard import _HEDGE

    if _SOFTENER.search(text):
        return True
    return any(m.group(0).lower() not in ("chance", "chances") for m in _HEDGE.finditer(text))


POINT_KEPT = 0.4         # share of a sentence's content words a rewrite must keep for its point to survive
POINT_ELSEWHERE = 0.6    # ...unless this share of them is already said in another message


def lost_points(original: str, rewritten: str, elsewhere: str = "") -> tuple[list[str], list[str]]:
    """(sentences whose point the rewrite deleted, hedged sentences it made certain).

    Compression keeps each sentence's point in fewer words; deletion drops it.
    A sentence is deleted when the rewrite keeps under POINT_KEPT of its content
    words, unless another message already says it (a restatement may go). The
    shadow replay of 2026-11-19 shipped a Hinglish rewrite that deleted "Aapko
    aise partner pasand aa sakta hai jo samjhautar ho…" — the answer itself."""
    from app.evaluation.chart_claims import split_sentences
    from app.evaluation.conciseness import _content

    new_sentences = [(s, _content(s)) for s in split_sentences(rewritten)]
    kept_words = set().union(*(w for _, w in new_sentences)) if new_sentences else set()
    other = _content(elsewhere)
    deleted, certain = [], []
    for sentence in split_sentences(original):
        words = _content(sentence)
        if len(words) < 3:
            continue
        if len(words & kept_words) / len(words) < POINT_KEPT:
            if len(words & other) / len(words) < POINT_ELSEWHERE:
                deleted.append(sentence)
            continue
        if _hedged(sentence) and new_sentences:
            # The sentences that now carry this point (a split may spread it over
            # two) must still hedge it somewhere.
            carriers = [s for s, w in new_sentences if len(words & w) >= 2]
            if carriers and not any(_hedged(s) for s in carriers):
                certain.append(sentence)
    return deleted, certain


# Recommendation verbs. Every distinct one in a bubble must survive its rewrite:
# "if you stay organised and avoid unnecessary spending" lost "avoid unnecessary
# spending" while the sentence itself survived (replay, 2026-11-19).
_ADVICE_MARKER = re.compile(
    r"\b(?:avoid|try|focus|consider|make\s+sure|be\s+careful|watch\s+out|prioriti[sz]e|save|invest|limit|nurture|"
    r"seek|choose|practi[cs]e|keep|stay)\b|"
    r"\b(?:kijiye|karein|rakhein|rakhiye|bachiye|bachein|chahiye|sochiye|apnaiye|badhaiye|dijiye)\b|"
    r"कीजिए|करें|रखें|बचें|चाहिए|सोचें|अपनाएं|बढ़ाएं", re.I)


def advice_markers(text: str) -> set[str]:
    return {re.sub(r"\s+", " ", m.group(0).lower()) for m in _ADVICE_MARKER.finditer(text or "")}


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]


def _bubbles(text: str) -> list[str]:
    """The reply as the user SEES it (app.prompts.segments). Planning on the
    model's raw paragraphs flagged a 113-word paragraph that the user is shown
    as four 23–34-word bubbles (live shadow, 2026-11-19): a rewrite nobody needed."""
    from app.prompts.segments import segment_reply

    return segment_reply(text or "")


def _words(text: str) -> int:
    return len(text.split())


def _is_question(text: str) -> bool:
    return text.rstrip().endswith(("?", "？"))


def plan(text: str, depth: str | None) -> list[int]:
    """Indices of the displayed bubbles worth rewriting: content bubbles over
    LONG_WORDS. Questions, system notices and simple answers are never touched."""
    if depth == "simple":
        return []
    return [i for i, p in enumerate(_bubbles(text))
            if _words(p) > LONG_WORDS and not _is_question(p) and not is_system_sentence(p)]


MAX_CONTENT_MESSAGES = 5


def message_budget(paragraphs: list[str], flagged: list[int]) -> dict[int, str]:
    """How many messages each flagged message may become. Squeezing a 45-word,
    three-fact message into ONE short message made the model drop a fact
    (replay 2026-11-19), so a message may become two while the reply stays
    within MAX_CONTENT_MESSAGES content messages."""
    content = sum(1 for p in paragraphs if not _is_question(p))
    room = max(0, MAX_CONTENT_MESSAGES - content)
    budget: dict[int, str] = {}
    for i in flagged:
        words = _words(paragraphs[i])
        if words > 110:
            budget[i] = "2 or 3" if room >= 2 else "2" if room >= 1 else "1"
            room -= min(room, 2)
        elif words > LONG_WORDS and room >= 1:
            budget[i] = "1 or 2"
            room -= 1
        else:
            budget[i] = "1"
    return budget


def build_messages(text: str, flagged: list[int], language: str) -> list[dict[str, str]]:
    paragraphs = _bubbles(text)
    budget = message_budget(paragraphs, flagged)
    numbered = "\n\n".join(
        f"{n}. [shorten to {budget[i]} message(s)]\n{paragraphs[i]}" for n, i in enumerate(flagged, 1))
    whole = "\n\n".join(paragraphs)
    user = (f"Language: {language}.\n\nWHOLE REPLY (context only):\n{whole}\n\n"
            f"MESSAGES TO SHORTEN:\n{numbered}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _others(reply: str | None, message: str) -> str:
    """The rest of the reply: what every displayed bubble except `message` says."""
    # "not in", not "!=": the bubble may be the message with a short neighbour merged in.
    return "\n\n".join(p for p in _bubbles(reply or "") if message.strip() not in p)


def _parse(raw: str) -> dict[str, str] | None:
    start, end = (raw or "").find("{"), (raw or "").rfind("}")
    try:
        data = json.loads(raw[start:end + 1])
    except (ValueError, TypeError):
        return None
    return {str(k): str(v).strip() for k, v in data.items()} if isinstance(data, dict) else None


def check_message(original: str, rewritten: str, *, facts: ChartFacts, style_id: str | None,
                  allowed: set[str], reply_before: str | None = None, reply_after: str | None = None,
                  max_parts: int = 3) -> str:
    """'ok' or the reason this one rewritten message is rejected. Pure.

    Chart facts, dates and new claims are judged on the WHOLE reply with only
    this message replaced (`reply_before` -> `reply_after`): dropping a
    placement another message already states is the repetition being removed,
    not a lost fact. Hedging and certainty are judged on the message itself."""
    from app.evaluation.rewrite_guard import _absolute
    from app.nodes.validate import _segment_language_ok

    if not rewritten:
        return "empty"
    if leaks_prompt(rewritten) or has_placeholder(rewritten) or rewritten.lstrip().startswith(("{", "[")):
        return "malformed_or_leaked"
    parts = _paragraphs(rewritten)
    if len(parts) > max_parts:
        return "too_many_messages"
    if not all(is_complete(p) for p in parts):
        return "incomplete_sentence"
    if any(_words(p) > (MAX_WORDS_AFTER if len(parts) == 1 else SPLIT_MAX_WORDS) for p in parts):
        return "still_long"
    # One message must really be shorter; a split into short messages meets the
    # bubble-size target without dropping anything, so it may only not grow.
    limit = (1 - MIN_SAVING) * _words(original) if len(parts) == 1 else _words(original)
    if _words(rewritten) > limit:
        return "not_shorter"
    if any(_is_question(p) for p in parts) and not _is_question(original):
        return "added_question"
    if not all(_segment_language_ok(style_id, p) for p in parts):
        return "language_changed"
    if invented_names(rewritten, allowed) and not invented_names(original, allowed):
        return "added_name"
    if (_ADVICE.search(original) and not _ADVICE.search(rewritten)) or \
            advice_markers(original) - advice_markers(rewritten):
        return "dropped_advice"
    if _DOSHA.search(original) and not _DOSHA.search(rewritten):
        return "dropped_dosha"
    if _hedged(original) and not _hedged(rewritten):
        return "facts:removed hedging"
    if _absolute(rewritten) and not _absolute(original):
        return "facts:added certainty"
    guard = compare_rewrite(reply_before or original, reply_after or rewritten, facts)
    if not guard.ok:
        return "facts:" + guard.reasons[0][:80]
    others = _others(reply_before, original)
    deleted, certain = lost_points(original, rewritten, others)
    if deleted:
        return "dropped_point"
    if certain:
        return "certainty_raised"
    if repeats_other_bubble(rewritten, others):
        return "adds_repetition"
    return "ok"


def repeats_other_bubble(rewritten: str, others: str, share: float = 0.8) -> bool:
    """A rewritten sentence that restates another bubble: the model pulled a
    sentence from the context into its rewrite, and it would ship twice (Hindi
    shadow case, 2026-11-19: the Venus-in-the-7th sentence)."""
    from app.evaluation.chart_claims import split_sentences
    from app.evaluation.conciseness import _content

    said = [_content(s) for s in split_sentences(others)]
    for sentence in split_sentences(rewritten):
        words = _content(sentence)
        if len(words) >= 4 and any(len(words & s) / len(words) >= share for s in said if s):
            return True
    return False


def display_gain(before: str, after: str) -> bool:
    """The rewrite must improve what the user SEES: fewer long bubbles and no
    longer bubble than before. Shortening one bubble can let the segmenter merge
    it with a neighbour into a longer one (Hinglish shadow case: 48 -> 54 words)."""
    old = [_words(b) for b in _bubbles(before)]
    new = [_words(b) for b in _bubbles(after)]
    if not old or not new:
        return False
    return max(new) <= max(old) and sum(w > LONG_WORDS for w in new) < sum(w > LONG_WORDS for w in old)


def skip_reason(state: dict[str, Any]) -> str | None:
    if state.get("generation_failed") or not (state.get("response") or "").strip():
        return "no_answer"
    if state.get("answer_depth") == "simple" or state.get("intent") in ("smalltalk", "clarification", "off_topic"):
        return "simple_or_nonastro"
    v = state.get("validation") or {}
    if "safe_fallback" in str(v.get("action", "")) or v.get("language_gate") == "fallback":
        return "safe_reply"
    if str(state.get("model_used") or "").startswith(("rule:", "safety-gate", "none")):
        return "deterministic_reply"
    return None


def _optional_llm() -> Any:
    """The rewrite model as an OPTIONAL call: one attempt, and its 429s never
    open the circuit that the users' main answers depend on."""
    import dataclasses

    from app.llm.llm_factory import LLM

    return LLM(dataclasses.replace(get_llm(ROLE_HUMANIZE).config, optional=True))


def _call(messages: list[dict[str, str]], state: dict[str, Any], timeout: float) -> str:
    """The rewrite call, bounded by `timeout`. On timeout the call finishes in the
    background (it is still metered) and TimeoutError is raised here."""
    def run() -> str:
        llm = _optional_llm()
        with llm_purpose("concise_rewrite", PROMPT_VERSION):
            return llm.complete(
                messages, session_id=state["session_id"], user_id=state["user_id"],
                request_id=state["request_id"], temperature=0.0, max_tokens=600).text

    future = _pool.submit(contextvars.copy_context().run, run)
    try:
        return future.result(timeout=timeout)
    except FutureTimeout as exc:
        raise TimeoutError(f"concise rewrite exceeded {timeout:.0f}s") from exc


def attempt(state: dict[str, Any], mode: str) -> tuple[dict[str, Any], str | None]:
    """One rewrite and its verdict. Never raises. Returns (report, new_text or None)."""
    started = time.perf_counter()
    text = state.get("response") or ""
    paragraphs = _bubbles(text)
    flagged = plan(text, state.get("answer_depth"))
    report: dict[str, Any] = {"mode": mode, "prompt_version": PROMPT_VERSION, "flagged": len(flagged),
                              "messages": len(paragraphs), "words_before": _words(text)}
    if not flagged:
        return {**report, "status": "skipped", "reason": "already_concise"}, None

    from app.prompts.language_policy import LANGUAGE_OF_STYLE

    style_id = state.get("response_style")
    language = LANGUAGE_OF_STYLE.get((style_id or "").lower(), "english")
    kundli = state.get("kundli")
    facts = build_chart_facts(kundli) if kundli is not None else ChartFacts(available=False)
    users = [getattr(m, "content", "") for m in state.get("history") or [] if getattr(m, "role", "") == "user"]
    allowed = allowed_names(kundli, [*users, state.get("question") or ""])
    try:
        answers = _parse(_call(build_messages(text, flagged, language), state,
                               settings.flags.concise_rewrite_timeout_seconds))
    except TimeoutError:
        return {**report, "status": "timeout", "ms": int((time.perf_counter() - started) * 1000)}, None
    except Exception as exc:  # noqa: BLE001 — a failed rewrite never costs the user their answer
        info = getattr(exc, "info", None)
        return {**report, "status": "failed", "reason": info.label() if info else type(exc).__name__,
                "ms": int((time.perf_counter() - started) * 1000)}, None
    if answers is None:
        return {**report, "status": "rejected", "reason": "unparseable_output",
                "ms": int((time.perf_counter() - started) * 1000)}, None

    budget = message_budget(paragraphs, flagged)
    new_paragraphs, verdicts, proposed, detail = list(paragraphs), [], list(paragraphs), []
    for n, i in enumerate(flagged, 1):
        candidate = answers.get(str(n), "").strip()
        trial = [*new_paragraphs[:i], candidate, *new_paragraphs[i + 1:]]
        verdict = check_message(paragraphs[i], candidate, facts=facts, style_id=style_id, allowed=allowed,
                                reply_before=text, reply_after="\n\n".join(trial),
                                max_parts=int(budget[i][-1]))
        verdicts.append(verdict)
        proposed[i] = candidate or paragraphs[i]
        detail.append({"words_before": _words(paragraphs[i]),
                       "words_proposed": [_words(p) for p in _paragraphs(candidate)], "verdict": verdict})
        if verdict == "ok":
            new_paragraphs[i] = candidate
    report["verdicts"], report["messages_detail"] = verdicts, detail
    if settings.flags.concise_shadow_log_text and mode == "shadow":
        # Evaluation only (CONCISE_SHADOW_LOG_TEXT): the texts a human reviewer needs.
        report["text"] = {"original": text, "proposed": "\n\n".join(proposed),
                          "would_ship": "\n\n".join(new_paragraphs)}
    accepted = verdicts.count("ok")
    report["accepted"], report["rejected"] = accepted, len(verdicts) - accepted
    new_text = "\n\n".join(new_paragraphs)
    if not accepted:
        return {**report, "status": "rejected", "reason": verdicts[0],
                "ms": int((time.perf_counter() - started) * 1000)}, None

    # Whole-reply check: the combination must keep every fact and add no contradiction.
    whole = compare_rewrite(text, new_text, facts)
    contradictions = {c.describe() for c in validate_reply(new_text, facts).contradictions} \
        - {c.describe() for c in validate_reply(text, facts).contradictions}
    if not whole.ok or contradictions:
        reason = (whole.reasons or sorted(contradictions) or ["whole_reply_check"])[0][:90]
        return {**report, "status": "rejected", "reason": "whole:" + reason,
                "ms": int((time.perf_counter() - started) * 1000)}, None
    if not display_gain(text, new_text):
        return {**report, "status": "rejected", "reason": "whole:no_display_gain",
                "ms": int((time.perf_counter() - started) * 1000)}, None
    from app.evaluation.conciseness import restated_sentences

    if len(restated_sentences(new_text)) > len(restated_sentences(text)):
        # A restatement that ended up inside one merged bubble, beyond the per-bubble check.
        return {**report, "status": "rejected", "reason": "whole:adds_repetition",
                "ms": int((time.perf_counter() - started) * 1000)}, None
    report.update(status="applied" if mode == "on" else "shadow_would_apply",
                  reason="ok" if accepted == len(flagged) else "partial", words_after=_words(new_text),
                  ms=int((time.perf_counter() - started) * 1000))
    return report, new_text


def _record_shadow(request_id: str, report: dict[str, Any], language: str | None) -> None:
    mine = [c for c in usage_tracker.for_request(request_id) if c.purpose == "concise_rewrite"]
    line = {"ts": time.time(), "request_id": request_id, "language": language, **report,
            "input_tokens": sum(c.input_tokens or 0 for c in mine),
            "output_tokens": sum(c.output_tokens or 0 for c in mine)}
    try:
        path = SHADOW_LOG if SHADOW_LOG.is_absolute() else BASE_DIR / SHADOW_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError:
        pass


def concise_rewrite_node(state: dict[str, Any]) -> dict[str, Any]:
    mode = settings.flags.concise_rewrite_mode
    if mode not in ("shadow", "on"):
        return {}
    validation = dict(state.get("validation") or {})
    why = skip_reason(state)
    if why is None and not plan(state.get("response") or "", state.get("answer_depth")):
        why = "already_concise"
    if why is not None:
        validation["concise_rewrite"] = {"mode": mode, "status": "skipped", "reason": why}
        return {"validation": validation}

    if mode == "shadow":
        snapshot = dict(state)
        language = (state.get("language") or {}).get("language")

        def work() -> None:
            report, _ = attempt(snapshot, "shadow")
            _record_shadow(snapshot["request_id"], report, language)

        if settings.flags.evaluation_async:
            from app.observability import background

            ok = background.try_submit(work)
            validation["concise_rewrite"] = ({"mode": mode, "status": "shadow_pending"} if ok
                                             else {"mode": mode, "status": "skipped", "reason": "queue_full"})
        else:
            work()
            validation["concise_rewrite"] = {"mode": mode, "status": "shadow_recorded"}
        return {"validation": validation}

    report, new_text = attempt(state, "on")
    validation["concise_rewrite"] = report
    log.info("concise rewrite", extra={k: report.get(k) for k in ("status", "reason", "flagged", "accepted",
                                                                   "words_before", "words_after", "ms")})
    if new_text is not None:
        return {"response": new_text, "validation": validation}
    return {"validation": validation}


__all__ = ["concise_rewrite_node", "plan", "check_message", "attempt", "LONG_WORDS", "PROMPT_VERSION"]
