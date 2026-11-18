"""Independent quality checker, shadow mode.

A second model that did NOT write the reply (Nemotron 3 Ultra, configured with
NVIDIA_QUALITY_CHECKER_MODEL) scores a sample of delivered replies against the
Kundli facts and records PASS / REPAIR / REJECT / UNAVAILABLE. It is advisory:

* it runs AFTER the reply is on its way (EVALUATION_ASYNC), so it adds no latency;
* it never edits, blocks or delays a reply; deterministic validation against the
  chart engine is the only gate;
* it is told nothing about what the deterministic checks found, so its opinion is
  independent; the two are compared afterwards (`agreement`) to find replies the
  deterministic layer passed that the checker disliked, and the reverse;
* UNAVAILABLE (call failed, unparseable, circuit open, model not independent) is
  recorded as such and is never read as a pass;
* it has no fallback model: falling back to the generating model would let the
  author approve itself.

Promoting it to a gate is a separate, measured decision (see docs). When that
happens it must move before the humanizer; in shadow mode it checks the text the
user actually received.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from app.config.settings import BASE_DIR, settings
from app.llm.llm_factory import ROLE_QUALITY, LLMNotConfigured, active_provider, get_llm
from app.llm.usage_tracker import llm_purpose, usage_tracker
from app.logging_setup import get_logger
from app.observability.llm_trace import shipped, validation_status
from app.prompts.roles.quality_checker import QUALITY_PROMPT_VERSION, parse_verdict, quality_messages
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.quality_check")

SKIP_INTENTS = frozenset({"smalltalk", "off_topic", "clarification", "safety_gated"})
SHADOW_LOG = Path("logs") / "quality_checker.jsonl"


def _sampled() -> bool:
    rate = settings.flags.quality_checker_sample_rate
    return rate >= 1.0 or (rate > 0.0 and random.random() < rate)


def skip_reason(state: ChatbotState, delivered: dict[str, Any]) -> str | None:
    """Why this turn is not checked, or None. Cheap, deterministic, no model call."""
    if settings.flags.quality_checker_mode != "shadow":
        return "off"
    if not settings.nvidia.quality_checker or active_provider() != "nvidia":
        return "not_configured"           # no model configured, or the NVIDIA provider is not the active one
    if state.get("generation_failed") or not (state.get("response") or "").strip():
        return "no_answer"
    if delivered["source"] not in ("primary", "fallback"):
        return f"not_model_generated:{delivered['source']}"      # safe replies, notices, gates: nothing to judge
    if state.get("intent") in SKIP_INTENTS or state.get("answer_depth") == "simple":
        return "simple_or_nonastro"
    author = {delivered.get("model"), getattr(delivered.get("call"), "requested_model", None)}
    if settings.nvidia.quality_checker in author:
        return "not_independent"                                  # the checker must not grade its own work
    return None


def run_checker(*, request_id: str, session_id: str, user_id: str, facts_block: str, question: str,
                history_text: str, reply: str) -> dict[str, Any]:
    """One checker call. Never raises. Returns the recorded outcome."""
    started = time.perf_counter()
    out: dict[str, Any] = {"status": "UNAVAILABLE", "model": settings.nvidia.quality_checker,
                           "prompt_version": QUALITY_PROMPT_VERSION}
    try:
        with llm_purpose("quality_check", QUALITY_PROMPT_VERSION):
            raw = get_llm(ROLE_QUALITY).complete(
                quality_messages(facts_block, question, history_text, reply),
                session_id=session_id, user_id=user_id, request_id=request_id).text
        verdict = parse_verdict(raw)
        if verdict is None:
            out["reason"] = "unparseable_output"
        else:
            out.update(status=verdict.decision, scores=verdict.model_dump(exclude={"issues", "decision"}),
                       issues=verdict.issues)
    except LLMNotConfigured:
        out["reason"] = "not_configured"
    except Exception as exc:  # noqa: BLE001 — advisory: a failure is a recorded status, never an error
        info = getattr(exc, "info", None)
        out["reason"] = info.label() if info else type(exc).__name__
    out["latency_ms"] = int((time.perf_counter() - started) * 1000)
    return out


def agreement(status: str, det_status: str) -> str:
    """How the checker and the deterministic layer compare on this reply."""
    if status == "UNAVAILABLE":
        return "checker_unavailable"
    checker_flags = status in ("REPAIR", "REJECT")
    det_intervened = det_status not in ("passed", "not_validated")
    if checker_flags and not det_intervened:
        return "checker_only"            # a possible deterministic miss: worth a human look
    if checker_flags and det_intervened:
        return "both_flagged"
    return "deterministic_only" if det_intervened else "both_clean"


def _record(request_id: str, outcome: dict[str, Any], det_status: str, source: str) -> None:
    outcome["agreement"] = agreement(outcome["status"], det_status)
    line = {"ts": time.time(), "request_id": request_id, "response_source": source,
            "deterministic": det_status, **outcome}
    log.info("quality check", extra={"request_id": request_id, "qc_status": outcome["status"],
                                     "qc_agreement": outcome["agreement"], "qc_ms": outcome.get("latency_ms")})
    try:
        path = SHADOW_LOG if SHADOW_LOG.is_absolute() else BASE_DIR / SHADOW_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _history_text(state: ChatbotState) -> str:
    return "\n".join(f"{m.role}: {m.content[:300]}" for m in (state.get("history") or [])[-2:])


def quality_check_node(state: ChatbotState) -> dict[str, Any]:
    if settings.flags.quality_checker_mode == "off":
        return {}
    delivered = shipped(state, usage_tracker.for_request(state["request_id"]))
    why = skip_reason(state, delivered)
    if why is not None:
        return {"quality_check": {"status": "SKIPPED", "reason": why}} if why != "not_configured" else {}
    if not _sampled():
        return {"quality_check": {"status": "SKIPPED", "reason": "unsampled"}}

    kundli = state.get("kundli")
    kwargs = dict(request_id=state["request_id"], session_id=state["session_id"], user_id=state["user_id"],
                  facts_block=kundli.to_prompt_block(state.get("intent") or "general") if kundli is not None else "",
                  question=state["question"], history_text=_history_text(state), reply=state["response"])
    det_status, source = validation_status(state), delivered["source"]

    def work() -> dict[str, Any]:
        outcome = run_checker(**kwargs)
        _record(kwargs["request_id"], outcome, det_status, source)
        return outcome

    if settings.flags.evaluation_async:
        from app.observability import background

        # try_submit never runs the job on this thread: a full queue drops the sample
        # rather than delaying the reply it is checking.
        if not background.try_submit(work):
            return {"quality_check": {"status": "SKIPPED", "reason": "queue_full"}}
        return {"quality_check": {"status": "PENDING"}}
    return {"quality_check": work()}


__all__ = ["quality_check_node", "run_checker", "skip_reason", "agreement"]
