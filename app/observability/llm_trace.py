"""One line per model call of a turn, with what validation made of its output.

Built at the end of the turn from the usage records (role, purpose, model the
provider actually reported, fallback status, route, prompt version, latency,
tokens, outcome) and the turn's validation report. It holds no prompt, reply,
user name or birth data, so it is safe for ordinary logs.

The LLM judge (purpose "evaluate") runs after the reply is sent when
EVALUATION_ASYNC is on, so it is not in this trace; its own usage row carries
purpose="evaluate" in the ledger.
"""

from __future__ import annotations

from typing import Any


def _verdict(purpose: str | None, success: bool, state: dict[str, Any]) -> str:
    if not success:
        return "call_failed"
    v = state.get("validation") or {}
    if purpose == "answer":
        action = v.get("action")
        if action is None:
            return "not_validated" if state.get("generation_failed") else "passed"
        return "passed" if action == "none" else f"intervened:{action}"
    if purpose == "repair":
        return "rejected" if v.get("primary_rejected") else "accepted"
    if purpose == "validation_fallback":
        return "rejected" if v.get("bolt_rejected") else "accepted"
    if purpose == "language_rewrite":
        return "rejected" if v.get("language_gate_rewrite_rejected") else str(v.get("language_gate") or "accepted")
    if purpose == "grammar_correction":
        lq = v.get("language_quality") or {}
        if lq.get("llm_discarded_by_guard"):
            return "rejected:rewrite_guard"
        return f"accepted:{lq.get('llm_accepted', 0)}/{lq.get('llm_sentences', 0)}"
    if purpose == "grounding_verify":
        return "used" if (v.get("grounding_verifier") or {}).get("used") else "not_used"
    if purpose == "classify":
        return f"intent={state.get('intent')} via {state.get('intent_method')}"
    if purpose == "humanize":
        h = v.get("humanizer") or {}
        return f"{h.get('status', 'n/a')}:{h.get('reason', '')}".rstrip(":")
    if purpose == "quality_check":
        q = state.get("quality_check") or {}
        return str(q.get("status") or "PENDING")      # asynchronous: the verdict is in the shadow log
    return "n/a"


def build_trace(calls: list[Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "role": c.role, "purpose": c.purpose, "provider": c.provider, "model": c.model,
        "requested_model": c.requested_model,
        "fallback": c.fallback_used, "attempt": c.attempt, "route": c.route,
        "prompt_version": c.prompt_version, "latency_ms": c.latency_ms,
        "input_tokens": c.input_tokens, "output_tokens": c.output_tokens, "token_source": c.token_source,
        "success": c.success, "error_code": c.error_code,
        "verdict": _verdict(c.purpose, c.success, state),
    } for c in calls]


def _last_success(calls: list[Any], purpose: str, model: str | None = None) -> Any:
    hits = [c for c in calls if c.success and c.purpose == purpose and (model is None or c.model == model)]
    return hits[-1] if hits else None


def _only_system_text(text: str) -> bool:
    from app.evaluation.chart_claims import split_sentences
    from app.safety.notices import is_system_sentence

    sentences = split_sentences(text or "")
    return bool(sentences) and all(is_system_sentence(s) for s in sentences)


def shipped(state: dict[str, Any], calls: list[Any]) -> dict[str, Any]:
    """Which content reached the user, and from where.

    source: primary | fallback | deterministic | safe_response. A candidate
    that validation rejected is never named here: the call is chosen from the
    decision that shipped (the validation action), not from call order.
    """
    v = state.get("validation") or {}
    action = str(v.get("action") or "none")
    final_gate = str((v.get("final_gate") or {}).get("action") or "")
    model_used = str(state.get("model_used") or "")
    out: dict[str, Any] = {"source": None, "purpose": None, "call": None, "model": None, "provider": None,
                           "fallback": False, "edits": []}

    if model_used == "safety-gate" or model_used.startswith("rule:"):
        out.update(source="deterministic", purpose=model_used, model=model_used)
    elif state.get("generation_failed"):
        out.update(source="safe_response", purpose="outage", model="none")
    elif ("safe_fallback" in action or final_gate == "safe_fallback" or action == "prompt_leak_blocked"
          or v.get("language_gate") == "fallback" or action == "no_chart_fallback"):
        out.update(source="safe_response", purpose="safe_reply", model="none")
    elif _only_system_text(state.get("response") or ""):
        out.update(source="deterministic", purpose="system_notice", model="none")
    else:
        if action == "bolt" or "+bolt" in action:
            purpose = "validation_fallback"
        elif str(v.get("language_gate") or "").startswith("rewritten"):
            purpose = "language_rewrite"
        elif action.startswith("regenerated"):
            purpose = "repair"
        else:
            purpose = "answer"
        call = (_last_success(calls, purpose, model_used or None) if purpose == "answer" else None) \
            or _last_success(calls, purpose)
        fallback = bool(call and call.fallback_used) or purpose == "validation_fallback"
        out.update(source="fallback" if fallback else "primary", purpose=purpose, call=call, fallback=fallback,
                   model=call.model if call else (model_used or None), provider=call.provider if call else None)
    lq = v.get("language_quality") or {}
    if lq.get("llm_accepted"):
        out["edits"].append("grammar_correction")
    if (v.get("humanizer") or {}).get("status") == "applied":
        out["edits"].append("humanizer")
    if "stripped" in action or final_gate == "stripped":
        out["edits"].append("sentences_removed")
    for key, label in (("dosha_policy", "dosha_notice"), ("integrity_notice", "integrity_notice"),
                       ("followup_added", "followup_added")):
        if v.get(key):
            out["edits"].append(label)
    return out


def validation_status(state: dict[str, Any]) -> str:
    """passed | the action taken | not_validated (gated, clarified, outage)."""
    v = state.get("validation") or {}
    if not v or state.get("generation_failed"):
        return "not_validated"
    action = str(v.get("action") or "none")
    final_gate = str((v.get("final_gate") or {}).get("action") or "")
    parts = [a for a in (action if action != "none" else "", f"final_gate:{final_gate}" if final_gate else "") if a]
    return "+".join(parts) or "passed"


def turn_outcome(state: dict[str, Any]) -> dict[str, Any]:
    """The deterministic gates' decisions on the final text."""
    v = state.get("validation") or {}
    return {k: v[k] for k in ("action", "dosha_policy", "final_gate", "integrity_notice", "followup_added",
                              "language_gate") if k in v}


__all__ = ["build_trace", "turn_outcome", "shipped", "validation_status"]
