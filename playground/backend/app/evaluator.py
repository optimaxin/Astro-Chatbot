"""Automated (LLM-as-judge) evaluation of one experiment's response.

Scores are an AI estimate, labelled as such in the UI, never ground truth.
The judge's own tokens and cost are reported separately so they are never
mixed into the tested model's numbers.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.catalog import Catalog, calculate_cost
from app.providers import ProviderError, get_provider

METRICS = ("relevance", "accuracy", "human_likeness", "tone", "conciseness",
           "context_awareness", "emoji_usage")

JUDGE_SYSTEM = """You are a strict reviewer of chatbot replies. You score one reply to one user message.

Score each metric from 0 to 10 (decimals allowed):
- relevance: does it answer what was actually asked?
- accuracy: is it consistent with the provided context and free of invented facts? If no context was given, judge plausibility and hedging.
- human_likeness: does it read like a person texting, not a report or a template?
- tone: warm, respectful, appropriate for the user?
- conciseness: no padding, no repetition, no over-explaining?
- context_awareness: does it use the system prompt, context and conversation history well?
- emoji_usage: are emojis used appropriately? Use null if the reply has no emojis and none were expected.

Return ONLY a JSON object, no markdown:
{"scores": {"relevance": n, "accuracy": n, "human_likeness": n, "tone": n, "conciseness": n, "context_awareness": n, "emoji_usage": n or null},
 "feedback": "2-3 sentences of concrete feedback",
 "strengths": ["short item", "..."],
 "improvements": ["short item", "..."]}"""


def _judge_prompt(experiment: dict[str, Any]) -> str:
    parts = []
    if experiment.get("system_prompt"):
        parts.append(f"SYSTEM PROMPT GIVEN TO THE MODEL:\n{experiment['system_prompt']}")
    if experiment.get("context"):
        parts.append(f"CONTEXT GIVEN TO THE MODEL:\n{experiment['context']}")
    history = (experiment.get("effective_request") or {}).get("messages", [])[:-1]
    turns = [f"{m['role']}: {m['content']}" for m in history if m["role"] in ("user", "assistant")]
    if turns:
        parts.append("EARLIER CONVERSATION:\n" + "\n".join(turns[-10:]))
    parts.append(f"USER MESSAGE:\n{experiment['user_prompt']}")
    parts.append(f"REPLY TO EVALUATE:\n{experiment['response']}")
    return "\n\n".join(parts)


def parse_judge_output(text: str) -> dict[str, Any]:
    """Extract the judge's JSON, tolerating code fences and stray prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("judge did not return JSON")
    data = json.loads(match.group(0))
    raw_scores = data.get("scores") or {}
    scores: dict[str, float | None] = {}
    for metric in METRICS:
        value = raw_scores.get(metric)
        try:
            scores[metric] = None if value is None else round(max(0.0, min(10.0, float(value))), 1)
        except (TypeError, ValueError):
            scores[metric] = None
    present = [v for v in scores.values() if v is not None]
    return {
        "scores": scores,
        "overall": round(sum(present) / len(present), 1) if present else None,
        "feedback": str(data.get("feedback") or "").strip(),
        "strengths": [str(s) for s in (data.get("strengths") or [])][:6],
        "improvements": [str(s) for s in (data.get("improvements") or [])][:6],
    }


def evaluate(experiment: dict[str, Any], catalog: Catalog,
             provider_id: str | None = None, model_id: str | None = None) -> dict[str, Any]:
    if provider_id and model_id:
        target = (provider_id, model_id)
    elif catalog.evaluator:
        target = catalog.evaluator
    else:
        raise ProviderError("evaluator", "No evaluator model is configured.")
    model = catalog.model(*target)
    if model is None:
        raise ProviderError("evaluator", f"Evaluator model {target[1]!r} is not in the catalog.")
    provider = catalog.providers[model.provider]

    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": _judge_prompt(experiment)}]
    started = time.perf_counter()
    result = get_provider(provider).generate(model, messages, {"temperature": 0.0, "max_tokens": 700})
    try:
        verdict = parse_judge_output(result.text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ProviderError(provider.name, f"The evaluator returned an unreadable verdict ({exc}).") from exc

    return {
        **verdict,
        "label": "AI Evaluation",
        "evaluator": {"provider": provider.id, "model": model.id},
        "evaluator_usage": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            **calculate_cost(model.pricing, result.input_tokens, result.output_tokens),
        },
    }
