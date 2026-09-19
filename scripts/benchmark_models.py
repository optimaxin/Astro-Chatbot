"""Compare OpenRouter models on the same questions and the same kundli.

    python -m scripts.benchmark_models --models "model-a,model-b" [--limit 5]

Every model is asked the identical question set (`tests/test_questions.py`)
against the identical chart, each in its own fresh session, and is graded by
the same evaluator model. That is what makes the numbers comparable: the only
variable is the generation model.

The report shows quality, reward, latency, tokens and estimated cost side by
side, and deliberately does not declare a winner. The cheapest model is not
automatically the best one — the choice is a quality-versus-cost tradeoff, and
this script exists to make that tradeoff visible rather than to make it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings as settings_module  # noqa: E402
from app.config.settings import ModelSettings, settings  # noqa: E402
from app.console import LINE, RULE, fail_mark, safe_print  # noqa: E402
from app.graph.chatbot_graph import run_turn  # noqa: E402
from app.memory.redis_memory import memory  # noqa: E402
from tests.test_questions import QUESTIONS  # noqa: E402

DIMENSIONS = (
    "relevance", "personalization", "groundedness",
    "conversation_quality", "safety", "completeness",
)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def run_model(model: str, limit: int | None, user_id: str) -> dict[str, Any]:
    """Run the whole question set against one model."""
    original = settings_module.settings.models
    # Swap only the generation model; the classifier and evaluator stay fixed
    # so every model is judged by the same judge.
    object.__setattr__(
        settings_module.settings,
        "models",
        ModelSettings(
            main=model,
            fallback="",  # no fallback — a failure must show as a failure
            classifier=original.classifier,
            evaluator=original.evaluator,
            main_temperature=original.main_temperature,
            main_max_tokens=original.main_max_tokens,
            classifier_temperature=original.classifier_temperature,
            evaluator_temperature=original.evaluator_temperature,
        ),
    )

    questions = list(QUESTIONS)[:limit] if limit else list(QUESTIONS)
    turns: list[dict[str, Any]] = []

    try:
        for question in questions:
            session_id = f"bench-{uuid.uuid4().hex[:8]}"
            try:
                # Replay the question's context so follow-ups are fair.
                for role, content in question.context:
                    memory.save_message(session_id, role, content)  # type: ignore[arg-type]

                state = run_turn(question.question, session_id, user_id)
                metrics = state.get("metrics") or {}
                quality = metrics.get("quality") or {}
                scores = quality.get("scores") or {}

                turns.append(
                    {
                        "question_id": question.id,
                        "expected_intent": question.expected_intent,
                        "intent": metrics.get("intent"),
                        "intent_correct": metrics.get("intent") == question.expected_intent,
                        "failed": bool(metrics.get("generation_failed")),
                        "latency_ms": metrics.get("latency_ms", 0),
                        "input_tokens": metrics.get("input_tokens"),
                        "output_tokens": metrics.get("output_tokens"),
                        "estimated_cost_usd": metrics.get("estimated_cost_usd", 0.0),
                        "native_cost_usd": metrics.get("native_cost_usd"),
                        "reward": metrics.get("reward", 0.0),
                        "scores": scores,
                        "deterministic_failures": quality.get("deterministic_failures") or [],
                        "response": state.get("response", ""),
                    }
                )
                status = fail_mark() if turns[-1]["failed"] else "."
                safe_print(f"  {status} {question.id:<24} reward={turns[-1]['reward']:.2f}")
            finally:
                memory.clear_session(session_id)
    finally:
        object.__setattr__(settings_module.settings, "models", original)

    ok = [t for t in turns if not t["failed"]]
    return {
        "model": model,
        "turns": turns,
        "summary": {
            "questions": len(turns),
            "failures": len(turns) - len(ok),
            "intent_accuracy": _mean([1.0 if t["intent_correct"] else 0.0 for t in turns]),
            "reward": _mean([t["reward"] for t in ok]),
            **{dim: _mean([t["scores"].get(dim, 0.0) for t in ok if t["scores"]]) for dim in DIMENSIONS},
            "avg_latency_ms": _mean([float(t["latency_ms"]) for t in ok]),
            "avg_input_tokens": _mean([float(t["input_tokens"] or 0) for t in ok]),
            "avg_output_tokens": _mean([float(t["output_tokens"] or 0) for t in ok]),
            "total_estimated_cost_usd": sum(t["estimated_cost_usd"] for t in ok),
            "cost_per_question_usd": _mean([t["estimated_cost_usd"] for t in ok]),
        },
    }


def print_report(results: list[dict[str, Any]]) -> None:
    safe_print()
    safe_print(LINE)
    safe_print("BENCHMARK RESULTS")
    safe_print(LINE)
    safe_print()

    rows: list[tuple[str, str]] = [
        ("Reward (0-1)", "reward"),
        ("Relevance", "relevance"),
        ("Personalization", "personalization"),
        ("Groundedness", "groundedness"),
        ("Conversation", "conversation_quality"),
        ("Safety", "safety"),
        ("Completeness", "completeness"),
        ("Intent accuracy", "intent_accuracy"),
        ("Avg latency (ms)", "avg_latency_ms"),
        ("Avg input tokens", "avg_input_tokens"),
        ("Avg output tokens", "avg_output_tokens"),
        ("Cost / question", "cost_per_question_usd"),
        ("Failures", "failures"),
    ]

    width = 22
    header = "Metric".ljust(width) + "".join(r["model"][:20].ljust(width) for r in results)
    safe_print(header)
    safe_print("-" * len(header))

    for label, key in rows:
        line = label.ljust(width)
        for result in results:
            value = result["summary"].get(key, 0)
            if key in {"cost_per_question_usd"}:
                cell = f"${value:.6f}"
            elif key in {"failures"}:
                cell = str(int(value))
            elif key in {"avg_latency_ms", "avg_input_tokens", "avg_output_tokens"}:
                cell = f"{value:,.0f}"
            elif key == "intent_accuracy":
                cell = f"{value * 100:.0f}%"
            elif key == "reward":
                cell = f"{value:.3f}"
            else:
                cell = f"{value:.1f}"
            line += cell.ljust(width)
        safe_print(line)

    safe_print()
    safe_print(RULE)
    safe_print("Pick on the quality/cost tradeoff you actually want. The cheapest")
    safe_print("model is not automatically the right one, and neither is the")
    safe_print("highest-scoring one if it costs ten times as much per turn.")
    safe_print(RULE)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OpenRouter models on the same question set")
    parser.add_argument("--models", required=True, help="comma-separated OpenRouter model ids")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N questions")
    parser.add_argument("--user", default=settings.default_user_id, help="user id whose kundli to use")
    parser.add_argument("--out", default=None, help="write the full JSON result to this path")
    args = parser.parse_args()

    if not settings.openrouter.configured:
        safe_print(f"{fail_mark()} OPENROUTER_API_KEY is not set")
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        safe_print(f"{fail_mark()} no models given")
        return 1

    safe_print(LINE)
    safe_print("TREDEV ASTRO AI — MODEL BENCHMARK")
    safe_print(LINE)
    safe_print(f"Questions : {args.limit or len(QUESTIONS)}")
    safe_print(f"Kundli    : {settings.kundli.source} (user {args.user})")
    safe_print(f"Judge     : {settings.models.evaluator}")
    safe_print()

    results: list[dict[str, Any]] = []
    for model in models:
        safe_print(f"Running {model} ...")
        started = time.perf_counter()
        results.append(run_model(model, args.limit, args.user))
        safe_print(f"  done in {time.perf_counter() - started:.1f}s")
        safe_print()

    print_report(results)

    out_path = Path(args.out) if args.out else settings.base_dir / "logs" / (
        f"benchmark-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    safe_print(f"Full results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
