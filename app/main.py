"""Terminal chat interface.

    cd ai-core
    python -m app.main

Commands: `exit` / `quit`, `/clear`, `/metrics`, `/config`, `/help`.
"""

from __future__ import annotations

import argparse
import textwrap
from typing import Any

from app.config.settings import settings
from app.console import LINE, RULE, fail_mark, ok_mark, safe_print, warn_mark
from app.graph.chatbot_graph import run_turn
from app.llm.llm_factory import describe_roles
from app.llm.usage_tracker import usage_tracker
from app.logging_setup import setup_logging
from app.memory.redis_memory import memory
from app.persona.registry import get_persona, list_personas

WRAP_WIDTH = 72


def _money(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value == 0:
        return "$0.000000"
    return f"${value:.6f}" if value >= 0.000001 else f"${value:.2e}"


def _tokens(value: int | None) -> str:
    return f"{value:,}" if value is not None else "unavailable"


def print_banner(session_id: str, user_id: str, persona_id: str = "") -> None:
    safe_print(LINE)
    safe_print("        TREDEV ASTRO AI")
    safe_print(LINE)
    safe_print()
    safe_print(f"Session : {session_id}")
    safe_print(f"User    : {user_id}")
    safe_print(f"Model   : {settings.models.main}")
    if persona_id:
        safe_print(f"Persona : {get_persona(persona_id).name}")
    safe_print(f"Kundli  : {settings.kundli.source}")
    safe_print()

    warnings: list[str] = []
    if not settings.openrouter.configured:
        warnings.append("OPENROUTER_API_KEY is not set — the assistant cannot answer")
    if not settings.redis.configured:
        warnings.append("Upstash Redis is not configured — this session will not remember anything")
    if not settings.pinecone.configured:
        warnings.append("Pinecone is not configured — answers will not be grounded in retrieved knowledge")
    for warning in warnings:
        safe_print(f"{warn_mark()} {warning}")
    if warnings:
        safe_print("    Run: python -m scripts.test_connections")
        safe_print()

    safe_print("Type 'exit' to quit.")
    safe_print("Type '/clear' to clear conversation.")
    safe_print("Type '/metrics' to show session metrics.")
    safe_print("Type '/usage' to show this user's token + cost meter.")
    safe_print()


def print_response(text: str) -> None:
    safe_print()
    safe_print("Astro AI:")
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            safe_print()
            continue
        for line in textwrap.wrap(paragraph, width=WRAP_WIDTH):
            safe_print(f"  {line}")
    safe_print()


def print_turn_metrics(metrics: dict[str, Any]) -> None:
    safe_print(RULE)
    safe_print("TURN METRICS")
    safe_print(RULE)

    confidence = metrics.get("intent_confidence", 0.0)
    method = metrics.get("intent_method", "")
    suffix = f" ({confidence:.2f}, {method})" if method else ""
    safe_print(f"Intent          : {metrics.get('intent', 'unknown')}{suffix}")
    safe_print(f"Model           : {metrics.get('model', 'none')}"
               f"{'  [fallback]' if metrics.get('fallback_used') else ''}")
    safe_print(f"Kundli          : {metrics.get('kundli_source', '?')}"
               f"{'' if metrics.get('kundli_available') else '  (unavailable)'}")
    safe_print(f"RAG chunks      : {metrics.get('rag_chunks', 0)}"
               f"{'' if metrics.get('rag_available') else '  (unavailable)'}")

    safe_print(f"Input Tokens    : {_tokens(metrics.get('input_tokens'))}")
    safe_print(f"Output Tokens   : {_tokens(metrics.get('output_tokens'))}")
    safe_print(f"Total Tokens    : {_tokens(metrics.get('total_tokens'))}")
    if metrics.get("cached_tokens"):
        safe_print(f"Cached Tokens   : {_tokens(metrics.get('cached_tokens'))}")
    if metrics.get("reasoning_tokens"):
        safe_print(f"Reasoning Tokens: {_tokens(metrics.get('reasoning_tokens'))}")
    safe_print(f"Latency         : {metrics.get('latency_ms', 0):,} ms"
               f"  (turn {metrics.get('turn_latency_ms', 0):,} ms)")
    safe_print(f"Estimated Cost  : {_money(metrics.get('estimated_cost_usd'))}")
    if metrics.get("native_cost_usd") is not None:
        safe_print(f"Provider Cost   : {_money(metrics.get('native_cost_usd'))}")
    if metrics.get("llm_calls", 0) > 1:
        safe_print(f"Turn Cost (all) : {_money(metrics.get('turn_cost_usd'))}"
                   f"  across {metrics['llm_calls']} calls")

    quality = metrics.get("quality") or {}
    scores = quality.get("scores")
    if scores:
        safe_print()
        judge_note = "" if quality.get("judge_used") else "  (judge unavailable — deterministic only)"
        safe_print(f"Quality{judge_note}")
        safe_print(f"Relevance       : {scores.get('relevance', 0):.0f}/10")
        safe_print(f"Personalization : {scores.get('personalization', 0):.0f}/10")
        safe_print(f"Groundedness    : {scores.get('groundedness', 0):.0f}/10")
        safe_print(f"Conversation    : {scores.get('conversation_quality', 0):.0f}/10")
        safe_print(f"Safety          : {scores.get('safety', 0):.0f}/10")
        safe_print(f"Completeness    : {scores.get('completeness', 0):.0f}/10")
        failures = quality.get("deterministic_failures") or []
        if failures:
            safe_print(f"Checks failed   : {', '.join(failures)}")
        safe_print()
        safe_print(f"Reward          : {metrics.get('reward', 0.0):.2f}")

    errors = metrics.get("errors") or []
    if errors:
        safe_print()
        for error in errors:
            safe_print(f"{warn_mark()} {error}")

    safe_print(RULE)
    safe_print()


def print_session_metrics(session_id: str) -> None:
    totals = usage_tracker.totals()
    safe_print()
    safe_print(RULE)
    safe_print("SESSION METRICS")
    safe_print(RULE)
    safe_print(f"Session         : {session_id}")
    safe_print(f"Messages stored : {memory.count_messages(session_id)}")
    safe_print(f"LLM calls       : {totals['calls']}"
               f"  ({totals['failed_calls']} failed)")
    safe_print(f"Input Tokens    : {totals['input_tokens']:,}")
    safe_print(f"Output Tokens   : {totals['output_tokens']:,}")
    safe_print(f"Total Tokens    : {totals['total_tokens']:,}")
    if totals["tokens_unavailable_for"]:
        safe_print(f"                  ({totals['tokens_unavailable_for']} call(s) reported no usage)")
    safe_print(f"Avg Latency     : {totals['avg_latency_ms']:,} ms")
    safe_print(f"Estimated Cost  : {_money(totals['estimated_cost_usd'])}")
    if totals["native_cost_usd"]:
        safe_print(f"Provider Cost   : {_money(totals['native_cost_usd'])}")
    safe_print(RULE)
    safe_print()


def print_usage_meter(user_id: str) -> None:
    """Per-user lifetime totals, not just this process's."""
    from app.llm.llm_factory import ROLE_MAIN
    from app.llm.usage_ledger import aggregate, read_ledger, user_totals

    safe_print()
    safe_print(RULE)
    safe_print(f"USAGE METER — {user_id}")
    safe_print(RULE)

    records = list(read_ledger())
    for label, role in (("This user, all roles", None), ("This user, answers only", ROLE_MAIN)):
        summary = aggregate(records, group_by="user_id", role=role).get(user_id)
        if summary is None:
            continue
        exact = "" if summary.tokens_exact else "  (undercount: provider omitted usage)"
        safe_print(f"{label}")
        safe_print(f"    Calls          : {summary.calls}")
        safe_print(f"    Input tokens   : {summary.input_tokens:,}")
        safe_print(f"    Output tokens  : {summary.output_tokens:,}")
        safe_print(f"    Total tokens   : {summary.total_tokens:,}{exact}")
        safe_print(f"    Estimated cost : {_money(summary.estimated_cost_usd)}")
        safe_print(f"    Cost per call  : {_money(summary.cost_per_call_usd)}")
        safe_print()

    live = user_totals(user_id)
    if live:
        safe_print(f"Live Redis counters : {live['total_tokens']:,} tokens, "
                   f"{_money(live['estimated_cost_usd'])}")
    else:
        safe_print(f"{warn_mark()} Redis counters unavailable (Upstash not configured);")
        safe_print("    the on-disk ledger above is the source of truth.")
    safe_print()
    safe_print("Full report: python -m scripts.usage_report")
    safe_print(RULE)
    safe_print()


def print_config() -> None:
    safe_print()
    safe_print(RULE)
    safe_print("CONFIGURATION")
    safe_print(RULE)
    for role, cfg in describe_roles().items():
        price = cfg["price_per_1m"]
        safe_print(f"{role}")
        safe_print(f"    model    : {cfg['model']}")
        safe_print(f"    fallback : {cfg['fallback'] or '(none)'}")
        safe_print(f"    price/1M : ${price['input']} in / ${price['output']} out")
    safe_print(f"Kundli source   : {settings.kundli.source}")
    safe_print(f"RAG enabled     : {settings.enable_rag}")
    safe_print(f"Evaluation      : {settings.enable_evaluation}")
    safe_print(f"History window  : {settings.redis.window} of {settings.redis.max_messages} kept")
    safe_print(RULE)
    safe_print()


def print_help() -> None:
    safe_print()
    safe_print("  exit, quit  — leave the chat")
    safe_print("  /clear      — erase this session's conversation from Redis")
    safe_print("  /metrics    — token, latency and cost totals for this session")
    safe_print("  /usage      — lifetime token and cost meter for this user")
    safe_print("  /personas   — the astrologer personas available")
    safe_print("  /config     — active model routing and settings")
    safe_print("  /help       — this list")
    safe_print()


def handle_command(command: str, session_id: str, user_id: str) -> bool:
    """Returns True if the input was a command. Exits the process on `exit`."""
    lowered = command.strip().lower()

    if lowered in {"exit", "quit", "/exit", "/quit"}:
        print_session_metrics(session_id)
        safe_print("Goodbye.")
        raise SystemExit(0)

    if lowered == "/clear":
        if memory.clear_session(session_id):
            safe_print(f"{ok_mark()} conversation cleared for session '{session_id}'")
        else:
            safe_print(f"{fail_mark()} could not clear the session — is Upstash Redis configured?")
        safe_print()
        return True

    if lowered == "/metrics":
        print_session_metrics(session_id)
        return True

    if lowered == "/usage":
        print_usage_meter(user_id)
        return True

    if lowered == "/personas":
        safe_print()
        for persona in list_personas():
            safe_print(f"  {persona.id:<24} {persona.name} — {persona.specialization}")
        safe_print()
        safe_print("  Restart with --persona <id> to switch.")
        safe_print()
        return True

    if lowered == "/config":
        print_config()
        return True

    if lowered in {"/help", "/?"}:
        print_help()
        return True

    if lowered.startswith("/"):
        safe_print(f"{fail_mark()} unknown command '{command}'. Try /help")
        safe_print()
        return True

    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Tredev Astro AI — terminal chat")
    parser.add_argument("--session", default=settings.default_session_id, help="session id")
    parser.add_argument("--user", default=settings.default_user_id, help="user id")
    parser.add_argument(
        "--persona",
        default=settings.api.default_persona_id,
        choices=[p.id for p in list_personas()],
        help="astrologer persona",
    )
    parser.add_argument("--no-metrics", action="store_true", help="hide the per-turn metrics block")
    args = parser.parse_args()

    setup_logging()
    print_banner(args.session, args.user, args.persona)

    while True:
        try:
            question = input("You:\n  ").strip()
        except (EOFError, KeyboardInterrupt):
            safe_print()
            print_session_metrics(args.session)
            safe_print("Goodbye.")
            return 0

        if not question:
            continue
        if handle_command(question, args.session, args.user):
            continue

        state = run_turn(question, args.session, args.user, args.persona)
        print_response(state.get("response", ""))
        if not args.no_metrics:
            print_turn_metrics(state.get("metrics") or {})


if __name__ == "__main__":
    raise SystemExit(main())
