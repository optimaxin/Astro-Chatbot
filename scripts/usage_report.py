"""Token and cost report, per user / per session / per model.

    python -m scripts.usage_report                     # per user, all roles
    python -m scripts.usage_report --by session
    python -m scripts.usage_report --by model
    python -m scripts.usage_report --answers-only      # exclude classifier/judge overhead
    python -m scripts.usage_report --user test-user-001
    python -m scripts.usage_report --json

Reads `logs/usage.jsonl`, the complete append-only record of every LLM call.
When Upstash Redis is configured, `--user` also shows that user's live Redis
counters, which are the O(1) figures a quota check or billing lookup would
use in production.

Every call is attributed to a role:

    job-main-turn         the user-visible answer
    job-classify-intent   intent classification
    job-evaluate-response the quality judge

`--answers-only` restricts the report to `job-main-turn`, which is what you
want when asking "what did this user's answers cost". The default includes
everything, which is what you want when asking "what did this user cost us".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.console import LINE, RULE, safe_print, warn_mark  # noqa: E402
from app.llm.llm_factory import ROLE_MAIN  # noqa: E402
from app.llm.usage_ledger import (  # noqa: E402
    aggregate,
    ledger_path,
    read_ledger,
    user_totals,
)

GROUPS = {"user": "user_id", "session": "session_id", "model": "model", "role": "role"}


def _money(value: float) -> str:
    if value == 0:
        return "$0.000000"
    return f"${value:.6f}" if value >= 0.000001 else f"${value:.2e}"


def print_table(summaries: dict, group: str) -> None:
    rows = sorted(summaries.values(), key=lambda s: s.estimated_cost_usd, reverse=True)
    if not rows:
        safe_print("No usage recorded yet.")
        return

    key_width = max(len(group), *(len(r.key) for r in rows)) + 2
    header = (
        f"{group.upper():<{key_width}}"
        f"{'CALLS':>7}{'INPUT':>12}{'OUTPUT':>12}{'TOTAL':>12}"
        f"{'COST':>14}{'$/CALL':>13}{'LATENCY':>10}"
    )
    safe_print(header)
    safe_print("-" * len(header))

    for row in rows:
        exact = "" if row.tokens_exact else "*"
        safe_print(
            f"{row.key:<{key_width}}"
            f"{row.calls:>7}"
            f"{row.input_tokens:>12,}"
            f"{row.output_tokens:>12,}"
            f"{row.total_tokens:>11,}{exact:<1}"
            f"{_money(row.estimated_cost_usd):>14}"
            f"{_money(row.cost_per_call_usd):>13}"
            f"{row.avg_latency_ms:>9,}ms"
        )

    safe_print("-" * len(header))
    safe_print(
        f"{'TOTAL':<{key_width}}"
        f"{sum(r.calls for r in rows):>7}"
        f"{sum(r.input_tokens for r in rows):>12,}"
        f"{sum(r.output_tokens for r in rows):>12,}"
        f"{sum(r.total_tokens for r in rows):>12,}"
        f"{_money(sum(r.estimated_cost_usd for r in rows)):>14}"
    )

    if any(not r.tokens_exact for r in rows):
        safe_print()
        safe_print(f"{warn_mark()} * token totals are an undercount: the provider reported no")
        safe_print("    usage for some calls, and those are never guessed at.")

    failed = sum(r.failed_calls for r in rows)
    if failed:
        safe_print(f"{warn_mark()} {failed} failed call(s) are counted but cost nothing.")


def print_user_detail(user_id: str, records: list[dict]) -> None:
    safe_print(LINE)
    safe_print(f"USER: {user_id}")
    safe_print(LINE)

    own = [r for r in records if r.get("user_id") == user_id]
    if not own:
        safe_print("No ledger entries for this user.")
        return

    for label, role in (("All roles", None), ("Answers only", ROLE_MAIN)):
        summaries = aggregate(own, group_by="user_id", role=role)
        summary = summaries.get(user_id)
        if summary is None:
            continue
        safe_print()
        safe_print(f"{label}")
        safe_print(f"    Calls          : {summary.calls} ({summary.failed_calls} failed)")
        safe_print(f"    Input tokens   : {summary.input_tokens:,}")
        safe_print(f"    Output tokens  : {summary.output_tokens:,}")
        safe_print(f"    Total tokens   : {summary.total_tokens:,}"
                   f"{'' if summary.tokens_exact else '  (undercount — see note below)'}")
        if summary.cached_tokens:
            safe_print(f"    Cached tokens  : {summary.cached_tokens:,}")
        safe_print(f"    Estimated cost : {_money(summary.estimated_cost_usd)}")
        if summary.native_cost_usd:
            safe_print(f"    Provider cost  : {_money(summary.native_cost_usd)}")
        safe_print(f"    Cost per call  : {_money(summary.cost_per_call_usd)}")
        safe_print(f"    Avg latency    : {summary.avg_latency_ms:,} ms")
        safe_print(f"    Models         : {', '.join(summary.models) or '(none)'}")
        if role is None:
            safe_print(f"    Calls by role  : {dict(summary.by_role)}")

    live = user_totals(user_id)
    safe_print()
    if live:
        safe_print("Live Redis counters (the O(1) path for quotas and billing)")
        safe_print(f"    Calls          : {live['calls']}")
        safe_print(f"    Input tokens   : {live['input_tokens']:,}")
        safe_print(f"    Output tokens  : {live['output_tokens']:,}")
        safe_print(f"    Total tokens   : {live['total_tokens']:,}")
        safe_print(f"    Estimated cost : {_money(live['estimated_cost_usd'])}")
    else:
        safe_print(f"{warn_mark()} No Redis counters — Upstash is not configured, so the")
        safe_print("    ledger above is the only source. Both agree once Redis is set up.")

    safe_print()
    safe_print(RULE)
    safe_print("Most recent calls")
    safe_print(RULE)
    for record in own[-10:]:
        tokens = (
            f"{record.get('input_tokens')}/{record.get('output_tokens')}"
            if record.get("input_tokens") is not None
            else "usage unavailable"
        )
        safe_print(
            f"  {str(record.get('role')):<22} {str(record.get('model'))[:32]:<34} "
            f"{tokens:>18}  {_money(float(record.get('total_cost_usd') or 0)):>12}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-user token and cost report")
    parser.add_argument("--by", choices=sorted(GROUPS), default="user", help="grouping (default: user)")
    parser.add_argument("--user", help="detailed report for one user id")
    parser.add_argument("--answers-only", action="store_true",
                        help=f"only {ROLE_MAIN} calls — excludes classifier and judge overhead")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    path = ledger_path()
    records = list(read_ledger())

    if not records:
        safe_print(f"{warn_mark()} No usage recorded yet ({path} is empty or missing).")
        safe_print("    Run the chatbot first:  python -m app.main")
        return 0

    role = ROLE_MAIN if args.answers_only else None

    if args.user:
        if args.json:
            summaries = aggregate(records, group_by="user_id", role=role)
            summary = summaries.get(args.user)
            safe_print(json.dumps(summary.to_dict() if summary else {}, indent=2))
        else:
            print_user_detail(args.user, records)
        return 0

    summaries = aggregate(records, group_by=GROUPS[args.by], role=role)

    if args.json:
        safe_print(json.dumps({k: v.to_dict() for k, v in summaries.items()}, indent=2))
        return 0

    safe_print(LINE)
    safe_print("TREDEV ASTRO AI — USAGE REPORT")
    safe_print(LINE)
    safe_print(f"Ledger  : {path}")
    safe_print(f"Calls   : {len(records)}")
    safe_print(f"Scope   : {'answers only (' + ROLE_MAIN + ')' if role else 'all roles'}")
    safe_print()
    print_table(summaries, args.by)
    safe_print()
    if not args.answers_only:
        safe_print("Tip: --answers-only excludes classifier and judge calls, showing what")
        safe_print("     the user-visible answers alone cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
