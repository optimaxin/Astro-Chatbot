"""Filters and aggregations shared by the admin analytics API and the Excel export.

Aggregation runs in Python over the matching rows so that days are bucketed
in the report timezone, and so both consumers compute identical numbers.
Costs are always kept per currency — different currencies are never summed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import HTTPException

from app.settings import settings

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STATUSES = ("success", "failed")


def tz() -> ZoneInfo:
    return ZoneInfo(settings.report_timezone)


def local_dt(epoch: float) -> datetime:
    """Naive local datetime (Excel cannot hold tz-aware values)."""
    return datetime.fromtimestamp(epoch, tz()).replace(tzinfo=None)


def _day_start(day: date) -> float:
    return datetime(day.year, day.month, day.day, tzinfo=tz()).timestamp()


def parse_filters(date_from: str | None = None, date_to: str | None = None, provider: str | None = None,
                  model: str | None = None, status: str | None = None, session_id: str | None = None,
                  q: str | None = None) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for label, raw, key, shift in (("from", date_from, "since", 0), ("to", date_to, "until", 1)):
        if raw:
            if not _DATE_RE.fullmatch(raw):
                raise HTTPException(400, f"'{label}' must be YYYY-MM-DD.")
            try:
                day = date.fromisoformat(raw)
            except ValueError as exc:
                raise HTTPException(400, f"'{label}' is not a valid date.") from exc
            filters[key] = _day_start(day + timedelta(days=shift))
    if status and status not in STATUSES:
        raise HTTPException(400, "status must be 'success' or 'failed'.")
    for key, value in (("provider", provider), ("model", model), ("status", status),
                       ("session_id", session_id), ("q", q)):
        if value:
            filters[key] = value.strip()[:200]
    return filters


def _num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _add_cost(bucket: dict[str, float], row: dict[str, Any], field: str = "total_cost") -> None:
    if row.get(field) is not None and row.get("currency"):
        bucket[row["currency"]] = bucket.get(row["currency"], 0.0) + float(row[field])


def _tokens(acc: dict[str, Any], row: dict[str, Any]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        acc[key] += int(_num(row.get(key)))


def _new() -> dict[str, Any]:
    return {"requests": 0, "success": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "cost": {}, "latency_sum": 0, "latency_n": 0,
            "sessions": set(), "models": set()}


def _feed(acc: dict[str, Any], row: dict[str, Any]) -> None:
    acc["requests"] += 1
    acc["success" if row["status"] == "success" else "failed"] += 1
    _tokens(acc, row)
    _add_cost(acc["cost"], row)
    if row.get("latency_ms") is not None:
        acc["latency_sum"] += row["latency_ms"]
        acc["latency_n"] += 1
    if row.get("session_id"):
        acc["sessions"].add(row["session_id"])
    acc["models"].add(f"{row['provider']}/{row['model']}")


def _finish(acc: dict[str, Any]) -> dict[str, Any]:
    return {
        "requests": acc["requests"], "success": acc["success"], "failed": acc["failed"],
        "input_tokens": acc["input_tokens"], "output_tokens": acc["output_tokens"],
        "total_tokens": acc["total_tokens"],
        "cost": {k: round(v, 8) for k, v in acc["cost"].items()},
        "avg_latency_ms": round(acc["latency_sum"] / acc["latency_n"]) if acc["latency_n"] else None,
        "unique_sessions": len(acc["sessions"]),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = _new()
    clients: set[str] = set()
    by_model: dict[tuple[str, str], dict[str, Any]] = defaultdict(_new)
    by_provider: dict[str, dict[str, Any]] = defaultdict(_new)
    daily: dict[str, dict[str, Any]] = defaultdict(_new)
    daily_model: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {**_new(), "input_cost": {}, "output_cost": {}})
    failures: Counter[str] = Counter()
    sessions: dict[str, dict[str, Any]] = {}

    for row in rows:
        day = local_dt(row["created_at"]).date().isoformat()
        for acc in (total, by_model[(row["provider"], row["model"])], by_provider[row["provider"]], daily[day]):
            _feed(acc, row)
        dm = daily_model[(day, row["provider"], row["model"])]
        _feed(dm, row)
        _add_cost(dm["input_cost"], row, "input_cost")
        _add_cost(dm["output_cost"], row, "output_cost")
        if row.get("client_id"):
            clients.add(row["client_id"])
        if row["status"] != "success":
            failures[(row.get("error") or "Unknown error")[:160]] += 1
        sid = row.get("session_id") or "(none)"
        s = sessions.setdefault(sid, {**_new(), "tester_name": None, "first": row["created_at"],
                                      "last": row["created_at"]})
        _feed(s, row)
        s["tester_name"] = row.get("tester_name") or s["tester_name"]
        s["first"] = min(s["first"], row["created_at"])
        s["last"] = max(s["last"], row["created_at"])

    return {
        "totals": {**_finish(total), "unique_clients": len(clients)},
        "by_model": [{"provider": p, "model": m, **_finish(a)}
                     for (p, m), a in sorted(by_model.items(), key=lambda kv: -kv[1]["requests"])],
        "by_provider": [{"provider": p, **_finish(a)}
                        for p, a in sorted(by_provider.items(), key=lambda kv: -kv[1]["requests"])],
        "daily": [{"date": d, **_finish(a)} for d, a in sorted(daily.items())],
        "daily_model": [{"date": d, "provider": p, "model": m, **_finish(a),
                         "input_cost": {k: round(v, 8) for k, v in a["input_cost"].items()},
                         "output_cost": {k: round(v, 8) for k, v in a["output_cost"].items()}}
                        for (d, p, m), a in sorted(daily_model.items())],
        "failures": [{"error": e, "count": c} for e, c in failures.most_common(20)],
        "sessions": [{"session_id": sid, "tester_name": s["tester_name"], **_finish(s),
                      "models": sorted(s["models"]), "first": s["first"], "last": s["last"]}
                     for sid, s in sorted(sessions.items(), key=lambda kv: kv[1]["first"])],
    }
