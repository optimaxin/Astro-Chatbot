"""The admin Excel report: eight sheets built straight from SQLite.

Untrusted text (prompts, responses, feedback, tester names) is written as a
typed string with Excel's quote-prefix, so a value such as "=HYPERLINK(...)"
is shown, never evaluated. Characters XML cannot hold are dropped and cells
are capped at Excel's 32,767-character limit (with a visible marker).
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Callable

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from app.evaluator import METRICS
from app.reports import aggregate, local_dt
from app.settings import settings

FILENAME = "AI_Playground_Company_Testing_Logs.xlsx"
EXCEL_CELL_LIMIT = 32767
_TRUNCATED = "… [truncated: exceeds Excel's cell limit]"
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")

TITLE = Font(bold=True, size=14)
HEADER = Font(bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="1F2937")
OK_FILL = PatternFill("solid", fgColor="DCFCE7")
FAIL_FILL = PatternFill("solid", fgColor="FEE2E2")
TOTAL_FONT = Font(bold=True)
WRAP = Alignment(wrap_text=True, vertical="top")

INT = "#,##0"
COST = "0.000000"
MS = "#,##0"
DATE = "yyyy-mm-dd"
TIME = "hh:mm:ss"
DATETIME = "yyyy-mm-dd hh:mm:ss"


def safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = ILLEGAL_CHARACTERS_RE.sub("", str(value))
    if len(text) > EXCEL_CELL_LIMIT:
        text = text[: EXCEL_CELL_LIMIT - len(_TRUNCATED)] + _TRUNCATED
    return text


# (header, width, value-getter, number format or "text")
Column = tuple[str, int, Callable[[dict[str, Any]], Any], str | None]


def _write_sheet(wb: Workbook, title: str, heading: str, subtitle: str, columns: list[Column],
                 rows: list[dict[str, Any]], status_key: str | None = None,
                 totals: dict[int, Any] | None = None) -> Worksheet:
    ws = wb.create_sheet(title)
    ws.cell(row=1, column=1, value=heading).font = TITLE
    ws.cell(row=2, column=1, value=safe_text(subtitle))
    header_row = 4
    for i, (name, width, _, _) in enumerate(columns, start=1):
        cell = ws.cell(row=header_row, column=i, value=name)
        cell.font, cell.fill = HEADER, HEADER_FILL
        ws.column_dimensions[get_column_letter(i)].width = width
    r = header_row
    for r, row in enumerate(rows, start=header_row + 1):
        for i, (_, _, get, fmt) in enumerate(columns, start=1):
            value = get(row)
            if fmt == "text":
                value = safe_text(value)
            cell = ws.cell(row=r, column=i, value=value)
            if fmt == "text" and isinstance(value, str):
                # Assignment of "=..." makes openpyxl store a formula; force a string.
                cell.data_type = "s"
                if value.startswith(_FORMULA_START):
                    cell.quotePrefix = True
                if len(value) > 60:
                    cell.alignment = WRAP
            elif fmt:
                cell.number_format = fmt
        if status_key:
            status = row.get(status_key)
            fill = OK_FILL if status == "success" else FAIL_FILL if status == "failed" else None
            if fill:
                for i, (name, *_rest) in enumerate(columns, start=1):
                    if name == "Status":
                        ws.cell(row=r, column=i).fill = fill
    last_data = r
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
    ws.auto_filter.ref = f"A{header_row}:{get_column_letter(len(columns))}{max(last_data, header_row)}"
    if totals and rows:
        tr = last_data + 2
        ws.cell(row=tr, column=1, value="Total").font = TOTAL_FONT
        for col, value in totals.items():
            cell = ws.cell(row=tr, column=col, value=value)
            cell.font = TOTAL_FONT
            cell.number_format = columns[col - 1][3] or INT
    return ws


def _cost_text(costs: dict[str, float]) -> str | None:
    return ", ".join(f"{v:.6f} {k}" for k, v in sorted(costs.items())) or None


def _usd(costs: dict[str, float]) -> float | None:
    return costs.get("USD")


def build_workbook(rows: list[dict[str, Any]], filter_note: str) -> bytes:
    agg = aggregate(rows)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    sub = f"Generated {generated} · times in {settings.report_timezone} · {filter_note}"
    wb = Workbook()
    wb.remove(wb.active)

    def ev(row: dict[str, Any]) -> dict[str, Any]:
        return row.get("evaluation") or {}

    logs: list[Column] = [
        ("Experiment ID", 13, lambda r: r["id"], INT),
        ("Request ID", 24, lambda r: r.get("request_id"), "text"),
        ("Session ID", 18, lambda r: r.get("session_id"), "text"),
        ("Tester", 16, lambda r: r.get("tester_name"), "text"),
        ("Date", 12, lambda r: local_dt(r["created_at"]).date(), DATE),
        ("Time", 10, lambda r: local_dt(r["created_at"]).time(), TIME),
        ("Provider", 12, lambda r: r["provider"], "text"),
        ("Model", 30, lambda r: r["model"], "text"),
        ("System prompt", 40, lambda r: r.get("system_prompt"), "text"),
        ("User prompt", 50, lambda r: r.get("user_prompt"), "text"),
        ("Context", 40, lambda r: r.get("context"), "text"),
        ("Generated response", 60, lambda r: r.get("response"), "text"),
        ("Humanized response", 40, lambda r: r.get("humanized_response"), "text"),
        ("Input tokens", 12, lambda r: r.get("input_tokens"), INT),
        ("Output tokens", 12, lambda r: r.get("output_tokens"), INT),
        ("Total tokens", 12, lambda r: r.get("total_tokens"), INT),
        ("Usage source", 13, lambda r: r.get("usage_source"), "text"),
        ("Input cost", 12, lambda r: r.get("input_cost"), COST),
        ("Output cost", 12, lambda r: r.get("output_cost"), COST),
        ("Total cost", 12, lambda r: r.get("total_cost"), COST),
        ("Currency", 9, lambda r: r.get("currency"), "text"),
        ("Latency (ms)", 12, lambda r: r.get("latency_ms"), MS),
        ("First token latency (ms)", 14, lambda r: None, MS),
        ("Status", 10, lambda r: r["status"], "text"),
        ("Error", 40, lambda r: r.get("error"), "text"),
    ]
    t = agg["totals"]
    _write_sheet(wb, "All Test Logs", "All Test Logs",
                 sub + " · first-token latency is not measured (responses are not streamed)",
                 logs, rows, "status",
                 {14: t["input_tokens"], 15: t["output_tokens"], 16: t["total_tokens"], 20: _usd(t["cost"])})

    model_cols: list[Column] = [
        ("Provider", 14, lambda r: r["provider"], "text"),
        ("Model", 34, lambda r: r["model"], "text"),
        ("Total requests", 14, lambda r: r["requests"], INT),
        ("Successful", 12, lambda r: r["success"], INT),
        ("Failed", 10, lambda r: r["failed"], INT),
        ("Input tokens", 14, lambda r: r["input_tokens"], INT),
        ("Output tokens", 14, lambda r: r["output_tokens"], INT),
        ("Total tokens", 14, lambda r: r["total_tokens"], INT),
        ("Total cost (USD)", 16, lambda r: _usd(r["cost"]), COST),
        ("Other currencies", 18, lambda r: _cost_text({k: v for k, v in r["cost"].items() if k != "USD"}), "text"),
        ("Avg latency (ms)", 16, lambda r: r["avg_latency_ms"], MS),
    ]
    _write_sheet(wb, "Model Summary", "Model Summary", sub, model_cols, agg["by_model"],
                 totals={3: t["requests"], 4: t["success"], 5: t["failed"], 6: t["input_tokens"],
                         7: t["output_tokens"], 8: t["total_tokens"], 9: _usd(t["cost"])})

    token_cols: list[Column] = [
        ("Date", 12, lambda r: r["date"], "text"),
        ("Provider", 14, lambda r: r["provider"], "text"),
        ("Model", 34, lambda r: r["model"], "text"),
        ("Requests", 10, lambda r: r["requests"], INT),
        ("Input tokens", 14, lambda r: r["input_tokens"], INT),
        ("Output tokens", 14, lambda r: r["output_tokens"], INT),
        ("Total tokens", 14, lambda r: r["total_tokens"], INT),
        ("Avg tokens / request", 18, lambda r: round(r["total_tokens"] / r["requests"]) if r["requests"] else None, INT),
    ]
    _write_sheet(wb, "Token Analysis", "Token Analysis", sub, token_cols, agg["daily_model"],
                 totals={4: t["requests"], 5: t["input_tokens"], 6: t["output_tokens"], 7: t["total_tokens"]})

    cost_rows = [{**r, "currency": cur, "in": r["input_cost"].get(cur), "out": r["output_cost"].get(cur),
                  "tot": r["cost"].get(cur)}
                 for r in agg["daily_model"] for cur in sorted(r["cost"])]
    cost_cols: list[Column] = [
        ("Date", 12, lambda r: r["date"], "text"),
        ("Provider", 14, lambda r: r["provider"], "text"),
        ("Model", 34, lambda r: r["model"], "text"),
        ("Input cost", 14, lambda r: r["in"], COST),
        ("Output cost", 14, lambda r: r["out"], COST),
        ("Total estimated cost", 18, lambda r: r["tot"], COST),
        ("Currency", 10, lambda r: r["currency"], "text"),
    ]
    _write_sheet(wb, "Cost Analysis", "Cost Analysis",
                 sub + " · one row per currency; currencies are never added together", cost_cols, cost_rows)

    daily_cols: list[Column] = [
        ("Date", 12, lambda r: r["date"], "text"),
        ("Total experiments", 16, lambda r: r["requests"], INT),
        ("Successful", 12, lambda r: r["success"], INT),
        ("Failed", 10, lambda r: r["failed"], INT),
        ("Unique sessions", 15, lambda r: r["unique_sessions"], INT),
        ("Total tokens", 14, lambda r: r["total_tokens"], INT),
        ("Total cost (USD)", 16, lambda r: _usd(r["cost"]), COST),
        ("Other currencies", 18, lambda r: _cost_text({k: v for k, v in r["cost"].items() if k != "USD"}), "text"),
    ]
    _write_sheet(wb, "Daily Statistics", "Daily Statistics", sub, daily_cols, agg["daily"],
                 totals={2: t["requests"], 3: t["success"], 4: t["failed"], 6: t["total_tokens"], 7: _usd(t["cost"])})

    failed = [r for r in rows if r["status"] != "success"]
    fail_cols: list[Column] = [
        ("Experiment ID", 13, lambda r: r["id"], INT),
        ("Timestamp", 20, lambda r: local_dt(r["created_at"]), DATETIME),
        ("Provider", 12, lambda r: r["provider"], "text"),
        ("Model", 30, lambda r: r["model"], "text"),
        ("Prompt", 60, lambda r: r.get("user_prompt"), "text"),
        ("Error details", 60, lambda r: r.get("error"), "text"),
        ("HTTP status", 11, lambda r: (r.get("metadata") or {}).get("status_code"), INT),
        ("Status", 10, lambda r: r["status"], "text"),
    ]
    _write_sheet(wb, "Failed Requests", "Failed Requests", sub, fail_cols, failed, "status")

    reviewed = [r for r in rows if r.get("evaluation") or r.get("human_rating") or r.get("human_feedback")
                or r.get("edited_response")]
    fb_cols: list[Column] = [
        ("Experiment ID", 13, lambda r: r["id"], INT),
        ("Provider", 12, lambda r: r["provider"], "text"),
        ("Model", 30, lambda r: r["model"], "text"),
        *[(m.replace("_", " ").title(), 12, (lambda key: lambda r: (ev(r).get("scores") or {}).get(key))(m), "0.0")
          for m in METRICS],
        ("AI overall score", 14, lambda r: r.get("evaluation_overall"), "0.0"),
        ("AI feedback", 50, lambda r: ev(r).get("feedback"), "text"),
        ("Evaluator model", 24, lambda r: (ev(r).get("evaluator") or {}).get("model"), "text"),
        ("Human rating", 12, lambda r: r.get("human_rating"), "text"),
        ("Human feedback", 50, lambda r: r.get("human_feedback"), "text"),
        ("Edited response", 60, lambda r: r.get("edited_response"), "text"),
    ]
    _write_sheet(wb, "Feedback & Evaluation", "Feedback & Evaluation",
                 sub + " · AI scores are an automated estimate, not ground truth", fb_cols, reviewed)

    sess_cols: list[Column] = [
        ("Session ID", 22, lambda r: r["session_id"], "text"),
        ("Tester", 18, lambda r: r.get("tester_name"), "text"),
        ("Total experiments", 16, lambda r: r["requests"], INT),
        ("Models tested", 50, lambda r: ", ".join(r["models"]), "text"),
        ("Total tokens", 14, lambda r: r["total_tokens"], INT),
        ("Total cost (USD)", 16, lambda r: _usd(r["cost"]), COST),
        ("First activity", 20, lambda r: local_dt(r["first"]), DATETIME),
        ("Last activity", 20, lambda r: local_dt(r["last"]), DATETIME),
    ]
    _write_sheet(wb, "Session Summary", "Session Summary", sub, sess_cols, agg["sessions"])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
