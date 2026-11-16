from __future__ import annotations

import io
import time
from dataclasses import replace

import pytest
from openpyxl import load_workbook

from app import admin_auth, storage
from app.excel_export import EXCEL_CELL_LIMIT, safe_text

from .conftest import ADMIN_PASSWORD

SHEETS = ["All Test Logs", "Model Summary", "Token Analysis", "Cost Analysis", "Daily Statistics",
          "Failed Requests", "Feedback & Evaluation", "Session Summary"]


def _seed(**overrides) -> dict:
    row = {"provider": "gemini", "model": "gemini-2.5-flash", "user_prompt": "Hi", "status": "success",
           "response": "Hello", "session_id": "s-1", "client_id": "client-aaaa1111", "input_tokens": 100,
           "output_tokens": 50, "total_tokens": 150, "input_cost": 0.00003, "output_cost": 0.000125,
           "total_cost": 0.000155, "currency": "USD", "latency_ms": 800, **overrides}
    return storage.create_experiment(row)


# ── authentication ───────────────────────────────────────────────────
def test_admin_routes_reject_anonymous_callers(client) -> None:
    for path in ("/api/admin/me", "/api/admin/analytics", "/api/admin/experiments", "/api/admin/experiments/1",
                 "/api/admin/models", "/api/admin/audit-logs", "/api/admin/export/excel", "/api/admin/backup"):
        assert client.get(path).status_code == 401, path


def test_wrong_password_is_rejected_and_audited(client) -> None:
    assert client.post("/api/admin/login", json={"password": "nope"}).status_code == 401
    logs, _ = storage.audit_logs(10, 0)
    assert logs[0]["action"] == "admin.login_failed"


def test_login_sets_a_hardened_cookie_and_me_works(client) -> None:
    res = client.post("/api/admin/login", json={"password": ADMIN_PASSWORD})
    cookie = res.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/api/admin" in cookie
    assert client.get("/api/admin/me").json()["authenticated"] is True


def test_login_is_rate_limited(client) -> None:
    for _ in range(admin_auth.LOGIN_ATTEMPTS):
        client.post("/api/admin/login", json={"password": "wrong"})
    assert client.post("/api/admin/login", json={"password": ADMIN_PASSWORD}).status_code == 429


def test_logout_needs_csrf_and_ends_the_session(admin) -> None:
    assert admin.post("/api/admin/logout").status_code == 403
    assert admin.post("/api/admin/logout", headers={"X-CSRF-Token": admin.csrf}).status_code == 200
    assert admin.get("/api/admin/me").status_code == 401


def test_expired_session_is_rejected(admin) -> None:
    with storage._lock:
        conn = storage._connection()
        with conn:
            conn.execute("UPDATE admin_sessions SET expires_at = ?", (time.time() - 1,))
    assert admin.get("/api/admin/me").status_code == 401


def test_unconfigured_admin_password_fails_closed(client, monkeypatch) -> None:
    monkeypatch.setattr(admin_auth, "settings", replace(admin_auth.settings, admin_password_hash=""))
    assert client.post("/api/admin/login", json={"password": ADMIN_PASSWORD}).status_code == 503


# ── analytics and experiments ────────────────────────────────────────
def test_analytics_match_the_database(admin) -> None:
    _seed()
    _seed(session_id="s-2", client_id="client-bbbb2222")
    _seed(status="failed", response=None, error="Rate limit exceeded.", input_tokens=None, output_tokens=None,
          total_tokens=None, input_cost=None, output_cost=None, total_cost=None)
    body = admin.get("/api/admin/analytics").json()
    t = body["totals"]
    assert (t["requests"], t["success"], t["failed"]) == (3, 2, 1)
    assert t["total_tokens"] == 300 and t["cost"]["USD"] == pytest.approx(0.00031)
    assert t["unique_sessions"] == 2 and t["unique_clients"] == 2
    assert body["by_model"][0]["requests"] == 3
    assert body["failures"] == [{"error": "Rate limit exceeded.", "count": 1}]


def test_experiment_filters_and_pagination(admin) -> None:
    for i in range(5):
        _seed(user_prompt=f"prompt {i}")
    _seed(provider="openrouter", model="openai/gpt-4o-mini", status="failed", error="x")
    page = admin.get("/api/admin/experiments", params={"page_size": 2, "page": 2}).json()
    assert page["total"] == 6 and len(page["items"]) == 2
    assert admin.get("/api/admin/experiments", params={"status": "failed"}).json()["total"] == 1
    assert admin.get("/api/admin/experiments", params={"model": "gemini-2.5-flash"}).json()["total"] == 5
    assert admin.get("/api/admin/experiments", params={"q": "prompt 3"}).json()["total"] == 1
    assert admin.get("/api/admin/experiments", params={"q": "100%"}).json()["total"] == 0
    assert admin.get("/api/admin/experiments", params={"from": "2001-13-01"}).status_code == 400
    assert admin.get("/api/admin/experiments", params={"from": "2001-01-01", "to": "2001-01-02"}).json()["total"] == 0


def test_viewing_an_experiment_is_audited(admin) -> None:
    exp = _seed()
    assert admin.get(f"/api/admin/experiments/{exp['id']}").json()["response"] == "Hello"
    logs = admin.get("/api/admin/audit-logs").json()["items"]
    assert logs[0]["action"] == "admin.view_experiment" and logs[0]["details"] == {"experiment_id": exp["id"]}


# ── Excel ────────────────────────────────────────────────────────────
def _workbook(admin, **params):
    res = admin.get("/api/admin/export/excel", params=params)
    assert res.status_code == 200
    assert "AI_Playground_Company_Testing_Logs.xlsx" in res.headers["content-disposition"]
    return load_workbook(io.BytesIO(res.content))


def _data_rows(ws) -> list[tuple]:
    rows = []
    for row in ws.iter_rows(min_row=5, values_only=True):
        if row[0] in (None, "Total"):
            break
        rows.append(row)
    return rows


def test_excel_has_all_sheets_and_every_record(admin) -> None:
    for _ in range(3):
        _seed()
    _seed(status="failed", error="boom", response=None, total_cost=None, input_cost=None, output_cost=None,
          input_tokens=None, output_tokens=None, total_tokens=None)
    wb = _workbook(admin)
    assert wb.sheetnames == SHEETS
    logs = _data_rows(wb["All Test Logs"])
    assert len(logs) == 4
    assert len(_data_rows(wb["Failed Requests"])) == 1
    total_row = next(r for r in wb["All Test Logs"].iter_rows(values_only=True) if r[0] == "Total")
    assert total_row[15] == 450 and total_row[19] == pytest.approx(0.000465)
    assert wb["All Test Logs"].freeze_panes == "A5"
    assert wb["All Test Logs"].auto_filter.ref.startswith("A4:")


def test_excel_filters(admin) -> None:
    _seed()
    _seed(model="gemini-2.5-flash-lite")
    _seed(status="failed", error="boom")
    assert len(_data_rows(_workbook(admin, model="gemini-2.5-flash-lite")["All Test Logs"])) == 1
    assert len(_data_rows(_workbook(admin, status="failed")["All Test Logs"])) == 1
    assert len(_data_rows(_workbook(admin, session_id="nope")["All Test Logs"])) == 0


def test_excel_neutralises_formulas_and_preserves_text(admin) -> None:
    nasty = '=HYPERLINK("http://evil","click")'
    multiline = "line one\nline two — नमस्ते 🙏"
    _seed(user_prompt=nasty, response=multiline, tester_name="@admin")
    ws = _workbook(admin)["All Test Logs"]
    prompt = ws.cell(row=5, column=10)
    assert prompt.value == nasty and prompt.data_type == "s" and prompt.quotePrefix
    assert ws.cell(row=5, column=12).value == multiline
    assert ws.cell(row=5, column=4).data_type == "s"


def test_excel_keeps_currencies_apart(admin) -> None:
    _seed()
    _seed(currency="EUR", total_cost=1.0, input_cost=0.4, output_cost=0.6)
    rows = _data_rows(_workbook(admin)["Cost Analysis"])
    assert sorted(r[6] for r in rows) == ["EUR", "USD"]


def test_safe_text_strips_control_chars_and_caps_length() -> None:
    assert safe_text("a\x00b\x1fc") == "abc"
    assert len(safe_text("x" * 40000)) == EXCEL_CELL_LIMIT


def test_backup_download_is_a_valid_database(admin, tmp_path) -> None:
    _seed()
    res = admin.get("/api/admin/backup")
    assert res.status_code == 200 and res.content[:15] == b"SQLite format 3"


def test_database_survives_a_restart(tmp_path) -> None:
    path = tmp_path / "persist.db"
    storage.reset_for_tests(path)
    exp = _seed()
    storage.close()
    storage.reset_for_tests(path)
    assert storage.get_experiment(exp["id"])["response"] == "Hello"
