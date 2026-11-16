"""Admin API — every route except /login requires an admin session."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Cookie, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import admin_auth, storage
from app.auth import client_ip
from app.catalog import get_catalog
from app.excel_export import FILENAME, build_workbook
from app.reports import aggregate, parse_filters

router = APIRouter(prefix="/api/admin")
admin = [Depends(admin_auth.require_admin)]

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class LoginRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=256)


def _filters(date_from: str | None = Query(None, alias="from"), date_to: str | None = Query(None, alias="to"),
             provider: str | None = None, model: str | None = None, status: str | None = None,
             session_id: str | None = None, q: str | None = None) -> dict[str, Any]:
    return parse_filters(date_from, date_to, provider, model, status, session_id, q)


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response) -> dict[str, str]:
    return admin_auth.login(request, response, body.password)


@router.post("/logout", dependencies=[Depends(admin_auth.require_csrf)])
def logout(request: Request, response: Response, pg_admin: str | None = Cookie(default=None)) -> dict[str, bool]:
    admin_auth.logout(request, response, pg_admin)
    return {"ok": True}


@router.get("/me")
def me(token: str = Depends(admin_auth.require_admin)) -> dict[str, Any]:
    return {"authenticated": True, "csrf_token": admin_auth.csrf_for(token)}


@router.get("/analytics", dependencies=admin)
def analytics(filters: dict[str, Any] = Depends(_filters)) -> dict[str, Any]:
    result = aggregate(storage.all_experiments(filters))
    result.pop("daily_model")
    result["sessions"] = len(result["sessions"])
    return result


@router.get("/experiments", dependencies=admin)
def experiments(filters: dict[str, Any] = Depends(_filters), page: int = Query(1, ge=1),
                page_size: int = Query(25, ge=1, le=200)) -> dict[str, Any]:
    items, total = storage.search_experiments(filters, page_size, (page - 1) * page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/experiments/{experiment_id}", dependencies=admin)
def experiment(experiment_id: int, request: Request) -> dict[str, Any]:
    found = storage.get_experiment(experiment_id)
    if found is None:
        raise HTTPException(404, "Experiment not found.")
    storage.audit("admin.view_experiment", client_ip(request), {"experiment_id": experiment_id})
    return found


@router.get("/models", dependencies=admin)
def models() -> dict[str, Any]:
    catalog = get_catalog()
    return {
        "catalog": [{"provider": m.provider, "model": m.id, "label": m.label,
                     "configured": catalog.providers[m.provider].configured,
                     "pricing": m.pricing.snapshot() if m.pricing else None}
                    for m in catalog.models.values()],
        "seen": storage.distinct_values(),
    }


@router.get("/audit-logs", dependencies=admin)
def audit_logs(page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
    items, total = storage.audit_logs(page_size, (page - 1) * page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/export/excel", dependencies=admin)
def export_excel(request: Request, filters: dict[str, Any] = Depends(_filters)) -> Response:
    rows = storage.all_experiments(filters)
    shown = {k: v for k, v in request.query_params.items() if k in {"from", "to", "provider", "model", "status",
                                                                     "session_id", "q"}}
    note = "filters: " + (", ".join(f"{k}={v}" for k, v in shown.items()) if shown else "none (all records)")
    data = build_workbook(rows, note)
    storage.audit("admin.export_excel", client_ip(request), {"filters": shown, "rows": len(rows)})
    return Response(data, media_type=XLSX,
                    headers={"Content-Disposition": f'attachment; filename="{FILENAME}"'})


@router.get("/backup", dependencies=admin)
def backup(request: Request, tasks: BackgroundTasks) -> FileResponse:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = Path(tempfile.mkdtemp()) / f"playground-backup-{stamp}.db"
    storage.backup_to(dest)
    tasks.add_task(lambda: (dest.unlink(missing_ok=True), dest.parent.rmdir()))
    storage.audit("admin.download_backup", client_ip(request))
    return FileResponse(dest, media_type="application/octet-stream", filename=dest.name)
