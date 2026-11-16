"""AI Model Playground — one service: tester API, admin API and the built frontend."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import storage
from app.admin_routes import router as admin_router
from app.auth import client_id, rate_limit, require_auth, spend_guard
from app.catalog import Catalog, calculate_cost, get_catalog
from app.evaluator import evaluate
from app.providers import ProviderError, filter_params, get_provider
from app.schemas import ChatRequest, EvaluateRequest, FeedbackRequest
from app.settings import settings

logging.basicConfig(level=settings.log_level,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("playground")


@asynccontextmanager
async def lifespan(_: FastAPI):
    storage.init()
    if settings.production and not settings.admin_password_hash:
        log.warning("ADMIN_PASSWORD_HASH is unset: the admin dashboard cannot be used.")
    yield
    storage.close()


app = FastAPI(title="AI Model Playground", version="0.2.0", lifespan=lifespan,
              docs_url=None if settings.production else "/docs", redoc_url=None,
              openapi_url=None if settings.production else "/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Playground-Client"],
)

_CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'")


@app.middleware("http")
async def guard(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > settings.max_body_bytes:
        return JSONResponse({"detail": "Request body too large."}, status_code=413)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Content-Security-Policy", _CSP)
    if settings.production:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error: %s", type(exc).__name__)
    return JSONResponse({"detail": "Internal error."}, status_code=500)


PREFIX = "/api/playground"
protected = [Depends(require_auth)]
generation = [Depends(require_auth), Depends(rate_limit), Depends(spend_guard)]


def _estimate_tokens(text: str | None) -> int:
    return round(len(text or "") / 4)


def build_messages(req: ChatRequest) -> list[dict[str, str]]:
    """The exact message list sent to the provider."""
    system_parts = [req.system_prompt.strip()] if req.system_prompt and req.system_prompt.strip() else []
    if req.context and req.context.strip():
        system_parts.append(f"CONTEXT:\n{req.context.strip()}")
    messages: list[dict[str, str]] = []
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    messages.extend({"role": t.role, "content": t.content} for t in req.history)
    messages.append({"role": "user", "content": req.message})
    return messages


def _resolve(catalog: Catalog, provider_id: str, model_id: str):
    model = catalog.model(provider_id, model_id)
    if model is None:
        raise HTTPException(400, f"Unknown model {model_id!r} for provider {provider_id!r}.")
    return catalog.providers[provider_id], model


def _owned(experiment_id: int, client: str) -> dict[str, Any]:
    """A tester may only read or change their own experiments."""
    found = storage.get_experiment(experiment_id)
    if found is None or found.get("client_id") != client:
        raise HTTPException(404, "Experiment not found.")
    return found


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, Any]:
    storage.usage_since(time.time())  # proves the database is reachable
    return {"status": "ok"}


@app.get(f"{PREFIX}/models", dependencies=protected)
def models() -> dict[str, Any]:
    catalog = get_catalog()
    return {
        "providers": [
            {"id": p.id, "name": p.name, "configured": p.configured, "kind": p.kind,
             "supported_params": list(p.supported_params)}
            for p in catalog.providers.values()
        ],
        "models": [
            {"id": m.id, "provider": m.provider, "label": m.label,
             "pricing": None if m.pricing is None else {
                 "input_per_1m": m.pricing.input_per_1m,
                 "output_per_1m": m.pricing.output_per_1m,
                 "currency": m.pricing.currency}}
            for m in catalog.models.values()
        ],
        "evaluator": None if catalog.evaluator is None else
        {"provider": catalog.evaluator[0], "model": catalog.evaluator[1]},
        "limits": {"rate_limit_per_minute": settings.rate_limit_per_minute,
                   "max_prompt_chars": settings.max_prompt_chars},
    }


@app.post(f"{PREFIX}/chat", dependencies=generation)
def chat(req: ChatRequest, client: str = Depends(client_id)) -> dict[str, Any]:
    """Run one experiment. Response, tokens, cost and latency share one record."""
    if req.request_id:
        existing = storage.find_by_request_id(req.request_id)
        if existing is not None:
            if existing.get("client_id") != client:
                raise HTTPException(409, "request_id already used.")
            return existing

    request_start = time.perf_counter()
    catalog = get_catalog()
    provider, model = _resolve(catalog, req.provider, req.model)
    if not provider.configured:
        raise HTTPException(400, f"{provider.name} is not configured on this playground.")

    aicore = provider.kind == "aicore"
    if aicore:
        # The chatbot builds its own prompt and keeps its own memory: only the
        # message (and birth details from Context) are sent, so only those are recorded.
        req = req.model_copy(update={"system_prompt": None, "history": []})
    messages = build_messages(req)
    requested = req.params.model_dump(exclude_none=True)
    sent = filter_params(provider, requested)
    session_id = req.session_id or f"s-{uuid.uuid4().hex[:10]}"
    effective_request = {
        "provider": provider.id,
        "base_url": provider.base_url or provider.kind,
        "model": model.id,
        "params_sent": {**sent, **model.request_overrides},
        "params_ignored": sorted(set(requested) - set(sent)),
        "messages": [m for m in messages if m["role"] != "system"] if aicore else messages,
    }
    estimate = {
        "system_prompt": _estimate_tokens(req.system_prompt),
        "context": 0 if aicore else _estimate_tokens(req.context),
        "history": sum(_estimate_tokens(t.content) for t in req.history),
        "user_message": _estimate_tokens(req.message),
    }
    base = {
        "request_id": req.request_id or uuid.uuid4().hex,
        "session_id": session_id, "client_id": client,
        "tester_name": (req.tester_name or "").strip() or None,
        "provider": provider.id, "model": model.id,
        "system_prompt": req.system_prompt, "user_prompt": req.message, "context": req.context,
        "params": requested, "effective_request": effective_request,
        "pricing_snapshot": model.pricing.snapshot() if model.pricing else None,
    }

    try:
        result = get_provider(provider).generate(model, messages, requested,
                                                 session_id=session_id, context=req.context)
    except ProviderError as exc:
        latency_ms = int((time.perf_counter() - request_start) * 1000)
        log.warning("generation failed provider=%s model=%s reason=%s", provider.id, model.id, exc.reason)
        return storage.create_experiment({
            **base, "status": "failed", "error": exc.reason, "latency_ms": latency_ms,
            "usage_source": "unavailable",
            "metadata": {"provider_name": provider.name, "status_code": exc.status_code,
                         "prompt_breakdown_estimate": estimate},
        })

    cost = calculate_cost(model.pricing, result.input_tokens, result.output_tokens)
    cost_source = "playground pricing" if cost["total_cost"] is not None else None
    if cost["total_cost"] is None and result.provider_cost is not None and provider.kind == "aicore":
        # ai-core prices its own turn (classifier, RAG, generation, judge); use that figure.
        cost = {**cost, "total_cost": result.provider_cost, "currency": "USD"}
        cost_source = "ai-core reported (whole turn)"
    degraded = bool(result.extra.get("degraded"))
    if not result.text:
        error = "The model returned an empty response."
    elif degraded:
        error = "ai-core could not reach its model and returned its fallback reply."
    else:
        error = None
    latency_ms = int((time.perf_counter() - request_start) * 1000)
    experiment = storage.create_experiment({
        **base,
        "status": "failed" if error else "success",
        "error": error,
        "response": result.text,
        "model_version": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "usage_source": result.usage_source,
        "input_cost": cost["input_cost"], "output_cost": cost["output_cost"],
        "total_cost": cost["total_cost"], "currency": cost["currency"],
        "latency_ms": latency_ms,
        "provider_latency_ms": result.provider_latency_ms,
        "metadata": {
            "provider_name": provider.name,
            "model_reported": result.model,
            "finish_reason": result.finish_reason,
            "reasoning_tokens": result.reasoning_tokens,
            "cached_tokens": result.cached_tokens,
            "provider_reported_cost": result.provider_cost,
            "cost_source": cost_source,
            "tokens_reported_by_provider": result.input_tokens is not None,
            "prompt_breakdown_estimate": estimate,
            "overhead_ms": latency_ms - result.provider_latency_ms,
            **({"ai_core": result.extra} if result.extra else {}),
        },
    })
    log.info("experiment=%s provider=%s model=%s tokens=%s latency_ms=%s",
             experiment["id"], provider.id, model.id, result.total_tokens, latency_ms)
    return experiment


@app.post(f"{PREFIX}/evaluate", dependencies=generation)
def evaluate_experiment(req: EvaluateRequest, client: str = Depends(client_id)) -> dict[str, Any]:
    experiment = _owned(req.experiment_id, client)
    if experiment["status"] != "success" or not experiment.get("response"):
        raise HTTPException(400, "Only a successful response can be evaluated.")
    try:
        verdict = evaluate(experiment, get_catalog(), req.provider, req.model)
    except ProviderError as exc:
        raise HTTPException(502, f"Evaluation failed ({exc.provider}): {exc.reason}") from exc
    return storage.update_experiment(req.experiment_id, {
        "evaluation": verdict, "evaluation_overall": verdict["overall"],
    })  # type: ignore[return-value]


@app.post(f"{PREFIX}/feedback", dependencies=protected)
def feedback(req: FeedbackRequest, client: str = Depends(client_id)) -> dict[str, Any]:
    _owned(req.experiment_id, client)
    return storage.update_experiment(req.experiment_id, {
        "human_rating": req.rating,
        "human_feedback": (req.feedback or "").strip() or None,
        "edited_response": (req.edited_response or "").strip() or None,
        "feedback_at": time.time(),
    })  # type: ignore[return-value]


@app.get(f"{PREFIX}/experiments", dependencies=protected)
def experiments(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                client: str = Depends(client_id)) -> list[dict[str, Any]]:
    return storage.list_experiments(limit, offset, client_id=client)


@app.get(f"{PREFIX}/experiments/{{experiment_id}}", dependencies=protected)
def experiment(experiment_id: int, client: str = Depends(client_id)) -> dict[str, Any]:
    return _owned(experiment_id, client)


app.include_router(admin_router)


# ── the built frontend (same origin, one service) ────────────────────
def _mount_frontend(static_dir: Path) -> None:
    index = static_dir / "index.html"
    if not index.is_file():
        log.info("no built frontend at %s; serving the API only", static_dir)
        return
    if (static_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "Not found.")
        candidate = (static_dir / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(static_dir):
            return FileResponse(candidate)
        # Client-side routes (/, /admin, ...) all get the app shell.
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


_mount_frontend(settings.static_dir)
