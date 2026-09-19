"""FastAPI surface over the existing LangGraph pipeline.

    uvicorn app.api.server:app --reload --port 8000
    python -m app.api.server              # same thing, reads API_HOST/API_PORT

This module is deliberately thin. It validates input, applies a rate limit,
calls `run_turn`, and reshapes the metrics the graph already produces. It
contains no astrology, no prompt text and no model ids — adding an endpoint
must never become a second place where business logic lives.

Two properties the handlers rely on:

* `run_turn` never raises. A dependency outage comes back as a degraded state
  with `errors` on the metrics, not as an exception, so a handler that leaked
  a stack trace to a browser would have to work at it.
* The terminal (`app.main`) and this API call exactly the same function, so
  they cannot drift apart in behaviour.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.schemas import (
    ChartResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    PersonaOut,
    Source,
    Usage,
)
from app.config.settings import settings
from app.graph.chatbot_graph import get_graph, run_turn
from app.kundli.kundli_service import KundliContext, get_kundli
from app.llm.llm_factory import describe_roles
from app.logging_setup import get_logger, setup_logging
from app.memory.redis_memory import memory
from app.persona.registry import get_persona, is_known_persona, list_personas
from app.rag.pinecone_client import pinecone_client

log = get_logger("api")

# Messages a client is allowed to see. Anything else is logged in full and
# reported generically, so an internal error can never leak a path or a key.
GENERIC_ERROR = "The assistant is temporarily unavailable. Please try again."


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    # Compile the graph at startup rather than inside the first request, so
    # the first user does not pay for it and a broken graph fails loudly here.
    get_graph()
    log.info(
        "api starting",
        extra={
            "openrouter_configured": settings.openrouter.configured,
            "pinecone_configured": settings.pinecone.configured,
            "redis_configured": settings.redis.configured,
            "kundli_source": settings.kundli.source,
        },
    )
    yield
    log.info("api stopping")


app = FastAPI(
    title="Tredev Astro — AI Core",
    description="Vedic astrology chatbot: kundli facts + Pinecone RAG + Redis memory + LLM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.api.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── rate limiting ─────────────────────────────────────────────────────
def rate_limit(request: Request) -> None:
    """Per-user fixed window, enforced in Redis so it holds across processes.

    Fails open when Redis is down (see `RedisMemory.check_rate_limit`). It
    guards spend, not security — the LLM key is never reachable from a client.
    """
    if not settings.api.rate_limit_enabled:
        return
    # Body is not parsed yet at dependency time, so key on the peer address.
    # Per-user limiting happens inside the handler where user_id is known.
    client = request.client.host if request.client else "unknown"
    allowed, _ = memory.check_rate_limit(f"ip:{client}", settings.api.rate_limit_per_minute * 5, 60)
    if not allowed:
        log.warning("rate limited", extra={"scope": "ip"})
        raise _too_many_requests()


class _RateLimited(Exception):
    pass


def _too_many_requests() -> _RateLimited:
    return _RateLimited()


@app.exception_handler(_RateLimited)
async def _rate_limited_handler(_: Request, __: _RateLimited) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limited", "detail": "Too many requests. Please slow down."},
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence. Logs everything, returns nothing revealing."""
    log.exception("unhandled api error", extra={"path": request.url.path})
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": GENERIC_ERROR})


# ── response assembly ─────────────────────────────────────────────────
# Errors recorded on the state are developer-facing (they carry exception type
# names). These map them to something a user may read without exposing
# internals. Anything unmatched becomes a single generic notice.
_NOTICE_MAP = (
    ("rag unavailable", "Knowledge retrieval is unavailable, so this answer draws on the chart alone."),
    ("kundli unavailable", "Your birth chart could not be loaded, so no placements are stated."),
    ("history unavailable", "Conversation memory is unavailable, so earlier messages were not used."),
    ("session state unavailable", "Conversation memory is unavailable, so earlier messages were not used."),
    ("generation failed", "The astrology model could not be reached."),
)


def _notices(errors: list[str]) -> list[str]:
    out: list[str] = []
    for error in errors:
        lowered = error.lower()
        match = next((msg for prefix, msg in _NOTICE_MAP if lowered.startswith(prefix)), None)
        if match and match not in out:
            out.append(match)
    return out


def _to_response(state: dict[str, Any], session_id: str, persona_id: str) -> ChatResponse:
    metrics: dict[str, Any] = state.get("metrics") or {}
    return ChatResponse(
        answer=state.get("response", ""),
        persona=metrics.get("persona", persona_id),
        session_id=session_id,
        request_id=metrics.get("request_id", ""),
        intent=metrics.get("intent", "unknown"),
        sources=[Source(**s) for s in metrics.get("sources", [])],
        chart_facts_used=metrics.get("chart_facts_used", []),
        usage=Usage(
            input_tokens=metrics.get("input_tokens"),
            output_tokens=metrics.get("output_tokens"),
            total_tokens=metrics.get("total_tokens"),
            tokens_available=bool(metrics.get("tokens_available")),
            model=metrics.get("model"),
            estimated_cost_usd=metrics.get("estimated_cost_usd"),
            provider_cost_usd=metrics.get("native_cost_usd"),
            turn_total_tokens=metrics.get("turn_total_tokens"),
            turn_cost_usd=metrics.get("turn_cost_usd"),
            llm_calls=metrics.get("llm_calls", 0),
        ),
        latency_ms=metrics.get("turn_latency_ms", 0),
        degraded=bool(metrics.get("generation_failed")),
        notices=_notices(metrics.get("errors") or []),
    )


# ── endpoints ─────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
def chat(payload: ChatRequest) -> ChatResponse:
    """One conversational turn.

    The client sends who is asking and what they asked — never the chart. The
    chart is resolved server-side from `user_id` and cached in Redis, so a
    browser never handles chart data it did not need to see and cannot
    substitute a chart of its own.
    """
    session_id = payload.resolved_session_id()
    persona_id = payload.persona_id or settings.api.default_persona_id
    if payload.persona_id and not is_known_persona(payload.persona_id):
        log.info("unknown persona requested, falling back", extra={"requested": payload.persona_id})
    persona_id = get_persona(persona_id).id

    allowed, remaining = memory.check_rate_limit(
        f"user:{payload.user_id}", settings.api.rate_limit_per_minute, 60
    )
    if settings.api.rate_limit_enabled and not allowed:
        log.warning("rate limited", extra={"scope": "user", "user_id": payload.user_id})
        raise _too_many_requests()

    started = time.perf_counter()
    state = run_turn(payload.question, session_id, payload.user_id, persona_id)
    response = _to_response(dict(state), session_id, persona_id)

    log.info(
        "chat handled",
        extra={
            "session_id": session_id,
            "user_id": payload.user_id,
            "persona": persona_id,
            "intent": response.intent,
            "sources": len(response.sources),
            "degraded": response.degraded,
            "rate_limit_remaining": remaining,
            "api_latency_ms": int((time.perf_counter() - started) * 1000),
        },
    )
    return response


@app.get("/personas", response_model=list[PersonaOut])
def personas() -> list[PersonaOut]:
    """The personas a client may offer. System prompts are never returned."""
    return [PersonaOut(**p.to_dict()) for p in list_personas()]


@app.get("/chart/{user_id}", response_model=ChartResponse)
def chart(user_id: str) -> ChartResponse:
    """The compact chart facts for a user, for a frontend that wants to render
    placements alongside the conversation. Same cache the chat path uses."""
    cached = memory.get_chart_facts(user_id)
    context = KundliContext.from_facts(user_id, cached) if cached else None
    if context is None:
        context = get_kundli(user_id)
        if context.available:
            memory.save_chart_facts(user_id, context.to_facts())
    if not context.available:
        return ChartResponse(user_id=user_id, available=False, error="chart not available for this user")
    return ChartResponse(user_id=user_id, available=True, facts=context.to_facts())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Reports each dependency separately.

    `configured` and `reachable` are different facts and are kept apart: a
    missing key and an outage need different fixes. Only `llm` is required for
    the service to answer at all; the rest degrade.
    """
    redis_ok, redis_detail = memory.ping() if settings.redis.configured else (False, "not configured")
    pinecone_ok, pinecone_detail = (
        pinecone_client.ping() if settings.pinecone.configured else (False, "not configured")
    )
    dependencies: dict[str, Any] = {
        "llm": {
            "configured": settings.openrouter.configured,
            "provider": "openrouter",
            "models": {role: cfg["model"] for role, cfg in describe_roles().items()},
        },
        "pinecone": {
            "configured": settings.pinecone.configured,
            "reachable": pinecone_ok,
            "detail": pinecone_detail,
            "index": settings.pinecone.index_name,
        },
        "redis": {
            "configured": settings.redis.configured,
            "reachable": redis_ok,
            "detail": redis_detail,
        },
        "kundli": {"source": settings.kundli.source},
    }
    status = "ok" if settings.openrouter.configured else "degraded"
    return HealthResponse(status=status, dependencies=dependencies)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.api.server:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
