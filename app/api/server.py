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

import hashlib
import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.schemas import (
    BirthDetails,
    ChartResponse,
    Citation,
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PersonaOut,
    ResponseStyleOut,
    Source,
    Usage,
)
from app.api.auth import (
    AuthRequired,
    AuthUnavailable,
    Forbidden,
    Identity,
    check_session_secret,
    parse_anon_token,
    resolve_identity,
    scoped_session,
    secret_fingerprint,
    verify_bearer,
)
from app.config.settings import settings
from app.graph.chatbot_graph import get_graph, run_turn
from app.kundli.kundli_service import KundliContext, get_kundli, kundli_from_birth
from app.llm.llm_factory import active_provider, describe_roles, provider_configured
from app.llm.reliability import breaker
from app.logging_setup import get_logger, setup_logging
from app.observability import trace
from app.memory.redis_memory import memory
from app.evaluation import evaluation_store
from app.persona.registry import get_persona, is_known_persona, list_personas
from app.prompts.reply_format import reply_messages
from app.prompts.response_style import AUTO_STYLE_ID, is_known_style, list_styles
from app.rag.pinecone_client import pinecone_client

log = get_logger("api")

# Messages a client is allowed to see. Anything else is logged in full and
# reported generically, so an internal error can never leak a path or a key.
GENERIC_ERROR = "The assistant is temporarily unavailable. Please try again."


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    check_session_secret()
    # Compile the graph at startup rather than inside the first request, so
    # the first user does not pay for it and a broken graph fails loudly here.
    get_graph()
    log.info(
        "api starting",
        extra={
            "llm_provider": active_provider(),
            "llm_configured": provider_configured(),
            "openrouter_configured": settings.openrouter.configured,
            "nvidia_configured": settings.nvidia.configured,
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


def _chart_cache_key(birth: BirthDetails) -> str:
    """A stable id for one set of birth details.

    Derived rather than random so the same visitor asking a second question
    reuses the chart already computed for them, and so two people with
    identical birth details share one cached chart instead of two. Only the
    five fields that actually determine the chart go into the hash — a
    different spelling of the place name must not produce a second entry.
    """
    canonical = json.dumps(
        {
            "date": birth.date,
            "time": birth.time,
            "tz": birth.timezoneOffsetMinutes,
            # Rounded to ~11m of precision: far finer than birth-time accuracy
            # can justify, while stopping float noise from splitting the cache.
            "lat": round(birth.latitude, 4),
            "lon": round(birth.longitude, 4),
        },
        sort_keys=True,
    )
    return "birth-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _resolve_chart(birth: BirthDetails) -> tuple[KundliContext, str]:
    """Chart for a visitor who just filled in the birth form. Never raises.

    Returns the chart and the cache key, which doubles as the `user_id` for
    the turn so usage and history attribute to this chart rather than to a
    shared anonymous bucket.
    """
    cache_key = _chart_cache_key(birth)

    cached = memory.get_chart_facts(cache_key)
    if cached:
        context = KundliContext.from_facts(cache_key, cached)
        if context is not None:
            return context, cache_key

    context = kundli_from_birth(birth.model_dump(exclude_none=True), cache_key)
    if context.available:
        memory.save_chart_facts(cache_key, context.to_facts())
    return context, cache_key


def _to_response(
    state: dict[str, Any], session_id: str, persona_id: str, style_id: str
) -> ChatResponse:
    metrics: dict[str, Any] = state.get("metrics") or {}
    answer = state.get("response", "")
    citations = metrics.get("citations") or []
    # The reply's bubbles: the model's own topic paragraphs (REPLY_SHAPE).
    messages = reply_messages(answer)
    # Visible attribution of knowledge-base material goes at the end of the
    # LAST bubble, never as a bubble of its own. Added here rather than in the
    # graph so it never enters chat history.
    if citations and settings.flags.citation_display == "footer" and not metrics.get("generation_failed"):
        from app.rag.grounding import citation_footer

        footer = citation_footer(citations, style_id)
        answer = f"{answer}\n\n{footer}"
        messages = [*messages[:-1], f"{messages[-1]}\n\n{footer}"] if messages else [footer]
    return ChatResponse(
        answer=answer,
        messages=messages,
        persona=metrics.get("persona", persona_id),
        response_style=style_id,
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
            provider=metrics.get("provider"),
            fallback_used=bool(metrics.get("fallback_used")),
            token_source=metrics.get("token_source"),
            estimated_total_tokens=metrics.get("estimated_total_tokens"),
        ),
        latency_ms=metrics.get("turn_latency_ms", 0),
        citations=[Citation(**c) for c in metrics.get("citations") or []],
        degraded=bool(metrics.get("generation_failed")),
        notices=_notices(metrics.get("errors") or []),
    )


# ── endpoints ─────────────────────────────────────────────────────────
@app.exception_handler(AuthRequired)
async def _auth_required(_: Request, exc: AuthRequired) -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": "auth_required", "detail": exc.detail})


@app.exception_handler(Forbidden)
async def _forbidden(_: Request, exc: Forbidden) -> JSONResponse:
    return JSONResponse(status_code=403, content={"error": "forbidden", "detail": exc.detail})


@app.exception_handler(AuthUnavailable)
async def _auth_unavailable(_: Request, __: AuthUnavailable) -> JSONResponse:
    # Fails closed: an identity we cannot check is not trusted.
    return JSONResponse(status_code=503, content={"error": "auth_unavailable", "detail": GENERIC_ERROR})


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
def chat(payload: ChatRequest,
         authorization: str | None = Header(default=None),
         x_chat_session_token: str | None = Header(default=None)) -> ChatResponse:
    """One conversational turn, traced end to end (see app.observability)."""
    with trace.request("pending"):
        with trace.span("api.auth"):
            identity = resolve_identity(
                authorization=authorization,
                session_token=x_chat_session_token or payload.session_token,
                body_user_id=payload.user_id, has_birth=payload.birth is not None)
        return _chat(payload, identity)


# Graph nodes -> the progress stage a client may show while it waits.
_PROGRESS_STAGES = {"load_context": "reading_chart", "classify_intent": "understanding",
                    "retrieve_knowledge": "consulting_knowledge", "generate_response": "writing",
                    "validate_reply": "checking"}
# Characters per `delta` event when the validated reply is streamed.
_STREAM_CHUNK_CHARS = 48


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunks(text: str, size: int = _STREAM_CHUNK_CHARS) -> list[str]:
    """Split on whitespace into ~`size`-char pieces that concatenate back to
    exactly `text` (newlines and paragraph breaks included)."""
    import re as _re

    pieces, current = [], ""
    for token in _re.findall(r"\S+\s*|\s+", text):
        if current and len(current) + len(token) > size:
            pieces.append(current)
            current = ""
        current += token
    if current:
        pieces.append(current)
    return pieces


@app.post("/chat/stream", dependencies=[Depends(rate_limit)])
def chat_stream(payload: ChatRequest,
                authorization: str | None = Header(default=None),
                x_chat_session_token: str | None = Header(default=None)):
    """The same turn as POST /chat, delivered as Server-Sent Events.

    One assistant TURN (`message_id`) made of 1–4 semantic SEGMENTS (bubbles):

        start          {message_id}
        status         {message_id, stage}             0..n, while the graph runs
        segment_start  {message_id, segment_id, index}
        delta          {message_id, segment_id, text}  append to that segment
        segment_end    {message_id, segment_id, index, text}   the segment's full text
        done           {message_id, segments: [{segment_id, index, text}], ...ChatResponse}
        error          {message_id, detail}            instead of done, on failure

    Segments are the model's own topic paragraphs (app/prompts/segments.py);
    `done.messages` holds the same texts in order and is authoritative.

    Text is streamed only AFTER chart/language validation: a draft that the
    validator would have corrected must never reach the screen, so token-level
    streaming of the raw draft is deliberately not used. The turn is persisted
    exactly once (by the graph's save_metrics node), whatever the client does.
    """
    import queue
    import threading
    import uuid

    from fastapi.responses import StreamingResponse

    # Auth errors surface as normal HTTP errors, before the stream opens.
    with trace.span("api.auth"):
        identity = resolve_identity(
            authorization=authorization,
            session_token=x_chat_session_token or payload.session_token,
            body_user_id=payload.user_id, has_birth=payload.birth is not None)

    message_id = f"msg_{uuid.uuid4().hex[:16]}"
    events: "queue.Queue[tuple[str, dict[str, Any]] | None]" = queue.Queue()

    def progress(node: str) -> None:
        stage = _PROGRESS_STAGES.get(node)
        if stage:
            events.put(("status", {"message_id": message_id, "stage": stage}))

    def work() -> None:
        try:
            with trace.request("pending"):
                response = _chat(payload, identity, on_progress=progress)
            segments = []
            for index, text in enumerate(response.messages or ([response.answer] if response.answer else [])):
                segment_id = f"{message_id}-s{index}"
                events.put(("segment_start", {"message_id": message_id, "segment_id": segment_id,
                                              "index": index}))
                for piece in _chunks(text):
                    events.put(("delta", {"message_id": message_id, "segment_id": segment_id, "text": piece}))
                events.put(("segment_end", {"message_id": message_id, "segment_id": segment_id,
                                            "index": index, "text": text}))
                segments.append({"segment_id": segment_id, "index": index, "text": text})
            events.put(("done", {"message_id": message_id, "segments": segments, **response.model_dump()}))
        except _RateLimited:
            events.put(("error", {"message_id": message_id, "error": "rate_limited",
                                  "detail": "Too many requests. Please slow down."}))
        except Exception:  # noqa: BLE001 — never leak internals into the stream
            log.exception("stream turn failed")
            events.put(("error", {"message_id": message_id, "error": "internal_error",
                                  "detail": GENERIC_ERROR}))
        finally:
            events.put(None)

    threading.Thread(target=work, name="chat-stream", daemon=True).start()

    def stream():
        yield _sse("start", {"message_id": message_id})
        while True:
            item = events.get()
            if item is None:
                return
            yield _sse(*item)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _chat(payload: ChatRequest, identity: Identity, on_progress: Any = None) -> ChatResponse:
    """One conversational turn.

    The client never sends a chart. It sends either

    * `user_id` alone — the chart is looked up server-side and cached, or
    * `birth` — five validated scalars from the birth form, from which the
      Tredev calculators recompute the chart here.

    The second path is what lets a logged-out visitor on the Kundli page chat
    about the chart they just generated. Either way the chart the model sees
    was produced by our own engine, never by the browser.
    """
    session_id = payload.resolved_session_id()
    persona_id = payload.persona_id or settings.api.default_persona_id
    if payload.persona_id and not is_known_persona(payload.persona_id):
        log.info("unknown persona requested, falling back", extra={"requested": payload.persona_id})
    persona_id = get_persona(persona_id).id

    if (payload.response_style and payload.response_style.lower() != AUTO_STYLE_ID
            and not is_known_style(payload.response_style)):
        log.info("unknown response style requested, falling back",
                 extra={"requested": payload.response_style})
    # The reply language is decided inside the graph (load_context) by the
    # central policy, which can also see the conversation's saved preference.
    requested_style = payload.response_style if is_known_style(payload.response_style) else AUTO_STYLE_ID

    # Limits key on the server-derived identity, never on the body's user_id
    # (which a client could rotate to dodge the limit).
    allowed, remaining = memory.check_rate_limit(
        identity.key, settings.api.rate_limit_per_minute, 60
    )
    if settings.api.rate_limit_enabled and not allowed:
        log.warning("rate limited", extra={"scope": "user", "identity": identity.kind})
        raise _too_many_requests()

    kundli: KundliContext | None = None
    turn_user_id = identity.user_id
    if payload.birth is not None:
        with trace.span("api.resolve_chart"):
            kundli, chart_key = _resolve_chart(payload.birth)
        if identity.kind == "legacy":
            # Previous behaviour: the chart's cache key is the turn's user id.
            turn_user_id = chart_key

    # History lives under the caller's identity: a guessed or reused session
    # id (the widget's ids are derived from birth details) can only ever reach
    # the caller's own conversation.
    started = time.perf_counter()
    state = run_turn(
        payload.question,
        scoped_session(identity, session_id),
        turn_user_id,
        persona_id,
        kundli=kundli,
        chat_bubbles=payload.chat_bubbles,
        response_style=requested_style,
        on_progress=on_progress,
    )
    style_id = state.get("response_style") or requested_style
    response = _to_response(dict(state), session_id, persona_id, style_id)
    response.session_token = identity.new_session_token
    if response.request_id and identity.kind != "legacy":
        memory.set_request_owner(response.request_id, identity.key)

    log.info(
        "chat handled",
        extra={
            "session_id": session_id,
            "identity": identity.kind,
            "persona": persona_id,
            "response_style": style_id,
            "language_source": (state.get("language") or {}).get("source"),
            "intent": response.intent,
            "chart_from_birth": payload.birth is not None,
            "chart_available": kundli.available if kundli else None,
            "sources": len(response.sources),
            "bubbles": len(response.messages),
            "degraded": response.degraded,
            "rate_limit_remaining": remaining,
            "api_latency_ms": int((time.perf_counter() - started) * 1000),
        },
    )
    return response


@app.get("/chat/history")
def chat_history(session_id: str, limit: int = 20,
                 authorization: str | None = Header(default=None),
                 x_chat_session_token: str | None = Header(default=None)) -> dict[str, Any]:
    """The caller's own conversation, for restoring the chat after a reload.

    Each assistant turn comes back as the SAME bubbles it was shown as: the
    stored reply text is re-segmented with the same pure function the live
    path uses, so reload never merges or duplicates bubbles. Only the caller's
    own history is reachable (identity-scoped key). Without an identity there
    is simply no history — this endpoint never issues a session token.
    """
    identity: Identity | None = None
    token = _bearer_token(authorization)
    if token and settings.auth.mode != "off":
        user_id = verify_bearer(token)
        if user_id:
            identity = Identity("user", user_id, f"user:{user_id}")
    if identity is None:
        anon_id = parse_anon_token(x_chat_session_token)
        if anon_id and settings.auth.mode != "required":
            identity = Identity("anon", f"anon:{anon_id}", f"anon:{anon_id}")
    if identity is None or not session_id or len(session_id) > 200:
        return {"session_id": session_id, "turns": []}
    limit = max(1, min(limit, 40))
    turns: list[dict[str, Any]] = []
    for message in memory.get_history(scoped_session(identity, session_id), limit=limit * 2):
        if message.role == "user":
            turns.append({"role": "user", "text": message.content})
        else:
            turns.append({"role": "assistant", "messages": reply_messages(message.content)})
    return {"session_id": session_id, "turns": turns}


@app.get("/personas", response_model=list[PersonaOut])
def personas() -> list[PersonaOut]:
    """The personas a client may offer. System prompts are never returned."""
    return [PersonaOut(**p.to_dict()) for p in list_personas()]


@app.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(rate_limit)])
def feedback(payload: FeedbackRequest,
             authorization: str | None = Header(default=None),
             x_chat_session_token: str | None = Header(default=None)) -> FeedbackResponse:
    """Record what a user thought of one reply.

    This is the signal that matters most — the automated reward score is only
    an estimate of it — so the two are stored separately and never merged.

    A user changing their mind updates their existing verdict rather than
    adding a second one, so the helpful-rate cannot be skewed by double
    clicks. A storage failure is reported as `persisted: false` but still
    returns 200: the button should not appear broken because analytics is
    down, and the vote is not worth failing a request over.
    """
    user_id = payload.user_id
    if settings.auth.mode != "off":
        identity = _feedback_identity(authorization, x_chat_session_token)
        owner = memory.get_request_owner(payload.request_id)
        # Only the identity that received an answer may rate it, so analytics
        # cannot be skewed by voting on other people's replies.
        if owner and owner != identity.key:
            raise Forbidden("You can only rate answers from your own conversation.")
        user_id = identity.user_id
    persisted = evaluation_store.write_feedback(
        request_id=payload.request_id,
        user_id=user_id,
        feedback_type=payload.feedback_type,
        conversation_id=payload.conversation_id,
        rating=payload.rating,
        feedback_reason=payload.feedback_reason,
        feedback_comment=payload.feedback_comment,
    )
    log.info(
        "feedback recorded",
        extra={
            "request_id": payload.request_id,
            "feedback_type": payload.feedback_type,
            "has_reason": bool(payload.feedback_reason),
            "has_rating": payload.rating is not None,
            "persisted": persisted,
        },
    )
    return FeedbackResponse(
        recorded=True,
        request_id=payload.request_id,
        feedback_type=payload.feedback_type,
        persisted=persisted,
    )


@app.get("/response-styles", response_model=list[ResponseStyleOut])
def response_styles() -> list[ResponseStyleOut]:
    """The reply styles a client may offer. Prompt text is never returned."""
    return [ResponseStyleOut(**s.to_dict()) for s in list_styles()]


def _bearer_token(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None


def _feedback_identity(authorization: str | None, session_token: str | None) -> Identity:
    token = _bearer_token(authorization)
    if token:
        user_id = verify_bearer(token)
        if user_id:
            return Identity("user", user_id, f"user:{user_id}")
    anon_id = parse_anon_token(session_token)
    if anon_id and settings.auth.mode != "required":
        return Identity("anon", f"anon:{anon_id}", f"anon:{anon_id}")
    raise AuthRequired("Please sign in to leave feedback.")


@app.get("/chart/{user_id}", response_model=ChartResponse)
def chart(user_id: str, authorization: str | None = Header(default=None)) -> ChartResponse:
    """The compact chart facts for a user, for a frontend that wants to render
    placements alongside the conversation. Same cache the chat path uses.

    A stored chart is personal data: only its signed-in owner may read it."""
    if settings.auth.mode != "off":
        token = _bearer_token(authorization)
        caller = verify_bearer(token) if token else None
        if not caller:
            raise AuthRequired("Please sign in to view your chart.")
        if caller != user_id:
            raise Forbidden("You can only view your own chart.")
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
    provider = active_provider()
    llm_configured = provider_configured(provider)
    roles = describe_roles()
    dependencies: dict[str, Any] = {
        "llm": {
            "configured": llm_configured,
            "provider": provider,
            "models": {role: cfg["model"] for role, cfg in roles.items()},
            "fallbacks": {role: cfg["fallback"] for role, cfg in roles.items()},
            # Open circuits mean a model is currently being skipped.
            "circuits": breaker.snapshot(settings.reliability.circuit_cooldown_seconds),
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
        # Same fingerprint on every worker = they share one signing key.
        "auth": {"mode": settings.auth.mode, "session_secret": secret_fingerprint()},
    }
    status = "ok" if llm_configured else "degraded"
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
