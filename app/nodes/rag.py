"""Knowledge retrieval node.

RAG is treated as optional infrastructure. If Pinecone or the embedding
service is unavailable the node records why and returns an empty result;
`build_context` then tells the model to answer from general principles and to
be explicit about uncertainty. The turn continues either way — losing
retrieval degrades an answer, it should not fail one.
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from app.config.settings import settings

from app.logging_setup import get_logger
from app.persona.registry import get_persona
from app.rag.retriever import retrieve
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.rag")

# Greetings and thanks carry nothing to retrieve against; searching for them
# only wastes an embedding call and pollutes the prompt.
SKIP_INTENTS = {"smalltalk"}

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="aicore-rag")


def retrieve_knowledge_node(state: ChatbotState) -> dict[str, Any]:
    intent = state.get("intent", "general_astrology")

    if intent in SKIP_INTENTS:
        from app.rag.retriever import RetrievalResult

        log.info("retrieval skipped", extra={"intent": intent, "reason": "intent needs no knowledge"})
        return {"retrieval": RetrievalResult(chunks=[], used_filter=False, widened=False, available=True)}

    # Search with the question plus a little recent context, so a short
    # follow-up like "and next year?" still retrieves something useful.
    history = state.get("history") or []
    recent_user = [m.content for m in history if m.role == "user"][-1:]
    query = " ".join([*recent_user, state["question"]]).strip()

    persona = get_persona(state.get("persona_id"))
    kwargs = dict(query=query, intent=intent, top_k=persona.top_k,
                  extra_domains=persona.domains or None)
    timeout = settings.pinecone.timeout_seconds
    if timeout > 0:
        # Bounded: a single Pinecone embed call was measured hanging for 24.5 s.
        # On timeout the turn takes the existing "retrieval unavailable" path
        # (answer from the chart, user-visible notice) instead of stalling.
        ctx = contextvars.copy_context()
        future = _pool.submit(ctx.run, retrieve, **kwargs)
        try:
            result = future.result(timeout=timeout)
        except FutureTimeout:
            from app.rag.retriever import RetrievalResult

            log.warning("retrieval timed out", extra={"timeout_s": timeout})
            result = RetrievalResult.unavailable(f"timeout after {timeout:.0f}s")
    else:
        result = retrieve(**kwargs)

    errors = list(state.get("errors", []))
    if not result.available and result.error:
        errors.append(f"rag unavailable: {result.error}")

    return {"retrieval": result, "errors": errors}


__all__ = ["retrieve_knowledge_node"]
