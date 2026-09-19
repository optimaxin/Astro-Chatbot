"""Knowledge retrieval node.

RAG is treated as optional infrastructure. If Pinecone or the embedding
service is unavailable the node records why and returns an empty result;
`build_context` then tells the model to answer from general principles and to
be explicit about uncertainty. The turn continues either way — losing
retrieval degrades an answer, it should not fail one.
"""

from __future__ import annotations

from typing import Any

from app.logging_setup import get_logger
from app.persona.registry import get_persona
from app.rag.retriever import retrieve
from app.state.chatbot_state import ChatbotState

log = get_logger("nodes.rag")

# Greetings and thanks carry nothing to retrieve against; searching for them
# only wastes an embedding call and pollutes the prompt.
SKIP_INTENTS = {"smalltalk"}


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
    result = retrieve(
        query=query,
        intent=intent,
        top_k=persona.top_k,
        extra_domains=persona.domains or None,
    )

    errors = list(state.get("errors", []))
    if not result.available and result.error:
        errors.append(f"rag unavailable: {result.error}")

    return {"retrieval": result, "errors": errors}


__all__ = ["retrieve_knowledge_node"]
