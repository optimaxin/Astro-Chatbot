"""Chatbot retrieval over the Tradeva Premium knowledge corpus.

The corpus (506 source files, 8,967 chunks, Pinecone index
`tredev-astro-knowledge`, namespace `tradeva-astro-v1`, BAAI/bge-m3 dense
1024-d) and its retrieval service already exist in the Premium project
(`D:\\Tradeva Premium\\kb_rag`, Stage 5; Stage 6 report generation builds on
it). This module does not re-embed, re-index or copy any of it. It calls that
shared service over HTTP (`POST /api/rag/retrieve`) with a chatbot-specific
retrieval plan, so report and chatbot share one corpus and one retrieval
implementation while keeping independent configuration.

Why HTTP rather than querying Pinecone directly from here: the corpus was
embedded with bge-m3, which ai-core cannot produce — it embeds with Pinecone's
hosted llama-text-embed-v2 (a different vector space), NVIDIA's hosted bge-m3
reached end of life on 2026-08-25 (verified: HTTP 410), Pinecone does not host
bge-m3, and ai-core has no torch runtime. The Premium service owns the
pinned bge-m3 model, so it is the one place that can embed queries correctly.

Chatbot retrieval plan (per turn, at most `max_queries` parallel calls):

* chart-fact queries — the user's REAL placements for the asked topic, sent
  as `structured_entities` (e.g. marriage -> {"house_lord": 7, "house": 6}
  when the chart's 7th lord sits in the 6th; {"planet": "Venus", "house": 7}
  for a 7th-house occupant). The Premium filter builder turns these into exact
  metadata filters, so the evidence is about this chart, not a lookalike.
* a dasha query when the question is about timing or the running period.
* the question itself, semantically, for anything the chart facts miss.

Then: drop case studies (they contain other people's birth data and dates),
remedies/gemstones/transits unless asked for, review-flagged chunks and
duplicates; keep at most `max_chunks`, chart-fact matches first.

Never raises. Any failure returns `None` so the caller falls back to the
legacy namespace — a knowledge outage degrades an answer, it never fails one.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from app.config.settings import settings
from app.logging_setup import get_logger
from app.observability.trace import traced
from app.rag.retriever import RetrievalResult, RetrievedChunk

log = get_logger("rag.premium")

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="aicore-premium-rag")

_ORD = {n: f"{n}{'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'}" for n in range(1, 13)}

# Houses whose LORD placement is looked up per intent, most important first.
_LORD_HOUSES: dict[str, tuple[int, ...]] = {
    "marriage": (7, 2), "relationship": (7, 4), "love": (5, 7), "career": (10, 6),
    "finance": (11, 2), "health": (1, 6), "personal_growth": (1, 9), "general_astrology": (1,),
}
# Houses whose OCCUPANTS are looked up per intent.
_OCCUPIED_HOUSES: dict[str, tuple[int, ...]] = {
    "marriage": (7,), "relationship": (7, 4), "love": (5, 7), "career": (10,), "finance": (2, 11),
    "health": (6, 1), "personal_growth": (1,), "general_astrology": (1,),
}
_TOPIC_WORD = {"marriage": "marriage", "relationship": "relationships and family", "love": "love",
               "career": "career", "finance": "wealth", "health": "health", "personal_growth": "personality",
               "general_astrology": "effects"}

_TIMING = re.compile(r"\b(?:when|kab|timing|time|period|dasha|mahadasha|antardasha|year|saal|age|umar)\b"
                     r"|कब|समय|दशा|अवधि", re.I)
_DASHA_ASK = re.compile(r"\b(?:dasha|mahadasha|antardasha|dasa)\b|दशा", re.I)
_ASKS_REMEDY = re.compile(r"\b(?:remed\w*|upay\w*|upaay|totka|mantra|puja|pooja|donate|daan)\b|उपाय", re.I)
_ASKS_GEM = re.compile(r"\b(?:gem\w*|stone|ratna|ruby|emerald|sapphire|pearl|coral|diamond)\b|रत्न", re.I)
_ASKS_TRANSIT = re.compile(r"\b(?:transit\w*|gochar|sade\s*sati|sadesati)\b|गोचर|साढ़े", re.I)
# Chunks describing someone else's chart: never shown to the model, whose job
# is to read THIS chart (and who would happily copy a stranger's dates).
_FOREIGN_CHART = re.compile(r"case stud|date of (?:marriage|birth)|\bdob\b|answer key", re.I)


@dataclass(frozen=True)
class SubQuery:
    query: str
    entities: dict[str, Any]
    applies_to: str | None      # the chart fact this evidence is about, or None for a semantic query
    priority: int               # lower = more specific to this chart


def _cap(name: str | None) -> str | None:
    return name.capitalize() if name else None


def plan_queries(question: str, intent: str, facts: Any) -> list[SubQuery]:
    """The retrieval plan for one turn. Pure — tested without a network."""
    plan: list[SubQuery] = []
    intent = intent if intent in _LORD_HOUSES else "general_astrology"
    topic = _TOPIC_WORD[intent]
    available = bool(getattr(facts, "available", False))

    # A question ABOUT the dasha puts dasha evidence first; a topic question
    # asking "when" puts the topic's main house lord first and the running
    # period second.
    dasha_question = bool(_DASHA_ASK.search(question or ""))
    if available and (dasha_question or _TIMING.search(question or "")) and getattr(facts, "mahadasha", None):
        md, ad = _cap(facts.mahadasha), _cap(facts.antardasha)
        if ad and ad != md:
            plan.append(SubQuery(f"{md} Mahadasha {ad} Antardasha",
                                 {"planets": [md, ad], "category_slug": "dasha"},
                                 f"current {md} Mahadasha / {ad} Antardasha", 0 if dasha_question else 1))
        if dasha_question or not ad or ad == md:
            plan.append(SubQuery(f"{md} Mahadasha",
                                 {"planet": md, "category_slug": "dasha"},
                                 f"current {md} Mahadasha", 0 if dasha_question else 1))

    if available and facts.houses:
        for rank, house in enumerate(_LORD_HOUSES[intent]):
            fact = facts.houses.get(house)
            if fact and fact.lord_house:
                plan.append(SubQuery(
                    f"{_ORD[house]} lord in the {_ORD[fact.lord_house]} house",
                    {"house_lord": house, "house": fact.lord_house},
                    f"{_ORD[house]} lord {fact.lord.capitalize()} in the {_ORD[fact.lord_house]} house",
                    0 if rank == 0 and not dasha_question else 2))
        for house in _OCCUPIED_HOUSES[intent]:
            fact = facts.houses.get(house)
            for graha in (fact.occupants if fact else [])[:1]:
                plan.append(SubQuery(
                    f"{graha.capitalize()} in the {_ORD[house]} house",
                    {"planet": graha.capitalize(), "house": house},
                    f"{graha.capitalize()} in the {_ORD[house]} house", 2))
            if fact and not fact.occupants:
                plan.append(SubQuery(
                    f"{fact.sign} in the {_ORD[house]} house",
                    {"rashi": fact.sign, "house": house},
                    f"{fact.sign} on the {_ORD[house]} house", 3))

    plan.append(SubQuery(question.strip()[:400], {}, None, 5))
    # Most chart-specific first; de-duplicate identical filters.
    seen: set[str] = set()
    unique: list[SubQuery] = []
    for sub in sorted(plan, key=lambda s: s.priority):
        key = repr(sorted(sub.entities.items()))
        if key not in seen:
            seen.add(key)
            unique.append(sub)
    limit = settings.premium_knowledge.max_queries
    # Always keep the semantic query as the last slot.
    semantic = [s for s in unique if s.applies_to is None]
    specific = [s for s in unique if s.applies_to is not None][: max(0, limit - len(semantic[:1]))]
    return specific + semantic[:1]


def _allowed(result: dict[str, Any], question: str) -> bool:
    category = str(result.get("category_slug") or "")
    facet = str(result.get("facet") or "")
    head = f"{result.get('section_title') or ''} {result.get('subsection_title') or ''} {category}"
    if result.get("needs_review"):
        return False
    if category == "case_studies" or _FOREIGN_CHART.search(head) or _FOREIGN_CHART.search(
            str(result.get("text") or "")[:400]):
        return False
    if (category == "remedies" or facet == "remedies") and not _ASKS_REMEDY.search(question):
        return False
    if category == "gemstones" and not _ASKS_GEM.search(question):
        return False
    if (category == "transit" or "transit" in head.lower()) and not _ASKS_TRANSIT.search(question):
        return False
    return _consistent_house(result)


_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
          "XI": 11, "XII": 12}
_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
          "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12}
_LORD_IN = re.compile(r"lord\s+in\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+house"
                      r"|lord\s+in\s+(?:the\s+)?([IVX]{1,4})\s+house"
                      r"|lord\s+in\s+(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
                      r"eleventh|twelfth)\s+house", re.I)


def _consistent_house(result: dict[str, Any]) -> bool:
    """Drop a house-lord chunk whose own title names a different house than
    its metadata. Found while integrating: a chunk tagged house_lord=1,
    house=12 is titled "Effects of I House Lord in XI House"."""
    if result.get("house") is None or result.get("house_lord") is None:
        return True
    head = f"{result.get('section_title') or ''} {result.get('subsection_title') or ''} "            f"{str(result.get('text') or '')[:160]}"
    for match in _LORD_IN.finditer(head):
        digits, roman, word = match.groups()
        named = int(digits) if digits else (_ROMAN.get(roman.upper()) if roman else _WORDS.get(word.lower()))
        if named and named != int(result["house"]):
            return False
    return True


# Results of identical sub-queries, reused. The corpus is static (re-ingestion
# is a deliberate, rare event) and chart-fact sub-queries repeat constantly —
# every user whose 7th lord sits in the 6th asks the same one — while each
# call costs a CPU bge-m3 embedding (~0.5–1 s, serialised) plus a Pinecone
# query. Only knowledge passages are cached, never anything about a user.
_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 4096


def _cache_key(sub: SubQuery) -> str:
    return json.dumps([sub.query.strip().lower(), sorted(sub.entities.items())], ensure_ascii=False, default=str)


def cache_clear() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


@traced("premium_rag.query")
def _call(sub: SubQuery) -> list[dict[str, Any]]:
    import httpx

    cfg = settings.premium_knowledge
    key = _cache_key(sub)
    ttl = cfg.cache_seconds
    if ttl > 0:
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
        if hit and time.monotonic() - hit[0] < ttl:
            raw = hit[1]
            return [{**r, "_applies_to": sub.applies_to, "_priority": sub.priority, "_entities": sub.entities}
                    for r in raw]
    body: dict[str, Any] = {"query": sub.query, "top_k": cfg.top_k_per_query,
                            "max_context_tokens": 1500}
    if sub.entities:
        body["structured_entities"] = sub.entities
    response = httpx.post(f"{cfg.url.rstrip('/')}/api/rag/retrieve", json=body, timeout=cfg.timeout_seconds)
    response.raise_for_status()
    raw = list(response.json().get("results") or [])
    if ttl > 0:
        with _CACHE_LOCK:
            if len(_CACHE) >= _CACHE_MAX:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = (time.monotonic(), raw)
    return [{**r, "_applies_to": sub.applies_to, "_priority": sub.priority, "_entities": sub.entities}
            for r in raw]


def matches_entities(result: dict[str, Any]) -> bool:
    """A chart-fact chunk must be about exactly the requested fact.

    The Premium service relaxes its filter when a strict match returns few
    candidates, and the relaxed stages happily return the mirror image: a
    query for "10th lord in the 6th" came back with "6th lord in the 10th",
    and a dasha query with a nakshatra chunk (found in the retrieval eval).
    Those are not evidence about this chart, so they are dropped. Semantic
    (non chart-fact) sub-queries are not constrained."""
    # The chatbot reads the English half of the bilingual corpus; the model
    # writes the reply in the user's language itself.
    if result.get("language") not in (None, "en"):
        return False
    if not result.get("_applies_to"):
        return True
    wanted = result.get("_entities") or {}
    for key in ("house", "house_lord", "planet", "rashi", "category_slug"):
        if key in wanted and result.get(key) != wanted[key]:
            return False
    if "planets" in wanted:
        got = {str(p) for p in result.get("planets") or []}
        if got != set(wanted["planets"]):
            return False
        # "Venus-Mercury Antardasha" (Venus Mahadasha) is not "Mercury-Venus".
        if wanted.get("category_slug") == "dasha" and len(wanted["planets"]) == 2:
            pair = "-".join(wanted["planets"]).lower()
            head = f"{result.get('subsection_title') or ''} {result.get('section_title') or ''}".lower()
            if pair not in head.replace(" - ", "-"):
                return False
    return True


def _to_chunk(result: dict[str, Any]) -> RetrievedChunk:
    cfg = settings.premium_knowledge
    text = str(result.get("text") or "").strip()
    if len(text) > cfg.max_chars_per_chunk:
        cut = text[: cfg.max_chars_per_chunk]
        text = cut[: cut.rfind(" ")] + " …" if " " in cut else cut
    applies = result.get("_applies_to")
    if applies:
        text = f"(Applies to this chart: {applies}.)\n{text}"
    document = str(result.get("document_title") or result.get("category") or "Tradeva knowledge").strip()
    topic = str(result.get("subsection_title") or result.get("section_title") or "").strip()
    return RetrievedChunk(text=text, score=float(result.get("score") or 0.0),
                          domain=str(result.get("category_slug") or "unknown"),
                          topic=topic or "general", document=document.title() if document.isupper() else document,
                          source="tradeva-premium")


def gather(question: str, intent: str, kundli: Any) -> tuple[list[SubQuery], list[dict[str, Any]], int]:
    """Run the plan's sub-queries in parallel: (plan, raw results, failures)."""
    cfg = settings.premium_knowledge
    from app.kundli.chart_facts import build_chart_facts

    facts = build_chart_facts(kundli) if kundli is not None else None
    plan = plan_queries(question, intent, facts)
    futures = [_pool.submit(_call, sub) for sub in plan]
    results: list[dict[str, Any]] = []
    failures = 0
    for future in futures:
        try:
            results.extend(future.result(timeout=cfg.timeout_seconds + 1))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            log.warning("premium retrieval sub-query failed", extra={"error_type": type(exc).__name__})
    return plan, results, failures


def consistent_with_chart(result: dict[str, Any], facts: Any) -> bool:
    """A chunk describing a specific placement ("7th lord in the 3rd house",
    "Venus in the 5th") is kept only if THIS chart has that placement. A
    semantic match on "my 7th house" otherwise brought back passages for
    every possible 7th-lord position, which a model can mistake for a
    reading of this chart."""
    if facts is None or not getattr(facts, "available", False):
        return True
    house, lord_of, planet = result.get("house"), result.get("house_lord"), result.get("planet")
    try:
        if lord_of and house and facts.houses:
            fact = facts.houses.get(int(lord_of))
            return bool(fact and fact.lord_house == int(house))
        if planet and house and str(result.get("category_slug")) in _PLACEMENT_CATEGORIES:
            return facts.house_of(str(planet).lower()) == int(house)
    except (TypeError, ValueError):
        return False
    return True


_PLACEMENT_CATEGORIES = {"planets_in_houses", "retrograde_planets", "vargottam_combust_planets", "marriage",
                         "career", "medical", "child"}


def select(results: list[dict[str, Any]], question: str, facts: Any = None) -> list[dict[str, Any]]:
    """Filter, de-duplicate and cap raw results; chart-fact evidence first."""
    cfg = settings.premium_knowledge
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    kept: list[dict[str, Any]] = []
    specific = 0
    semantic_available = any(not r.get("_applies_to") and _allowed(r, question)
                             and consistent_with_chart(r, facts)
                             and float(r.get("score") or 0) >= cfg.min_score for r in results)
    for result in sorted(results, key=lambda r: (r["_priority"], -float(r.get("score") or 0))):
        cid = str(result.get("chunk_id") or "")
        head = re.sub(r"\W+", " ", str(result.get("text") or "")[:200].lower())
        if cid in seen_ids or head in seen_text or not _allowed(result, question) \
                or not matches_entities(result) or not consistent_with_chart(result, facts):
            continue
        if float(result.get("score") or 0) < cfg.min_score:
            continue
        if result.get("_applies_to") and specific >= cfg.max_chunks - 1 and semantic_available:
            continue        # keep one slot for the question's own semantic match
        seen_ids.add(cid)
        seen_text.add(head)
        kept.append(result)
        specific += 1 if result.get("_applies_to") else 0
        if len(kept) >= cfg.max_chunks:
            break
    return kept


def retrieve_premium(question: str, intent: str, kundli: Any) -> RetrievalResult | None:
    """Evidence for one chatbot turn, or None when the Premium service is not
    configured or unreachable (the caller then uses the legacy namespace)."""
    if not settings.premium_knowledge.configured:
        return None
    plan, results, failures = gather(question, intent, kundli)
    if failures == len(plan):
        return None
    from app.kundli.chart_facts import build_chart_facts

    kept = select(results, question, build_chart_facts(kundli) if kundli is not None else None)
    chunks = [_to_chunk(r) for r in kept]
    log.info("premium retrieval complete", extra={
        "intent": intent, "sub_queries": len(plan), "candidates": len(results), "chunks": len(chunks),
        "failed_sub_queries": failures,
        "applies_to": [r.get("_applies_to") for r in kept if r.get("_applies_to")]})
    return RetrievalResult(chunks=chunks, used_filter=True, widened=False, available=True)


__all__ = ["retrieve_premium", "plan_queries", "gather", "select", "matches_entities",
           "consistent_with_chart", "SubQuery"]
