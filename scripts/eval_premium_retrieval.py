"""Retrieval evaluation: Tradeva Premium corpus (chatbot plan) vs the legacy namespace.

    PREMIUM_RAG_URL=http://127.0.0.1:8100 python -m scripts.eval_premium_retrieval

For each question (asked against the screenshot chart) a chunk counts as:

* relevant   — it is about the asked topic: a chart-fact chunk for one of the
               topic's houses/planets/periods, or a chunk whose category belongs
               to the topic (rule table below, decided before looking at results)
* grounded   — a chart-fact chunk whose metadata matches what the Kundli engine
               computed for this chart (e.g. house_lord=7 AND house=6 because the
               7th lord Mercury really sits in the 6th)
* excluded   — must never appear: case studies (other people's charts),
               remedies/gemstones/transits when not asked for

Reports precision@k, grounding rate, excluded-chunk rate, empty-result rate and
latency for both sources. No LLM calls.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.kundli.chart_facts import build_chart_facts  # noqa: E402
from app.kundli.kundli_service import KundliContext  # noqa: E402
from app.rag import premium_knowledge  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402
from scripts.kundli_quality_bench import screenshot_chart  # noqa: E402

# topic -> (intent, relevant Premium categories, relevant legacy domains)
TOPICS = {
    "marriage": ("marriage", {"marriage", "house_lords_in_houses", "planets_in_houses", "dasha",
                              "vargottam_combust_planets", "retrograde_planets", "conjunctions", "blank_chart"},
                 {"relationships", "houses", "planets", "dashas"}),
    "career": ("career", {"career", "house_lords_in_houses", "planets_in_houses", "dasha", "blank_chart",
                          "raj_yog", "conjunctions"}, {"career", "houses", "planets", "dashas"}),
    "finance": ("finance", {"house_lords_in_houses", "planets_in_houses", "dasha", "raj_yog", "blank_chart"},
                {"finance", "houses", "planets", "dashas"}),
    "dasha": ("general_astrology", {"dasha", "house_lords_in_houses", "planets_in_houses"}, {"dashas", "planets"}),
    "family": ("relationship", {"house_lords_in_houses", "planets_in_houses", "blank_chart", "bhavat_bhavam"},
               {"relationships", "houses", "planets"}),
    "health": ("health", {"house_lords_in_houses", "planets_in_houses", "medical", "blank_chart"},
               {"health", "houses", "planets"}),
}

CASES = [
    ("marriage", "When is a good period for marriage?"),
    ("marriage", "What does my 7th house say about married life?"),
    ("marriage", "Meri shaadi kab hogi?"),
    ("marriage", "मेरी शादी के बारे में बताइए।"),
    ("marriage", "What does Venus in my 7th house mean?"),
    ("career", "Tell me about my career."),
    ("career", "Mera career kaisa rahega?"),
    ("career", "Is this a good time to change jobs?"),
    ("career", "मेरा करियर कैसा रहेगा?"),
    ("career", "Who is my 10th lord and what does its placement mean?"),
    ("finance", "How will my financial growth be?"),
    ("finance", "Paise ke maamle mein kundli kya kehti hai?"),
    ("finance", "What does my 11th house say about gains?"),
    ("dasha", "What is my current Dasha about?"),
    ("dasha", "Abhi kaunsi Mahadasha chal rahi hai aur uska kya asar hai?"),
    ("dasha", "मेरी वर्तमान दशा के बारे में बताइए।"),
    ("family", "Tell me about my family and relationships."),
    ("family", "Ghar aur mummy ke baare mein kundli kya kehti hai?"),
    ("health", "Anything in my chart about my health?"),
    ("health", "मेरे स्वास्थ्य के बारे में कुंडली क्या कहती है?"),
]


def _grounded(chunk_meta: dict, facts) -> bool | None:
    """True/False for a chart-fact chunk; None when the chunk is not tied to a chart fact."""
    hl, house, planet = chunk_meta.get("house_lord"), chunk_meta.get("house"), chunk_meta.get("planet")
    if hl and house:
        fact = facts.houses.get(int(hl))
        return bool(fact and fact.lord_house == int(house))
    if planet and house:
        return facts.house_of(str(planet).lower()) == int(house)
    return None


def evaluate_premium(facts, chart) -> list[dict]:
    rows = []
    for topic, question in CASES:
        intent, relevant_cats, _ = TOPICS[topic]
        started = time.perf_counter()
        _, raw, _ = premium_knowledge.gather(question, intent, chart)
        uniq = premium_knowledge.select(raw, question, facts)
        latency = int((time.perf_counter() - started) * 1000)
        excluded_raw = [r for r in raw if not premium_knowledge._allowed(r, question)]
        topic_houses = {*premium_knowledge._LORD_HOUSES.get(intent, ()),
                        *premium_knowledge._OCCUPIED_HOUSES.get(intent, ())}

        def is_relevant(m: dict) -> bool:
            if m.get("category_slug") not in relevant_cats:
                return False
            if m.get("category_slug") == "house_lords_in_houses":
                return m.get("house_lord") in topic_houses          # a lord of a house this topic is read from
            if m.get("category_slug") == "dasha":
                return str(m.get("planet") or "").lower() in {facts.mahadasha, facts.antardasha} or \
                    {p.lower() for p in m.get("planets") or []} <= {facts.mahadasha, facts.antardasha}
            return True

        relevant = [m for m in uniq if is_relevant(m)]
        grounded_flags = [g for g in (_grounded(m, facts) for m in uniq) if g is not None]
        rows.append({"topic": topic, "question": question, "chunks": len(uniq), "relevant": len(relevant),
                     "chart_fact_chunks": len(grounded_flags), "grounded": sum(grounded_flags),
                     "excluded_in_output": sum(1 for m in uniq if not premium_knowledge._allowed(m, question)),
                     "excluded_filtered": len(excluded_raw), "latency_ms": latency,
                     "categories": [m.get("category_slug") for m in uniq]})
    return rows


def evaluate_legacy() -> list[dict]:
    rows = []
    for topic, question in CASES:
        intent, _, relevant_domains = TOPICS[topic]
        started = time.perf_counter()
        result = retrieve(question, intent=intent)
        latency = int((time.perf_counter() - started) * 1000)
        chunks = result.chunks if result.available else []
        rows.append({"topic": topic, "question": question, "chunks": len(chunks),
                     "relevant": sum(1 for c in chunks if c.domain in relevant_domains),
                     "chart_fact_chunks": 0, "grounded": 0, "excluded_in_output": 0, "latency_ms": latency,
                     "categories": [c.domain for c in chunks]})
    return rows


def summary(rows: list[dict]) -> dict:
    total = sum(r["chunks"] for r in rows)
    facts_total = sum(r["chart_fact_chunks"] for r in rows)
    return {
        "questions": len(rows),
        "avg_chunks": round(total / len(rows), 2),
        "precision_%": round(100 * sum(r["relevant"] for r in rows) / max(1, total), 1),
        "chart_fact_chunks_%": round(100 * facts_total / max(1, total), 1),
        "chart_grounding_%": round(100 * sum(r["grounded"] for r in rows) / max(1, facts_total), 1) if facts_total else None,
        "excluded_chunks_in_output": sum(r["excluded_in_output"] for r in rows),
        "empty_results_%": round(100 * sum(1 for r in rows if r["chunks"] == 0) / len(rows), 1),
        "latency_ms_avg": round(statistics.mean(r["latency_ms"] for r in rows)),
        "latency_ms_p95": sorted(r["latency_ms"] for r in rows)[int(0.95 * (len(rows) - 1))],
    }


def main() -> None:
    chart = screenshot_chart(KundliContext)
    facts = build_chart_facts(chart)
    out = {}
    if settings.premium_knowledge.configured:
        premium_rows = evaluate_premium(facts, chart)
        out["premium"] = {"summary": summary(premium_rows), "rows": premium_rows}
    legacy_rows = evaluate_legacy()
    out["legacy"] = {"summary": summary(legacy_rows), "rows": legacy_rows}
    path = Path(__file__).resolve().parents[1] / "logs" / "evals" / "retrieval-premium-vs-legacy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, block in out.items():
        print(name, json.dumps(block["summary"]))
    print("->", path)


if __name__ == "__main__":
    main()
