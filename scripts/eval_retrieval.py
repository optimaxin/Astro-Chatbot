"""Measure production retrieval (Pinecone) against labelled queries.

    python -m scripts.eval_retrieval                 # current settings + variants
    python -m scripts.eval_retrieval --out logs/evals/retrieval.json

Each query is labelled with the knowledge-base section(s) that answer it
("document::topic-prefix"). Every query is embedded ONCE; the vector is then
replayed against Pinecone under several configurations, so variants are
compared on identical inputs and the knowledge base is never re-embedded.

Metrics per configuration:
  hit@k      a relevant section is among the chunks handed to the model
  precision  relevant chunks / chunks handed to the model
  MRR        1 / rank of the first relevant chunk
  recall@10  a relevant section is anywhere in the raw top 10 (no cut-offs):
             separates "not retrievable" from "retrieved but filtered out"
  dupes      duplicate chunk texts handed to the model
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.nodes.classifier import classify_by_keywords, classify_unambiguous  # noqa: E402
from app.rag import retriever  # noqa: E402
from app.rag.embeddings import embed_query  # noqa: E402

# (query, intent, [relevant "document::topic-prefix", ...])
CASES: list[tuple[str, str, list[str]]] = [
    ("Is this a good time for me to change my job?", "career", ["career.txt::job_change", "career.txt::timing_professional"]),
    ("How does Saturn affect my career?", "career", ["career.txt::saturn_and_career", "planets.txt::saturn_shani"]),
    ("Should I start my own business or stay employed?", "career", ["career.txt::job_change"]),
    ("Will I get a promotion this year?", "career", ["career.txt::career_growth", "career.txt::timing_professional"]),
    ("Can Rahu give success in an unusual career like tech or media?", "career", ["career.txt::rahu_ketu", "planets.txt::rahu_ambition"]),
    ("What does the 10th house say about my work?", "career", ["houses.txt::the_10th_house", "career.txt::the_core_factors"]),
    ("Career mein growth kab milegi?", "career", ["career.txt::timing_professional", "career.txt::career_growth"]),
    ("Naukri badalni chahiye ya nahi?", "career", ["career.txt::job_change", "career.txt::career_growth"]),
    ("When will I get married?", "marriage", ["relationships.txt::timing_of_marriage"]),
    ("What does my 7th house say about my spouse?", "marriage", ["houses.txt::the_7th_house", "relationships.txt::the_core_factors"]),
    ("Am I manglik and does it affect marriage?", "marriage", ["relationships.txt::manglik_or_kuja"]),
    ("How is kundli matching and guna milan done?", "marriage", ["relationships.txt::compatibility_and_guna", "nakshatras.txt::gana_nadi"]),
    ("Meri shaadi kab hogi?", "marriage", ["relationships.txt::timing_of_marriage"]),
    ("मेरी शादी कब होगी?", "marriage", ["relationships.txt::timing_of_marriage"]),
    ("We fight a lot, what does astrology say about our relationship?", "relationship", ["relationships.txt::difficulties_in_an_existing"]),
    ("Is love marriage indicated differently from arranged marriage?", "love", ["relationships.txt::love_relationships_versus"]),
    ("How does Venus affect my relationships?", "love", ["relationships.txt::venus_and_the_quality", "planets.txt::venus_shukra"]),
    ("Will I become wealthy?", "finance", ["finance.txt::the_houses_that_govern", "finance.txt::dhana_yogas"]),
    ("Which houses show money in a kundli?", "finance", ["finance.txt::the_houses_that_govern", "houses.txt::the_2nd_house", "houses.txt::the_11th_house"]),
    ("Should I invest in the stock market now?", "finance", ["finance.txt::speculation_investment"]),
    ("I have a lot of debt, what does my chart say?", "finance", ["finance.txt::debt_expenditure"]),
    ("Paisa kab aayega?", "finance", ["finance.txt::timing_financial"]),
    ("What is a dhana yoga?", "finance", ["finance.txt::dhana_yogas"]),
    ("Which planet governs which part of the body?", "health", ["health.txt::what_each_planet_governs"]),
    ("I feel anxious all the time, does my chart explain it?", "health", ["health.txt::mental_and_emotional", "health.txt::responding_to_a_person"]),
    ("Does Sade Sati affect health?", "health", ["health.txt::sade_sati_dasha", "dashas.txt::sade_sati"]),
    ("What is the Vimshottari dasha system?", "general_astrology", ["dashas.txt::what_the_vimshottari"]),
    ("What is the difference between mahadasha and antardasha?", "general_astrology", ["dashas.txt::mahadasha_antardasha"]),
    ("What happens in Rahu mahadasha?", "general_astrology", ["dashas.txt::rahu_and_jupiter"]),
    ("What is Sade Sati?", "general_astrology", ["dashas.txt::sade_sati"]),
    ("Sade sati kya hoti hai?", "general_astrology", ["dashas.txt::sade_sati"]),
    ("What is a nakshatra?", "general_astrology", ["nakshatras.txt::what_a_nakshatra"]),
    ("Why does the birth nakshatra matter so much?", "general_astrology", ["nakshatras.txt::why_the_birth"]),
    ("What is the ascendant or lagna?", "general_astrology", ["houses.txt::the_1st_house"]),
    ("What are kendra and trikona houses?", "general_astrology", ["houses.txt::kendra_trikona"]),
    ("Which sign is Jupiter exalted in?", "general_astrology", ["rashis.txt::exaltation_and_debilitation"]),
    ("What is the nature of Scorpio?", "general_astrology", ["rashis.txt::scorpio_vrishchika"]),
    ("Are Rahu and Ketu real planets?", "general_astrology", ["planets.txt::rahu_ambition", "planets.txt::ketu_detachment"]),
    ("What are natural benefics and malefics?", "general_astrology", ["planets.txt::natural_benefics"]),
    ("What does the Moon signify in astrology?", "general_astrology", ["planets.txt::moon_chandra"]),
    ("नक्षत्र क्या होता है?", "general_astrology", ["nakshatras.txt::what_a_nakshatra"]),
    ("Do remedies like gemstones really work?", "personal_growth", ["personal_growth.txt::remedies_and_the_role"]),
    ("What is my life purpose according to my chart?", "personal_growth", ["personal_growth.txt::handling_questions_about_purpose"]),
    ("Am I spiritually inclined?", "personal_growth", ["personal_growth.txt::spiritual_inclination"]),
    ("Can astrology predict everything about my future?", "personal_growth", ["personal_growth.txt::what_an_astrological_reading", "career.txt::the_limits"]),
]


def _relevant(chunk: retriever.RetrievedChunk, labels: list[str]) -> bool:
    for label in labels:
        doc, _, prefix = label.partition("::")
        if chunk.document == doc and chunk.topic.startswith(prefix):
            return True
    return False


def _config_chunks(vector: list[float], intent: str, top_k: int, relative_min: float,
                   use_filter: bool) -> list[retriever.RetrievedChunk]:
    """Mirror retriever.retrieve() with explicit knobs, on a fixed vector."""
    domains = retriever.INTENT_DOMAINS.get(intent, retriever.FOUNDATION_DOMAINS) if use_filter else None
    chunks = retriever._to_chunks(retriever._query(vector, top_k, domains), settings.pinecone.min_score)
    if use_filter and len(chunks) < max(1, top_k // 2):
        seen = {c.text for c in chunks}
        chunks.extend(c for c in retriever._to_chunks(retriever._query(vector, top_k, None),
                                                       settings.pinecone.min_score) if c.text not in seen)
    chunks.sort(key=lambda c: c.score, reverse=True)
    if chunks and relative_min > 0:
        floor = chunks[0].score * relative_min
        chunks = [c for c in chunks if c.score >= floor]
    return chunks[:top_k]


CONFIGS = {
    "current (k=4, rel=0.55, filter)": dict(top_k=4, relative_min=0.55, use_filter=True),
    "k=6, rel=0.55, filter": dict(top_k=6, relative_min=0.55, use_filter=True),
    "k=4, rel=0.40, filter": dict(top_k=4, relative_min=0.40, use_filter=True),
    "k=4, rel=0.55, no filter": dict(top_k=4, relative_min=0.55, use_filter=False),
    "k=6, rel=0.40, filter": dict(top_k=6, relative_min=0.40, use_filter=True),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(settings.base_dir / "logs" / "evals" / "retrieval.json"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for query, labelled_intent, labels in CASES:
        vector = embed_query(query)
        quick = classify_unambiguous(query, has_history=False)
        pipeline_intent = quick.intent if quick else classify_by_keywords(query).intent
        raw = retriever._to_chunks(retriever._query(vector, 10, None), 0.0)
        rank = next((i + 1 for i, c in enumerate(raw) if _relevant(c, labels)), None)
        row = {"query": query, "labelled_intent": labelled_intent, "pipeline_intent": pipeline_intent,
               "labels": labels, "raw_rank_of_first_relevant": rank,
               "raw_top3": [f"{c.document}::{c.topic[:40]} ({c.score:.3f})" for c in raw[:3]], "configs": {}}
        for name, cfg in CONFIGS.items():
            chunks = _config_chunks(vector, labelled_intent, **cfg)
            ranks = [i + 1 for i, c in enumerate(chunks) if _relevant(c, labels)]
            row["configs"][name] = {
                "n": len(chunks), "hit": bool(ranks), "relevant": len(ranks),
                "mrr": 1 / ranks[0] if ranks else 0.0,
                "dupes": len(chunks) - len({c.text for c in chunks}),
            }
        rows.append(row)
        mark = "hit " if row["configs"]["current (k=4, rel=0.55, filter)"]["hit"] else "MISS"
        print(f"  {mark} raw_rank={str(rank):<4} {query[:60]}", flush=True)

    summary: dict[str, Any] = {}
    for name in CONFIGS:
        c = [r["configs"][name] for r in rows]
        summary[name] = {
            "hit_rate": round(statistics.fmean(x["hit"] for x in c), 3),
            "precision": round(sum(x["relevant"] for x in c) / max(1, sum(x["n"] for x in c)), 3),
            "mrr": round(statistics.fmean(x["mrr"] for x in c), 3),
            "avg_chunks": round(statistics.fmean(x["n"] for x in c), 2),
            "duplicates": sum(x["dupes"] for x in c),
        }
    summary["recall@10 (raw, no cut-offs)"] = round(
        statistics.fmean(r["raw_rank_of_first_relevant"] is not None for r in rows), 3)
    summary["intent agreement (pipeline vs label)"] = round(
        statistics.fmean(r["pipeline_intent"] == r["labelled_intent"] for r in rows), 3)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print()
    for name, s in summary.items():
        print(f"{name:<42} {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
