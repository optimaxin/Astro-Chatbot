"""Generate a human-review batch of real chatbot answers.

    python -m scripts.build_human_eval                      # 60 items
    python -m scripts.build_human_eval --out docs/human-eval/batch.csv

Runs representative questions through the real pipeline (same prompt,
chart context, validation and citations users get) and writes a CSV for
reviewers: one row per answer, with everything a reviewer needs to judge it
and empty rubric columns. Reviewers fill those in (see
docs/human-eval/RUBRIC.md); `scripts/score_human_eval.py` aggregates them.

No score in this file is produced by a model. The model name is kept in a
separate column so reviewers can be blinded by hiding it.
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.llm.usage_tracker import set_traffic_source  # noqa: E402

set_traffic_source("evaluation")

from app.graph.chatbot_graph import run_turn  # noqa: E402
from app.kundli.kundli_service import SampleFileKundliProvider  # noqa: E402
from app.memory.redis_memory import memory  # noqa: E402
from app.observability import background  # noqa: E402
from app.prompts.response_style import resolve_style  # noqa: E402

# (category, question) — single-turn items
SINGLE: list[tuple[str, str]] = [
    ("career", "How does my career look? I'm a software developer."),
    ("career", "Is this a good time to switch jobs?"),
    ("career", "Should I start my own business or keep my job?"),
    ("career", "Which planet in my chart supports my profession?"),
    ("career", "Will I get a promotion this year?"),
    ("career", "What does my 10th house say?"),
    ("career", "I feel stuck at work. Does my chart show why?"),
    ("career", "Is government service suited to my chart?"),
    ("marriage", "What does my chart say about marriage?"),
    ("marriage", "When is a good period for marriage for me?"),
    ("marriage", "What kind of partner suits me according to my 7th house?"),
    ("marriage", "Am I manglik?"),
    ("marriage", "Will I have a love marriage or an arranged marriage?"),
    ("marriage", "We argue a lot. What does my chart say about my relationships?"),
    ("marriage", "What does Venus in my chart mean for love?"),
    ("finance", "Will I make good money? Tell me from my chart."),
    ("finance", "Which houses in my chart show wealth?"),
    ("finance", "Is now a good time to invest in stocks?"),
    ("finance", "I have debts. What does my chart say?"),
    ("finance", "Do I have any dhana yoga?"),
    ("finance", "Should I buy property this year?"),
    ("general", "Which mahadasha am I running and what does it mean?"),
    ("general", "What is my ascendant and moon sign?"),
    ("general", "Is Sade Sati affecting me?"),
    ("general", "What is a nakshatra, and what is mine?"),
    ("general", "Do I have Kaal Sarp dosha?"),
    ("general", "Which planets are retrograde in my chart?"),
    ("general", "What does it mean that Sun and Venus are together in my chart?"),
    ("general", "Can astrology predict everything about my future?"),
    ("hinglish", "Meri career growth kab achhi hogi?"),
    ("hinglish", "Mera Guru kis ghar mein baitha hai?"),
    ("hinglish", "Meri shaadi ke yog kaise hain?"),
    ("hinglish", "Shani ka mere upar kya asar hai abhi?"),
    ("hinglish", "Paisa kab aayega? Chart dekh ke batao."),
    ("hinglish", "Mera lagna aur rashi kya hai?"),
    ("hinglish", "Kya mujhe naukri badalni chahiye?"),
    ("hinglish", "Mangal dosh hai kya mere chart mein?"),
    ("hinglish", "Meri health ke liye chart kya kehta hai?"),
    ("hinglish", "Venus mahadasha mein kya hoga?"),
    ("hindi", "मेरे करियर के बारे में मेरी कुंडली क्या कहती है?"),
    ("hindi", "मेरा गुरु किस भाव में है?"),
    ("hindi", "मेरी शादी कब होगी?"),
    ("hindi", "अभी मेरी कौन सी महादशा चल रही है?"),
    ("hindi", "क्या मेरी कुंडली में कोई दोष है?"),
    ("hindi", "मेरा लग्न क्या है और इसका क्या अर्थ है?"),
    ("hindi", "क्या साढ़ेसाती मुझ पर चल रही है?"),
    ("hindi", "धन के लिए मेरी कुंडली क्या बताती है?"),
]
# (category, first turn, follow-up) — the follow-up is the item reviewed
FOLLOW_UPS: list[tuple[str, str, str]] = [
    ("follow-up", "I'm a teacher and I'm 30.", "Is this a good time to change jobs?"),
    ("follow-up", "What does my chart say about marriage?", "And what about the next two years?"),
    ("follow-up", "Which mahadasha am I in?", "What changes when the antardasha changes?"),
    ("follow-up", "Meri shaadi kab hogi?", "Aur career ke baare mein?"),
    ("follow-up", "How is my career looking?", "Should I study further instead?"),
    ("follow-up", "Is Saturn troubling me?", "What can I practically do about it?"),
    ("follow-up", "Do I have any doshas?", "Are you sure? My friend said I'm manglik."),
    ("follow-up", "मेरी कुंडली में धन योग है?", "और नौकरी के बारे में?"),
    ("follow-up", "Mera rashi Vrishabh hai na?", "Toh mera lucky colour kya hai?"),
    ("follow-up", "What's my moon sign?", "What does that say about my emotions?"),
    ("follow-up", "Will I travel abroad?", "When exactly?"),
    ("follow-up", "Hi!", "Can you look at my career?"),
    ("follow-up", "Which house is my Jupiter in?", "Is that good for career?"),
]
RUBRIC_COLUMNS = ["reviewer", "chart_accuracy_1to5", "grounding_1to5", "helpfulness_1to5",
                  "language_naturalness_1to5", "safety_ok_YN", "overconfident_YN",
                  "would_send_YN", "notes"]


def _ask(question: str, session: str) -> dict:
    style = resolve_style(None, question, settings.api.default_response_style)
    state = run_turn(question, session, settings.default_user_id, chat_bubbles=True, response_style=style.id)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(settings.base_dir / "docs" / "human-eval" /
                                             f"batch-{date.today():%Y%m%d}.csv"))
    args = parser.parse_args()
    chart = SampleFileKundliProvider().get_kundli(settings.default_user_id)
    chart_summary = "; ".join(chart.facts_used())

    rows = []
    items = [(c, None, q) for c, q in SINGLE] + list(FOLLOW_UPS)
    for i, (category, first, question) in enumerate(items, 1):
        session = f"eval-human-{uuid.uuid4().hex[:8]}"
        context = ""
        if first:
            prior = _ask(first, session)
            context = f"USER: {first}\nASSISTANT: {prior.get('response', '')}"
        state = _ask(question, session)
        metrics = state.get("metrics") or {}
        memory.clear_session(session)
        rows.append({
            "item_id": f"H{i:03d}", "category": category,
            "language": resolve_style(None, question, settings.api.default_response_style).id,
            "question": question, "conversation_context": context,
            "chart_facts_available": chart_summary,
            "answer": state.get("response", ""),
            "citations": "; ".join(c["title"] for c in metrics.get("citations") or []),
            "model_hidden_from_reviewers": metrics.get("model", ""),
            "latency_ms": metrics.get("turn_latency_ms", ""),
            **{c: "" for c in RUBRIC_COLUMNS},
        })
        print(f"  {rows[-1]['item_id']} [{category}] {question[:60]}", flush=True)
    background.drain(30)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as handle:   # BOM: Excel shows Hindi correctly
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} items -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
