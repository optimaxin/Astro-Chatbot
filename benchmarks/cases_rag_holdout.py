"""Held-out retrieval-grounded cases (20) — NOT part of the 220-case benchmark.

Written on 2026-11-13, *before* the grounding prompt was changed for RC1, to
check that prompt changes made after reading the 40 RAG failures generalise
instead of fitting those 40 cases. New fictional passages and questions; the
same rule-based scoring (benchmarks/scoring.py) as the main RAG suite.

Run with: python -m benchmarks.run --dataset holdout --grounding production
13 answerable, 7 not answerable.
"""

from __future__ import annotations

from benchmarks.cases_rag import ABSTAIN, RAG_SYSTEM

GEMSTONE = ("Tredev Gemstone Store policy (2026): Every gemstone ships with a lab certificate from GemTest "
            "Jaipur. Rings are made to order and dispatched within 7 working days. Unworn gemstones can be "
            "returned within 10 days for a 90% refund; worn or resized rings cannot be returned. "
            "Delivery is free above Rs 5,000; below that a Rs 149 delivery fee applies.")
SUNITA = ("Astrologer profile - Dr. Sunita Rao. Based in Pune. 17 years of practice in Vedic astrology and "
          "vastu. Consults in Marathi, Hindi and English. Fee on Tredev Astro: Rs 40 per minute.")
FIRSTCALL = ("Tredev Astro offer: a user's first consultation on the platform is billed at 50% of the "
             "astrologer's per-minute fee for up to 15 minutes; minutes beyond 15 are billed at the full fee.")
ARJUN = ("User chart summary - Arjun M. (chart_id KND-5120). Lagna: Sagittarius. Moon: Gemini, Ardra "
         "nakshatra. Mars: Leo in the 9th house. Current Mahadasha: Saturn (until March 2033).")
LUCKY = ("Tredev lucky-colour guide 2026, by Lagna: Aries red, Taurus white, Gemini green, Cancer silver, "
         "Leo gold, Virgo olive, Libra pink, Scorpio maroon, Sagittarius yellow, Capricorn navy, "
         "Aquarius electric blue, Pisces sea green.")
GRAHAN = ("Shastri Panchang 2026 edition - eclipse table (Varanasi): Lunar eclipse on 3 March 2026, visible, "
          "sutak begins 9 hours before the eclipse starts; eclipse from 15:20 to 18:47. "
          "Solar eclipse on 12 August 2026, not visible from India, no sutak observed.")
NOTICE_OLD = ("Tredev Astro notice (January 2026): Live chat support is available from 9 AM to 9 PM IST, "
              "every day.")
NOTICE_NEW = ("Tredev Astro notice (September 2026): Live chat support hours change to 8 AM to 11 PM IST on "
              "weekdays and 10 AM to 6 PM IST on weekends. This replaces all earlier support-hour notices.")
COURSE = ("Tredev Academy course - Foundations of Jyotish: 24 recorded lessons and 4 live doubt sessions, "
          "fee Rs 4,999. A certificate is issued after passing the final quiz with at least 60%.")
WALLET = ("Tredev Wallet terms: recharges of Rs 500 or more receive 10% bonus credit. Bonus credit expires "
          "90 days after the recharge. Wallet money can be used for consultations and pooja bookings.")


def _c(cid: str, prompt: str, context: list[str], **checks) -> dict:
    return {"id": cid, "category": "rag", "prompt": prompt, "system": RAG_SYSTEM,
            "context": [{"id": f"S{i}", "text": t} for i, t in enumerate(context, 1)],
            "checks": checks}


def _no(cid: str, prompt: str, context: list[str]) -> dict:
    return _c(cid, prompt, context, answerable=False, any=ABSTAIN)


HOLDOUT_CASES: list[dict] = [
    # single facts
    _c("rho-001", "Which lab certifies the gemstones sold on Tredev?", [GEMSTONE],
       any=["GemTest"], answerable=True, cite=True),
    _c("rho-002", "How many years of experience does Dr. Sunita Rao have?", [SUNITA],
       regex=r"\b17\b", answerable=True, cite=True),
    _c("rho-003", "When does bonus wallet credit expire?", [WALLET],
       regex=r"\b90\b", answerable=True, cite=True),
    # arithmetic over stated values
    _c("rho-004", "I returned an unworn Rs 8,000 gemstone within a week. How much do I get back?", [GEMSTONE],
       regex=r"7,?200", answerable=True, cite=True),
    _c("rho-005", "It's my first consultation and I talk to Dr. Sunita Rao for 10 minutes. What will it cost?",
       [SUNITA, FIRSTCALL], regex=r"\b200\b", answerable=True, cite=True),
    _c("rho-006", "If I recharge my wallet with Rs 1,000, how much bonus credit do I get?", [WALLET],
       regex=r"\b100\b", answerable=True, cite=True),
    _c("rho-007", "At what time does sutak begin for the March 2026 lunar eclipse?", [GRAHAN],
       any=["06:20", "6:20"], answerable=True, cite=True),
    _c("rho-008", "I'm buying a Rs 3,000 ring. What is the total with delivery?", [GEMSTONE],
       regex=r"3,?149", answerable=True, cite=True),
    # lookup keyed by a value from another passage
    _c("rho-009", "What is Arjun's lucky colour for 2026?", [ARJUN, LUCKY],
       any=["yellow"], answerable=True, cite=True),
    # superseded notice
    _c("rho-010", "Until what time is live chat support available on a Wednesday?", [NOTICE_OLD, NOTICE_NEW],
       any=["11 PM", "11 pm", "11:00 PM", "23:00"], answerable=True, cite=True),
    _c("rho-011", "Is a sutak observed for the August 2026 solar eclipse?", [GRAHAN],
       any=["no sutak", "not observed", "isn't observed", "is not observed", "no, ", "no."],
       answerable=True, cite=True),
    _c("rho-012", "How many live doubt sessions does the Foundations of Jyotish course include?", [COURSE],
       regex=r"\b(4|four)\b", answerable=True, cite=True),
    _c("rho-013", "Can wallet money be used to book a pooja?", [WALLET],
       any=["yes", "can be used"], answerable=True, cite=True),
    # not answerable: neighbouring attribute present, asked attribute absent
    _no("rho-014", "Which nakshatra is Arjun's Lagna in?", [ARJUN]),
    _no("rho-015", "In which house is Arjun's Moon?", [ARJUN]),
    # not answerable: silence is not a yes or a no
    _no("rho-016", "Can I get my ring resized later for free?", [GEMSTONE]),
    _no("rho-017", "Is the Foundations of Jyotish course available in Hindi?", [COURSE]),
    # not answerable: totals are not derivable from a few retrieved profiles
    _no("rho-018", "How many vastu experts does Tredev Astro have?", [SUNITA]),
    # not answerable: asked value is missing from the table
    _no("rho-019", "How long does gemstone delivery take after dispatch?", [GEMSTONE]),
    _no("rho-020", "What is the pass mark for the Advanced Jyotish course?", [COURSE]),
]
