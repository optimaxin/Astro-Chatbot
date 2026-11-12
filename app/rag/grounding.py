"""Source-grounded answers: how retrieved knowledge is shown to the model,
and how the model's citations are checked before a user sees them.

Measured failure modes this addresses (benchmark RAG cases, 2026-11-13):
overreach on unanswerable questions (the sources are silent and the model
answers "No, ..."), fabricated details with a citation attached, correct
answers with no citation, and passages combined incorrectly. Retrieval
itself was measured separately at 97.8% hit@4 and was not the problem.

Every retrieved passage gets a tag ([K1], [K2], …) and a title built only
from its real metadata (file and section heading) — nothing is invented.
After generation:

* tags that do not correspond to a retrieved passage are removed and
  counted (`invalid`) — a citation can never point at something that was
  not retrieved;
* valid tags become a `citations` list (id + title) for the API;
* tags are removed from the visible text; the sources used are shown as a
  short footer line (CITATION_DISPLAY=footer, default) or left to the client
  (metadata) or kept inline (inline).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_DOC_TITLES = {
    "career.txt": "Career guide", "dashas.txt": "Dasha guide", "finance.txt": "Finance guide",
    "health.txt": "Health guide", "houses.txt": "Houses guide", "nakshatras.txt": "Nakshatra guide",
    "personal_growth.txt": "Personal growth guide", "planets.txt": "Planets guide",
    "rashis.txt": "Rashi guide", "relationships.txt": "Relationships guide",
}


def source_title(document: str, topic: str) -> str:
    """Human title from real metadata only: "Career guide — Job change switching and …"."""
    doc = _DOC_TITLES.get(document) or document.rsplit(".", 1)[0].replace("_", " ").title()
    section = topic.replace("_", " ").strip()
    section = section[:1].upper() + section[1:] if section else ""
    return f"{doc} — {section}" if section else doc


def grounding_rules(tag: str = "K") -> str:
    return (
        "USING THE RETRIEVED KNOWLEDGE\n"
        f"- When a sentence relies on a passage, end it with that passage's tag, e.g. [{tag}2] "
        f"(or [{tag}1][{tag}3]). Use only tags listed above; never invent one.\n"
        "- Answer the exact question in your first sentence, then support it.\n"
        "- State only what the passages actually say. First find the words that state the very thing "
        "asked: the same person, the same item, the same attribute. A neighbouring value is not the "
        "answer: one planet's or point's sign is not another's, a sign is not a house, the Lagna is not "
        "a planet, and a rule for one product or plan is not a rule for another.\n"
        "- If no passage states it, say so plainly (e.g. \"the information I have doesn't mention "
        "that\") and stop there. Silence is never a yes or a no: do not conclude that something is "
        "not allowed, unavailable, free, included or excluded because a passage leaves it out, and "
        "never add a date, number, name or rule they do not contain.\n"
        "- Passages are a few search results, not a complete catalogue. Never give a total or count of "
        "things on the platform (astrologers, experts, users, services) unless a passage states it.\n"
        "- When the answer follows from stated values, do the working and give the final figure: apply "
        "the fee, percentage, discount, bonus or charge to the amount; add or subtract the stated "
        "hours from the stated time (going back past midnight means the previous day). Show the short "
        "calculation and cite every passage you used.\n"
        "- When one passage is a table keyed by some value (by sign, date or plan), take that key from "
        "the other passage exactly as written, check it is the same kind of value the table is keyed "
        "by (a Moon sign is not a Lagna), then read that row.\n"
        "- When passages conflict, use the one that says it replaces the other or is newer.\n"
        "- Facts from the USER KUNDLI block are chart facts: state them plainly, never tag them. "
        "Astrology you know that no passage states must be marked as general (\"generally\", "
        "\"traditionally\") and is never tagged.\n"
        "- Do not write \"according to the context/documents\"; the tag is the attribution."
    )


_TAG = re.compile(r"\[\s*([A-Za-z])\s*(\d{1,2})(?:\s*,\s*[A-Za-z]?\s*\d{1,2})*\s*\]")
_TAG_NUMS = re.compile(r"\d{1,2}")


@dataclass
class CitationResult:
    text: str
    cited: list[int] = field(default_factory=list)      # 1-based indices of real passages, in order
    invalid: list[str] = field(default_factory=list)    # tags that pointed at nothing
    had_tags: bool = False


def process_citations(text: str, n_sources: int, tag: str = "K", keep_inline: bool = False) -> CitationResult:
    """Validate and (by default) remove [K#] tags. Pure."""
    cited: list[int] = []
    invalid: list[str] = []
    had = False

    def repl(match: re.Match) -> str:
        nonlocal had
        if match.group(1).upper() != tag.upper():
            return match.group(0)
        had = True
        good = []
        for num in _TAG_NUMS.findall(match.group(0)):
            index = int(num)
            if 1 <= index <= n_sources:
                good.append(index)
                if index not in cited:
                    cited.append(index)
            else:
                invalid.append(f"{tag}{index}")
        if keep_inline and good:
            return "".join(f"[{tag}{i}]" for i in good)
        return ""

    cleaned = _TAG.sub(repl, text)
    cleaned = re.sub(r"[ \t]+([.,;:!?।])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return CitationResult(text=cleaned, cited=cited, invalid=invalid, had_tags=had)


VERIFIER_SYSTEM = (
    "You check an answer against its sources before it is sent to a user. You are strict.\n"
    "Rules:\n"
    "1. A statement about the topic is supported only if a source states it explicitly. Something "
    "merely implied, a count of what happens to be listed, or a yes/no inferred from the sources "
    "being silent is NOT supported.\n"
    "2. Remove or correct every unsupported statement. If the sources then do not answer the "
    "question, the answer must say plainly that this is not covered (it may mention what is).\n"
    "3. Recheck any arithmetic or combination of sources step by step and fix mistakes.\n"
    "4. Keep supported content, its [{tag}#] citations, the language, script and tone. Chart facts "
    "from a USER KUNDLI block and clearly-marked general astrology may stay without citations.\n"
    "Output only the final answer — no notes about what you changed."
)


def verification_messages(question: str, sources: list[tuple[str, str]], draft: str,
                          tag: str = "K") -> list[dict[str, str]]:
    """`sources` = [(tag_id, text)] exactly as the model saw them."""
    block = "\n".join(f"[{sid}] {text}" for sid, text in sources)
    return [
        {"role": "system", "content": VERIFIER_SYSTEM.replace("{tag}", tag)},
        {"role": "user", "content": f"SOURCES:\n{block}\n\nQUESTION:\n{question}\n\nDRAFT ANSWER:\n{draft}"},
    ]


def citation_footer(citations: list[dict[str, Any]], style_id: str | None) -> str:
    if not citations:
        return ""
    label = {"hindi": "स्रोत (Tredev ज्ञानकोष)", "whatsapp_hinglish": "Source (Tredev knowledge base)"}.get(
        (style_id or "").lower(), "Source (Tredev knowledge base)")
    return f"📚 {label}: " + "; ".join(c["title"] for c in citations)


__all__ = ["source_title", "grounding_rules", "process_citations", "CitationResult", "citation_footer"]
