"""The correction notes and message lists for validate.py's repair calls.

Moved out of app/nodes/validate.py unchanged (file-size limit); validate.py
re-exports `_correction_note` for existing callers and tests.
"""

from __future__ import annotations

from typing import Any

from app.evaluation.chart_claims import ValidationResult


def _correction_note(result: ValidationResult, fix_language: bool, style_id: str | None,
                     fix_unsupported: bool, fix_placeholder: bool = False,
                     fix_machinery: bool = False, hinglish: dict | None = None) -> str:
    lines = ["CORRECTION REQUIRED (an automatic check wrote this, not the user). Your previous reply has problems:"]
    for claim in result.contradictions:
        lines.append(f"- It {claim.describe()}. Use the chart value.")
    if fix_unsupported:
        for claim in result.unsupported:
            lines.append(f"- It asserts something the chart data does not contain ({claim.kind}: "
                         f"{claim.subject} -> {claim.value}). Remove it or mark it clearly as general astrology.")
    if fix_placeholder:
        lines.append("- It contains an unfilled template slot in <angle brackets>. Replace it with the real "
                     "chart fact from the USER KUNDLI block, or remove it.")
    if fix_machinery:
        lines.append("- It talks about internal machinery (\"chart data\", \"retrieved knowledge\", context or "
                     "documents). Say the same thing in the user's terms, e.g. \"your chart doesn't show this\".")
    if hinglish:
        words = ", ".join(hinglish.get("suspect_transliteration", [])[:8])
        if words:
            lines.append(f"- These words look like English words spelled the Hindi way: {words}. Write every "
                         "English word in its correct English spelling (partner, commitment, relationship, "
                         "emotional connection).")
        if hinglish.get("address_mixed"):
            lines.append("- It mixes \"tum\" and \"aap\". Address the user as \"aap\" throughout.")
    if fix_language:
        if (style_id or "") == "whatsapp_hinglish":
            lines.append("- It is not Roman-script Hinglish. Every sentence must be Hinglish: Hindi grammar and "
                         "function words (hai, mein, aap, ka/ki/ke, ...) with the English astrology terms kept as "
                         "they are, in Latin letters only — no Devanagari, not even in brackets.")
        elif (style_id or "") == "hindi":
            lines.append("- It is not Hindi in Devanagari script. Rewrite every sentence in Devanagari Hindi, "
                         "addressing the user as आप.")
        else:
            lines.append("- It is not English. Rewrite every sentence in natural English — no Hindi or Hinglish "
                         "words, even though earlier messages may have been in Hindi or Hinglish.")
    # Always restate the language: a repair once fixed a placement and drifted
    # from Hinglish into English.
    lines.append("Required language: " + _LANGUAGE_NAME.get(style_id or "", "the user's language") + ".")
    lines.append("Rewrite the reply fixing ONLY these problems. Keep the same meaning, tone and length, "
                 "keep the same paragraphs (one topic each, a blank line between them), "
                 "and use only facts from the USER KUNDLI block. Output only the reply.")
    return "\n".join(lines)


_LANGUAGE_NAME = {
    "whatsapp_hinglish": "Roman-script Hinglish (Hindi and English mixed, Latin letters only, no Devanagari)",
    "hindi": "Hindi in Devanagari script",
    "english": "English", "formal": "English", "detailed": "English",
}


_ABOUT_ITSELF = ("[This question is about the user's OWN birth chart and life. Answer it from their chart. Do not "
                 "talk about yourself, an AI model, or who made you.]")


def _about_itself(draft: str) -> bool:
    """Most of the draft is the model describing itself ("नहीं, मैं एनवीआईडीआईए के
    शोधकर्ताओं द्वारा बनाया गया एक भाषा मॉडल हूँ।" for "क्या मेरी शादी हो चुकी है?" —
    the user's own marriage, read as a question to the bot). A rewrite that
    "keeps the same meaning" only repeats it, so the answer is generated fresh."""
    from app.evaluation.chart_claims import split_sentences
    from app.evaluation.response_quality import _IDENTITY

    sentences = split_sentences(draft)
    return bool(sentences) and 2 * sum(1 for s in sentences if _IDENTITY.search(s)) >= len(sentences)


def _repair_messages(state: dict[str, Any], draft: str, note: str, language_problem: bool) -> list[dict]:
    """A language rewrite drops the earlier turns (see language_repair_messages)."""
    messages = list(state.get("messages") or [])
    if _about_itself(draft) and messages and messages[-1].get("role") == "user":
        question = dict(messages[-1])
        question["content"] = f"{question['content']}\n\n{_ABOUT_ITSELF}"
        return [*messages[:-1], question]
    if language_problem:
        from app.prompts.language import language_repair_messages

        return language_repair_messages(messages, draft, note, state.get("response_style"))
    return [*messages, {"role": "assistant", "content": draft}, {"role": "user", "content": note}]
