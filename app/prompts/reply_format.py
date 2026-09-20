"""How a reply is shaped for a chat bubble UI, and how it is split back out.

The two halves of one contract live together on purpose: the instruction that
asks the model for several short messages, and the splitter that turns its
answer back into that list. If they drift apart the UI silently degrades to
one long bubble, and a reviewer should not have to read two files to notice.

The model is asked to separate messages with a blank line rather than a custom
token (`<<<SPLIT>>>` and friends). A blank line is something a chat model
already produces reliably, it survives a model swap, and when the model
ignores the instruction entirely the fallback below still produces something
sensible instead of leaking a marker into the user's face.
"""

from __future__ import annotations

import re

# Appended to the system prompt only when the caller wants bubbles. The
# terminal REPL does not, so it keeps the single-block answer it always had.
CHAT_BUBBLE_STYLE = """HOW TO FORMAT THIS REPLY
- You are replying inside a small chat window, so write the way a person texts, not the way a report reads.
- Break your answer into 2 to 4 short messages. Put a blank line between each message.
- Each message is one or two sentences. Never write a long paragraph.
- Plain, everyday words. If you use an astrology term, explain it in half a sentence.
- No headings, no bullet points, no numbered lists, no markdown.
- Do not label the messages. Just write them, separated by blank lines."""

# A bubble longer than this reads as a wall of text in a 360px-wide panel, so
# an over-long single message is split again on sentence boundaries.
MAX_BUBBLE_CHARS = 320
# Below this, a trailing fragment is merged back into the previous bubble
# rather than standing alone as a stray "Yes." bubble.
MIN_BUBBLE_CHARS = 25
MAX_BUBBLES = 5


def _split_long(text: str) -> list[str]:
    """Break an over-long message on sentence boundaries, never mid-sentence."""
    if len(text) <= MAX_BUBBLE_CHARS:
        return [text]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > MAX_BUBBLE_CHARS and current:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def split_into_messages(answer: str) -> list[str]:
    """Turn one model answer into the list of bubbles the UI renders.

    Always returns at least one message for a non-empty answer — a UI that
    receives an empty list would show nothing at all, which is a worse failure
    than showing one long bubble.
    """
    answer = (answer or "").strip()
    if not answer:
        return []

    # Preferred path: the model followed the instruction and used blank lines.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", answer) if b.strip()]

    # Fallback: it returned one block anyway. Single newlines are the next
    # best boundary; failing that, `_split_long` cuts on sentences.
    if len(blocks) == 1:
        lines = [ln.strip() for ln in blocks[0].split("\n") if ln.strip()]
        if len(lines) > 1:
            blocks = lines

    # Strip list markers the model may have added despite being asked not to —
    # they render as a literal "- " inside a bubble.
    blocks = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", b) for b in blocks]

    # Fold a too-short block into the one before it, so a stray "Yes." does
    # not get a bubble of its own.
    merged: list[str] = []
    for block in blocks:
        if merged and len(block) < MIN_BUBBLE_CHARS:
            merged[-1] = f"{merged[-1]} {block}"
        else:
            merged.append(block)

    # The cap limits how many *blocks the model chose to write* — it is there
    # to stop bubble spam, not to bound length. Overflow is joined rather than
    # dropped, because dropping it would silently lose the model's words.
    if len(merged) > MAX_BUBBLES:
        head, tail = merged[: MAX_BUBBLES - 1], merged[MAX_BUBBLES - 1 :]
        merged = [*head, " ".join(tail)]

    # Size-splitting comes last, so no bubble can exceed the readable budget —
    # including the joined overflow above. This may push the final count past
    # MAX_BUBBLES, which is the right trade: several readable bubbles beat one
    # unreadable wall of text.
    messages: list[str] = []
    for block in merged:
        messages.extend(_split_long(block))

    return [m for m in messages if m]


__all__ = ["CHAT_BUBBLE_STYLE", "split_into_messages", "MAX_BUBBLES", "MAX_BUBBLE_CHARS"]
