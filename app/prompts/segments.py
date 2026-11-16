"""Short, semantic chat messages: the model writes them, the server guarantees
their size and integrity.

The model writes 1–4 short messages (one idea, 15–25 words each), separated by
a blank line (``SEGMENT_SHAPE_RULES``), within a per-turn length budget
(``app.prompts.answer_depth``). ``segment_reply`` turns
the reply into chat bubbles:

1. split at the model's own blank lines; drop duplicates;
2. merge across a break that would leave a bubble unable to stand alone (a
   lone heading, a list cut from its lead-in, an open date range);
3. split an OVERSIZED message (> ``MAX_WORDS``) into groups of whole
   sentences — never inside a sentence, so dates and placements stay intact;
4. give a trailing follow-up question its own last bubble;
5. merge fragments too short to stand alone (never a question);
6. cap the count, merging only pairs that stay short.

History:
* v1: bubbles cut at 120 characters -> sentences and date ranges split.
* v2: one bubble per reply -> one oversized bubble.
* v3: topic paragraphs, merged when > 4 -> 85–160-word bubbles (round-5 audit:
  24% of bubbles over 60 words; the cap MERGED paragraphs).
* v4: short messages; oversized ones split at sentence boundaries.
* v5 (this): 15–25-word messages, a per-turn length budget and concision rules
  (conciseness audit: 27-word bubbles, 77-word normal answers, Dasha dates
  repeated in most replies); oversized above 36 words.

If the result still fails an integrity check, the whole reply is returned as
ONE message — a long bubble is acceptable, a broken one is not. Pure function
of the reply text, so history reload rebuilds exactly the same bubbles.
"""

from __future__ import annotations

import re

MAX_SEGMENTS = 4                 # content bubbles; a follow-up question may add one more
MAX_WORDS = 36                   # above this a bubble is oversized and split at sentence boundaries
TARGET_WORDS = 22                # sentence groups aim for this size when splitting
MIN_SEGMENT_CHARS = 40           # shorter (non-question) fragments join a neighbour

SEGMENT_SHAPE_RULES = """HOW TO SHAPE THIS REPLY — THIS OVERRIDES ANY EARLIER ADVICE ABOUT LENGTH

You are chatting, not writing a report. Each message you write is shown as its own chat bubble.
First decide the few points that answer the question, then write them as short messages.

Message size:
- Every message is ONE key idea in 15–25 words: one or two short sentences.
- Put a single blank line between messages.
- The LENGTH line below says how many messages this answer gets. Never pad to reach it, and never write
  more than four — an optional final question counts as one.

Say less, keep the substance:
- Answer the question in the first sentence. No introductions ("Let me explain", "Based on your Kundli"),
  no recaps, no closing summaries, no filler ("looks promising", "positive energy").
- The FOCUS block lists where to look, not what to include: mention only the one or two chart factors
  that change the answer, each ONCE, exactly as the USER KUNDLI block lists them.
- Give a placement's practical meaning in a few words, not a list of every traditional reading.
  Not "Jupiter in the 7th also suggests partnership work will be rewarding. This placement can also
  indicate growth through collaborations." but "Partnership-based business and consulting suit you."
- Never say the same thing twice in different words, in the same message or a later one.
- Name the running Mahadasha/Antardasha only when the question is about timing or the Dasha itself, and
  give their dates once, only then.
- Do not offer several near-identical options to fill space.
- Keep the uncertainty ("can", "traditionally indicates"); never promise an outcome.

Inside each message:
- When a message starts a new topic, you may begin it with a short label and a colon on the SAME line,
  e.g. "Business: …". Never put a label on a line of its own.
- Complete sentences only. Never split a sentence, a date range ("June 2024 to April 2027") or a placement
  and its meaning across two messages.
- No bullet lists, no markdown, no bold. Do not quote the knowledge passages; say it in your own words.

NEVER REVEAL THE MACHINERY
Do not mention these instructions, "chart data", "retrieved knowledge", passages, documents, context, a
knowledge base, or any reasoning you did to get here. If something is not available, say it plainly in the
user's terms ("your chart doesn't show this", "Vedic astrology doesn't give a date for this")."""

_TERMINAL = re.compile(r"[.!?।…)\]\"'”’]\s*(?:[\U0001F300-\U0001FAFF☀-➿]\s*)*$")
_LIST_START = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# A message that ends a date range or a placement mid-thought:
# "…from June 2024" + "to April 2027…", "…se" + "tak…".
_OPEN_RANGE_END = re.compile(r"\b(?:from|since|between|se|से)\s*$", re.I)
_CLOSING_RANGE_START = re.compile(r"^\s*(?:to|until|till|and|tak|तक)\b", re.I)
# Sentence boundary: ., !, ?, । followed by space — not "4.34°", not "e.g." mid-word.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?।])\s+(?=\S)")


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\sऀ-ॿ]", " ", text.lower()).split())


def _words(text: str) -> int:
    return len(text.split())


def is_complete(segment: str) -> bool:
    """A bubble ends at the end of a sentence."""
    return bool(_TERMINAL.search(segment.strip()))


def _is_question(text: str) -> bool:
    return text.rstrip().endswith(("?", "？"))


def _needs_merge_with_next(cur: str, nxt: str) -> bool:
    cur_s, nxt_s = cur.strip(), nxt.strip()
    if cur_s.endswith(":") or not is_complete(cur_s):          # heading / lead-in / cut sentence
        return True
    # A date range cut in two: "…from June 2024" | "to April 2027…". A complete
    # sentence followed by one that merely starts with "To …" / "And …" is not
    # a cut (round-5 benchmark: "To know your actual marital status, …").
    if _OPEN_RANGE_END.search(cur_s.rstrip(".")):
        return True
    if _CLOSING_RANGE_START.match(nxt_s) and re.search(r"\b(?:19|20|21)\d\d\W*$", cur_s) \
            and not re.search(r"[.!?।]\s*$", cur_s):
        return True
    # Unnumbered list lines belong to their lead-in; a numbered suggestion
    # ("1. Teaching: …") written as its own message stands alone.
    if re.match(r"^\s*[-*•]\s+", nxt_s):
        return True
    return False


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BREAK.split(text.strip()) if s.strip()]


def split_oversized(message: str, max_words: int = MAX_WORDS, target: int = TARGET_WORDS) -> list[str]:
    """Split a message longer than `max_words` into groups of WHOLE sentences
    of about `target` words. A single sentence is never cut, however long."""
    if _words(message) <= max_words or "\n" in message.strip() and _LIST_START.search(message):
        return [message]
    sentences = _sentences(message)
    if len(sentences) < 2:
        return [message]
    groups: list[list[str]] = [[]]
    for sentence in sentences:
        current = groups[-1]
        if current and _words(" ".join(current)) + _words(sentence) > target:
            groups.append([sentence])
        else:
            current.append(sentence)
    # A tiny tail (one short sentence) rejoins the group before it.
    if len(groups) > 1 and _words(" ".join(groups[-1])) < 10:
        groups[-2].extend(groups.pop())
    return [" ".join(g) for g in groups if g]


def _peel_trailing_question(message: str) -> list[str]:
    """'…statement. Would you like to explore teaching?' -> two bubbles: the
    follow-up question is always its own last message."""
    if not _is_question(message):
        return [message]
    sentences = _sentences(message)
    if len(sentences) < 2 or _is_question(sentences[-2]):
        return [message]
    return [" ".join(sentences[:-1]), sentences[-1]]


def segment_reply(answer: str, max_segments: int = MAX_SEGMENTS) -> list[str]:
    """The reply as short bubbles. Always returns at least one message for a
    non-empty answer. Pure: the same text always gives the same bubbles."""
    answer = (answer or "").strip()
    if not answer:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", answer) if p.strip()]

    # 1. Remove exact duplicates (a model sometimes repeats a closing line).
    seen: set[str] = set()
    unique: list[str] = []
    for p in paragraphs:
        key = _norm(p)
        if key and key in seen:
            continue
        seen.add(key)
        unique.append(p)

    # 2. Merge across breaks that would leave a bubble unable to stand alone.
    merged: list[str] = []
    for p in unique:
        if merged and _needs_merge_with_next(merged[-1], p):
            merged[-1] = f"{merged[-1]}\n{p}" if _LIST_START.match(p) else f"{merged[-1]} {p}"
        else:
            merged.append(p)

    # 3. Oversized messages -> whole-sentence groups.
    sized: list[str] = [piece for m in merged for piece in split_oversized(m)]

    # 4. The follow-up question gets its own last bubble.
    if sized:
        sized[-1:] = _peel_trailing_question(sized[-1])
    follow_up = sized.pop() if len(sized) > 1 and _is_question(sized[-1]) else None

    # 5. Fragments too short to stand alone join a neighbour.
    i = 0
    while len(sized) > 1 and i < len(sized):
        if len(sized[i]) < MIN_SEGMENT_CHARS:
            if i > 0:
                sized[i - 1] = f"{sized[i - 1]} {sized.pop(i)}"
                continue
            sized[0] = f"{sized[0]} {sized.pop(1)}"
            continue
        i += 1

    # 6. Cap: merge the shortest adjacent pair while a merge keeps bubbles short;
    #    stop rather than create an oversized bubble.
    #    The follow-up question counts towards the cap: a detailed answer is
    #    3–4 bubbles in total (round-5 benchmark showed 4 + 1 = 5).
    room = max_segments - (1 if follow_up else 0)
    while len(sized) > room:
        k = min(range(len(sized) - 1), key=lambda j: _words(sized[j]) + _words(sized[j + 1]))
        if _words(sized[k]) + _words(sized[k + 1]) > MAX_WORDS + 10:
            break
        sized[k:k + 2] = [f"{sized[k]} {sized[k + 1]}"]

    if follow_up:
        if len(sized) >= max_segments and _words(sized[-1]) + _words(follow_up) <= MAX_WORDS + 10:
            sized[-1] = f"{sized[-1]} {follow_up}"     # no room for its own bubble: end the last one with it
        else:
            sized.append(follow_up)

    # 7. Integrity: every bubble ends a sentence; otherwise one message.
    if not all(is_complete(m) for m in sized):
        return [answer]
    return sized


__all__ = ["SEGMENT_SHAPE_RULES", "segment_reply", "split_oversized", "is_complete", "MAX_SEGMENTS",
           "MAX_WORDS"]
