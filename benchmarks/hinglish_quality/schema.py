"""Data model of the Hinglish Quality Benchmark.

Every example is a chatbot-style RESPONSE (with the question it answers):

* kind="gold"     — a natural, correct reply; the evaluator must not flag it;
* kind="negative" — a reply with known problems, annotated with each error
                    (category, the wrong span, the fix) and its corrected
                    alternative, which is itself a gold reference.

Chart facts in the astrology examples follow the benchmark chart
(scripts/kundli_quality_bench.screenshot_chart): Pisces Lagna, Moon in the
1st, Mars in the 2nd (Aries), Rahu 3rd, Saturn 4th (Gemini), Sun + Mercury +
Jupiter in the 6th (Leo), Venus 7th (Virgo), Ketu 9th; Mercury Mahadasha
January 2021 – January 2038, Venus Antardasha June 2024 – April 2027.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

# Error taxonomy -> the dimension it lowers.
CATEGORIES: dict[str, str] = {
    # grammar
    "gender_agreement": "grammar", "number_agreement": "grammar", "postposition": "grammar",
    "verb_form": "grammar", "sentence_structure": "grammar",
    # vocabulary
    "spelling": "vocabulary", "transliteration": "vocabulary", "non_word": "vocabulary",
    "word_choice": "vocabulary", "collocation": "vocabulary",
    # naturalness
    "literal_translation": "naturalness", "heavy_register": "naturalness", "word_order": "naturalness",
    "verbose": "naturalness", "filler": "naturalness",
    # meaning preservation
    "fact_change": "meaning", "fact_dropped": "meaning", "certainty": "meaning", "unsafe_claim": "meaning", "machinery": "meaning",
    # language consistency
    "register_mix": "consistency", "gender_assumption": "consistency", "script_mixing": "consistency",
    "language_switch": "consistency",
}
DIMENSIONS = ("grammar", "naturalness", "vocabulary", "meaning", "consistency")
LANGUAGES = ("hinglish", "hindi", "english")


@dataclass(frozen=True)
class Error:
    category: str
    span: str
    fix: str = ""

    @property
    def dimension(self) -> str:
        return CATEGORIES[self.category]


@dataclass(frozen=True)
class Example:
    id: str
    language: str                 # hinglish | hindi | english
    domain: str                   # astrology | conversational
    topic: str                    # career | marriage | health | property | dasha | placement | chat
    question: str
    text: str
    kind: str                     # gold | negative
    corrected: str | None = None
    errors: tuple[Error, ...] = ()
    source: str = "authored"      # screenshot | benchmark-output | brief | authored
    review: str = "author"        # author | native:<reviewer>  (see reviews.json)
    split: str = field(default="")

    @property
    def reference(self) -> str:
        """The gold text this example's meaning is checked against."""
        return self.corrected if self.kind == "negative" and self.corrected else self.text


def assign_split(example: Example) -> Example:
    """dev | holdout. Deterministic (crc32 of the id), so a run is repeatable.
    About a third of the negatives are held out from rule development; the
    screenshot examples are always dev (they are the acceptance cases)."""
    if example.split:
        return example
    held = example.kind == "negative" and example.source != "screenshot" and zlib.crc32(example.id.encode()) % 3 == 0
    return Example(**{**example.__dict__, "split": "holdout" if held else "dev"})


def builder(language: str, domain: str):
    """g(...) for gold and n(...) for negative examples of one language/domain."""

    def g(id_: str, topic: str, question: str, text: str, source: str = "authored") -> Example:
        return Example(id_, language, domain, topic, question, text, "gold", source=source)

    def n(id_: str, topic: str, question: str, wrong: str, corrected: str,
          errors: list[tuple[str, str, str]], source: str = "authored") -> Example:
        for category, _, _ in errors:
            assert category in CATEGORIES, category
        return Example(id_, language, domain, topic, question, wrong, "negative", corrected,
                       tuple(Error(*e) for e in errors), source=source)

    return g, n
