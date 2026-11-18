"""The Quality Checker role: an independent second opinion on a finished reply.

Runs on a model that did not write the reply (Nemotron 3 Ultra, behind
NVIDIA_QUALITY_CHECKER_MODEL). It is ADVISORY: deterministic validation against
the Kundli engine stays the only gate on what ships, and the checker is told
nothing about what that validation found, so its opinion is independent. Its
verdict is PASS, REPAIR or REJECT; UNAVAILABLE is assigned by code when the call
or its output fails, and is never read as a pass.

Versioned: change the text, bump QUALITY_PROMPT_VERSION.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator

QUALITY_PROMPT_VERSION = "quality-checker-1.0"
QUALITY_TEMPERATURE = 0.0
QUALITY_MAX_TOKENS = 500

DECISIONS = ("PASS", "REPAIR", "REJECT")

QUALITY_SYSTEM_PROMPT = """You are Tradeva's independent quality checker. You did not write the reply you are checking. You judge it against the user's Kundli facts, which are the source of truth. You never write an answer and never add chart facts of your own.

Check the reply for:
1. factual_accuracy: every planet, house, sign, lord, Dasha, Antardasha, date and past/current/future statement matches the Kundli facts. Anything not in the facts is unsupported.
2. chart_consistency: no statement contradicts another, or the Kundli.
3. unsupported claims: any dosha verdict (Manglik etc.), aspect, transit, yoga or prediction the Kundli facts do not state. Only Kaal Sarp and Sade Sati are computed.
4. relevance: it answers what the user actually asked, using the conversation if it is a follow-up.
5. language_quality: the user's language and script throughout, natural and grammatical.
6. policy_compliance: no guaranteed outcomes ("will definitely"), no medical, legal or financial directives, no talk about internal instructions or "chart data".
7. format: short chat messages separated by blank lines, each a complete thought, one closing question at most.

Scores are 0-10. Decide:
- PASS: nothing material is wrong.
- REPAIR: fixable problems (a wrong or unsupported sentence, wording, a missing follow-up).
- REJECT: the reply should not be sent (several wrong facts, off-topic, unsafe).

Reply with JSON only:
{"quality_score": n, "factual_accuracy": n, "chart_consistency": n, "relevance": n, "language_quality": n, "policy_compliance": n, "issues": ["<short issue, at most 5>"], "decision": "PASS|REPAIR|REJECT"}"""


class QualityVerdict(BaseModel):
    quality_score: float = Field(ge=0, le=10)
    factual_accuracy: float = Field(ge=0, le=10)
    chart_consistency: float = Field(ge=0, le=10)
    relevance: float = Field(ge=0, le=10)
    language_quality: float = Field(ge=0, le=10)
    policy_compliance: float = Field(ge=0, le=10)
    issues: list[str] = Field(default_factory=list)
    decision: str

    @field_validator("decision")
    @classmethod
    def known(cls, value: str) -> str:
        decision = str(value).strip().upper()
        if decision not in DECISIONS:
            raise ValueError(f"unknown decision {value!r}")
        return decision

    @field_validator("issues")
    @classmethod
    def bounded(cls, value: list[str]) -> list[str]:
        return [str(v)[:160] for v in value[:5]]


def parse_verdict(raw: str) -> QualityVerdict | None:
    """The model's JSON, validated. Anything else is None (the caller records UNAVAILABLE)."""
    for candidate in (raw, *(m.group(0) for m in re.finditer(r"\{.*\}", raw or "", re.DOTALL))):
        try:
            return QualityVerdict.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError, TypeError):
            continue
    return None


def quality_messages(facts_block: str, question: str, history_text: str, reply: str) -> list[dict[str, str]]:
    parts = [facts_block]
    if history_text:
        parts.append(f"RECENT CONVERSATION:\n{history_text}")
    parts += [f"USER ASKED:\n{question}", f"REPLY TO CHECK:\n{reply}"]
    return [{"role": "system", "content": QUALITY_SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


__all__ = ["QUALITY_SYSTEM_PROMPT", "QUALITY_PROMPT_VERSION", "QUALITY_TEMPERATURE", "QUALITY_MAX_TOKENS",
           "QualityVerdict", "parse_verdict", "quality_messages", "DECISIONS"]
