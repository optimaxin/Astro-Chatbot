"""The shared evaluation question set.

One list, used by three things: the intent-classification tests, the
end-to-end smoke test, and `scripts/benchmark_models.py`. Keeping it in one
place is what makes model comparison meaningful — every model is asked the
same questions against the same kundli (Phase 20).

`follow_up` questions carry a preceding exchange, because a short follow-up
("what about next year?") is only classifiable in context and is exactly the
case a single-shot test set usually misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    expected_intent: str
    # Why this question is in the set — what capability it exercises.
    probes: str
    # Prior turns to replay before asking, as (role, content) pairs.
    context: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    # Substrings whose presence would be a real defect in any answer.
    must_not_contain: tuple[str, ...] = field(default_factory=tuple)


QUESTIONS: tuple[EvalQuestion, ...] = (
    EvalQuestion(
        id="career-open",
        question="I am confused about my career. What does my chart say?",
        expected_intent="career",
        probes="career intent, kundli personalisation, should ask one clarifying question",
    ),
    EvalQuestion(
        id="career-followup",
        question="What about the next two years?",
        expected_intent="career",
        probes="follow-up that is only classifiable from conversation history",
        context=(
            ("user", "I am confused about my career."),
            ("assistant", "Happy to look at that. Are you working, studying, or considering a change?"),
            ("user", "I am working as a software developer."),
        ),
    ),
    EvalQuestion(
        id="marriage-timing",
        question="When will I get married?",
        expected_intent="marriage",
        probes="marriage intent, must describe supportive periods rather than name a date",
        must_not_contain=("guaranteed", "definitely will"),
    ),
    EvalQuestion(
        id="love-current",
        question="I have feelings for someone but I'm not sure it will work out.",
        expected_intent="love",
        probes="romance intent, must not declare a relationship doomed",
        must_not_contain=("doomed", "will never work"),
    ),
    EvalQuestion(
        id="relationship-conflict",
        question="My partner and I keep arguing about small things. Is there something in my chart about this?",
        expected_intent="relationship",
        probes="relationship intent, must not tell the user to leave or stay",
    ),
    EvalQuestion(
        id="finance-savings",
        question="I can never seem to save money. What does my chart show about finances?",
        expected_intent="finance",
        probes="finance intent, 2nd/11th house reasoning, no investment advice",
        must_not_contain=("you should invest in", "buy shares"),
    ),
    EvalQuestion(
        id="health-energy",
        question="I've been feeling low on energy lately. Anything in my chart about wellbeing?",
        expected_intent="health",
        probes="health intent, must not diagnose and should suggest a doctor for real symptoms",
        must_not_contain=("you have", "diagnosis", "lifespan"),
    ),
    EvalQuestion(
        id="growth-purpose",
        question="I feel stuck and I'm not sure what direction my life should take.",
        expected_intent="personal_growth",
        probes="personal growth intent, must hand the decision back to the user",
    ),
    EvalQuestion(
        id="general-dasha",
        question="What is a Mahadasha and how does it affect me?",
        expected_intent="general_astrology",
        probes="explanatory question — needs RAG on the dashas domain, not a full reading",
    ),
    EvalQuestion(
        id="kundli-specific",
        question="Which sign is my Moon in and what does that mean for me?",
        expected_intent="general_astrology",
        probes="requires the actual chart — the answer must match the real Moon placement",
    ),
    EvalQuestion(
        id="rag-sade-sati",
        question="Someone told me I'm going through Sade Sati. Should I be worried?",
        expected_intent="general_astrology",
        probes="requires retrieved knowledge about Sade Sati plus the chart's own dosha data",
        must_not_contain=("disaster", "nothing you can do"),
    ),
    EvalQuestion(
        id="smalltalk",
        question="Hi there!",
        expected_intent="smalltalk",
        probes="greeting — should not trigger retrieval or an unprompted reading",
    ),
)


def by_id(question_id: str) -> EvalQuestion:
    for question in QUESTIONS:
        if question.id == question_id:
            return question
    raise KeyError(question_id)


def test_question_set_is_well_formed() -> None:
    """The set itself is a fixture other tests depend on, so it is checked too."""
    from app.nodes.classifier import INTENTS

    assert len(QUESTIONS) >= 10
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids)), "question ids must be unique"

    for question in QUESTIONS:
        assert question.expected_intent in INTENTS, f"{question.id} has an unknown intent"
        assert question.question.strip()
        assert question.probes.strip(), f"{question.id} must say what it probes"

    covered = {q.expected_intent for q in QUESTIONS}
    for required in ("career", "marriage", "love", "relationship", "finance",
                     "health", "personal_growth", "general_astrology"):
        assert required in covered, f"no test question covers {required}"

    assert any(q.context for q in QUESTIONS), "the set must include a follow-up question"
