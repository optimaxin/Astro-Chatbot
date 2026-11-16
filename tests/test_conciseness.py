"""Conciseness pass (2026-11): per-turn length budget, 15–25-word bubbles,
safe redundancy removal. Offline; the live benchmark is
scripts/kundli_quality_bench.py."""

from __future__ import annotations

import pytest

from app.evaluation.conciseness import drop_repeated_date_ranges, redundant_pairs, restated_sentences
from app.evaluation.response_quality import apply_safe_fixes
from app.prompts.answer_depth import BUDGETS, classify_depth, length_line
from app.prompts.segments import MAX_WORDS, SEGMENT_SHAPE_RULES, segment_reply


# ── depth: decided in code, one budget per turn ───────────────────────
@pytest.mark.parametrize("question,intent,depth", [
    ("What is my Lagna?", "general_astrology", "simple"),
    ("Mera lagna kya hai?", "general_astrology", "simple"),
    ("मेरी राशि क्या है?", "general_astrology", "simple"),
    ("Jupiter kis house mein hai?", "general_astrology", "simple"),
    ("theek hai", "relationship", "simple"),
    ("I want you to reply in english", "career", "simple"),
    ("What is my current Dasha about?", "general_astrology", "normal"),
    ("Who is the lord of my 10th house and where is it placed?", "career", "normal"),
    ("So is my 7th lord strong?", "marriage", "normal"),
    ("Tell me about my career.", "career", "detailed"),
    ("Mera career kaisa rahega?", "career", "detailed"),
    ("मुझे मेरी शादी के बारे में बताइए।", "marriage", "detailed"),
    ("Give me a detailed career analysis with timing.", "career", "complex"),
    ("Mera partner kaisa hoga aur shaadi kab hogi? Detail mein batao", "marriage", "complex"),
    ("मेरी शादी का पूरा विश्लेषण बताइए — योग, जीवनसाथी, समय और दशा।", "marriage", "complex"),
])
def test_answer_depth(question: str, intent: str, depth: str) -> None:
    switch = question.startswith("I want you to reply")
    assert classify_depth(question, intent, switch_only=switch) == depth


def test_a_tag_question_is_not_a_second_question() -> None:
    assert classify_depth("My Mercury is in the 7th house, right? What does that mean?", "general_astrology") \
        == "normal"


def test_budgets_match_the_brief() -> None:
    assert BUDGETS["simple"].words == "15–30 words" and BUDGETS["normal"].words.startswith("30–60")
    assert BUDGETS["detailed"].words.startswith("60–100") and BUDGETS["complex"].words.startswith("80–120")
    for depth in BUDGETS:
        line = length_line(depth)
        assert "15–25 words" in line and "never leave out a chart fact" in line


def test_the_length_line_reaches_the_prompt(kundli) -> None:
    from app.prompts.builder import build_system_prompt
    from app.rag.retriever import RetrievalResult

    system = build_system_prompt("career", kundli, RetrievalResult.unavailable("x"), chat_bubbles=True,
                                 response_style="english", length_line=length_line("detailed"))
    assert "LENGTH FOR THIS ANSWER: THREE messages" in system
    assert system.index("LENGTH FOR THIS ANSWER") > system.index("HOW TO SHAPE THIS REPLY")


def test_the_length_is_restated_on_the_question_once_there_is_history(kundli) -> None:
    from app.memory.redis_memory import Message
    from app.prompts.answer_depth import turn_length_note
    from app.prompts.builder import build_messages
    from app.prompts.language import leaks_prompt
    from app.rag.retriever import RetrievalResult

    note = turn_length_note("detailed")
    assert "THREE messages" in note and "60–100 words" in note and leaks_prompt(note)
    retrieval = RetrievalResult.unavailable("x")
    fresh = build_messages("What about my career?", "career", kundli, retrieval, [], chat_bubbles=True,
                           response_style="english", length_note=note)
    assert fresh[-1]["content"] == "What about my career?"          # first turn: system prompt only
    history = [Message(role="user", content="What is my Dasha?", timestamp=0.0),
               Message(role="assistant", content="You are in Mercury Mahadasha. " * 20, timestamp=1.0)]
    later = build_messages("What about my career?", "career", kundli, retrieval, history, chat_bubbles=True,
                           response_style="english", length_note=note)
    assert later[-1]["content"] == f"What about my career?\n\n{note}"
    assert later[2]["content"] == history[1].content                 # stored history unchanged


def test_the_shape_rules_carry_the_concision_rules() -> None:
    for phrase in ("ONE key idea in 15–25 words", "Never say the same thing twice",
                   "only when the question is about timing or the Dasha itself",
                   "lists where to look, not what to include", "Partnership-based business and consulting"):
        assert phrase in SEGMENT_SHAPE_RULES, phrase


# ── segmentation keeps working at the new size ────────────────────────
def test_a_40_word_message_is_split_between_sentences() -> None:
    text = ("Your 10th house is Sagittarius and its lord Jupiter sits in the 6th house. "
            "Traditionally this favours service-oriented work such as medicine, law or finance, where steady "
            "effort, patience and problem-solving are rewarded slowly over the coming years.")
    assert len(text.split()) > MAX_WORDS
    bubbles = segment_reply(text)
    assert len(bubbles) == 2 and " ".join(bubbles) == text


def test_a_long_single_sentence_is_never_cut() -> None:
    text = ("Your running Mercury Mahadasha from January 2021 to January 2038 together with the Venus Antardasha "
            "from June 2024 to April 2027 is traditionally linked with partnership, communication and service "
            "work in your chart.")
    assert segment_reply(text) == [text]


def test_short_answers_stay_in_one_bubble() -> None:
    assert segment_reply("Your Lagna is Pisces, ruled by Jupiter in the 6th house.") == \
        ["Your Lagna is Pisces, ruled by Jupiter in the 6th house."]


def test_detailed_answer_keeps_three_to_four_bubbles() -> None:
    text = ("Your 10th house is Sagittarius, and its lord Jupiter sits in the 6th house.\n\n"
            "Service-oriented work such as medicine, law or finance suits this placement.\n\n"
            "Growth comes through steady effort and a reputation for solving problems.\n\n"
            "Would you like to look at business or job options next?")
    bubbles = segment_reply(text)
    assert 3 <= len(bubbles) <= 4 and bubbles[-1].endswith("?")


# ── redundancy: only the safe cases are removed ───────────────────────
def test_a_restatement_with_no_new_fact_is_removed() -> None:
    text = ("Venus in your 7th house supports harmony and affection in partnership. "
            "Venus in your 7th house also supports affection and harmony in partnership. "
            "Saturn in the 4th asks for patience at home.")
    assert restated_sentences(text) == [1]
    fixed, report = apply_safe_fixes(text, "english", "marriage")
    assert "also supports" not in fixed and "Saturn in the 4th" in fixed
    assert report.restated_removed == 1


def test_a_similar_sentence_with_a_new_chart_fact_is_kept() -> None:
    text = ("Venus in your 7th house supports harmony and affection in partnership. "
            "Jupiter in your 7th house also supports harmony and affection in partnership.")
    assert restated_sentences(text) == []


def test_paraphrases_are_left_to_the_prompt_not_deleted() -> None:
    text = ("Jupiter in the 7th house suggests consulting or partnership-based work will be rewarding. "
            "This placement can also indicate growth through business collaborations.")
    assert restated_sentences(text) == []


def test_a_dasha_range_is_given_once_per_reply() -> None:
    text = ("You are in Mercury Mahadasha (January 2021 to January 2038) with Venus Antardasha "
            "(June 2024 to April 2027). The Venus Antardasha (June 2024 to April 2027) favours partnership.")
    fixed, removed = drop_repeated_date_ranges(text)
    assert removed == 1 and fixed.count("(June 2024 to April 2027)") == 1
    assert "(January 2021 to January 2038)" in fixed


def test_dates_already_given_are_dropped_unless_timing_is_asked() -> None:
    previous = "You are in Mercury Mahadasha (January 2021 to January 2038)."
    text = "Mercury Mahadasha (January 2021 to January 2038) favours communication work."
    fixed, removed = drop_repeated_date_ranges(text, previous, "How is my career?")
    assert removed == 1 and fixed == "Mercury Mahadasha favours communication work."
    kept, removed = drop_repeated_date_ranges(text, previous, "When does this period end?")
    assert removed == 0 and kept == text


def test_hindi_date_ranges() -> None:
    text = "बुध की महादशा (जनवरी 2021 से जनवरी 2038) चल रही है। बुध की महादशा (जनवरी 2021 से जनवरी 2038) शुभ है।"
    fixed, removed = drop_repeated_date_ranges(text)
    assert removed == 1 and fixed.count("2038") == 1


def test_redundant_pairs_measure() -> None:
    assert redundant_pairs("Venus in the 7th supports harmony and affection in partnership. "
                           "Venus supports affection and harmony in partnership life.") == 1
    assert redundant_pairs("Your Lagna is Pisces. Saturn sits in the 4th house of home.") == 0


# ── follow-up stays short ──────────────────────────────────────────────
def test_a_long_follow_up_is_not_kept() -> None:
    from app.prompts.followup import is_good_followup

    assert is_good_followup("Would you like to explore business or job options next?")
    assert not is_good_followup("Would you like me to explain in more detail how the Venus Antardasha and the "
                                "Mercury Mahadasha together affect your business and job options?")


def test_every_trailing_question_falls_under_the_follow_up_decision() -> None:
    # Not allowed: even a short "statement + question" loses the question.
    fixed, _ = apply_safe_fixes("Achha, theek hai. Samajh aaya?", "hinglish", "relationship",
                                followup_allowed=False)
    assert fixed == "Achha, theek hai."
    # Allowed but a probe about the user's plans, not an offer: removed.
    fixed, _ = apply_safe_fixes("Career ke liye service field achha hai. "
                                "Kya aap service ya legal field mein career options dekh rahe hain?",
                                "hinglish", "career", followup_allowed=True)
    assert not fixed.endswith("?")
    # Allowed and a specific offer: kept.
    fixed, report = apply_safe_fixes("Career ke liye service field achha hai. "
                                     "Kya aap business ya job options jaanna chahenge?",
                                     "hinglish", "career", followup_allowed=True)
    assert fixed.endswith("jaanna chahenge?") and report.follow_up_kept == 1


def test_a_simple_answer_becomes_one_bubble() -> None:
    from app.evaluation.conciseness import join_simple_answer

    text = "Aapki Moon sign Pisces hai.\n\nMoon aapke 1st house mein hai."
    joined = join_simple_answer(text, "simple")
    assert segment_reply(joined) == ["Aapki Moon sign Pisces hai. Moon aapke 1st house mein hai."]
    assert join_simple_answer(text, "normal") == text


def test_the_dasha_follow_up_suggestion_names_no_period() -> None:
    from app.prompts.followup import decide

    decision = decide("Meri dasha ka career par kya asar hai?", "general_astrology", "hinglish")
    assert "agli" not in decision.suggestion.lower() and "next" not in decision.suggestion.lower()
