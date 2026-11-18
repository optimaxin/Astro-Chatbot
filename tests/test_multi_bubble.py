"""Multi-bubble conversational replies (2026-11-18 product brief).

An advice question is answered in about 4–5 chat bubbles — understanding,
chart insight, more insight, practical guidance, then an optional follow-up —
and a simple question in fewer. The formatter only re-cuts the validated
answer: these tests pin the shape AND that no bubble cut can change a fact.
Ground truth is the sample chart (Lagna Virgo; 10th Gemini, lord Mercury in
the 12th; Moon in the 7th; Jupiter Scorpio/3rd; Venus in the 11th).
"""

from __future__ import annotations

import re

import pytest

from app.kundli.chart_facts import build_chart_facts
from app.prompts.answer_depth import classify_depth, length_line
from app.prompts.followup import decide, prompt_line
from app.prompts.reply_format import reply_messages
from app.prompts.segments import MAX_WORDS, is_complete, segment_reply

ENGLISH_MARRIAGE = (
    "I understand you're hoping for clarity on marriage, and your chart shows a supportive picture rather than a "
    "blocked one.\n\n"
    "The Moon sits in your 7th house of marriage, which traditionally points to an emotionally caring partner.\n\n"
    "Your 7th lord Jupiter is in your 3rd house, so a relationship can grow through conversations, short travel "
    "or your social circle.\n\n"
    "Keep meeting people through shared interests, and give a promising relationship time to deepen before you "
    "decide.\n\n"
    "Would you like to look at your partner's nature next?")

HINGLISH_CAREER = (
    "Career ko lekar aapki chinta samajh aati hai, aur aapki kundli mein mehnat ka achha result dikh raha hai.\n\n"
    "Aapka 10th house Gemini hai aur uska lord Mercury 12th house mein hai, jo analytical ya behind-the-scenes "
    "kaam ki taraf ishaara karta hai.\n\n"
    "Venus 11th house mein hai, jo network aur dosto ke through income badhne ka sanket deta hai.\n\n"
    "Abhi apni skills par focus kijiye aur ek-do achhe contacts ke saath regular baat banaye rakhiye.\n\n"
    "Kya aap jaanna chahenge ki business aapke liye kaisa rahega?")

HINDI_FINANCE = (
    "धन को लेकर आपकी चिंता स्वाभाविक है, और आपकी कुंडली में धीरे-धीरे स्थिरता बनने के संकेत हैं।\n\n"
    "शुक्र आपके ग्यारहवें भाव में है, जो आय और लाभ का भाव है, इसलिए मेल-जोल से कमाई के अवसर मिल सकते हैं।\n\n"
    "बृहस्पति आपके तीसरे भाव में है, जो अपनी मेहनत और पहल से आगे बढ़ने का संकेत देता है।\n\n"
    "हर महीने थोड़ी बचत तय कीजिए और जल्दबाज़ी में बड़ा निवेश करने से बचिए।\n\n"
    "क्या आप जानना चाहेंगे कि आय बढ़ने का समय कब अनुकूल है?")

ENGLISH_RELATIONSHIP = (
    "It's natural to want more peace in your relationships, and your chart suggests you value emotional "
    "closeness.\n\n"
    "The Moon in your 7th house of partnership makes you sensitive to how a partner treats you.\n\n"
    "Mercury, your Lagna lord, sits in your 12th house, so you may hold back your feelings more than you "
    "realise.\n\n"
    "Try saying what you need early and calmly, before small hurts build up.\n\n"
    "Would you like to explore relationship compatibility next?")

ADVICE = [("english", ENGLISH_MARRIAGE, "marriage", "When will I get married?"),
          ("whatsapp_hinglish", HINGLISH_CAREER, "career", "Meri career growth kaisi rahegi?"),
          ("hindi", HINDI_FINANCE, "finance", "मेरी आर्थिक स्थिति कैसी रहेगी?"),
          ("english", ENGLISH_RELATIONSHIP, "relationship", "How will my relationships be?")]


@pytest.fixture
def facts(kundli):
    return build_chart_facts(kundli)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s.strip()]


# ── the shape ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("style,answer,intent,question", ADVICE, ids=["en-marriage", "hinglish-career",
                                                                       "hindi-finance", "en-relationship"])
def test_an_advice_answer_is_four_to_five_concise_bubbles(style, answer, intent, question, facts) -> None:
    bubbles = reply_messages(answer, facts)
    assert 4 <= len(bubbles) <= 5
    for bubble in bubbles:
        assert is_complete(bubble), bubble
        assert 1 <= len(_sentences(bubble)) <= 3 and len(bubble.split()) <= MAX_WORDS, bubble


@pytest.mark.parametrize("style,answer,intent,question", ADVICE, ids=["en", "hinglish", "hindi", "en-rel"])
def test_each_bubble_keeps_the_users_language(style, answer, intent, question, facts) -> None:
    from app.nodes.validate import _segment_language_ok

    for bubble in reply_messages(answer, facts):
        assert _segment_language_ok(style, bubble), bubble


@pytest.mark.parametrize("question,intent", [
    ("When will I get married?", "marriage"), ("Meri career growth kaisi rahegi?", "career"),
    ("How will my relationships be?", "relationship"), ("मेरी आर्थिक स्थिति कैसी रहेगी?", "finance"),
])
def test_advice_questions_are_asked_for_the_four_part_structure(question, intent) -> None:
    depth = classify_depth(question, intent)
    line = length_line(depth, question)
    assert depth == "detailed" and "FOUR messages" in line
    for part in ("direct answer", "most important chart fact", "one more relevant", "practical guidance"):
        assert part in line, part


def test_the_shape_prompt_states_the_product_balance() -> None:
    from app.prompts.segments import SEGMENT_SHAPE_RULES

    assert "60% helpful human guidance, 40% meaningful astrology" in SEGMENT_SHAPE_RULES
    assert "never write more than five messages in total" in SEGMENT_SHAPE_RULES
    assert "no labels or headings" in SEGMENT_SHAPE_RULES and "Explain every astrology term" in SEGMENT_SHAPE_RULES
    assert 'begin it with a short label' not in SEGMENT_SHAPE_RULES          # report-style labels are gone


# ── simple questions get fewer bubbles ────────────────────────────────
@pytest.mark.parametrize("question,intent,depth,messages", [
    ("What is my Lagna?", "general_astrology", "simple", "ONE message"),
    ("Mera lagna kya hai?", "general_astrology", "simple", "ONE message"),
    ("What does my 10th house lord say about work?", "career", "normal", "TWO or THREE messages"),
])
def test_simple_questions_are_asked_for_fewer_bubbles(question, intent, depth, messages) -> None:
    assert classify_depth(question, intent) == depth
    assert f"LENGTH FOR THIS ANSWER: {messages}" in length_line(depth, question)


def test_a_simple_answer_stays_one_bubble_and_a_normal_one_two_or_three(facts) -> None:
    assert reply_messages("Your Lagna is Virgo, which traditionally gives a practical, careful nature.", facts) == [
        "Your Lagna is Virgo, which traditionally gives a practical, careful nature."]
    normal = ("Your 10th house is Gemini and its lord Mercury sits in your 12th house.\n\n"
              "That traditionally points to analytical, research or behind-the-scenes work, often with foreign links.")
    assert len(reply_messages(normal, facts)) == 2


# ── bubbles are semantic units ────────────────────────────────────────
@pytest.mark.parametrize("style,answer,intent,question", ADVICE, ids=["en", "hinglish", "hindi", "en-rel"])
def test_no_sentence_is_split_between_bubbles_and_nothing_is_lost(style, answer, intent, question, facts) -> None:
    bubbles = reply_messages(answer, facts)
    assert [s for b in bubbles for s in _sentences(b)] == _sentences(answer)


def test_an_oversized_message_is_cut_only_between_whole_sentences() -> None:
    long = ("Your 7th lord Jupiter sits in your 3rd house, which traditionally ties relationships to communication. "
            "That means a partner may come through conversations, writing, short trips or your siblings' circle. "
            "It also suggests that talking openly and honestly keeps a relationship healthy and close over the "
            "years ahead.")
    assert len(long.split()) > MAX_WORDS
    bubbles = segment_reply(long)
    assert len(bubbles) >= 2 and " ".join(bubbles) == long and all(is_complete(b) for b in bubbles)
    assert all(b in _sentences(long) or set(_sentences(b)) <= set(_sentences(long)) for b in bubbles)


def test_a_date_range_and_a_list_never_break_across_bubbles() -> None:
    assert len(segment_reply("Your Venus Antardasha runs from June 2024\n\nto April 2027, a supportive period.")) == 1
    listed = "These help your career:\n\n- discipline in daily work\n- one mentor you trust"
    assert len(segment_reply(listed)) == 1


def test_repeated_content_is_not_shown_twice(facts) -> None:
    answer = ("The Moon sits in your 7th house of marriage, which traditionally points to a caring partner.\n\n"
              "Your 7th lord Jupiter is in your 3rd house, which supports meeting someone through your circle.\n\n"
              "Traditionally the Moon in your 7th house of marriage points to a caring partner.\n\n"
              "The Moon sits in your 7th house of marriage, which traditionally points to a caring partner.")
    bubbles = reply_messages(answer, facts)
    assert len(bubbles) == 2
    assert sum("Moon" in b for b in bubbles) == 1


def test_bubbles_that_share_words_but_not_numbers_are_kept(facts) -> None:
    answer = ("Your Mercury Antardasha runs from February 2025 to December 2027 and supports learning new skills.\n\n"
              "Your Ketu Antardasha runs from December 2027 to February 2029 and supports learning inner skills.")
    assert len(segment_reply(answer)) == 2


# ── the follow-up ─────────────────────────────────────────────────────
@pytest.mark.parametrize("style,answer,intent,question", ADVICE, ids=["en", "hinglish", "hindi", "en-rel"])
def test_the_last_bubble_is_one_relevant_follow_up(style, answer, intent, question, facts) -> None:
    from app.prompts.followup import is_good_followup

    bubbles = reply_messages(answer, facts)
    assert bubbles[-1].endswith("?") and _sentences(bubbles[-1]) == [bubbles[-1]]
    assert sum(b.rstrip().endswith("?") for b in bubbles) == 1
    assert is_good_followup(bubbles[-1], answer="\n\n".join(bubbles[:-1]), question=question)


def test_advice_questions_normally_get_a_personalised_follow_up() -> None:
    decision = decide("When will I get married?", "marriage", "english")
    assert decision.allowed
    line = prompt_line(decision)
    assert "normally end with ONE short" in line and "never ask for something they already said" in line
    assert "build on what they already told you" in line


@pytest.mark.parametrize("question,intent,previous", [
    ("What is my Lagna?", "general_astrology", None),          # a plain fact
    ("Hello", "smalltalk", None),                               # a greeting
    ("Thanks", "smalltalk", None),
    ("When will I get married?", "marriage", "Would you like to look at your partner's nature next?"),
])
def test_no_follow_up_when_it_would_not_help(question, intent, previous) -> None:
    decision = decide(question, intent, "english", previous)
    assert not decision.allowed
    assert "do NOT end with a question" in prompt_line(decision)


def test_a_follow_up_that_repeats_what_was_already_covered_is_not_good() -> None:
    from app.prompts.followup import is_good_followup

    answer = "Your partner's nature looks caring and emotionally warm, shown by the Moon in your 7th house."
    assert not is_good_followup("Would you like to know your partner's nature?", answer=answer)
    assert is_good_followup("Would you like to look at marriage timing next?", answer=answer)


def _turn(text, *, allowed=True, depth="detailed", intent="marriage", language="english", action="none",
          question="When will I get married?"):
    return ({"response": text, "followup": {"allowed": allowed}, "answer_depth": depth, "intent": intent,
             "language": {"language": language}, "question": question, "history": []},
            {"validation": {"action": action}})


@pytest.mark.parametrize("language,intent,marker", [
    ("english", "marriage", "Would you like to look at"), ("hinglish", "career", "Kya aap"),
    ("hindi", "finance", "क्या आप"),
])
def test_a_missing_follow_up_is_added_when_allowed(language, intent, marker) -> None:
    """Live 2026-11-18: the model ended 0/11 advice answers with the follow-up."""
    from app.prompts.followup import apply_followup

    body = "Your Moon sits in your 7th house, which points to a caring partner."
    out = apply_followup(*_turn(body, language=language, intent=intent))
    bubbles = segment_reply(out["response"])
    assert out["response"].startswith(body) and bubbles[-1].startswith(marker) and bubbles[-1].endswith("?")
    assert out["validation"]["followup_added"] is True


@pytest.mark.parametrize("kwargs", [
    {"allowed": False},                                     # the policy said no follow-up this turn
    {"depth": "simple"},                                    # a simple answer stays one bubble
    {"action": "regenerated+safe_fallback"},                # never after the safe reply
    {"intent": "smalltalk"},                                # no topic to offer
])
def test_no_follow_up_is_added_when_it_should_not_be(kwargs) -> None:
    from app.prompts.followup import apply_followup

    state, out = _turn("Your Lagna is Virgo, which gives a practical nature.", **kwargs)
    assert apply_followup(state, out) == out


def test_a_follow_up_the_model_already_wrote_is_not_doubled() -> None:
    from app.prompts.followup import apply_followup

    state, out = _turn("Your Moon is in your 7th house.\n\nWould you like to look at marriage timing next?")
    assert apply_followup(state, out) == out


def test_the_added_follow_up_never_offers_what_was_just_covered() -> None:
    from app.prompts.followup import apply_followup

    covered = "Your partner's nature looks caring. Marriage timing favours your current Dasha period."
    state, out = _turn(covered)
    assert apply_followup(state, out) == out


def test_the_length_line_asks_for_the_follow_up_as_its_own_message() -> None:
    assert "the FOLLOW-UP question" in length_line("detailed", "Meri shaadi kab hogi?", True)
    assert "FOLLOW-UP" not in length_line("detailed", "Meri shaadi kab hogi?", False)
    assert "FOLLOW-UP" not in length_line("simple", "What is my Lagna?", True)


def test_a_system_notice_goes_before_the_closing_follow_up() -> None:
    """Live 2026-11-18: the chart-integrity notice landed after the follow-up question."""
    from app.nodes.factual_gate import with_notice

    text = "Your Lagna is Virgo.\n\nWould you like to look at marriage timing next?"
    out = with_notice(text, "Note: your chart data needs a re-check.")
    assert segment_reply(out)[-1] == "Would you like to look at marriage timing next?"
    assert "Note: your chart data needs a re-check." in out
    assert with_notice("Your Lagna is Virgo.", "Note: x.").endswith("Note: x.")


def test_no_follow_up_is_invented_by_the_formatter(facts) -> None:
    answer = ("Your Lagna is Virgo, which gives a practical nature.\n\n"
              "Mercury, your Lagna lord, sits in your 12th house, linking you to research and foreign places.")
    assert not any(b.endswith("?") for b in reply_messages(answer, facts))


# ── factual integrity of the formatting ───────────────────────────────
@pytest.mark.parametrize("style,answer,intent,question", ADVICE, ids=["en", "hinglish", "hindi", "en-rel"])
def test_all_verified_chart_facts_survive_formatting(style, answer, intent, question, facts) -> None:
    from app.evaluation.chart_claims import validate_reply

    before = {(c.kind, c.subject, str(c.value)) for c in validate_reply(answer, facts).claims if c.status == "verified"}
    joined = "\n\n".join(reply_messages(answer, facts))
    after = {(c.kind, c.subject, str(c.value)) for c in validate_reply(joined, facts).claims if c.status == "verified"}
    assert before and before <= after
    assert not validate_reply(joined, facts).contradictions


@pytest.mark.parametrize("drift", [
    lambda b: [b[0].replace("3rd", "4th")] + b[1:],                         # a placement changed
    lambda b: [x for x in b if "Jupiter" not in x],                         # a verified fact dropped
    lambda b: b[:-1] + ["Your Sade Sati is also running now."] + b[-1:],    # a claim added
    lambda b: [x.replace("traditionally points", "definitely means") for x in b],   # certainty raised
    lambda b: [b[0][:40]] + b[1:],                                          # a sentence cut in half
])
def test_a_formatter_that_changes_anything_falls_back_to_the_validated_answer(monkeypatch, drift, facts) -> None:
    from app.prompts import segments

    answer = ("Your 7th lord Jupiter is in your 3rd house, which traditionally points to meeting someone through "
              "your circle.\n\n"
              "The Moon sits in your 7th house of marriage, which traditionally points to a caring partner.\n\n"
              "Would you like to look at marriage timing next?")
    honest = segments.segment_reply(answer)
    monkeypatch.setattr(segments, "segment_reply", lambda text, *a, **k: drift(list(honest)))
    assert reply_messages(answer, facts) == [answer]


def test_formatting_without_a_chart_still_runs_the_chart_free_checks() -> None:
    assert len(reply_messages(ENGLISH_MARRIAGE)) == 5


# ── API, streaming and history stay compatible ────────────────────────
def test_the_api_returns_the_bubbles_and_the_unchanged_full_answer(kundli) -> None:
    from app.api.server import _to_response

    body = _to_response({"response": ENGLISH_MARRIAGE, "metrics": {"errors": []}, "kundli": kundli},
                        "s", "vedic_general", "english")
    assert len(body.messages) == 5 and body.answer == ENGLISH_MARRIAGE
    assert body.messages[-1].endswith("?")


def test_the_stream_delivers_five_ordered_bubbles_of_one_turn(monkeypatch) -> None:
    from tests.test_screenshot_regressions import _sse_events, _stream_client

    calls: list[int] = []
    client = _stream_client(monkeypatch, HINGLISH_CAREER, calls)
    events = _sse_events(client.post("/chat/stream", json={"user_id": "u", "question": "career?"}).text)
    starts = [d for k, d in events if k == "segment_start"]
    done = events[-1][1]
    assert events[-1][0] == "done" and len(starts) == 5 == len(done["messages"])
    for seg in starts:
        rebuilt = "".join(d["text"] for k, d in events if k == "delta" and d["segment_id"] == seg["segment_id"])
        assert rebuilt == done["messages"][seg["index"]]
    assert done["answer"] == HINGLISH_CAREER and calls == [1]


def test_history_reload_rebuilds_the_same_bubbles() -> None:
    """The stored text is the full answer; bubbles are a pure function of it."""
    assert reply_messages(HINDI_FINANCE) == reply_messages(HINDI_FINANCE) == segment_reply(HINDI_FINANCE)
