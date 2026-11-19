"""Guarded concise rewrite, invented-name guard and multi-area classification
(2026-11-19 response-quality brief). Ground truth is the sample chart (Lagna
Virgo; 10th Gemini, lord Mercury in the 12th; Moon in the 7th; Venus in the 11th)."""

from __future__ import annotations

import dataclasses
import json
import time

import pytest

from app.config import settings as settings_module
from app.nodes import concise
from tests.llm_fakes import reply, use_nvidia

# 36-45 words: shown to the user as ONE long bubble (the segmenter only splits above 45).
LONG_FACT = ("Your 10th house is Gemini and its lord Mercury sits in your 12th house, which traditionally points to "
             "research, writing or work done behind the scenes. This can also suggest that you do your best work "
             "quietly, away from crowds.")
LONG_ADVICE = ("In everyday terms, you may enjoy roles where you can focus deeply, and you should try to build a quiet "
               "routine that lets you research and write regularly, because steady practice in this kind of work can "
               "bring you recognition over time.")
SHORT = "Venus in your 11th house can bring support through friends and networks."
QUESTION = "Would you like to look at business, a job, or a specific field next?"
DRAFT = "\n\n".join([LONG_FACT, LONG_ADVICE, SHORT, QUESTION])

# Compressions: every sentence's point kept in fewer words (none deleted).
GOOD_FACT = ("Your 10th house is Gemini and its lord Mercury sits in your 12th house, which can point to research "
             "or behind-the-scenes work. You may work best quietly.")
GOOD_ADVICE = ("You may enjoy roles that need deep focus. Try a quiet routine for research and writing; steady "
               "practice can bring recognition.")


def _flags(**overrides):
    s = settings_module.settings
    object.__setattr__(s, "flags", dataclasses.replace(s.flags, **overrides))


def _state(kundli, text=DRAFT, **extra):
    return {"response": text, "kundli": kundli, "response_style": "english", "answer_depth": "detailed",
            "intent": "career", "question": "Tell me about my career", "history": [], "validation": {},
            "model_used": "nvidia/nemotron-3.5-lightning-30b-a3b", "language": {"language": "english"},
            "session_id": "s", "user_id": "u", "request_id": "r", **extra}


def _answer(**messages):
    return reply(json.dumps(messages))


# ── what gets rewritten ───────────────────────────────────────────────
def test_only_long_content_messages_are_planned() -> None:
    assert concise.plan(DRAFT, "detailed") == [0, 1]
    assert concise.plan("\n\n".join([SHORT, SHORT.replace("Venus", "Moon"), QUESTION]), "detailed") == []
    assert concise.plan(DRAFT, "simple") == []
    long_question = "Would you " + "really " * 40 + "like to know more?"
    assert concise.plan(long_question, "detailed") == []


def test_a_concise_reply_makes_no_model_call(monkeypatch, kundli) -> None:
    _flags(concise_rewrite_mode="on")
    _, client = use_nvidia(monkeypatch, [])
    out = concise.concise_rewrite_node(_state(kundli, "\n\n".join([SHORT, QUESTION])))
    assert out["validation"]["concise_rewrite"]["reason"] == "already_concise" and client.requests == []


def test_off_mode_does_nothing(monkeypatch, kundli) -> None:
    _, client = use_nvidia(monkeypatch, [])
    assert concise.concise_rewrite_node(_state(kundli)) == {} and client.requests == []


# ── accepted rewrites ─────────────────────────────────────────────────
def test_long_bubbles_are_shortened_and_every_fact_is_kept(monkeypatch, kundli) -> None:
    _flags(concise_rewrite_mode="on")
    _, client = use_nvidia(monkeypatch, [_answer(**{"1": GOOD_FACT, "2": GOOD_ADVICE})])
    out = concise.concise_rewrite_node(_state(kundli))
    report = out["validation"]["concise_rewrite"]
    assert report["status"] == "applied" and report["accepted"] == 2, report
    assert out["response"] == "\n\n".join([GOOD_FACT, GOOD_ADVICE, SHORT, QUESTION])
    assert report["words_after"] < report["words_before"]
    assert len(client.requests) == 1
    sent = client.requests[0]["messages"][-1]["content"]
    assert SHORT in sent and "MESSAGES TO SHORTEN" in sent and sent.count("[shorten to") == 2


def test_one_bad_bubble_is_rejected_and_the_good_one_still_ships(monkeypatch, kundli) -> None:
    _flags(concise_rewrite_mode="on")
    dropped_advice = "Steady practice in this kind of work can bring recognition over time."
    use_nvidia(monkeypatch, [_answer(**{"1": GOOD_FACT, "2": dropped_advice})])
    out = concise.concise_rewrite_node(_state(kundli))
    report = out["validation"]["concise_rewrite"]
    assert report["verdicts"] == ["ok", "dropped_advice"] and report["reason"] == "partial"
    assert out["response"] == "\n\n".join([GOOD_FACT, LONG_ADVICE, SHORT, QUESTION])


# ── rejected rewrites keep the original ───────────────────────────────
@pytest.mark.parametrize("bad,reason", [
    ("Your 10th house is Gemini and its lord Mercury sits in your 10th house, which can point to research.", "facts"),
    ("Your 10th lord Mercury can point to research or behind-the-scenes work.", "facts"),        # dropped Gemini/12th
    ("Your 10th house is Gemini and its lord Mercury sits in your 12th house, which will definitely bring research "
     "work.", "facts"),                                                                           # certainty
    ("Rahul, your 10th house is Gemini and its lord Mercury sits in your 12th house, which can point to research.",
     "added_name"),
    ("Your 10th house is Gemini and its lord Mercury sits in your 12th house. Shall we look at research work?",
     "added_question"),
    ("Your 10th house is Gemini and its lord Mercury sits in your 12th house, which can point to <field>.",
     "malformed_or_leaked"),
    ("Your 10th house is Gemini and its lord Mercury sits in your 12th house, which can point to research", "incomplete"),
])
def test_a_rewrite_that_changes_the_answer_is_discarded(monkeypatch, kundli, bad, reason) -> None:
    _flags(concise_rewrite_mode="on")
    _, client = use_nvidia(monkeypatch, [_answer(**{"1": bad, "2": LONG_ADVICE})])
    out = concise.concise_rewrite_node(_state(kundli))
    report = out["validation"]["concise_rewrite"]
    assert report["verdicts"][0].startswith(reason), report
    assert "response" not in out, "nothing accepted: the validated original ships"
    assert len(client.requests) == 1, "never retried"


def test_unparseable_output_keeps_the_original(monkeypatch, kundli) -> None:
    _flags(concise_rewrite_mode="on")
    _, client = use_nvidia(monkeypatch, [reply("Sure! Here is a shorter version of your reply.")])
    out = concise.concise_rewrite_node(_state(kundli))
    assert out["validation"]["concise_rewrite"]["reason"] == "unparseable_output" and "response" not in out
    assert len(client.requests) == 1


def test_a_model_error_keeps_the_original(monkeypatch, kundli) -> None:
    from tests.llm_fakes import http_error

    _flags(concise_rewrite_mode="on")
    use_nvidia(monkeypatch, [http_error(500), http_error(500), http_error(500), http_error(500)])
    out = concise.concise_rewrite_node(_state(kundli))
    assert out["validation"]["concise_rewrite"]["status"] == "failed" and "response" not in out


def test_a_slow_rewrite_times_out_and_keeps_the_original(monkeypatch, kundli) -> None:
    _flags(concise_rewrite_mode="on", concise_rewrite_timeout_seconds=0.2)

    class Slow:
        def complete(self, *args, **kwargs):
            time.sleep(1.0)
            raise AssertionError("never used")

    monkeypatch.setattr(concise, "_optional_llm", lambda: Slow())
    started = time.perf_counter()
    out = concise.concise_rewrite_node(_state(kundli))
    assert out["validation"]["concise_rewrite"]["status"] == "timeout" and "response" not in out
    assert time.perf_counter() - started < 0.9


@pytest.mark.parametrize("extra", [
    {"answer_depth": "simple"}, {"generation_failed": True}, {"model_used": "rule:clarify"},
    {"validation": {"action": "regenerated+safe_fallback"}},
])
def test_simple_failed_and_deterministic_replies_are_never_rewritten(monkeypatch, kundli, extra) -> None:
    _flags(concise_rewrite_mode="on")
    _, client = use_nvidia(monkeypatch, [])
    out = concise.concise_rewrite_node(_state(kundli, **extra))
    assert out["validation"]["concise_rewrite"]["status"] == "skipped" and client.requests == []


def test_shadow_mode_records_but_never_changes_the_reply(monkeypatch, kundli, tmp_path) -> None:
    _flags(concise_rewrite_mode="shadow")
    monkeypatch.setattr(concise, "SHADOW_LOG", tmp_path / "shadow.jsonl")
    use_nvidia(monkeypatch, [_answer(**{"1": GOOD_FACT, "2": GOOD_ADVICE})])
    out = concise.concise_rewrite_node(_state(kundli))
    assert "response" not in out and out["validation"]["concise_rewrite"]["status"] == "shadow_recorded"
    line = json.loads((tmp_path / "shadow.jsonl").read_text(encoding="utf-8"))
    assert line["status"] == "shadow_would_apply" and "text" not in json.dumps(line).lower().replace("context", "")


def test_the_graph_runs_the_rewrite_between_validation_and_the_humanizer() -> None:
    from app.graph.chatbot_graph import build_graph

    edges = {(e[0], e[1]) for e in build_graph().get_graph().edges}
    assert ("validate_reply", "concise_rewrite") in edges and ("concise_rewrite", "humanize") in edges


# ── invented names ────────────────────────────────────────────────────
@pytest.mark.parametrize("text,removed,fixed", [
    ("Aapke 7th house mein Chandra hai.\n\nRahul, aapko pyaar mein patience rakhni chahiye.", ["Rahul"],
     "Aapke 7th house mein Chandra hai.\n\nAapko pyaar mein patience rakhni chahiye."),
    ("Hi Priya, your 10th house is Gemini.", ["Priya"], "Your 10th house is Gemini."),
    ("Rahul ji, aapka career achha hai.", ["Rahul"], "Aapka career achha hai."),
    ("राहुल जी, आपकी कुंडली में शुक्र ग्यारहवें भाव में है।", ["राहुल"], "आपकी कुंडली में शुक्र ग्यारहवें भाव में है।"),
])
def test_an_invented_name_is_removed_and_the_sentence_kept(text, removed, fixed) -> None:
    from app.evaluation.names import strip_invented_names

    assert strip_invented_names(text, set()) == (fixed, removed)


@pytest.mark.parametrize("text", [
    "Traditionally, you thrive in service roles. So, focus on skills.",
    "Venus, your 11th-house planet, supports friendships.", "Saturn, aapke 6th mein hai.",
    "Haan, aapki Mahadasha chal rahi hai.",
])
def test_discourse_words_and_planets_are_not_names(text) -> None:
    from app.evaluation.names import strip_invented_names

    assert strip_invented_names(text, set()) == (text, [])


def test_a_name_from_the_profile_or_the_user_is_kept(kundli) -> None:
    from app.evaluation.names import allowed_names, strip_invented_names

    chart = dataclasses.replace(kundli, birth={**(kundli.birth or {}), "name": "Ankit Sharma"})
    allowed = allowed_names(chart, ["hello"])
    assert strip_invented_names("Ankit, your 10th house is Gemini.", allowed)[1] == []
    assert strip_invented_names("Meera, aapka career achha hai.", allowed_names(None, ["Mera naam Meera hai"]))[1] == []


def test_validation_removes_an_invented_name(monkeypatch, kundli) -> None:
    from app.nodes.validate import validate_reply_node

    use_nvidia(monkeypatch, [])
    state = {**_state(kundli, "Your 10th house is Gemini.\n\nRahul, you can build steady skills in research."),
             "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "career?"}]}
    out = validate_reply_node(state)
    assert "Rahul" not in out["response"] and out["validation"]["invented_names_removed"] == ["Rahul"]


# ── several named life areas are answered, not asked about ────────────
@pytest.mark.parametrize("question", [
    "Give me a detailed analysis of my Kundli for career and marriage",
    "Meri kundli ka career aur shaadi ke liye detail mein analysis karo",
    "मेरी कुंडली का करियर और विवाह के लिए विस्तार से विश्लेषण कीजिए",
])
def test_a_multi_area_question_is_answered_without_an_llm_classification(monkeypatch, question) -> None:
    from app.nodes.classifier import classify, decide_routing

    _, client = use_nvidia(monkeypatch, [])
    result, method = classify(question)
    assert (result.intent, method, decide_routing(result, method, False)) == ("career", "keyword", "answer")
    assert result.sub_intent == "marriage" and client.requests == []


def test_a_multi_area_prompt_reads_every_named_area(kundli) -> None:
    from app.prompts.builder import build_system_prompt
    from app.rag.retriever import RetrievalResult

    system = build_system_prompt("career", kundli, RetrievalResult.unavailable("x"), chat_bubbles=True,
                                 response_style="english",
                                 question="Give me a detailed analysis of my Kundli for career and marriage")
    assert "FOCUS: CAREER" in system and "FOCUS: MARRIAGE" in system
    single = build_system_prompt("career", kundli, RetrievalResult.unavailable("x"), question="How is my career?")
    assert "FOCUS: MARRIAGE" not in single


def test_a_sign_name_is_not_a_life_area() -> None:
    from app.nodes.classifier import life_areas

    assert life_areas("मेरा लग्न धनु है?") == [] and life_areas("Will my job and money improve?") == [
        "career", "finance"]


def test_the_modal_may_is_not_the_month_may(kundli) -> None:
    from app.evaluation.rewrite_guard import compare_rewrite
    from app.kundli.chart_facts import build_chart_facts

    facts = build_chart_facts(kundli)
    assert compare_rewrite("You may enjoy quiet research work.", "You can enjoy quiet research work.", facts).ok
    dated = compare_rewrite("This period started in May 2026.", "This period started in 2026.", facts)
    assert not dated.ok and any("months changed" in r for r in dated.reasons)


# ── follow-ups answer the new question ────────────────────────────────
@pytest.mark.parametrize("question,expected", [
    ("Which is more compatible to me?", True), ("Kaun zyada compatible hai mere liye?", True),
    ("मेरे लिए कौन ज़्यादा अनुकूल है?", True), ("Who is more compatible with me?", True),
    ("Which career is better for me?", False), ("Which is better, teaching or law?", False),
    ("What kind of partner is most compatible with me?", False), ("Tell me more.", False),
])
def test_a_comparison_that_names_nothing_is_asked_about(question, expected) -> None:
    from app.nodes.clarify import is_unanchored_comparison, route_after_load_context

    assert is_unanchored_comparison(question) is expected
    history = [object()]                                   # mid-conversation: history does not name the options
    assert (route_after_load_context({"question": question, "history": history}) == "clarify") is expected


def test_the_comparison_clarification_is_in_the_users_language() -> None:
    from app.nodes.clarify import clarify_node

    hinglish = clarify_node({"question": "Kaun zyada compatible hai mere liye?", "response_style": "whatsapp_hinglish",
                             "session_id": "s"})
    assert hinglish["response"].startswith("Aap kin options ko compare") and hinglish["model_used"] == "rule:clarify"
    assert "तुलना" in clarify_node({"question": "मेरे लिए कौन ज़्यादा अनुकूल है?", "response_style": "hindi",
                                    "session_id": "s"})["response"]


def test_sentences_copied_from_the_previous_reply_are_dropped_when_new_content_remains() -> None:
    from app.evaluation.conciseness import repeated_from_previous

    previous = ("Venus in your 7th house can bring a caring partner. A partner who listens well suits you.\n\n"
                "Would you like to look at marriage timing next?")
    follow_up = ("Venus in your 7th house can bring a caring partner. Marriage timing is linked to your 7th lord's "
                 "period. The Venus Antardasha until April 2027 is traditionally supportive.")
    assert repeated_from_previous(follow_up, previous) == [0]
    only_repeat = "Venus in your 7th house can bring a caring partner. Short new line here."
    assert repeated_from_previous(only_repeat, previous) == []              # too little would remain


def test_messages_on_single_lines_become_separate_bubbles() -> None:
    from app.prompts.segments import segment_reply

    text = ("The previous Antardasha was Ketu from June 2023 to June 2024.\n"
            "This period traditionally suggests detachment and reorientation rather than outer achievement.\n"
            "Focus on inner work and rest during such phases.")
    assert len(segment_reply(text)) == 3
    assert segment_reply("Here are two options:\n- teaching\n- law") == ["Here are two options:\n- teaching\n- law"]


# ── repetition across bubbles and the bubble budget ───────────────────
def test_a_placement_already_stated_elsewhere_may_be_dropped(monkeypatch, kundli) -> None:
    """Removing a restatement is the point; the fact survives in the other bubble."""
    _flags(concise_rewrite_mode="on")
    first = "Your 10th house is Gemini and its lord Mercury sits in your 12th house."
    repeat = ("Because Mercury, your 10th lord, sits in your 12th house, you may enjoy work that is done quietly, "
              "away from crowds and noise, where you can research, write and think deeply about the long-term "
              "questions that interest you.")
    text = "\n\n".join([first, repeat, QUESTION])
    short = ("So you may enjoy quiet work away from crowds, where you can research, write and think deeply about "
             "long-term questions.")
    use_nvidia(monkeypatch, [_answer(**{"1": short})])
    out = concise.concise_rewrite_node(_state(kundli, text))
    assert out["validation"]["concise_rewrite"]["verdicts"] == ["ok"]
    assert out["response"] == "\n\n".join([first, short, QUESTION])


def test_the_only_mention_of_a_fact_may_not_be_dropped(monkeypatch, kundli) -> None:
    _flags(concise_rewrite_mode="on")
    use_nvidia(monkeypatch, [_answer(**{"1": "Your 10th lord can point to research or behind-the-scenes work.",
                                        "2": GOOD_ADVICE})])
    out = concise.concise_rewrite_node(_state(kundli))
    assert out["validation"]["concise_rewrite"]["verdicts"][0].startswith("facts:dropped verified fact")


def test_a_long_message_may_become_two_only_while_the_reply_stays_within_five() -> None:
    paragraphs = [LONG_FACT, LONG_ADVICE, SHORT, QUESTION]
    assert concise.message_budget(paragraphs, [0, 1]) == {0: "1 or 2", 1: "1 or 2"}
    crowded = [LONG_FACT, LONG_ADVICE, SHORT, SHORT, SHORT, QUESTION]          # already five content messages
    assert concise.message_budget(crowded, [0, 1]) == {0: "1", 1: "1"}


def test_more_messages_than_allowed_is_rejected(kundli) -> None:
    from app.kundli.chart_facts import build_chart_facts

    split = "Your 10th house is Gemini.\n\nIts lord Mercury sits in your 12th house, which can point to research."
    assert concise.check_message(LONG_FACT, split, facts=build_chart_facts(kundli), style_id="english",
                                 allowed=set(), max_parts=1) == "too_many_messages"


def test_a_split_into_two_short_messages_is_accepted_without_a_word_saving(kundli) -> None:
    from app.kundli.chart_facts import build_chart_facts

    facts = build_chart_facts(kundli)
    split = ("Your 10th house is Gemini and its lord Mercury sits in your 12th house.\n\n"
             "That can point to research, writing or behind-the-scenes work, and you may do your best work "
             "quietly, away from crowds.")
    assert concise.check_message(LONG_FACT, split, facts=facts, style_id="english", allowed=set(),
                                 reply_before=DRAFT, reply_after=DRAFT.replace(LONG_FACT, split), max_parts=2) == "ok"
    longer = split + " This can also bring you recognition from people who value careful, honest work."
    assert concise.check_message(LONG_FACT, longer, facts=facts, style_id="english", allowed=set(),
                                 reply_before=DRAFT, reply_after=DRAFT.replace(LONG_FACT, longer),
                                 max_parts=2) in ("not_shorter", "still_long")


# ── a follow-up that re-sends the previous answer ─────────────────────
PREVIOUS = ("Your 10th house is Gemini and its lord Mercury sits in your 12th house.\n\n"
            "That can point to research, writing or behind-the-scenes work.\n\n"
            "So try to build a quiet routine for research and writing.")


def _repeat_state(kundli, draft):
    from app.memory.redis_memory import Message

    return {**_state(kundli, draft, question="And how will this year be?"),
            "history": [Message(role="user", content="Tell me about my career", timestamp=0.0),
                        Message(role="assistant", content=PREVIOUS, timestamp=1.0)],
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "And how will this year be?"}]}


def test_a_word_for_word_repeat_is_regenerated_once(monkeypatch, kundli) -> None:
    from app.nodes.repeat_guard import apply_repeat_guard

    fresh = "This year your focus can shift to learning new skills at work, which can help you grow steadily."
    _, client = use_nvidia(monkeypatch, [reply(fresh)])
    state = _repeat_state(kundli, PREVIOUS)
    out = apply_repeat_guard(state, {"response": PREVIOUS})
    assert out["response"] == fresh and out["validation"]["repeat_guard"]["action"] == "regenerated"
    assert len(client.requests) == 1 and "repeats your previous answer" in client.requests[0]["messages"][-1]["content"]


def test_a_regeneration_that_repeats_again_gets_the_honest_offer(monkeypatch, kundli) -> None:
    from app.nodes.repeat_guard import FALLBACK, apply_repeat_guard

    _, client = use_nvidia(monkeypatch, [reply(PREVIOUS)])
    out = apply_repeat_guard(_repeat_state(kundli, PREVIOUS), {"response": PREVIOUS})
    assert out["response"] == FALLBACK["english"] and out["validation"]["repeat_guard"]["reason"] == "still_repeats"
    assert len(client.requests) == 1, "never retried"


def test_a_new_answer_on_the_same_topic_is_left_alone(monkeypatch, kundli) -> None:
    from app.nodes.repeat_guard import apply_repeat_guard, repeats_previous

    _, client = use_nvidia(monkeypatch, [])
    new = ("This year the Venus Antardasha can support teamwork and client work.\n\n"
           "Try to take on one new responsibility before April 2027.")
    assert not repeats_previous(new, PREVIOUS)
    assert apply_repeat_guard(_repeat_state(kundli, new), {"response": new}) == {"response": new}
    assert client.requests == []


# ── compress, do not delete (2026-11-19 shadow review) ────────────────
HL_PARTNER = ("Aapko aise partner pasand aa sakta hai jo samjhautar ho, intellectual ho aur visual ya creative "
              "qualities rakhte ho. Venus Virgo ke karan partner ka swasthya ya daily routine se related hona bhi ho "
              "sakta hai.")


def test_deleting_the_partner_description_is_rejected(kundli) -> None:
    """The shipped replay rewrite kept only the second sentence: the answer itself was gone."""
    from app.kundli.chart_facts import build_chart_facts

    facts = build_chart_facts(kundli)
    deleted = "Venus Virgo ke karan partner ka swasthya ya routine se related ho sakta hai."
    reply_before = "Aapka 7th house important hai.\n\n" + HL_PARTNER
    assert concise.check_message(HL_PARTNER, deleted, facts=facts, style_id="whatsapp_hinglish", allowed=set(),
                                 reply_before=reply_before,
                                 reply_after=reply_before.replace(HL_PARTNER, deleted)) == "dropped_point"
    compressed = ("Aapko samjhautar, intellectual aur creative partner pasand aa sakta hai. Venus Virgo ke karan "
                  "partner ka health ya daily routine se juda hona bhi ho sakta hai.")
    assert concise.check_message(HL_PARTNER, compressed, facts=facts, style_id="whatsapp_hinglish", allowed=set(),
                                 reply_before=reply_before,
                                 reply_after=reply_before.replace(HL_PARTNER, compressed)) == "ok"


def test_a_hedged_prediction_made_certain_is_rejected_sentence_by_sentence() -> None:
    original = ("This period can bring steady gains through teamwork and patient effort. "
                "You may also find support from senior colleagues at work.")
    certain = "This period brings steady gains through teamwork and effort. Senior colleagues may support you at work."
    assert concise.lost_points(original, certain) == ([], [original.split(". ")[0] + "."])


def test_a_restatement_of_another_message_may_still_be_dropped() -> None:
    other = "A partner who is calm and patient suits you, and steady routines help your relationship."
    original = "A calm and patient partner suits you. Your Moon in the 7th house adds emotional sensitivity to bonds."
    deleted, _ = concise.lost_points(original, "Your Moon in the 7th house adds emotional sensitivity.", other)
    assert deleted == []


def test_shadow_text_is_logged_only_when_evaluation_asks_for_it(monkeypatch, kundli, tmp_path) -> None:
    monkeypatch.setattr(concise, "SHADOW_LOG", tmp_path / "shadow.jsonl")
    for flag in (False, True):
        _flags(concise_rewrite_mode="shadow", concise_shadow_log_text=flag)
        use_nvidia(monkeypatch, [_answer(**{"1": GOOD_FACT, "2": GOOD_ADVICE})])
        concise.concise_rewrite_node(_state(kundli))
    first, second = [json.loads(x) for x in (tmp_path / "shadow.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "text" not in first and second["text"]["original"] == DRAFT
    assert [d["words_before"] for d in first["messages_detail"]] == [len(LONG_FACT.split()), len(LONG_ADVICE.split())]


# ── what the user sees, not just each bubble (2026-11-19 shadow review) ──
def test_a_rewrite_that_restates_another_bubble_is_rejected() -> None:
    others = "Your Venus sits in your 7th house, which traditionally signifies married life and beauty."
    rewrite = ("Your Venus sits in your 7th house, which signifies married life and beauty. This placement can "
               "bring harmony in marriage.")
    assert concise.repeats_other_bubble(rewrite, others)
    assert not concise.repeats_other_bubble("This placement can bring harmony and understanding in marriage.", others)


def test_a_rewrite_must_improve_the_displayed_bubbles() -> None:
    long_a = " ".join(["word"] * 40) + "."
    long_b = " ".join(["other"] * 38) + "."
    assert concise.display_gain(f"{long_a}\n\n{long_b}", f"{' '.join(['short'] * 20)}.\n\n{long_b}")
    # Same number of long bubbles, or a longer maximum, is no gain.
    assert not concise.display_gain(f"{long_a}\n\nTiny one here, quite short.",
                                    f"{' '.join(['word'] * 44)}.\n\nTiny one here, quite short.")


@pytest.mark.parametrize("language,label,body", [
    ("hindi", "पहला संदेश (सीधा उत्तर):\n", "आपकी कुंडली में दशम भाव धनु राशि में है।"),
    ("english", "First message (direct answer): ", "Your 10th lord Jupiter sits in your 6th house."),
    ("english", "Message 2: ", "Your 10th lord Jupiter sits in your 6th house."),
    ("hinglish", "Doosra message: ", "Aapka 10th lord Jupiter 6th house mein hai, jo service work dikhata hai."),
    ("hinglish", "pahla sandesh: ", "Aapka 10th lord Jupiter 6th house mein hai, jo service work dikhata hai."),
])
def test_copied_message_labels_are_removed(language, label, body) -> None:
    from app.evaluation.response_quality import apply_safe_fixes

    fixed, report = apply_safe_fixes(f"{body}\n\n{label}{body.replace('10th', '9th')}", language, "career")
    assert label.strip() not in fixed and fixed.endswith(body.replace("10th", "9th"))
    assert any(f.startswith("format_note_removed") for f in report.terminology_fixes)


def test_dropping_traditionally_from_a_reading_is_a_certainty_change() -> None:
    """Replay 2026-11-19: "Traditionally, this points to work…" became "…points to work…"."""
    original = ("Its lord Jupiter sits in your 6th house. Traditionally, this points to work that involves helping, "
                "service or solving problems for others.")
    rewritten = "Its lord Jupiter in the 6th house points to work involving helping, service or solving problems."
    assert concise.lost_points(original, rewritten)[1] == [original.split(". ", 1)[1]]
    kept = ("Its lord Jupiter in the 6th house traditionally points to work involving helping, service or solving "
            "problems.")
    assert concise.lost_points(original, kept) == ([], [])


def test_a_dropped_advice_clause_is_caught_even_when_its_sentence_survives(kundli) -> None:
    from app.kundli.chart_facts import build_chart_facts

    original = ("In everyday terms, you may find your income more stable, and expenses easier to manage, if you stay "
                "organised and avoid unnecessary spending. So one practical step: keep a simple record of income.")
    dropped = "Your income may feel more stable if you stay organised. Keep a simple record of income."
    assert concise.check_message(original, dropped, facts=build_chart_facts(kundli), style_id="english",
                                 allowed=set()) == "dropped_advice"
    assert concise.advice_markers(original) == {"stay", "avoid", "keep"}


def test_will_get_the_chance_is_not_a_hedge() -> None:
    original = "Agar aap kaam kar rahe hain, toh yeh samay aapko naye skills seekhne ka chance de sakta hai."
    certain = "Agar aap kaam kar rahe hain, toh naye skills seekhne ka chance milega."
    assert concise.lost_points(original, certain)[1] == [original]
