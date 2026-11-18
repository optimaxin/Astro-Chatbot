"""Final quality gate (2026-11): invented conjunctions, wrong lords stated as
"traditionally", unasked Dasha paragraphs, identity leaks, garbled words,
repeated follow-ups, and benchmark judge reliability. Offline."""

from __future__ import annotations

import pytest

from app.evaluation.chart_claims import validate_reply
from app.kundli.chart_facts import build_chart_facts


def _bench_facts():
    """The benchmark chart (Pisces Lagna, 7th lord Mercury, Sun/Mercury/Jupiter in the 6th)."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("_kqb_gate", root / "scripts" / "kundli_quality_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from app.kundli.kundli_service import KundliContext

    return build_chart_facts(module.screenshot_chart(KundliContext))


FACTS = None


def _f():
    global FACTS
    if FACTS is None:
        FACTS = _bench_facts()
    return FACTS


def _statuses(text: str, kind: str) -> list[str]:
    return [c.status for c in validate_reply(text, _f()).claims if c.kind == kind]


# ── 1. relations between planets ──────────────────────────────────────
@pytest.mark.parametrize("text,status", [
    # the live browser reply: right houses, invented conjunction
    ("Rahu in Taurus (3rd house) and Saturn in Gemini (4th house), both conjunct your Moon and Jupiter.",
     "contradicted"),
    ("Your Mars is conjunct Venus.", "contradicted"),
    ("Aapke chart mein Sun aur Mercury ki yuti 6th house mein hai.", "verified"),
    ("Jupiter 6th mein hai, jo Sun ke saath yuti banata hai.", "verified"),
    ("Venus conjunct Mars gives passion.", "generic"),          # general astrology, not this chart
])
def test_conjunction_claims_are_checked_against_the_engine(text: str, status: str) -> None:
    assert _statuses(text, "conjunction") == [status]


def test_a_negated_or_list_sentence_is_not_a_conjunction() -> None:
    assert _statuses("Moon aur Jupiter ek saath nahi hain.", "conjunction") == []
    assert _statuses("Jupiter ke saath Sun, Mercury bhi 6th mein hai.", "conjunction") == []


def test_an_aspect_on_this_chart_is_unsupported_not_verified() -> None:
    assert _statuses("Shani 7th ko aspect karta hai, isliye shaadi mein der ho sakti hai.", "aspect") == ["unsupported"]
    assert _statuses("शनि की दृष्टि सातवें भाव पर है।", "aspect") == ["unsupported"]
    assert _statuses("Traditionally Saturn aspects the 3rd, 7th and 10th houses from itself.", "aspect") == []


def test_traditionally_does_not_excuse_a_wrong_house_lord() -> None:
    # HQB final run ha-m13: the 7th lord is Mercury in this chart.
    assert _statuses("Traditionally 7th house ka lord Venus yahan relationship ka sukh deta hai.",
                     "lordship") == ["contradicted"]
    assert _statuses("Agar 7th house ka lord Venus ho, to shaadi achhi hoti hai.", "lordship") == ["generic"]


def test_two_periods_in_one_sentence_are_checked_clause_by_clause() -> None:
    # Bench chart: Venus Antardasha to April 2027, then Sun (to Feb 2028), then Moon.
    wrong = ("Venus Antardasha (until April 2027) may slow things, and after April 2027, Moon's antardasha can "
             "help.")
    right = ("Venus Antardasha (until April 2027) may slow things, and after April 2027, Sun's antardasha can "
             "help.")
    assert _statuses(wrong, "dasha_dates") == ["contradicted"]
    assert _statuses(right, "dasha_dates") == ["verified"]
    assert _statuses("You are in Mercury Mahadasha (January 2021 to January 2038) with Venus Antardasha "
                     "(June 2024 to April 2027).", "dasha_dates") == ["verified"]


def test_a_hindi_reply_naming_the_wrong_lord_in_devanagari_is_caught() -> None:
    assert _statuses("4th house lord शनि 4th house में है, जिससे घर में कठिनाइयाँ आती हैं।", "lordship") == ["contradicted"]
    assert _statuses("आपकी कुंडली में 4th lord बुध 6th house में है।", "lordship") == ["verified"]


# ── 2. conciseness: the unasked Dasha paragraph ───────────────────────
_SATURN = ("Shani aapki Kundli mein Gemini rashi mein 4th house mein hai. Isse ghar mein der se stability aati hai.\n\n"
           "Abhi aapki Mercury Mahadasha chal rahi hai (January 2021 se January 2038), aur current Antardasha "
           "Venus hai (June 2024 se April 2027).\n\nYe period communication ko highlight karta hai.")


def test_an_unasked_dasha_paragraph_is_removed_from_a_normal_answer() -> None:
    from app.evaluation.conciseness import drop_unasked_dasha

    out, removed = drop_unasked_dasha(_SATURN, "Shani meri kundli mein kahan hai?", "normal")
    assert removed == 2 and "Mahadasha" not in out and "Ye period" not in out and "4th house" in out


@pytest.mark.parametrize("question,depth", [
    ("Meri dasha kya chal rahi hai?", "normal"),          # asked about the Dasha
    ("Shaadi kab hogi?", "normal"),                        # timing
    ("Mera career kaisa rahega?", "detailed"),             # broad answers keep their timing factor
])
def test_the_dasha_stays_when_it_is_relevant(question: str, depth: str) -> None:
    from app.evaluation.conciseness import drop_unasked_dasha

    assert drop_unasked_dasha(_SATURN, question, depth) == (_SATURN, 0)


def test_the_dasha_stays_when_the_answer_would_not_stand_without_it() -> None:
    from app.evaluation.conciseness import drop_unasked_dasha

    text = "Abhi aapki Mercury Mahadasha chal rahi hai. Venus 7th mein hai."
    assert drop_unasked_dasha(text, "Venus kahan hai?", "normal") == (text, 0)


def test_a_normal_answer_far_over_budget_loses_its_last_elaborations() -> None:
    from app.evaluation.conciseness import trim_to_budget

    text = ("Your 9th house lord Mars in the 2nd house ties your purpose to family, wealth and guiding others. "
            "You gain success through family resources and connections, and your upbringing gives you values.\n\n"
            "This placement also suggests earning through foreign connections or long-distance travel, and an "
            "inheritance from your father.\n\n"
            "The 9th house lord in the 2nd creates a strong foundation of wealth through family and spiritual values.\n\n"
            "Over time this combination can also bring recognition within your community, steady savings and a "
            "reputation for honesty in money matters.\n\n"
            "Many people with this placement also find meaning in teaching younger relatives what they learned.\n\n"
            "Would you like to explore the timing of financial growth?")
    assert len(text.split()) > 100                     # the normal ceiling (2–3 messages, 40–80 words)
    out, removed = trim_to_budget(text, "normal")
    assert removed >= 1 and len(out.split()) <= 100
    assert out.startswith("Your 9th house lord Mars") and out.endswith("financial growth?")   # answer + follow-up kept
    assert trim_to_budget(text, "detailed") == (text, 0)


def test_two_sentences_are_never_trimmed() -> None:
    from app.evaluation.conciseness import trim_to_budget

    long_two = " ".join(["word"] * 50) + ". " + " ".join(["more"] * 50) + "."
    assert trim_to_budget(long_two, "normal") == (long_two, 0)


def test_a_fact_lookup_is_asked_for_one_line_of_meaning() -> None:
    from app.prompts.answer_depth import length_line

    assert "plus one short clause" in length_line("simple", "What is my moon sign?")
    assert "plus one short clause" not in length_line("simple", "theek hai")
    assert "no Dasha unless" in length_line("normal", "Shani ka mere ghar par kya asar hai?")


def test_a_fact_lookup_carries_its_length_note_from_the_first_turn(kundli) -> None:
    from app.prompts.answer_depth import is_fact_lookup, turn_length_note
    from app.prompts.builder import build_messages
    from app.rag.retriever import RetrievalResult

    q = "What is my moon sign?"
    note = turn_length_note("simple", q)
    assert is_fact_lookup("simple", q) and "not the bare fact alone" in note
    msgs = build_messages(q, "general_astrology", kundli, RetrievalResult.unavailable("x"), [], chat_bubbles=True,
                          response_style="english", length_note=note, length_note_always=True)
    assert msgs[-1]["content"] == f"{q}\n\n{note}"
    assert not is_fact_lookup("simple", "theek hai")
    # a normal-depth question that asks only who/where/until when
    assert is_fact_lookup("normal", "Who is the lord of my 10th house and where is it placed?")
    assert is_fact_lookup("normal", "मेरे दशम भाव का स्वामी कौन है और वह कहाँ है?")
    assert not is_fact_lookup("normal", "Shani kahan hai aur uska kya matlab hai?")


# ── 3. identity and source-title leaks ─────────────────────────────────
def test_the_model_never_names_itself() -> None:
    from app.evaluation.response_quality import IDENTITY_REPLY, apply_safe_fixes

    whole = "नहीं, मेरा विकास NVIDIA के शोधकर्ताओं द्वारा किया गया है।"
    assert apply_safe_fixes(whole, "hindi", "general_astrology")[0] == IDENTITY_REPLY["hindi"]
    mixed = "Main Nemotron hoon. Aapka Lagna Pisces hai."
    fixed, _ = apply_safe_fixes(mixed, "hinglish", "general_astrology")
    assert "Nemotron" not in fixed and "Lagna Pisces" in fixed


def test_a_reply_about_the_model_itself_is_regenerated_fresh_not_rewritten() -> None:
    from app.nodes.repair_prompts import _repair_messages

    state = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "पहले"},
                          {"role": "assistant", "content": "जवाब"}, {"role": "user", "content": "क्या मेरी शादी हो चुकी है?"}],
             "response_style": "hindi"}
    draft = "नहीं, मैं एनवीआईडीआईए के शोधकर्ताओं द्वारा बनाया गया एक भाषा मॉडल हूँ।"
    msgs = _repair_messages(state, draft, "CORRECTION REQUIRED", False)
    assert all(m["content"] != draft for m in msgs)                 # the draft is not handed back
    assert msgs[-1]["role"] == "user" and "OWN birth chart" in msgs[-1]["content"]
    normal = _repair_messages(state, "Aapka Lagna Pisces hai.", "CORRECTION REQUIRED", False)
    assert normal[-2]["content"] == "Aapka Lagna Pisces hai."


def test_a_quoted_source_breadcrumb_is_machinery() -> None:
    from app.evaluation.response_quality import mentions_machinery

    assert mentions_machinery('Traditionally ise "10TH HOUSE LORD › Tenth house lord in the sixth house" kaha jata hai.')


# ── 4. garbled words: detected by near-miss, corrected only in context ─
@pytest.mark.parametrize("word,anchor", [("assar", "asar"), ("aspekt", "aspect"), ("Mahadashi", "mahadasha"),
                                         ("naahin", "nahin"), ("partnershi", "partnership"), ("kiundli", "kundli")])
def test_slips_of_real_words_are_suspect(word: str, anchor: str) -> None:
    from app.prompts.hinglish_lexicon import suspect_words

    assert suspect_words(f"Iska {word} dikh sakta hai.") == [(word, anchor)]


def test_real_words_near_an_anchor_are_not_suspect() -> None:
    from app.prompts.hinglish_lexicon import suspect_words

    text = ("Agar aap June mein disha badalte hain, to main banata hoon ki shadi aur wealth par sahi asar "
            "padta hai; Mahadasha, Antardasha aur partnership opportunities favourable hain.")
    assert suspect_words(text) == []


def test_a_garbled_word_goes_to_the_contextual_correction_not_a_dictionary() -> None:
    from app.evaluation.hinglish_grammar import correct_deterministic, detect

    text = "Gharelu maahol par uska assar dikh sakta hai."
    assert any(i.category == "non_word" and i.span == "assar" and not i.severe for i in detect(text, "hinglish"))
    assert correct_deterministic(text, "hinglish")[0] == text       # no word list rewrites it


def test_the_benchmark_gold_text_has_no_suspect_word() -> None:
    from app.prompts.hinglish_lexicon import suspect_words
    from benchmarks.hinglish_quality import load_examples

    flagged = [(e.id, suspect_words(t)) for e in load_examples() if e.language == "hinglish"
               for t in (e.text if e.kind == "gold" else None, e.corrected) if t and suspect_words(t)]
    assert not flagged, flagged[:5]


# ── 5. a follow-up never offers what was just said ─────────────────────
def test_a_follow_up_offering_the_answer_again_is_dropped() -> None:
    from app.prompts.followup import is_good_followup

    answer = "आपके लिए जीवनसाथी स्थिर, परिपक्व और स्वतंत्र स्वभाव के होंगे।"
    question = "मुझे मेरी शादी के बारे में बताइए।"
    assert not is_good_followup("क्या आप जीवनसाथी का स्वभाव जानना चाहेंगे?", None, answer, question)
    assert is_good_followup("क्या आप विवाह का समय जानना चाहेंगे?", None, answer, question)


# ── 6. benchmark reliability ───────────────────────────────────────────
def test_the_judge_waits_out_rate_limits_and_reports_failures(tmp_path, monkeypatch) -> None:
    import benchmarks.judge_runner as jr

    monkeypatch.setattr(jr.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("rate_limited:429")
        return '{"score": 4}'

    runner = jr.JudgeRunner("t", "v1", cache_path=tmp_path / "c.json", min_interval=0, attempts=4)
    assert runner.run(["a"], [], call=flaky) == ({"score": 4}, "judged")
    assert runner.run(["a"], [], call=flaky) == ({"score": 4}, "cached")          # no second call
    assert calls["n"] == 3

    def broken(messages):
        raise RuntimeError("circuit open for nvidia/x")

    verdict, status = runner.run(["b"], [], call=broken)
    assert verdict is None and status == "failed:circuit_open"
    cov = jr.coverage(["judged", "cached", "failed:circuit_open"], {"safe_fallback": 1})
    assert cov["with_verdict"] == 2 and cov["failed"] == {"failed:circuit_open": 1}
    assert cov["excluded_from_judge"] == {"safe_fallback": 1}


def test_a_canned_reply_is_never_a_language_quality_success() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import kq_coverage
    from app.nodes.validate import safe_fallback

    row = {"answer": safe_fallback("whatsapp_hinglish"), "failed": False, "fallback_used": False}
    assert kq_coverage.row_status(row) == "safe_fallback"
    assert kq_coverage.row_status({"answer": "Aapka Lagna Pisces hai.", "fallback_used": True}) == "fallback_model"
    assert kq_coverage.row_status({"answer": "", "failed": True}) == "failed"
