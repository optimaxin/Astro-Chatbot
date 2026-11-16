"""Contextual Hinglish grammar layer (app/evaluation/hinglish_grammar.py,
app/nodes/language_quality.py).

Regression set: the benchmark dataset (benchmarks/hinglish_quality) —
150 Hinglish, 50 Hindi and 50 English examples. Gold text must come through
untouched (no unnecessary rewriting); negatives must be detected or fixed;
no fix may change a chart fact. The live effect is measured by
scripts/hinglish_quality_bench.py, not here.
"""

from __future__ import annotations

import json

import pytest

from app.evaluation.conciseness import _chart_tokens
from app.evaluation.hinglish_grammar import correct_deterministic, detect, verify_sentence
from benchmarks.hinglish_quality import load_examples

EXAMPLES = load_examples()
LANGUAGE_CATEGORIES = {"gender_agreement", "number_agreement", "postposition", "verb_form", "spelling",
                       "transliteration", "non_word", "collocation", "heavy_register", "register_mix",
                       "gender_assumption", "word_choice"}


def _gold_texts(language: str) -> list[tuple[str, str]]:
    out = [(e.id, e.text) for e in EXAMPLES if e.language == language and e.kind == "gold"]
    out += [(f"{e.id}:corrected", e.corrected) for e in EXAMPLES
            if e.language == language and e.kind == "negative" and e.corrected]
    return out


def test_the_regression_set_has_the_required_size() -> None:
    by_language = {lang: sum(1 for e in EXAMPLES if e.language == lang) for lang in ("hinglish", "hindi", "english")}
    assert by_language["hinglish"] >= 100 and by_language["hindi"] >= 50 and by_language["english"] >= 50


# ── 1. the screenshot examples ─────────────────────────────────────────
@pytest.mark.parametrize("wrong,fixed", [
    ("Aapka 12th house property abroad se joda hota hai.", "Aapka 12th house property abroad se juda hota hai."),
    ("Ghar kharidne mein samay-samay par delays ho sakte hain.", "Ghar kharidne mein samay-samay par deri ho sakti hai."),
    ("Abhi aap Venus Antardash mein Mercury Mahadash chal rahe hain.",
     "Abhi aapki Venus Antardasha mein Mercury Mahadasha chal rahi hai."),
    ("Relationship mein samasya aa sakta hai.", "Relationship mein samasya aa sakti hai."),
    ("Aapki business mein partner se fayda hoga.", "Aapke business mein partner se fayda hoga."),
    ("Aap ka kundli bahut interesting hai!", "Aapki kundli bahut interesting hai!"),
    ("Bahut sare log yeh sawaal poochte hai.", "Bahut sare log yeh sawaal poochte hain."),
    ("Aapka Lagna Pisces hain.", "Aapka Lagna Pisces hai."),
    ("Mercury Mahadasha January 2021 to January 2038 tak chalegi.",
     "Mercury Mahadasha January 2021 se January 2038 tak chalegi."),
])
def test_context_decided_fixes(wrong: str, fixed: str) -> None:
    assert detect(wrong, "hinglish")
    assert correct_deterministic(wrong, "hinglish")[0] == fixed
    assert not detect(fixed, "hinglish")


@pytest.mark.parametrize("text", [
    "Saturn ka effect health issues la sakta hai.",
    "Saturn 4th mein domestic harmony ke nuksan de sakta hai.",
    "Aapke career mein badhyaari hogi kyunki Jupiter 6th mein hai.",
])
def test_meaning_dependent_errors_go_to_the_model_not_a_dictionary(text: str) -> None:
    """Detected, but no word list decides the new wording."""
    assert detect(text, "hinglish")
    assert correct_deterministic(text, "hinglish")[0] == text


# ── 2. no unnecessary rewriting ────────────────────────────────────────
@pytest.mark.parametrize("language", ["hinglish", "hindi", "english"])
def test_gold_text_is_left_alone(language: str) -> None:
    flagged = [(i, t) for i, t in _gold_texts(language) if detect(t, language)
               or correct_deterministic(t, language)[0] != t]
    assert not flagged, flagged[:5]


def test_preserved_english_terms_stay_english() -> None:
    text = ("Aapke career mein partnership aur business se professional growth ho sakti hai. Mahadasha aur "
            "Antardasha dono relationship aur communication par focus dikhati hain.")
    assert correct_deterministic(text, "hinglish")[0] == text and not detect(text, "hinglish")


# ── 3. negatives are caught ───────────────────────────────────────────
def test_hinglish_negatives_are_detected_or_fixed() -> None:
    relevant = [e for e in EXAMPLES if e.language == "hinglish" and e.kind == "negative"
                and {x.category for x in e.errors} & LANGUAGE_CATEGORIES]
    caught = [e for e in relevant if detect(e.text, "hinglish") or correct_deterministic(e.text, "hinglish")[0] != e.text]
    assert len(relevant) >= 50
    assert len(caught) / len(relevant) >= 0.85, [e.id for e in relevant if e not in caught]


@pytest.mark.parametrize("wrong,fixed", [
    ("बुध की महादशा चल रहा है।", "बुध की महादशा चल रही है।"),
    ("आपकी कुंडली में 4था भाव का स्वामी बुध है।", "आपकी कुंडली में चौथे भाव का स्वामी बुध है।"),
])
def test_hindi_agreement_fixes(wrong: str, fixed: str) -> None:
    assert correct_deterministic(wrong, "hindi")[0] == fixed


# ── 4. facts never change ─────────────────────────────────────────────
def test_no_deterministic_fix_changes_a_chart_fact() -> None:
    for e in EXAMPLES:
        for text in filter(None, (e.text, e.corrected)):
            fixed, _ = correct_deterministic(text, e.language)
            assert _chart_tokens(fixed) == _chart_tokens(text), (e.id, text, fixed)


@pytest.mark.parametrize("original,corrected,ok,reason", [
    ("Saturn ka effect health issues la sakta hai.", "Saturn ke kaaran health-related dikkat aa sakti hai.", True, ""),
    ("Saturn ka effect health issues la sakta hai.", "Jupiter ke kaaran health-related dikkat aa sakti hai.", False,
     "chart facts changed"),
    ("Saturn ka effect health issues la sakta hai.", "Saturn ke kaaran pakka health-related dikkat aayegi.", False,
     "added certainty"),
    ("Saturn ka effect health issues la sakta hai.", "Saturn dikkat.", False, "length ratio"),
    ("Saturn ka effect health issues la sakta hai.", "Saturn ka effect health issues la sakta hai.", False, "not better"),
    ("Saturn ka effect health issues la sakta hai.", "शनि के कारण स्वास्थ्य में दिक्कत आ सकती है।", False, ""),
])
def test_a_model_correction_is_verified(original: str, corrected: str, ok: bool, reason: str) -> None:
    verdict = verify_sentence(original, corrected, "hinglish")
    assert verdict.accepted is ok and verdict.reason.startswith(reason)


# ── 5. the node: one targeted call, boundaries kept, safe failure ──────
class _FakeLLM:
    def __init__(self, answer: str | Exception) -> None:
        self.answer, self.calls, self.prompts = answer, 0, []

    def complete(self, messages, **_):
        self.calls += 1
        self.prompts.append(messages[-1]["content"])
        if isinstance(self.answer, Exception):
            raise self.answer
        return type("R", (), {"text": self.answer})()


def _run(monkeypatch, text: str, llm: _FakeLLM, style: str = "whatsapp_hinglish"):
    import app.llm.llm_factory as factory
    from app.nodes.language_quality import apply_language_quality

    monkeypatch.setattr(factory, "get_llm", lambda *a, **k: llm)
    state = {"response": text, "response_style": style, "session_id": "s", "user_id": "u", "request_id": "r"}
    return apply_language_quality(state, {"response": text})


def test_only_flagged_sentences_go_to_the_model(monkeypatch) -> None:
    text = ("Aapka Lagna Pisces hai, aur iska lord Jupiter 6th house mein hai.\n\n"
            "Saturn ka effect health issues la sakta hai.")
    llm = _FakeLLM(json.dumps({"1": "Saturn ke kaaran health-related dikkat aa sakti hai."}))
    out = _run(monkeypatch, text, llm)
    assert llm.calls == 1 and "Lagna Pisces" not in llm.prompts[0]
    assert out["response"] == ("Aapka Lagna Pisces hai, aur iska lord Jupiter 6th house mein hai.\n\n"
                               "Saturn ke kaaran health-related dikkat aa sakti hai.")
    report = out["validation"]["language_quality"]
    assert report["llm_accepted"] == 1 and report["issues_after"] == 0


def test_natural_hinglish_makes_no_model_call(monkeypatch) -> None:
    llm = _FakeLLM("{}")
    text = "Aapka 10th house Sagittarius hai aur iska lord Jupiter 6th house mein hai."
    out = _run(monkeypatch, text, llm)
    assert llm.calls == 0 and out.get("response", text) == text


def test_a_rejected_correction_keeps_the_original(monkeypatch) -> None:
    text = "Saturn ka effect health issues la sakta hai. Venus 7th house mein hai."
    out = _run(monkeypatch, text, _FakeLLM(json.dumps({"1": "Mars ke kaaran health-related dikkat aa sakti hai."})))
    assert "Saturn ka effect" in out.get("response", text)
    assert out["validation"]["language_quality"]["llm_rejected"] == ["chart facts changed"]


def test_a_failed_model_call_leaves_the_reply(monkeypatch) -> None:
    text = "Saturn ka effect health issues la sakta hai. Venus 7th house mein hai."
    out = _run(monkeypatch, text, _FakeLLM(TimeoutError("x")))
    assert out.get("response", text) == text


def test_an_unreadable_sentence_is_dropped_or_the_safe_reply_used(monkeypatch) -> None:
    from app.nodes.validate import safe_fallback

    two = "Aapke career mein badhyaari hogi. Jupiter 6th house mein hai."
    out = _run(monkeypatch, two, _FakeLLM(TimeoutError("x")))
    assert out["response"] == "Jupiter 6th house mein hai."
    one = "Aapke career mein badhyaari hogi."
    out = _run(monkeypatch, one, _FakeLLM(TimeoutError("x")))
    assert out["response"] == safe_fallback("whatsapp_hinglish")


def test_a_copied_english_passage_is_corrected_or_dropped(monkeypatch) -> None:
    # Live browser test: a legacy knowledge passage pasted into a Hinglish reply.
    text = ("Shani aapki Kundli mein 4th house mein hai.\n\n"
            "Body parts ruled: bones, teeth, nervous system, joints throughout body. "
            "Shani in 4th can cause domestic unrest, emotional detachment from mother, or property-related "
            "matters needing patience.\n\nShani ke kaaran bones aur joints par dhyan dene ki zaroorat hai.")
    out = _run(monkeypatch, text, _FakeLLM(TimeoutError("x")))
    assert out["response"] == ("Shani aapki Kundli mein 4th house mein hai.\n\n"
                               "Shani ke kaaran bones aur joints par dhyan dene ki zaroorat hai.")
    assert out["validation"]["language_quality"]["dropped_sentences"] == 2
    fixed = json.dumps({"1": "Yeh bones, teeth, nervous system aur joints se juda hai.",
                        "2": "4th house mein Shani ghar mein thodi asthirta aur property ke maamlon mein deri de sakta hai."})
    out = _run(monkeypatch, text, _FakeLLM(fixed))
    assert "Yeh bones, teeth, nervous system aur joints se juda hai." in out["response"]
    assert "Body parts" not in out["response"] and "domestic unrest" not in out["response"]


@pytest.mark.parametrize("text,expected", [
    # live browser test, conciseness round
    ("Abhi aapka Rahu Mahadasha chal rahi hai.", "Abhi aapki Rahu Mahadasha chal rahi hai."),
    ("Ghar ka mahaul shant rahega.\n\naur Shani joints par asar daal sakta hai.",
     "Ghar ka mahaul shant rahega.\n\nAur Shani joints par asar daal sakta hai."),
    # unchanged: the planet is skipped only before a Dasha word; decimals are not sentence ends
    ("Aapka Saturn chart mein 4th house mein hai.", "Aapka Saturn chart mein 4th house mein hai."),
    ("Aapka Lagna Aquarius 4.34° hai.", "Aapka Lagna Aquarius 4.34° hai."),
])
def test_browser_found_fixes(text, expected) -> None:
    assert correct_deterministic(text, "hinglish")[0] == expected


@pytest.mark.parametrize("text", [
    "Career: aapka 10th house Scorpio hai, lord Mars wahi baitha hai.",    # label, but Hinglish
    "Partner ka nature calm, mature aur independent ho sakta hai.",
])
def test_hinglish_with_a_label_or_list_is_not_copied_english(text) -> None:
    assert not [i for i in detect(text, "hinglish") if i.category == "copied_english"]


def test_english_replies_are_never_rewritten(monkeypatch) -> None:
    llm = _FakeLLM("{}")
    text = "Your Lagna are Pisces."
    out = _run(monkeypatch, text, llm, style="english")
    assert llm.calls == 0 and out.get("response", text) == text


def test_the_layer_can_be_switched_off(monkeypatch) -> None:
    import app.nodes.language_quality as layer

    monkeypatch.setattr(layer, "_mode", lambda: "off")
    llm = _FakeLLM("{}")
    text = "Relationship mein samasya aa sakta hai."
    out = _run(monkeypatch, text, llm)
    assert out.get("response", text) == text and llm.calls == 0


def test_grammar_guidance_reaches_only_hinglish_prompts(kundli, monkeypatch) -> None:
    import app.nodes.language_quality as layer
    from app.prompts.builder import build_system_prompt
    from app.rag.retriever import RetrievalResult

    hinglish = build_system_prompt("career", kundli, RetrievalResult.unavailable("x"), chat_bubbles=True,
                                   response_style="whatsapp_hinglish")
    english = build_system_prompt("career", kundli, RetrievalResult.unavailable("x"), chat_bubbles=True,
                                  response_style="english")
    assert "HINGLISH GRAMMAR" in hinglish and "HINGLISH GRAMMAR" not in english
    monkeypatch.setattr(layer, "_mode", lambda: "off")
    assert "HINGLISH GRAMMAR" not in build_system_prompt("career", kundli, RetrievalResult.unavailable("x"),
                                                         chat_bubbles=True, response_style="whatsapp_hinglish")


@pytest.mark.parametrize("sentence,contradicted", [
    ("Shani 4th mein hai, jo 6th lord Mercury ke saath rahta hai.", True),
    ("Aapki Kundli mein 4th house lord Saturn 4th mein hai.", True),
    ("Aapke 10th house ka lord Jupiter hai, jo 6th house mein hai.", False),
    ("Your 7th lord Mercury sits in the 6th house.", False),
])
def test_nth_lord_claims_are_checked(sentence: str, contradicted: bool) -> None:
    """Found by the Hinglish benchmark: wrong lords in "6th lord Mercury" form passed the validator."""
    from app.evaluation.chart_claims import validate_reply
    from benchmarks.hinglish_quality.facts import _bench_chart

    facts, _ = _bench_chart()
    assert bool(validate_reply(sentence, facts).contradictions) is contradicted
