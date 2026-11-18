"""Factual-integrity regression suite (live E2E 2026-11-18 follow-up).

Dosha verdict policy, sign-lordship validation, Dasha past/current/future
status, Devanagari numerals, the rewrite guard, chart integrity, cold
follow-ups, and the repair -> Bolt -> safe-reply chain. Deterministic: model
calls are scripted, the chart is fixed, "today" is pinned.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.evaluation.chart_claims import validate_reply
from app.evaluation.rewrite_guard import check_repair, compare_rewrite
from app.kundli.chart_facts import build_chart_facts, render_prompt_block
from tests.llm_fakes import reply, use_nvidia
from tests.test_chatbot_mvp import FakeRedis

MAIN = "nvidia/nemotron-3.5-lightning-30b-a3b"
BOLT = "nvidia/nemotron-3.5-bolt-e4b"

# Venus Mahadasha; Mars Antardasha is PAST, Mercury is CURRENT, Ketu is FUTURE (today = 2026-11-18).
_ANTARDASHAS = [
    ("venus", "2009-02-03", "2012-06-03"), ("sun", "2012-06-03", "2013-06-03"),
    ("moon", "2013-06-03", "2015-02-03"), ("mars", "2015-02-03", "2016-04-03"),
    ("rahu", "2016-04-03", "2019-04-03"), ("jupiter", "2019-04-03", "2021-12-03"),
    ("saturn", "2021-12-03", "2025-02-03"), ("mercury", "2025-02-03", "2027-12-03"),
    ("ketu", "2027-12-03", "2029-02-03"),
]


@pytest.fixture(autouse=True)
def _ledger_to_tmp(tmp_path, monkeypatch):
    from app.llm import usage_ledger

    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: tmp_path / "usage.jsonl")
    monkeypatch.setattr(usage_ledger, "_increment_redis", lambda record: False)


@pytest.fixture(autouse=True)
def _flags_restored():
    from app.config import settings as settings_module

    s = settings_module.settings
    original = s.flags
    yield
    object.__setattr__(s, "flags", original)


def _mode(monkeypatch, mode: str) -> None:
    from app.config import settings as settings_module

    s = settings_module.settings
    object.__setattr__(s, "flags", dataclasses.replace(s.flags, chart_validation_mode=mode))


@pytest.fixture
def chart(monkeypatch, kundli):
    import app.kundli.kundli_service as ks

    class _Date(ks.date):
        @classmethod
        def today(cls):
            return ks.date(2026, 11, 18)

    monkeypatch.setattr(ks, "date", _Date)
    return dataclasses.replace(
        kundli,
        current_mahadasha={"lord": "venus", "startsAt": "2009-02-03", "endsAt": "2029-02-03",
                           "antardashas": [{"lord": lord, "startsAt": s, "endsAt": e} for lord, s, e in _ANTARDASHAS]},
        doshas={"kaalSarp": {"isKaalSarp": False},
                "sadeSati": {"active": True, "phase": "peak", "moonRashi": "Pisces", "saturnTransitRashi": "Pisces"}})


@pytest.fixture
def facts(chart):
    return build_chart_facts(chart)


def _safe(text: str, style: str) -> bool:
    """The localized safe reply. The deterministic quality pass may trim its
    closing question ("Could you ask it once more?"), so compare the opening."""
    from app.nodes.validate import safe_fallback

    return text.startswith(safe_fallback(style)[:45])


def _bad(text, facts):
    result = validate_reply(text, facts)
    return [c.describe() for c in result.contradictions]


def _state(chart, draft, style="english", question="When will I get married?"):
    return {"response": draft, "kundli": chart, "response_style": style, "question": question, "intent": "marriage",
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": question}],
            "session_id": "s", "user_id": "u", "request_id": "r"}


# ══ 1-3. Dosha verdicts: blocked both ways, localized ═════════════════
MANGLIK_PRESENT = {
    "en": "Yes, you are Manglik because Mars is in your 1st house.",
    "hinglish": "Haan, aap Manglik hain kyunki Mars 1st house mein hai.",
    "hi": "हाँ, आप मांगलिक हैं क्योंकि मंगल पहले भाव में है।",
}
MANGLIK_ABSENT = {
    "en": "Your chart has no Manglik dosha.",
    "hinglish": "Aapki Kundli mein Manglik dosha nahi hai.",
    "hi": "आपकी कुंडली में मंगल दोष नहीं है।",
}
STYLE = {"en": "english", "hinglish": "whatsapp_hinglish", "hi": "hindi"}


@pytest.mark.parametrize("lang", ["en", "hinglish", "hi"])
@pytest.mark.parametrize("table", [MANGLIK_PRESENT, MANGLIK_ABSENT], ids=["present", "absent"])
def test_manglik_verdict_is_blocked_either_way_and_replaced_by_the_localized_notice(table, lang, facts) -> None:
    from app.nodes.factual_gate import apply_dosha_policy
    from app.safety.notices import DOSHA_UNAVAILABLE

    claims = validate_reply(table[lang], facts).claims
    assert any(c.kind == "dosha_verdict" and c.status == "unsupported" for c in claims)

    text, report = apply_dosha_policy(table[lang], "Kya main Manglik hoon?", STYLE[lang], facts)
    assert report["action"] == "dosha_unavailable"
    assert text == " ".join(DOSHA_UNAVAILABLE[lang])
    assert table[lang] not in text and "Mars" not in text and "मंगल" not in text    # no verdict, no inference


@pytest.mark.parametrize("verdict", [
    "Aapko Pitra dosha hai.", "You do not have Kemadruma dosha.", "Aapki kundli mein koi dosha nahi hai.",
    "Shrapit dosha is present in your chart.", "आपकी कुंडली में पितृ दोष है।", "Guru Chandal dosha ban raha hai.",
])
def test_other_unsupported_dosha_verdicts_are_blocked_too(verdict, facts) -> None:
    from app.nodes.factual_gate import apply_dosha_policy

    assert any(c.kind == "dosha_verdict" for c in validate_reply(verdict, facts).claims)
    keep = "Aapka Jupiter 3rd house mein hai, jo communication aur short travel ko support karta hai."
    text, report = apply_dosha_policy(f"{keep} {verdict}", "career?", "whatsapp_hinglish", facts)   # not asked
    assert report["action"] == "dosha_unavailable" and report["asked"] is False
    assert keep in text and verdict not in text


def test_a_direct_dosha_question_keeps_none_of_the_models_answer(facts) -> None:
    """Live E2E 2026-11-18, "Am I Manglik?": "Your Mars is in the 1st house. However, classical practice
    lists several cancellations… the chart alone does not treat it as a barrier." No sentence names
    the dosha, yet together they are a verdict, so the whole answer goes."""
    from app.nodes.factual_gate import apply_dosha_policy
    from app.safety.notices import DOSHA_UNAVAILABLE

    implied = ("Your Mars is in the 1st house (Virgo). However, classical practice lists several cancellations, "
               "and many astrologers find it not decisive on its own. The chart alone does not treat it as a "
               "barrier.")
    text, report = apply_dosha_policy(implied, "Am I Manglik?", "english", facts)
    assert text == " ".join(DOSHA_UNAVAILABLE["en"]) and report["asked"] is True
    hedged = "Aapki Kundli mein Mars 1st house mein hai. Toh yeh dosha lag sakta hai, par effect kam hota hai."
    text, _ = apply_dosha_policy(hedged, "Kya mujhe Manglik dosha hai?", "whatsapp_hinglish", facts)
    assert text == " ".join(DOSHA_UNAVAILABLE["hinglish"])


def test_a_general_dosha_question_keeps_the_computed_facts(facts) -> None:
    """"Do I have any dosha?" -> Sade Sati and Kaal Sarp ARE computed and stay; the rest is the notice."""
    from app.nodes.factual_gate import apply_dosha_policy
    from app.safety.notices import DOSHA_UNAVAILABLE

    draft = ("Aapki Sade Sati abhi chal rahi hai. Aapki Kundli mein Kaal Sarp dosha nahi hai. "
             "Mars 1st house mein hai, isliye Manglik ka asar halka hai. Yogas bhi achhe hain.")
    text, _ = apply_dosha_policy(draft, "Mere paas koi bhi dosha hai?", "whatsapp_hinglish", facts)
    assert text.startswith("Aapki Sade Sati abhi chal rahi hai. Aapki Kundli mein Kaal Sarp dosha nahi hai.")
    assert "Mars 1st house" not in text and "Yogas" not in text and text.endswith(DOSHA_UNAVAILABLE["hinglish"][-1])


def test_a_leftover_fragment_is_not_kept_next_to_the_notice(facts) -> None:
    """Live E2E: a "Why?" after a marriage question ended "The calculation simply isn't available here."
    plus the notice. When the model brought a dosha up unasked, a scrap of what is left is dropped."""
    from app.nodes.factual_gate import apply_dosha_policy
    from app.safety.notices import DOSHA_UNAVAILABLE

    text, report = apply_dosha_policy("Delay hota hai. Aap Manglik hain isliye shaadi mein deri hai.",
                                      "Why?", "whatsapp_hinglish", facts)
    assert report["asked"] is False and text == " ".join(DOSHA_UNAVAILABLE["hinglish"])


def test_the_chart_prompt_tells_the_model_not_to_opine_on_doshas(chart) -> None:
    assert "Give NO verdict on Manglik" in render_prompt_block(chart)


def test_computed_doshas_are_not_blocked(facts) -> None:
    """Kaal Sarp and Sade Sati ARE computed; they are judged against the engine, not blocked."""
    from app.nodes.factual_gate import apply_dosha_policy

    for sentence in ("Aapki Kundli mein Kaal Sarp dosha nahi hai.", "Aapki Sade Sati abhi chal rahi hai."):
        assert apply_dosha_policy(sentence, "Kya mujhe Kaal Sarp dosha hai?", "whatsapp_hinglish", facts) == (sentence, {})


def test_the_notice_is_not_itself_removed_and_is_not_added_twice(facts) -> None:
    from app.nodes.factual_gate import apply_dosha_policy

    once, _ = apply_dosha_policy("Your Lagna is Virgo, so practical thinking helps you a great deal. "
                                 "You are Manglik.", "Manglik dosha hai?", "english", facts)
    twice, report = apply_dosha_policy(once, "Manglik dosha hai?", "english", facts)
    assert twice == once and report == {}


def test_a_dosha_verdict_does_not_trigger_a_repair_call(monkeypatch, chart) -> None:
    from app.nodes.validate import validate_reply_node

    _mode(monkeypatch, "repair")
    _, client = use_nvidia(monkeypatch, [])
    out = validate_reply_node(_state(chart, "Aapko Manglik dosha hai. Jupiter aapke 3rd house mein hai, jo "
                                            "communication ko support karta hai.", style="whatsapp_hinglish"))
    assert not client.requests
    assert "Aapko Manglik dosha hai" not in out["response"] and "Jupiter aapke 3rd house" in out["response"]
    assert out["validation"]["dosha_policy"]["action"] == "dosha_unavailable"


def test_dosha_policy_also_holds_without_a_chart(monkeypatch, empty_kundli) -> None:
    from app.nodes.validate import validate_reply_node

    _, client = use_nvidia(monkeypatch, [])
    out = validate_reply_node(_state(empty_kundli, "Aap Manglik hain.", style="whatsapp_hinglish",
                                     question="Kya main Manglik hoon?"))
    assert "Manglik hain" not in out["response"] and not client.requests


# ══ 4-5. Sign lordship, three languages ═══════════════════════════════
@pytest.mark.parametrize("text", [
    "Virgo is ruled by Mercury.", "The lord of Virgo is Mercury.", "Mercury rules Virgo and Gemini.",
    "Kanya ka swami Budh hai.", "Kanya rashi ka lord Budh hai.", "Budh Kanya ka swami hai.",
    "कन्या राशि का स्वामी बुध है।", "बुध कन्या का स्वामी है।", "Scorpio ka swami Mangal hai.",
    "Kanya ka swami Mangal nahi hai.",
])
def test_correct_sign_lordship_passes(text, facts) -> None:
    assert [c for c in validate_reply(text, facts).claims if c.kind.startswith("sign_lord")
            and c.status != "verified"] == []
    assert any(c.kind.startswith("sign_lord") for c in validate_reply(text, facts).claims)


@pytest.mark.parametrize("text", [
    "Virgo is ruled by Mars.", "The lord of Virgo is Jupiter.", "Mars rules Aries and Virgo.",
    "Virgo ka lord Mangal hai.", "Kanya ka swami hi Mangal hai.", "Mangal Kanya ka swami hai.",
    "कन्या का स्वामी मंगल है।", "मंगल कन्या राशि का स्वामी है।", "Kanya ka swami Budh nahi hai.",
    "Your Virgo lagna ka lord Mars is strong.",
])
def test_incorrect_sign_lordship_is_a_contradiction(text, facts) -> None:
    assert _bad(text, facts), text


@pytest.mark.parametrize("text,bad", [
    # Shipped live 2026-11-18 inside a 7-bubble answer: Mercury rules the 1st and 10th, not the 12th.
    ("Mercury rules your 1st and 12th houses, so this period is about communication.", True),
    ("Venus rules the 3rd and 9th houses in your chart.", True),
    ("Mercury rules your 1st and 10th houses.", False),
    ("Jupiter rules your 4th and 7th houses.", False),
    ("Manglik dosha forms when Mars is in the 1st, 4th, 7th, 8th or 12th house.", False),
])
def test_a_lordship_over_two_houses_is_checked(text, bad, facts) -> None:
    assert bool([c for c in validate_reply(text, facts).contradictions if c.kind == "lordship"]) == bad


def test_sign_lordship_is_checked_even_with_no_chart(empty_kundli) -> None:
    nothing = build_chart_facts(empty_kundli)
    assert _bad("Virgo ka lord Mangal hai.", nothing) and not _bad("Kanya ka swami Budh hai.", nothing)


def test_a_house_lord_is_not_mistaken_for_a_sign_lord(facts) -> None:
    assert _bad("Aapka 7th house Pisces hai aur uska lord Jupiter 3rd mein hai.", facts) == []
    assert _bad("Aapka 10th house Gemini ka lord Mercury 12th mein hai.", facts) == []


def test_the_live_wrong_lord_sentence_is_stripped_before_it_ships(monkeypatch, chart) -> None:
    from app.nodes.validate import validate_reply_node

    _mode(monkeypatch, "auto")
    use_nvidia(monkeypatch, [])
    draft = ("Aapki Kundli mein Mars Virgo lagna mein hai. Virgo ka swami hi Mangal hai. "
             "Mercury 12th house mein hai, jo kharcha dikhata hai.")
    out = validate_reply_node(_state(chart, draft, style="whatsapp_hinglish"))
    assert "swami hi Mangal" not in out["response"] and "Mercury 12th house" in out["response"]


# ══ 6-7. Dasha status: past / current / future ════════════════════════
def test_dasha_sequence_has_the_expected_status(facts) -> None:
    assert facts.dasha_status("mars", "antar")[0] == "past"
    assert facts.dasha_status("mercury", "antar")[0] == "current"
    assert facts.dasha_status("ketu", "antar")[0] == "future"
    assert facts.dasha_status("venus", "maha")[0] == "current"
    assert facts.dasha_status("mars", "maha")[0] == "not_current"


@pytest.mark.parametrize("text", [
    "वर्तमान में आप शुक्र महादशा के अंतर्गत हैं, और उसकी अंतर्दशा मंगल चल रही है (फरवरी 2015 से अप्रैल 2016 तक)।",
    "Aapki Mars Antardasha abhi chal rahi hai.", "Mars Antardasha is currently running.",
    "अंतर्दशा मंगल चल रही है।", "Aap abhi Mars Mahadasha mein hain.",
    "Mercury Antardasha chal rahi hai aur Mars ki Antardasha aayegi.",
])
def test_past_dasha_described_as_current_or_future_is_a_contradiction(text, facts) -> None:
    assert any("Dasha" in c.describe() or "dasha" in c.describe() for c in validate_reply(text, facts).contradictions)


@pytest.mark.parametrize("text", [
    "Aapki Mercury Antardasha khatam ho chuki hai.", "Mercury Antardasha ended in 2027.",
    "Mercury ki Antardasha aane wali hai.", "Venus Mahadasha khatam ho chuki hai.",
    "Ketu Antardasha abhi chal rahi hai.",
])
def test_current_dasha_described_as_past_or_future_is_a_contradiction(text, facts) -> None:
    assert [c for c in validate_reply(text, facts).contradictions if c.kind == "dasha_status"], text


@pytest.mark.parametrize("text", [
    "Aapki Mercury Antardasha chal rahi hai.", "Mars Antardasha 2015 se 2016 tak chali thi.",
    "Ketu ki Antardasha aane wali hai.", "Venus Mahadasha aur Mercury Antardasha chal rahi hai.",
    "आपकी महादशा शुक्र की है और अंतर्दशा बुध की चल रही है।", "मंगल की अंतर्दशा चली थी।",
    "Venus dasha mein shaadi ke chances achhe hain.", "Jab Mars ki dasha chal rahi hoti hai, energy badhti hai.",
])
def test_correct_dasha_status_and_unrelated_dasha_talk_pass(text, facts) -> None:
    assert _bad(text, facts) == []


@pytest.mark.parametrize("text", [
    # Shipped live on 2026-11-18: Mercury (CURRENT) described as over, Ketu (FUTURE) as running.
    "Aapki Mercury Antardasha 2025 ke January se 2027 ke November tak chali. Ab aap Ketu Antardasha mein hai.",
    "Aapki Mercury Antardasha 2025 se 2027 tak chali.", "Ab aap Ketu Antardasha mein hain.",
    "Aap Ketu Antardasha mein hain.", "आप अब केतु अंतर्दशा में हैं।", "The Mercury Antardasha ran from 2025 to 2027.",
])
def test_perfective_and_now_phrasings_are_caught(text, facts) -> None:
    assert [c for c in validate_reply(text, facts).contradictions if c.kind == "dasha_status"], text


@pytest.mark.parametrize("text,bad", [
    # Shipped live 2026-11-18 for the FUTURE Ketu period.
    ("Aapki Ketu antardasha December 2027 se shuru hui hai.", True), ("Your Ketu Antardasha has started.", True),
    ("केतु अंतर्दशा शुरू हो गई है।", True), ("Aapki Mercury antardasha February 2025 se shuru hui hai.", False),
    ("Mars antardasha February 2015 se shuru hui thi.", False), ("बुध अंतर्दशा शुरू हो चुकी है।", False),
])
def test_has_started_is_checked_against_the_dasha_sequence(text, bad, facts) -> None:
    assert bool([c for c in validate_reply(text, facts).contradictions if c.kind == "dasha_status"]) == bad


@pytest.mark.parametrize("text", [
    # Shipped live on 2026-11-18: start and end swapped; an undatable Mahadasha given a date.
    "Ketu antardasha January 2029 se shuru ho jayegi.", "Ketu Antardasha starts in 2029.",
    "Mercury Antardasha 2026 mein khatam hogi.", "Mars Mahadasha January 2015 se shuru hui thi.",
])
def test_a_wrong_start_or_end_year_or_an_undatable_mahadasha_never_passes(text, facts) -> None:
    result = validate_reply(text, facts)
    assert result.contradictions or [c for c in result.unsupported if c.kind == "dasha_period"], text


@pytest.mark.parametrize("text", [
    "Ketu Antardasha December 2027 se February 2029 tak rahegi.", "Mercury Antardasha 2025 mein shuru hui.",
    "Venus Mahadasha February 2009 se February 2029 tak chal rahi hai aur Mercury Antardasha February 2025 se "
    "December 2027 tak.", "बुध अंतर्दशा फरवरी २०२५ से दिसंबर २०२७ तक चल रही है।",
])
def test_correct_start_and_end_years_pass(text, facts) -> None:
    result = validate_reply(text, facts)
    assert not result.contradictions and not [c for c in result.unsupported if c.kind == "dasha_period"]


@pytest.mark.parametrize("text", [
    "The Saturn Antardasha ran from 2021 to 2025.", "Saturn Antardasha 2021 se 2025 tak chali.",
    "Ab aap Mercury Antardasha mein hain.", "आप बुध अंतर्दशा में हैं।", "Ketu Antardasha ki dikkat chali aa rahi hai.",
])
def test_the_same_phrasings_pass_when_they_are_true(text, facts) -> None:
    assert [c for c in validate_reply(text, facts).contradictions if c.kind == "dasha_status"] == []


@pytest.mark.parametrize("text", [
    "No, your Mars Antardasha is not running right now.", "Nahi, aapki Mars Antardasha abhi nahi chal rahi.",
    "नहीं, आपकी मंगल अंतर्दशा अभी नहीं चल रही है।", "Aapki Mercury Antardasha khatam nahi hui hai.",
    "Ketu ki Antardasha abhi shuru nahi hui.", "Venus Mahadasha has not ended.", "Ketu Antardasha has not started yet.",
])
def test_a_correct_denial_about_a_dasha_is_not_flagged(text, facts) -> None:
    """Live E2E 2026-11-18: "No, your Mars Antardasha is not running right now" was read as
    "is running". Users ask false-premise questions; the right answer must survive."""
    assert [c for c in validate_reply(text, facts).contradictions if c.kind == "dasha_status"] == []


@pytest.mark.parametrize("text", [
    "Your Mercury Antardasha is not running right now.", "Aapki Mercury Antardasha nahi chal rahi hai.",
    "Aapki Mars Antardasha khatam nahi hui hai.", "Mercury Antardasha has not started yet.",
    "Venus Mahadasha has not started yet.",
])
def test_a_wrong_denial_about_a_dasha_is_a_contradiction(text, facts) -> None:
    assert [c for c in validate_reply(text, facts).contradictions if c.kind == "dasha_status"], text


# ── the chart cache must keep the Dasha sequence (live E2E 2026-11-18) ──
def test_cached_chart_keeps_the_dasha_sequence_and_saturns_transit(chart) -> None:
    """A cache hit used to rebuild the chart with ONE Antardasha, so past/future status
    (and every date check) silently stopped working on every cached turn."""
    from app.kundli.kundli_service import KundliContext

    restored = KundliContext.from_facts("u", chart.to_facts())
    before, after = build_chart_facts(chart), build_chart_facts(restored)
    assert after.antardashas == before.antardashas and len(after.antardashas) == 9
    assert after.dasha_status("mars", "antar")[0] == "past" and after.dasha_status("ketu", "antar")[0] == "future"
    assert after.saturn_transit_sign == before.saturn_transit_sign == "Pisces"
    assert _bad("Aapki Mars Antardasha abhi chal rahi hai.", after)


def test_an_old_cache_entry_without_the_sequence_still_loads(chart) -> None:
    from app.kundli.kundli_service import KundliContext

    old = chart.to_facts()
    old["dashas"].pop("antardashas")
    old.pop("saturnTransitRashi")
    facts = build_chart_facts(KundliContext.from_facts("u", old))
    assert facts.available and len(facts.antardashas) == 1 and facts.antardasha == "mercury"
    assert facts.dasha_status("mars", "antar") == (None, "")        # unknown, not guessed
    assert _bad("Aapki Mars Antardasha abhi chal rahi hai.", facts) == [
        "says dasha antardasha = mars; chart: mercury"]              # the older check still catches it


# ══ 8-9. Rewrite guard ════════════════════════════════════════════════
ORIGINAL = ("Aapka 7th lord Jupiter Scorpio mein 3rd house mein hai, aur Moon 7th house mein hai. "
            "Aapki Venus Mahadasha chal rahi hai, aur Mercury Antardasha February 2025 se December 2027 tak chal rahi hai. "
            "Isse shaadi ke chances ban sakte hain.")


def test_a_faithful_rewrite_is_accepted(facts) -> None:
    rewrite = ("Aapke 7th house ka lord Jupiter Scorpio mein 3rd house mein baitha hai, aur 7th house mein Moon hai. "
               "Venus Mahadasha chal rahi hai, aur Mercury Antardasha February 2025 se December 2027 tak chalegi. "
               "Isse shaadi ke chances ban sakte hain.")
    assert compare_rewrite(ORIGINAL, rewrite, facts).ok


def test_a_translation_with_different_phrasing_is_accepted(facts) -> None:
    english = ("Your 10th house is Gemini and its lord Mercury sits in the 12th house.")
    hinglish = "Aapka 10th house Gemini hai, aur iska lord Mercury 12th house mein hai."
    assert compare_rewrite(english, hinglish, facts).ok


def test_a_rewrite_that_introduces_sade_sati_is_rejected(facts) -> None:
    rewrite = ORIGINAL + " Saath hi aapki Sade Sati chal rahi hai, jo shaadi mein deri la sakti hai."
    verdict = compare_rewrite(ORIGINAL, rewrite, facts)
    assert not verdict.ok and any("sade_sati" in r for r in verdict.reasons)


@pytest.mark.parametrize("addition", [
    " Aapko Manglik dosha bhi hai.", " Saturn aapke 7th house ko aspect karta hai.",
    " Saturn abhi aapke 10th house par gochar kar raha hai.",
])
def test_a_rewrite_that_introduces_a_dosha_aspect_or_transit_is_rejected(addition, facts) -> None:
    assert not compare_rewrite(ORIGINAL, ORIGINAL + addition, facts).ok


def test_a_rewrite_that_drops_verified_placements_is_rejected(facts) -> None:
    rewrite = ("Aapki Venus Mahadasha chal rahi hai, aur Mercury Antardasha February 2025 se December 2027 tak "
               "chal rahi hai. Isse shaadi ke chances ban sakte hain.")
    verdict = compare_rewrite(ORIGINAL, rewrite, facts)
    assert not verdict.ok and any("dropped verified fact" in r for r in verdict.reasons)


@pytest.mark.parametrize("old,new,reason", [
    ("Aapki Mercury Antardasha chal rahi hai.", "Aapki Mercury Antardasha khatam ho chuki hai.", "Dasha status"),
    ("Mars Antardasha 2015 se 2016 tak chali thi.", "Mars Antardasha abhi chal rahi hai.", "Dasha status"),
])
def test_a_rewrite_that_flips_dasha_status_is_rejected(old, new, reason, facts) -> None:
    verdict = compare_rewrite(old, new, facts)
    assert not verdict.ok and any(reason in r or "rewrite" in r for r in verdict.reasons)


@pytest.mark.parametrize("new", [
    ORIGINAL.replace("2027", "2028"), ORIGINAL.replace("February 2025", "March 2025"),
    ORIGINAL.replace("ban sakte hain", "pakka ban jayenge"), ORIGINAL.replace("ban sakte hain", "ban jaate hain").replace(
        "Isse", "Isse zaroor"),
    ORIGINAL.replace("Jupiter Scorpio mein 3rd house", "Jupiter Scorpio mein 4th house"),
])
def test_a_rewrite_that_changes_dates_placements_or_certainty_is_rejected(new, facts) -> None:
    assert not compare_rewrite(ORIGINAL, new, facts).ok


def test_a_repair_may_not_bring_in_a_new_risky_claim(facts) -> None:
    draft = "Aapka Jupiter 10th house mein hai."
    assert check_repair(draft, "Aapka Jupiter 3rd house mein hai.", facts).ok
    assert not check_repair(draft, "Aapka Jupiter 3rd house mein hai, aur Sade Sati chal rahi hai.", facts).ok
    assert not check_repair(draft, "Aapka Jupiter 3rd house mein hai, aur aap Manglik hain.", facts).ok


def test_the_language_gate_discards_a_rewrite_that_adds_sade_sati(monkeypatch, chart) -> None:
    """A language rewrite that smuggles in a Sade Sati claim is thrown away, not retried."""
    from app.nodes.validate import _language_gate

    english = "Your Jupiter sits in the 3rd house, which supports communication."
    smuggled = "Aapka Jupiter 3rd house mein hai, aur aapki Sade Sati chal rahi hai, jo communication ko support karta hai."
    _, client = use_nvidia(monkeypatch, [reply(smuggled)])
    out = _language_gate({**_state(chart, "", "whatsapp_hinglish"), "validation": {}}, {"response": english})
    assert len(client.requests) == 1, "one rewrite, never retried"
    assert "Sade Sati" not in out["response"]
    assert out["validation"]["language_gate_rewrite_rejected"]


def test_the_grammar_correction_is_discarded_when_it_changes_a_fact(monkeypatch, chart) -> None:
    from app.nodes.language_quality import _targeted_correction

    class Issue:
        sentence, category, span, hint, severe = 0, "x", "x", "x", False

    text = "Aapka Jupiter 3rd house mein baitha hai. Isse communication strong rehti hai."
    _, client = use_nvidia(monkeypatch, [reply('{"1": "Aapka Jupiter 4th house mein baitha hai."}')])
    out, stats = _targeted_correction({**_state(chart, text, "whatsapp_hinglish"), "kundli": chart},
                                      text, [Issue()], "hinglish")
    assert out == text and stats["llm_accepted"] == 0


# ══ 10. Devanagari numerals ═══════════════════════════════════════════
def test_numeral_normalisation_is_one_for_one() -> None:
    from app.evaluation.text_normalize import normalize_numerals

    assert normalize_numerals("०१२३४५६७८९") == "0123456789"
    assert len(normalize_numerals("७वें भाव २०२७")) == len("७वें भाव २०२७")


@pytest.mark.parametrize("text,expected", [
    ("शुक्र आपके १२वें भाव में है।", "says venus is in the 12th house; chart: 11th house"),
    ("शुक्र आपके १२ वें भाव में बैठा है।", "says venus is in the 12th house; chart: 11th house"),
    ("बृहस्पति आपके ४ घर में है।", "says jupiter is in the 4th house; chart: 3rd house"),
])
def test_devanagari_house_numbers_are_validated(text, expected, facts) -> None:
    assert expected in _bad(text, facts)


def test_devanagari_house_numbers_that_are_true_pass(facts) -> None:
    assert _bad("शुक्र आपके ११वें भाव में है। बृहस्पति ३ भाव में है।", facts) == []


def test_devanagari_aspect_numeral_is_an_unsupported_claim(facts) -> None:
    result = validate_reply("पारंपरिक रूप से, शनि की ७वें भाव पर दृष्टि विवाह में विलंब का संकेत देती है।", facts)
    assert result.unsupported and not result.contradictions


def test_devanagari_years_and_dasha_periods_are_validated(facts) -> None:
    wrong_year = "आपकी बुध अंतर्दशा २०३५ में खत्म होगी।"
    assert validate_reply(wrong_year, facts).claims, "the year must be seen"
    assert _bad("बुध अंतर्दशा फरवरी २०२५ से दिसंबर २०२७ तक चल रही है।", facts) == []
    assert _bad("बुध अंतर्दशा फरवरी २०२५ से दिसंबर २०३५ तक चल रही है।", facts)
    assert [c for c in validate_reply("बुध अंतर्दशा २०२७ में खत्म हो चुकी है।", facts).contradictions
            if c.kind == "dasha_status"]


def test_devanagari_numerals_are_normalised_before_the_no_chart_guard() -> None:
    from app.nodes.validate import _guard_without_chart

    guarded = _guard_without_chart("आपकी शुक्र महादशा २०२६ से २०४६ तक है। आप अपने करियर पर ध्यान दें और मेहनत "
                                   "करते रहें, इससे आगे चलकर अच्छे नतीजे मिल सकते हैं।", "hindi")
    assert guarded is not None and "२०४६" not in guarded["response"]


# ══ 11-12. Cold vs contextual follow-ups ══════════════════════════════
COLD = ["Kab tak?", "Aur batao", "Why?", "Iska reason?", "kab tak", "कब तक?", "और बताइए", "Phir?", "What else?"]
SELF_CONTAINED = ["Meri shaadi kab hogi?", "When will I get married?", "Bhai add kar 2+3",
                  "Meri career growth kaisi rahegi and when can I expect promotion?", "Hello", "Mera Jupiter kahan hai?"]


@pytest.mark.parametrize("text", COLD)
def test_a_cold_followup_is_ambiguous_and_asks_a_question(text) -> None:
    from app.nodes.clarify import clarify_node, is_ambiguous_followup, route_after_load_context

    assert is_ambiguous_followup(text)
    state = {"question": text, "history": [], "response_style": "whatsapp_hinglish", "session_id": "s"}
    assert route_after_load_context(state) == "clarify"
    out = clarify_node(state)
    assert out["model_used"] == "rule:clarify" and out["intent"] == "clarification" and "?" in out["response"]
    assert not any(planet in out["response"] for planet in ("Venus", "Mercury", "Dasha", "Mahadasha"))


@pytest.mark.parametrize("text", COLD)
def test_the_same_message_with_history_goes_through_the_normal_pipeline(text, history) -> None:
    from app.nodes.clarify import route_after_load_context

    assert route_after_load_context({"question": text, "history": history}) == "classify_intent"


@pytest.mark.parametrize("text", SELF_CONTAINED)
def test_self_contained_questions_are_never_asked_to_clarify(text) -> None:
    from app.nodes.clarify import is_ambiguous_followup, route_after_load_context

    assert not is_ambiguous_followup(text)
    assert route_after_load_context({"question": text, "history": []}) == "classify_intent"


@pytest.mark.parametrize("style,marker", [("english", "Could you tell me"), ("whatsapp_hinglish", "Aap kis baare mein"),
                                          ("hindi", "आप किस बारे में")])
def test_the_clarification_is_in_the_users_language(style, marker) -> None:
    from app.safety.notices import clarification

    assert marker in clarification(style)


@pytest.fixture
def turn_env(monkeypatch, chart, retrieval):
    from app.graph import chatbot_graph
    from app.memory.redis_memory import RedisMemory, memory
    from app.nodes import context as context_node, rag as rag_node

    fake = FakeRedis()
    monkeypatch.setattr(memory, "_client", fake)
    monkeypatch.setattr(RedisMemory, "configured", property(lambda self: True))
    monkeypatch.setattr(context_node, "get_kundli", lambda user_id: chart)
    monkeypatch.setattr(rag_node, "retrieve", lambda *a, **k: retrieval)
    monkeypatch.setattr(chatbot_graph, "settings",
                        dataclasses.replace(chatbot_graph.settings, enable_evaluation=False))
    return fake


def test_full_graph_cold_kab_tak_asks_and_makes_no_model_call(monkeypatch, turn_env) -> None:
    from app.graph.chatbot_graph import run_turn

    _, client = use_nvidia(monkeypatch, [])
    state = run_turn("Kab tak?", "cold-1", "u1")
    assert state["response"] and "?" in state["response"]
    assert state["metrics"]["llm_calls"] == 0 and state["metrics"]["intent"] == "clarification"
    assert not client.requests
    assert state["metrics"]["history_persisted"]               # the question is remembered for the next turn


def test_full_graph_kab_tak_after_a_marriage_question_is_answered_from_context(monkeypatch, turn_env) -> None:
    from app.graph.chatbot_graph import run_turn
    from app.memory.redis_memory import memory

    memory.save_turn("ctx-1", "Meri shaadi kab hogi?", "Aapki Venus Mahadasha chal rahi hai.", {})
    steps = [reply('{"intent": "marriage", "confidence": 0.9}'), reply("Venus Mahadasha February 2029 tak chal rahi hai.")]
    _, client = use_nvidia(monkeypatch, steps)
    state = run_turn("Kab tak?", "ctx-1", "u1")
    assert state["metrics"]["intent"] != "clarification" and client.requests
    assert "2029" in state["response"]


# ══ 13. Contradictory signs and houses ════════════════════════════════
@pytest.fixture
def clashing(chart):
    planets = [dict(p) for p in chart.planets]
    next(p for p in planets if p["id"] == "mars")["house"] = 10        # Mars is in Virgo (the Lagna): 1st, not 10th
    return dataclasses.replace(chart, planets=planets)


def test_house_sign_contradiction_withholds_every_house_and_invents_no_replacement(clashing) -> None:
    facts = build_chart_facts(clashing)
    assert facts.house_sign_mismatch == ["mars"] and facts.integrity_problems == ["house_sign_mismatch"]
    assert not facts.houses and not facts.whole_sign
    assert all(p["house"] is None for p in facts.planets.values())
    assert facts.planets["mars"]["sign"] == "Virgo" and facts.lagna == "Virgo"     # signs are kept


def test_the_prompt_tells_the_model_the_houses_are_withheld(clashing) -> None:
    block = render_prompt_block(clashing)
    assert "CHART INTEGRITY PROBLEM" in block and "withheld" in block
    assert "10th house" not in block and "| 1st house" not in block


@pytest.mark.parametrize("text", ["Mars aapke 10th house mein hai.", "Mars aapke 1st house mein hai.",
                                  "Aapka 10th house Gemini hai."])
def test_any_house_statement_is_unconfirmable_when_houses_are_withheld(text, clashing) -> None:
    result = validate_reply(text, build_chart_facts(clashing))
    assert result.unsupported and not result.contradictions


@pytest.mark.parametrize("style,marker", [("english", "contradict its signs"), ("whatsapp_hinglish", "match nahi kar rahe"),
                                          ("hindi", "मेल नहीं खातीं")])
def test_the_integrity_warning_is_localized_and_appended(style, marker, clashing, monkeypatch) -> None:
    from app.nodes.validate import validate_reply_node

    _mode(monkeypatch, "auto")
    use_nvidia(monkeypatch, [])
    plain = {"english": "Your Lagna is Virgo.", "whatsapp_hinglish": "Aapka Lagna Virgo hai.",
             "hindi": "आपका लग्न कन्या है।"}[style]
    out = validate_reply_node(_state(clashing, plain, style=style))
    assert plain in out["response"] and marker in out["response"]
    assert out["validation"]["integrity_notice"]["problems"] == ["house_sign_mismatch"]


def test_conflicting_planet_records_get_the_same_warning(chart, monkeypatch) -> None:
    from app.nodes.validate import validate_reply_node

    jupiter = next(p for p in chart.planets if p["id"] == "jupiter")
    dup = dataclasses.replace(chart, planets=[*chart.planets, {**jupiter, "rashi": "Leo", "house": 12}])
    _mode(monkeypatch, "auto")
    use_nvidia(monkeypatch, [])
    out = validate_reply_node(_state(dup, "Aapka Lagna Virgo hai.", style="whatsapp_hinglish"))
    assert "Jupiter ke records" in out["response"]


def test_a_healthy_chart_gets_no_warning(chart, monkeypatch) -> None:
    from app.nodes.validate import validate_reply_node

    _mode(monkeypatch, "auto")
    use_nvidia(monkeypatch, [])
    out = validate_reply_node(_state(chart, "Aapka Lagna Virgo hai.", style="whatsapp_hinglish"))
    assert out.get("response", "Aapka Lagna Virgo hai.") == "Aapka Lagna Virgo hai."
    assert "integrity_notice" not in out.get("validation", {})


# ══ 14-16. Primary repair -> Bolt -> safe reply ═══════════════════════
WRONG = "Your career house is strong.\n\nJupiter sits there, so growth is likely.\n\nKeep going."
RIGHT = "Your career house holds no planet.\n\nJupiter sits in your 3rd house, so courage helps.\n\nKeep going."
STILL_WRONG = "Your Jupiter sits in the 10th house, so your career will grow.\n\nVenus is in your 12th house."


def _run(monkeypatch, chart, script, mode="repair", draft=WRONG):
    from app.nodes.validate import validate_reply_node

    _mode(monkeypatch, mode)
    _, client = use_nvidia(monkeypatch, script)
    return validate_reply_node(_state(chart, draft)), client


def test_primary_repair_that_works_never_reaches_bolt(monkeypatch, chart) -> None:
    out, client = _run(monkeypatch, chart, [reply(RIGHT)])
    assert out["response"] == RIGHT and out["validation"]["action"] == "regenerated"
    assert client.models == [MAIN]


def test_primary_repair_failure_is_followed_by_one_bolt_answer(monkeypatch, chart) -> None:
    out, client = _run(monkeypatch, chart, [reply(STILL_WRONG), reply(RIGHT)])
    assert out["response"] == RIGHT
    assert out["validation"]["action"] == "regenerated+bolt"
    assert client.models == [MAIN, BOLT], "one primary repair, then Bolt, nothing more"
    assert out["validation"]["primary_rejected"]
    note = client.requests[1]["messages"][-1]["content"]
    assert "rejected" in note and "Jupiter sits in the 10th house" not in note     # failures named, text not repeated


def test_bolts_answer_is_validated_against_the_same_facts(monkeypatch, chart) -> None:
    out, client = _run(monkeypatch, chart, [reply(STILL_WRONG), reply(STILL_WRONG)])
    assert _safe(out["response"], "english")
    assert out["validation"]["action"] == "regenerated+safe_fallback"
    assert client.models == [MAIN, BOLT] and out["validation"]["bolt_rejected"]
    assert "10th house" not in out["response"]


def test_both_models_failing_validation_return_the_localized_safe_reply(monkeypatch, chart) -> None:
    from app.nodes.validate import safe_fallback, validate_reply_node

    _mode(monkeypatch, "repair")
    wrong_hinglish = "Aapka Jupiter 10th house mein hai.\n\nVenus aapke 12th house mein hai."
    use_nvidia(monkeypatch, [reply(wrong_hinglish), reply(wrong_hinglish)])
    out = validate_reply_node(_state(chart, wrong_hinglish, style="whatsapp_hinglish"))
    assert out["response"] == safe_fallback("whatsapp_hinglish") and "Kripya" in out["response"]


def test_bolt_unreachable_ends_in_the_safe_reply(monkeypatch, chart) -> None:
    from tests.llm_fakes import http_error

    out, client = _run(monkeypatch, chart, [reply(STILL_WRONG), http_error(400, "bad"), http_error(400, "bad")])
    assert _safe(out["response"], "english")


def test_bolt_answer_in_the_wrong_language_is_not_accepted(monkeypatch, chart) -> None:
    from app.nodes.validate import safe_fallback, validate_reply_node

    _mode(monkeypatch, "repair")
    hinglish_wrong = "Aapka Jupiter 10th house mein hai.\n\nVenus aapke 12th house mein hai."
    devanagari = "आपका बृहस्पति तीसरे भाव में है, जो संवाद को सहारा देता है।"
    use_nvidia(monkeypatch, [reply(hinglish_wrong), reply(devanagari)])
    out = validate_reply_node(_state(chart, hinglish_wrong, style="whatsapp_hinglish"))
    assert out["response"] == safe_fallback("whatsapp_hinglish")


def test_auto_mode_still_strips_a_single_bad_sentence_without_any_model_call(monkeypatch, chart) -> None:
    draft = ("Your Lagna is Virgo.\n\nJupiter sits in your 10th house.\n\nVenus is in your 11th house, which "
             "supports gains.")
    out, client = _run(monkeypatch, chart, [], mode="auto", draft=draft)
    assert not client.requests and "10th house" not in out["response"] and "11th house" in out["response"]
    assert out["validation"]["action"] == "stripped"


def test_technical_failures_keep_their_existing_fallback(monkeypatch, chart) -> None:
    """Timeouts, 5xx, 429, empty and malformed answers fall back inside the gateway, as before."""
    from tests.llm_fakes import http_error, timeout_error

    # No retry on the same model: straight to Bolt.
    for failure in (timeout_error(), reply(""), reply('{"status": "succ')):
        out, client = _run(monkeypatch, chart, [failure, reply(RIGHT)])
        assert out["response"] == RIGHT and client.models == [MAIN, BOLT]
    # 5xx and 429 are retried on the same model first, then answered.
    for failure in (http_error(500), http_error(429)):
        out, client = _run(monkeypatch, chart, [failure, reply(RIGHT)])
        assert out["response"] == RIGHT and client.models[0] == MAIN


def test_the_final_gate_catches_what_slipped_through_a_rewrite(chart) -> None:
    from app.nodes.factual_gate import final_gate
    from app.nodes.validate import safe_fallback

    facts = build_chart_facts(chart)
    text, report = final_gate("Your Lagna is Virgo, so practical thinking helps you.\n\nJupiter sits in your 10th house.",
                              facts, "english", safe_fallback)
    assert text == "Your Lagna is Virgo, so practical thinking helps you." and report["action"] == "stripped"
    text, report = final_gate("Jupiter sits in your 10th house.", facts, "english", safe_fallback)
    assert text == safe_fallback("english") and report["action"] == "safe_fallback"


def test_log_mode_records_only_but_policy_gates_stay_on(monkeypatch, chart) -> None:
    from app.nodes.validate import validate_reply_node

    _mode(monkeypatch, "log")
    use_nvidia(monkeypatch, [])
    out = validate_reply_node(_state(chart, "Jupiter sits in your 10th house, which many people read as career "
                                            "visibility. You are Manglik."))
    assert "10th house" in out["response"]                      # chart claims are not touched in log mode
    assert "You are Manglik" not in out["response"]             # the dosha policy is not optional
