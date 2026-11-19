"""Regression tests built from real production screenshots (2026-11).

Each test quotes the reply (or reply pattern) the production chatbot actually
sent, and pins the component that now prevents it. The chart below matches
the one visible in the screenshots: Pisces Lagna, Mercury (7th lord) in the
6th with the Sun and Jupiter, Venus in the 7th (Virgo), Saturn (11th lord) in
the 4th, 10th house Sagittarius, Mercury Mahadasha from 2021.

No network: LLM calls go to the fake client in tests/llm_fakes.py.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from app.config import settings as settings_module
from app.kundli.chart_facts import build_chart_facts, render_prompt_block
from app.kundli.kundli_service import KundliContext
from tests.llm_fakes import reply, use_nvidia


# ── the screenshot user's chart ───────────────────────────────────────
def _planet(pid: str, rashi: str, house: int, deg: float = 10.0) -> dict[str, Any]:
    return {"id": pid, "rashi": rashi, "house": house, "degreeInSign": deg,
            "nakshatra": "Magha", "nakshatraPada": 1, "retrograde": False}


ANTARDASHAS = [
    ("mercury", "2021-01-10", "2023-06-07"), ("ketu", "2023-06-07", "2024-06-03"),
    ("venus", "2024-06-03", "2027-04-03"), ("sun", "2027-04-03", "2028-02-09"),
    ("moon", "2028-02-09", "2029-07-09"), ("mars", "2029-07-09", "2030-07-06"),
    ("rahu", "2030-07-06", "2033-01-24"), ("jupiter", "2033-01-24", "2035-04-30"),
    ("saturn", "2035-04-30", "2038-01-10"),
]


@pytest.fixture
def chart(monkeypatch) -> KundliContext:
    # Pin "today" inside the Venus Antardasha so the active period is stable.
    import app.kundli.kundli_service as ks

    class _Date(ks.date):
        @classmethod
        def today(cls):
            return ks.date(2026, 11, 15)

    monkeypatch.setattr(ks, "date", _Date)
    return KundliContext(
        user_id="screenshot-user", source="test", available=True,
        birth={"date": "2001-03-02", "time": "06:10"},
        ascendant={"rashi": "Pisces", "degreeInSign": 12.0},
        planets=[_planet("sun", "Leo", 6), _planet("moon", "Pisces", 1), _planet("mars", "Aries", 2),
                 _planet("mercury", "Leo", 6), _planet("jupiter", "Leo", 6), _planet("venus", "Virgo", 7),
                 _planet("saturn", "Gemini", 4), _planet("rahu", "Taurus", 3), _planet("ketu", "Scorpio", 9)],
        current_mahadasha={"lord": "mercury", "startsAt": "2021-01-10", "endsAt": "2038-01-10",
                           "antardashas": [{"lord": l, "startsAt": s, "endsAt": e} for l, s, e in ANTARDASHAS]},
    )


@pytest.fixture
def facts(chart):
    return build_chart_facts(chart)


def _flags(monkeypatch, **overrides):
    base = settings_module.settings
    patched = dataclasses.replace(base, flags=dataclasses.replace(base.flags, **overrides))
    monkeypatch.setattr(settings_module, "settings", patched)
    import app.nodes.validate as validate_module

    monkeypatch.setattr(validate_module, "settings", patched)
    return patched


def _state(chart, draft, style="english", question="When is a good period for marriage?", intent="marriage"):
    return {"response": draft, "kundli": chart, "response_style": style, "question": question,
            "intent": intent,
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": question}],
            "session_id": "s", "user_id": "u", "request_id": "r"}


# ══ 1. Semantic segments, never fragments ══════════════════════════════
SCREENSHOT_MARRIAGE_REPLY = (
    "The 7th lord Mercury and Venus are both active in your current Mercury Mahadasha, "
    "with the Venus Antardasha running from June 2024 to April 2027.\n\n"
    "The classical indications are a Mahadasha or Antardasha of the 7th lord, of Venus, of a planet "
    "sitting in the 7th house, or of the lord of the 2nd or 11th.")

DETAILED_MARRIAGE_REPLY = (
    "Your 7th house is Virgo and Venus sits in it, while its lord Mercury is placed in the 6th house. "
    "Traditionally this links marriage with work, service and a practical approach to commitment.\n\n"
    "The partner indicated by this placement is often a working professional, organised and "
    "detail-minded, who values reliability over grand gestures.\n\n"
    "For timing, the classical indications are periods of the 7th lord, of Venus, or of a planet in "
    "the 7th house. Your Venus Antardasha runs from June 2024 to April 2027.\n\n"
    "Overall, the Mercury Mahadasha from January 2021 to January 2038 supports this area, and how it "
    "plays out also depends on the strength of these planets and the Navamsa.")


def test_the_screenshot_date_range_is_never_cut_across_bubbles() -> None:
    """Screenshot 1: "…from January 2021" in one bubble, "to January 2028" in the next."""
    from app.prompts.segments import segment_reply

    cut = ("Venus Antardasha period, which runs from January 2021\n\n"
           "to January 2028. The classical indications are periods of the 7th lord.")
    assert segment_reply(cut) == [cut.replace("\n\n", " ")]
    for segment in segment_reply(SCREENSHOT_MARRIAGE_REPLY):
        assert "June 2024 to April 2027" in segment or "June 2024" not in segment


def test_a_simple_answer_is_one_bubble() -> None:
    from app.prompts.segments import segment_reply

    assert segment_reply("Your Lagna is Pisces, with the Moon placed in it.") == [
        "Your Lagna is Pisces, with the Moon placed in it."]


def test_a_detailed_answer_becomes_its_topic_paragraphs() -> None:
    from app.prompts.segments import segment_reply

    segments = segment_reply(DETAILED_MARRIAGE_REPLY)
    assert len(segments) == 4
    assert segments == [p.strip() for p in DETAILED_MARRIAGE_REPLY.split("\n\n")]
    assert all(s.rstrip().endswith(".") for s in segments), "every bubble ends a sentence"


def test_never_more_than_five_bubbles_and_no_text_lost() -> None:
    from app.prompts.segments import segment_reply

    paragraphs = [f"Paragraph {i} explains one more chart factor in a complete sentence of its own." for i in
                  range(1, 8)]
    segments = segment_reply("\n\n".join(paragraphs))
    assert len(segments) == 5
    assert _words(" ".join(segments)) == _words(" ".join(paragraphs))


def _words(text: str) -> list[str]:
    return text.split()


@pytest.mark.parametrize("raw, expected_count", [
    # a heading on its own line belongs to its content
    ("Marriage indications:\n\nYour 7th house is Virgo with Venus in it, which is traditionally supportive.", 1),
    # a list belongs to its lead-in
    ("Your chart shows these supportive factors for marriage this year:\n\n- Venus in the 7th house\n"
     "- The Venus Antardasha until April 2027.", 1),
    # a fragment too short to stand alone joins its neighbour
    ("Haan.\n\nAapki Kundli mein 7th house Virgo hai aur usmein Venus hai, jo marriage ke liye supportive hai.", 1),
    # a duplicated closing paragraph is dropped
    ("Your Mercury Mahadasha runs from January 2021 to January 2038, a period of learning and work.\n\n"
     "Your Mercury Mahadasha runs from January 2021 to January 2038, a period of learning and work.", 1),
])
def test_segments_that_cannot_stand_alone_are_merged(raw, expected_count) -> None:
    from app.prompts.segments import segment_reply

    assert len(segment_reply(raw)) == expected_count


def test_an_unrepairable_reply_falls_back_to_one_complete_message() -> None:
    from app.prompts.segments import segment_reply

    raw = "Your 7th house is Virgo and its lord Mercury sits in\n\nthe 6th house, traditionally"
    assert segment_reply(raw) == [raw]


def test_hindi_segments_keep_whole_sentences() -> None:
    from app.prompts.segments import segment_reply

    raw = ("आपकी कुंडली में सप्तम भाव कन्या राशि का है और उसमें शुक्र विराजमान है।\n\n"
           "अभी बुध महादशा में शुक्र की अंतर्दशा जून 2024 से अप्रैल 2027 तक चल रही है, जो विवाह के लिए "
           "पारंपरिक रूप से अनुकूल मानी जाती है।")
    segments = segment_reply(raw)
    assert len(segments) == 2 and all(s.endswith("।") for s in segments)


@pytest.mark.parametrize("answer, count", [
    (SCREENSHOT_MARRIAGE_REPLY, 2),
    (DETAILED_MARRIAGE_REPLY, 4),
    ("Aapki Kundli mein 10th house Sagittarius hai, jiska lord Jupiter 6th house mein hai.", 1),
])
def test_the_api_returns_the_semantic_segments(answer, count) -> None:
    from app.api.server import _to_response

    body = _to_response({"response": answer, "metrics": {"errors": []}}, "s", "vedic_general", "english")
    assert len(body.messages) == count
    assert body.answer == answer, "the full reply (what history stores) is unchanged"


def test_the_prompt_asks_for_topic_paragraphs_not_fixed_bubbles(chart) -> None:
    from app.prompts.builder import build_system_prompt
    from app.rag.retriever import RetrievalResult

    from app.prompts.answer_depth import length_line

    system = build_system_prompt("marriage", chart, RetrievalResult.unavailable("x"), chat_bubbles=True,
                                 response_style="english", length_line=length_line("simple"))
    assert "Send 2 or 3 short messages" not in system
    assert "never write more than five messages in total" in system
    assert "LENGTH FOR THIS ANSWER: ONE message" in system and "Never pad" in system


# ── streaming lifecycle ───────────────────────────────────────────────
def _sse_events(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _stream_client(monkeypatch, answer: str, calls: list[int], fail: bool = False):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from app.api import server

    def fake_run_turn(question, session_id, user_id, persona_id, **kwargs):
        calls.append(1)
        progress = kwargs.get("on_progress")
        for node in ("load_context", "classify_intent", "retrieve_knowledge", "generate_response",
                     "validate_reply", "save_metrics"):
            progress(node)
        if fail:
            raise RuntimeError("boom")
        return {"response": answer, "response_style": "hindi",
                "metrics": {"request_id": "req-1", "errors": [], "generation_failed": False}}

    monkeypatch.setattr(server, "run_turn", fake_run_turn)
    # The per-user limit lives in the real Redis: repeated test runs inside one
    # minute hit it and turned `done` into a rate-limited `error` (flaky).
    monkeypatch.setattr(server.memory, "check_rate_limit", lambda *a, **k: (True, 99))
    return fastapi_testclient.TestClient(server.app)


@pytest.mark.parametrize("answer", [
    SCREENSHOT_MARRIAGE_REPLY,
    DETAILED_MARRIAGE_REPLY,
    "अभी आपकी बुध महादशा चल रही है और यह जनवरी 2038 तक चलेगी।\n\nशुक्र की अंतर्दशा जून 2024 से अप्रैल 2027 तक है।",
    "Aapki Kundli mein Venus 7th house mein hai. " * 30,
])
def test_stream_delivers_ordered_segments_of_one_turn(monkeypatch, answer) -> None:
    from app.prompts.segments import segment_reply

    calls: list[int] = []
    client = _stream_client(monkeypatch, answer, calls)
    raw = client.post("/chat/stream", json={"user_id": "u", "question": "shaadi kab?"}).text
    events = _sse_events(raw)
    kinds = [k for k, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert "status" in kinds and "error" not in kinds
    assert len({data["message_id"] for _, data in events}) == 1, "one logical turn"

    expected = segment_reply(answer)
    starts = [d for k, d in events if k == "segment_start"]
    ends = [d for k, d in events if k == "segment_end"]
    assert [d["index"] for d in starts] == list(range(len(expected))) == [d["index"] for d in ends]
    assert len({d["segment_id"] for d in starts}) == len(expected), "distinct segment ids"
    for seg in starts:
        rebuilt = "".join(d["text"] for k, d in events if k == "delta" and d["segment_id"] == seg["segment_id"])
        assert rebuilt == expected[seg["index"]], "deltas rebuild exactly that segment"
    done = events[-1][1]
    assert done["messages"] == expected == [s["text"] for s in done["segments"]]
    assert done["answer"] == answer
    assert calls == [1], "the turn runs (and is persisted) exactly once"


def test_stream_reports_failure_as_one_error_event(monkeypatch) -> None:
    calls: list[int] = []
    client = _stream_client(monkeypatch, "unused", calls, fail=True)
    events = _sse_events(client.post("/chat/stream", json={"user_id": "u", "question": "hi"}).text)
    assert [k for k, _ in events][0] == "start" and events[-1][0] == "error"
    assert "boom" not in json.dumps(events), "internals never leak"
    assert not any(k == "delta" for k, _ in events)


# ══ 2. No unnecessary follow-up questions ══════════════════════════════
PROBE_ENDINGS = [
    "What kind of partner are you looking for — someone traditional, modern, or a mix?",
    "अभी तुम्हारी क्या योजनाएं हैं?",
    "क्या तुम्हे कोई विशेष तारीख या महीना पहले से सूझ रहा है?",
    "Kya tum software development, data analytics, ya koi specific tech role dekh rahe ho?",
    "Kya aap poochna chahte hain ki partner kaise ho sakta hai? Ya koi dusra sawal hai?",
    "Koi job change soch rahe ho ya current role mein growth?",
    "Kya tum koi specific relationship question poochna chahte ho, ya family se related koi concern?",
]


@pytest.mark.parametrize("ending", PROBE_ENDINGS)
def test_screenshot_probe_questions_are_removed(ending) -> None:
    from app.evaluation.response_quality import apply_safe_fixes

    body = ("Your 7th house is Virgo and Venus sits there. Its lord Mercury is in the 6th house. "
            "Traditionally this links partnership with service and daily work.")
    fixed, report = apply_safe_fixes(f"{body} {ending}", "english", "marriage")
    assert fixed == body
    assert report.follow_up_removed >= 1


def test_a_genuine_clarification_is_kept() -> None:
    from app.evaluation.response_quality import apply_safe_fixes

    text = "Which house would you like me to look at?"
    assert apply_safe_fixes(text, "english", "general_astrology")[0] == text
    text = "Could you tell me what you mean by that?"
    assert apply_safe_fixes(f"I'm not sure I follow. {text}", "english", "unknown")[0].endswith(text)


def test_no_prompt_part_tells_the_model_to_ask_before_answering() -> None:
    from app.persona.registry import PERSONAS
    from app.prompts.builder import BASE_SYSTEM_PROMPT, DOMAIN_MODULES
    from app.prompts.response_style import STYLES

    texts = [BASE_SYSTEM_PROMPT, *DOMAIN_MODULES.values(), *(p.system_prompt for p in PERSONAS.values())]
    for text in texts:
        lowered = text.lower()
        assert "ask that first" not in lowered and "ask that one thing first" not in lowered
        assert "before interpreting" not in lowered
    for style in STYLES.values():
        example = style.instruction.split("WRITE LIKE THIS")[-1].split("NOT LIKE THIS")[0]
        assert "?" not in example, f"{style.id} example ends on a question"


# ══ 3. Language: detection, priority and persistence ═══════════════════
@pytest.mark.parametrize("question, language", [
    ("What is my current Mahadasha?", "english"),
    ("Tell me about my career", "english"),
    ("Meri shaadi kab hogi?", "hinglish"),
    ("Mera career kaisa rahega?", "hinglish"),
    ("मेरी कुंडली में क्या है?", "hindi"),
    ("मुझे मेरी शादी के बारे में बताइए।", "hindi"),
    ("Can you explain in Hindi?", "hindi"),
    ("Ab English mein answer do", "english"),
    # screenshot questions answered in the wrong language
    ("I am married or not?", "english"),
    ("Cau you tell me about my family and relationship", "english"),
])
def test_language_table_from_the_spec(question, language) -> None:
    from app.prompts.language_policy import resolve_language

    assert resolve_language(question).language == language


def test_an_explicit_english_request_persists_over_later_hinglish() -> None:
    """Screenshot 9-10: after "I want you to reply in english" the next reply's
    question was Hinglish again, and "Mera rebirth kb hoga?" got Hinglish."""
    from app.prompts.language_policy import resolve_language

    first = resolve_language("I want you to reply in english")
    assert first.language == "english" and first.persist == "english" and first.switch_only
    later = resolve_language("Mera rebirth kb hoga?", saved_preference=first.persist)
    assert later.language == "english" and later.source == "saved"


def test_hindi_request_persists_and_a_new_request_replaces_it() -> None:
    from app.prompts.language_policy import resolve_language

    hindi = resolve_language("Kya aap hindi me jawab de sakte hai")
    assert hindi.language == "hindi" and hindi.persist == "hindi" and hindi.switch_only
    assert resolve_language("Meri shaadi ka sahi time kya hai", saved_preference="hindi").language == "hindi"
    back = resolve_language("Reply in English from now on", saved_preference="hindi")
    assert back.language == "english" and back.persist == "english"


def test_without_a_preference_each_question_sets_its_own_language() -> None:
    from app.prompts.language_policy import resolve_language

    assert resolve_language("Tell me about my career").language == "english"
    assert resolve_language("Mera career kaisa rahega?").language == "hinglish"
    assert resolve_language("ok", earlier_user_messages=["Mera career kaisa rahega?"]).language == "hinglish"


def test_asking_for_a_translation_is_not_a_language_switch() -> None:
    from app.prompts.language_policy import explicit_request

    assert explicit_request("What does Lagna mean in English?") is None
    assert explicit_request("What is my rashi called in Hindi?") is None


def test_a_client_pinned_style_still_wins() -> None:
    from app.prompts.language_policy import resolve_language

    decision = resolve_language("Meri shaadi kab hogi?", saved_preference="hindi", requested_style="formal")
    assert decision.style_id == "formal" and decision.source == "client"


@pytest.mark.parametrize("style, text", [
    # screenshot: English question, Hinglish answer
    ("english", "Chart dekhne ke baad lag raha hai ki marriage ke chances theek-thenek hain. "
                "7th house Virgo mein hai, aur usmein Venus bhi hai."),
    ("english", "Family aur relationship dono tumhari kundli ke different houses se judge honge."),
    # screenshot: Devanagari words inside Hinglish
    ("whatsapp_hinglish", "Tumhari kundli ke 11th house lord Saturn 4th house mein hai, jo technology "
                          "aur backend systems ke liye generally good संकेत देता है।"),
    # Hindi requested, English answer
    ("hindi", "Your current Mahadasha is Mercury and the Antardasha is Venus."),
])
def test_wrong_language_replies_are_rejected(style, text) -> None:
    from app.prompts.language import reply_matches

    assert not reply_matches(style, text)


@pytest.mark.parametrize("style, text", [
    ("english", "Your current Mahadasha is Mercury, with the Venus Antardasha until April 2027."),
    ("whatsapp_hinglish", "Aapki Kundli mein 10th house Sagittarius hai, jiska lord Jupiter 6th house mein hai."),
    ("hindi", "आपकी कुंडली में दशम भाव धनु राशि का है और उसका स्वामी बृहस्पति छठे भाव में है। career के लिए "
              "यह सेवा से जुड़ा माना जाता है।"),
])
def test_right_language_replies_pass(style, text) -> None:
    from app.prompts.language import reply_matches

    assert reply_matches(style, text)


def test_language_lock_is_respectful_and_consistent() -> None:
    from app.prompts.language import language_lock

    assert '"aap"' in language_lock("whatsapp_hinglish")
    assert "आप" in language_lock("hindi") and "तुम" in language_lock("hindi")   # told not to mix
    assert "even if earlier messages" in language_lock("english")


def test_the_explicit_preference_is_saved_to_session_state(monkeypatch, chart) -> None:
    from app.nodes import generate

    saved: dict[str, Any] = {}
    monkeypatch.setattr(generate.memory, "save_turn",
                        lambda sid, q, a, state: saved.update(state) or True)
    import time

    generate.save_metrics_node({
        "request_id": "r", "session_id": "s", "user_id": "u", "question": "Reply in English",
        "response": "Sure — I'll continue in English.", "started_at": time.perf_counter(),
        "language": {"language": "english", "persisted": "english", "source": "explicit"},
        "kundli": chart,
    })
    assert saved["language_preference"] == "english"
    assert saved["last_language"] == "english"


def test_bookkeeping_is_never_shown_to_the_model_as_user_facts() -> None:
    from app.prompts.builder import user_facts

    assert user_facts({"last_intent": "career", "last_turn_at": 1.0, "language_preference": "english",
                       "occupation": "teacher"}) == {"occupation": "teacher"}


def test_a_language_ability_sentence_does_not_leak_into_the_next_answer() -> None:
    """Screenshot 3: "Meri shaadi ka sahi time kya hai" was answered starting with
    "नहीं, मैं हिंदी में जवाब दे सकता हूँ 😊"."""
    from app.evaluation.response_quality import apply_safe_fixes

    # In production it was its own bubble, i.e. its own paragraph.
    text = ("नहीं, मैं हिंदी में जवाब दे सकता हूँ 😊\n\nआपका सप्तम भाव कन्या राशि का है और उसमें शुक्र है। "
            "अभी बुध महादशा में शुक्र की अंतर्दशा चल रही है।")
    fixed, report = apply_safe_fixes(text, "hindi", "marriage", "Meri shaadi ka sahi time kya hai")
    assert "हिंदी में जवाब" not in fixed and fixed.startswith("आपका सप्तम भाव")
    assert report.language_meta_removed == 1
    # …but it is the right reply when the user just asked to switch.
    confirm = "हाँ, मैं हिंदी में जवाब दे सकता हूँ। आगे से हिंदी में ही बात करेंगे।"
    assert apply_safe_fixes(confirm, "hindi", "unknown", "Kya aap hindi me jawab de sakte hai")[0] == confirm


# ══ 4. Terminology ═════════════════════════════════════════════════════
def test_half_translated_dasha_terms_are_repaired() -> None:
    """Screenshot 3-4: "बुध महाdasha", "शुक्र की अन्तर्दasha"."""
    from app.prompts.terminology import hybrid_tokens, normalize_terms

    text, fixes = normalize_terms("अभी आपकी बुध महाdasha चल रही है, जिसमें शुक्र की अन्तर्दasha है।", "hindi")
    assert text == "अभी आपकी बुध महादशा चल रही है, जिसमें शुक्र की अंतर्दशा है।"
    assert fixes and not hybrid_tokens(text)


def test_devanagari_glosses_are_removed_from_hinglish() -> None:
    """Screenshot 4: "Tumhare career ke liye 10th house (करियर) Sagittarius (धनु) hai"."""
    from app.prompts.language import reply_matches
    from app.prompts.terminology import normalize_terms

    raw = ("Aapke career ke liye 10th house (करियर) Sagittarius (धनु) hai, jiska lord Jupiter "
           "(बृहस्पति) 6th house (सेवा/श्रम) mein hai.")
    text, _ = normalize_terms(raw, "hinglish")
    assert text == "Aapke career ke liye 10th house Sagittarius hai, jiska lord Jupiter 6th house mein hai."
    assert reply_matches("whatsapp_hinglish", text)


def test_terminology_line_names_one_form_per_language() -> None:
    from app.prompts.terminology import terminology_line

    assert "महादशा" in terminology_line("hindi") and "अंतर्दशा" in terminology_line("hindi")
    assert "Budh" in terminology_line("hinglish") and "Shani" in terminology_line("hinglish")
    assert "Mercury" in terminology_line("english") and "बुध" not in terminology_line("english")


# ══ 5. Chart grounding ═════════════════════════════════════════════════
def test_mercurys_placement_in_the_7th_is_caught(facts) -> None:
    """Screenshot 2: "Mercury's placement in the 7th and Venus there give this
    period strong marriage potential." Mercury is in the 6th."""
    from app.evaluation.chart_claims import validate_reply

    result = validate_reply("Mercury's placement in the 7th and Venus there give this period strong "
                            "marriage potential.", facts)
    assert [(c.subject, c.value, c.truth) for c in result.contradictions] == [("mercury", 7, 6)]


def test_devanagari_placements_and_lordships_are_validated(facts) -> None:
    """Screenshot 3: "आपका 7वें घर का स्वामी बुध और शुक्र 7वें घर में हैं।" was never checked."""
    from app.evaluation.chart_claims import validate_reply

    assert validate_reply("शुक्र 7वें घर में है।", facts).claims[0].status == "verified"
    wrong = validate_reply("बुध सातवें घर में है।", facts)
    assert wrong.contradictions and wrong.contradictions[0].truth == 6
    lord = validate_reply("आपके 7वें घर का स्वामी बुध है।", facts)
    assert any(c.kind == "lordship" and c.status == "verified" for c in lord.claims)
    bad_lord = validate_reply("आपके 7वें घर का स्वामी शनि है।", facts)
    assert any(c.kind == "lordship" and c.status == "contradicted" for c in bad_lord.claims)


def test_invented_antardasha_dates_are_caught(facts) -> None:
    """Screenshot 1/4: "Venus Antardasha period, which runs from January 2021 to January 2028"."""
    from app.evaluation.chart_claims import validate_reply

    bad = validate_reply("You are in the Venus Antardasha, which runs from January 2021 to January 2028.", facts)
    assert any(c.kind == "dasha_dates" and c.status == "contradicted" for c in bad.claims)
    good = validate_reply("The Venus Antardasha runs from June 2024 to April 2027, inside your Mercury "
                          "Mahadasha (2021 to 2038).", facts)
    assert all(c.status == "verified" for c in good.claims if c.kind == "dasha_dates")


def test_the_model_is_given_the_antardasha_dates(chart) -> None:
    block = render_prompt_block(chart, "marriage")
    assert "current Antardasha: Venus (June 2024 to April 2027)" in block
    assert "Antardasha sequence within the Mercury Mahadasha" in block
    assert "Jupiter January 2033 to April 2035" in block


def test_a_wrong_placement_in_history_is_not_fed_back(chart) -> None:
    from app.memory.redis_memory import Message
    from app.nodes.context import clean_history

    history = [Message("user", "When is a good period for marriage?", 0.0),
               Message("assistant", "Mercury sits in your 7th house. Venus is in your 7th house.", 0.0)]
    cleaned, removed = clean_history(history, chart)
    assert removed == 1
    assert cleaned[1].content == "Venus is in your 7th house."
    assert cleaned[0].content == history[0].content


def test_contradicting_draft_is_repaired_before_the_user_sees_it(monkeypatch, chart) -> None:
    from app.nodes.validate import validate_reply_node

    _flags(monkeypatch, chart_validation_mode="repair")
    right = "Venus sits in your 7th house, and its lord Mercury is in the 6th."
    _, client = use_nvidia(monkeypatch, [reply(right)])
    out = validate_reply_node(_state(chart, "Mercury sits in your 7th house, so marriage looks strong."))
    assert out["response"] == right
    note = client.requests[0]["messages"][-1]["content"]
    assert "mercury is in the 7th house; chart: 6th house" in note


# ══ 6. Questions a chart cannot answer literally ═══════════════════════
def test_am_i_married_gets_the_life_fact_note() -> None:
    """Screenshot 6: "I am married or not?" -> "marriage ke chances theek-thenek hain"
    and "Navamsa (D9) check karna zaroori hai ki shadi actually hui ya nahi"."""
    from app.prompts.builder import question_note

    for q in ("I am married or not?", "Am I married?", "Kya meri shaadi ho chuki hai?",
              "क्या मेरी शादी हो चुकी है?"):
        note = question_note(q)
        assert note and "cannot tell whether someone IS married" in note, q


def test_rebirth_gets_the_beyond_chart_note() -> None:
    """Screenshot 10: "Mera rebirth kb hoga?" -> "Chart data mein rebirth … mention nahi hai
    aur retrieved knowledge bhi is baare mein kuch nahi batata"."""
    from app.prompts.builder import question_note

    note = question_note("Mera rebirth kb hoga?")
    assert note and "does not give that" in note and "never talk about 'data'" in note


def test_a_language_switch_gets_a_one_line_confirmation_note() -> None:
    from app.prompts.builder import question_note

    assert "ONE short sentence" in question_note("I want you to reply in english", switch_only=True)


def test_machinery_talk_triggers_one_repair(monkeypatch, chart) -> None:
    from app.evaluation.response_quality import mentions_machinery
    from app.nodes.validate import validate_reply_node

    leaked = ("Chart data mein rebirth ya antim janam ki date mention nahi hai aur retrieved knowledge bhi "
              "is baare mein kuch nahi batata.")
    assert mentions_machinery(leaked)
    _flags(monkeypatch, chart_validation_mode="repair")
    fixed = ("Vedic astrology birth chart se rebirth ki koi date nahi batata. Parampara mein moksha ka vichar "
             "12th house aur Ketu se joda jaata hai.")
    _, client = use_nvidia(monkeypatch, [reply(fixed)])
    out = validate_reply_node(_state(chart, leaked, style="whatsapp_hinglish", question="Mera rebirth kb hoga?",
                                     intent="general_astrology"))
    assert out["response"] == fixed and len(client.requests) == 1
    assert "internal machinery" in client.requests[0]["messages"][-1]["content"]


def test_filler_verdicts_are_measured() -> None:
    from app.evaluation.response_quality import assess

    report = assess("Chart dekhne ke baad lag raha hai ki marriage ke chances theek-thenek hain.",
                    "hinglish", "marriage")
    assert report.filler


def test_a_repeated_sentence_is_removed() -> None:
    from app.evaluation.response_quality import apply_safe_fixes

    text = ("Your 10th house is Sagittarius. Its lord Jupiter sits in the 6th house. "
            "Your 10th house is Sagittarius.")
    fixed, report = apply_safe_fixes(text, "english", "career")
    assert fixed == "Your 10th house is Sagittarius. Its lord Jupiter sits in the 6th house."
    assert report.repeated_removed == 1


# ══ 7. Premium knowledge corpus: retrieval plan and safety filters ═════
def test_marriage_plan_queries_this_charts_real_placements(facts) -> None:
    from app.rag.premium_knowledge import plan_queries

    plan = plan_queries("When is a good period for marriage?", "marriage", facts)
    entities = [p.entities for p in plan]
    assert {"planets": ["Mercury", "Venus"], "category_slug": "dasha"} in entities
    assert any(e.get("house_lord") == 7 and e.get("house") == 6 for e in entities)
    assert plan[-1].applies_to is None, "the question's own semantic query is always kept"
    assert len(plan) <= settings_module.settings.premium_knowledge.max_queries


def test_career_plan_uses_the_10th_lord_and_no_dasha_without_timing(facts) -> None:
    from app.rag.premium_knowledge import plan_queries

    plan = plan_queries("Mera career kaisa rahega?", "career", facts)
    assert any(p.entities.get("house_lord") == 10 and p.entities.get("house") == 6 for p in plan)
    assert not any(p.entities.get("category_slug") == "dasha" for p in plan)


@pytest.mark.parametrize("result, question, allowed", [
    ({"category_slug": "case_studies", "text": "Case: DOB 1990"}, "marriage?", False),
    ({"category_slug": "marriage", "section_title": "Timing of Divorce", "text": "Date of Marriage: 2014"},
     "marriage?", False),
    ({"category_slug": "remedies", "text": "Chant…"}, "career?", False),
    ({"category_slug": "remedies", "text": "Chant…"}, "koi upay batao", True),
    ({"category_slug": "transit", "text": "Transit of Jupiter…"}, "career?", False),
    ({"category_slug": "planets_in_houses", "planet": "Venus", "house": 7, "text": "Venus in the 7th"},
     "marriage?", True),
    ({"category_slug": "house_lords_in_houses", "house_lord": 1, "house": 12,
      "subsection_title": "Effects of I House Lord in XI House", "text": "…"}, "career?", False),
])
def test_retrieved_chunks_are_filtered_for_the_chatbot(result, question, allowed) -> None:
    from app.rag.premium_knowledge import _allowed

    assert _allowed(result, question) is allowed


def test_relaxed_mirror_image_matches_are_not_used_as_chart_evidence() -> None:
    """Found in the retrieval eval: a query for "10th lord in the 6th" came back
    (relaxed filter) with "6th lord in the 10th"; a dasha query with a nakshatra chunk."""
    from app.rag.premium_knowledge import matches_entities

    wanted = {"house_lord": 10, "house": 6, "language": "en"}
    assert matches_entities({"house_lord": 10, "house": 6, "_applies_to": "x", "_entities": wanted})
    assert not matches_entities({"house_lord": 6, "house": 10, "_applies_to": "x", "_entities": wanted})
    dasha = {"planet": "Mercury", "category_slug": "dasha", "language": "en"}
    assert not matches_entities({"planet": "Mercury", "category_slug": "nakshatra",
                                 "_applies_to": "x", "_entities": dasha})
    assert matches_entities({"category_slug": "career", "_applies_to": None, "_entities": {}})


def _premium_on(monkeypatch):
    base = settings_module.settings
    patched = dataclasses.replace(base, premium_knowledge=dataclasses.replace(
        base.premium_knowledge, url="http://premium.test", source="premium"))
    import app.nodes.rag as rag_node
    import app.rag.premium_knowledge as premium

    for module in (settings_module, premium, rag_node):
        monkeypatch.setattr(module, "settings", patched)
    return premium


def test_premium_evidence_is_annotated_deduplicated_and_capped(monkeypatch, chart) -> None:
    premium = _premium_on(monkeypatch)

    def fake_call(sub):
        base = {"category_slug": "house_lords_in_houses", "house_lord": 7, "house": 6,
                "document_title": "7TH HOUSE LORD", "section_title": "Seventh House Lord in the Sixth House",
                "subsection_title": "Effects", "score": 0.62, "text": "7th House Lord in the 6th House — Effects"}
        return [{**base, "chunk_id": "a", "_applies_to": sub.applies_to, "_priority": sub.priority},
                {**base, "chunk_id": "a", "_applies_to": sub.applies_to, "_priority": sub.priority}]

    monkeypatch.setattr(premium, "_call", fake_call)
    result = premium.retrieve_premium("When is a good period for marriage?", "marriage", chart)
    assert result is not None and result.available
    assert len(result.chunks) == 1, "the same chunk from several sub-queries appears once"
    assert result.chunks[0].text.startswith("(Applies to this chart:")
    assert result.chunks[0].source == "tradeva-premium"


def test_premium_outage_falls_back_to_the_legacy_namespace(monkeypatch, chart) -> None:
    premium = _premium_on(monkeypatch)
    import app.nodes.rag as rag_node
    from app.rag.retriever import RetrievalResult

    monkeypatch.setattr(premium, "_call", lambda sub: (_ for _ in ()).throw(ConnectionError("down")))
    legacy = RetrievalResult(chunks=[], used_filter=True, widened=False, available=True)
    monkeypatch.setattr(rag_node, "retrieve", lambda **kwargs: legacy)
    assert premium.retrieve_premium("career?", "career", chart) is None
    out = rag_node.retrieve_knowledge_node({"intent": "career", "question": "How is my career?",
                                            "kundli": chart, "history": [], "errors": []})
    assert out["retrieval"] is legacy


def test_premium_is_off_unless_configured() -> None:
    from app.rag.premium_knowledge import retrieve_premium

    if not settings_module.settings.premium_knowledge.configured:
        assert retrieve_premium("career?", "career", None) is None


# ══ 8. Metrics carry versions for evaluation ═══════════════════════════
def test_turn_metrics_record_prompt_and_validator_versions(monkeypatch, chart) -> None:
    import time

    from app.evaluation.response_quality import VALIDATOR_VERSION
    from app.nodes import generate
    from app.prompts.builder import PROMPT_VERSION

    monkeypatch.setattr(generate.memory, "save_turn", lambda *a, **k: True)
    out = generate.save_metrics_node({
        "request_id": "r", "session_id": "s", "user_id": "u", "question": "q", "response": "a",
        "started_at": time.perf_counter(), "kundli": chart, "response_style": "english",
        "language": {"language": "english", "source": "detected"},
        "validation": {"action": "regenerated"},
    })
    m = out["metrics"]
    assert m["prompt_version"] == PROMPT_VERSION and m["validator_version"] == VALIDATOR_VERSION
    assert m["language"] == "english" and m["regenerations"] == 1


# ══ 9. Found by the 100-turn benchmark (after the first upgrade run) ═══
@pytest.mark.parametrize("style, text", [
    ("whatsapp_hinglish", "Aapki Kundli mein Lagna Pisces hai aur ye personality ko nिखार deta hai."),
    ("whatsapp_hinglish", "Padhai ke liye aapki gतिविधियों par focus rahega."),
    ("hindi", "आपकी कुंडली में 7वां भाव कन्या है। spouse ko working professional banata hai aur ye career "
              "focus hota hai।"),
    ("hindi", "विवाह का विचार दशा-अंतर्दasha से किया जाता है।"),
])
def test_script_glitches_and_roman_hindi_inside_hindi_are_rejected(style, text) -> None:
    from app.prompts.language import reply_matches

    assert not reply_matches(style, text)


def test_attribution_left_dangling_by_a_removed_citation_is_cleaned() -> None:
    """"[K1] ke hisaab se …" became "… hai. ke hisaab se …" once the tag was stripped."""
    from app.evaluation.response_quality import apply_safe_fixes

    fixed, _ = apply_safe_fixes("Aapki Kundli mein 1st lord Jupiter 6th house mein hai. ke hisaab se yeh "
                                "mehnat aur seva dikhata hai.", "hinglish", "personal_growth")
    assert fixed == "Aapki Kundli mein 1st lord Jupiter 6th house mein hai. Yeh mehnat aur seva dikhata hai."


def test_machinery_sentence_that_survives_repair_is_dropped() -> None:
    from app.evaluation.response_quality import apply_safe_fixes

    fixed, report = apply_safe_fixes(
        "Your chart has Venus in the 7th house. I don't have yoga confirmations from the available data. "
        "Venus there is traditionally linked with harmony in partnership.", "english", "general_astrology")
    assert "available data" not in fixed and report.machinery_removed == 1


def test_a_kb_title_presented_as_a_yoga_is_a_claim(facts) -> None:
    from app.evaluation.chart_claims import validate_reply

    result = validate_reply('Jupiter 6th house mein hone se "Shatru Hanta" yoga banata hai.', facts)
    assert any(c.kind == "yoga" for c in result.claims)


def test_mid_word_script_switches_in_hinglish_are_written_in_roman() -> None:
    from app.prompts.language import reply_matches
    from app.prompts.terminology import normalize_terms

    text, fixes = normalize_terms("Mercury ka 6th house mein होना, shaadi mein deरी ka sanket hai; "
                                  "vyavसाय ke liye accha hai.", "hinglish")
    assert "transliterated_devanagari" in fixes
    assert text == "Mercury ka 6th house mein hona, shaadi mein deri ka sanket hai; vyavsaay ke liye accha hai."
    assert reply_matches("whatsapp_hinglish", text)


def test_missing_space_between_scripts_in_hindi_is_restored() -> None:
    from app.prompts.terminology import normalize_terms

    text, _ = normalize_terms("पार्टनरशिप के लिएharmony है और संभावनाtraditionally अच्छी है।", "hindi")
    assert text == "पार्टनरशिप के लिए harmony है और संभावना traditionally अच्छी है।"


def test_markdown_bold_is_removed_for_the_plain_text_bubble() -> None:
    from app.evaluation.response_quality import apply_safe_fixes

    fixed, _ = apply_safe_fixes("Abhi **Budh Mahadasha** chal rahi hai, jo **January 2038 tak** rahegi.",
                                "hinglish", "general_astrology")
    assert fixed == "Abhi Budh Mahadasha chal rahi hai, jo January 2038 tak rahegi."


# ══ 10. Found in the live run: chart unavailable -> invented chart ═════
def test_with_no_chart_an_invented_reading_is_replaced(empty_kundli) -> None:
    """Calculator unreachable: the model still wrote "lagna Vrishabha … 10th lord Shani …
    Shani Mahadasha 2023 to 2048". With no chart there is nothing it may state."""
    from app.nodes.validate import NO_CHART_MESSAGE, validate_reply_node

    draft = ("Aapki Kundli mein lagna Vrishabha lag rahi hai aur Chandra 10th house ka lord hai. "
             "10th lord Shani Makar mein hi 10th ghar mein hai.\n\n"
             "Abhi Shani Mahadasha chal rahi hai, jo December 2023 se December 2048 tak hai.")
    state = {"response": draft, "kundli": empty_kundli, "response_style": "whatsapp_hinglish",
             "question": "Mere career k bare me batao", "intent": "career", "messages": [],
             "session_id": "s", "user_id": "u", "request_id": "r"}
    out = validate_reply_node(state)
    assert out["response"] == NO_CHART_MESSAGE["hinglish"]
    assert out["validation"]["action"] == "no_chart_fallback"


def test_with_no_chart_general_knowledge_is_kept(empty_kundli) -> None:
    from app.nodes.validate import validate_reply_node

    draft = ("In Vedic astrology the 7th house is traditionally read for marriage and partnership. "
             "Venus is generally the significator of relationships.")
    state = {"response": draft, "kundli": empty_kundli, "response_style": "english",
             "question": "What does the 7th house mean?", "intent": "general_astrology", "messages": [],
             "session_id": "s", "user_id": "u", "request_id": "r"}
    out = validate_reply_node(state)
    assert out.get("response", draft) == draft


def test_with_no_chart_a_repair_cannot_reintroduce_a_reading(monkeypatch, empty_kundli) -> None:
    """Live: the draft was only in the wrong language; the language repair then
    wrote a full fictional reading. The no-chart guard runs on the final text too."""
    from app.nodes.validate import NO_CHART_MESSAGE, validate_reply_node

    _flags(monkeypatch, chart_validation_mode="repair")
    invented = ("Aapki Kundli mein 10th house Taurus hai aur uska lord Venus 2nd house mein hai. "
                "Abhi Venus Mahadasha chal rahi hai, jo 2025 ke end tak rahegi.")
    _, client = use_nvidia(monkeypatch, [reply(invented)])
    state = {"response": "Your career depends on many things.", "kundli": empty_kundli,
             "response_style": "whatsapp_hinglish", "question": "Mere career k bare me batao",
             "intent": "career", "messages": [{"role": "system", "content": "s"}],
             "session_id": "s", "user_id": "u", "request_id": "r"}
    out = validate_reply_node(state)
    assert len(client.requests) == 1, "the language repair ran"
    assert out["response"] == NO_CHART_MESSAGE["hinglish"]


# ══ 11. Found in the final benchmark rounds ════════════════════════════
def test_wrong_sign_on_a_house_is_caught_in_hindi(facts) -> None:
    """"आपकी कुंडली में 7वां भाव कुंभ राशि है, जिसका स्वामी शुक्र है" — the 7th is Virgo
    (lord Mercury). Two gaps: "7वां" was not a house, and "विराजमान " (ending in न)
    made the whole sentence look negated."""
    from app.evaluation.chart_claims import validate_reply

    result = validate_reply("आपकी कुंडली में 7वां भाव कुंभ राशि है, जिसका स्वामी शुक्र है और वह 7वें भाव में "
                            "ही विराजमान है।", facts)
    kinds = {(c.kind, c.status) for c in result.claims}
    assert ("house_sign", "contradicted") in kinds and ("lordship", "contradicted") in kinds
    assert validate_reply("Your 7th house is Virgo.", facts).claims[0].status == "verified"
    assert validate_reply("बुध 7वें भाव में नहीं है।", facts).claims == [], "real negation still skipped"


def test_a_pure_language_switch_gets_a_fixed_confirmation_without_a_model_call(monkeypatch) -> None:
    """Benchmark: "Kya aap hindi me jawab de sakte hai" produced a looping apology."""
    from app.nodes.generate import generate_response_node
    from app.prompts.language_policy import SWITCH_CONFIRMATION

    _, client = use_nvidia(monkeypatch, [])
    out = generate_response_node({"messages": [{"role": "user", "content": "x"}],
                                  "question": "Kya aap hindi me jawab de sakte hai",
                                  "language": {"language": "hindi", "switch_only": True},
                                  "history": [], "session_id": "s", "user_id": "u", "request_id": "r"})
    assert out["response"] == SWITCH_CONFIRMATION["hindi"] and client.requests == []


def test_explain_in_hindi_re_answers_the_previous_question() -> None:
    from app.prompts.builder import question_note

    note = question_note("Can you explain in Hindi?", switch_only=True,
                         previous_question="When is a good period for marriage?")
    assert "When is a good period for marriage?" in note and "previous question" in note


def test_a_looping_short_sentence_is_collapsed() -> None:
    from app.evaluation.response_quality import apply_safe_fixes

    loop = "मुझे हिंदी में उत्तर देना चाहिए था। " + "मुझे खेद है। " * 20
    fixed, report = apply_safe_fixes(loop.strip(), "hindi", "unknown")
    assert fixed.count("मुझे खेद है") == 1 and report.repeated_removed == 19


def test_talking_about_the_passages_is_machinery() -> None:
    from app.evaluation.response_quality import mentions_machinery

    assert mentions_machinery("The passages note that partners can be demanding.")
    assert not mentions_machinery("Classical texts traditionally read this as a demanding partnership.")


def test_lordship_phrasings_found_in_benchmark_review_are_checked(facts) -> None:
    """"Mercury is the 6th and 9th lord" and "Your 7th house is Virgo, with Venus as its
    lord" both reached users in the benchmark (Mercury rules the 4th and 7th here)."""
    from app.evaluation.chart_claims import validate_reply

    bad = validate_reply("Mercury is the 6th and 9th lord, and it sits in the house of health.", facts)
    assert {(c.value, c.status) for c in bad.claims if c.kind == "lordship"} == {(6, "contradicted"),
                                                                               (9, "contradicted")}
    venus = validate_reply("Your 7th house is Virgo, with Venus as its lord placed in the 7th house.", facts)
    assert any(c.kind == "lordship" and c.status == "contradicted" for c in venus.claims)
    good = validate_reply("Your 7th house is Virgo, with its lord Mercury placed in the 6th house.", facts)
    assert all(c.status == "verified" for c in good.claims)


# ══ 12. Natural Hinglish (real-user report, second round) ══════════════
# Root cause of most of these: the letter-by-letter Devanagari->Roman fallback
# added in round one ("आपको" -> "aapako", "पार्टनर" -> "paartanar").
@pytest.mark.parametrize("wrong, right", [
    ("paartanar", "partner"),
    ("committment", "commitment"),
    ("maanasik kanekshan", "emotional connection"),
    ("aapako", "aapko"),
    ("bataate hain kee", "batate hain ki"),
    ("soch-samajh ke bolta hai", "soch-samajhkar baat karta hai"),
    ("rileshanship", "relationship"),
    ("parsnaliti", "personality"),
    ("kariyar", "career"),
    ("kamyunikeshan", "communication"),
    ("oporchunities", "opportunities"),
    ("fainenshiyal growth", "financial growth"),
    ("mairij", "marriage"),
    ("praiktikal", "practical"),
    ("intellijent", "intelligent"),
    ("enalisis", "analysis"),
    ("konfidens", "confidence"),
    ("responsibiliti", "responsibility"),
    ("kompatibiliti", "compatibility"),
    ("profeshanal", "professional"),
    ("divelapment", "development"),
    ("ejukeshan", "education"),
    ("committment period", "commitment period"),
])
def test_hinglish_vocabulary_is_normalised(wrong, right) -> None:
    from app.prompts.hinglish import normalize_hinglish

    assert normalize_hinglish(wrong)[0] == right


def test_the_screenshot_sentence_is_repaired() -> None:
    from app.prompts.hinglish import hinglish_issues, normalize_hinglish

    fixed, _ = normalize_hinglish("Aapako partner pasand aaega jo soch-samajh ke bolta hai. Jyotishi bataate "
                                  "hain kee paartanar ke saath committment aur maanasik kanekshan zaroori hai.")
    assert fixed == ("Aapko partner pasand aaega jo soch-samajhkar baat karta hai. Jyotishi batate hain ki "
                     "partner ke saath commitment aur emotional connection zaroori hai.")
    assert hinglish_issues(fixed) == {}


def test_devanagari_words_are_romanised_the_way_people_write() -> None:
    from app.prompts.transliterate import to_roman

    assert to_roman("आपको ऐसा पार्टनर पसंद आएगा जो कमिटमेंट और मानसिक कनेक्शन को महत्व देता है। "
                    "ज्योतिषी बताते हैं की") == (
        "aapko aisa partner pasand aayega jo commitment aur emotional connection ko mahatva deta hai. "
        "jyotishi batate hain ki")


def test_an_unknown_devanagari_word_is_not_guessed_in_a_hinglish_reply() -> None:
    """A guess is how "paartanar" happened; an unknown word is left for the language repair."""
    from app.prompts.language import reply_matches
    from app.prompts.terminology import normalize_terms

    text = "Aapki Kundli mein yeh जानकारी zaroori hai aur career ke liye accha sanket hai."
    out, _ = normalize_terms(text, "hinglish")
    assert "जानकारी" in out
    assert not reply_matches("whatsapp_hinglish", out), "still flagged, so the repair rewrites it"


def test_correct_english_and_chart_facts_are_never_touched() -> None:
    from app.prompts.hinglish import normalize_hinglish

    text = ("Aapki Kundli mein 7th house Virgo hai, uska lord Mercury 6th house mein hai, aur Venus "
            "Antardasha June 2024 se April 2027 tak chalegi. Partner practical aur professional ho sakta hai, "
            "aur relationship mein commitment aur communication important rahenge.")
    assert normalize_hinglish(text) == (text, [])


def test_masculine_possessive_before_a_postposition_takes_the_oblique_form() -> None:
    from app.prompts.hinglish import normalize_hinglish

    assert normalize_hinglish("Aapka career mein growth ki opportunities ban sakti hain.")[0] == \
        "Aapke career mein growth ki opportunities ban sakti hain."
    # the postposition belongs to "Pisces" here: unchanged
    assert normalize_hinglish("Aapka Lagna Pisces mein hai.")[0] == "Aapka Lagna Pisces mein hai."
    # feminine stays feminine
    assert normalize_hinglish("Aapki Kundli mein Venus hai.")[0] == "Aapki Kundli mein Venus hai."


def test_remaining_hindi_style_spellings_trigger_one_targeted_repair(monkeypatch, chart) -> None:
    from app.nodes.validate import validate_reply_node

    _flags(monkeypatch, chart_validation_mode="repair")
    fixed = ("Aapki Kundli mein Venus 7th house mein hai, isliye partner ke saath emotional connection aur "
             "communication strong reh sakta hai.")
    _, client = use_nvidia(monkeypatch, [reply(fixed)])
    draft = ("Aapki Kundli mein Venus 7th house mein hai, isliye paarthanar ke saath kanaikshan aur "
             "kommyunikeshan strong reh sakta hai.")
    out = validate_reply_node(_state(chart, draft, style="whatsapp_hinglish", question="Mera partner kaisa hoga?"))
    assert len(client.requests) == 1
    note = client.requests[0]["messages"][-1]["content"]
    assert "spelled the Hindi way" in note and "kommyunikeshan" in note
    assert out["response"] == fixed


def test_mixed_tum_and_aap_is_reported() -> None:
    from app.prompts.hinglish import hinglish_issues

    assert "address_mixed" in hinglish_issues("Aapki Kundli mein Venus hai aur tumhara partner practical hoga.")


# ── combined: segmentation + Hinglish + grounding together ────────────
COMBINED_HINGLISH_DRAFT = (
    "Aapki Kundli mein 7th house Virgo hai aur usmein Venus hai, jabki 7th lord Mercury 6th house mein hai. "
    "Traditionally isse marriage mein practical approach aur committment ka sanket milta hai.\n\n"
    "Aapako aisa paartanar pasand aa sakta hai jo soch-samajh ke bolta hai aur maanasik kanekshan ko "
    "importance deta hai.\n\n"
    "Abhi Mercury Mahadasha mein Venus Antardasha June 2024 se April 2027 tak chal rahi hai, jise marriage "
    "ke liye supportive maana jata hai. Final result baaki placements par bhi depend karta hai.")


def test_hinglish_fixes_and_segments_hold_together(monkeypatch, chart) -> None:
    from app.evaluation.chart_claims import validate_reply
    from app.kundli.chart_facts import build_chart_facts
    from app.nodes.validate import validate_reply_node
    from app.prompts.hinglish import hinglish_issues
    from app.prompts.language import reply_matches
    from app.prompts.segments import segment_reply

    _, client = use_nvidia(monkeypatch, [])
    out = validate_reply_node(_state(chart, COMBINED_HINGLISH_DRAFT, style="whatsapp_hinglish",
                                     question="Mera partner kaisa hoga aur shaadi kab hogi?"))
    final = out.get("response", COMBINED_HINGLISH_DRAFT)
    assert client.requests == [], "fixed deterministically, no regeneration"
    segments = segment_reply(final)
    assert len(segments) == 3
    for segment in segments:
        assert reply_matches("whatsapp_hinglish", segment), segment
        assert hinglish_issues(segment) == {}, segment
        assert segment.rstrip().endswith(".")
    assert "June 2024 se April 2027 tak" in segments[2], "date range intact and unchanged"
    assert "partner" in segments[1] and "emotional connection" in segments[1] and "aapko" in segments[1].lower()
    assert not validate_reply(final, build_chart_facts(chart)).contradictions


# ── found in the round-3 benchmark ────────────────────────────────────
def test_a_single_devanagari_word_in_a_long_hinglish_reply_fails_the_language_check() -> None:
    """"…career ki ओर इशारा karta hai" passed the 0.5% tolerance, so no repair ran."""
    from app.prompts.language import reply_matches

    long = "Aapki Kundli mein 10th house Sagittarius hai aur uska lord Jupiter 6th house mein hai. " * 6
    assert not reply_matches("whatsapp_hinglish", long + "Yeh career ki ओर इशारा karta hai.")


def test_devanagari_left_after_the_repair_is_romanised_not_shown() -> None:
    from app.evaluation.response_quality import apply_safe_fixes
    from app.prompts.language import reply_matches

    text = "Aapki Kundli career ki ओर इशारा karta है। Ghar ka घरेलू माहौल shant rahega."
    fixed, _ = apply_safe_fixes(text, "hinglish", "career")
    assert fixed == "Aapki Kundli career ki or ishara karta hai. Ghar ka gharelu maahaul shant rahega."
    assert reply_matches("whatsapp_hinglish", fixed)


def test_simple_questions_are_told_not_to_add_a_dasha_paragraph() -> None:
    from app.prompts.segments import SEGMENT_SHAPE_RULES

    from app.prompts.answer_depth import classify_depth, length_line

    assert classify_depth("What is my Lagna?", "general_astrology") == "simple"
    assert "ONE message" in length_line("simple")
    assert "only when the question is about timing or the Dasha itself" in SEGMENT_SHAPE_RULES


def test_a_closing_offer_is_removed_like_a_follow_up_question() -> None:
    from app.evaluation.response_quality import apply_safe_fixes

    fixed, report = apply_safe_fixes("Aapka Lagna Pisces hai, aur iska lord Jupiter 6th house mein hai. "
                                     "Koi aur chart fact poochni ho to pooch sakti hain!", "hinglish",
                                     "general_astrology")
    assert fixed == "Aapka Lagna Pisces hai, aur iska lord Jupiter 6th house mein hai."
    assert report.follow_up_removed == 1


def test_a_reply_cut_off_mid_sentence_is_trimmed_to_its_last_sentence() -> None:
    """Benchmark ex-10: a Hindi analysis ended "…चुनौतियाँ आ सकती हैं, लेकिन शुक्र"."""
    from app.evaluation.response_quality import apply_safe_fixes
    from app.prompts.segments import is_complete

    fixed, _ = apply_safe_fixes("शुक्र सप्तम भाव में है और विवाह के लिए अनुकूल है। बुध की स्थिति के कारण कभी-कभी "
                                "देरी आ सकती है, लेकिन शुक्र", "hindi", "marriage")
    assert fixed == "शुक्र सप्तम भाव में है और विवाह के लिए अनुकूल है।" and is_complete(fixed)


def test_aap_never_gets_a_gendered_verb() -> None:
    from app.prompts.hinglish import normalize_hinglish

    assert normalize_hinglish("Isse aap emotionally sensitive ho sakti hain.")[0] == \
        "Isse aap emotionally sensitive ho sakte hain."
    # a third person keeps its own grammar
    assert normalize_hinglish("Partner thodi demanding ho sakti hai.")[0] == "Partner thodi demanding ho sakti hai."


def test_the_last_resort_reply_is_in_the_users_language() -> None:
    from app.nodes.validate import SAFE_FALLBACK, safe_fallback
    from app.prompts.language import reply_matches

    assert reply_matches("whatsapp_hinglish", safe_fallback("whatsapp_hinglish"))
    assert reply_matches("hindi", safe_fallback("hindi"))
    assert safe_fallback("english") == SAFE_FALLBACK


# ══ 13. Round 4: production hardening ══════════════════════════════════
def test_roman_hindi_inside_a_hindi_reply_is_written_in_devanagari() -> None:
    """Round-3 benchmark: "आपकी कुंडली में 5th house ka sign Cancer है … mein situate है"."""
    from app.prompts.language import reply_matches
    from app.prompts.terminology import normalize_terms

    raw = ("आपकी कुंडली में 5th house ka sign Cancer है, जिस का lord Moon 1st house (Lagna) mein situate है। "
           "Moon aapki Kundli mein 10.0° par है।")
    assert not reply_matches("hindi", raw)
    fixed, fixes = normalize_terms(raw, "hindi")
    assert fixed == ("आपकी कुंडली में 5th house का sign Cancer है, जिस का lord चंद्र 1st house (लग्न) में स्थित है। "
                     "चंद्र आपकी कुंडली में 10.0° par है।")
    assert reply_matches("hindi", fixed) and any(f.startswith("devanagarized") for f in fixes)


def _gate_state(chart, text, style):
    return {**_state(chart, text, style=style, question="Mera career kaisa rahega?", intent="career"),
            "validation": {}}


def test_the_language_gate_rewrites_a_reply_that_is_still_wrong(monkeypatch, chart) -> None:
    from app.nodes.validate import _language_gate

    good = "Aapki Kundli mein 10th house Sagittarius hai, aur iska lord Jupiter 6th house mein hai."
    _, client = use_nvidia(monkeypatch, [reply(good)])
    out = _language_gate(_gate_state(chart, "", "whatsapp_hinglish"),
                         {"response": "Your 10th house is Sagittarius and its lord Jupiter sits in the 6th house."})
    assert out["response"] == good and out["validation"]["language_gate"] == "rewritten"
    assert len(client.requests) == 1


def test_the_language_gate_sends_the_localised_fallback_when_the_rewrite_fails(monkeypatch, chart) -> None:
    from app.nodes.validate import _language_gate, safe_fallback

    still_wrong = "Your 10th house is Sagittarius and its lord Jupiter sits in the 6th house."
    _, client = use_nvidia(monkeypatch, [reply(still_wrong)])
    out = _language_gate(_gate_state(chart, "", "hindi"), {"response": still_wrong})
    assert out["response"] == safe_fallback("hindi") and out["validation"]["language_gate"] == "fallback"


def test_the_gate_checks_every_bubble_not_only_the_whole_reply(monkeypatch, chart) -> None:
    """A long Hindi reply can pass as a whole while one bubble is entirely English."""
    from app.nodes.validate import _segments_ok

    hindi = ("आपकी कुंडली में सप्तम भाव कन्या राशि का है और उसमें शुक्र विराजमान है, जो विवाह के लिए पारंपरिक "
             "रूप से अनुकूल माना जाता है। ") * 3
    mixed = hindi.strip() + "\n\nThe current Venus Antardasha runs from June 2024 to April 2027 and supports this."
    assert not _segments_ok("hindi", mixed)
    assert _segments_ok("hindi", hindi.strip())


@pytest.mark.parametrize("style", ["english", "whatsapp_hinglish", "hindi"])
def test_every_canned_reply_is_in_the_users_language(style) -> None:
    from app.nodes.generate import failure_message
    from app.nodes.validate import NO_CHART_MESSAGE, safe_fallback
    from app.prompts.language import deflection, reply_matches
    from app.prompts.language_policy import LANGUAGE_OF_STYLE, SWITCH_CONFIRMATION

    language = LANGUAGE_OF_STYLE[style]
    for text in (failure_message(style), safe_fallback(style), NO_CHART_MESSAGE[language],
                 deflection(style), SWITCH_CONFIRMATION[language]):
        assert reply_matches(style, text), (style, text)


def test_the_fallback_model_receives_the_same_language_lock(monkeypatch) -> None:
    """Language preference must survive a primary-model failure: the fallback
    model gets the identical message list, language lock included."""
    from app.nodes.generate import generate_response_node
    from tests.llm_fakes import http_error

    lock = "LANGUAGE LOCK: reply in natural Roman-script Hinglish"
    messages = [{"role": "system", "content": f"rules\n\n{lock}"}, {"role": "user", "content": "career?"}]
    _, client = use_nvidia(monkeypatch, [http_error(503)] * 4 + [reply("Aapka career steady rahega.")] * 4)
    out = generate_response_node({"messages": messages, "question": "career?", "language": {},
                                  "session_id": "s", "user_id": "u", "request_id": "r"})
    assert out["response"] == "Aapka career steady rahega."
    models = {r["model"] for r in client.requests}
    assert len(models) >= 2, "the fallback model answered"
    assert all(r["messages"][0]["content"].endswith(lock) for r in client.requests)


def test_identical_premium_sub_queries_hit_the_cache(monkeypatch) -> None:
    import app.rag.premium_knowledge as premium

    base = settings_module.settings
    patched = dataclasses.replace(base, premium_knowledge=dataclasses.replace(
        base.premium_knowledge, url="http://premium.test", source="premium", cache_seconds=3600))
    monkeypatch.setattr(premium, "settings", patched)
    premium.cache_clear()
    calls = []

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"chunk_id": "c1", "score": 0.6, "text": "7th lord in the 6th"}]}

    import httpx

    monkeypatch.setattr(httpx, "post", lambda *a, **k: calls.append(1) or _Resp())
    sub = premium.SubQuery("7th lord in the 6th house", {"house_lord": 7, "house": 6}, "7th lord", 0)
    first, second = premium._call(sub), premium._call(sub)
    assert calls == [1], "the second identical query is served from the cache"
    assert first == second and first[0]["_applies_to"] == "7th lord"
    premium.cache_clear()


def test_a_sign_named_on_a_house_is_not_read_as_the_moon_sign(facts) -> None:
    """Round-4 probe: "10th house (Sagittarius rashi)" cost a repair as a Moon-sign contradiction."""
    from app.evaluation.chart_claims import validate_reply

    result = validate_reply("Career ke liye 10th house (Sagittarius rashi) dekha jata hai.", facts)
    assert not result.contradictions
    assert validate_reply("Aapki Moon sign Sagittarius hai.", facts).contradictions, "real Moon-sign error still caught"


def test_copied_knowledge_base_headings_are_lower_cased() -> None:
    from app.evaluation.response_quality import apply_safe_fixes

    fixed, _ = apply_safe_fixes("Venus Antardasha mein aap Charitable Inclination aur Beautiful Clothing ki taraf "
                                "jhukaav mehsoos kar sakte hain.", "hinglish", "general_astrology")
    assert "charitable inclination aur beautiful clothing" in fixed
    assert "Venus Antardasha" in fixed, "proper nouns keep their capitals"


def test_mechanical_hinglish_reference_pairs_are_fixed_deterministically() -> None:
    from app.prompts.hinglish import normalize_hinglish
    from benchmarks.hinglish_reference import PAIRS

    mechanical = [p for p in PAIRS if p.mechanical]
    assert len(mechanical) >= 5
    for pair in mechanical:
        assert normalize_hinglish(pair.awkward)[0] == pair.natural, pair.awkward


def test_detailed_life_area_questions_are_described_in_the_segment_rules() -> None:
    from app.prompts.answer_depth import classify_depth, length_line

    for question, intent in [("Tell me about my career.", "career"), ("Meri shaadi kab hogi?", "marriage"),
                             ("मेरी आर्थिक स्थिति कैसी रहेगी?", "finance")]:
        assert classify_depth(question, intent) == "detailed", question
    assert "FOUR messages" in length_line("detailed") and "70–110 words" in length_line("detailed")


def test_a_self_description_instead_of_an_answer_is_repaired() -> None:
    """Round-4 benchmark: "क्या मेरी शादी हो चुकी है?" was answered with
    "मैं एनवीआईडीआईए के शोधकर्ताओं द्वारा बनाया गया एक भाषा मॉडल हूं"."""
    from app.evaluation.response_quality import mentions_machinery

    assert mentions_machinery("नहीं, मैं एनवीआईडीआईए के शोधकर्ताओं द्वारा बनाया गया एक भाषा मॉडल हूं।")
    assert mentions_machinery("I am a language model created by NVIDIA.")
    assert not mentions_machinery("Your 7th house is Virgo and Venus sits there.")


def test_an_attribution_phrase_goes_with_its_removed_citation_tag() -> None:
    """Live browser test: "इस स्थिति को [K1] के अनुसार …" was shown as "इस स्थिति को के अनुसार …"."""
    from app.nodes.validate import _TAG_ATTRIBUTION

    assert _TAG_ATTRIBUTION.sub("", "इस स्थिति को [K1] के अनुसार विवाह देर से होता है।") == \
        "इस स्थिति को विवाह देर से होता है।"
    assert _TAG_ATTRIBUTION.sub("", "According to [K2], the 7th lord in the 6th brings delays.") == \
        "the 7th lord in the 6th brings delays."
    assert _TAG_ATTRIBUTION.sub("", "Venus 7th house mein hai [K1].") == "Venus 7th house mein hai [K1]."


def test_an_english_leaning_hinglish_bubble_does_not_send_the_reply_to_the_fallback() -> None:
    """Round-4 quiet run: the screenshot case (ex-06) was replaced by the fallback
    because one Hinglish bubble leaned on English terms."""
    from app.nodes.validate import _segment_language_ok

    assert _segment_language_ok("whatsapp_hinglish", "Venus Antardasha June 2024 se April 2027 tak chalegi.")
    assert _segment_language_ok("whatsapp_hinglish",
                                "Traditionally, 7th lord ka 6th house mein hona working professional partner dikhata hai.")
    assert not _segment_language_ok("whatsapp_hinglish", "Traditionally this placement indicates that your partner "
                                    "will be a hardworking professional with a strong focus on career growth.")
    assert not _segment_language_ok("whatsapp_hinglish", "Aapki Kundli mein शुक्र 7th house mein hai.")


def test_repairs_are_told_to_keep_the_paragraph_structure() -> None:
    """Round-4: a repair collapsed a four-topic career analysis into one bubble."""
    from app.evaluation.chart_claims import ValidationResult
    from app.nodes.validate import _correction_note

    assert "keep the same paragraphs" in _correction_note(ValidationResult(), True, "english", False)


def test_a_failed_gate_rewrite_does_not_discard_a_reply_whose_language_is_right(monkeypatch, chart) -> None:
    """Degraded-service run: the gate's rewrite call timed out and correct-language
    Hindi answers were replaced by the fallback because one bubble was weak."""
    from app.nodes.validate import _language_gate
    from tests.llm_fakes import http_error

    hindi = ("आपकी कुंडली में सप्तम भाव कन्या राशि का है और उसमें शुक्र विराजमान है, जो विवाह के लिए पारंपरिक "
             "रूप से अनुकूल माना जाता है। ") * 3
    text = hindi.strip() + "\n\nThe current Venus Antardasha runs from June 2024 to April 2027 and supports this."
    use_nvidia(monkeypatch, [http_error(503)] * 12)
    out = _language_gate(_gate_state(chart, "", "hindi"), {"response": text})
    assert out["response"] == text and out["validation"]["language_gate"] == "original_whole_reply_ok"


# ══ 14. Round 5: short bubbles and selective follow-ups ════════════════
# The user's screenshot: one ~120-word Hinglish bubble ("…Passage yeh bhi kehta
# hai ki…"), a long English bubble, and a closing "please share your exact
# birth time" offer although the chart was already computed.
SCREENSHOT_LONG_BUBBLE = (
    "Aapki Kundli mein 10th house Sagittarius hai, jis ka lord Jupiter 7th house mein Virgo mein hai aur wahan "
    "Jupiter aur Ketu ke saath yuti banayi hai. Traditionally 10th lord ki 7th mein sthiti business aur "
    "partnership skills ka sign maani jaati hai, jo career mein growth ki taraf le jaati hai. Aapki Kundli mein "
    "Rahu 1st house mein hai, jo leadership quality aur confidence deta hai. 11th house lord Saturn 5th house "
    "mein hai, jo creative fields ya teaching ki taraf le ja sakta hai.")


def _wc(text: str) -> int:
    return len(text.split())


def test_an_oversized_bubble_is_split_at_sentence_boundaries() -> None:
    from app.prompts.segments import MAX_WORDS, segment_reply

    segments = segment_reply(SCREENSHOT_LONG_BUBBLE)
    assert len(segments) >= 2
    assert all(_wc(s) <= MAX_WORDS for s in segments)
    assert " ".join(segments) == SCREENSHOT_LONG_BUBBLE, "no word lost, no sentence cut"
    assert all(s.endswith(".") for s in segments)


def test_a_single_long_sentence_is_never_cut() -> None:
    from app.prompts.segments import segment_reply

    sentence = ("Your current Mahadasha is Jupiter, running from November 2020 to November 2036, and within it the "
                "Mercury Antardasha runs from July 2025 to October 2027, which traditionally favours communication, "
                "analysis, contracts, writing and every kind of work that depends on careful thinking.")
    assert segment_reply(sentence) == [sentence]


def test_the_cap_never_merges_into_an_oversized_bubble() -> None:
    """v3 merged a 5th paragraph into its neighbour (the 120-word bubble)."""
    from app.prompts.segments import MAX_WORDS, segment_reply

    paragraphs = [f"Point {i}: this message explains one chart factor in two short sentences. It stays brief."
                  for i in range(1, 6)]
    segments = segment_reply("\n\n".join(paragraphs))
    assert all(_wc(s) <= MAX_WORDS + 10 for s in segments)


@pytest.mark.parametrize("raw, count", [
    ("Your Lagna is Aquarius 4.34°.", 1),                                                     # A simple
    ("Your 10th house is Sagittarius and its lord Jupiter sits in the 7th house.\n\n"           # B normal
     "Traditionally this links your career with partnerships and business.", 2),
    (SCREENSHOT_LONG_BUBBLE + "\n\nAbhi aapki Jupiter Mahadasha November 2020 se November 2036 tak chal rahi hai, "
     "aur Mercury Antardasha July 2025 se October 2027 tak rahegi.", 5),                        # C detailed
])
def test_bubble_counts_follow_the_shape_of_the_answer(raw, count) -> None:
    from app.prompts.segments import segment_reply

    assert len(segment_reply(raw)) == count


def test_a_numbered_suggestion_stands_as_its_own_bubble() -> None:
    from app.prompts.segments import segment_reply

    raw = ("1. Business: Jupiter in the 7th house traditionally supports partnership-based work.\n\n"
           "2. Teaching: Saturn in the 5th house is traditionally linked with teaching and research.")
    assert len(segment_reply(raw)) == 2


def test_a_follow_up_question_gets_its_own_last_bubble() -> None:
    from app.prompts.segments import segment_reply

    raw = ("Your 10th lord Jupiter sits in the 7th house, which traditionally favours partnership work. "
           "Would you like to explore business, teaching or corporate career options?")
    segments = segment_reply(raw)
    assert segments[-1] == "Would you like to explore business, teaching or corporate career options?"
    assert len(segments) == 2


# ── E. follow-up decisions ────────────────────────────────────────────
@pytest.mark.parametrize("question, intent, previous, allowed", [
    ("Tell me about some career fields", "career", None, True),
    ("Meri shaadi kab hogi aur partner kaisa hoga?", "marriage", None, True),
    ("What is my Lagna?", "general_astrology", None, False),
    ("Mera lagna kya hai?", "general_astrology", None, False),
    ("Am I married or not?", "marriage", None, False),
    ("hello there", "smalltalk", None, False),
    ("Tell me about my career", "career", "Your 10th house is Sagittarius. Would you like to explore teaching?",
     False),
])
def test_follow_ups_are_offered_only_when_a_next_step_exists(question, intent, previous, allowed) -> None:
    from app.prompts.followup import decide

    assert decide(question, intent, "english", previous).allowed is allowed


@pytest.mark.parametrize("question, good", [
    ("Would you like to explore business, teaching or corporate career options?", True),
    ("Kya aap partner ka nature ya shaadi ki timing jaanna chahenge?", True),
    ("Do you want to know more?", False),
    ("Kya aap kuch aur jaanna chahenge?", False),
    ("What kind of partner are you looking for?", False),
    ("Are you considering changing your job?", False),
])
def test_only_specific_topic_offers_count_as_good_follow_ups(question, good) -> None:
    from app.prompts.followup import is_good_followup

    assert is_good_followup(question) is good


def test_a_permitted_follow_up_is_kept_and_extras_removed() -> None:
    from app.evaluation.response_quality import apply_safe_fixes

    text = ("Your 10th lord Jupiter sits in the 7th house, which traditionally favours partnership work. "
            "Do you want to know more? Would you like to explore business, teaching or corporate career options?")
    fixed, report = apply_safe_fixes(text, "english", "career", "Tell me about career fields",
                                     followup_allowed=True)
    assert fixed.endswith("Would you like to explore business, teaching or corporate career options?")
    assert "know more" not in fixed and report.follow_up_kept == 1 and report.follow_up_removed == 1


def test_no_follow_up_survives_when_none_is_permitted() -> None:
    from app.evaluation.response_quality import apply_safe_fixes

    text = "Your Lagna is Aquarius, ruled by Saturn. Would you like to explore your personality traits?"
    fixed, _ = apply_safe_fixes(text, "english", "general_astrology", "What is my Lagna?", followup_allowed=False)
    assert fixed == "Your Lagna is Aquarius, ruled by Saturn."


def test_the_same_follow_up_is_not_asked_twice() -> None:
    from app.prompts.followup import is_good_followup

    previous = "Your 10th lord sits in the 7th. Would you like to explore business or teaching career options?"
    assert not is_good_followup("Would you like to explore teaching or business career options?", previous)


def test_the_screenshot_closing_lines_are_removed() -> None:
    """"If you have a specific timing or future question, please share your exact birth time…"
    after a chart was already computed; and "Passage yeh bhi kehta hai…"."""
    from app.evaluation.response_quality import apply_safe_fixes, mentions_machinery

    text = ("Your 10th house is Sagittarius and its lord Jupiter sits in the 7th house. "
            "This analysis is based on your Kundli facts and traditional reading. If you have a specific timing "
            "or future question, please share your exact birth time so I can give you a detailed analysis.")
    fixed, _ = apply_safe_fixes(text, "english", "career", "career?", followup_allowed=True)
    assert "birth time" not in fixed
    assert mentions_machinery("Passage yeh bhi kehta hai ki career mein growth hogi.")


def test_the_prompt_carries_the_follow_up_decision(chart) -> None:
    from app.prompts.builder import build_system_prompt
    from app.prompts.followup import FollowUpDecision, prompt_line
    from app.rag.retriever import RetrievalResult

    yes = prompt_line(FollowUpDecision(True, "x", "business or teaching"))
    no = prompt_line(FollowUpDecision(False, "x"))
    system = build_system_prompt("career", chart, RetrievalResult.unavailable("x"), chat_bubbles=True,
                                 response_style="english", followup_line=yes)
    assert yes in system and "ONE short (under 15 words), specific question" in yes
    assert "do NOT end with a question" in no


# ── H. persistence: reload rebuilds the same bubbles ─────────────────
def test_history_reload_returns_the_same_bubbles(monkeypatch) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from app.api import auth, server
    from app.memory.redis_memory import Message
    from app.prompts.reply_format import reply_messages

    reply_text = SCREENSHOT_LONG_BUBBLE + "\n\nWould you like to explore business or teaching career options?"
    stored = [Message("user", "Mere career k bare me batao", 1.0), Message("assistant", reply_text, 2.0)]
    seen = {}

    def fake_history(key, limit=None):
        seen["key"] = key
        return stored

    monkeypatch.setattr(server.memory, "get_history", fake_history)
    monkeypatch.setattr(server, "parse_anon_token", lambda token: "abc" if token == "tok" else None)
    client = fastapi_testclient.TestClient(server.app)
    body = client.get("/chat/history", params={"session_id": "kundli-x"},
                      headers={"X-Chat-Session-Token": "tok"}).json()
    assert seen["key"] == "anon:abc|kundli-x", "only the caller's own conversation"
    assert body["turns"][0] == {"role": "user", "text": "Mere career k bare me batao"}
    assert body["turns"][1]["messages"] == reply_messages(reply_text)
    assert len(body["turns"][1]["messages"]) >= 3
    # no identity -> no history, and no token issued
    empty = client.get("/chat/history", params={"session_id": "kundli-x"}).json()
    assert empty["turns"] == []
    del auth


@pytest.mark.parametrize("question", [
    "क्या आप विवाह के अन्य पहलुओं के बारे में भी जानना चाहेंगे?",
    "Kya aap jaanna chahenge ki 7th house mein kya koi other planet bhi hai, ya aap kisi specific area (jaise "
    "career ya emotional connection) par zyada focus karna chahenge?",
])
def test_vague_or_double_follow_ups_from_the_smoke_run_are_rejected(question) -> None:
    from app.prompts.followup import is_good_followup

    assert not is_good_followup(question)


def test_a_sentence_starting_with_to_is_not_treated_as_a_cut_date_range() -> None:
    """Round-5 benchmark: "To know your actual marital status, …" was glued to the
    previous bubble as if it continued a date range."""
    from app.prompts.segments import segment_reply

    raw = ("You are running Mercury Mahadasha (January 2021 to January 2038) with Venus Antardasha (June 2024 "
           "to April 2027).\n\nTo know your actual marital status, the chart alone cannot confirm it.")
    assert len(segment_reply(raw)) == 2
    cut = "Your Venus Antardasha runs from June 2024\n\nto April 2027, a supportive period for marriage."
    assert len(segment_reply(cut)) == 1


def test_four_content_bubbles_and_the_follow_up_make_five() -> None:
    """Multi-bubble brief (2026-11-18): four content bubbles + one follow-up = 5, never more."""
    from app.prompts.segments import segment_reply

    content = [f"Point {i} explains one chart factor briefly in a single short sentence of its own."
               for i in range(1, 5)]
    raw = "\n\n".join(content) + "\n\nWould you like to explore the partner's nature or marriage timing?"
    segments = segment_reply(raw)
    assert len(segments) == 5
    assert segments[-1] == "Would you like to explore the partner's nature or marriage timing?"
    six = "\n\n".join(content + ["Point 5 explains one more chart factor in a short sentence of its own."])
    assert len(segment_reply(six + "\n\nWould you like to explore marriage timing?")) == 5


def test_a_mostly_devanagari_hinglish_reply_is_not_romanised_wholesale() -> None:
    """Live browser test: a whole Hindi answer to a Hinglish question was romanised
    into "paramparik roop se … lanbe samay"; it must go to the language rewrite instead."""
    from app.evaluation.response_quality import apply_safe_fixes
    from app.prompts.language import reply_matches

    hindi = "आपकी कुंडली में सातवां भाव सिंह है, और उसका स्वामी सूर्य पहले भाव में है।"
    fixed, _ = apply_safe_fixes(hindi, "hinglish", "marriage")
    assert fixed == hindi and not reply_matches("whatsapp_hinglish", fixed)


def test_an_all_lower_case_english_reply_is_recapitalised() -> None:
    from app.evaluation.response_quality import apply_safe_fixes

    fixed, _ = apply_safe_fixes("your lagna is aquarius, and the 10th house is scorpio. you are running the "
                                "rahu mahadasha until august 2038.", "english", "career")
    assert fixed == ("Your Lagna is Aquarius, and the 10th house is Scorpio. You are running the Rahu Mahadasha "
                     "until August 2038.")


@pytest.mark.parametrize("question", [
    "क्या आप कोई अन्य विवरण जानना चाहेंगे?",          # live browser test, round 5
    "Would you like more details?",
    "Kya aap aur details jaanna chahenge?",
])
def test_generic_detail_offers_are_not_follow_ups(question) -> None:
    from app.prompts.followup import is_good_followup

    assert not is_good_followup(question)
