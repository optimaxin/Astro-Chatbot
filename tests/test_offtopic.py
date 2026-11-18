"""Off-topic handling: deterministic rules first, then the classifier's off_topic
label, routed to the existing localized scope reply with no astrology model call.
Astrology vocabulary (match, score, career, house, period, phone) must never trigger it."""

from __future__ import annotations

import dataclasses

import pytest

from app.nodes.classifier import (
    IntentResult,
    classify_intent_node,
    decide_routing,
    route_after_classify,
    scope_reply_node,
)
from app.safety.block_messages import block_message
from app.safety.categories import SafetyCategory
from app.safety.offtopic_rules import in_scope_signal, offtopic_rule
from app.safety.relevance import check_relevance
from tests.llm_fakes import reply, use_nvidia
from tests.test_model_roles import turn_env  # noqa: F401 — fixture: real graph, fake Redis, fixed chart

BOLT = "nvidia/nemotron-3.5-bolt-e4b"

OFF_TOPIC = {
    "english": ["My phone is not charging", "Who is the best cricketer ever?", "What is the capital of France?",
                "What is the score of yesterday's match", "Best biryani in Delhi?", "Who invented the telephone?",
                "laptop won't turn on", "price of petrol", "Latest bollywood movie review"],
    "hinglish": ["Mera phone charge nahi ho raha", "IPL ka score kya hai", "Kal ka match kaun jeeta?",
                 "laptop hang ho raha hai", "fransh ki rajdhani kya hai", "biryani recipe batao",
                 "WiFi nahi chal raha"],
    "hindi": ["मेरा फोन चार्ज नहीं हो रहा", "क्रिकेट मैच का स्कोर बताओ", "फ्रांस की राजधानी क्या है?",
              "बिरयानी की रेसिपी बताओ"],
}
# Astrology-related uses of the same words, and personal-life questions that mention a device or a sport.
IN_SCOPE = [
    "Is my match compatible with my partner?", "What is my guna milan score?", "career period kab shuru hoga?",
    "Will I become a cricketer?", "Meri shaadi kab hogi?", "my 10th house is Gemini, what does it say?",
    "Is Saturn period good for my phone business?", "My mobile app startup, will it succeed?",
    "Kya meri kundli mein technology career hai?", "Is it a good time to buy a new phone?", "When will I get promotion?",
    "Does the score of my chart look good?", "Will my team win the match I am playing tomorrow?",
    "mera business kaisa chalega, laptop shop hai", "What does Rahu period mean for me?",
    "Kundli match karwana hai", "Mera cricket career kaisa rahega?", "मेरा करियर कैसा रहेगा?", "मेरी शादी कब होगी?",
    "मेरा जन्म फोन के पास हुआ, मेरा भविष्य कैसा है?", "Aur batao", "Kab tak?", "Why?",
]


# ── 8. off-topic in English, Hindi and Hinglish: caught before any model runs ──
@pytest.mark.parametrize("language", ["english", "hinglish", "hindi"])
def test_off_topic_messages_are_caught_deterministically(language) -> None:
    for text in OFF_TOPIC[language]:
        check = check_relevance(text)
        assert check.blocked and check.category == SafetyCategory.out_of_scope, text
        assert check.rule.startswith(("offtopic_", "sports_", "general_", "weather", "news")), (text, check.rule)


@pytest.mark.parametrize("style,lang", [("english", "en"), ("whatsapp_hinglish", "hinglish"), ("hindi", "hi")])
def test_the_off_topic_reply_is_the_existing_localized_scope_reply(style, lang) -> None:
    check = check_relevance("My phone is not charging", lang)
    assert check.reply == block_message(SafetyCategory.out_of_scope, lang)
    assert len(check.reply) > 40


# ── 9. astrology terminology is never off-topic ───────────────────────
@pytest.mark.parametrize("text", IN_SCOPE)
def test_astrology_and_personal_life_messages_are_never_flagged_off_topic(text) -> None:
    assert offtopic_rule(text) is None, text
    assert not check_relevance(text).blocked, text


def test_the_in_scope_guard_recognises_vocabulary_and_personal_framing() -> None:
    for text in ("kundli ke hisaab se", "Saturn transit", "7th house lord", "मेरा करियर", "will my career grow"):
        assert in_scope_signal(text), text
    for text in ("my phone is slow", "IPL score", "recipe of pasta"):
        assert not in_scope_signal(text), text


# ── the classifier: structured output and routing ─────────────────────
def test_the_structured_classification_validates_and_defaults_safely() -> None:
    full = IntentResult.model_validate({"intent": "Off Topic", "confidence": 0.9, "sub_intent": "phone repair",
                                        "is_astrology_related": False, "is_follow_up": False, "language": "hinglish",
                                        "routing_decision": "scope_reply"})
    assert (full.intent, full.routing_decision, full.is_astrology_related) == ("off_topic", "scope_reply", False)
    legacy = IntentResult.model_validate({"intent": "career", "confidence": 0.8})      # an old-style reply
    assert legacy.routing_decision == "answer" and legacy.is_astrology_related and not legacy.is_follow_up
    assert IntentResult.model_validate({"intent": "career", "routing_decision": "nonsense"}).routing_decision == "answer"
    with pytest.raises(ValueError):
        IntentResult.model_validate({"intent": "weather_report"})


@pytest.mark.parametrize("result,method,history,expected", [
    (IntentResult(intent="off_topic", confidence=0.9), "llm", False, "scope_reply"),
    (IntentResult(intent="off_topic", confidence=0.5), "llm", False, "clarify"),            # unsure: ask, don't block
    (IntentResult(intent="off_topic", confidence=0.9, is_follow_up=True), "llm", True, "answer"),
    (IntentResult(intent="career", confidence=0.9), "llm", False, "answer"),
    (IntentResult(intent="career", confidence=0.2), "llm", False, "clarify"),               # ambiguous, no context
    (IntentResult(intent="career", confidence=0.2), "llm", True, "answer"),                 # with history: keep going
    (IntentResult(intent="marriage", confidence=0.6, routing_decision="clarify"), "llm", False, "clarify"),
    (IntentResult(intent="general_astrology", confidence=0.3), "keyword-fallback", False, "answer"),
    (IntentResult(intent="marriage", confidence=0.9, is_follow_up=True), "llm", True, "answer"),
])
def test_routing_decision(result, method, history, expected) -> None:
    assert decide_routing(result, method, history) == expected


def test_a_contextual_follow_up_keeps_the_conversation_intent(monkeypatch) -> None:
    from app.nodes.classifier import classify
    from app.config import settings as settings_module

    s = settings_module.settings
    object.__setattr__(s, "flags", dataclasses.replace(s.flags, classifier_mode="llm"))
    use_nvidia(monkeypatch, [reply('{"intent": "marriage", "confidence": 0.9, "is_follow_up": true, '
                                   '"is_astrology_related": true, "language": "hinglish", "routing_decision": "answer"}')])
    result, method = classify("Aur kya indication hai?", "user: Meri shaadi kab hogi?\nassistant: Venus Dasha.")
    assert (result.intent, result.is_follow_up, method) == ("marriage", True, "llm")


def test_the_graph_routes_by_the_decision() -> None:
    assert route_after_classify({"routing_decision": "scope_reply"}) == "scope_reply"
    assert route_after_classify({"routing_decision": "clarify"}) == "clarify"
    assert route_after_classify({"routing_decision": "answer"}) == "retrieve_knowledge"
    assert route_after_classify({}) == "retrieve_knowledge"


@pytest.mark.parametrize("style,lang", [("english", "en"), ("whatsapp_hinglish", "hinglish"), ("hindi", "hi")])
def test_the_scope_reply_node_is_localized_and_makes_no_call(style, lang, monkeypatch) -> None:
    _, client = use_nvidia(monkeypatch, [])
    out = scope_reply_node({"language": {"style": style}, "session_id": "s"})
    assert out["response"] == block_message(SafetyCategory.out_of_scope, lang)
    assert out["model_used"] == "rule:scope" and out["generation_failed"] is False and not client.requests


# ── end to end through the real graph ─────────────────────────────────
def _turn(monkeypatch, question, script, **kw):
    from app.graph.chatbot_graph import run_turn

    _, client = use_nvidia(monkeypatch, script)
    state = run_turn(question, f"off-{abs(hash(question)) % 10**6}", "u1", **kw)
    return state, client


@pytest.mark.parametrize("language,style", [("english", "english"), ("hinglish", "whatsapp_hinglish"),
                                            ("hindi", "hindi")])
def test_a_rule_caught_off_topic_message_makes_no_model_call_at_all(monkeypatch, turn_env, language, style) -> None:  # noqa: F811
    state, client = _turn(monkeypatch, OFF_TOPIC[language][0], [], response_style=style)
    assert not client.requests and state["metrics"]["llm_calls"] == 0
    assert state["metrics"]["response_source"] == "deterministic" and state["metrics"]["intent"] == "safety_gated"
    assert state["response"] == block_message(SafetyCategory.out_of_scope, {"english": "en", "hinglish": "hinglish",
                                                                         "hindi": "hi"}[language])


def test_a_message_the_rules_miss_is_labelled_off_topic_by_the_classifier_and_gets_no_astrology_call(
        monkeypatch, turn_env) -> None:  # noqa: F811
    question = "Which is better, a Samsung or a OnePlus?"
    assert not check_relevance(question).blocked, "no deterministic rule covers this one"
    state, client = _turn(monkeypatch, question,
                          [reply('{"intent": "off_topic", "confidence": 0.93, "is_astrology_related": false, '
                                 '"routing_decision": "scope_reply"}')], response_style="english")
    assert client.models == [BOLT], "only the classifier ran: no answer, repair, judge or retrieval call"
    assert state["intent"] == "off_topic" and state["response"] == block_message(SafetyCategory.out_of_scope, "en")
    assert state["metrics"]["response_source"] == "deterministic" and state["metrics"]["llm_calls"] == 1
    assert state["classification"]["is_astrology_related"] is False


def test_an_unsure_off_topic_label_asks_instead_of_blocking(monkeypatch, turn_env) -> None:  # noqa: F811
    state, client = _turn(monkeypatch, "Which is better, a Samsung or a OnePlus?",
                          [reply('{"intent": "off_topic", "confidence": 0.5, "routing_decision": "clarify"}')],
                          response_style="english")
    assert client.models == [BOLT] and "?" in state["response"]
    assert state["response"] != block_message(SafetyCategory.out_of_scope, "en")
    assert state["metrics"]["response_source"] == "deterministic"
