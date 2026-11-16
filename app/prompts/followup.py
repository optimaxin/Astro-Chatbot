"""When may a reply end with ONE follow-up question — and which question?

The decision is made in code, per turn, before generation (so the prompt can
say "ask one" or "ask none"), and enforced again after generation (so a
question the model adds anyway is removed, and a permitted one is checked).

Rules (2026-11 brief):
* at most one follow-up per assistant turn, as the last bubble;
* only when a natural next step exists: a topical question (career, marriage,
  relationships, finance, growth, dasha) answered in more than a line;
* never after a simple factual question, a life-fact question, a language
  switch, a greeting, or when the previous assistant turn already ended with
  a question;
* it must OFFER a specific next topic ("Would you like to explore business,
  teaching or corporate career options?"), never probe the user's private
  plans ("What kind of partner are you looking for?") and never be generic
  ("Do you want to know more?").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TOPICAL_INTENTS = {"career", "marriage", "relationship", "love", "finance", "personal_growth", "general_astrology"}

# Next steps a follow-up may offer, per intent and language. Examples for the
# prompt, not templates: the model words its own question.
_SUGGESTIONS = {
    "career": {"english": "business, a job, or a specific field like teaching or finance",
               "hinglish": "business, job ya teaching/finance jaisa koi specific field",
               "hindi": "व्यवसाय, नौकरी या शिक्षण जैसा कोई विशेष क्षेत्र"},
    "marriage": {"english": "the partner's nature or marriage timing",
                 "hinglish": "partner ka nature ya shaadi ki timing",
                 "hindi": "जीवनसाथी का स्वभाव या विवाह का समय"},
    "relationship": {"english": "relationship compatibility or family life",
                     "hinglish": "relationship compatibility ya family life",
                     "hindi": "रिश्तों में अनुकूलता या पारिवारिक जीवन"},
    "love": {"english": "relationship compatibility or marriage timing",
             "hinglish": "relationship compatibility ya shaadi ki timing",
             "hindi": "रिश्तों में अनुकूलता या विवाह का समय"},
    "finance": {"english": "savings and investments or income growth periods",
                "hinglish": "savings ya income growth ke periods",
                "hindi": "बचत या आय बढ़ने के समय"},
    "personal_growth": {"english": "strengths to build on or the current Dasha's lessons",
                        "hinglish": "aapki strengths ya current Dasha ke lessons",
                        "hindi": "आपकी क्षमताएँ या वर्तमान दशा की सीख"},
    # Not "the next Antardasha": the model named a wrong next period in the
    # follow-up itself (conciseness benchmark: "अगली अंतर्दशा केतु की…").
    "general_astrology": {"english": "how the current Dasha affects career or relationships",
                          "hinglish": "current Dasha ka career ya relationships par asar",
                          "hindi": "वर्तमान दशा का करियर या रिश्तों पर असर"},
}

# A question that is a plain fact lookup ("What is my Lagna?"): answer and stop.
_SIMPLE_FACT = re.compile(
    r"\b(?:what|which)\s+is\s+my\s+(?:lagna|ascendant|rashi|moon sign|sun sign|nakshatra|mahadasha|dasha)\b"
    r"|\b(?:which|what)\s+house\s+is\b|\bwhere\s+is\s+my\s+\w+\b"
    r"|\b(?:mera|meri)\s+(?:lagna|rashi|nakshatra)\s+(?:kya|kaun|kaunsa|kaunsi)\b"
    r"|\b(?:kis|kaunse)\s+(?:house|bhav|ghar)\s+mein\b"
    r"|(?:मेरा|मेरी)\s+(?:लग्न|राशि|नक्षत्र)\s+(?:क्या|कौन)|किस\s+भाव\s+में",
    re.I)

# An OFFER of a next topic, in each language.
_OFFER = re.compile(
    r"\b(?:would you like (?:to|me to)|shall i|should i|do you want me to|want me to)\b"
    r"|\b(?:kya aap|aap)\b[^?]{0,80}\b(?:jaanna|samajhna|dekhna|explore karna|janna)\s+(?:chahenge|chahte hain|chahengi)\b"
    r"|\bbataun\b|\bbatau\b|\bsamjhaun\b"
    r"|क्या आप[^?]{0,80}(?:जानना|समझना|देखना)\s+चाहेंगे|बताऊँ|बताऊं|समझाऊँ",
    re.I)
# Questions that only invite "more" without naming anything.
_GENERIC = re.compile(
    r"\b(?:know more|learn more|anything else|any other (?:question|query)|something else|any questions|"
    r"kuch aur|aur kuch|koi aur sawal|koi dusra sawal|other aspects|anya pahlu\w*|anya pehlu\w*|"
    r"(?:kisi|any) specific area|(?:other|more|further|any) details?|anya vivaran|aur details?|"
    r"koi aur jaankari)\b|कुछ और|कोई और सवाल|अन्य पहलु|अन्य विवरण|और विवरण|कोई और जानकारी|अधिक जानकारी",
    re.I)


@dataclass(frozen=True)
class FollowUpDecision:
    allowed: bool
    reason: str
    suggestion: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"allowed": self.allowed, "reason": self.reason}


def _ends_with_question(text: str | None) -> bool:
    return bool(text) and text.rstrip().endswith(("?", "？"))


def decide(question: str, intent: str | None, language: str, previous_assistant: str | None = None,
           switch_only: bool = False) -> FollowUpDecision:
    """Pure. `previous_assistant` is the last assistant message in history."""
    from app.prompts.builder import _BEYOND_CHART, _LIFE_FACT

    q = (question or "").strip()
    if switch_only:
        return FollowUpDecision(False, "language switch")
    if (intent or "") not in TOPICAL_INTENTS:
        return FollowUpDecision(False, f"intent {intent or 'unknown'}")
    if _ends_with_question(previous_assistant):
        return FollowUpDecision(False, "previous turn already asked a question")
    if _SIMPLE_FACT.search(q) or len(q.split()) <= 3:
        return FollowUpDecision(False, "simple factual question")
    if _LIFE_FACT.search(q) or _BEYOND_CHART.search(q):
        return FollowUpDecision(False, "question a chart cannot answer literally")
    suggestion = _SUGGESTIONS.get(intent or "", {}).get(language, "")
    return FollowUpDecision(True, "topical question with a natural next step", suggestion)


def prompt_line(decision: FollowUpDecision) -> str:
    if not decision.allowed:
        return ("FOLLOW-UP: do NOT end with a question this time. Finish the answer and stop.")
    example = f" (for example, offering {decision.suggestion})" if decision.suggestion else ""
    return ("FOLLOW-UP: if a natural next step exists, end with ONE short (under 15 words), specific question that offers it"
            f"{example}, as its own last message. Offer a chart topic — never ask about the user's private plans "
            "or preferences, never ask a generic \"want to know more?\", never more than one question. If the "
            "answer is already complete, end without a question.")


_SCAFFOLD = set("""
would you like to know explore learn more about me shall should do want tell explain look at next your the a an or
and of in for on what how which this that i it its is are be with into from deeper detail details further
kya aap jaanna janna chahenge chahengi chahte hain samajhna dekhna explore karna ke baare mein ki ka ko aapke
aapki aapka main bataun batau samjhaun aur ya bhi is iske uske se par hai
क्या आप जानना चाहेंगे समझना देखना के बारे में की का को आपके आपकी आपका मैं बताऊँ बताऊं समझाऊँ और या भी इस से पर है
""".split())


def _topic_words(text: str) -> set[str]:
    from app.evaluation.conciseness import _STOP

    return {w for w in re.findall(r"[\wऀ-ॿ]+", (text or "").lower())
            if len(w) > 2 and w not in _SCAFFOLD and w not in _STOP}


def offers_what_was_said(sentence: str, *said: str | None) -> bool:
    """The follow-up offers a topic the answer (or the question, or the previous
    reply) already covered: "क्या आप जीवनसाथी का स्वभाव जानना चाहेंगे?" right
    after three bubbles on the partner's nature (live browser test, Hindi)."""
    offered = _topic_words(sentence)
    known = set().union(*(_topic_words(t) for t in said if t))
    return bool(offered) and len(offered & known) / len(offered) >= 0.6


def is_good_followup(sentence: str, previous_assistant: str | None = None, answer: str | None = None,
                     question: str | None = None) -> bool:
    """A permitted follow-up: an offer of a specific next topic, not a repeat."""
    s = (sentence or "").strip()
    if not s.endswith(("?", "？")) or not _OFFER.search(s) or _GENERIC.search(s):
        return False
    # One short question: longer ones were two questions glued together
    # ("…ya aap kisi specific area par focus karna chahenge?").
    if len(s.split()) > 18 or s.count("?") > 1:
        return False
    if (answer or question or previous_assistant) and offers_what_was_said(s, answer, question, previous_assistant):
        return False
    if previous_assistant:
        prev_q = [p for p in re.split(r"(?<=[.!?।])\s+", previous_assistant.strip()) if p.endswith("?")]
        norm = lambda t: set(re.findall(r"[\wऀ-ॿ]+", t.lower()))  # noqa: E731
        for p in prev_q[-1:]:
            a, b = norm(p), norm(s)
            if a and len(a & b) / len(a | b) > 0.6:
                return False          # the same question again, reworded
    return True


__all__ = ["FollowUpDecision", "decide", "prompt_line", "is_good_followup", "TOPICAL_INTENTS"]
