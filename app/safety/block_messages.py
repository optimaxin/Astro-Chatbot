"""Canned replies served when a gate fires, in the user's own language.

A gated turn never reaches the model, so these strings *are* the answer. Two
things follow from that.

First, they are written in all three languages the product serves. Answering a
Devanagari user in English at the exact moment they disclosed self-harm is the
worst possible time to switch language, so the resolver mirrors whatever
`response_style.detect_style` saw in their message.

Second, the helplines are real and India-first, matching where the users are:
iCall 9152987821, KIRAN 1800-599-0019 (24x7), emergency 112, Women Helpline
181, Childline 1098. If these are ever edited, they must be verified against
the operators' current published numbers — a wrong number here is worse than
no number.
"""

from __future__ import annotations

from app.safety.categories import SafetyCategory

DEFAULT_LANG = "en"
SUPPORTED_LANGS: tuple[str, ...] = ("en", "hi", "hinglish")

#: `response_style.detect_style` ids -> the canned-language keys used here.
_STYLE_TO_LANG: dict[str, str] = {
    "english": "en",
    "hindi": "hi",
    "whatsapp_hinglish": "hinglish",
    "formal": "en",
    "detailed": "en",
}


def resolve_lang(style_id: str | None) -> str:
    """Map a response-style id onto a canned-reply language key."""
    if not style_id:
        return DEFAULT_LANG
    key = style_id.strip().lower()
    if key in SUPPORTED_LANGS:
        return key
    return _STYLE_TO_LANG.get(key, DEFAULT_LANG)


# The assistant is openly an AI here (BASE_SYSTEM_PROMPT says so plainly), so
# unlike a human-persona product these replies may say "I". What they must not
# do is soften the referral: the whole point is that the user goes to a doctor,
# a lawyer, a SEBI-registered advisor or a helpline rather than to a chart.
_FALLBACK: dict[str, str] = {
    "en": (
        "I can't help with this one directly — it needs a qualified professional. "
        "If there's something else on your mind, I'm here for chart-based guidance."
    ),
    "hi": (
        "इसमें मैं सीधे मदद नहीं कर सकता — इसके लिए किसी योग्य पेशेवर की ज़रूरत है। "
        "अगर कुछ और पूछना हो, तो कुंडली से जुड़े सवालों पर मैं ज़रूर मदद करूँगा।"
    ),
    "hinglish": (
        "Ismein main seedhe madad nahi kar sakta — iske liye qualified professional "
        "chahiye. Agar kuch aur poochna ho, to kundli se jude sawaalon par main "
        "zaroor help karunga."
    ),
}


BLOCK_MESSAGES: dict[SafetyCategory, dict[str, str]] = {
    SafetyCategory.self_harm_crisis: {
        "en": (
            "I hear you, and I'm really sorry you're in this much pain. Please reach "
            "out to someone right now — you don't have to go through this alone:\n\n"
            "- iCall (free, India): 9152987821 (Mon-Sat, 8am-10pm)\n"
            "- KIRAN (24x7): 1800-599-0019\n"
            "- Emergency: 112\n\n"
            "If you can, please talk to a trusted person near you or a mental health "
            "professional today. A birth chart is not the right kind of help for what "
            "you're going through right now."
        ),
        "hi": (
            "मैं सुन रहा हूँ, और मुझे सच में दुख है कि आप इतनी तकलीफ़ में हैं। कृपया अभी किसी से "
            "बात करें — आप अकेले नहीं हैं:\n\n"
            "- iCall (मुफ़्त, भारत): 9152987821 (सोम-शनि, सुबह 8 से रात 10 बजे)\n"
            "- KIRAN (24x7): 1800-599-0019\n"
            "- आपातकाल: 112\n\n"
            "कृपया आज ही किसी विश्वसनीय व्यक्ति या मानसिक स्वास्थ्य पेशेवर से बात करें। इस समय "
            "जो आप झेल रहे हैं, उसके लिए कुंडली सही मदद नहीं है।"
        ),
        "hinglish": (
            "Main sun raha hoon, aur mujhe sach mein dukh hai ki aap itni takleef mein "
            "hain. Please abhi kisi se baat karein — aap akele nahi hain:\n\n"
            "- iCall (free, India): 9152987821 (Mon-Sat, 8am-10pm)\n"
            "- KIRAN (24x7): 1800-599-0019\n"
            "- Emergency: 112\n\n"
            "Aaj hi kisi trusted person ya mental health professional se zaroor baat "
            "karein. Is waqt aap jo jhel rahe hain, uske liye kundli sahi madad nahi hai."
        ),
    },
    SafetyCategory.abuse_coercion: {
        "en": (
            "What you're describing isn't something a chart should decide — your "
            "safety matters more than any reading. If you're in immediate danger, "
            "please call 112. For confidential support, the Women Helpline (181) and "
            "Childline (1098) are available 24x7."
        ),
        "hi": (
            "आप जो बता रहे हैं, वह कुंडली से तय करने की बात नहीं है — आपकी सुरक्षा किसी भी "
            "भविष्यवाणी से ज़्यादा ज़रूरी है। अगर आप तुरंत ख़तरे में हैं, तो 112 पर कॉल करें। "
            "गोपनीय सहायता के लिए महिला हेल्पलाइन (181) और चाइल्डलाइन (1098) 24x7 उपलब्ध हैं।"
        ),
        "hinglish": (
            "Aap jo bata rahe hain, woh kundli se decide karne ki baat nahi hai — aapki "
            "safety kisi bhi reading se zyada zaroori hai. Agar aap turant khatre mein "
            "hain, to 112 par call karein. Confidential support ke liye Women Helpline "
            "(181) aur Childline (1098) 24x7 available hain."
        ),
    },
    SafetyCategory.medical: {
        "en": (
            "I can't diagnose anything or tell you what to take — that needs a "
            "qualified doctor. Please see a medical professional about this. Once you "
            "have their guidance, I'm glad to look at timing and supportive periods "
            "in your chart."
        ),
        "hi": (
            "मैं न कोई निदान कर सकता हूँ और न ही दवा बता सकता हूँ — इसके लिए योग्य डॉक्टर "
            "चाहिए। कृपया पहले किसी चिकित्सक से मिलें। उनकी सलाह के बाद, कुंडली में समय और "
            "अनुकूल अवधि पर मैं ज़रूर बात करूँगा।"
        ),
        "hinglish": (
            "Main na koi diagnosis kar sakta hoon aur na hi dawai bata sakta hoon — "
            "iske liye qualified doctor chahiye. Pehle kisi doctor se milein. Unki "
            "guidance ke baad, kundli mein timing aur supportive periods par main "
            "zaroor baat karunga."
        ),
    },
    SafetyCategory.legal: {
        "en": (
            "I can't give legal advice or predict how a specific case will go. Please "
            "talk to a qualified lawyer about your situation. After that, I'm happy to "
            "talk through general timing themes with you."
        ),
        "hi": (
            "मैं कानूनी सलाह नहीं दे सकता, और न ही यह बता सकता हूँ कि कोई मुकदमा किस ओर "
            "जाएगा। कृपया अपनी स्थिति के बारे में किसी योग्य वकील से बात करें। उसके बाद, समय "
            "से जुड़े सामान्य विषयों पर हम बात कर सकते हैं।"
        ),
        "hinglish": (
            "Main legal advice nahi de sakta, aur na hi yeh bata sakta hoon ki koi case "
            "kis taraf jayega. Please apni situation ke baare mein kisi qualified lawyer "
            "se baat karein. Uske baad, general timing themes par hum baat kar sakte hain."
        ),
    },
    SafetyCategory.financial: {
        "en": (
            "I can't tell you whether to invest, buy, sell or borrow — that's a "
            "financial decision and a chart should not be the thing that settles it. "
            "Please consult a SEBI-registered financial advisor. Once you have that "
            "advice, I can look at supportive timing windows with you."
        ),
        "hi": (
            "निवेश करना है या नहीं, खरीदना है या बेचना — यह मैं नहीं बता सकता। यह वित्तीय "
            "निर्णय है और इसे कुंडली के आधार पर तय नहीं करना चाहिए। कृपया SEBI-रजिस्टर्ड "
            "फाइनेंशियल एडवाइज़र से सलाह लें। उसके बाद, अनुकूल समय पर हम बात कर सकते हैं।"
        ),
        "hinglish": (
            "Invest karna hai ya nahi, kharidna hai ya bechna — yeh main nahi bata sakta. "
            "Yeh financial decision hai aur ise kundli ke basis par tay nahi karna chahiye. "
            "Please SEBI-registered financial advisor se salah lein. Uske baad, supportive "
            "timing windows par hum baat kar sakte hain."
        ),
    },
    SafetyCategory.out_of_scope: {
        "en": (
            "That's outside what I can help with — I read birth charts, so I stick to "
            "your chart and the life questions around it. But I'd genuinely like to "
            "help there: ask me about your career, relationships, health, finances, or "
            "the timing of something on your mind."
        ),
        "hi": (
            "यह मेरे दायरे से बाहर है — मैं कुंडली पढ़ता हूँ, इसलिए मैं आपकी कुंडली और जीवन से "
            "जुड़े सवालों तक ही सीमित रहता हूँ। पर उसमें मैं ज़रूर मदद करना चाहूँगा: अपने करियर, "
            "रिश्तों, स्वास्थ्य, धन, या किसी बात के समय के बारे में पूछिए।"
        ),
        "hinglish": (
            "Yeh mere dayre se bahar hai — main kundli padhta hoon, isliye main aapki "
            "kundli aur life ke sawaalon tak hi rehta hoon. Par usmein main zaroor help "
            "karna chahunga: apne career, rishton, health, paise, ya kisi baat ki timing "
            "ke baare mein poochiye."
        ),
    },
    SafetyCategory.prompt_injection: {
        "en": (
            "I can't act on that. If you have an astrology question — about your chart, "
            "timing, or a situation in your life — I'm glad to help."
        ),
        "hi": (
            "मैं इस पर अमल नहीं कर सकता। अगर आपके पास ज्योतिष से जुड़ा कोई सवाल है — अपनी "
            "कुंडली, समय, या जीवन की किसी स्थिति के बारे में — तो मुझे मदद करके खुशी होगी।"
        ),
        "hinglish": (
            "Main is par amal nahi kar sakta. Agar aapke paas jyotish se juda koi sawaal "
            "hai — apni kundli, timing, ya life ki kisi situation ke baare mein — to main "
            "khushi se madad karunga."
        ),
    },
}

# Both PII rules serve the same "please don't paste that here" reply.
_PII_MESSAGE: dict[str, str] = {
    "en": (
        "Please don't share phone numbers or card details here — I don't need them, "
        "and this isn't a secure place for them. For a chart reading I only need your "
        "date, time and place of birth."
    ),
    "hi": (
        "कृपया यहाँ फ़ोन नंबर या कार्ड की जानकारी न भेजें — मुझे उनकी ज़रूरत नहीं है, और यह "
        "उसके लिए सुरक्षित जगह नहीं है। कुंडली के लिए मुझे सिर्फ़ आपकी जन्म तिथि, समय और "
        "स्थान चाहिए।"
    ),
    "hinglish": (
        "Please yahan phone number ya card details na bhejein — mujhe unki zaroorat "
        "nahi hai, aur yeh uske liye secure jagah nahi hai. Kundli ke liye mujhe sirf "
        "aapki birth date, time aur place chahiye."
    ),
}

BLOCK_MESSAGES[SafetyCategory.pii_phone] = _PII_MESSAGE
BLOCK_MESSAGES[SafetyCategory.pii_credit_card] = _PII_MESSAGE


def block_message(category: SafetyCategory, lang: str | None = None) -> str:
    """The canned reply for `category`, in `lang` (an id or a language key).

    Falls back to English, and then to a generic professional-referral line, so
    a category added without copy still produces something safe to ship rather
    than an empty bubble or a KeyError on the user's worst day.
    """
    key = resolve_lang(lang)
    variants = BLOCK_MESSAGES.get(category)
    if not variants:
        return _FALLBACK.get(key, _FALLBACK[DEFAULT_LANG])
    return variants.get(key) or variants.get(DEFAULT_LANG) or _FALLBACK[DEFAULT_LANG]


def crisis_block_message(lang: str | None = None) -> str:
    """The self-harm reply. Named separately because it is the one callers grep for."""
    return block_message(SafetyCategory.self_harm_crisis, lang)


__all__ = [
    "BLOCK_MESSAGES",
    "DEFAULT_LANG",
    "SUPPORTED_LANGS",
    "block_message",
    "crisis_block_message",
    "resolve_lang",
]
