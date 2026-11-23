"""Canned replies served when a gate fires, in the user's own language.

A gated turn never reaches the model, so these strings *are* the answer. Two
things follow from that.

First, they are written in all three languages the product serves. Answering a
Devanagari user in English at the exact moment they disclosed self-harm is the
worst possible time to switch language, so the resolver mirrors whatever
`response_style.detect_style` saw in their message.

Second, the helplines are real and India-first, matching where the users are:
emergency 112, Tele-MANAS 14416, iCall 9152987821, Women Helpline 181,
Childline 1098. If these are ever edited, they must be verified against
the operators' current published numbers — a wrong number here is worse than
no number.

Verified 2026-11-23 against official sources (PIB, DGHS, iCall/TISS, MWCD):
112 national emergency (police, ambulance, fire; ERSS), Tele-MANAS 14416 or
1800-891-4416 (MoHFW, free, 24x7, many languages) as the first mental-health
line, iCall 9152987821 (TISS, Mon-Sat 10am-8pm), Women Helpline 181 and
Childline 1098 (both integrated with 112). KIRAN 1800-599-0019 is deliberately
not listed: it reportedly merged into Tele-MANAS and its current status could
not be confirmed.
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
    # Safety replies are ONE paragraph with a list (single newlines), so they ship
    # as one bubble at once: the chat paces multi-bubble replies, and an emergency
    # message must not arrive over several seconds.
    SafetyCategory.self_harm_crisis: {
        "en": (
            "I hear you, and I'm really sorry you're in this much pain. Are you safe right now — "
            "have you taken anything or hurt yourself? If you have, please call 112 or go to the "
            "nearest hospital immediately.\n"
            "- Emergency: 112\n"
            "- Tele-MANAS (free, 24x7): 14416 or 1800-891-4416\n"
            "- iCall (free, India): 9152987821 (Mon-Sat, 10am-8pm)\n"
            "Please don't stay alone — call or go to someone you trust right now. A birth chart is "
            "not the right kind of help for what you're going through, but people can help you, and you are not alone in this."
        ),
        "hi": (
            "मैं सुन रहा हूँ, और मुझे सच में दुख है कि आप इतनी तकलीफ़ में हैं। क्या आप अभी सुरक्षित "
            "हैं — क्या आपने कुछ खा लिया है या खुद को चोट पहुँचाई है? अगर हाँ, तो तुरंत 112 पर कॉल "
            "करें या सबसे नज़दीकी अस्पताल जाएँ।\n"
            "- आपातकाल: 112\n"
            "- Tele-MANAS (मुफ़्त, 24x7): 14416 या 1800-891-4416\n"
            "- iCall (मुफ़्त, भारत): 9152987821 (सोम-शनि, सुबह 10 से शाम 8 बजे)\n"
            "कृपया अकेले न रहें — अभी किसी भरोसेमंद व्यक्ति को फ़ोन करें या उनके पास जाएँ। आप जो "
            "झेल रहे हैं, उसके लिए कुंडली सही मदद नहीं है, लेकिन लोग आपकी मदद कर सकते हैं — आप अकेले नहीं हैं।"
        ),
        "hinglish": (
            "Main sun raha hoon, aur mujhe sach mein dukh hai ki aap itni takleef mein hain. Kya aap "
            "abhi safe hain — kya aapne kuch kha liya hai ya khud ko hurt kiya hai? Agar haan, to "
            "turant 112 par call karein ya sabse paas ke hospital jaayein.\n"
            "- Emergency: 112\n"
            "- Tele-MANAS (free, 24x7): 14416 or 1800-891-4416\n"
            "- iCall (free, India): 9152987821 (Mon-Sat, 10am-8pm)\n"
            "Please akele mat rahiye — abhi kisi trusted insaan ko call karein ya unke paas jaayein. "
            "Is takleef ke liye kundli sahi madad nahi hai, lekin log aapki madad kar sakte hain — aap akele nahi hain."
        ),
    },
    SafetyCategory.poisoning_emergency: {
        "en": (
            "I'm really worried about you, and I want you to be safe. Please treat this as a medical "
            "emergency right now. Have you already taken something? "
            "If you have swallowed poison, pills or any chemical, call 112 or go to the nearest "
            "hospital immediately — don't wait for symptoms.\n"
            "- Emergency: 112\n"
            "- Tele-MANAS (free, 24x7): 14416 or 1800-891-4416\n"
            "Please call someone you trust right now to come and stay with you. If you haven't taken "
            "anything yet, keep it away from you."
        ),
        "hi": (
            "मुझे आपकी बहुत चिंता है, और मैं चाहता हूँ कि आप सुरक्षित रहें। कृपया इसे अभी मेडिकल इमरजेंसी मानें। "
            "क्या आपने कुछ ले लिया है? अगर आपने ज़हर, गोलियाँ या "
            "कोई केमिकल निगल लिया है, तो तुरंत 112 पर कॉल करें या सबसे नज़दीकी अस्पताल जाएँ — लक्षणों का "
            "इंतज़ार न करें।\n"
            "- आपातकाल: 112\n"
            "- Tele-MANAS (मुफ़्त, 24x7): 14416 या 1800-891-4416\n"
            "कृपया किसी भरोसेमंद व्यक्ति को अभी फ़ोन करें कि वह आपके पास आकर रहे। अगर अभी तक कुछ नहीं लिया "
            "है, तो उसे खुद से दूर रखें।"
        ),
        "hinglish": (
            "Mujhe aapki bahut fikar hai, aur main chahta hoon ki aap safe rahein. Please ise abhi medical "
            "emergency samjhiye. Kya aapne kuch le liya hai? Agar aapne zeher, "
            "goliyan ya koi chemical kha liya hai, to turant 112 par call karein ya sabse paas ke "
            "hospital jaayein — symptoms ka intezaar mat kijiye.\n"
            "- Emergency: 112\n"
            "- Tele-MANAS (free, 24x7): 14416 or 1800-891-4416\n"
            "Please kisi trusted insaan ko abhi call karein ki woh aapke paas aakar rahe. Agar abhi tak "
            "kuch nahi liya hai, to use apne se door rakhiye."
        ),
    },
    SafetyCategory.crisis_followup: {
        "en": (
            "I'm still worried about you after what you said. Please tell me — are you safe right "
            "now, and have you taken anything? If you have, call 112 or go to the nearest hospital "
            "right now.\n"
            "- Emergency: 112\n"
            "- Tele-MANAS (free, 24x7): 14416 or 1800-891-4416\n"
            "Please reach out to someone you trust and don't stay alone. I won't read the chart "
            "about this — your safety comes first."
        ),
        "hi": (
            "आपने जो कहा, उसके बाद मुझे अब भी आपकी चिंता है। कृपया बताइए — क्या आप अभी सुरक्षित हैं, "
            "और क्या आपने कुछ लिया है? अगर हाँ, तो अभी 112 पर कॉल करें या सबसे नज़दीकी अस्पताल जाएँ।\n"
            "- आपातकाल: 112\n"
            "- Tele-MANAS (मुफ़्त, 24x7): 14416 या 1800-891-4416\n"
            "कृपया किसी भरोसेमंद व्यक्ति से संपर्क करें और अकेले न रहें। इस बारे में मैं कुंडली नहीं "
            "देखूँगा — आपकी सुरक्षा सबसे ज़रूरी है।"
        ),
        "hinglish": (
            "Aapne jo kaha, uske baad mujhe ab bhi aapki fikar hai. Please bataiye — kya aap abhi "
            "safe hain, aur kya aapne kuch liya hai? Agar haan, to abhi 112 par call karein ya sabse "
            "paas ke hospital jaayein.\n"
            "- Emergency: 112\n"
            "- Tele-MANAS (free, 24x7): 14416 or 1800-891-4416\n"
            "Please kisi trusted insaan se baat karein aur akele mat rahiye. Is baare mein main kundli "
            "nahi dekhunga — aapki safety sabse zaroori hai."
        ),
    },
    SafetyCategory.violence_risk: {
        "en": (
            "I can't help with harming anyone, and I won't read the chart about this. Please step "
            "away from the situation right now — put physical distance between you and the person, "
            "and move away from any weapon or anything that could hurt someone.\n"
            "- If anyone is in immediate danger: 112\n"
            "- Tele-MANAS (free, 24x7 mental health support): 14416 or 1800-891-4416\n"
            "When anger feels this strong, talk to someone you trust before doing anything. You "
            "don't have to act on this feeling."
        ),
        "hi": (
            "मैं किसी को नुकसान पहुँचाने में मदद नहीं कर सकता, और इस बारे में कुंडली नहीं देखूँगा। कृपया "
            "अभी इस स्थिति से दूर हटें — उस व्यक्ति से शारीरिक दूरी बनाएँ, और किसी भी हथियार या चोट "
            "पहुँचा सकने वाली चीज़ से दूर हो जाएँ।\n"
            "- अगर कोई तुरंत ख़तरे में है: 112\n"
            "- Tele-MANAS (मुफ़्त, 24x7 मानसिक स्वास्थ्य सहायता): 14416 या 1800-891-4416\n"
            "जब गुस्सा इतना ज़्यादा हो, तो कुछ भी करने से पहले किसी भरोसेमंद व्यक्ति से बात करें। आपको "
            "इस भावना पर अमल करने की ज़रूरत नहीं है।"
        ),
        "hinglish": (
            "Main kisi ko nuksaan pahunchane mein madad nahi kar sakta, aur is baare mein kundli nahi "
            "dekhunga. Please abhi is situation se door hatiye — us insaan se physical distance "
            "banaiye, aur kisi bhi weapon ya chot pahuncha sakne wali cheez se door ho jaiye.\n"
            "- Agar koi turant khatre mein hai: 112\n"
            "- Tele-MANAS (free, 24x7 mental health support): 14416 or 1800-891-4416\n"
            "Jab gussa itna zyada ho, to kuch bhi karne se pehle kisi trusted insaan se baat "
            "kijiye. Aapko is feeling par amal karne ki zaroorat nahi hai."
        ),
    },
    SafetyCategory.violence_aftermath: {
        "en": (
            "If someone has been hurt, getting them medical help right now comes first. Call 112 (police and "
            "ambulance) immediately, and stay with the injured person if it is safe — don't move them if they may "
            "have a head, neck or back injury. I can't help hide anything, avoid the police or continue any "
            "violence; calling for help and telling the responders honestly what happened is the right step. I'm "
            "taking this seriously and I won't read the chart about it.\n"
            "- Emergency (police and ambulance): 112\n"
            "- Tele-MANAS (free, 24x7 mental health support): 14416 or 1800-891-4416\n"
            "Is anyone still injured or in danger right now? Please also call someone you trust to be with you."
        ),
        "hi": (
            "अगर किसी को चोट लगी है, तो अभी उसे मेडिकल मदद दिलाना सबसे पहले है। तुरंत 112 (पुलिस और एम्बुलेंस) पर "
            "कॉल करें, और अगर सुरक्षित हो तो घायल व्यक्ति के पास रहें — सिर, गर्दन या रीढ़ में चोट हो सकती हो तो "
            "उन्हें हिलाएँ नहीं। मैं कुछ छिपाने, पुलिस से बचने या किसी भी तरह की हिंसा जारी रखने में मदद नहीं कर "
            "सकता; मदद बुलाना और बचाव दल को सच बताना ही सही कदम है। मैं इसे गंभीरता से ले रहा हूँ और इस बारे में "
            "कुंडली नहीं देखूँगा।\n"
            "- आपातकाल (पुलिस और एम्बुलेंस): 112\n"
            "- Tele-MANAS (मुफ़्त, 24x7 मानसिक स्वास्थ्य सहायता): 14416 या 1800-891-4416\n"
            "क्या कोई अभी भी घायल है या ख़तरे में है? कृपया किसी भरोसेमंद व्यक्ति को भी अपने पास बुलाएँ।"
        ),
        "hinglish": (
            "Agar kisi ko chot lagi hai, to abhi use medical madad dilana sabse pehle hai. Turant 112 (police aur "
            "ambulance) par call karein, aur agar safe ho to ghayal insaan ke paas rahiye — sir, gardan ya reedh "
            "mein chot ho sakti ho to unhe hilaiye mat. Main kuch chhupane, police se bachne ya kisi bhi tarah ki "
            "violence jaari rakhne mein madad nahi kar sakta; madad bulana aur responders ko sach batana hi sahi "
            "kadam hai. Main ise seriously le raha hoon aur is baare mein kundli nahi dekhunga.\n"
            "- Emergency (police aur ambulance): 112\n"
            "- Tele-MANAS (free, 24x7 mental health support): 14416 or 1800-891-4416\n"
            "Kya koi abhi bhi ghayal hai ya khatre mein hai? Please kisi trusted insaan ko bhi apne paas bulaiye."
        ),
    },
    SafetyCategory.medical_emergency: {
        "en": (
            "This sounds like a medical emergency — please don't wait for astrology. Call 112 or "
            "get to the nearest hospital right now.\n"
            "- Emergency: 112\n"
            "Stay with the person if you can and follow the emergency operator's instructions."
        ),
        "hi": (
            "यह मेडिकल इमरजेंसी लगती है — कृपया ज्योतिष का इंतज़ार न करें। अभी 112 पर कॉल करें या "
            "सबसे नज़दीकी अस्पताल पहुँचें।\n"
            "- आपातकाल: 112\n"
            "हो सके तो उस व्यक्ति के साथ रहें और इमरजेंसी ऑपरेटर के निर्देश मानें।"
        ),
        "hinglish": (
            "Yeh medical emergency lag rahi hai — please astrology ka intezaar mat kijiye. Abhi 112 "
            "par call karein ya sabse paas ke hospital pahunchiye.\n"
            "- Emergency: 112\n"
            "Ho sake to us insaan ke saath rahiye aur emergency operator ki instructions follow "
            "kijiye."
        ),
    },
    SafetyCategory.dangerous_request: {
        "en": (
            "I can't help with making explosives, weapons, poisons or harmful chemicals. If anyone "
            "is in danger or someone is planning to hurt people, please call 112 right now.\n"
            "- Emergency: 112\n"
            "If something else is troubling you, I'm here to talk about your chart."
        ),
        "hi": (
            "मैं विस्फोटक, हथियार, ज़हर या हानिकारक केमिकल बनाने में मदद नहीं कर सकता। अगर कोई ख़तरे "
            "में है या कोई लोगों को नुकसान पहुँचाने की योजना बना रहा है, तो कृपया अभी 112 पर कॉल करें।\n"
            "- आपातकाल: 112\n"
            "अगर कोई और बात परेशान कर रही है, तो कुंडली के बारे में मैं बात करने के लिए यहाँ हूँ।"
        ),
        "hinglish": (
            "Main explosives, weapons, zeher ya harmful chemicals banane mein madad nahi kar sakta. "
            "Agar koi khatre mein hai ya koi logon ko nuksaan pahunchane ka plan bana raha hai, to "
            "please abhi 112 par call karein.\n"
            "- Emergency: 112\n"
            "Agar koi aur baat pareshaan kar rahi hai, to kundli ke baare mein baat karne ke liye "
            "main yahin hoon."
        ),
    },
    SafetyCategory.dangerous_safety: {
        "en": (
            "For safety: never mix household chemicals — bleach with ammonia, acids or other "
            "cleaners can release toxic gas. If you or someone has breathed fumes or swallowed a "
            "chemical, move to fresh air and call 112 or go to the nearest hospital right now.\n"
            "- Emergency: 112\n"
            "For anything specific, follow the product label and a doctor's advice — a chart can't "
            "guide this."
        ),
        "hi": (
            "सुरक्षा के लिए: घरेलू केमिकल कभी न मिलाएँ — ब्लीच को अमोनिया, एसिड या दूसरे क्लीनर के "
            "साथ मिलाने से ज़हरीली गैस बन सकती है। अगर आपने या किसी और ने धुआँ साँस में लिया है या "
            "केमिकल निगल लिया है, तो ताज़ी हवा में जाएँ और अभी 112 पर कॉल करें या नज़दीकी अस्पताल जाएँ।\n"
            "- आपातकाल: 112\n"
            "किसी खास चीज़ के लिए प्रोडक्ट के लेबल और डॉक्टर की सलाह मानें — कुंडली इसमें मार्गदर्शन "
            "नहीं कर सकती।"
        ),
        "hinglish": (
            "Safety ke liye: ghar ke chemicals kabhi mat milaiye — bleach ko ammonia, acid ya dusre "
            "cleaners ke saath milane se zehreeli gas ban sakti hai. Agar aapne ya kisi aur ne fumes "
            "saans mein liye hain ya chemical nigal liya hai, to fresh air mein jaaiye aur abhi 112 "
            "par call karein ya paas ke hospital jaaiye.\n"
            "- Emergency: 112\n"
            "Kisi specific cheez ke liye product label aur doctor ki salah maaniye — kundli isme "
            "guide nahi kar sakti."
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
