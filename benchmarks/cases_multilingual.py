"""Multilingual cases (35): 15 Hinglish, 8 Hindi (Devanagari), 12 other Indian languages.

`checks.language` is the expected reply language. `any` lists common words in the
expected language/script (e.g. "your", "is", or the astrology term itself) so a reply
in the right language almost certainly contains at least one of them.
"""

from __future__ import annotations

SYSTEM = "Reply in the same language and script the user writes in."

HINGLISH_WORDS = ["aap", "hai", "hoga", "hogi", "mein", "karein", "kijiye", "aapki", "aapke", "ke liye"]
HINDI_WORDS = ["है", "आप", "में", "के"]


def _m(n: int, prompt: str, language: str, any_words: list[str], system: str = SYSTEM, **extra) -> dict:
    return {"id": f"ml-{n:03d}", "category": "multilingual", "prompt": prompt, "system": system,
            **extra, "checks": {"language": language, "any": any_words}}


MULTILINGUAL_CASES: list[dict] = [
    # ---- Hinglish (Roman-script Hindi-English mix) --------------------------
    _m(1, "Meri shaadi kab hogi? Meri DOB 14-02-1996 hai, time 08:20 AM, Lucknow.", "hinglish",
       HINGLISH_WORDS, consistency=True),
    _m(2, "Career mein growth kab milegi? Abhi job mein bahut stress hai.", "hinglish", HINGLISH_WORDS),
    _m(3, "Sade sati kya hoti hai? Simple language mein samjhao please.", "hinglish",
       HINGLISH_WORDS + ["shani", "saturn"]),
    _m(4, "Mangal dosh ka kya matlab hai? Kya isse shaadi mein problem hoti hai?", "hinglish", HINGLISH_WORDS),
    _m(5, "Mera rashi Vrishabh hai, aaj ka din kaisa rahega?", "hinglish", HINGLISH_WORDS),
    _m(6, "Bhai, Rahu ki mahadasha chal rahi hai, bahut confusion hai life mein. Kya karu?", "hinglish",
       HINGLISH_WORDS),
    _m(7, "Kundli matching mein 36 mein se 18 gun mile hain. Kya hum shaadi kar sakte hain?", "hinglish",
       HINGLISH_WORDS),
    _m(8, "Naya ghar lene ke liye accha muhurat kaise nikalte hain?", "hinglish", HINGLISH_WORDS),
    _m(9, "Pandit ji ne bola Shani weak hai, toh kya mujhe neelam pehnna chahiye?", "hinglish", HINGLISH_WORDS),
    _m(10, "Nakshatra aur rashi mein kya difference hai?", "hinglish", HINGLISH_WORDS),
    _m(11, "Mujhe naya business start karna hai. Kaunsa din shubh mana jata hai?", "hinglish", HINGLISH_WORDS),
    _m(12, "Mera beta board exam mein fail ho gaya. Kya uski kundli mein koi problem hai?", "hinglish",
       HINGLISH_WORDS),
    _m(13, "Ketu agar 12th house mein ho toh kya effect hota hai?", "hinglish", HINGLISH_WORDS),
    _m(14, "Kya Ekadashi ka vrat rakhna zaroori hai? Iske kya fayde hain?", "hinglish", HINGLISH_WORDS),
    _m(15, "Achha, toh mera lucky colour kya hai? Meri rashi Simha hai.", "hinglish", HINGLISH_WORDS,
       history=[
           {"role": "user", "content": "Hello ji, main Pune se hoon. Thoda astrology ke baare mein jaanna tha."},
           {"role": "assistant", "content": "Namaste! Zaroor, aap jo bhi poochna chahte hain poochiye. "
                                            "Main aapki madad karunga."},
       ]),
    # ---- Hindi (Devanagari) --------------------------------------------------
    _m(16, "मेरी शादी कब होगी? मेरा जन्म 3 मई 1995 को सुबह 7:10 बजे इंदौर में हुआ था।", "hindi_devanagari",
       HINDI_WORDS, consistency=True),
    _m(17, "साढ़े साती क्या होती है? आसान भाषा में बताइए।", "hindi_devanagari", HINDI_WORDS + ["शनि"]),
    _m(18, "राहु काल क्या होता है और इसमें कौन से काम नहीं करने चाहिए?", "hindi_devanagari", HINDI_WORDS),
    _m(19, "मेरी कुंडली में मंगल दोष है, क्या उपाय करूँ?", "hindi_devanagari", HINDI_WORDS + ["मंगल"]),
    _m(20, "नौकरी में तरक्की कब मिलेगी? बहुत मेहनत कर रहा हूँ।", "hindi_devanagari", HINDI_WORDS),
    _m(21, "बृहस्पति का गोचर मेरे लिए कैसा रहेगा? मेरी राशि मकर है।", "hindi_devanagari",
       HINDI_WORDS + ["गुरु", "बृहस्पति"]),
    _m(22, "Please answer in Hindi using Devanagari script: what is a nakshatra and how many are there?",
       "hindi_devanagari", HINDI_WORDS + ["नक्षत्र"],
       system="Follow the user's stated language preference for the reply."),
    _m(23, "एकादशी व्रत का क्या महत्व है?", "hindi_devanagari", HINDI_WORDS + ["एकादशी"]),
    # ---- Tamil ---------------------------------------------------------------
    _m(24, "சனி பெயர்ச்சி என்றால் என்ன? எளிமையாக விளக்குங்கள்.", "tamil", ["சனி", "உங்கள்", "இது"],
       consistency=True),
    _m(25, "என் திருமணம் எப்போது நடக்கும்?", "tamil", ["திருமண", "உங்கள்", "ஜாதக"]),
    _m(26, "நட்சத்திரம் என்றால் என்ன?", "tamil", ["நட்சத்திர", "27", "இது"]),
    # ---- Bengali -------------------------------------------------------------
    _m(27, "শনির সাড়ে সাতি কী? সহজ ভাষায় বুঝিয়ে বলুন।", "bengali", ["শনি", "আপনার", "সাড়ে"]),
    _m(28, "আমার কেরিয়ারে কবে উন্নতি হবে?", "bengali", ["আপনার", "কেরিয়ার", "কর্ম"]),
    # ---- Marathi -------------------------------------------------------------
    _m(29, "साडेसाती म्हणजे काय? सोप्या भाषेत सांगा.", "marathi", ["आहे", "शनी", "तुमच्या"]),
    _m(30, "माझे लग्न कधी होईल?", "marathi", ["आहे", "तुमच्या", "लग्न"]),
    _m(31, "मंगळ दोष म्हणजे काय?", "marathi", ["आहे", "मंगळ"]),
    # ---- Gujarati ------------------------------------------------------------
    _m(32, "મંગળ દોષ શું છે? સરળ ભાષામાં સમજાવો.", "gujarati", ["છે", "મંગળ", "તમારી", "તમારા"]),
    _m(33, "મારી નોકરીમાં પ્રગતિ ક્યારે થશે?", "gujarati", ["છે", "તમારી", "તમારા", "નોકરી"]),
    # ---- Telugu --------------------------------------------------------------
    _m(34, "ఏలినాటి శని అంటే ఏమిటి?", "telugu", ["శని", "మీ", "ఉంది"]),
    _m(35, "నా పెళ్లి ఎప్పుడు అవుతుంది?", "telugu", ["మీ", "పెళ్లి", "వివాహ"]),
]
