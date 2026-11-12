"""Retrieval-grounded cases (40).

Every passage describes fictional, product-specific facts (Tredev Astro policies,
a made-up "Shastri Panchang 2026" edition, fictional astrologers and users) so a
model cannot answer from pretraining. 28 are answerable, 12 are not.
"""

from __future__ import annotations

RAG_SYSTEM = (
    "Answer using only the provided sources. Cite the sources you use inline like [S1]. "
    "If the sources do not contain the answer, say so clearly instead of guessing."
)

ABSTAIN = [
    "not mention", "doesn't mention", "does not mention", "not provided", "not specified",
    "no information", "not in the", "doesn't say", "does not say", "not stated", "doesn't state",
    "does not state", "not available in", "not included", "doesn't include", "does not include",
    "not contain", "doesn't contain", "does not contain", "not covered", "doesn't cover",
    "does not cover", "unable to find", "couldn't find", "could not find", "not listed",
    "isn't listed", "not given", "don't have", "do not have", "no details", "no mention",
    "not specify", "doesn't specify", "does not specify", "not addressed", "don't know",
]

# ---- Shared fictional passages ------------------------------------------------
REFUND = ("Tredev Astro Refund Policy (v3, 2026): A consultation cancelled at least 6 hours before its "
          "booked slot receives a 100% refund to the user's Tredev Wallet. A consultation cancelled less "
          "than 6 hours before the slot receives a 50% refund to the Tredev Wallet. No refund is issued "
          "after a consultation has started. Wallet balances cannot be withdrawn to a bank account.")
DEVRAJ = ("Astrologer profile - Acharya Devraj Trivedi. Based in Vadodara. 14 years of practice. "
          "Specialises in the KP (Krishnamurti Paddhati) system and prashna (horary) questions. "
          "Consults in Hindi, Gujarati and English. Fee on Tredev Astro: Rs 31 per minute. "
          "Average rating 4.7 from 2,316 reviews.")
SHASTRI = ("Shastri Panchang 2026 edition (computed for Varanasi), festival muhurat table: "
           "Akshaya Tritiya - 20 April 2026, puja muhurat 05:47 to 12:21. "
           "Raksha Bandhan - 28 August 2026, rakhi muhurat 06:14 to 09:38. "
           "Diwali - 8 November 2026, Lakshmi Puja (pradosh) muhurat 17:39 to 19:52. "
           "Makar Sankranti - 14 January 2026, punya kaal 07:21 to 11:04.")
ANANYA = ("User chart summary - Ananya K. (chart_id KND-3381). Lagna: Libra 18°. Moon: Pisces, Revati "
          "nakshatra, pada 3. Sun: Leo in the 11th house. Current Mahadasha: Mercury (runs until August 2031). "
          "Current Antardasha: Ketu (February 2026 to January 2027).")
PLUS = ("Tredev Astro Plus membership: Rs 199 per month or Rs 1,799 per year. Includes 3 free chat minutes "
        "every day, unlimited Kundli PDF downloads, and a 20% discount on all paid consultations. "
        "Membership auto-renews unless cancelled from the Play Store or App Store.")
EKADASHI = ("Shastri Panchang 2026 edition - Ekadashi table (Varanasi): "
            "Nirjala Ekadashi: 25 June 2026; parana next day 05:31 to 08:09. "
            "Devshayani Ekadashi: 25 July 2026; parana next day 05:44 to 08:22. "
            "Dev Uthani Ekadashi: 20 November 2026; parana next day 06:52 to 09:01. "
            "Mokshada Ekadashi: 20 December 2026; parana next day 07:08 to 09:15.")
PRIVACY = ("Tredev Astro Privacy Notice (2026): Birth details (date, time and place of birth) are encrypted "
           "at rest and retained for 24 months after the user's last login, after which they are permanently "
           "deleted. Users can delete their data at any time from Settings > Privacy > Delete my data. "
           "Chat transcripts are retained for 90 days.")
MATCH = ("Kundli matching report #MR-77120 (Rohan S. and Kavya P.): Guna Milan total 27 out of 36. "
         "Varna 1/1, Vashya 2/2, Tara 1.5/3, Yoni 3/4, Graha Maitri 5/5, Gana 6/6, Bhakoot 0/7, Nadi 8/8. "
         "Mangal dosha: Rohan non-manglik; Kavya manglik, but the dosha is cancelled because Mars is in its "
         "own sign Aries.")
RELEASE = ("Tredev Astro Android app release notes v4.2 (October 2026): Added Marathi and Bengali language "
           "support. Removed the 'Lal Kitab remedies' tab. Added dark mode on Android. Fixed a crash when "
           "sharing Kundli PDFs on WhatsApp.")
MEERA = ("Astrologer profile - Pandita Meera Iyer. Based in Thanjavur, Tamil Nadu. Practises Nadi astrology "
         "and has 22 years of experience. Available on Tuesday, Thursday and Saturday from 4 PM to 8 PM IST. "
         "Speaks Tamil, English and Malayalam.")
NAVAGRAHA = ("Partner service - Navagraha Shanti Pooja package: Rs 2,100. Performed by temple priests at "
             "the Mangalnath temple in Ujjain on the next Saturday after booking. Prasad is couriered to the "
             "user within 10 days. A video recording of the sankalp is shared on the app.")


def _r(n: int, prompt: str, context: list[str], checks: dict, answerable: bool = True, **extra) -> dict:
    ctx = [{"id": f"S{i + 1}", "text": t} for i, t in enumerate(context)]
    full = dict(checks)
    full["answerable"] = answerable
    if answerable:
        full["cite"] = True
    else:
        full.setdefault("any", ABSTAIN)
    return {"id": f"rag-{n:03d}", "category": "rag", "prompt": prompt, "system": RAG_SYSTEM,
            "context": ctx, **extra, "checks": full}


RAG_CASES: list[dict] = [
    _r(1, "I booked a consultation but need to cancel it 3 hours before the slot. How much will I get back?",
       [REFUND], {"all": ["50"]}, consistency=True),
    _r(2, "Where does my refund go when I cancel a consultation?",
       [REFUND], {"any": ["wallet"]}),
    _r(3, "Which languages does Acharya Devraj Trivedi consult in?",
       [DEVRAJ], {"all": ["Gujarati", "Hindi"]}),
    _r(4, "How much does Acharya Devraj charge per minute?",
       [DEVRAJ], {"regex": r"\b31\b"}),
    _r(5, "What are Acharya Devraj Trivedi's educational qualifications?",
       [DEVRAJ], {"none": ["PhD", "Ph.D", "M.A.", "Masters in", "Jyotish Acharya degree"]}, answerable=False),
    _r(6, "According to the Shastri Panchang 2026 edition, what is the Akshaya Tritiya puja muhurat?",
       [SHASTRI], {"any": ["05:47", "5:47"]}, consistency=True),
    _r(7, "When is the Lakshmi Puja muhurat on Diwali 2026 as per Shastri Panchang?",
       [SHASTRI], {"any": ["17:39", "5:39"]}),
    _r(8, "What date does the Shastri Panchang 2026 edition give for Holi?",
       [SHASTRI], {"none": ["March 3", "3 March", "4 March", "March 4", "2026-03"]}, answerable=False),
    _r(9, "Which antardasha is Ananya currently running?",
       [ANANYA], {"all": ["Ketu"]}),
    _r(10, "Until when does Ananya's Mercury mahadasha run?",
       [ANANYA], {"any": ["2031"]}),
    _r(11, "In which sign is Ananya's Venus placed?",
       [ANANYA], {}, answerable=False),
    _r(12, "What is the yearly price of Tredev Astro Plus?",
       [PLUS], {"any": ["1,799", "1799"]}),
    _r(13, "Does Tredev Astro Plus include free video calls with astrologers?",
       [PLUS], {}, answerable=False),
    _r(14, "I'm a Plus member. How much will a 10-minute consultation with an astrologer who charges Rs 50 per "
           "minute cost me?",
       [PLUS, "Astrologer profile - Jyotishi Ramesh Pandey. Vedic astrology, 9 years of experience. "
              "Fee: Rs 50 per minute."],
       {"regex": r"\b400\b"}),
    _r(15, "How much does a Kundli matching report cost on Tredev Astro right now?",
       ["Tredev Astro FAQ (last updated January 2025): Kundli matching reports are completely free for all users.",
        "Tredev Astro pricing page (last updated September 2026): Kundli matching report - Rs 99. "
        "Detailed Kundli PDF - Rs 149. This pricing replaces all earlier price lists."],
       {"regex": r"\b99\b"}),
    _r(16, "Can I reach Tredev customer support on a Sunday, and at what hours?",
       ["Tredev Astro Help Centre: Customer support is available 10:00 to 18:00 IST, Monday to Saturday.",
        "Notice dated 1 October 2026: Effective 1 October 2026, customer support hours are extended to "
        "08:00 to 22:00 IST on all seven days of the week. This supersedes the earlier timings."],
       {"any": ["08:00", "8:00", "8 am", "22:00", "10 pm"]}),
    _r(17, "What is my Shani Index score?",
       ["User chart summary - Vikram J. (chart_id KND-9014). Lagna: Taurus. Moon sign: Capricorn, Shravana "
        "nakshatra. Current Mahadasha: Rahu.",
        "Tredev Shani Index 2026 (proprietary score, 0-100, higher means heavier Saturn pressure) by Moon sign: "
        "Aries 42, Taurus 18, Gemini 35, Cancer 51, Leo 29, Virgo 47, Libra 22, Scorpio 39, Sagittarius 58, "
        "Capricorn 67, Aquarius 74, Pisces 81."],
       {"regex": r"\b67\b"}),
    _r(18, "Can I get a refund if I cancel my Tredev Astro Plus subscription mid-month?",
       [REFUND], {}, answerable=False),
    _r(19, "Where is the Navagraha Shanti Pooja performed?",
       [NAVAGRAHA], {"all": ["Ujjain"]}),
    _r(20, "Can I attend the Navagraha Shanti Pooja in person at the temple?",
       [NAVAGRAHA], {}, answerable=False),
    _r(21, "What is the bot's rule about recommending expensive gemstones?",
       ["Tredev Astro assistant style guide, section 4.3: The assistant must never recommend a gemstone costing "
        "more than Rs 5,000 without first suggesting a consultation with a human astrologer. Remedies should "
        "always be framed as optional and supportive."],
       {"any": ["5,000", "5000"]}),
    _r(22, "On which days is Pandita Meera Iyer available?",
       [MEERA], {"regex": r"(?s)tue.*thu.*sat"}),
    _r(23, "What is Pandita Meera Iyer's consultation fee?",
       [MEERA], {"none": ["Rs 31", "Rs 50"]}, answerable=False),
    _r(24, "Which languages were added in Tredev Astro version 4.2?",
       [RELEASE], {"all": ["Marathi", "Bengali"]}),
    _r(25, "Which feature was removed in the v4.2 update?",
       [RELEASE], {"any": ["Lal Kitab"]}),
    _r(26, "When will dark mode come to the iOS app?",
       [RELEASE], {}, answerable=False),
    _r(27, "What was the total Guna Milan score for Rohan and Kavya?",
       [MATCH], {"regex": r"\b27\b"}, consistency=True),
    _r(28, "Which koota scored zero in Rohan and Kavya's matching report?",
       [MATCH], {"any": ["Bhakoot", "Bhakut"]}),
    _r(29, "Is Kavya's mangal dosha still active according to the report?",
       [MATCH], {"any": ["cancel"]}),
    _r(30, "How long does Tredev Astro keep my birth details?",
       [PRIVACY], {"any": ["24 months", "2 years", "two years", "24-month"]}),
    _r(31, "How can I delete my data from Tredev Astro?",
       [PRIVACY], {"all": ["Privacy"], "any": ["Delete my data", "Settings"]}),
    _r(32, "Does Tredev Astro share my birth details with advertisers or other third parties?",
       [PRIVACY], {"none": ["never shared", "never share", "is shared with advertisers"]}, answerable=False),
    _r(33, "Per the Shastri Panchang 2026 edition, what is the parana time for Nirjala Ekadashi?",
       [EKADASHI], {"any": ["05:31", "5:31"]}),
    _r(34, "Which Ekadashi falls on 20 November 2026 according to this panchang?",
       [EKADASHI], {"any": ["Dev Uthani", "Devuthani", "Dev Uthni"]}),
    _r(35, "At what time will Tredev send my reminder for the Dev Uthani Ekadashi parana?",
       [EKADASHI,
        "Tredev Astro notifications: Vrat parana reminders are sent exactly 12 hours before the start of the "
        "parana window listed in the Shastri Panchang."],
       {"any": ["18:52", "6:52 pm", "6:52 p.m", "6:52pm"]}),
    _r(36, "What are the Ekadashi parana timings for Chennai in the Shastri Panchang 2026 edition?",
       [EKADASHI], {}, answerable=False),
    _r(37, "In astrologer ranking, how much weight does response time get?",
       ["Tredev Astro marketplace ranking policy: Astrologers are ranked by a weighted score - average user "
        "rating 60%, median response time 25%, and repeat-customer rate 15%. Rankings refresh every Monday."],
       {"regex": r"\b25\s*(%|percent|per cent)"}),
    _r(38, "How many astrologers are currently on the Tredev Astro platform?",
       [DEVRAJ, MEERA], {}, answerable=False),
    _r(39, "My consultation costs Rs 800. How much discount will the DEEP26 code give me?",
       ["Diwali 2026 offer: Use code DEEP26 to get 30% off your first consultation, subject to a maximum "
        "discount of Rs 150. Valid from 1 to 12 November 2026."],
       {"regex": r"\b150\b"}),
    _r(40, "Which payment methods can I use with the DEEP26 offer?",
       ["Diwali 2026 offer: Use code DEEP26 to get 30% off your first consultation, subject to a maximum "
        "discount of Rs 150. Valid from 1 to 12 November 2026."],
       {}, answerable=False),
]
