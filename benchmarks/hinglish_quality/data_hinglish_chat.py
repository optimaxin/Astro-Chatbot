"""50 general conversational Hinglish examples: 30 gold, 20 negative.

Greetings, clarifications, limits of astrology, app questions, emotional
moments — the non-chart turns of a real conversation."""

from __future__ import annotations

from benchmarks.hinglish_quality.schema import builder

g, n = builder("hinglish", "conversational")

EXAMPLES = [
    g("hc-01", "chat", "Hi",
      "Namaste! Aap apne career, shaadi, paise ya current Dasha ke baare mein pooch sakte hain."),
    g("hc-02", "chat", "Thank you", "Aapka swagat hai! Kabhi bhi kuch aur poochna ho toh zaroor poochiye."),
    g("hc-03", "chat", "Hinglish mein baat karo", "Bilkul, ab se jawab Hinglish mein milenge."),
    g("hc-04", "chat", "Mujhe samajh nahi aaya", "Koi baat nahi, ise aasaan shabdon mein samjhate hain."),
    g("hc-05", "chat", "Aap kaun ho?",
      "Main Tredev Astro ka assistant hoon. Aapki Kundli ke basis par aapke sawaalon ke jawab deta hoon."),
    g("hc-06", "chat", "Kya yeh sach hota hai?",
      "Astrology sambhavnayein aur tendencies batati hai, pakki guarantee nahi. Apne faisle soch-samajhkar lena "
      "hamesha behtar hai."),
    g("hc-07", "chat", "Birth time pata nahi hai",
      "Koi baat nahi. Birth time ke bina Lagna aur houses sahi se nahi nikalte, lekin Moon sign ke basis par kuch "
      "baatein batayi ja sakti hain."),
    g("hc-08", "chat", "Aaj ka din kaisa rahega?",
      "Daily prediction ke liye current transits chahiye, jo is chart mein calculate nahi kiye gaye hain. Aap apni "
      "current Dasha ke baare mein pooch sakte hain."),
    g("hc-09", "chat", "Mera mood off hai",
      "Yeh sunkar bura laga. Agar aap baat karna chahein toh main yahan hoon, aur kisi apne se baat karna bhi "
      "madadgar ho sakta hai."),
    g("hc-10", "chat", "Kya upay karun?",
      "Roz thodi der meditation aur Thursday ko daan jaise simple upay traditionally achhe mane jaate hain. Koi "
      "bhi upay apni suvidha ke hisaab se karein."),
    g("hc-11", "chat", "Gemstone pehnu?",
      "Gemstone pehnne se pehle kisi experienced astrologer ko poori Kundli dikhana behtar hai. Bina salah ke "
      "gemstone pehnna theek nahi mana jata."),
    g("hc-12", "chat", "Birth time kitna exact chahiye?",
      "Jitna sahi birth time ho, utna achha. Kuch minute ka fark bhi Lagna badal sakta hai."),
    g("hc-13", "chat", "Aapka jawab bahut lamba hai", "Theek hai, ab se jawab chhote aur seedhe rahenge."),
    g("hc-14", "chat", "Kya meri kundli match hogi?",
      "Kundli milan ke liye dono logon ki birth details chahiye. Aap partner ki details share karein toh Guna Milan "
      "dekha ja sakta hai."),
    g("hc-15", "chat", "Ok", "Theek hai! Aur kuch jaanna ho toh bataiye."),
    g("hc-16", "chat", "Kya aap free ho?", "Haan, aap apna sawaal pooch sakte hain."),
    g("hc-17", "chat", "Mujhe exam ki tension hai",
      "Exam ki tension hona normal hai. Thoda break lekar padhai ka plan banaiye, aur zarurat lage toh kisi "
      "teacher se baat kariye."),
    g("hc-18", "chat", "Kal interview hai", "All the best! Achhi tarah taiyari kariye aur aaram se jawab dijiye."),
    g("hc-19", "chat", "Kya meri baatein save rehti hain?",
      "Haan, yeh chat save rehti hai, toh aap baad mein bhi apni baatcheet dekh sakte hain."),
    g("hc-20", "chat", "Panchang kya hota hai?",
      "Panchang ek Hindu calendar hai jo tithi, nakshatra, yoga, karan aur vaar batata hai. Shubh muhurat dekhne ke "
      "liye iska use hota hai."),
    g("hc-21", "chat", "Rashi aur Lagna mein kya fark hai?",
      "Rashi woh sign hai jisme janm ke samay Moon tha. Lagna woh sign hai jo janm ke samay east mein rise ho raha "
      "tha."),
    g("hc-22", "chat", "Nakshatra kya hai?",
      "Nakshatra sky ke 27 hisson mein se ek hai. Aapka janm nakshatra Moon ki position se nikala jata hai."),
    g("hc-23", "chat", "Sade Sati kya hoti hai?",
      "Jab Saturn aapki Moon sign se pehle wali, usi aur agli sign se guzarta hai, us lagbhag saadhe saat saal ke "
      "samay ko Sade Sati kehte hain."),
    g("hc-24", "chat", "Kya aap galat bhi ho sakte ho?",
      "Haan, astrology ek interpretation hai, isliye ise guidance ki tarah lijiye, aakhri faisle ki tarah nahi."),
    g("hc-25", "chat", "Mera naam kya hai?", "Aapne abhi tak apna naam nahi bataya hai. Aap chahein toh bata sakte hain."),
    g("hc-26", "chat", "Kya main phir se kundli bana sakta hoon?",
      "Haan, aap nayi birth details daal kar dobara Kundli bana sakte hain."),
    g("hc-27", "chat", "Kya aap Hindi samajhte ho?",
      "Haan, aap Hindi, English ya Hinglish, kisi bhi bhasha mein pooch sakte hain."),
    g("hc-28", "chat", "Bahut achha laga baat karke", "Mujhe bhi achha laga! Jab bhi zarurat ho, wapas aaiye."),
    g("hc-29", "chat", "Muhurat kaise nikalte hain?",
      "Muhurat ke liye tithi, nakshatra aur din dekhe jaate hain. Kisi khaas kaam ke liye panchang se shubh samay "
      "chuna jata hai."),
    g("hc-30", "chat", "Kya grahan ka asar hota hai?",
      "Traditional astrology mein grahan ke samay naya kaam shuru na karne ki salah di jaati hai. Iska asar har "
      "vyakti ki poori Kundli par nirbhar karta hai."),
    n("hc-31", "chat", "Main kya pooch sakta hoon?", "Aap kya jaanna chahte ho?", "Aap kya jaanna chahte hain?",
      [("register_mix", "Aap ... chahte ho", "Aap ... chahte hain")]),
    n("hc-32", "chat", "Meri madad karoge?", "Main aapko madad karunga.", "Main aapki madad karunga.",
      [("postposition", "aapko madad", "aapki madad")]),
    n("hc-33", "chat", "Mera sawaal theek hai?", "Aapka sawaal bahut achhi hai.", "Aapka sawaal bahut achha hai.",
      [("gender_agreement", "sawaal ... achhi", "sawaal ... achha")]),
    n("hc-34", "chat", "Aapko kya chahiye?", "Kripya apna janm samay pradaan karein.",
      "Kripya apna birth time bata dijiye.",
      [("heavy_register", "pradaan karein", "bata dijiye")]),
    n("hc-35", "chat", "Kuch samjhe?", "Main samjha nahi aapka sawaal.", "Mujhe aapka sawaal samajh nahi aaya.",
      [("word_order", "Main samjha nahi aapka sawaal", "Mujhe aapka sawaal samajh nahi aaya")]),
    n("hc-36", "chat", "Chart kaise banega?", "Tum apna birth time batao, aap ka chart banega.",
      "Aap apna birth time bataiye, phir aapka chart banega.",
      [("register_mix", "Tum ... batao ... aap", "Aap ... bataiye")]),
    n("hc-37", "chat", "Aap kya bata sakte ho?", "Aapko kaunsi jaankari chahiye hai?", "Aapko kya jaankari chahiye?",
      [("verb_form", "chahiye hai", "chahiye")]),
    n("hc-38", "chat", "Thanks", "Dhanyavaad aapke sawaal ke liye, main khush hoon jawab dene mein.",
      "Sawaal poochne ke liye shukriya!",
      [("literal_translation", "main khush hoon jawab dene mein", "shukriya")]),
    n("hc-39", "chat", "Meri help karoge?", "Mujhe aapki madad karne mein khushi hogi hai.",
      "Mujhe aapki madad karke khushi hogi.",
      [("verb_form", "hogi hai", "hogi")]),
    n("hc-40", "chat", "Yeh jaankari kaam aayegi?", "Ye information aapke liye helpful hoga.",
      "Yeh information aapke liye helpful hogi.",
      [("gender_agreement", "information ... hoga", "information ... hogi")]),
    n("hc-41", "chat", "Mujhe tension hai", "Aapko tension lene ka zarurat nahi hai.",
      "Aapko tension lene ki zarurat nahi hai.",
      [("gender_agreement", "ka zarurat", "ki zarurat")]),
    n("hc-42", "chat", "Kya log yeh poochte hain?", "Bahut sare log yeh sawaal poochte hai.",
      "Bahut saare log yeh sawaal poochte hain.",
      [("number_agreement", "poochte hai", "poochte hain")]),
    n("hc-43", "chat", "Thanks", "Aapka swaagat hain.", "Aapka swagat hai.",
      [("number_agreement", "swaagat hain", "swagat hai")]),
    n("hc-44", "chat", "Sab theek hoga?", "Aap bilkul chinta mat karo, sab theek hoga.",
      "Aap chinta mat kijiye; soch-samajhkar kadam uthane se cheezein behtar ho sakti hain.",
      [("register_mix", "Aap ... mat karo", "Aap ... mat kijiye"), ("certainty", "sab theek hoga", "")]),
    n("hc-45", "chat", "History save hoti hai?", "Is chat ka itihaas save rehta hai.",
      "Yeh chat history save rehti hai.",
      [("literal_translation", "chat ka itihaas", "chat history")]),
    n("hc-46", "chat", "Aap kaun ho?", "Main ek AI bhasha model hoon jo NVIDIA ne banaya.",
      "Main Tredev Astro ka assistant hoon.",
      [("machinery", "AI bhasha model ... NVIDIA", "Tredev Astro ka assistant")], source="benchmark-output"),
    n("hc-47", "chat", "Meri details sahi hain?", "Aapka birth details galat lag raha hai.",
      "Aapki birth details galat lag rahi hain.",
      [("gender_agreement", "Aapka birth details ... lag raha", "Aapki birth details ... lag rahi"),
       ("number_agreement", "lag raha hai", "lag rahi hain")]),
    n("hc-48", "chat", "Kitna time lagega?", "Thoda wait kijiye, main check kar raha hoon aapki kundli ko.",
      "Thoda wait kijiye, main aapki Kundli check kar raha hoon.",
      [("word_order", "check kar raha hoon aapki kundli ko", "aapki Kundli check kar raha hoon")]),
    n("hc-49", "chat", "Muhurat kaise nikalta hai?",
      "Muhurat nikalne ke liye panchang ka upyog kiya jaata hai jismein tithi nakshatra ityadi ka vishleshan hota hai.",
      "Muhurat nikalne ke liye panchang dekha jata hai, jisme tithi aur nakshatra dekhe jaate hain.",
      [("heavy_register", "upyog ... ityadi ka vishleshan", "dekha jata hai")]),
    n("hc-50", "chat", "Meri kundli kaisi hai?", "Aap ka kundli bahut interesting hai!",
      "Aapki Kundli kaafi interesting hai!",
      [("gender_agreement", "Aap ka kundli", "Aapki Kundli")]),
]

assert len(EXAMPLES) == 50, len(EXAMPLES)
