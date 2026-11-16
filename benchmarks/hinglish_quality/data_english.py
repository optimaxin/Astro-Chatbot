"""50 English examples: 30 gold, 20 negative."""

from __future__ import annotations

from benchmarks.hinglish_quality.schema import builder

g, n = builder("english", "astrology")
gc, nc = builder("english", "conversational")

EXAMPLES = [
    g("en-01", "placement", "What is my Lagna?", "Your Lagna is Pisces, and its lord Jupiter sits in your 6th house."),
    g("en-02", "career", "How will my career be?",
      "Your 10th house is Sagittarius, with its lord Jupiter in the 6th house. Service-oriented fields such as "
      "finance, law or healthcare suit this placement."),
    g("en-03", "marriage", "When will I get married?",
      "A chart cannot give an exact date. Your Venus Antardasha (June 2024 to April 2027) is traditionally "
      "supportive for marriage, as Venus sits in your 7th house."),
    g("en-04", "dasha", "What is my current Dasha?",
      "You are in Mercury Mahadasha (January 2021 to January 2038), with the Venus Antardasha running until April "
      "2027."),
    g("en-05", "placement", "Where is Saturn?",
      "Saturn is in your 4th house, in Gemini. It is linked with home responsibilities and stability that builds "
      "slowly."),
    g("en-06", "marriage", "What kind of partner will I have?",
      "Your 7th house is Virgo and holds Venus. Your partner may be practical, thoughtful and attentive to detail."),
    g("en-07", "property", "Will I settle abroad?",
      "Your 12th lord Saturn sits in the 4th house. Opportunities abroad are possible, but settling there may take "
      "time and steady effort."),
    g("en-08", "health", "How is my health?",
      "Three planets in your 6th house are traditionally linked with good recovery. If you have any symptoms, "
      "please see a doctor."),
    g("en-09", "placement", "Do I have any yogas?",
      "Sun and Mercury sit together in your 6th house, which is known as Budhaditya yoga. It is linked with a sharp, "
      "analytical mind."),
    g("en-10", "career", "Is this a good time to change jobs?",
      "Your Venus Antardasha runs until April 2027. A well-planned change during this period can work in your "
      "favour."),
    g("en-11", "placement", "What does my Moon sign say?",
      "Your Moon sign is Pisces, with the Moon in your 1st house. You may be sensitive and quick to sense how "
      "others feel."),
    g("en-12", "placement", "What does Mars in my 2nd house mean?",
      "Mars is in its own sign Aries in your 2nd house. It can make your speech direct and your drive to earn "
      "strong."),
    g("en-13", "dasha", "What comes after the Venus Antardasha?",
      "The Sun Antardasha follows, from April 2027 to February 2028."),
    g("en-14", "marriage", "Am I married?",
      "Your chart cannot confirm whether you are married. Venus in your 7th house does show supportive indications "
      "for relationships."),
    g("en-15", "property", "Will I buy a house?",
      "Saturn in your 4th house suggests property comes with patience. Planning your finances carefully will help."),
    gc("en-16", "chat", "Hi", "Hello! You can ask me about your career, marriage, finances or current Dasha."),
    gc("en-17", "chat", "Thanks", "You're welcome! Feel free to ask anything else."),
    gc("en-18", "chat", "Is astrology accurate?",
       "Astrology describes tendencies and possibilities, not guarantees. It works best as guidance alongside your "
       "own judgement."),
    g("en-19", "placement", "When will I be reborn?",
      "Vedic astrology does not give a date for rebirth from a birth chart. Traditionally, the 12th house and Ketu "
      "are linked with liberation."),
    g("en-20", "career", "What does my 11th house say?",
      "Your 11th house is Capricorn, with its lord Saturn in the 4th house. Income tends to grow steadily through "
      "consistent effort."),
    gc("en-21", "chat", "Should I wear a gemstone?",
       "Before wearing a gemstone, it is best to have your full chart reviewed by an experienced astrologer."),
    gc("en-22", "chat", "What is Sade Sati?",
       "Sade Sati is the roughly seven-and-a-half-year period when Saturn moves through the signs before, on and "
       "after your Moon sign."),
    g("en-23", "property", "Tell me about my mother",
      "Saturn in your 4th house is traditionally linked with a disciplined, responsible mother. Patience helps in "
      "that relationship."),
    g("en-24", "placement", "What about children?",
      "Your 5th house is Cancer, with its lord the Moon in your 1st house. This is linked with a strong emotional "
      "bond with children."),
    g("en-25", "placement", "Where is Rahu?",
      "Rahu is in your 3rd house, in Taurus. It is linked with courage and a taste for new experiences."),
    gc("en-26", "chat", "I feel anxious",
       "I'm sorry you're feeling this way. Talking to someone you trust or a counsellor can really help."),
    gc("en-27", "chat", "What is a Nakshatra?",
       "A Nakshatra is one of 27 lunar mansions in the sky. Your birth Nakshatra comes from the Moon's position at "
       "birth."),
    g("en-28", "career", "Will my finances improve?",
      "Your 11th lord Saturn sits in the 4th house, so gains build gradually. The Mercury Mahadasha favours income "
      "from analytical or communication work."),
    gc("en-29", "chat", "Can you explain in simpler words?",
       "Of course. Your chart points to steady growth through hard work rather than sudden success."),
    g("en-30", "placement", "Where is Ketu?",
      "Ketu is in your 9th house, in Scorpio. It is linked with an independent view of religion and philosophy."),
    n("en-31", "placement", "What is my Lagna?", "Your Lagna are Pisces.", "Your Lagna is Pisces.",
      [("number_agreement", "Lagna are", "Lagna is")]),
    n("en-32", "marriage", "When will I marry?", "You will definitely get married in 2025.",
      "Your Venus Antardasha (June 2024 to April 2027) is traditionally supportive for marriage, but a chart cannot "
      "give an exact year.",
      [("fact_change", "2025", "June 2024 to April 2027"), ("certainty", "definitely", "")]),
    n("en-33", "career", "Career?", "Your carrer will grow in the finanse field.",
      "Your career may grow in the finance field.",
      [("spelling", "carrer", "career"), ("spelling", "finanse", "finance"), ("certainty", "will grow", "may grow")]),
    n("en-34", "placement", "Where is Saturn?", "Saturn is in your 7th house, so marriage will be delayed.",
      "Saturn is in your 4th house, which is linked with home responsibilities.",
      [("fact_change", "7th house", "4th house"), ("certainty", "will be delayed", "")]),
    n("en-35", "career", "How is my career?", "Aapka career achha rahega because Jupiter is in 6th house.",
      "Jupiter in your 6th house supports a career in service-oriented fields.",
      [("language_switch", "Aapka career achha rahega", "")]),
    n("en-36", "marriage", "My partner?", "Your partner will be very understanding person.",
      "Your partner may be a very understanding person.",
      [("sentence_structure", "be very understanding person", "be a very understanding person"),
       ("certainty", "will be", "may be")]),
    n("en-37", "placement", "Venus in 7th?", "Venus in 7th house give you a loving partner.",
      "Venus in your 7th house can bring a loving partner.",
      [("number_agreement", "Venus ... give", "Venus ... can bring")]),
    n("en-38", "career", "Growth?", "As per the retrieved knowledge passages, your chart shows growth.",
      "Your chart shows steady growth through effort.",
      [("machinery", "retrieved knowledge passages", "")]),
    n("en-39", "dasha", "Current Mahadasha?", "You are running the Mercury Mahadasha since January 2021 to January 2038.",
      "You are in the Mercury Mahadasha, which runs from January 2021 to January 2038.",
      [("postposition", "since ... to", "from ... to")]),
    n("en-40", "health", "My health?", "Your health will face many diseases.",
      "Your 6th house suggests paying attention to your routine; please see a doctor for any symptoms.",
      [("unsafe_claim", "will face many diseases", ""), ("collocation", "health will face", "")]),
    n("en-41", "career", "10th lord?", "Your 10th lord Mercury sits in 6th house.",
      "Your 10th lord Jupiter sits in your 6th house.",
      [("fact_change", "Mercury", "Jupiter")]),
    n("en-42", "career", "Success?", "The chart is indicating that you might possibly be getting success.",
      "The chart indicates possible success.",
      [("verbose", "is indicating that you might possibly be getting", "indicates possible")]),
    n("en-43", "property", "My mother?", "Your mother will be disciplined and she will be strict and she will be responsible.",
      "Your mother may be disciplined and responsible.",
      [("verbose", "and she will be ... and she will be", ""), ("certainty", "will be", "may be")]),
    n("en-44", "marriage", "Relationship?", "Your relationship and commitment are looking promising with positive energy.",
      "Venus in your 7th house supports commitment in relationships.",
      [("filler", "looking promising with positive energy", "")]),
    n("en-45", "property", "Abroad?", "You has good chances in foreign.", "You have good chances of opportunities abroad.",
      [("number_agreement", "You has", "You have"), ("word_choice", "in foreign", "abroad")]),
    n("en-46", "placement", "Mars?", "Mars in its own sign make your speech direct.",
      "Mars in its own sign makes your speech direct.",
      [("number_agreement", "Mars ... make", "Mars ... makes")]),
    n("en-47", "placement", "Moon sign?", "Your Moon sign is Pisces (मीन).", "Your Moon sign is Pisces.",
      [("script_mixing", "(मीन)", "")], source="benchmark-output"),
    n("en-48", "dasha", "Sun Antardasha?", "Sun Antardasha will start from April 2027 till February 2028.",
      "The Sun Antardasha runs from April 2027 to February 2028.",
      [("postposition", "from ... till", "from ... to")]),
    n("en-49", "property", "Should I buy property now?", "You should invest in property now, the stars are favourable.",
      "A chart should not decide an investment; Saturn in your 4th house suggests patience with property matters.",
      [("unsafe_claim", "You should invest in property now", "")]),
    n("en-50", "career", "11th house?", "Your 11th house is Capricorn, its lord Saturn is in 4th house, gains will come slow.",
      "Your 11th house is Capricorn, with its lord Saturn in the 4th house, so gains may come slowly.",
      [("sentence_structure", "comma splice", "with ... so"), ("word_choice", "come slow", "come slowly")]),
]

assert len(EXAMPLES) == 50, len(EXAMPLES)
