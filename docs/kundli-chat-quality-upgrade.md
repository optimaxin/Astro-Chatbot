# Kundli chatbot: quality and intelligence upgrade (2026-11-15)

Prompt version `kundli-chat-2.0`, validator version `kq-validator-1.0`. Every behaviour below is driven by the production screenshots of 2026-11. Each fix has a regression test in `tests/test_screenshot_regressions.py`, which holds 92 tests.

## 1. What the screenshots showed, and why

| # | Symptom (production) | Root cause | Fixed in |
|---|---|---|---|
| 1 | One answer arrived as 4–7 bubbles. A date range was cut in two: "…from January 2021" in one bubble, "to January 2028" in the next. | **This was by design, not a streaming bug.** `reply_format.STRUCTURE_RULES` asked the model for "2 or 3 short messages, under 20 words". `split_into_messages` then hard-cut every bubble at 120 characters, splitting at a comma or space, and dropped anything past 6 bubbles. The widget rendered each item in `messages[]` as its own bubble behind a fake typing delay. | New default `REPLY_SHAPE=single`: one reply is one message (`reply_format.reply_messages`). The widget creates one bubble per reply and fills it from a stream. |
| 2 | Every reply ended with a probe, e.g. "What kind of partner are you looking for?", "Kya tum software development… dekh rahe ho?", "अभी तुम्हारी क्या योजनाएं हैं?" | `CLOSING_QUESTION_RULE` said "Your final message is ALWAYS a short question… no exception". All three style examples ended on a question. The career module and persona said "If you don't know their field, ask that first". The love persona said "Find out what is happening for them before interpreting". | Those rules are removed. The prompt now says "end when the explanation is complete". `response_quality.unneeded_follow_up` deterministically cuts a trailing probe from any route. A genuine clarification, or a reply to an unclassifiable message, is kept. |
| 3 | An English question got a Hinglish answer: "I am married or not?" → "marriage ke chances theek-thenek hain". | `reply_matches("english", …)` only checked the script. Roman-script Hinglish passed as English. The model followed the Hinglish history. | `reply_matches` now rejects Hinglish grammar in English replies (`looks_hinglish`). The English lock says "even if earlier messages were in Hindi or Hinglish". |
| 4 | "I want you to reply in english" worked for one reply. The next Hinglish question got Hinglish again. | Explicit requests were never remembered. The language was re-detected from every message. | Central policy `app/prompts/language_policy.py`. Priority: explicit request, then saved preference (Redis session state `language_preference`), then the message's own language, then history, then English. |
| 5 | A Hinglish reply had Devanagari glosses: "10th house (करियर) Sagittarius (धनु)". Hindi replies had half-Latin terms: "महाdasha", "अन्तर्दasha". The address form mixed तुम and आप. | There was no terminology control, and Hinglish was allowed up to 30% Devanagari. | `app/prompts/terminology.py` gives one vocabulary per language. `normalize_terms` repairs hybrid tokens, removes glosses and transliterates stray Devanagari. Styles and locks require "aap"/"आप". |
| 6 | "Mercury's placement in the 7th…" (Mercury is in the 6th) was not caught. "आपका 7वें घर का स्वामी बुध और शुक्र 7वें घर में हैं" was not checked at all. | Three validator gaps. (a) "X's placement in" was read as general astrology. (b) Python's `\b` never fires after a Devanagari vowel sign, so "…घर में है" produced no claim. (c) The Hindi word "न" counted as a negation even as a word ending ("विराजमान "), so the whole sentence was skipped. | `chart_claims.py`: a Devanagari-aware word end, Hindi lordship ("का स्वामी"), the standalone-only "न", the 7वां/सप्तम/दशम ordinals, new `house_sign` claims, and the lordship forms "X is the Nth lord" / "with X as its lord". |
| 7 | "Venus Antardasha… January 2021 to January 2028" (invented dates). | The prompt gave the Antardasha *lord* with no dates, and nothing validated years. | The prompt now carries the current Antardasha's dates and the full Antardasha sequence. A new `dasha_dates` claim checks every year stated for a named period. |
| 8 | "Chart data mein rebirth… aur retrieved knowledge bhi… kuch nahi batata" | The persona rule said "say plainly that your reference material is thin". There was no deterministic check. | The rule is reworded. `mentions_machinery` triggers the existing single repair, and as a last resort the sentence is dropped. A question note covers "beyond chart" questions such as rebirth or death dates. |
| 9 | "Am I married?" got an implied verdict plus "D9 check karna zaroori hai ki shadi actually hui ya nahi". | Nothing distinguished life-fact questions. | `builder.question_note`: a chart cannot establish real-world status. The reply says so and then gives the traditional indications. |
| 10 | "Meri shaadi ka sahi time kya hai" began "नहीं, मैं हिंदी में जवाब दे सकता हूँ 😊" (the previous turn's answer, echoed). | The prompt never said "reply only to the latest message". | That rule is now in the prompt. `language_meta_sentences` removes capability sentences unless the user just asked to switch. A pure switch request gets a fixed confirmation with no model call. |
| 11 | "jo technology aur backend systems ke liye generally good संकेत देता है" | The model inferred an industry the chart does not state. | The career module forbids naming industries the chart does not point to. |

**Found while testing (not in the screenshots):**

- **With the calculator unreachable, the model invented a whole chart.** It wrote "lagna Vrishabha… Shani Mahadasha 2023 to 2048" even though the prompt said no chart was available, and the validator had no facts to check against. `validate._guard_without_chart` now removes every chart-specific sentence, and replaces the reply with an honest "couldn't load your chart" if too little remains. It also runs after a repair, because a repair once wrote a reading too.
- Nemotron Lightning switches script mid-word ("vyavसाय", "nिखार"), produced an apology loop ("मुझे खेद है।" ×40), and copied a knowledge-base title as a "yoga". All three now have deterministic guards and tests.

## 2. How a turn works now

```text
POST /chat  or  POST /chat/stream (SSE)
  input_guardrail ─► load_context ─► classify_intent ─► retrieve_knowledge ─► build_context
                     │                                   │                     │
                     ├ Redis history + session state     ├ Premium corpus via   ├ master prompt v2 + question note
                     ├ chart (engine) = source of truth  │ kb_rag (chart-fact   ├ chart block with AD dates
                     └ language_policy.resolve_language  │ sub-queries), legacy ├ history with wrong chart
                                                         │ namespace fallback   │ sentences removed
  ─► generate_response (Lightning → Bolt fallback; language switch = fixed reply, no model call)
  ─► validate_reply: terminology clean-up → no-chart guard → chart claims + language + machinery
                     → ONE repair → strip leftovers → citations → deterministic quality fixes
  ─► guardrail_out ─► evaluate_response (sampled/async) ─► save_metrics (persist once; language pref)
```

**Streaming.** `POST /chat/stream` emits these events, all under one `message_id`:

1. `start`
2. `status` (one per pipeline stage)
3. `delta` (zero or more pieces of the reply)
4. `done`, carrying the full `ChatResponse`, or `error` if the turn failed

Text is streamed only after validation. Token-level streaming of the raw draft is deliberately not used, because the validator must see the whole draft before any of it reaches the screen. As a result, streaming gives one message lifecycle and live progress, but no earlier first token. `POST /chat` is unchanged apart from returning `messages` with one element. The widget falls back to it automatically.

## 3. Tradeva Premium knowledge base

These facts come from auditing `D:\Tradeva Premium`:

- **Corpus:** 8,967 chunks in index `tredev-astro-knowledge`, namespace `tradeva-astro-v1`.
- **Embeddings:** BAAI/bge-m3, dense with CLS pooling, normalised, 1024 dimensions.
- **Metadata:** `category_slug`, `planet`, `house`, `house_lord`, `rashi`, `facet` and `language` (en/hi).
- **Existing retrieval:** the Stage 5 service `kb_rag` (`POST /api/rag/retrieve` with `structured_entities`), which Stage 6 builds on.

**Compatibility.** These were checked, not assumed:

- ai-core embeds queries with Pinecone-hosted `llama-text-embed-v2`. That is a different vector space, so it cannot query this namespace.
- NVIDIA's hosted `baai/bge-m3` returned **HTTP 410: end of life on 2026-08-25**.
- Pinecone does not host bge-m3.
- ai-core has no torch runtime.

**Decision: the chatbot calls the existing `kb_rag` service over HTTP.** This required no re-embedding, no re-indexing and no change to any Premium code, so Stage 6 is untouched. The chatbot configuration lives in ai-core only: `PREMIUM_RAG_URL`, `KNOWLEDGE_SOURCE`, and the `PREMIUM_RAG_*` limits.

**Chatbot retrieval plan.** This is `app/rag/premium_knowledge.py`.

1. Up to three parallel sub-queries are built from the user's real placements for the topic. Examples: marriage → `{house_lord: 7, house: 6}` because this chart's 7th lord sits in the 6th; a timing question → the running Mahadasha/Antardasha pair. The user's question runs as one more semantic query.
2. Results then pass four filters:
   - a strict metadata match, which drops the service's relaxed "mirror image" matches (a query for "10th lord in 6th" returned "6th lord in 10th")
   - chart consistency, which drops passages about placements this chart does not have
   - removal of case studies (other people's birth data), and of remedies, gemstones and transits unless the user asked for them
   - de-duplication, with a cap of 4 chunks
3. Each kept chunk is tagged "(Applies to this chart: …)".
4. If the service is down or returns nothing, the legacy namespace is used.

**Retrieval evaluation** (`scripts/eval_premium_retrieval.py`, 20 questions across EN, Hinglish and Hindi on the screenshot chart):

| | Premium (chatbot plan) | Legacy namespace |
|---|---|---|
| precision (topic-relevant chunks) | **98.0%** | 97.3%* |
| chunks about THIS chart's actual placements | **67.3%** | 0% |
| chart grounding of those chunks | **100%** | n/a |
| excluded material (case studies and similar) in output | 0 | 0 |
| empty results (falls back to legacy) | 5% | 0% |
| latency, average / p95 | 2.5 s / 7.1 s (local CPU bge-m3) | 1.0 s / 1.0 s |

\*Legacy relevance can only be judged by coarse domain ("houses", "planets"). Its 90 chunks are generic, so it is not comparable on specificity.

**Corpus data-quality findings** (the corpus was not modified):

- At least one chunk is tagged `house_lord=1, house=12` but titled "I House Lord in XI House". It is filtered by a title/metadata consistency check.
- Strict filters often relax to neighbouring placements. This is handled by the strict re-check described above.

## 4. Benchmark: 100 turns, before vs after

`benchmarks/kundli_quality_cases.py` holds 25 English, 25 Hinglish and 25 Hindi questions, plus 25 multi-turn turns that include the screenshot conversations. `scripts/kundli_quality_bench.py` runs them through the real `run_turn` of each version: "before" from a worktree of commit `69f68a0`, "after" from this checkout. Both use the screenshot chart, live NVIDIA models and the same scorer. That scorer uses the current deterministic validators plus an LLM judge (Bolt).

| Metric (100 turns) | Before | After |
|---|---|---|
| Reply in the required language | 81% (Hindi 52%, multi-turn 80%) | **100%** (every group) |
| Unneeded closing question (deterministic) | 65% | **0%** |
| Unneeded question (LLM judge) | 73% | **2%** |
| Replies with a wrong chart fact (validator) | 4% | **0%** |
| Unsupported chart claims | 2 | 1 |
| Fragmented replies (more than 1 bubble) | 88% (4.5 bubbles avg) | **0%** (1 bubble) |
| Half-translated / mid-word script tokens | 6% | **0%** |
| Judge: relevance (1–5) | 4.29 | **4.85** |
| Judge: completeness (1–5) | 3.54 | **4.17** |
| Judge: naturalness (1–5) | 3.94 | **4.22** |
| Judge: answers the question | 87% | **96%** |
| Failed turns | 0 | 0 |

Notes:

- Language consistency is scored by the stricter checker for both versions. Under the old checker, "before" Hinglish-in-English answers would have passed.
- The "contradiction" rows use the deterministic chart validator, which has known blind spots. Manual review of earlier rounds found three new phrasings, all now covered.

## 5. Tokens and latency

| | Before | After |
|---|---|---|
| Turn latency, average / p50 / p95 | 3.8 s / 3.3 s / 8.2 s | 6.3 s / 5.7 s / 12.3 s |
| Generation, average | 2.1 s | 3.0 s (answers are complete: about 850 vs 400 characters) |
| Retrieval, average | 0.8 s | 1.9 s (Premium bge-m3 on a local CPU) |
| Validation, average | 0.4 s | 0.9 s |
| Tokens per turn (all calls) | 6,140 | 6,312 (+3%) |
| LLM calls per turn | 1.87 | 1.83 |

Latency is the cost of this upgrade. About half comes from the locally hosted Premium retrieval, and that half shrinks with a proper host or `PREMIUM_RAG_MAX_QUERIES=2`. The rest comes from longer, complete answers. Token use is essentially flat, and a pure language switch now costs zero tokens.

All existing token metering is unchanged. Each turn now also records these fields in its metrics and logs:

- `prompt_version`
- `validator_version`
- `language`
- `language_source`
- `response_style`
- `regenerations`
- `validation.quality`

No database schema was changed. `ai_usage` and `ai_evaluations` columns are as before.

## 6. Files

**ai-core, new:**

- `app/prompts/language_policy.py`
- `app/prompts/terminology.py`
- `app/prompts/transliterate.py`
- `app/evaluation/response_quality.py`
- `app/rag/premium_knowledge.py`
- `benchmarks/kundli_quality_cases.py`
- `scripts/kundli_quality_bench.py`
- `scripts/eval_premium_retrieval.py`
- `tests/test_screenshot_regressions.py`

**ai-core, changed:**

- `app/prompts/builder.py` (master prompt v2, question notes)
- `app/prompts/reply_format.py`
- `app/prompts/response_style.py`
- `app/prompts/language.py`
- `app/persona/registry.py`
- `app/kundli/chart_facts.py` (Antardasha dates)
- `app/evaluation/chart_claims.py`
- `app/nodes/validate.py`
- `app/nodes/context.py`
- `app/nodes/generate.py`
- `app/nodes/rag.py`
- `app/state/chatbot_state.py`
- `app/graph/chatbot_graph.py`
- `app/api/server.py` (`/chat/stream`, single message)
- `app/config/settings.py`
- `app/evaluation/evaluator.py`
- tests whose asserted behaviour intentionally changed: `test_chatbot_mvp.py`, `test_grounding.py`, `test_language_guard.py`, `test_pipeline_quality.py`, `test_style_detection.py`

**web/frontend:**

- `src/sections/KundliSection/KundliChatWidget.tsx`: one bubble per reply, streaming, a status line, and a language-aware greeting
- `src/services/aiCoreService.ts` (`askStream`)
- `src/services/sse.ts` and `src/services/sse.test.ts`
- `package.json` (`test:sse`)

## 7. Flags and rollback

| Flag | Default | Off value restores |
|---|---|---|
| `REPLY_SHAPE` | `segments` | `single`: one bubble per reply; `bubbles`: the old 120-character splitter plus closing question |
| `REPLY_QUALITY_FIXES` | `true` | `false`: measure only |
| `KNOWLEDGE_SOURCE` / `PREMIUM_RAG_URL` | `premium` / unset | unset URL or `legacy`: the legacy namespace only |
| `PREMIUM_RAG_TIMEOUT_SECONDS`, `PREMIUM_RAG_MAX_QUERIES`, `PREMIUM_RAG_TOP_K`, `PREMIUM_RAG_MAX_CHUNKS`, `PREMIUM_RAG_MIN_SCORE`, `PREMIUM_RAG_CACHE_SECONDS` | 6 / 3 / 2 / 4 / 0.48 / 3600 | chatbot-only limits |
| `CHART_VALIDATION_MODE` | `auto` | `repair`: always regenerate once |

## 8. Remaining issues and production readiness

- **The Premium corpus needs a hosted `kb_rag`.** The service needs torch, transformers and the 2.2 GB bge-m3 model. On a laptop CPU it adds about 0.9 s per turn on average, with a 7 s p95 under load. Until it is hosted, set no `PREMIUM_RAG_URL`, and the chatbot keeps using the legacy namespace. The alternative is to re-embed the 8,967 chunks with `llama-text-embed-v2` into a new namespace (roughly 2.7M tokens) and query Pinecone directly. That is a decision for the owners, because it duplicates vectors.
- **The model still makes script slips.** Lightning sometimes leaves Latin words with Hindi grammar inside a Hindi reply. One repair pass does not always fix it.
- **The validator does not catch every possible phrasing of a wrong placement.** Each phrasing found so far is covered by a test.
- **Classical "hard outcomes" are still surfaced.** For the 7th lord in the 6th, passages mention separation. The prompt now asks for gentle, conditional framing, and a human review of tone is recommended.
- **Streaming starts after validation**, so there is no time-to-first-token gain.
- **Refresh does not restore the conversation in the widget**, which was already the case before this change. Nothing is duplicated, because the turn is persisted once on the server.

## 9. Round 3: semantic multi-message replies and natural Hinglish

Prompt version is unchanged in purpose; `REPLY_SHAPE` now defaults to `segments`.

### Root causes

**Hinglish spelling ("paartanar", "aapako", "maanasik kanekshan", "bataate hain kee").** These were produced by the round-two fallback that rewrote stray Devanagari words into Roman script letter by letter: "पार्टनर" became "paartanar", "आपको" became "aapako". English loanwords cannot be recovered from their Devanagari spelling by phonetic rules. Misspellings such as "committment" and grammar such as "aapka career mein" come from the model itself.

**One oversized bubble.** This was the round-two design. The bubble splitter was removed because it cut sentences and date ranges, but nothing replaced it with topic-level segmentation.

### What changed

- **`app/prompts/segments.py`.** The model writes 1–4 topic paragraphs:
  - simple question: 1
  - normal question: 2
  - detailed question: 3
  - complex analysis: 3–4

  `segment_reply` splits only at the model's own paragraph breaks. It merges a break when the bubble on either side could not stand alone: a heading on its own line, a list separated from its lead-in, an open date range, a fragment under 60 characters, an unfinished sentence, or a duplicate. More than four paragraphs are merged down to four. If integrity still fails, the reply is sent as one complete message. There is no character-count splitting anywhere.
- **Streaming.** `/chat/stream` events are `start`, `status`, then per segment `segment_start`, `delta` and `segment_end`, then `done`, which carries `messages` (one per bubble) and `segments`. There is one turn id and one id per segment. The turn is persisted once, as the full reply, and history rebuilds the same bubbles. The widget shows one bubble per segment, with feedback on the last bubble.
- **`app/prompts/hinglish.py`.**
  - A Devanagari→Hinglish lexicon, including English loanwords ("पार्टनर" → partner).
  - A vocabulary of always-wrong forms, with each wrong form mapped to its correct spelling.
  - Context-scoped grammar rules: masculine oblique before a postposition, "aap" with gender-neutral verbs, and the "kee" conjunction.
  - `hinglish_issues` for the validator and the benchmark.
- **`app/prompts/transliterate.py`, rewritten.** It looks words up in the lexicon first. For other words it applies Hinglish chat conventions: schwa deletion, short vowels after the first syllable, and "ki" rather than "kee". A free-standing word it does not know is not guessed in the first pass; the model rewrites the reply instead.
- **Validator.**
  - Hinglish allows zero Devanagari.
  - Two or more remaining Hindi-style spellings of English words, or mixed "tum"/"aap", trigger one targeted repair that names the words.
  - Devanagari still present after the repair is romanised.
  - Closing offers ("Koi aur … pooch sakti hain!") are removed like follow-up questions.
  - A reply cut off mid-sentence is trimmed to its last complete sentence.

### Benchmark: 112 turns

The run used the 100-turn core plus 12 depth-labelled questions, including the real-user Hinglish case. "v2" is the round-two code, run from a snapshot; "v3" is this round. Both used the same chart, models, Premium retrieval and scorer.

| Metric (112 turns) | v2 (one bubble) | v3 (this round) |
|---|---|---|
| Simple questions answered in 1 bubble | 100% | **100%** |
| Detailed questions answered in 2–4 bubbles | 0% | **87%** |
| Complex questions answered in 3–4 bubbles | 0% | **100%** |
| Bubbles per reply, average | 1 | 2.59 |
| Bubbles per reply, by question type (simple / normal / detailed / complex) | 1 / 1 / 1 / 1 | 1 / 2.5 / 2.9 / 3.8 |
| Most bubbles in one reply | 1 | 4 |
| Bubbles that stop mid-sentence | 3.6% | **0%** |
| Date ranges split across bubbles | 0% | **0%** |
| Duplicate bubbles | 0% | **0%** |
| Hinglish replies with a spelling, script or grammar issue | 0% | **0%** |
| Misspelled English words (dictionary check) | 0% | **0%** |
| Hindi-style spellings of English words | 0% | **0%** |
| Reply in the required language | 98.2% | 96.4% (Hindi 88%) |
| Replies with a wrong chart fact | 0% | **0%** |
| Unneeded closing question (deterministic check) | 0% | 0% |
| LLM judge: language quality, all languages / Hinglish only (out of 5) | 4.49 / 4.00 | 4.53 / 4.06 |
| LLM judge: relevance / completeness / naturalness (out of 5) | 4.84 / 4.29 / 4.03 | 4.82 / 4.28 / 3.97 |
| LLM judge: unneeded question | 6.2% | 4.5% |
| Latency: average / median / 95th percentile | 4.5 / 4.3 / 7.9 s | 5.6 / 5.0 / 10.5 s |
| Tokens per turn | 6,128 | 6,926 (+13%) |

**Where the extra time goes, per turn on average** (round 2 → round 3):

| Pipeline step | Round 2 | Round 3 | Difference |
|---|---|---|---|
| Generation | 2.50 s | 2.82 s | +0.32 s (multi-paragraph answers are longer) |
| Validation | 0.60 s | 0.95 s | +0.35 s (more targeted repairs) |
| Retrieval | 1.03 s | 1.40 s | +0.37 s (local CPU contention on the bge-m3 service) |

**Language failures in the final run.** There were four:

- **Three Hindi replies.** After the one repair they still mixed in Latin-script Hinglish, for example "5th house ka sign Cancer है". This is a model limitation. A deterministic fix cannot rewrite these sentences safely, so it remains open.
- **One Hinglish reply.** It hit the last-resort fallback, which was English. That fallback is now language-specific, and a test covers it.

**Stage 6 Premium report generation is unaffected.** `D:\Tradeva Premium` has a clean git status, and no file there is newer than the first ai-core change. The Pinecone index, namespace and embeddings are untouched.


## 10. Round 4: production hardening

### What changed

**Language: 100% target**
- **Hindi replies that slip into Roman Hindi.** In a Hindi reply, Roman-script Hindi words are rewritten in Devanagari: "5th house ka sign" becomes "5th house का sign", and planet and sign names become बुध, कन्या and so on. This lives in `devanagarize_hindi` and runs automatically.
- **Final language gate** (`validate._language_gate`). The whole reply must pass, and so must every bubble it will be shown as. A failing reply gets one language-only rewrite; if that also fails, the user gets the fallback message in their own language.
  - Hinglish bubbles have their own rule (`_segment_language_ok`): no Devanagari, and not a full English sentence with zero Hindi words. The stricter whole-reply rule had sent the screenshot case to the fallback.
- **Every canned reply is in the user's language.** This covers the outage message, the last-resort fallback, the "chart unavailable" message, the instruction deflection and the language-switch confirmation. A test checks all five in all three languages.
- **Fallback models keep the language.** A test checks that the backup model receives the same message list, language lock included. A live check confirmed it, below.

**Hinglish naturalness**
- The prompt now tells the model to paraphrase the knowledge base in its own words, prefer everyday words over Sanskrit-heavy ones, avoid "situate hai", and use gender-neutral "aap" forms.
- English headings copied from the knowledge base ("Charitable Inclination, Beautiful Clothing") are lower-cased.
- A self-description given instead of an answer ("मैं … एक भाषा मॉडल हूं") now triggers a repair.
- An attribution phrase left dangling after a citation tag is removed ("…को [K1] के अनुसार…" became "…को के अनुसार…") is now deleted along with the tag.
- `benchmarks/hinglish_reference.py` holds 20 awkward→natural Hinglish pairs. Their status is **draft: not yet reviewed by a native speaker.** They serve two purposes: regression tests for the mechanical cases, and calibration of the LLM judge.

**Segmentation**
- The rules spell out that "How will my career / finances / family / marriage be?" is a detailed question with two or three topics.
- Repairs are told to keep the paragraph structure. A repair had collapsed a four-topic answer into one bubble.

**Performance**
- `CHART_VALIDATION_MODE` now defaults to `auto`. Wrong sentences are removed without a model call when a real answer remains; otherwise the reply is regenerated once.
- Identical Premium sub-queries are cached in memory (`PREMIUM_RAG_CACHE_SECONDS`, 1 hour by default). The corpus is static.
- A validator false positive was removed: "10th house (Sagittarius rashi)" was being read as a Moon-sign claim.

### Investigation findings

- **Repairs.** A 25-turn probe found that most repairs fix real model errors in chart facts. They protect accuracy, so they were made cheaper, not removed.
- **Prompt size.** About 17,000 characters, or 4,600 input tokens, per main call.
- **Retrieval.** The Premium service embeds queries one at a time on the CPU, at about 0.5–0.9 s per sub-query. This is the largest cost we control. Hosting the service on a GPU, or setting `PREMIUM_RAG_MAX_QUERIES=2`, would cut it further.
- **LLM judge calibration.** On the 20 reference pairs, the Bolt judge rated the natural version higher in 8 pairs and called 11 a tie (averages: natural 3.90, awkward 3.35). It cannot reliably measure Hinglish naturalness, so the Hinglish judge score should not be treated as a quality gate until native speakers review the reference set.

### Final 112-turn benchmark

"v3" is round three; "final" is this round's final code. Both used the same chart, models, Premium retrieval and scorer.

| Metric | v3 | final |
|---|---|---|
| Reply in the required language | 96.4% | **100%** (every group) |
| Simple questions answered in 1 bubble / detailed in 2–4 / complex in 3–4 | 100% / 87% / 100% | **100% / 89% / 100%** |
| Bubbles that stop mid-sentence / date ranges split / duplicate bubbles | 0 / 0 / 0 | 0 / 0 / 0 |
| Replies with a wrong chart fact | 0% | 0% |
| Hinglish replies with a vocabulary, script or grammar issue | 0% | 2.8% (1 suspect transliteration) |
| LLM judge: Hinglish language quality (uncalibrated, see above) | 4.06 | 4.14 |
| LLM judge: unneeded question | 4.5% | **0.9%** |
| LLM judge: relevance / completeness / naturalness | 4.82 / 4.28 / 3.97 | 4.79 / 4.25 / 4.03 |
| Tokens per turn | 6,926 | 6,626 |
| Replies replaced by the language fallback | — | 1 of 112 |
| Failed turns | 0 | 2 (model unreachable) |

**Latency: target not met, and today's runs cannot be compared with earlier rounds.** Both final-code runs coincided with degraded external services:

| Per turn, average | v3 run (earlier, healthy services) | final run | clean re-run |
|---|---|---|---|
| Turn latency, average / p50 / p95 | 5.6 / 5.0 / 10.5 s | 12.9 / 8.3 / 35.4 s | 12.1 / 9.1 / 32.6 s |
| Redis load / save (Upstash), unchanged code | 73 / 90 ms | 696 / 1,693 ms | 671 / 1,461 ms |
| Classifier LLM call, unchanged code | 450 ms | 1,955 ms | 1,585 ms |
| Main model call | 3.8 s | 8.4 s | 6.5 s |
| Turns answered by the backup model because the primary was slow | 0.9% | 8.0% | 45.5% |
| Validation (this round's work) | 0.95 s | 0.92 s | **0.76 s** |
| Tokens per turn (this round's work) | 6,926 | 6,626 | 6,624 |

**Spot checks at the same time.** In isolation the services were fast: Redis about 80 ms, and NVIDIA about 0.6 s for a 4,000-token prompt. Both took 1.4–6 s on the first call. A sequential 8-turn A/B test averaged 6.9 s per turn with the Premium corpus and 5.8 s with the legacy knowledge base, so the Premium service costs about 0.4–1.1 s per turn.

What this round changed and can show:
- Validation is cheaper (0.95 → 0.5–0.76 s).
- Validator rewrites dropped (39 → 23 in the quiet run).
- Tokens are down 4–14%.

End-to-end latency needs re-measuring when Redis and NVIDIA are healthy, or from production telemetry in the admin Token Meter, which already records latency per model.

**One more fix from this investigation.** Under degraded NVIDIA, the language gate's own rewrite call timed out. Seven correct-language Hindi answers were replaced by "please ask again" because of one weak bubble. The gate now delivers a candidate that passes the whole-reply language check instead of the fallback. A test covers it, and 12 of 12 Hindi questions passed sequentially.

### Production verification

| Check | Result | Evidence |
|---|---|---|
| ai-core on :8000 restarted on the final code with the Premium service | Yes. `/chat/stream` emits segment events, and replies cite Premium categories (`house_lords_in_houses`). | `logs/ai-core-8000.log` |
| Real browser, through the actual TredevaAstro UI | Passed 6/6 turns on the final code: real birth form and Kundli generation, then English, Hindi and Hinglish questions, a simple question, an explicit switch, and a persistence check. Bubbles per reply: 4, 2, 3, 1, 1, 4. Every bubble ended on a complete sentence. | `web/frontend/e2e/kundliChat.e2e.mjs`, `logs/browser-e2e/*.png`, `results.json` |
| Streaming | Every turn showed the full event sequence: start, status, segment_start, delta, segment_end, done. | `logs/browser-e2e/results.json` |
| Conversation persistence | After "I want you to reply in english", the next question ("Mera career kaisa rahega?") was answered in English. The preference was read back from the server-side session. | browser results, turns 5–6 |
| NVIDIA fallback route | The primary model was deliberately broken on a test instance. Bolt answered (`fallback_used=true`). The English preference survived into a Hinglish and a Hindi question, answers still came in 3 segments, and the chart facts were consistent. | `logs/fallback-live-check.txt` |
| Stage 6 Premium report generation | Not affected. `D:\Tradeva Premium` git status is clean. Its offline suites (kb_report, kb_interpret, kb_kundli, premium_api; live tests excluded) passed: 577 passed. | — |
| ai-core offline tests | 792 passed | `pytest --ignore=tests/test_integration.py` |

## 11. Round 5: shorter replies, Hinglish grammar layer, Hinglish benchmark

Nothing in this round touches Pinecone, the knowledge base, embeddings, retrieval, Kundli Engine calculations or Premium report generation. Segmentation, streaming, the provider chain and token tracking are unchanged.

### What changed

| Area | Change | Files |
|---|---|---|
| Length | `classify_depth` picks one budget per turn (simple 15–30 words in 1 message, normal 30–60 in 2, detailed 60–100 in 3–4, complex 80–120 in 3–4). The budget goes into the system prompt and, once the conversation has history, is repeated as a short note on the current question. Long earlier replies were pulling later replies to the same length. | `app/prompts/answer_depth.py`, `app/prompts/builder.py`, `app/nodes/context.py` |
| Redundancy | Restated sentences that add no new chart fact are removed. A Dasha date range is given once per reply, and not repeated from the previous reply unless timing is asked. Simple answers are joined into one message. | `app/evaluation/conciseness.py` |
| Follow-up | At most one follow-up of 15 words or fewer, and only a specific offer. | `app/prompts/followup.py` |
| Hinglish grammar | One pass after generation: detect, apply deterministic fixes (possessive gender, "se juda", sentence capitals, known misspellings), send only the flagged sentences in one LLM correction call, and verify each correction (no new planets, houses, signs, dates or numbers; length kept). If the correction fails, severe sentences are dropped. If nothing is left, a language-matched safe reply is sent. There are no retry loops. | `app/evaluation/hinglish_grammar.py`, `app/nodes/validate.py` |
| Language switching | After a Devanagari Hindi turn, the model answered Roman-Hinglish questions in Devanagari, and both repair calls copied that draft, so the user got the fallback. Now the current question carries a one-line language note when the previous reply was in another language, repair calls drop the older history, and a wrong-script draft is regenerated rather than rewritten. | `app/prompts/language.py`, `app/nodes/repair_prompts.py` |
| Copied English | An English knowledge-base passage pasted into a Hinglish reply, including "Label: a, b, c" lists, is now a severe issue. | `app/evaluation/hinglish_grammar.py` |
| Benchmark | Hinglish Quality Benchmark (HQB): 250 examples, rules, fact checks, a cached LLM judge, a gold set, a dashboard and regression detection. It is independent of the chatbot. | `benchmarks/hinglish_quality/`, `scripts/hinglish_quality_bench.py`, `docs/hinglish-quality-benchmark.md` |

Flags: `HINGLISH_GRAMMAR=off|detect|correct`. `correct` is the default, and `off` restores the earlier behaviour.

### Conciseness: 112 live turns, r5final vs c4 (final code)

| Metric | Before | After |
|---|---|---|
| Words per reply | 81.8 | 64.2 |
| Normal / detailed / complex, words | 77.3 / 102.9 / 90.2 | 60.8 / 76.7 / 100.6 |
| Replies within the length budget | 50.9% | 69.6% |
| Replies over the length budget | 33.0% | 11.6% |
| Messages of 25 words or fewer | 47.6% | 66.0% |
| Messages over 36 words | 20.7% | 4.3% |
| Sentences per message | 1.49 | 1.28 |
| Simple answers in 1 message | 100% | 100% |
| Detailed answers in 3–4 messages | 82.6% | 87.0% |
| Replies with a redundant pair | 2.7% | 0.9% |
| Replies repeating Dasha dates | 0.9% | 0% |
| Follow-up asked | 7.1% | 0.9% |
| Chart contradictions, unsupported claims | 0, 0 | 0, 0 |
| Language consistency | 100% | 100% |
| Latency, average / p95 | 4.7 / 9.6 s | 2.6 / 5.6 s |
| LLM calls per turn | 1.76 | 2.13 |

The NVIDIA free tier rate-limited the judge, so only 2 of the 112 c4 turns were judged and there is no judge score for conciseness. The latency gain mostly comes from faster external services during this run, not from this round. The grammar correction call adds about 0.25 s when it fires.

### Hinglish Quality Benchmark: 250 live chatbot turns

`cb-before` uses `HINGLISH_GRAMMAR=off`. `cb-final` uses the final code. All runs were judged.

| Slice | Overall | Grammar | Pass rate | Fact gate | Errors |
|---|---|---|---|---|---|
| All | 4.820 → 4.846 | 4.766 → 4.838 | 89.2% → 96.0% | 1.6% → 1.2% | 153 → 148 |
| Hinglish (150) | 4.783 → 4.828 | 4.64 → 4.77 | 86.0% → 97.3% | 1.3% → 0.7% | 119 → 111 |
| Hindi (50) | 4.787 → 4.799 | 4.91 → 4.89 | 88% → 88% | 4% → 4% | 28 → 28 |
| English (50) | 4.967 → 4.948 | 5.0 → 4.99 | 100% → 100% | 0% → 0% | 6 → 9 |

Regression check: none beyond the thresholds (overall 0.1, any dimension 0.2, pass rate 5 points, fact gate 2 points). Hinglish naturalness moved −0.03, which is within the noise. Dashboard: `docs/hqb-report-final.html`, with before, after-grammar-layer and final, and a language filter.

### Correction mode: the benchmark's negative examples through the corrector

| Run | Negatives changed | Exactly fixed | Passing after | Gold left unchanged |
|---|---|---|---|---|
| Deterministic fixes only (v1) | — | 5.5% | 14.5% | 99.3% |
| Deterministic fixes only (v2) | 33.6% | 9.1% | 28.2% | 100% |
| Full layer with LLM correction | 48.2% | 10.9% | 40.0% | 100% |

The corrector never changed a correct gold sentence. Most negatives that are still not fixed involve word choice or naturalness, which the corrector deliberately leaves alone unless a rule flags them.

### Examples

| Before | After |
|---|---|
| "property abroad se joda hota hai", "samay-samay par delays ho sakte hain" | "se juda hota hai", "deri ho sakti hai" (deterministic, agreement adjusted) |
| "Saturn ka effect health issues la sakta hai" | Flagged and sent to the one correction call, e.g. "Saturn ke kaaran health-related dikkat aa sakti hai". The verifier rejects a version that changes the planet, adds certainty ("pakka") or changes script. |
| "domestic harmony ke nuksan de sakta hai", "badhyaari hogi" | Flagged and sent to the model, since the right wording depends on meaning. If no verified correction comes back, the sentence is dropped (the safe reply is used only if nothing is left). |
| "aapka Rahu Mahadasha chal raha hai" | "aapki Rahu Mahadasha chal rahi hai" |
| "aur Shani 4th house mein hai." (sentence start) | "Aur Shani 4th house mein hai." |
| Hinglish question after a Hindi turn: the safe fallback ("please ask again") | An answer in Roman Hinglish, in 4 messages |
| A pasted "Body parts ruled: bones, teeth, …" in a Hinglish reply | Corrected into Hinglish, or dropped |
| A detailed Hinglish question late in a long session: 6 messages, 173 words | 4 messages, 92 words |

### Browser verification (real UI, final code)

7 of 7 turns passed: real birth form, Kundli generation, then English Dasha, Hindi marriage, two Hinglish questions, a simple Lagna question, an explicit language switch and an English career question. Messages per turn were 4, 4, 4, 4, 1, 1, 4. Detailed and complex answers were 75–98 words. After a reload, all 23 messages came back with identical text. Screenshots: `logs/browser-e2e/00-greeting.png` … `07-english.png`, and `99-after-reload.png`.

### Limitations

- Lightning still produces some garbled Hinglish words ("mulyaman", "antarkarun", "samansooth", "aspekt", "assar"). The rules do not catch them; only the judge does.
- `chart_claims` has no conjunction check. One hallucinated "conjunct your Moon and Jupiter" was seen.
- The Hindi follow-up sometimes repeats something already answered. Hindi machinery leaks are rare but not zero.
- The judge is lenient. The rules and the dataset have the same author, and the dataset has not yet been reviewed by a native speaker (`export-review` / `import-review` are ready for that).
- NVIDIA free-tier rate limits affect the runs: 9.6% of HQB turns went to the fallback model, and the c4 judge scored only 2 turns.
- The Premium knowledge service occasionally fell back to the legacy namespace during runs. That is where the copied-English passages came from.
- Tests: 1,172 offline tests pass (`pytest -o addopts="" --deselect tests/test_integration.py`).
