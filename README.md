# Tredev Astro — AI Core

A conversational Vedic astrology assistant, built as a **standalone Python service** with no dependency on the existing frontend. Every turn reports its intent, tokens, latency, cost, quality scores and reward.

It is reachable two ways — `POST /chat` over HTTP, and a terminal REPL — and both call the same `run_turn`, so they cannot drift apart in behaviour.

Nothing in the existing Node backend or frontend was modified to build this.

```
Terminal  /  POST /chat
   ↓
AI Core  →  LangGraph
                ├── input_guardrail (crisis, regulated advice, PII, scope)
                │       └── gated? → canned reply, straight to save_metrics, no LLM call
                ├── load_context ──┬── conversation history  → Upstash Redis
                │                  ├── user kundli           → Tredev backend / sample
                │                  └── session state         → Upstash Redis
                ├── classify_intent                          → OpenRouter (classifier model)
                ├── retrieve_knowledge                       → Pinecone (astrology knowledge)
                ├── build_context   (dynamic system prompt)
                ├── generate_response                        → OpenRouter (main model, + fallback)
                ├── guardrail_out   (AI leak, false certainty, disease prediction)
                ├── evaluate_response                        → OpenRouter (separate judge model)
                └── save_metrics    (tokens, latency, cost, reward)
   ↓
Terminal  /  JSON response
```

## The data design that matters

Three kinds of data, three stores. Conflating them is the most common way a system like this goes wrong.

| Data | Nature | Where it lives |
|---|---|---|
| **User kundli** | Structured, user-specific, exactly retrievable | Tredev backend (`POST /api/calculators/kundli-full`), behind `app/kundli/kundli_service.py` |
| **Astrology knowledge** | General, unstructured, semantically searchable | Pinecone |
| **Conversation / session state** | Per-session, ephemeral, ordered | Upstash Redis |

A kundli is **never** put into Pinecone. "Which house is your Saturn in" is an exact lookup, not a similarity search — vectorising it would make a precise answer fuzzy for no benefit, and would store every user's personal chart in a shared index.

`data/sample_kundli.json` was produced by **running the real backend calculators** (`backend/app/services/astrology/*`), not hand-written, so terminal testing never exercises invented astrology.

## Setup

```bash
cd ai-core
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env          # then fill in the credentials below
```

### Required environment variables

| Variable | Needed for | Get it from |
|---|---|---|
| `OPENROUTER_API_KEY` | all LLM calls | https://openrouter.ai/keys |
| `PINECONE_API_KEY` | RAG | https://app.pinecone.io |
| `UPSTASH_REDIS_REST_URL` | memory | Upstash console → your database → REST API |
| `UPSTASH_REDIS_REST_TOKEN` | memory | same page |

Everything else in `.env.example` has a working default. `.env` is gitignored and must never be committed.

## Running it

```bash
# 1. Verify all three services independently
python -m scripts.test_connections

# 2. Load the astrology knowledge base into Pinecone (once)
python -m scripts.ingest_knowledge            # --dry-run to chunk without uploading
                                              # --recreate to wipe the namespace first

# 3a. Chat in the terminal
python -m app.main                            # --session my-session --user my-user
                                              # --persona career_finance

# 3b. Or serve the HTTP API
uvicorn app.api.server:app --reload --port 8000
python -m app.api.server                      # same thing, reads API_HOST/API_PORT
                                              # docs at http://localhost:8000/docs

# 4. Tests
python -m pytest                              # credential-gated tests self-skip

# 5. Per-user token and cost report
python -m scripts.usage_report                # grouped by user
python -m scripts.usage_report --answers-only # excludes classifier/judge overhead
python -m scripts.usage_report --user alice   # one user in detail

# 6. Compare models on the same questions and the same chart
python -m scripts.benchmark_models --models "model-a,model-b"
```

In the chat: `exit`, `/clear`, `/metrics`, `/usage`, `/personas`, `/config`, `/help`.

## The HTTP API

`POST /chat` is the production surface. The client sends **who is asking and what they asked — never the chart**. The chart is resolved server-side from `user_id` and cached in Redis under `chart:{user_id}`, so a browser never handles chart data it did not need and cannot substitute a chart of its own.

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "test-user-001",
    "session_id": "web-session-42",
    "persona_id": "career_finance",
    "question": "Is this a good period for a job change?"
  }'
```

```jsonc
{
  "answer": "To give you the best guidance, could you tell me a little about what you do now? …",
  "persona": "career_finance",
  "session_id": "web-session-42",
  "request_id": "0f3a…",
  "intent": "career",
  "sources": [
    { "id": "career.txt::job_change_switching_and_starting_a_business",
      "document": "career.txt", "domain": "career",
      "topic": "job_change_switching_and_starting_a_business", "score": 0.1411 }
  ],
  "chart_facts_used": ["lagna=Virgo", "moonSign=Pisces", "Sun=Cancer/H11", "…"],
  "usage": {
    "input_tokens": 2022, "output_tokens": 41, "total_tokens": 2063,
    "tokens_available": true, "model": "google/gemini-2.5-flash",
    "estimated_cost_usd": 0.0007091, "provider_cost_usd": 0.0007091,
    "turn_total_tokens": 3718, "turn_cost_usd": 0.00097915, "llm_calls": 3
  },
  "latency_ms": 14808,
  "degraded": false,
  "notices": []
}
```

| Endpoint | Purpose |
|---|---|
| `POST /chat` | one conversational turn |
| `GET /personas` | the personas a client may offer — never returns a system prompt |
| `GET /chart/{user_id}` | compact chart facts, for rendering placements alongside the chat |
| `GET /health` | per-dependency status; `configured` and `reachable` reported separately |
| `GET /docs` | generated OpenAPI docs |

`session_id` is optional and defaults to `user-{user_id}`. `persona_id` is optional; an unknown one falls back to `vedic_general` rather than erroring, because a bad persona should blur an answer's focus, not fail a request.

Two things the API will not do: it never returns a stack trace, an internal path or a credential (dependency failures arrive as plain-language `notices`), and it never accepts a chart from the client.

## Personas

Configuration data, not branching logic — `app/persona/registry.py`.

| id | Specialization | Knowledge domains searched |
|---|---|---|
| `vedic_general` | whole-chart reading | follows the classified intent |
| `love_relationships` | romance and emotional life | relationships, houses, planets |
| `career_finance` | work, income, money | career, finance, houses, planets, dashas |
| `marriage_compatibility` | marriage timing and matching | relationships, houses, planets, nakshatras |

A persona composes with — rather than replaces — the intent module, so the system prompt is:

```
identity + safety  →  persona  →  intent module  →  chart facts
                   →  retrieved knowledge  →  anti-hallucination rules
```

The rules go **last** on purpose: a model weights the end of a long system message heavily, and those are the part that has to survive a long context. A persona's `domains` are *unioned* with the intent's, never substituted, so a Career persona asked a health question can still reach health knowledge instead of answering from the wrong shelf.

Moving personas into Postgres later means reimplementing `get_persona` / `list_personas` against a table. Nothing else constructs a `Persona`.

## Ingesting a PDF

```bash
# 1. Drop the PDF into data/knowledge/
# 2. Declare its retrieval domain (a domain cannot be guessed from a filename,
#    and a chunk filed under the wrong one is unreachable by filtered search):
cat > data/knowledge/manifest.json <<'JSON'
{ "brihat_parashara.pdf": { "domain": "houses", "book": "BPHS" } }
JSON
# ...or pass --pdf-domain houses for a one-off.

python -m scripts.ingest_knowledge --dry-run     # inspect the chunks first
python -m scripts.ingest_knowledge
```

PDFs are split **chapter → heading → topic → overlapping sentence window**, in that order; the fixed window (`CHUNK_SIZE` / `CHUNK_OVERLAP`) is the last resort, because a chunk that starts mid-sentence retrieves badly however good the embedding model is. Each chunk is tagged with `book`, `chapter`, `system`, `topic`, and — when the text supports it on a clear majority — `planet` and `house`. A tag that cannot be determined is **omitted rather than blanked**, so it stays usable as a Pinecone filter.

Ingestion is offline only. A `/chat` request performs retrieval, never ingestion.

## Token and cost metering

Three levels, because they answer different questions.

**Per response** — shown in the terminal after every turn:

```
Input Tokens    : 1,610
Output Tokens   : 38
Estimated Cost  : $0.000578
Provider Cost   : $0.000578      <- OpenRouter's own figure, for cross-checking
Turn Cost (all) : $0.000837  across 3 calls
```

**Per user, durable** — every LLM call is appended to `logs/usage.jsonl` and,
when Upstash is configured, also increments `usage:user:{user_id}` counters in
Redis. The ledger is the complete audit record; the Redis counters are the
O(1) lookup a quota check or billing query needs, shared across processes.

```
USER     CALLS       INPUT      OUTPUT       TOTAL          COST       $/CALL
alice        6       5,136         253       5,389      $0.001480    $0.000247
bob          3       3,168         237       3,405      $0.001145    $0.000382
```

**Per role** — every call is attributed to the job that made it, so the cost of
the *answer* is separable from the cost of classifying and judging it:

| Role | Model | What it is |
|---|---|---|
| `job-main-turn` | `google/gemini-2.5-flash` | the user-visible answer |
| `job-classify-intent` | `google/gemini-2.5-flash-lite` | intent classification |
| `job-evaluate-response` | `openai/gpt-4o-mini` | the quality judge |

`--answers-only` restricts a report to `job-main-turn`. Without it you see the
true all-in cost per user, which is roughly 40% higher than the answer alone.

Measured on a real turn, the config-table estimate matched OpenRouter's own
reported cost to the cent ($0.000578 vs $0.000578), so the price table in
`.env` is verified rather than assumed.

## What each phase requirement maps to

| Concern | Where |
|---|---|
| Config (no hard-coded secrets) | `app/config/settings.py` — the only module that reads `os.environ` |
| Model routing + fallback | `app/llm/llm_factory.py` — nodes ask for a **role**, never a model id |
| Token + cost accounting | `app/llm/usage_tracker.py` |
| Conversation memory | `app/memory/redis_memory.py` |
| Kundli access | `app/kundli/kundli_service.py` |
| RAG | `app/rag/` + `scripts/ingest_knowledge.py` |
| Intent classification | `app/nodes/classifier.py` |
| Dynamic prompt assembly | `app/prompts/builder.py` |
| Quality evaluation | `app/evaluation/evaluator.py` |
| Reward | `app/evaluation/reward.py` |
| Orchestration | `app/graph/chatbot_graph.py` |
| Terminal UI | `app/main.py` |
| Structured logging | `app/logging_setup.py` → `logs/ai-core.jsonl` |
| HTTP API | `app/api/server.py` + `app/api/schemas.py` |
| Personas | `app/persona/registry.py` |
| PDF extraction + chunking | `app/rag/chunking.py` |
| Chart facts (`to_facts` / `from_facts`) | `app/kundli/kundli_service.py` |

### Three rules the code enforces rather than asserts

1. **Token counts are never invented.** They come from the provider's `usage` block. When the provider omits it, the count is `None` and prints as `unavailable` — never a plausible-looking guess. (`usage_tracker.extract_usage`)
2. **Kundli facts are never fabricated.** `evaluator.check_kundli_contradictions` parses chart claims out of the response and checks them against the real chart, deterministically, without asking an LLM. A contradiction caps the reward.
3. **A 200 OK is not a good answer.** Quality is measured by deterministic checks plus a judge running on a *different* model from the generator, so the generator never grades itself.

## Configuring models

Every LLM call routes through a role. Change a model by changing `.env` — no code touches a model id.

```
OPENROUTER_MODEL=…              # job-main-turn      (the answer)
OPENROUTER_FALLBACK_MODEL=…     # used when the main model errors or times out
CLASSIFIER_MODEL=…              # job-classify-intent
EVALUATOR_MODEL=…               # job-evaluate-response — keep this DIFFERENT from the main model
```

`MODEL_PRICES` (USD per 1M tokens) drives cost estimation. Free models correctly show a provider cost of `$0` while still producing a hypothetical estimate, so they stay comparable in a benchmark.

Moving this configuration to Postgres later means changing `ROLE_CONFIG` in `llm_factory.py` and nothing else.

## Failure behaviour

Each dependency degrades independently — one outage never ends a turn:

| Fails | Result |
|---|---|
| Pinecone / embeddings | Answers without retrieval; the prompt tells the model to be explicit about uncertainty |
| Upstash Redis | Session becomes stateless; conversation still works within the turn |
| Kundli source | Prompt says the chart is unavailable and instructs the model not to guess placements |
| Classifier model | Falls back to deterministic keyword classification |
| Evaluator model | Deterministic checks still score the turn; the metrics block says the judge was unavailable |
| Main model | Configured fallback model; if that fails too, an honest apology — never an invented reading |

## Current limitations

- **Upstash Redis is not connected.** The code is complete and tested against a fake, but `.env` holds only a database UUID where `UPSTASH_REDIS_REST_TOKEN` needs a JWT-style REST token, and `UPSTASH_REDIS_REST_URL` is empty. Until both are filled from the Upstash console → *REST API*, every session is stateless: no chat history, no chart cache, and rate limiting fails open. Nothing else breaks — each of those degrades independently by design.
- **No frontend is wired to `/chat` yet.** The API is running and documented; connecting the existing React app is the next step.
- **Kundli is one sample chart by default.** `KUNDLI_SOURCE=backend` fetches a live chart from the running Tredev backend, but mapping a real `user_id` to their stored birth details needs the authenticated user endpoint, which is the first piece of the next phase.
- **Metrics are in-process.** Per-turn records are logged to `logs/ai-core.jsonl` but not persisted to Postgres.
- **Knowledge base is small** (90 chunks, English only) and written for this service; it is a working foundation, not a reference corpus.
- **No safety classifier node yet** — safety is handled by prompt rules plus evaluator scoring, not a dedicated moderation pass.

## Next phase

1. Fill in the real Upstash `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN`, then confirm with `python -m scripts.test_connections`.
2. Wire `TredevBackendKundliProvider` to the authenticated user endpoint so a real `user_id` resolves to that user's stored birth details.
3. Point the React frontend at `POST /chat` and put a real auth check in front of it — `user_id` is currently trusted as sent.
4. Persist turn metrics to Postgres for cost dashboards and evaluation datasets.
5. Add the safety-classifier node before generation, and regeneration on a low reward.
6. Expand the knowledge base, and add Hindi (the backend already has a `language` flag and `hindiVocab.ts`).
7. Benchmark candidate models with `scripts/benchmark_models.py` and pick on quality-versus-cost.
