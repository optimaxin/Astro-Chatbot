# NVIDIA provider (beta) — architecture, configuration and operations

During the beta the Kundli chatbot generates answers with NVIDIA's free hosted models. OpenRouter is **not removed**. It stays fully configured and you switch back to it by changing `LLM_PROVIDER`. No code changes are needed to switch.

## Architecture

```text
graph node ──► get_llm(role, intent=…)                     app/llm/llm_factory.py   (gateway)
                 │
                 ├─ model_router.route_main(...)           app/llm/model_router.py  (which model)
                 ├─ model_registry.get_spec(model)         app/llm/model_registry.py (what it accepts)
                 ├─ providers.get_provider(LLM_PROVIDER)   app/llm/providers/{nvidia,openrouter,gemini}.py
                 ├─ reliability: classify / backoff / circuit   app/llm/reliability.py
                 ├─ streaming (LLM.stream)                 app/llm/streaming.py
                 └─ usage_tracker.record(LLMUsage)         app/llm/usage_tracker.py
                        └─ usage_ledger.write → logs/usage.jsonl, Redis counters,
                                                Postgres ai_usage (admin Token Meter)
```

The task brief named `llm.gateway.js`, `model.router.js` and similar files. ai-core is a Python service, so those map onto the existing module layout:

| Brief | Implemented as |
|---|---|
| `llm.gateway.js` | `app/llm/llm_factory.py`. The module name and its public API (`get_llm`, `LLM.complete`, `describe_roles`) are unchanged, so no caller changed. |
| `model.router.js` | `app/llm/model_router.py` |
| `model.registry.js` | `app/llm/model_registry.py` |
| `token.tracker.js` | `app/llm/usage_tracker.py`, extended |
| `providers/nvidia.provider.js` | `app/llm/providers/nvidia.py` |
| `providers/openrouter.provider.js` | `app/llm/providers/openrouter.py` (plus `gemini.py`, which already existed as a code path) |

**Provider interface** (`providers/base.py`). An adapter supplies only what differs between upstreams: its credentials and base URL, the error message when the key is missing, the model id for each role, and any vendor-only request fields. The gateway implements the following once, for every provider:

- chat completions, streaming and non-streaming
- tool calling (`tools`, `tool_choice`, parsed `tool_calls` on the result)
- structured output (`response_format` with `json_object` or `json_schema`)
- usage reporting
- error normalisation, retries, fallback and the circuit breaker

### Models and routing

| Role / route | Model (default) | When it is used |
|---|---|---|
| Main answer | `nvidia/nemotron-3.5-lightning-30b-a3b` | General conversation, RAG and reasoning |
| Main → fallback | `nvidia/nemotron-3.5-bolt-e4b` | Only after a technical failure of the main model (see below) |
| Image turns | `nvidia/nemotron-3.5-bolt-e4b` | Lightning is text-only (NVIDIA returns a 400 for image input), so the router sends images to the first vision-capable model |
| Light route | `nvidia/nemotron-3.5-bolt-e4b`, with Lightning as its fallback | Intents listed in `NVIDIA_LIGHT_INTENTS` (default `smalltalk`, which covers greetings and thanks) |
| Intent classifier | `nvidia/nemotron-3.5-bolt-e4b` → Lightning | Every turn; falls back to keyword rules if both fail |
| Quality judge | `nvidia/nemotron-3.5-bolt-e4b` | Sampled turns, controlled by `EVALUATION_SAMPLE_RATE` |
| Not routed | `mistralai/mistral-large-4-1t-a52b` | Benchmark only. Set `NVIDIA_MODEL` to it to make it primary again |

**Why Lightning instead of the planned Mistral Large 4.** The live tests on 2026-11-12, with the project's key, gave these results:

- **Mistral Large 4:** about 60 s in the queue, then about 2 tokens/s.
  - It reasons by default. With a 400-token budget, 399 tokens were hidden reasoning and the visible reply was empty.
  - `enable_thinking=false` has no effect, and `reasoning_effort="none"` is rejected with a 400. The lowest accepted value is `"low"`.
  - Even at `"low"`, one sentence took 139 s.
- **Lightning:** about 1–2 s once thinking is switched off.
- **Bolt:** under 1 s.

The decision was made on 2026-11-12. All model ids were verified in NVIDIA's live catalogue and on the live endpoint.

### Verified live capabilities (2026-11-12)

| | Lightning | Bolt | Mistral Large 4 |
|---|---|---|---|
| Plain completion | ✓ about 1–2 s, **only with** `chat_template_kwargs.enable_thinking=false`; without it all tokens go to reasoning and the reply is empty | ✓ under 1 s | ✓ 77–181 s |
| Reports token usage | ✓ | ✓ | ✓ |
| Streaming, with usage in the final chunk | ✓ | ✓ | (timed out) |
| JSON mode | ✓ | ✓ | (timed out) |
| Tool calling, with correct arguments | ✓ | ✓ | (timed out) |
| Image input | ✗ 400 "multimodal processing is not enabled" | ✓ read the text in the image correctly | (timed out) |
| Hinglish question answered in Hinglish | ✓ | ✗ answered in Devanagari | — |

These settings are recorded in `app/llm/model_registry.py`, so each model is sent only the reasoning switch it accepts.

Routing is a rule table, not a second classifier. It reuses the intent label the classifier already produces. Turns that carry an image go to the first model, in the order main → fallback → light, that is not known to be text-only. With the current defaults that is Bolt.

### When fallback happens (and when it doesn't)

| Failure | Retry same model? | Fall back? | Counts toward circuit? |
|---|---|---|---|
| Timeout | **no**: a stalled model goes straight to the fallback | yes | yes |
| Connection error | yes | yes | yes |
| 429 rate limited (honours `Retry-After`, capped) | yes | yes | yes |
| 5xx | yes | yes | yes |
| 404 model not found | no | yes | yes |
| Empty reply, open circuit, unsupported capability | no | yes | no |
| 401/403 auth | no | **no** (same key would fail) | no |
| 400/422 bad request | no* | **no** | no |
| Client cancelled a stream | no | no | no |

\* A 400 that names an optional parameter (`response_format`, `stream_options`, `chat_template_kwargs`, `temperature`) is a special case. The gateway records that the model rejects that parameter and repeats the call once without it. Later calls to that model omit the parameter from the start.

Limits:

- **Retries:** at most `LLM_MAX_RETRIES` per model (default 2, so 3 tries), with exponential backoff and full jitter.
- **Deadline:** `LLM_DEADLINE_SECONDS` (default 120) bounds all retries and the fallback together. Nothing retries indefinitely.
- **Circuit breaker:** opens after `LLM_CIRCUIT_FAILURE_THRESHOLD` consecutive technical failures. It stays open for `LLM_CIRCUIT_COOLDOWN_SECONDS`, then lets one trial request through. While the circuit is open, the model is skipped and its fallback answers. The breaker is per process, and `/health` shows it under `dependencies.llm.circuits`.

**No duplicate user messages.** Retries and fallback both happen inside one `LLM.complete()` call. Conversation history is written once, in `save_metrics`, and only when the turn succeeded. This is covered by `tests/test_llm_streaming_e2e.py::test_fallback_turn_saves_the_user_message_exactly_once`.

### RAG

RAG is unchanged. The Pinecone index, embedding model (`llama-text-embed-v2`), ingestion, chunking and retrieval were not touched, and the knowledge base was not regenerated. Embeddings come from Pinecone, not from the LLM provider, so switching provider has no effect on them.

Retrieved chunks reach the NVIDIA prompt through the same `build_context` node as before, and `sources` (document::topic plus score) are still returned on `/chat`. The system prompt gained one rule: when neither the chart nor the retrieved knowledge covers the question, the model must say so and mark any general view as general.

### Streaming

`LLM.stream()` provides:

- token deltas as they arrive
- retry and fallback **only before the first token**, so two models' output is never spliced into one message
- cancellation through `close()`, which closes the upstream HTTP response and is recorded as `cancelled`
- usage taken from the provider's final usage chunk

`/chat` itself still returns one JSON response, which is what the current frontend (`KundliChatWidget`) consumes; that client has never streamed. Exposing streaming to browsers would also need the output guardrail (`guardrail_out`) to run on partial text. That is a separate change and was deliberately not made here.

## Environment variables

Set these in `ai-core/.env`. `.env.example` could not be edited from this session (its directory is permission-restricted), so copy this block into `.env.example` by hand:

```bash
# ── LLM provider ─────────────────────────────────────────────
LLM_PROVIDER=nvidia                 # nvidia (default) | openrouter | gemini

# ── NVIDIA (build.nvidia.com) ────────────────────────────────
NVIDIA_API_KEY=                     # nvapi-...  server-side only, never sent to the browser
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_TIMEOUT_SECONDS=25
NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
NVIDIA_FALLBACK_MODEL=nvidia/nemotron-3.5-bolt-e4b
NVIDIA_CLASSIFIER_MODEL=nvidia/nemotron-3.5-bolt-e4b
NVIDIA_EVALUATOR_MODEL=nvidia/nemotron-3.5-bolt-e4b
NVIDIA_BENCHMARK_MODEL=mistralai/mistral-large-4-1t-a52b
NVIDIA_LIGHT_MODEL=nvidia/nemotron-3.5-bolt-e4b   # empty = send every intent to NVIDIA_MODEL
NVIDIA_LIGHT_INTENTS=smalltalk                    # comma-separated classifier intents

# ── Reliability (all providers) ──────────────────────────────
LLM_MAX_RETRIES=2
LLM_BACKOFF_BASE_SECONDS=0.5
LLM_BACKOFF_MAX_SECONDS=8
LLM_DEADLINE_SECONDS=120
LLM_CIRCUIT_FAILURE_THRESHOLD=5
LLM_CIRCUIT_COOLDOWN_SECONDS=30

# ── Shadow roles (advisory only; neither changes a delivered reply) ──
NVIDIA_QUALITY_CHECKER_MODEL=nvidia/nemotron-3-ultra-550b-a55b   # independent checker; empty = not configured
QUALITY_CHECKER_MODE=shadow         # shadow (default) | off
QUALITY_CHECKER_SAMPLE_RATE=0.1     # share of eligible replies checked, 0.0 - 1.0
NVIDIA_HUMANIZER_MODEL=             # empty = the main model (Lightning) with the humanizer's own prompt
HUMANIZER_MODE=shadow               # shadow (default: records, never applies) | on | off
HUMANIZER_MIN_ISSUES=1              # replies with fewer language issues are not humanized
HUMANIZER_CLEAN_SAMPLE_RATE=0.05    # shadow only: share of already-clean Hindi/Hinglish replies still tried and judged

# ── Usage tracking ───────────────────────────────────────────
USAGE_TRACKER_MAX_RECORDS=5000      # in-memory call records kept per process (oldest evicted); the ledger keeps all
```

The shadow roles run in a separate small background pool (`SHADOW_MAX_PENDING`, 50 jobs); when it is full a sample is dropped, never run on the request thread. `/health` reports `dependencies.background` (queue depth, drops, failures, memory). Read the results with `python -m scripts.shadow_report`.

The OpenRouter variables (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, …) are unchanged and are still read whenever `LLM_PROVIDER=openrouter`. The existing per-user limit (`RATE_LIMIT_PER_MINUTE`, enforced in Redis) still applies to `/chat`.

## API changes

`POST /chat`: the request is unchanged, and so is every existing response field. `usage` gained four optional fields:

| Field | Meaning |
|---|---|
| `usage.provider` | `nvidia`, `openrouter`, … |
| `usage.fallback_used` | the answer came from the fallback model |
| `usage.token_source` | `provider` (counts reported by the upstream), `estimated` (the upstream reported nothing, so `input/output/total_tokens` stay `null` and `estimated_total_tokens` holds a ~4 chars/token estimate), or `unavailable` |
| `usage.estimated_total_tokens` | set only when `token_source` is `estimated` |

`GET /health`: `dependencies.llm` now reports `provider`, whether that provider is `configured`, the `models` and `fallbacks` for each role, and `circuits` (the breaker state per model).

## Token Meter

- Every attempt is recorded, including failed retries. Each record carries: input, output and total tokens; model; provider; latency; success; `fallback_used`; a normalised `error_code` (for example `rate_limited:429` or `timeout`); `attempt`; `streamed`; `token_source`; timestamp; and `request_id`, `user_id` and `session_id`, the same identifiers as before.
- **Fixed during this work:** the Postgres insert used `ON CONFLICT (request_id, role) DO NOTHING`. When the primary model failed, its failure row landed first, and the successful fallback row was then discarded. The Token Meter therefore under-reported exactly the turns that used the fallback. The insert now upserts, and only over a non-success row. The surviving row keeps `fallback_used = true` and the primary's `error_code`, so admins can see what triggered the fallback.
- **No database migration is needed.** No columns were added. The new fields (`token_source`, `attempt`, the estimates) go to the JSONL ledger, and estimates never go into the token columns. The admin UI already groups by provider and model (`aiUsageRepository.getModelUsage`), so `nvidia` rows appear per model with no backend or frontend change.
- **Cost:** NVIDIA models have no `MODEL_PRICES` entry, so estimated cost is $0 and `actual_cost_usd` is `NULL`. The Token Meter's existing banner then says "no actual billing amount — on a free tier nothing is charged". Free usage is never presented as spend.

## Testing

```bash
cd ai-core
python -m pytest --deselect tests/test_integration.py      # offline, deterministic (mocked endpoints)
python -m pytest tests/test_integration.py                 # live: uses the active provider
python -m scripts.smoke_nvidia                             # live capability probe, the 3 models
python -m scripts.test_connections                         # live round-trip via the gateway
python -m benchmarks.run --mock                            # harness self-check, offline
python -m benchmarks.run                                   # live benchmark, 220 cases x 3 models
```

Test files added: `tests/test_llm_gateway.py` (providers, routing, error normalisation, retries, fallback, circuit breaker, capability downgrade, tool calls, token accounting), `tests/test_llm_streaming_e2e.py` (streaming, cancellation, RAG prompt integration, fallback without duplicate history, graceful total failure, the `/chat` contract the widget reads, the per-user 429, `/health`) and `tests/test_benchmark_harness.py`.

## Benchmark suite

`benchmarks/` holds the dataset and the harness.

- **Dataset:** 220 cases. 35 general, 40 RAG (12 of them unanswerable from the given sources), 30 reasoning, 25 coding, 35 multilingual (Hinglish, Hindi, Tamil, Bengali, Marathi, Gujarati, Telugu), 30 function calling (8 that must *not* call a tool), and 25 image cases. Images are rendered deterministically with Pillow.
- **Runs:** every model gets identical messages, tools, temperature (0.2) and `max_tokens` (900). Cases run case-major, so a slow period on the endpoint affects all models equally. Each model is pinned with no fallback. Cases marked for consistency run twice.
- **Judge:** an optional judge model that is not under test (default `nvidia/nemotron-3.5-super-vl-120b-a12b`; `nemotron-3-super-120b` and `mistral-large-2` return 404 for this account).
- **Kept out of the Token Meter:** benchmark calls are not written to the usage ledger.
- **Output:** results go to `logs/benchmarks/<timestamp>/`, as `results.jsonl`, `summary.json` and `report.md`. The report covers pass rate per category, RAG groundedness, tool-call accuracy, reply-language accuracy, judge scores, consistency, latency (mean/p50/p95) and reported tokens. It does not show costs.
- **Scale:** a full run is about 723 generation calls plus about 660 judge calls. At the default `--rpm 30` that is roughly 45–60 minutes.

## Deployment

1. Create an API key at <https://build.nvidia.com> and set `NVIDIA_API_KEY` in the ai-core environment (Render or the server's `.env`). Keep it server-side only. The browser talks to `/chat`, never to NVIDIA.
2. Set `LLM_PROVIDER=nvidia`. It is the code default, but an existing `.env` that names another provider overrides it.
3. Run `python -m scripts.smoke_nvidia`, then `python -m scripts.test_connections`.
4. Restart ai-core and check `GET /health`: expect `llm.provider = "nvidia"` and `configured: true`.
5. Watch the admin Token Meter's Models tab. `nvidia` rows should appear for each model.

No backend or frontend deploy is required.

## Rollback

- **Switch provider (instant, no code change):** set `LLM_PROVIDER=openrouter` and restart ai-core. All roles go back to the `OPENROUTER_*` models.
- **Keep NVIDIA but drop the light route:** set `NVIDIA_LIGHT_MODEL=` (empty).
- **Code rollback:** ai-core is not tracked in git yet, so commit it before deploying, so that a code rollback is a `git revert`.

## Known limitations

- **Mistral Large 4 is too slow on the free tier to serve live chat** (77–181 s per reply). Its JSON, tool and vision support could not be verified because every probe timed out.
- **Lightning can misstate chart placements.** In a live turn it implied Jupiter was in the career (10th) house when the chart has `Jupiter=Scorpio/H3`. The deterministic contradiction checker (`evaluator.check_kundli_contradictions`) only parses explicit "X in your Nth house" claims and did not catch this indirect phrasing.
- **Turn latency is about 11–16 s, but only about 5 s of that is model time.** The 3 calls are classifier ~1.5 s, answer ~1.3 s and judge ~2 s. The rest is Redis, Pinecone and metrics I/O that existed before this change. Lowering `EVALUATION_SAMPLE_RATE` takes the judge call off most turns.
- **Breaker state and learned capabilities are per process.** They are not shared through Redis.
- **Free-tier rate limits are not published per account.** Heavy traffic will hit 429s. The gateway backs off and falls back, but the free tier is not a production SLA.
- **Images are supported at the gateway and in the benchmark only.** The chat widget has no image upload, so "image queries" are not reachable from the UI yet.
- **The browser cannot stream.** Streaming exists at the gateway level only (see above).
- **Per-user limits fail open when Redis is down.** This is the existing behaviour.
