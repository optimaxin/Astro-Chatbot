# RC1 evaluation results (2026-11-13)

Everything below was measured on the code in this branch, from a developer machine, against the real NVIDIA,
Pinecone and Redis services. Nothing here was measured on the production host, and no human reviewed any answer.
**Completed** and **pending** items are marked separately. Raw outputs are in `logs/` (not committed).

## 1. Gate status

| Gate | Status | Evidence |
|---|---|---|
| Chart-fact accuracy | **PASS** | 100%: 0 of 78 checked claims wrong, 0 failures, in the 51-turn evaluation (also 71 and 74 claims in two earlier runs of the same set) |
| Test suite | **PASS** | 637 passed (offline + live integration), final code |
| Security probes (prompt injection, key disclosure, hidden reasoning, cross-user) | **PASS** | 8/8 on the production config and 8/8 on the reasoning-on candidate |
| Session, ownership, rate limit, fallback, circuit breaker | **PASS** | reliability probe 11/11 |
| Multi-worker sessions, shared `CHAT_SESSION_SECRET` | **PASS** | 5/5: instance A's token is accepted by B (same key) and rejected by a different key; both report the same key fingerprint |
| Private reasoning never reaches the client, ledger or logs | **PASS** | 12 offline tests, no leak in 279 reasoning-on benchmark replies, one live stream checked |
| Hedge (6 s) and timeout (25 s) behaviour | **PASS** | `timing_probe` 5/5; real worst case is **31 s** (6 s hedge, then the fallback's own 25 s timeout) |
| Test rows never reach production usage tables | **PASS** | `audit_traffic`: 0 test rows since the fix; found and fixed two leaks (see §6) |
| Language preservation, target 100% | **NOT MET** | 98.0% (Hinglish 24/25) in the 51-turn evaluation |
| RAG groundedness, target 95% | **NOT MET** | production config 82.5–92.5% across runs (§3) |
| Latency, median under 5 s and p95 under 8 s | **NOT MET / NOT CERTIFIED** | §4: runs disagree; cannot be certified from this machine |
| Human evaluation | **PENDING** | workflow ready, 0 reviewers have scored anything |
| Migration 061 | **PENDING APPROVAL** | rehearsed locally; not applied anywhere real (see `migration-058-audit.md`) |
| Latency from staging/production infrastructure | **PENDING** | no staging environment exists |
| Four-configuration end-to-end chart/language evaluation | **PARTIAL** | only the production configuration (Lightning, reasoning off) completed; the other three runs crashed on a console-encoding error and were not repeated |

## 2. Model configurations

### 2a. Benchmark (rule-based `benchmarks/scoring.py`, unchanged), same 42 cases per configuration

42 cases = the first 6 per category of the 220. Reasoning-on runs use `max_tokens` 4000 and reasoning-off 900, which
is a necessary difference (reasoning tokens count against the limit).

| Configuration | Pass | Errors | Call p50 | Call p95 | Output tokens (mean) |
|---|---:|---:|---:|---:|---:|
| Lightning, reasoning off | 64.3% | 13 | 1.1 s | 4.6 s | 133 |
| Lightning, reasoning on | 71.4% | 6 | 4.6 s | 12.4 s | 773 |
| Bolt, reasoning off | 52.4% | 9 | 0.9 s | 4.8 s | 74 |
| Bolt, reasoning on | 83.3% | 0 | 4.5 s | 12.5 s | 360 |

Reading it: the benchmark calls the model directly and bypasses the router, so image cases fail on Lightning (it is
text-only; production sends images to Bolt) and count as "errors" there. The reasoning category (math and logic) is
0/6 with reasoning off and 6/6 with it on for both models. Full 220, Bolt reasoning on: 89.1%, 0 errors; the earlier
reasoning-off full runs scored Lightning 77.7% and Bolt 81.8% (same scoring, run on a different date).

### 2b. Time to first visible token, streamed, 24 turns each, production-like system prompt

| Configuration | TTFT p50 | TTFT p95 | Total p50 | Total p95 |
|---|---:|---:|---:|---:|
| Lightning, reasoning off | 0.48 s | 1.17 s | 4.3 s | 7.1 s |
| Lightning, reasoning on | 21.7 s | 42.6 s | 26.8 s | 43.8 s |
| Bolt, reasoning off | 0.37 s | 0.54 s | 2.7 s | 4.5 s |
| Bolt, reasoning on | 5.2 s | 7.8 s | 6.7 s | 9.6 s |

`/chat` returns a finished answer today (no browser streaming), so users wait for the total, not the TTFT.

### 2c. Recommendation

**Keep Lightning (reasoning off) as primary and Bolt as fallback, which is the current production configuration.**
Reasoning-on wins the benchmark but loses where users feel it: Lightning with reasoning has a 22 s first token,
and the Bolt-with-reasoning routing profile was end-to-end median 12.7 s, p95 26 s, with 1.8% failures (§4).
`NVIDIA_MAIN_REASONING_MODELS` stays off. Workload routing (Bolt with reasoning only for hard knowledge questions)
is untested end to end and **not** recommended yet; it needs its own evaluation.

## 3. RAG groundedness (see `rag-groundedness.md`)

Original 40 RAG cases, production grounding protocol, rule-based scoring:

- Lightning reasoning off: 87.5%, 92.5%, 87.5%, 82.5% over four runs. Standard error per run is about ±5 pp, so the
  evidence rules added today did not measurably change it.
- Bolt reasoning off: 87.5%, 87.5%, 92.5%, 90.0%.
- Bolt reasoning on: **100%** twice, on the 40 and on 20 held-out cases, 120/120 in total, at about 5 s per call.
- Fabricated citations: **0** in all 13 runs. The chat no longer shows sources (§5).

## 4. Latency

| Run | Turns | Median | p95 | p99 | Failures | Fallback | 429s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Profile A, production settings | 108 | 8.3 s | 16.1 s | 19.9 s | 0% | 12% | 0 |
| Profile B, production settings | 108 | 7.8 s | 15.6 s | 17.1 s | 0% | 11% | 0 |
| End-to-end evaluation, hedging off | 51 | 4.6 s | 10.2 s | 11.2 s | 0% | 0% | 0 |
| Profile, Bolt-reasoning routing | 108 | 12.7 s | 26.1 s | 28.6 s | 1.8% | 8% | 0 |

- **The targets are not met.** Two independent production-settings runs (216 turns) agree on a median of about 8 s
  and p95 of about 16 s. The evaluation run looks better (4.6 s, 10.2 s) but uses a different question mix and had
  hedging off, so it cannot be used to claim the target.
- **Bottleneck:** the main-model call (median about 6 s inside a turn, against about 1 s for the same model on a
  short prompt). Prompts carry about 4,000 tokens (chart, passages, history). Pinecone is 0.3–0.6 s at p50.
  Redis reads are fast except for a few slow outliers.
- **Hedge:** the 6 s hedge fires on 11–12% of turns and the fallback answers them. NVIDIA returned **no** 429s.
- **Not done:** staging latency from production-equivalent infrastructure. This machine's network is not that.

## 5. Language preservation

This was the main regression found today. Lightning answered Hinglish questions in plain English about half the time
(47% correct on two probe questions) even before today's changes, and my new "answer in the first sentence" evidence
rule shortened replies and dropped it to 19%. Cause: the guard ignored replies under 15 words, so a six-word English
answer went through unchecked, and when it did fire the repair instruction only said "wrong language".

Fixes (committed): the guard judges short replies too; the repair says the reply is plain English and must be
Hinglish; the lock says one-line answers are not exempt. Results on the final code:

- 51-turn evaluation: english 13/13, hindi 13/13, **hinglish 24/25** (98.0%), up from 96.1%.
- Three hardest Hinglish questions, 10 runs each: 29/30 correct by the production guard, 25/30 by the benchmark
  checker. The checker is stricter on very short Hinglish ("Saturn 6th house hai." is flagged); its scoring was not
  changed.
- A dedicated short "rewrite into Hinglish" repair was tried and **removed**: it returned English every time.

Remaining gap: about one Hinglish reply in 25–30 is still English after repair. Citations in chat: by product decision
the chat never shows where information came from (`CITATION_DISPLAY=metadata`); sources stay in the API `citations`
field.

## 6. Data integrity

- `scripts/audit_traffic.py` (read-only): no test rows since the fix; the 11 timing-probe rows are preserved
  (labelled `user` until 061, become `debug`).
- Found and fixed two leaks: the integration tests wrote 203 `pytest-*` rows to `ai_evaluations`, and my timing probe
  wrote 11 stub rows to `ai_usage` (it disabled metering on the module, not on the tracker). `usage_store` now never
  labels a row with no user id as `user`.
- Decided (2026-11-13): signed anonymous visitors (`anon:`) count as real users. 061 leaves them labelled `user`,
  including the 28 older rows from 2026-11-12 that are probably RC testing but cannot be proven so.

## 7. Not done, or unreliable

- Human evaluation: no scores exist. The report prints "NO HUMAN REVIEWS RECORDED — beta gate NOT PASSED".
- The first four-configuration end-to-end runs crashed on Windows console encoding (printing Hindi) and wrote no
  results although the chain manifest said "ended". Only the production configuration was re-run.
- `profile-prod-a` spans about 4.5 hours of wall clock because the machine slept during it; its per-turn timings
  are unaffected (slowest turn 26 s), but its start and end times are not meaningful.
