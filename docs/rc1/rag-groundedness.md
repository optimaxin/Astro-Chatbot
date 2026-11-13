# RAG groundedness (RC1, 2026-11-13)

**Target: 95%. Result: not met by the production configuration (Lightning primary, reasoning off).
An opt-in configuration (Bolt with reasoning for the main answer) scored 100% on two runs. Before it can be
recommended, its end-to-end chart accuracy, language handling and latency must be measured (see latency.md).**

Method: same 40 RAG cases, same rule-based scoring (`benchmarks/scoring.py`, unchanged), and the same production
grounding protocol (`--grounding production`, citation post-processing as users get it). The LLM judge was
used in the first runs, but it kept failing (circuit opening) and does not affect the score, so the later runs
used `--judge none`. Pinecone index, embeddings, chunking, ingestion and retrieval settings are unchanged:
retrieval recall@4 is 97.8% (phase 3), so the failures are in answering, not retrieval.

## Held-out set (new)

`benchmarks/cases_rag_holdout.py` has 20 new cases: new fictional passages, 13 answerable and 7 not. They were
written **before** the prompt was changed, and they are not part of the 220-case benchmark. Because the prompt
rules below were written after reading the 40 cases' failures, the 40 cases are no longer an unbiased test,
and the held-out set is the check against overfitting.

## Results (pass rate = rag_groundedness; one row per run)

| Configuration | Prompt | Original 40: Lightning | Original 40: Bolt | Held-out 20: Lightning | Held-out 20: Bolt |
|---|---|---:|---:|---:|---:|
| production, 2026-11-12 | v0 | 87.5% (1 timeout) | 87.5% | n/a | n/a |
| production, RC1 baseline re-run | v0 | 92.5% | 87.5% | 70.0% | 60.0% |
| production, after rules, run 1 | v1 | 87.5% | 92.5% | 70.0% | 75.0% |
| production, after rules, run 2 | v1 | 82.5% | 90.0% | 75.0% | 75.0% |
| Lightning, reasoning on | v1 | 82.5%: 33/33 answered correctly, 7 timeouts at 25 s | n/a | 90.0%: 18/18 answered correctly, 2 timeouts | n/a |
| **Bolt, reasoning on, run 1** | v1 | n/a | **100%** | n/a | **100%** |
| **Bolt, reasoning on, run 2** | v1 | n/a | **100%** | n/a | **100%** |

Model-call latency (benchmark, single call, no retrieval):

| Configuration | p50 | p95 |
|---|---:|---:|
| Lightning, reasoning off | ~0.9–1.1 s | 2.3–5.2 s |
| Bolt, reasoning off | ~0.6–0.7 s | 1.1–1.2 s |
| Lightning, reasoning on | 13.5 s | 20.9–23.6 s (15% timeouts) |
| Bolt, reasoning on | 4.7–5.3 s | 6.9–7.2 s |

How to read this:

- **The prompt rules (v1) did not measurably change Lightning.** It scored 87.5/92.5% before and 87.5/82.5%
  after on the original 40, and 70% before and 70/75% after on the held-out set. With 40 cases, one run's
  standard error is about ±4.7 pp, so these differences are noise.
- **Bolt improved modestly:** from 87.5% to 90–92.5% on the original 40, and from 60% to 75% on the held-out set.
- **The dominant factor is reasoning, not the prompt.** With reasoning on, both models answered every case they
  completed correctly. Lightning with reasoning is too slow for chat. Bolt with reasoning is about 5 s per call.

## Failure classification (Lightning, reasoning off: every failure in the runs above)

| Category (brief's taxonomy) | Cases | What happened |
|---|---|---|
| Unsupported answer | rag-011 (all runs), rag-008, rho-015, rho-020 | Gave Venus's sign when only the Lagna is stated, invented a Holi date, a Moon house and a pass mark for the wrong course |
| Incorrect evidence interpretation | rag-020, rag-018, rag-032, rho-016, rho-017 | Read silence as "No" (e.g. "can I attend in person?" was answered "No, priests do it on your behalf"); rag-036 applied a Varanasi table to Chennai |
| Multi-source reasoning error | rag-017 (all runs), rag-038 (baseline), rho-014, rho-018 | Looked up a table using the Lagna instead of the Moon sign, and counted the retrieved profiles as the platform total |
| Arithmetic error | rho-007 (all runs), rag-014 | Didn't subtract 9 hours from 15:20, and applied the wrong benefit (free chat minutes) to a consultation |
| Missing citation | none for Lightning; Bolt rag-039 / rho-006 | Bolt abstained on an answerable case, so it gave no citation |
| Invalid citation attribution | **none** | **0 fabricated citation tags across all 13 runs** (validated by `process_citations`) |
| Infrastructure | rag-001 (2026-11-12) | 25 s timeout; it did not recur |

The rules in `app/rag/grounding.py` (v1) target exactly these categories, phrased generally:

- answer the exact question first;
- a neighbouring value is not the answer;
- silence is never a yes or a no;
- passages are not a catalogue;
- do the arithmetic;
- check the key a table is indexed by;
- prefer the newer of two conflicting passages.

The rules are kept because they don't hurt Lightning (within noise) and help Bolt. The Bolt-with-reasoning
runs used them, but Bolt with reasoning was not run without them, so their share of that 100% is unmeasured.
They are not enough to make Lightning without reasoning reliable.

## Citations

- Citations refer only to retrieved passages. A `[K#]` tag that points at nothing is removed and counted;
  0 occurred in the benchmark.
- Titles come from real document metadata (`source_title`), and the citation list is built only from tags
  that were actually used.
- Citations are not shown in the chat (default `CITATION_DISPLAY=metadata`, 2026-11-13); they are validated and returned in the API `citations` field. The opt-in footer bubble is added at the API layer and is never written into conversation
  history (phase 3; tests in `tests/test_grounding.py`).
- Whether each citation actually supports its sentence is not checked automatically. That is the
  "citation correctness" dimension in the human review (`docs/human-eval/RUBRIC.md`).

## The option, and what it would take

`NVIDIA_MAIN_REASONING_MODELS` is new, opt-in and off by default. The main answer of a listed model is
generated with reasoning; the classifier, evaluator and fallback calls are not.

The candidate configuration:

```
NVIDIA_MODEL=nvidia/nemotron-3.5-bolt-e4b
NVIDIA_FALLBACK_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
NVIDIA_MAIN_REASONING_MODELS=nvidia/nemotron-3.5-bolt-e4b
LLM_HEDGE_AFTER_SECONDS=10
```

It is **not** switched on. Lightning stays primary as instructed, because RAG groundedness is one dimension
only. The decision needs the end-to-end chart accuracy, language and latency comparison in `latency.md`, and
the owner's call on the latency trade-off.
