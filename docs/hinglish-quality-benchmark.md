# Hinglish Quality Benchmark (HQB)

A repeatable measure of whether the chatbot's replies sound natural, grammatical and correct to Hinglish
speakers. Hindi and English are measured the same way. The benchmark only reads replies. It never changes
how the chatbot behaves.

| Part | Where |
|---|---|
| Dataset (250 examples) | `benchmarks/hinglish_quality/data_*.py` |
| Deterministic checks | `benchmarks/hinglish_quality/rules.py` |
| Kundli-fact checks | `benchmarks/hinglish_quality/facts.py` |
| LLM judge (cached) | `benchmarks/hinglish_quality/judge.py` |
| Scoring | `benchmarks/hinglish_quality/scoring.py` |
| Runner, storage, regression gate | `scripts/hinglish_quality_bench.py` |
| Dashboard | `benchmarks/hinglish_quality/report.py` |
| Evaluator regression tests | `tests/test_hinglish_benchmark.py` |
| Native-reviewer sheet | `docs/human-eval/hinglish/review-sheet.csv` |

## 1. Dataset

There are 250 chatbot-style replies, each paired with the question it answers.

| Group | Examples | Gold | Negative |
|---|---|---|---|
| Hinglish, astrology | 100 | 50 | 50 |
| Hinglish, conversational | 50 | 30 | 20 |
| Hindi (Devanagari) | 50 | 30 | 20 |
| English | 50 | 30 | 20 |

**Topics covered.** Career, marriage and relationships, health, property and foreign settlement,
Mahadasha and Antardasha, planetary placements, and everyday conversation.

**Two kinds of example:**
- A **gold** example is a natural, correct reply. The evaluator must not flag it.
- A **negative** example contains known problems. Each problem is annotated as category, wrong span and
  fix. Every negative also carries a **corrected alternative**, which serves as its gold reference.

**Where the negatives come from:**
- The five examples from the production screenshots (`source="screenshot"`).
- Real chatbot output taken from earlier benchmark runs (`source="benchmark-output"`).
- Authored cases covering each error category.

**The chart.** Astrology facts follow the benchmark chart (`scripts/kundli_quality_bench.screenshot_chart`),
so meaning preservation can be checked mechanically.

**Splits.** About a third of the negatives form a deterministic *holdout* split (crc32 of the id). The
screenshot cases always stay in *dev*.

## 2. Dimensions

Each reply gets five scores from 1 to 5.

| Dimension | What it measures | Weight |
|---|---|---|
| Meaning preservation | Kundli facts, dates and meaning kept; no guarantees, invented specifics or system talk | **0.30** |
| Grammar accuracy | Gender and number agreement, verb forms, postpositions, sentence structure | **0.25** |
| Vocabulary accuracy | Roman-Hindi spelling, English words spelt in English, real words, natural Hindi–English combinations | 0.15 |
| Naturalness | Would a fluent speaker say it this way? No literal translation, heavy Sanskrit, English word order or filler | 0.15 |
| Language consistency | One language and script as expected, respectful *aap* throughout, no assumed gender | 0.15 |

**Error taxonomy.** Every error category (`schema.CATEGORIES`) maps to exactly one dimension. For example,
`gender_agreement` maps to grammar, `collocation` to vocabulary, `literal_translation` to naturalness,
`fact_change` to meaning, and `register_mix` to consistency.

## 3. Three signals, so no single LLM decides

1. **Rules** (`rules.py`). These are deterministic patterns, each wrong in every context it matches: a
   feminine noun with a masculine verb, *"X la sakta hai"* with an English noun, *"joda hota hai"*, *tum*
   mixed with *aap*, a singular verb with *hain*, Devanagari in Hinglish, and so on.
   - The rules are separate from the chatbot's own validator. A benchmark that re-ran the production
     rules would be grading the chatbot with its own answer key.
   - Precision comes first. The gold set is the false-positive test.
2. **Facts** (`facts.py`):
   - Chart contradictions, found with the production fact checker used as a measuring instrument.
   - House-lord statements. The production checker does not cover these, so the benchmark adds its own.
   - Invented years.
   - Chart facts of the reference that the reply dropped.
3. **Judge** (`judge.py`):
   - An LLM rates the five dimensions and lists the errors it finds.
   - Verdicts are cached on disk, keyed by judge version, language, question, text and reference, so
     re-runs are identical and free.
   - Calls are paced, because the provider rate-limits.
4. **Human review.** `export-review` writes a sheet for native or fluent reviewers. `import-review`
   stores their sign-off in `benchmarks/hinglish_quality/reviews.json`, and the dashboard and data show
   each example's review status.

## 4. Scoring

```
per dimension:   rule score = 5 − Σ penalties of rule/fact findings (floor 1)
                 penalties: grammar 1.5, vocabulary 1.0, naturalness 1.0, meaning 1.5, consistency 1.5
                 fact_change / machinery / unsafe_claim / language_switch → that dimension = 1
                 each dropped chart fact → meaning −1 (max −3)
                 if a rule fired there:   score = min(rule, judge)
                 else:                    score = (5 + judge) / 2     ← the judge alone can pull to 3, not lower
                 no judge verdict:        score = rule score
overall          = 0.30·meaning + 0.25·grammar + 0.15·vocabulary + 0.15·naturalness + 0.15·consistency
fact gate        a changed or invented Kundli fact, a machinery mention or an unsafe claim caps overall at 2.0
pass             overall ≥ 4.0, no fact gate, and no rule or fact finding (a clean reply)
```

Under the fact gate, a natural-sounding reply with a wrong fact can never score well.

## 5. Modes

`python scripts/hinglish_quality_bench.py <mode> --label <name> [--judge]` writes
`logs/hinglish-bench/<label>.json`. There are three modes:

- **`dataset`** checks the evaluator itself. It scores every gold text, every negative and every
  correction, and reports:
  - how much gold text is flagged;
  - how many negatives are detected;
  - how often a correction is ranked above its original;
  - per-category recall;
  - holdout detection.
- **`chatbot`** asks the live pipeline every example's question, in that example's language and against
  the benchmark chart, and scores the replies.
  - Failed turns are retried once. Any that still fail are reported and excluded from the quality scores,
    because a canned "please ask again" message would otherwise score as perfect language.
- **`correction`** runs the production post-generation language processing on every example. A negative
  should come out fixed; gold text should come out unchanged.

**Other commands:**
- **`judge RUN`** adds judge verdicts to a finished run. This lets a live run and its judging happen at
  different times.
- **`compare BASE NEW --fail-on-regression`** is the regression gate. It exits with code 1 if:
  - overall drops by more than 0.10; or
  - any dimension drops by more than 0.20; or
  - the pass rate drops by more than 5 points; or
  - the fact-gate rate rises by more than 2 points.

  It warns if the two runs used different dataset hashes.
- **`report RUN [RUN…] --out file.html`** builds the dashboard. It shows the overall score, the five
  dimensions, the error count, the most common error categories, and a before/after comparison when given
  more than one run. A filter switches between All, Hinglish, Hindi and English, and every example is
  listed with its errors.

## 6. What each result stores (for fair comparison between releases)

- `benchmark_version`, `scoring_version`, the weights, and `dataset_hash`;
- judge version and model;
- git commit, plus a flag for a dirty work tree;
- the chatbot's `prompt_version`, `validator_version`, and the provider, model and fallback model for each
  role;
- environment overrides used for the run (`HINGLISH_GRAMMAR`, `ENABLE_EVALUATION`, …).

## 7. Release procedure for any language change

```
python -m pytest tests/test_hinglish_benchmark.py tests/test_hinglish_grammar.py     # offline, seconds
ENABLE_EVALUATION=false python scripts/hinglish_quality_bench.py chatbot --label <release> --workers 1
python scripts/hinglish_quality_bench.py judge logs/hinglish-bench/<release>.json
python scripts/hinglish_quality_bench.py compare logs/hinglish-bench/<previous>.json \
       logs/hinglish-bench/<release>.json --fail-on-regression
python scripts/hinglish_quality_bench.py report logs/hinglish-bench/<previous>.json \
       logs/hinglish-bench/<release>.json --out docs/hqb-report-<release>.html
```

## 8. Known limitations

- **The rules and the dataset share an author.** Detection on this dataset is high partly by
  construction. The holdout split was also seen while the rules were refined, so it is not blind. The
  honest measures of quality are:
  - the live `chatbot` mode;
  - the judge;
  - native-speaker review.
- **Native review is still to be done.** Every example is `review="author"` (written by a fluent author)
  until reviewers sign off through the review sheet.
- **The judge is lenient.** The judge model (Nemotron Bolt) scores clearly awkward Hinglish at 4/5. It is
  therefore capped as one signal among three and cannot fail a reply on its own.
- **Live runs are limited by the NVIDIA rate limit.** Use `--workers 1` and `ENABLE_EVALUATION=false`,
  and keep the default turn delay.
