# Migration 058 audit (RC1, 2026-11-13)

Scope: `backend/migrations/058_ai_traffic_source_and_grounding.sql` in
`optimaxin/Tredev-astro`, and the corrective migration `061` written as a
result. Every production number below came from read-only queries
(`SET TRANSACTION READ ONLY`) run on 2026-11-13. No production row was written
during this audit.

## 1. Status: 058 is already applied in production

| Fact | Evidence |
|---|---|
| PR #1 (`feat/token-meter-model-analytics` → `main`) merged | merge commit `88e3c05`, 2026-11-12 21:13:01 UTC |
| 058 recorded in `_migrations` | `058_ai_traffic_source_and_grounding.sql`, applied_at 2026-11-12 21:14:32 UTC |
| Database | Supabase (`aws-0-ap-northeast-2.pooler.supabase.com`), 182 accounts in `users`. Treated as production. |

058 was recorded 91 s after the merge, which fits the production deploy running
it on start-up. The earlier RC note "migration 058 not applied" was wrong as of
that merge. The rest of this audit is a **post-hoc** review, and the staging
rehearsal in §6 was run afterwards for both 058 and its fix.

Schema drift seen while auditing: `_migrations` also lists
`058_astrologer_commission.sql`, `059_astrologer_intent.sql` and
`060_kundli_description_rewrites.sql`. The last of these is not in
`origin/main`. Two different files share the `058_` prefix. The runner keys on
the full filename, so both run, but the numbering is ambiguous.

## 2. What 058 did

- Added `ai_usage.traffic_source TEXT NOT NULL DEFAULT 'user'` with an index,
  and added the grounding columns plus `traffic_source` to `ai_evaluations`.
  This part is additive.
- Backfilled `ai_usage` rows created on or after 2026-11-12 whose session prefix or
  script user id marks them as internal.
- Did **not** backfill `ai_evaluations`: all 1,096 rows still say `user`.

## 3. How every row is classified (production, 1,756 rows)

The brief said 667 of 763 rows. That was the row count at an earlier dry-run;
the table has grown since, mostly from evaluation runs.

| Label after 058 | Rows | Why |
|---|---:|---|
| evaluation | 1,426 | session `eval-*`, user `test-user-999` (eval_pipeline / human-eval build) |
| debug | 76 | `dbg-*` (54), empty session + user `-` (11), `verify-*` / `nvidia-verify` (9), `connection-*` (2) |
| profiling | 72 | `profile-*` |
| security | 55 | `sec-*`, users `sec-user-a/b` |
| benchmark | 3 | role `job-benchmark-judge` |
| **user** | **124** | everything else (below) |

The 124 rows labelled `user`:

| Group | Rows | Dates (UTC) | Assessment |
|---|---:|---|---|
| `style-*` | 23 | 09-24, 1 min | internal: response-style testing |
| `tokenmeter-*` | 9 | 09-23, 12 min | internal: Token Meter development |
| `verify-*` | 6 | 09-26 | internal |
| `eval-*` | 3 | 09-25 | internal: before the 058 cutoff date |
| `local-*` | 2 | 11-09 | internal |
| `kundli-*`, users `birth-<hash>` | 53 | 09-23 → 09-26 | **unverifiable**: 3 user ids, none of them a registered account, recorded during development days |
| `anon:*\|kundli` | 28 | 11-12 18:58–19:12 | **kept as `user` by decision** (probably RC testing, unprovable): 8 sessions in 14 minutes. The signed anonymous sessions that produce this id format exist only in the unreleased ai-core RC, so these almost certainly came from local RC testing. |

**Finding:** none of the 124 rows can be tied to a real user, and 43 of them
are certainly internal. The production Token Meter's default "Real users" view
therefore shows internal and unverified traffic as real. This is the
silent misclassification the RC brief warned about.

No `ai_usage` user id matches an account in `users` (0 of 182). There was no
recorded chatbot usage between 2026-09-26 and 2026-11-09.

## 4. Could legitimate user traffic be misclassified as internal?

058's backfill rules match only fixed internal prefixes (`eval`, `profile`,
`sec`, `dbg`, `verify`, `connection`, `pipeline`, `bench`, `e2e`, `trace`)
or script user ids, and only for rows created on or after 2026-11-12. Real traffic uses `kundli-<chart>` or
`<identity>|<session>` ids, which never split to those prefixes. **No** real
row could have been labelled internal. The error runs the other way:
internal and unverifiable rows are labelled real.

Two root causes:
1. Local runs of ai-core (`uvicorn` for manual testing) default to
   `AI_TRAFFIC_SOURCE=user` and write to the production database. **Action:**
   set `AI_TRAFFIC_SOURCE=debug` in every non-production ai-core `.env`.
2. The live integration tests wrote 203 `pytest-*` rows to `ai_evaluations`
   because `evaluation_store.write_evaluation` was not stubbed. Fixed in this
   RC: `tests/conftest.py` now stubs it.

## 5. Corrective migration 061 (needs approval: not applied anywhere real)

`backend/migrations/061_ai_traffic_reclassify_history.sql` on branch
`feat/ai-analytics-rc1` (main repo):

1. Rows with internal session prefixes or 058's script user ids (`-`, `nvidia-verify`, …), on any date
   (prefixes now also include `style`, `tokenmeter`,
   `local` and `pytest`) → their internal label.
2. Remaining `user` rows created **before 2026-11-13** that belong to no
   registered account → **`unknown`**. This is a new label meaning "not
   verifiable either way". It is shown in its own Token Meter view
   ("Unverified history") and is excluded from both "Real users" and
   "Internal". **Signed anonymous visitors (`anon:<id>`) are excluded from
   this step**: by product decision (2026-11-13) they count as real users,
   including the 28 older rows from 2026-11-12, which are probably RC test
   traffic but cannot be proven so.
3. `ai_evaluations` takes the label of the `ai_usage` row with the same
   `request_id`. Evaluations with no usage row get the prefix rules, and then
   `unknown` if they are historical with no account.
4. Every change is written to `ai_traffic_reclassification` (table, row id,
   old value, new value, reason). Nothing is deleted.

Predicted production effect (read-only simulation of 061's exact predicates,
re-run 2026-11-13 08:10 UTC after the incident below):

| | Before 061 | After 061 |
|---|---:|---:|
| user | 135 | **28** (the signed anonymous visitors) |
| internal (all internal labels) | 2,833 | 2,887 |
| unknown | 0 | 53 |
| total | 2,968 | 2,968 |

(The total grew from 1,833 to 2,968 between the two runs of this simulation because the RC1 evaluation runs wrote
rows, all correctly labelled `evaluation` or `debug`; the `user` count did not change.)

### Incident found during RC1 (my error, now fixed)

On 2026-11-12, between 22:26 and 22:29 UTC, two local runs of `scripts/timing_probe.py` wrote
**11 stub rows** into production `ai_usage`, labelled `user`. They have user_id `-`, 9 tokens each,
and the scripted 6.5 s / 25 s latencies. The probe set `persist` on the `usage_tracker` *module*,
not on the tracker instance, so metering was never turned off. Fixes:

- the probe now disables the real tracker, asserts that it has, and tags itself `debug`;
- `usage_store` now refuses to label a row with no user id (`None`, `''` or `-`) as `user`, with a
  regression test;
- 061 now applies 058's script-user-id rule to all dates, so these 11 rows become `debug`.
  The re-rehearsal covered this, inserting the rows after 058 as happened in production.

The rows were **not deleted**, because no production write was approved. After a fixed probe run,
both the DB row count and the ledger line count were unchanged.

The column default stays `'user'`. The RC ai-core always writes the label
explicitly. The currently deployed (old) ai-core does not write it, and the
rows it produces come from real chat traffic, so `user` is the right default for them.

## 6. Reversibility and staging rehearsal

There is no staging database in this environment (only the production
Supabase). The rehearsal therefore used a **throw-away local PostgreSQL**:

- The schema was built by running the repo's own migrations in order. All
  succeeded except `052_consultation_slots_price_rename.sql`, which is
  unrelated and fails on an empty database (`price_paise` missing): more
  schema drift.
- The data is synthetic. It reproduces every pattern and count from §3 with
  fake ids, plus control rows: a registered account's history, and an
  anonymous row dated after the cutoff. **No production data was copied.**

Checks, all passed:

| Step | ai_usage user / internal / unknown | Check |
|---|---|---|
| backup (`pg_dump`) | n/a | 468 KB dump taken before any change |
| before (pre-058 schema) | column absent | 1,762 rows (plus 11 stub rows inserted after 058, as in production) |
| after 058 | 141 / 1,632 / 0 | production pattern (124 + 11 stub rows) plus 6 control rows |
| after 061 | 34 / 1,686 / 53 | the 28 anonymous rows and 6 control rows stay `user`; stub rows become `debug`; totals unchanged |
| 061 SQL run again | identical | idempotent |
| `rollback/061_down.sql` | 141 / 1,632 / 0 | restores the post-058 state exactly, including ai_evaluations |
| `rollback/058_down.sql` | column absent | columns and index removed |
| re-apply 058 + 061 | 34 / 1,686 / 53 | same result |

Rollback scripts live in `backend/migrations/rollback/`, which the runner
ignores because it reads only `migrations/*.sql`. `058_down.sql` is
destructive: it drops the grounding columns and their data. It is needed only
if the pre-058 Token Meter must be restored, and that code ignores the
columns anyway.

Rehearsal script: run with the scratch venv (`pgserver`, `psycopg`). It is not
committed because it targets this machine. It is described here so it can be
re-created.

## 7. Are existing reports accurate?

- **Before 061:** the Token Meter's default view counts 124 internal or
  unverified rows as real users. The model, latency and error panels are
  computed on those same rows, so they describe test traffic.
- **After 061:** "Real users" is empty until real traffic arrives through the
  RC ai-core. That is accurate: no verified real usage exists yet. Evaluation
  panels are driven by `ai_usage` (LEFT JOIN), so they follow the corrected
  labels.

## 8. Approval needed

- [ ] Approve applying 061 to production (it runs automatically when the
      backend starts from a `main` that contains it).
- [ ] Set `AI_TRAFFIC_SOURCE=debug` in local and dev ai-core environments.
- [ ] Decide whether to rename the duplicate `058_` files. Renaming an applied
      migration makes it run again; both files are idempotent, but don't rename
      without checking that.
