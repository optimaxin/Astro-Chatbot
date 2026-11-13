# RC1 release checklist, deployment order and rollback

Nothing here has been merged or deployed by the RC work. Each box needs an owner's sign-off.

## What is in the release

| Repo | Branch | Target | Contains |
|---|---|---|---|
| ai-core `sagar-goyal162005/Astro-Chatbot` | `feat/nvidia-kundli-chatbot` | `optimaxin/Astro-Chatbot:main` | NVIDIA gateway (Lightning primary, Bolt fallback), chart validation, grounded citations, server-derived identity, RC1 hardening (shared session secret, test isolation, RAG evidence rules, blinded human-eval workflow, probes) |
| main `optimaxin/Tredev-astro` | `feat/ai-analytics-rc1` | `main` | migration 061 (history reclassification, reversible), Token Meter "Unverified history" view, rollback scripts |

Already in production `main` (PR #1, merged 2026-11-12 21:13 UTC): migration 058, the
Token Meter model analytics, and the frontend `aiCoreService` identity headers. The
deployed (old) ai-core accepts those headers (`allow_headers=["*"]`) and ignores them.

## Dependencies

- ai-core RC → backend: verifies access tokens via `AUTH_BACKEND_URL` + `/api/auth/me`
  (already in production backend). Writes `traffic_source` and grounding columns only
  if they exist (058, already applied).
- Token Meter correctness → migration 061 (otherwise "Real users" shows 124 internal/
  unverified rows).
- Frontend identity headers → already deployed; harmless with any ai-core version.

## Deployment order

1. **Approve and merge** `feat/ai-analytics-rc1` → backend deploy runs **061** on start-up.
   Verify: `SELECT traffic_source, count(*) FROM ai_usage GROUP BY 1` → `user` 28 (the signed anonymous
   visitors, kept by decision), `unknown` 53 (plus any rows written since 2026-11-13), totals unchanged;
   `SELECT count(*) FROM ai_traffic_reclassification WHERE migration='061'` > 0.
2. **ai-core environment** (Render), before deploying code:
   - `CHAT_SESSION_SECRET` = one random value ≥32 chars, **identical on every worker/instance**
     (startup now refuses without it).
   - `AUTH_BACKEND_URL` = backend public URL; `AUTH_MODE=optional`.
   - `LLM_PROVIDER=nvidia`, `NVIDIA_API_KEY` (rotate: the current key was pasted into a chat).
   - `EVALUATION_SAMPLE_RATE=0.2`; leave `AI_TRAFFIC_SOURCE` unset (= `user`) in production only.
3. **Merge and deploy** ai-core `feat/nvidia-kundli-chatbot`.
4. **Smoke** (from a machine with the production `.env`, tagged as internal traffic):
   - After deploy, confirm the new behaviour: a Hinglish question gets a Hinglish answer, and no "Source" line appears in chat.
   - `GET /health` on each instance: `llm.configured`, `redis.reachable`, `pinecone.reachable`,
     and the same `auth.session_secret` fingerprint on every instance (never `ephemeral`).
   - `python -m scripts.test_connections`, `python -m scripts.security_probe`.
   - one anonymous chat in the web app, reload, confirm history persists.
5. **Token Meter**: "Real users" shows only post-deploy chat; "Internal" and
   "Unverified history" are separate.
6. **Before widening the beta**: two human reviewers complete the blinded sheets
   (`docs/human-eval/RUBRIC.md`) and the gate in `report-20261113.md` passes.

## Release checklist

- [ ] Human evaluation: 2 reviewers × 60 items, gate passed (currently **NOT PASSED — no reviews**)
- [ ] RAG groundedness decision recorded (target 95% **not met**: 82.5–92.5%, see `rag-groundedness.md` and `evaluation-results.md`)
- [ ] Latency re-measured from the production host (**not met** from a dev machine: median ~8 s, p95 ~16 s; see `evaluation-results.md` §4)
- [ ] Language preservation decision recorded (98.0% in the 51-turn run, target 100%; ~1 Hinglish reply in 25–30 stays English after repair)
- [ ] Migration 061 approved; 058 audit read (`migration-058-audit.md`)
- [ ] `CHAT_SESSION_SECRET` set identically on all instances; `/health` fingerprints match
- [ ] NVIDIA API key rotated
- [ ] `AI_TRAFFIC_SOURCE=debug` set in every non-production ai-core `.env`
- [x] Offline tests + live integration green: 637 passed on the final code (`evaluation-results.md`)
- [ ] Pre-existing `main` issues triaged (not caused by RC): frontend `tsc` error in
      `AstrologersPage.tsx` (`activeOfferPercent`), migration `052` fails on an empty DB,
      `060_kundli_description_rewrites.sql` applied in production but absent from `main`,
      two `058_*` files

## Rollback

| Layer | How | Notes |
|---|---|---|
| ai-core behaviour | flags, restart: `AUTH_MODE=off`, `GROUNDED_CITATIONS=false`, `LLM_HEDGE_AFTER_SECONDS=0`, `CHART_VALIDATION_MODE=log` | never disables chart validation entirely |
| ai-core provider | `LLM_PROVIDER=openrouter` (or `gemini`) | integrations kept intact |
| ai-core code | redeploy `optimaxin/Astro-Chatbot` `main` at `168476f` (the pre-NVIDIA code), or tag `rollback/pre-rag-phase` for the phase-2 state | old code ignores the new env vars |
| Migration 061 | `backend/migrations/rollback/061_down.sql` in a transaction (restores every relabelled row from `ai_traffic_reclassification`) | rehearsed: exact restore |
| Migration 058 | leave in place (old Token Meter ignores the columns); `rollback/058_down.sql` only if required | destructive: drops grounding columns |
| Token Meter UI | revert the `feat/ai-analytics-rc1` merge | |
| Frontend headers | none needed | harmless with any ai-core |
