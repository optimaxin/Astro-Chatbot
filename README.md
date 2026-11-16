# AI Model Playground

A public web app where testers evaluate the real **Tredev Kundli chatbot**, and compare raw AI models, after
entering only their name. Every generation is logged to Supabase Postgres with the tester, session, tokens, cost,
latency and evaluation. A password-protected admin dashboard shows tester-wise analytics and downloads Excel reports.

Everything runs on **Cloudflare Workers**: one Worker serves the React app (from Cloudflare's asset storage) and the
API. The database is the **existing Supabase project**, in its own `playground` schema, reached through **Cloudflare
Hyperdrive**. A Worker cannot connect to Supabase's pooler directly: Workers always verify TLS certificates, and the
pooler's certificate is issued by Supabase's own CA (`TLS handshake failed: invalid peer certificate: UnknownIssuer`).
Hyperdrive handles that connection and pools it.

```
Browser ──► /                 React playground (name screen first)
        ──► /admin            React admin dashboard (password)
        ──► /api/session      tester name → anonymous server-side session (HttpOnly cookie)
        ──► /api/playground/* tester API (needs a session)
        ──► /api/admin/*      admin API (Argon2id login, separate cookie, CSRF, audit log)
        ──► /api/health
                │  Cloudflare Worker (worker/src, Hono + TypeScript)
                ├── Hyperdrive ──► Supabase Postgres, schema `playground`
                ├── Kundli chatbot: POST {AI_CORE_URL}/chat   (the endpoint the website uses)
                └── NVIDIA · Gemini · OpenRouter (keys are Worker secrets, never sent to the browser)
```

## How testing works

1. **Name screen.** Opening the URL shows a welcome screen. *Start testing* stays disabled until the name is valid
   (1–80 characters, trimmed, at least one letter or number, no control characters). No email, phone or password.
2. **Anonymous session.** The server creates a session and sets a browser-session cookie (`HttpOnly`, `SameSite=Lax`,
   no expiry). Refreshing keeps it; closing the browser ends it, so the next visit asks again. Only a hash of the
   cookie token is stored. The tester name and session id used for logging always come from the server's session,
   never from the request.
3. **Testing modes.**
   - **Kundli Chatbot (default).** Messages go to the same chatbot the website uses, with the tester's birth details
     (sample chart prefilled) so the chatbot builds the Kundli itself. Its own system instructions, chart facts,
     knowledge retrieval, validation and conversation memory are untouched; nothing is generated twice. The
     playground records the provider and model the chatbot reports it used, its token usage, cost estimate, latency,
     the chart facts it grounded on, and its citations.
   - **Direct model.** NVIDIA, Google Gemini or OpenRouter, called directly with the tester's own prompt, model and
     supported parameters.
4. **Logged automatically.** Each generation is saved before it runs and finalised afterwards (success *or* failure),
   linked to the tester and session, and the work continues even if the browser disconnects. A retried request id never
   generates twice.

## Repository layout

```
frontend/   React + Vite + TypeScript (tester UI and /admin dashboard)
worker/     Cloudflare Worker API
  src/        routes, providers, evaluator, Excel export, store (all SQL)
  migrations/ SQL for the `playground` schema
  scripts/    migrate.mjs, hash-password.mjs
  tests/      84 tests against a real Postgres
```

## Run locally

Needs Node 22+ and a Postgres to point at (a throwaway Docker one is fine).

```bash
npm run install:all
docker run -d --name pgplay -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=pgtest -p 55432:5432 postgres:16-alpine
DATABASE_URL=postgres://postgres:postgres@localhost:55432/pgtest npm run migrate

# worker/.dev.vars  (gitignored; never commit it)
#   SESSION_SECRET=any-long-random-string
#   ADMIN_PASSWORD_HASH=<output of: npm run hash-password>
#   AI_CORE_URL=http://localhost:8000
#   NVIDIA_API_KEY=...
# Database for `wrangler dev` (local Hyperdrive emulation), as an environment variable in your shell:
export CLOUDFLARE_HYPERDRIVE_LOCAL_CONNECTION_STRING_HYPERDRIVE=postgres://postgres:postgres@localhost:55432/pgtest
#   (against Supabase instead: postgresql://...:5432/postgres?sslmode=require)

npm run build && npm run dev:worker       # http://localhost:8787  (Worker + built frontend)
# or, for frontend hot reload:  npm run dev:worker  and  npm run dev:web  (http://localhost:5180)
```

Tests (providers are mocked; every database test uses real Postgres, so never point `TEST_DATABASE_URL` at a real
database, because the tests truncate the playground tables):

```bash
npm test             # 84 tests, default DB: postgres://postgres:postgres@localhost:55432/pgtest
npm run typecheck
```

## Deploy to Cloudflare

You need: a Cloudflare account, the existing Supabase project's connection string (the same `DATABASE_URL` the Tredev
backend uses), the public URL of the deployed ai-core, and the provider keys you want testers to use.

**Plan note.** Admin login verifies an Argon2id hash, which takes about 0.3 s of CPU, and large Excel exports are CPU
work too. That needs the **Workers Paid** plan (the Free plan allows 10 ms of CPU per request). Free-plan users can
generate a lighter hash with `npm run hash-password -- --pbkdf2`, but heavy exports may still hit the limit.

1. **Create the tables (once).** *Already done for the Tredev Supabase project (2026-11-17).* The migration only
   creates objects inside a new `playground` schema; it does not read or change any existing table. Use a direct or
   session-pooler Supabase connection (DDL does not work through the transaction pooler):
   ```bash
   DATABASE_URL='postgres://...' npm run migrate
   ```
   Or paste `worker/migrations/0001_playground.sql` into the Supabase SQL editor. It is safe to run twice.
2. **Log in to Cloudflare:** `cd worker && npx wrangler login`
3. **Create the Hyperdrive connection to Supabase (once)**, using the Supabase **session pooler** string (port 5432):
   ```bash
   npx wrangler hyperdrive create ai-playground-db --connection-string="postgresql://postgres.<project-ref>:<password>@aws-0-ap-northeast-2.pooler.supabase.com:5432/postgres"
   ```
   Copy the `id` it prints into `worker/wrangler.jsonc`, replacing `REPLACE_WITH_HYPERDRIVE_ID`, and commit that change
   (the id is not a secret; the connection string stays inside Cloudflare).
4. **Make the admin password hash** and keep the password somewhere safe: `npm run hash-password`
5. **Set the secrets** (each command prompts for the value; nothing goes into the repo):
   ```bash
   cd worker
   npx wrangler secret put SESSION_SECRET        # long random string
   npx wrangler secret put ADMIN_PASSWORD_HASH   # from step 4
   npx wrangler secret put AI_CORE_URL           # e.g. https://<ai-core>.onrender.com  (no trailing slash)
   npx wrangler secret put NVIDIA_API_KEY        # also used by the AI evaluator
   npx wrangler secret put GEMINI_API_KEY        # optional
   npx wrangler secret put OPENROUTER_API_KEY    # optional
   ```
6. **Deploy:** `npm run deploy` (builds the frontend, then `wrangler deploy`). Wrangler prints the public
   `https://ai-playground.<your-subdomain>.workers.dev` URL.
7. **Verify on the live URL:** `/api/health` returns `{"status":"ok","database":"ok"}`; `/` shows the name screen; after
   entering a name, a Kundli Chatbot message gets a reply and the token meter shows the underlying provider and model;
   `/admin` accepts the password and shows that test under *Tester-wise results*; *Experiments → Excel reports* downloads
   a workbook with the tester name and session id.

Limits and defaults live in `worker/wrangler.jsonc` (`vars`): 20 generations/minute per connection, 1000 requests/day and
$5/day across all testers. The daily caps stop generation for everyone once reached.

## Admin dashboard (`/admin`)

- **Dashboard:** experiments, success and failure, tokens, estimated cost (per currency), latency, testers and sessions,
  average AI score and ratings, **tester-wise results**, **response-quality averages** (relevance, accuracy,
  human-likeness, tone, conciseness, context awareness, emoji use, language kept, chart consistency), plus charts for
  daily activity, testing mode, models, providers, tokens, spend and failure reasons. Filter by date, tester and mode.
- **Experiments:** filter by tester, session id, provider, model, testing mode, status, date range and text; open any
  experiment for the full prompt, response, edits, evaluation, pricing snapshot and the exact request.
- **Excel reports** (*Complete, Date-filtered, Model-wise, Tester-wise, Failed requests, Session-wise, Current filters*):
  nine sheets: All Test Logs, Model Summary, Token Analysis, Cost Analysis, Daily Statistics, Failed Requests,
  Feedback & Evaluation, Session Summary, **Tester Summary**. Tester names and session ids appear in every sheet where a
  row belongs to a tester or session. User text is written as plain strings, never formulas.
- **Audit log:** logins (including failed and blocked), logouts, viewed experiments and exports.

Admin authentication is separate from tester sessions: its own `SameSite=Strict` cookie, CSRF token on changes, and a
lockout after 5 wrong passwords in 15 minutes.

## How the numbers are produced

- **Tokens** are what the provider (or the chatbot) reports. If none are reported they stay empty and are labelled
  unavailable; they are never guessed.
- **Cost:** direct models = tokens × the per-million price in `worker/src/models.json`, with that price saved on each
  experiment. For the Kundli Chatbot it is the chatbot's own estimate for the reply (it exposes no input/output split).
  The chatbot's whole-turn tokens (classifier and quality-check calls included) are shown separately.
- **Latency** is measured by the Worker. **Time to first token is not available**: the chatbot validates the whole reply
  before sending it, and direct calls are not streamed, so the column stays blank.
- **AI evaluation** is a judge model's estimate, run as a separate call whose tokens are kept apart. Chart consistency is
  checked against the chart facts the chatbot reported plus a house/lord table derived from its lagna.

## Backups and data

The database is your Supabase project, so backups follow its plan (daily backups on paid plans). For an extra copy:
`pg_dump "$DATABASE_URL" --schema=playground -Fc -f playground.dump`. To purge old test data, delete from
`playground.experiments` older than a date, then any `playground.sessions` rows no longer referenced.

## Known limitations

- The AI judge is a small model and can misjudge. Treat scores as a guide, and set a stronger evaluator in
  `worker/src/models.json` if you have one.
- ai-core rate-limits per IP. All playground traffic reaches it from Cloudflare's network, so a very busy session could be
  throttled; such failures are recorded, not hidden.
- ai-core records playground turns in its own analytics as ordinary traffic; their session ids start with `playground-`.
- The tester name is self-declared and unverified.
- A daily cap is checked before each request, so the request in flight when the cap is crossed still completes.
