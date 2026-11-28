# AI Model Playground

A public web app where testers evaluate the real **Tredev Kundli chatbot**, and compare raw AI models, after
entering only their name. Every generation is logged to Supabase Postgres with the tester, session, tokens, cost,
latency and evaluation. A password-protected admin dashboard shows tester-wise analytics and downloads Excel reports.

It runs as **one Render Free web service** (Node.js + Express) that serves the built React app and the `/api/*`
routes from the same address. Data lives in the **existing Supabase project**, in its own `playground` schema.
Infrastructure cost: **$0/month**. Nothing here needs a paid Render plan, a paid database, or any add-on.

```
Browser ──► /                 React playground (name screen first)
        ──► /admin            React admin dashboard (password)
        ──► /healthz          liveness for Render (never touches the database)
        ──► /api/health       deep check: is Supabase reachable?
        ──► /api/session      tester name → anonymous server-side session (HttpOnly cookie)
        ──► /api/playground/* tester API (needs a session)
        ──► /api/admin/*      admin API (Argon2id login, separate cookie, CSRF, audit log)
                │  Render Free web service: Express (worker/src/node) hosting the Hono API (worker/src)
                ├── pool of ≤5 connections ──► Supabase Postgres (session pooler), schema `playground`
                ├── Kundli chatbot: POST {AI_CORE_URL}/chat   (the endpoint the website uses)
                ├── Reference Kundli: POST {TREDEV_BACKEND_URL}/api/calculators/kundli-full (same calculator ai-core itself uses)
                ├── Birthplace search: Amazon Location Service Places API (Suggest + GetPlace, API-key auth)
                └── NVIDIA · Gemini · OpenRouter (keys are server environment variables, never sent to the browser)
```

The API is a [Hono](https://hono.dev) application. Express hosts it on Node, serves `frontend/dist`, answers the health
check, caps request bodies, and sets security headers. The API code is runtime-neutral, so the previous Cloudflare Worker
entry point (`worker/src/index.ts`, `worker/wrangler.jsonc`) still works and is kept until the Render deployment has
been verified (see *Retiring the Cloudflare setup*).

## How testing works

1. **Name screen.** Opening the URL shows a welcome screen. *Start testing* stays disabled until the name is valid
   (1–80 characters, trimmed, at least one letter or number, no control characters). No email, phone or password.
2. **Anonymous session.** The server creates a session and sets a browser-session cookie (`HttpOnly`, `Secure`,
   `SameSite=Lax`, no expiry). Refreshing keeps it; closing the browser ends it, so the next visit asks again. Only a
   hash of the cookie token is stored. The tester name and session id used for logging always come from the server's
   session, never from the request. Sessions also expire server-side after 12 h idle / 24 h total.
3. **Kundli Chatbot testing.** Messages go to the same chatbot the website uses, with the tester's birth details
   (sample chart prefilled, or searched via Amazon Location Service when configured). Its own instructions, chart
   facts, knowledge retrieval, validation and memory are untouched; nothing is generated twice. The playground
   records the provider and model the chatbot reports, its token usage, cost estimate, latency, the chart facts it
   grounded on, and its citations. Direct-model testing (NVIDIA, Gemini, OpenRouter called directly) still works
   server-side but is not shown in the UI.
4. **Reference Kundli + response verification.** A tester can generate a reference chart for their birth details from
   the real Tredev calculator (`POST {TREDEV_BACKEND_URL}/api/calculators/kundli-full` — the same one ai-core itself
   calls; no second astrology engine). Each Kundli Chatbot reply is then checked against that chart: an LLM extracts
   the factual chart claims the reply makes (ascendant, planet placements, nakshatra, current dasha), and this app's
   own deterministic code — never the LLM — decides whether each one is SUPPORTED, CONTRADICTED, UNVERIFIABLE or not
   a chart fact at all, with a chart-fact consistency percentage (never shown as proof the whole reply is correct).
5. **Logged automatically.** Each generation is saved before it runs and finalised afterwards (success *or* failure),
   linked to the tester and session. The work continues and is logged even if the browser disconnects, and a retried
   request id never generates twice.

## Repository layout

```
render.yaml   Render Blueprint (free web service)
frontend/     React + Vite + TypeScript (tester UI and /admin dashboard)
worker/       The API (the folder keeps its old name because the Cloudflare config lives here too)
  src/          routes, providers, evaluator, Excel export, store (all SQL)
  src/node/     Express server (main.ts = entry point, server.ts = app)
  migrations/   SQL for the `playground` schema
  scripts/      migrate.mjs, hash-password.mjs, smoke.mjs
  tests/        130 tests against a real Postgres
```

## Deploy to Render Free

You need: a Render account (free), the existing Supabase project's connection string, the public URL of the deployed
ai-core, and the provider keys you want testers to use.

1. **Check the database (once).** *Already done for the Tredev Supabase project.* The migration only creates objects
   inside the `playground` schema and never reads or changes any other table. It is safe to run again:
   `DATABASE_URL='postgresql://…:5432/postgres' npm run migrate` prints `skip 0001_playground.sql` when up to date.
2. **Make the admin password hash:** `npm run install:all`, then `npm run hash-password`. Keep the password somewhere
   safe; keep the printed `$argon2id$…` line for step 4.
3. **Create the service.** In Render: *New → Blueprint*, pick the GitHub repo and the **`playground`** branch. Render reads
   [render.yaml](render.yaml) (plan `free`, Node 24, health check `/healthz`).
4. **Fill in the variables** Render asks for (the checklist below). `SESSION_SECRET` is generated for you.
5. **Deploy.** The first build takes a few minutes. Render prints `https://ai-playground-xxxx.onrender.com`.
6. **Verify:** `npm run smoke -- https://<your-service>.onrender.com` (add `ADMIN_PASSWORD='…'` to include admin login).
   It checks the health endpoints, the frontend, security headers, the tester session and cookie, the model catalogue
   and the admin login, without spending any model tokens. Then try a Kundli Chatbot message and the Excel export by hand.

Later changes: push to the `playground` branch and Render redeploys. Rollback is one click in Render's *Events* tab.

### Environment variables

| Variable | Required | Value |
|---|---|---|
| `DATABASE_URL` | yes | Supabase **session pooler** string, port 5432: `postgresql://postgres.<ref>:<password>@aws-0-ap-northeast-2.pooler.supabase.com:5432/postgres`. The transaction pooler (6543) also works. Do **not** use the direct `db.<ref>.supabase.co` host: on the free plan it is IPv6-only. |
| `SESSION_SECRET` | yes | 16+ random characters (Render generates it). Changing it signs admin out and resets client-address hashes. |
| `ADMIN_PASSWORD_HASH` | for `/admin` | Output of `npm run hash-password` (`$argon2id$…`). Unset = admin login disabled. |
| `AI_CORE_URL` | for Kundli mode | Public ai-core URL, no trailing slash. Unset = that mode is shown as unavailable. |
| `TREDEV_BACKEND_URL` | for Reference Kundli | Public Tredev backend URL, no trailing slash. Calls its existing, public `POST /api/calculators/kundli-full` — the same calculator ai-core itself uses, never a second engine. Unset = Reference Kundli generation returns 503. |
| `NVIDIA_API_KEY` | recommended | NVIDIA models, the AI evaluator, **and** the claim-extraction step of response verification. |
| `GEMINI_API_KEY`, `OPENROUTER_API_KEY` | optional | Unset = that provider's models are hidden. |
| `AWS_LOCATION_API_KEY` | optional | Amazon Location Service Places API key (API-key auth, no SigV4/IAM). Unset or `AWS_LOCATION_ENABLED` not `true` = the birthplace search UI is hidden; testers fall back to manual latitude/longitude/UTC-offset entry. |
| `AWS_LOCATION_REGION` | optional | Default `ap-south-1`. |
| `AWS_LOCATION_ENABLED` | optional | Must be `"true"` to turn on birthplace search. |
| `NODE_VERSION` | set in render.yaml | `24`. Node 24 has native Argon2: admin login took ~1.3 s on a 0.1-vCPU container, against ~5 s for the JavaScript fallback on a fast PC. On Node < 24.7 the server logs a warning at startup. |
| `PLAYGROUND_RATE_LIMIT_PER_MINUTE` | optional | Generations per minute per visitor address (default 20). |
| `PLAYGROUND_DAILY_REQUEST_LIMIT` / `PLAYGROUND_DAILY_COST_LIMIT_USD` | optional | Daily caps across all testers (default 1000 requests, $5). `0` turns a cap off. When reached, generation stops for everyone until the next day. |
| `PLAYGROUND_REPORT_TIMEZONE` | optional | Time zone for daily statistics (default `Asia/Kolkata`). |
| `DB_POOL_MAX` | optional | Connections kept to Supabase (default 5, max 20). |
| `PORT` | automatic | Render sets it; the server binds `0.0.0.0:$PORT`. |

The server refuses to start if `DATABASE_URL` or `SESSION_SECRET` is missing or malformed, and says which one in the deploy
log. Missing optional keys only log a warning.

## Free-tier behaviour

- **Render Free sleeps.** After ~15 minutes without traffic the service stops; the next visit takes up to about a minute
  to start. The page handles this: slow requests show a *"Waking up the server…"* notice and are retried automatically
  (a retried chat is safe: its request id makes it idempotent). `/healthz` never waits on the database, so a slow Supabase
  does not make Render restart the service.
- **No local storage is used.** Everything persistent is in Supabase; Excel reports are built in memory and streamed.
  Anything in flight during a restart is marked failed the next time it is looked at (after 150 s), not lost silently.
- **Supabase Free pauses** a project after about a week without activity. While paused, the site stays up, `/api/health`
  reports `degraded`, and API calls answer a retryable 503; restore the project in the Supabase dashboard.
- **Database blips** (a restart, a dropped connection, too many connections) are retried once on a fresh connection;
  if the database is still unreachable the tester sees "temporarily unreachable, try again" instead of an error.
- **Limits to keep in mind** (check the current Render and Supabase pricing pages, they change): Render Free is 512 MB RAM and
  a small CPU share, with a monthly instance-hour allowance; Supabase Free is 500 MB of database and a small connection
  budget (the pool is capped at 5 for that reason). One running instance is fine; do not scale it out.
- **AI providers are not guaranteed free or always up.** Their errors, rate limits and quota failures are recorded on the
  experiment (status `failed` with the reason), never hidden.
- To avoid sleeping you would need a paid plan or an external keep-alive pinger; neither is configured here.

## Security

- Provider keys, `DATABASE_URL`, `SESSION_SECRET` and the admin hash exist only as Render environment variables; the browser
  never sees them and logs print error types and database error codes, not messages.
- Admin: Argon2id hash, `SameSite=Strict` `HttpOnly` `Secure` cookie scoped to `/api/admin`, 12 h server-side expiry, logout,
  CSRF token on every change, an audit log, and login throttling: 5 failures / 15 min per address **and** 25 / 15 min across
  all addresses (so a guesser who rotates addresses is stopped too). The right password is refused while blocked.
- Tester sessions: random cookie token, only its hash stored, server-side idle and absolute expiry, cross-origin writes refused
  (the origin is checked against the public https host Render forwards).
- Abuse control: per-address rate limit, daily request and spend caps, request bodies over 2 MB or without a length refused
  before being read, prompt length cap, security headers and a strict content security policy on pages and API.
- The client address comes from the proxy headers. If those were forged the per-address limits could be evaded, but the
  daily caps and the global admin-login ceiling do not depend on them.

## Run locally

Needs Node 22+ (24 recommended) and a Postgres to point at (a throwaway Docker one is fine).

```bash
npm run install:all
docker run -d --name pgplay -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=pgtest -p 55432:5432 postgres:16-alpine
export DATABASE_URL='postgres://postgres:postgres@localhost:55432/pgtest'
npm run migrate

# worker/.env (gitignored; never commit it) is read by `npm run dev:node`:
#   DATABASE_URL=...  SESSION_SECRET=any-long-random-string  ADMIN_PASSWORD_HASH=<npm run hash-password>
#   AI_CORE_URL=http://localhost:8000  NVIDIA_API_KEY=...
npm run build && npm start         # http://localhost:10000 (server + built frontend)
# or, for frontend hot reload: npm run dev:node  and  npm run dev:web  (http://localhost:5180)
```

Tests (providers are mocked; every database test uses real Postgres, so never point `TEST_DATABASE_URL` at a real database,
because the tests truncate the playground tables):

```bash
npm test             # 103 tests, default DB: postgres://postgres:postgres@localhost:55432/pgtest
npm run typecheck
```

They include the production Express server over real HTTP sockets: static frontend, `/healthz`, proxy headers, body limits,
work that outlives a closed browser tab, reconnecting after Supabase drops connections, a database outage, and 10 testers
at once with the Excel export checked.

## Admin dashboard (`/admin`)

- **Dashboard:** experiments, success and failure, tokens, estimated cost (per currency), latency, testers and sessions,
  average AI score and ratings, **tester-wise results**, **response-quality averages**, plus charts for daily activity,
  testing mode, models, providers, tokens, spend and failure reasons. Filter by date, tester and mode.
- **Experiments:** filter by tester, session id, provider, model, testing mode, status, date range and text; open any
  experiment for the full prompt, response, edits, evaluation, pricing snapshot and the exact request.
- **Excel reports** (*Complete, Date-filtered, Model-wise, Tester-wise, Failed requests, Session-wise, Current filters*):
  nine sheets: All Test Logs, Model Summary, Token Analysis, Cost Analysis, Daily Statistics, Failed Requests,
  Feedback & Evaluation, Session Summary, **Tester Summary**. Tester names and session ids appear in every sheet where a
  row belongs to a tester or session. User text is written as plain strings, never formulas.
- **Audit log:** logins (including failed and blocked), logouts, viewed experiments and exports.

## How the numbers are produced

- **Tokens** are what the provider (or the chatbot) reports. If none are reported they stay empty and are labelled
  unavailable; they are never guessed.
- **Cost:** direct models = tokens × the per-million price in `worker/src/models.json`, with that price saved on each
  experiment. For the Kundli Chatbot it is the chatbot's own estimate for the reply.
- **Latency** is measured by the server. **Time to first token is not available**: the chatbot validates the whole reply
  before sending it, and direct calls are not streamed, so the column stays blank rather than showing an invented number.
- **AI evaluation** is a judge model's estimate, run as a separate call whose tokens are kept apart. Chart consistency is
  checked against the chart facts the chatbot reported plus a house/lord table derived from its lagna.

## Backups and data

The database is your Supabase project, so backups follow its plan (the free plan has no point-in-time backups). For an
extra copy: `pg_dump "$DATABASE_URL" --schema=playground -Fc -f playground.dump`. To purge old test data, delete from
`playground.experiments` older than a date, then any `playground.sessions` rows no longer referenced.

## Retiring the Cloudflare setup

`worker/src/index.ts`, `worker/wrangler.jsonc` and the `wrangler` dev dependency are only for the earlier Cloudflare
deployment (`npm run deploy:cloudflare`). After the Render service passes `npm run smoke` and a manual test, delete those
three, the `hyperdrive`/`ASSETS` handling in `src/db.ts` and `src/app.ts`, and `frontend/public/_headers`. Nothing on Render
depends on them. If a Hyperdrive config or Worker was ever created in Cloudflare, delete it there too.

## Known limitations

- The AI judge is a small model and can misjudge. Treat scores as a guide, and set a stronger evaluator in
  `worker/src/models.json` if you have one.
- ai-core rate-limits per IP. All playground traffic reaches it from Render's addresses, so a very busy session could be
  throttled; such failures are recorded, not hidden.
- ai-core records playground turns in its own analytics as ordinary traffic; their session ids start with `playground-`.
- The tester name is self-declared and unverified, and is **required** (the admin reports are organised by it).
- A daily cap is checked before each request, so the request in flight when the cap is crossed still completes.
