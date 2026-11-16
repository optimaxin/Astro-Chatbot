# AI Model Playground

A public web app where testers try the Tredev chatbot and raw models side by side, without signing up. Every
generation is logged automatically to SQLite with its tokens, cost and latency. A password-protected admin
dashboard shows analytics and downloads Excel reports.

It runs as **one service**: FastAPI serves the API and the built React app from the same origin. It is
separate from production and writes to its own database.

```
Browser ──► /             React playground (public, no login)
        ──► /admin        React admin dashboard (password)
        ──► /api/playground/*   tester API — rate-limited, daily spend cap, history scoped per browser
        ──► /api/admin/*        admin API — Argon2id login, HttpOnly session cookie, CSRF, audit log
        ──► /api/health
                │
                ├── SQLite (WAL) on the Render persistent disk: /var/data/playground.db
                └── Providers: ai-core /chat · NVIDIA · Gemini · OpenRouter  (keys stay on the server)
```

## What testers can test

| Provider | What it is |
|---|---|
| **Tredev Chatbot (ai-core)** — default | The real chatbot via its `/chat`: intent routing, knowledge base (Pinecone), guardrails, reply validation. One "model" per reply style (Hinglish, English, Hindi, Detailed). It keeps its own conversation memory, so the system prompt and parameters do not apply. A sample chart (12 Apr 1995, 06:30, New Delhi) is used unless the tester pastes birth details as JSON in *Context*: `{"date":"1995-04-12","time":"06:30","timezoneOffsetMinutes":330,"latitude":28.61,"longitude":77.21}` |
| NVIDIA | Nemotron 3.5 Lightning (the chatbot's main model) and Bolt (its fallback), called directly with the tester's own prompt |
| Gemini, OpenRouter | Raw models, for comparison |

Providers and models live in [`backend/config/models.json`](backend/config/models.json). A provider without its
key or URL is shown as "(no API key)" and cannot be used.

## Run locally

```bash
# backend (Python 3.12)
cd playground/backend
python -m venv .venv && .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
# Provide keys as environment variables, or reuse ai-core's:  PLAYGROUND_SHARED_ENV_FILE=../../.env
uvicorn app.main:app --port 8100 --reload

# frontend
cd playground/frontend
npm ci && npm run dev                                   # http://localhost:5180 (proxies /api to :8100)
```

To try the chatbot provider locally, run ai-core on :8000 and set `AI_CORE_URL=http://localhost:8000`.

Tests (52, all providers mocked):

```bash
cd playground/backend && .venv/Scripts/python -m pytest
cd playground/frontend && npm run build                # typecheck + production build
```

The production image is the one Render builds:

```bash
cd playground
docker build -t ai-playground .
docker run -p 10000:10000 -v playground-data:/var/data -e PLAYGROUND_COOKIE_SECURE=false \
  -e ADMIN_PASSWORD_HASH='<hash>' -e AI_CORE_URL=http://host.docker.internal:8000 -e NVIDIA_API_KEY=... ai-playground
```

## Deploy on Render

You need: a Render account with access to this GitHub repo, the **public URL of the deployed ai-core**, and the
provider keys you want testers to use.

1. **Make the admin password hash** (the password itself is never stored):
   ```bash
   cd playground/backend && .venv/Scripts/python -m scripts.hash_password
   ```
   Copy the printed `$argon2id$...` line.

2. **Create the service.** Either:
   - **Blueprint**: Render → *New* → *Blueprint* → pick this repo and give `playground/render.yaml` as the
     blueprint file path. Everything below except the secrets is filled in for you. If the form offers no
     file path option, use the manual setup instead. Or:
   - **By hand**: *New* → *Web Service* → this repo, then:

     | Setting | Value |
     |---|---|
     | Branch | `feat/nvidia-kundli-chatbot` (or whichever branch holds `playground/`) |
     | Language / Runtime | **Docker** |
     | Root Directory | `playground` |
     | Dockerfile Path | `./Dockerfile` |
     | Instance type | **Starter** or above. Free instances cannot have a disk. |
     | Region | The same region as ai-core, for lower latency |
     | Health Check Path | `/api/health` |
     | Build Filters → Included paths | `playground/**` (so ai-core commits do not redeploy the playground) |

   Render builds from the Dockerfile, so there is no separate build or start command. The image runs
   `uvicorn ... --host 0.0.0.0 --port $PORT`.

3. **Add a persistent disk** (*Advanced* → *Add Disk*): mount path **`/var/data`**, size 1 GB. The database is
   created at `/var/data/playground.db` on first start. Anything outside `/var/data` is lost on every deploy.

4. **Set environment variables** (*Environment*):

   | Variable | Required | Value |
   |---|---|---|
   | `ADMIN_PASSWORD_HASH` | yes | the hash from step 1 |
   | `SESSION_SECRET` | yes | a long random string (`python -c "import secrets; print(secrets.token_urlsafe(48))"`) |
   | `NODE_ENV` | yes | `production` (secure cookies, HSTS, no `/docs`) |
   | `PLAYGROUND_DB_PATH` | set by the image | `/var/data/playground.db` |
   | `AI_CORE_URL` | for the chatbot | public URL of ai-core, e.g. `https://<ai-core>.onrender.com` (no trailing slash) |
   | `NVIDIA_API_KEY` | for NVIDIA + the AI evaluator | preferably a key separate from ai-core's, so testers don't use up the chatbot's rate limit |
   | `GEMINI_API_KEY`, `OPENROUTER_API_KEY` | optional | only if you want those providers |
   | `PLAYGROUND_RATE_LIMIT_PER_MINUTE` | optional | per-IP generations per minute (default 20) |
   | `PLAYGROUND_DAILY_REQUEST_LIMIT` | optional | all testers combined, per day (default 1000; 0 = off) |
   | `PLAYGROUND_DAILY_COST_LIMIT_USD` | optional | stops generation for the day at this spend (default 5; 0 = off) |
   | `PLAYGROUND_REPORT_TIMEZONE` | optional | day boundaries for limits and reports (default `Asia/Kolkata`) |
   | `PLAYGROUND_ACCESS_TOKEN` | optional | set it to require a shared code from testers; leave unset for public access |

5. **Deploy**, then check on the live URL:
   - `https://<service>.onrender.com/api/health` → `{"status":"ok"}`
   - `/` opens the playground without a login, and the *Tredev Chatbot* provider answers a question
   - `/admin` asks for the password; a wrong password is refused; the dashboard shows the test you just ran
   - *Experiments → Excel reports → Complete* downloads `AI_Playground_Company_Testing_Logs.xlsx`
   - Trigger *Manual Deploy* once more, and confirm the experiment count is unchanged (the disk works)

6. Share the `/` URL with testers. Keep `/admin` and its password to the admin.

## Admin dashboard

- **Dashboard**: experiments, success and failure counts, tokens, estimated cost (kept separate per currency),
  average latency, anonymous sessions, most-used model. Charts: daily activity, requests by model, usage by
  provider, tokens per day, cost by model, daily spend, and failure reasons. Filter by date range.
- **Experiments**: search and filter by date, provider, model, status, session or text, page through results,
  and open any experiment to see the full prompt, response, edits, AI evaluation, pricing snapshot and the
  exact request sent.
- **Excel reports**: Complete, Date-filtered, Model-wise, Failed requests, Session-wise, or Current filters.
  Each workbook has eight sheets: All Test Logs, Model Summary, Token Analysis, Cost Analysis, Daily Statistics,
  Failed Requests, Feedback & Evaluation, Session Summary. User text is written as plain text, so formula
  injection does not work.
- **Audit log**: logins (including failed and blocked attempts), logouts, viewed experiments, exports, backups.

## Backups and restore

The Render disk survives restarts and deploys, but it is **not a backup**. Use independent copies as well:

- **Download**: *Backup DB* in the admin header saves a consistent snapshot (SQLite online backup API). Store it
  outside Render (e.g. the team's Drive) after each testing round.
- **Render snapshots**: Render takes daily disk snapshots, restorable from the disk's settings page.
- **From a Render shell**: `python -m scripts.backup_db backup /var/data/backups`

Restore:

1. Suspend the service (or stop the container) so nothing writes.
2. Upload the backup file to the disk (Render shell, or restore a Render snapshot instead).
3. Run `python -m scripts.backup_db restore /var/data/<backup>.db`. It checks integrity first and keeps the
   current database as `playground.db.before-restore-<time>`.
4. Resume the service. Older backups are migrated automatically on start.

## How the numbers are produced

- **Tokens** are what the provider reports. If a provider reports none, the counts stay empty, never guessed.
  The prompt-composition bar is a labelled ~4 characters/token estimate.
- **Cost** = tokens × the per-million price in `models.json`. The price used is saved with each experiment
  (`pricing_snapshot`), so later price edits don't rewrite history. A model with no price shows N/A. For the
  chatbot, the cost is ai-core's own figure for the whole turn and is labelled as such.
- **Latency** is measured on the server: total request time, provider call time and playground overhead.
- **AI evaluation** is a judge model's estimate, labelled as such. Testers can also add a good/bad rating, written feedback
  and an edited response. The original response is always kept.

## Known limitations

- Responses are not streamed, so time-to-first-token is not measured (the column exists in Excel, empty).
- The service must run as a single instance with one worker. SQLite, the rate limiter and the ai-core session
  tokens are per-process. That is plenty for 1–10 testers. A Render disk also allows only one instance.
- Chatbot sessions continue across turns only while the service runs. After a redeploy, a tester's next message
  starts a new ai-core conversation.
- ai-core rate-limits per IP (100/min by default). All playground traffic reaches it from one IP, so a
  very busy session could be throttled. Failures are recorded, not hidden.
- ai-core records playground turns in its own analytics as ordinary traffic. Their session ids start with
  `playground-`, so they can be filtered out.
- The daily spend cap counts what has been recorded, so the request in flight when the cap is crossed still
  completes. AI-evaluation calls are not counted towards the cap.
- The tester "name" is optional and unverified. The anonymous browser id only separates histories; it is not
  an identity.
