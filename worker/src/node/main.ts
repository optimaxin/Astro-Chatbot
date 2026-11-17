import { createServer } from 'node:http';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { argon2Backend } from '../crypto';
import { createPool } from '../db';
import type { Env } from '../env';
import { buildServer, tuneForProxy } from './server';

const here = dirname(fileURLToPath(import.meta.url));

// Configuration that makes the service unusable: stop right away so the deploy log says why.
const required = ['DATABASE_URL', 'SESSION_SECRET'] as const;
// Features that quietly switch off without these: warn, but keep serving.
const optional: Record<string, string> = {
  ADMIN_PASSWORD_HASH: 'admin login is disabled',
  AI_CORE_URL: 'Kundli chatbot mode is unavailable',
  NVIDIA_API_KEY: 'NVIDIA models and the AI evaluator are unavailable',
  GEMINI_API_KEY: 'Gemini models are unavailable',
  OPENROUTER_API_KEY: 'OpenRouter models are unavailable',
};

const env = { ...process.env } as unknown as Env;
const missing = required.filter((k) => !process.env[k]?.trim());
if (missing.length) {
  console.error(`Missing required environment variables: ${missing.join(', ')}. See the README for the checklist.`);
  process.exit(1);
}
if ((env.SESSION_SECRET ?? '').length < 16) {
  console.error('SESSION_SECRET must be at least 16 characters (use a long random string).');
  process.exit(1);
}
for (const [key, effect] of Object.entries(optional)) {
  if (!process.env[key]?.trim()) console.warn(`${key} is not set: ${effect}.`);
}

// Lazy: no connection is opened until the first query, so the server is up in milliseconds
// even when the database (Supabase free) is still waking up.
try {
  env.SQL_POOL = createPool(env);
} catch {
  // Never print the value: it contains the database password.
  console.error('DATABASE_URL is not a valid connection string. Expected postgresql://USER:PASSWORD@HOST:5432/postgres (special characters in the password must be percent-encoded).');
  process.exit(1);
}

if (argon2Backend() === 'javascript') {
  console.warn(`Node ${process.versions.node} has no native Argon2 (needs 24.7+): admin login falls back to a JavaScript implementation that takes seconds on a free instance. Set NODE_VERSION=24.`);
}

const staticDir = process.env.STATIC_DIR ?? resolve(here, '../../frontend/dist');
const built = buildServer(env, { staticDir });
const server = createServer(built.app);
tuneForProxy(server);

const port = Number(process.env.PORT) || 10000;
server.listen(port, '0.0.0.0', () => {
  console.log(`AI Playground listening on 0.0.0.0:${port}`);
  // Informational only: shows in the log whether Supabase answered during this cold start.
  const t0 = Date.now();
  env.SQL_POOL!`select 1`
    .then(() => console.log(`Database reachable (${Date.now() - t0} ms).`))
    .catch((err: unknown) => console.warn(
      `Database not reachable yet (${err instanceof Error ? err.name : 'error'} ${(err as { code?: string }).code ?? ''}); it will be retried on each request.`));
});

let stopping = false;
async function shutdown(signal: string) {
  if (stopping) return;
  stopping = true;
  console.log(`${signal} received: finishing ${built.pending()} background job(s) and closing.`);
  server.close();
  server.closeIdleConnections();
  await built.drain();
  process.exit(0);
}
process.on('SIGTERM', () => void shutdown('SIGTERM'));
process.on('SIGINT', () => void shutdown('SIGINT'));
// Log the error type only: messages can carry connection strings or provider details.
process.on('unhandledRejection', (reason) => console.error('unhandled rejection:', reason instanceof Error ? reason.name : 'unknown'));
process.on('uncaughtException', (err) => {
  console.error('uncaught exception:', err.name);
  process.exit(1);
});
