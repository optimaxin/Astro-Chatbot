import { Hono } from 'hono';
import { HTTPException } from 'hono/http-exception';
import { connect } from './db';
import { loadSettings, type Env } from './env';
import { fail, isHttps, requestOrigin, type AppEnv } from './http';
import { admin } from './routes/admin';
import { playground, session } from './routes/tester';
import * as store from './store';

export const app = new Hono<AppEnv>();

// Scripts and styles only from this origin (plus Google Fonts for typography).
export const CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; " +
  "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; " +
  "frame-ancestors 'none'; base-uri 'self'; form-action 'self'";

app.use('*', async (c, next) => {
  c.set('settings', loadSettings(c.env));
  await next();
  const h = c.res.headers;
  h.set('X-Content-Type-Options', 'nosniff');
  h.set('X-Frame-Options', 'DENY');
  h.set('Referrer-Policy', 'no-referrer');
  h.set('Content-Security-Policy', CSP);
  if (isHttps(c)) h.set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
});

app.use('/api/*', async (c, next) => {
  if (!c.env.SESSION_SECRET || c.env.SESSION_SECRET.length < 16) {
    throw fail(503, 'SESSION_SECRET is not configured (use at least 16 random characters).');
  }
  // Cross-site writes are refused outright: the browser always sends Origin on POST/DELETE.
  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(c.req.method)) {
    const origin = c.req.header('Origin');
    if (origin && origin !== requestOrigin(c)) throw fail(403, 'Cross-origin request refused.');
  }
  // Node keeps one shared pool; a Worker opens one short-lived connection per request.
  const pooled = c.env.SQL_POOL;
  const sql = pooled ?? connect(c.env);
  c.set('sql', sql);
  try {
    await next();
  } finally {
    if (!pooled) c.executionCtx.waitUntil(sql.end({ timeout: 5 }).catch(() => undefined));
  }
  c.res.headers.set('Cache-Control', 'no-store');
});

app.get('/api/health', async (c) => {
  try {
    try {
      await store.ping(c.get('sql'));
    } catch (err) {
      if (!isStaleConnection(err)) throw err;
      await new Promise((r) => setTimeout(r, 150)); // the pool discards the dead connection; ask again
      await store.ping(c.get('sql'));
    }
    return c.json({ status: 'ok', database: 'ok' });
  } catch {
    return c.json({ status: 'degraded', database: 'unreachable' }, 503);
  }
});

app.route('/api/session', session);
app.route('/api/playground', playground);
app.route('/api/admin', admin);
app.all('/api/*', () => {
  throw fail(404, 'Not found.');
});

// Errors that mean "the database could not be reached right now" (Supabase waking, restarting, out of
// connections, network blip). They are temporary, so the browser gets a retryable 503, not a generic 500.
const DB_UNAVAILABLE = new Set([
  'ECONNREFUSED', 'ECONNRESET', 'ETIMEDOUT', 'ENOTFOUND', 'EAI_AGAIN', 'EPIPE', 'CONNECT_TIMEOUT', 'CONNECTION_CLOSED',
  'CONNECTION_DESTROYED', 'CONNECTION_ENDED', 'CONNECTION_CLOSED', '57P01', '57P02', '57P03', '53300', '53400', '08000', '08003', '08006',
]);
const dbCode = (err: unknown) => {
  const code = (err as { code?: unknown } | null)?.code;
  return typeof code === 'string' ? code : '';
};
const isDbUnavailable = (err: unknown) => DB_UNAVAILABLE.has(dbCode(err));
// The pool handed out a connection the server had already closed (restart, failover, idle kill). A second
// attempt gets a fresh connection, so these are safe to retry; a refused or timed-out connect is not.
const STALE = new Set(['CONNECTION_CLOSED', 'CONNECTION_DESTROYED', 'CONNECTION_ENDED', 'ECONNRESET', 'EPIPE', '57P01', '57P02', '08003', '08006']);
const isStaleConnection = (err: unknown) => STALE.has(dbCode(err));
/** Set on a response that failed only because of a stale pooled connection; the Node server retries it once. */
export const RETRY_HEADER = 'X-Internal-Retry';

app.onError((err, c) => {
  if (err instanceof HTTPException) return c.json({ detail: err.message }, err.status);
  if (isDbUnavailable(err)) {
    console.error('database unavailable:', (err as unknown as { code: string }).code);
    c.header('Retry-After', '5');
    if (isStaleConnection(err)) c.header(RETRY_HEADER, '1');
    return c.json({ detail: 'The database is temporarily unreachable. Please try again in a few seconds.' }, 503);
  }
  // Log the error type only: messages can carry connection strings or provider details.
  // Type and database error code only (e.g. 28P01 = wrong password): messages can carry connection strings.
  console.error('unhandled error:', err instanceof Error ? err.name : 'unknown', dbCode(err));
  return c.json({ detail: 'Internal error.' }, 500);
});

// Anything that is not an API route is the built frontend (single-page app).
app.all('*', (c) => (c.env.ASSETS ? c.env.ASSETS.fetch(c.req.raw) : c.notFound()));

export type { Env };
