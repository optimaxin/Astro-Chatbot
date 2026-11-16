import { Hono } from 'hono';
import { HTTPException } from 'hono/http-exception';
import { connect } from './db';
import { loadSettings, type Env } from './env';
import { fail, type AppEnv } from './http';
import { admin } from './routes/admin';
import { playground, session } from './routes/tester';
import * as store from './store';

export const app = new Hono<AppEnv>();

// Scripts and styles only from this origin (plus Google Fonts for typography).
const CSP = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; " +
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
  if (new URL(c.req.url).protocol === 'https:') h.set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
});

app.use('/api/*', async (c, next) => {
  if (!c.env.SESSION_SECRET || c.env.SESSION_SECRET.length < 16) {
    throw fail(503, 'SESSION_SECRET is not configured (use at least 16 random characters).');
  }
  // Cross-site writes are refused outright: the browser always sends Origin on POST/DELETE.
  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(c.req.method)) {
    const origin = c.req.header('Origin');
    if (origin && origin !== new URL(c.req.url).origin) throw fail(403, 'Cross-origin request refused.');
  }
  const sql = connect(c.env);
  c.set('sql', sql);
  try {
    await next();
  } finally {
    c.executionCtx.waitUntil(sql.end({ timeout: 5 }).catch(() => undefined));
  }
  c.res.headers.set('Cache-Control', 'no-store');
});

app.get('/api/health', async (c) => {
  try {
    await store.ping(c.get('sql'));
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

app.onError((err, c) => {
  if (err instanceof HTTPException) return c.json({ detail: err.message }, err.status);
  // Log the error type only: messages can carry connection strings or provider details.
  console.error('unhandled error:', err instanceof Error ? err.name : 'unknown');
  return c.json({ detail: 'Internal error.' }, 500);
});

// Anything that is not an API route is the built frontend (single-page app).
app.all('*', (c) => c.env.ASSETS.fetch(c.req.raw));

export type { Env };
