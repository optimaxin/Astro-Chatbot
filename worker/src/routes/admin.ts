import { Hono, type MiddlewareHandler } from 'hono';
import { deleteCookie, getCookie, setCookie } from 'hono/cookie';
import { z } from 'zod';
import { catalog, findProvider, providerConfigured } from '../catalog';
import { hmacHex, randomToken, safeEqual, sha256Hex, verifyPassword } from '../crypto';
import { buildWorkbook, FILENAME } from '../excel';
import { clientIp, fail, isHttps, parse, readJson, type AppEnv, type Ctx } from '../http';
import { aggregate, describeFilters, FilterError, parseFilters, type RawFilters } from '../reports';
import * as store from '../store';

export const admin = new Hono<AppEnv>();

const COOKIE = 'pg_admin';
const PATH = '/api/admin';
const LOGIN_ATTEMPTS = 5;
const LOGIN_WINDOW_S = 15 * 60;
const XLSX = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';

const csrfFor = (c: Ctx, token: string) => hmacHex(c.env.SESSION_SECRET, 'csrf:' + token);

/** Every admin route except login needs a live admin session cookie. */
const requireAdmin: MiddlewareHandler<AppEnv> = async (c, next) => {
  const token = getCookie(c, COOKIE);
  if (!token || !(await store.adminSessionValid(c.get('sql'), await sha256Hex(token)))) {
    throw fail(401, 'Admin sign-in required.');
  }
  c.set('adminToken' as never, token as never);
  return next();
};

/** State-changing admin calls also need the CSRF token the session was issued with. */
const requireCsrf: MiddlewareHandler<AppEnv> = async (c, next) => {
  const token = c.get('adminToken' as never) as string;
  const sent = c.req.header('X-CSRF-Token');
  if (!sent || !(await safeEqual(sent, await csrfFor(c, token)))) throw fail(403, 'Missing or invalid CSRF token.');
  return next();
};

admin.post('/login', async (c) => {
  const sql = c.get('sql');
  const ip = clientIp(c);
  const { password } = parse(z.object({ password: z.string().min(1).max(256) }), await readJson(c));
  const hash = c.env.ADMIN_PASSWORD_HASH?.trim();
  if (!hash) throw fail(503, 'Admin login is not configured (ADMIN_PASSWORD_HASH is unset).');

  if (await store.countRecentFailedLogins(sql, ip, Date.now() / 1000 - LOGIN_WINDOW_S) >= LOGIN_ATTEMPTS) {
    await store.audit(sql, 'admin.login_blocked', ip);
    throw fail(429, 'Too many login attempts. Wait 15 minutes and try again.');
  }
  const check = await verifyPassword(password, hash);
  if (check === 'invalid-hash') throw fail(503, 'ADMIN_PASSWORD_HASH is not a valid password hash.');
  if (check !== 'ok') {
    await store.audit(sql, 'admin.login_failed', ip);
    throw fail(401, 'Incorrect password.');
  }
  const token = randomToken();
  const hours = c.get('settings').adminSessionHours;
  await store.createAdminSession(sql, await sha256Hex(token), Date.now() / 1000 + hours * 3600);
  setCookie(c, COOKIE, token, {
    httpOnly: true, secure: isHttps(c), sameSite: 'Strict', path: PATH, maxAge: hours * 3600,
  });
  await store.audit(sql, 'admin.login', ip);
  return c.json({ csrf_token: await csrfFor(c, token) });
});

admin.use('*', async (c, next) => (c.req.path.endsWith('/login') ? next() : requireAdmin(c, next)));

admin.post('/logout', requireCsrf, async (c) => {
  const token = getCookie(c, COOKIE);
  if (token) await store.deleteAdminSession(c.get('sql'), await sha256Hex(token));
  deleteCookie(c, COOKIE, { path: PATH, secure: isHttps(c) });
  await store.audit(c.get('sql'), 'admin.logout', clientIp(c));
  return c.json({ ok: true });
});

admin.get('/me', async (c) => {
  const token = c.get('adminToken' as never) as string;
  return c.json({ authenticated: true, csrf_token: await csrfFor(c, token) });
});

function filtersFrom(c: Ctx): { raw: RawFilters; filters: ReturnType<typeof parseFilters> } {
  const q = c.req.query();
  const raw: RawFilters = {
    from: q.from, to: q.to, provider: q.provider, model: q.model, status: q.status, mode: q.mode,
    tester: q.tester, session_id: q.session_id, q: q.q,
  };
  try {
    return { raw, filters: parseFilters(raw, c.get('settings').reportTimezone) };
  } catch (err) {
    if (err instanceof FilterError) throw fail(400, err.message);
    throw err;
  }
}

admin.get('/analytics', async (c) => {
  const { filters } = filtersFrom(c);
  const result = aggregate(await store.allExperiments(c.get('sql'), filters), c.get('settings').reportTimezone);
  const { daily_model: _dm, sessions, ...rest } = result;
  return c.json({ ...rest, sessions: sessions.length });
});

const pageParams = (c: Ctx, max: number, dflt: number) => ({
  page: Math.max(1, Number(c.req.query('page') ?? 1) || 1),
  pageSize: Math.min(max, Math.max(1, Number(c.req.query('page_size') ?? dflt) || dflt)),
});

admin.get('/experiments', async (c) => {
  const { filters } = filtersFrom(c);
  const { page, pageSize } = pageParams(c, 200, 25);
  const { items, total } = await store.searchExperiments(c.get('sql'), filters, pageSize, (page - 1) * pageSize);
  return c.json({ items, total, page, page_size: pageSize });
});

admin.get('/experiments/:id', async (c) => {
  const id = Number(c.req.param('id'));
  const exp = Number.isInteger(id) ? await store.getExperiment(c.get('sql'), id) : null;
  if (!exp) throw fail(404, 'Experiment not found.');
  await store.audit(c.get('sql'), 'admin.view_experiment', clientIp(c), { experiment_id: id });
  return c.json(exp);
});

admin.get('/models', async (c) => c.json({
  catalog: catalog.models.map((m) => ({
    provider: m.provider, model: m.id, label: m.label,
    configured: providerConfigured(c.env, findProvider(m.provider)!), pricing: m.pricing,
  })),
  seen: await store.distinctValues(c.get('sql')),
}));

admin.get('/audit-logs', async (c) => {
  const { page, pageSize } = pageParams(c, 200, 50);
  const { items, total } = await store.auditLogs(c.get('sql'), pageSize, (page - 1) * pageSize);
  return c.json({ items, total, page, page_size: pageSize });
});

admin.get('/export/excel', async (c) => {
  const { raw, filters } = filtersFrom(c);
  const rows = await store.allExperiments(c.get('sql'), filters);
  const data = await buildWorkbook(rows, describeFilters(raw), c.get('settings').reportTimezone);
  await store.audit(c.get('sql'), 'admin.export_excel', clientIp(c), {
    filters: Object.fromEntries(Object.entries(raw).filter(([, v]) => v)), rows: rows.length,
  });
  return new Response(data, { headers: { 'Content-Type': XLSX, 'Content-Disposition': `attachment; filename="${FILENAME}"` } });
});
