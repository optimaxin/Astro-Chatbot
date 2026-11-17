// The production Node server (Express hosting the Hono API + built frontend), exercised over real HTTP
// sockets against a real Postgres, exactly as Render runs it. Providers are mocked; nothing else is.
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { request as httpRequest, createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import postgres from 'postgres';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { createPool } from '../src/db';
import { buildServer, tuneForProxy, type BuiltServer } from '../src/node/server';
import { ADMIN_PASSWORD, DB_URL, chatbotReply, makeEnv, mockUpstream, openAiReply, resetDb, sql } from './helpers';

let dir: string;
let server: Server;
let built: BuiltServer;
let port: number;
let pool: ReturnType<typeof createPool>;

const HOST = 'playground.example.onrender.com';
const SECURE = { Host: HOST, Origin: `https://${HOST}`, 'X-Forwarded-Proto': 'https' };

interface Res { status: number; headers: Record<string, string | string[] | undefined>; text: string; json: any; buf?: Buffer }

function send(method: string, path: string, opts: {
  headers?: Record<string, string>; body?: unknown; raw?: string; chunked?: boolean; abortAfterMs?: number;
} = {}): Promise<Res> {
  return new Promise((resolve, reject) => {
    const payload = opts.raw ?? (opts.body !== undefined ? JSON.stringify(opts.body) : undefined);
    const headers: Record<string, string> = { ...opts.headers };
    if (payload !== undefined) headers['Content-Type'] = 'application/json';
    if (payload !== undefined && !opts.chunked) headers['Content-Length'] = String(Buffer.byteLength(payload));
    const req = httpRequest({ host: '127.0.0.1', port, method, path, headers }, (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (c: Buffer) => chunks.push(c));
      res.on('end', () => {
        const buf = Buffer.concat(chunks);
        const text = buf.toString('utf8');
        let json: any = null;
        try { json = JSON.parse(text); } catch { /* not JSON */ }
        resolve({ status: res.statusCode!, headers: res.headers, text, json, buf });
      });
    });
    req.on('error', reject);
    if (opts.abortAfterMs !== undefined) setTimeout(() => { req.destroy(); resolve({ status: 0, headers: {}, text: '', json: null }); }, opts.abortAfterMs);
    if (payload !== undefined) req.write(payload);
    req.end();
  });
}

/** A tester's browser: remembers cookies, shows up from its own address. */
function visitor(ip: string) {
  const jar = new Map<string, string>();
  const call = async (method: string, path: string, o: Parameters<typeof send>[2] = {}) => {
    const headers: Record<string, string> = { ...SECURE, 'CF-Connecting-IP': ip, ...o.headers };
    if (jar.size) headers.Cookie = [...jar].map(([k, v]) => `${k}=${v}`).join('; ');
    const r = await send(method, path, { ...o, headers });
    for (const sc of [r.headers['set-cookie'] ?? []].flat()) {
      const [name, value = ''] = sc!.split(';')[0]!.split('=');
      if (value === '' || /max-age=0/i.test(sc!)) jar.delete(name!); else jar.set(name!, value);
    }
    return r;
  };
  return {
    jar, call,
    start: (name: string) => call('POST', '/api/session', { body: { name } }),
    kundli: (message: string) => call('POST', '/api/playground/chat', { body: {
      mode: 'kundli_chatbot', message, request_id: `r-${Math.random().toString(36).slice(2, 12)}`, conversation_id: 'conv-0000000001' } }),
    direct: (message: string) => call('POST', '/api/playground/chat', { body: {
      mode: 'direct_provider', provider: 'gemini', model: 'gemini-2.5-flash', message,
      request_id: `r-${Math.random().toString(36).slice(2, 12)}`, conversation_id: 'conv-0000000001' } }),
  };
}

const isChatbot = (u: string) => u === 'http://ai-core.test/chat';

beforeAll(async () => {
  dir = mkdtempSync(join(tmpdir(), 'pg-static-'));
  mkdirSync(join(dir, 'assets'));
  writeFileSync(join(dir, 'index.html'), '<!doctype html><div id="root"></div>');
  writeFileSync(join(dir, 'assets', 'app-abc123.js'), 'console.log(1)');
  writeFileSync(join(dir, 'favicon.svg'), '<svg xmlns="http://www.w3.org/2000/svg"/>');
  const env = await makeEnv();
  delete env.ASSETS; // Node has no assets binding: Express serves the frontend
  pool = createPool({ ...env, DB_POOL_MAX: '5' });
  env.SQL_POOL = pool;
  built = buildServer(env, { staticDir: dir });
  server = createServer(built.app);
  tuneForProxy(server);
  await new Promise<void>((r) => server.listen(0, '0.0.0.0', r));
  port = (server.address() as AddressInfo).port;
});
afterAll(async () => {
  await new Promise((r) => server.close(r));
  await built.drain(2000);
  rmSync(dir, { recursive: true, force: true });
});
beforeEach(resetDb);
afterEach(() => vi.unstubAllGlobals());

describe('health and static frontend', () => {
  it('answers the platform health check without the database', async () => {
    const r = await send('GET', '/healthz');
    expect(r.status).toBe(200);
    expect(r.json).toEqual({ status: 'ok' });
  });

  it('/api/health checks the database', async () => {
    const r = await send('GET', '/api/health');
    expect(r.status).toBe(200);
    expect(r.json).toEqual({ status: 'ok', database: 'ok' });
  });

  it('serves the app shell for / and client-side routes like /admin, with security headers', async () => {
    for (const path of ['/', '/admin', '/admin/experiments']) {
      const r = await send('GET', path, { headers: { 'X-Forwarded-Proto': 'https' } });
      expect(r.status).toBe(200);
      expect(r.text).toContain('id="root"');
      expect(r.headers['cache-control']).toBe('no-cache');
      expect(r.headers['content-security-policy']).toContain("default-src 'self'");
      expect(r.headers['x-frame-options']).toBe('DENY');
      expect(r.headers['strict-transport-security']).toContain('max-age=');
    }
  });

  it('caches fingerprinted assets forever, returns 404 for a missing file and never the shell for it', async () => {
    const a = await send('GET', '/assets/app-abc123.js');
    expect(a.status).toBe(200);
    expect(a.headers['cache-control']).toBe('public, max-age=31536000, immutable');
    expect(a.headers['x-content-type-options']).toBe('nosniff');
    expect((await send('GET', '/assets/missing.js')).status).toBe(404);
    expect((await send('GET', '/favicon.svg')).headers['cache-control']).toBe('no-cache');
  });

  it('unknown API routes are JSON 404s, not the app shell', async () => {
    const r = await send('GET', '/api/nope');
    expect(r.status).toBe(404);
    expect(r.json.detail).toBe('Not found.');
    expect(r.headers['cache-control']).toBe('no-store');
  });

  it('does not reveal the framework', async () => {
    expect((await send('GET', '/healthz')).headers['x-powered-by']).toBeUndefined();
  });
});

describe('behind the platform proxy (https, host, client address)', () => {
  it('accepts the browser origin that matches the public https host and marks cookies Secure', async () => {
    const v = visitor('203.0.113.7');
    const r = await v.start('Asha');
    expect(r.status).toBe(201);
    const cookie = [r.headers['set-cookie']].flat().join(';');
    expect(cookie).toMatch(/pg_session=/);
    expect(cookie).toMatch(/HttpOnly/i);
    expect(cookie).toMatch(/Secure/i);
    expect(cookie).toMatch(/SameSite=Lax/i);
    expect(cookie).not.toMatch(/Max-Age|Expires/i);
  });

  it('refuses a cross-site origin', async () => {
    const v = visitor('203.0.113.8');
    const r = await v.call('POST', '/api/session', { body: { name: 'Eve' }, headers: { Origin: 'https://evil.example' } });
    expect(r.status).toBe(403);
  });

  it('refuses the old http origin when the public side is https', async () => {
    const v = visitor('203.0.113.9');
    const r = await v.call('POST', '/api/session', { body: { name: 'Eve' }, headers: { Origin: `http://${HOST}` } });
    expect(r.status).toBe(403);
  });

  it('rate-limits per client address, and a different address is unaffected', async () => {
    mockUpstream((c) => (isChatbot(c.url) ? Response.json(chatbotReply()) : undefined));
    const a = visitor('203.0.113.20');
    const b = visitor('203.0.113.21');
    await a.start('Asha');
    await b.start('Bala');
    const statuses: number[] = [];
    for (let i = 0; i < 22; i++) statuses.push((await a.kundli(`Question ${i}`)).status);
    expect(statuses.filter((s) => s === 200)).toHaveLength(20);
    expect(statuses.filter((s) => s === 429)).toHaveLength(2);
    expect((await b.kundli('Hello')).status).toBe(200);
  });
});

describe('request body limits', () => {
  it('rejects an oversized body before reading it', async () => {
    const v = visitor('203.0.113.30');
    await v.start('Asha');
    const big = JSON.stringify({ message: 'x'.repeat(2_100_000) });
    const r = await v.call('POST', '/api/playground/chat', { raw: big });
    expect(r.status).toBe(413);
  });

  it('requires a Content-Length on writes (no unbounded chunked bodies)', async () => {
    const v = visitor('203.0.113.31');
    const r = await v.call('POST', '/api/session', { raw: '{"name":"Asha"}', chunked: true });
    expect(r.status).toBe(411);
  });
});

describe('work that outlives the browser', () => {
  it('logs the generation even when the tester closes the tab mid-request', async () => {
    mockUpstream(async (c) => {
      if (!isChatbot(c.url)) return undefined;
      await new Promise((r) => setTimeout(r, 700));
      return Response.json(chatbotReply());
    });
    const v = visitor('203.0.113.40');
    await v.start('Asha');
    const gone = await v.call('POST', '/api/playground/chat', {
      abortAfterMs: 150,
      body: { mode: 'kundli_chatbot', message: 'Will I get promoted?', request_id: 'r-closed-tab-1', conversation_id: 'conv-0000000001' },
    });
    expect(gone.status).toBe(0); // the browser hung up
    // The server finishes the job on its own and records it.
    let row: any;
    for (let i = 0; i < 40 && row?.status !== 'success'; i++) {
      await new Promise((r) => setTimeout(r, 100));
      [row] = await sql`select status, tester_name, response from playground.experiments where request_id = 'r-closed-tab-1'`;
    }
    expect(row).toMatchObject({ status: 'success', tester_name: 'Asha' });
    expect(row.response).toContain('10th house');
    expect(built.pending()).toBe(0);
  }, 15_000);

  it('a retry of that request id returns the stored result and does not generate again', async () => {
    const calls = mockUpstream((c) => (isChatbot(c.url) ? Response.json(chatbotReply()) : undefined));
    const v = visitor('203.0.113.41');
    await v.start('Asha');
    const body = { mode: 'kundli_chatbot', message: 'Hello', request_id: 'r-retry-0001', conversation_id: 'conv-0000000001' };
    const first = await v.call('POST', '/api/playground/chat', { body });
    const second = await v.call('POST', '/api/playground/chat', { body });
    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    expect(second.json.id).toBe(first.json.id);
    expect(calls.filter((c) => isChatbot(c.url))).toHaveLength(1);
  });
});

describe('database failures and cold starts', () => {
  it('reconnects after the database dropped every connection', async () => {
    expect((await send('GET', '/api/health')).status).toBe(200);
    await sql`select pg_terminate_backend(pid) from pg_stat_activity where datname = current_database() and pid <> pg_backend_pid()`;
    for (const _ of [1, 2, 3]) expect((await send('GET', '/api/health')).json).toEqual({ status: 'ok', database: 'ok' });
    // and an ordinary write is transparently retried once on a fresh connection
    const v = visitor('203.0.113.50');
    await sql`select pg_terminate_backend(pid) from pg_stat_activity where datname = current_database() and pid <> pg_backend_pid()`;
    const started = await v.start('Asha');
    expect(started.status).toBe(201);
    expect(started.headers['x-internal-retry']).toBeUndefined();
    expect((await v.kundli('Hello')).status).toBe(200);
  });

  it('keeps serving the site and answers 503 (retryable) while the database is unreachable', async () => {
    const dead = await makeEnv({ DATABASE_URL: 'postgres://postgres:x@127.0.0.1:1/none' });
    delete dead.ASSETS;
    dead.SQL_POOL = createPool({ ...dead, DB_POOL_MAX: '2' });
    const b = buildServer(dead, { staticDir: dir });
    const s = createServer(b.app);
    await new Promise<void>((r) => s.listen(0, '127.0.0.1', r));
    const p = (s.address() as AddressInfo).port;
    const get = (path: string, method = 'GET', body?: string) => new Promise<{ status: number; headers: any; json: any; text: string }>((resolve, reject) => {
      const req = httpRequest({ host: '127.0.0.1', port: p, method, path, headers: body
        ? { 'Content-Type': 'application/json', 'Content-Length': String(Buffer.byteLength(body)) } : {} }, (res) => {
        let text = ''; res.on('data', (c) => (text += c));
        res.on('end', () => { let json: any = null; try { json = JSON.parse(text); } catch { /* html */ } resolve({ status: res.statusCode!, headers: res.headers, json, text }); });
      });
      req.on('error', reject); if (body) req.write(body); req.end();
    });
    try {
      expect((await get('/healthz')).status).toBe(200);
      expect((await get('/')).text).toContain('id="root"');
      const health = await get('/api/health');
      expect(health.status).toBe(503);
      expect(health.json).toEqual({ status: 'degraded', database: 'unreachable' });
      const start = await get('/api/session', 'POST', JSON.stringify({ name: 'Asha' }));
      expect(start.status).toBe(503);
      expect(start.headers['retry-after']).toBe('5');
      expect(start.json.detail).toMatch(/temporarily unreachable/);
      expect(JSON.stringify(start)).not.toMatch(/127\.0\.0\.1|postgres:/);
    } finally {
      await new Promise((r) => s.close(r));
      await b.drain(1000);
    }
  });
});

describe('10 testers at the same time', () => {
  it('keeps every tester separate and logs everything', async () => {
    mockUpstream(async (c) => {
      await new Promise((r) => setTimeout(r, 30 + Math.random() * 120));
      if (isChatbot(c.url)) return Response.json(chatbotReply());
      if (c.url.endsWith('/chat/completions')) return openAiReply('Direct answer');
      return undefined;
    });
    const testers = Array.from({ length: 10 }, (_, i) => ({ name: `Tester ${i + 1}`, v: visitor(`198.51.100.${i + 1}`) }));
    const t0 = Date.now();
    const results = await Promise.all(testers.map(async ({ name, v }) => {
      const start = await v.start(name);
      const out = await Promise.all([
        v.kundli('Career?'), v.kundli('Marriage?'), v.direct('Tell me about my career.'),
        v.call('GET', '/api/playground/models'), v.call('GET', '/api/session'),
      ]);
      return { name, start: start.status, statuses: out.map((r) => r.status) };
    }));
    const elapsed = Date.now() - t0;

    for (const r of results) {
      expect(r.start, r.name).toBe(201);
      expect(r.statuses, r.name).toEqual([200, 200, 200, 200, 200]);
    }
    expect(elapsed).toBeLessThan(10_000);

    // Every experiment is attached to its own tester and session; nothing is lost or mixed up.
    const rows = await sql`
      select s.tester_name as owner, e.tester_name, e.session_id = s.id as same_session, e.status
      from playground.experiments e join playground.sessions s on s.id = e.session_id`;
    expect(rows).toHaveLength(30);
    expect(rows.every((r) => r.owner === r.tester_name && r.same_session && r.status === 'success')).toBe(true);
    const perTester = await sql`select tester_name, count(*)::int as n from playground.experiments group by 1 order by 1`;
    expect(perTester).toHaveLength(10);
    expect(perTester.every((r) => r.n === 3)).toBe(true);
    expect((await sql`select count(*)::int as n from playground.experiments where status = 'pending'`)[0]!.n).toBe(0);

    // The admin sees all of it, over the same server, and can download the workbook.
    const admin = visitor('192.0.2.10');
    expect((await admin.call('POST', '/api/admin/login', { body: { password: ADMIN_PASSWORD } })).status).toBe(200);
    const list = await admin.call('GET', '/api/admin/experiments?page_size=200');
    expect(list.json.total).toBe(30);
    const xlsx = await admin.call('GET', '/api/admin/export/excel');
    expect(xlsx.status).toBe(200);
    expect(String(xlsx.headers['content-type'])).toContain('spreadsheetml');
    const { default: ExcelJS } = await import('exceljs');
    const wb = new ExcelJS.Workbook();
    await wb.xlsx.load(xlsx.buf as never);
    expect(wb.worksheets.map((w) => w.name)).toEqual(expect.arrayContaining(['All Test Logs', 'Tester Summary', 'Session Summary']));
    const text = JSON.stringify(wb.getWorksheet('All Test Logs')!.getSheetValues());
    for (const t of testers) expect(text).toContain(t.name);
  }, 60_000);
});

describe('database connection settings', () => {
  it('works with prepared statements off, as a transaction-mode pooler requires', async () => {
    const [row] = await pool`select current_setting('server_version_num')::int as v`;
    expect(row!.v).toBeGreaterThan(130000);
    const viaUrl = postgres(DB_URL, { max: 1, prepare: false, fetch_types: false });
    expect((await viaUrl`select 1 as one`)[0]!.one).toBe(1);
    await viaUrl.end();
  });
});
