import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { validateName } from '../src/http';
import { Browser, directChat, kundliChat, makeEnv, mockUpstream, openAiReply, resetDb, sql } from './helpers';

let env: Awaited<ReturnType<typeof makeEnv>>;
beforeEach(async () => {
  await resetDb();
  env = await makeEnv();
});
afterEach(() => vi.unstubAllGlobals());

describe('validateName', () => {
  it('trims, normalises and collapses whitespace', () => {
    expect(validateName('  Asha   Rao \n')).toEqual({ ok: true, name: 'Asha Rao' });
  });
  it('rejects empty, whitespace-only and non-string input', () => {
    for (const bad of ['', '   ', '\n\t', '  ', null, undefined, 42, {}]) expect(validateName(bad).ok).toBe(false);
  });
  it('enforces the 80-character limit by characters, not bytes', () => {
    expect(validateName('अ'.repeat(80)).ok).toBe(true);
    expect(validateName('a'.repeat(81)).ok).toBe(false);
  });
  it('rejects control characters but accepts Indic joiners and special characters', () => {
    expect(validateName('Bad\u0007Name').ok).toBe(false);
    // Line/paragraph separators are whitespace, so they are normalised to a plain space.
    expect(validateName('Line' + String.fromCharCode(0x2028) + 'Break')).toEqual({ ok: true, name: 'Line Break' });
    expect(validateName('क्‍ष').ok).toBe(true);
    expect(validateName("O'Brien <b>&</b> 王").ok).toBe(true);
  });
  it('requires at least one letter or number', () => {
    expect(validateName('!!! ...').ok).toBe(false);
    expect(validateName('42').ok).toBe(true);
  });
});

describe('name entry and sessions', () => {
  it('has no session until a name is submitted, and protects every tester route', async () => {
    const b = new Browser(env);
    expect((await b.json('/api/session')).status).toBe(401);
    expect((await b.json('/api/playground/models')).status).toBe(401);
    expect((await b.json('/api/playground/experiments')).status).toBe(401);
    expect((await directChat(b)).status).toBe(401);
    expect((await kundliChat(b)).status).toBe(401);
  });

  it('rejects invalid names with 422 and creates nothing', async () => {
    const b = new Browser(env);
    for (const name of ['', '   ', 'x'.repeat(81), 'bad\u0007', '???', 5]) {
      expect((await b.json('/api/session', { body: { name } })).status, JSON.stringify(name)).toBe(422);
    }
    expect((await b.json('/api/session', { body: {} })).status).toBe(422);
    expect((await b.json('/api/session', { rawBody: 'not json' })).status).toBe(400);
    expect((await sql`select count(*)::int as n from playground.sessions`)[0]!.n).toBe(0);
  });

  it('creates an anonymous session for a valid name and identifies the tester afterwards', async () => {
    const b = new Browser(env);
    const made = await b.json('/api/session', { body: { name: '  Asha   Rao  ' } });
    expect(made.status).toBe(201);
    expect(made.data.tester_name).toBe('Asha Rao');
    expect(made.data.session_id).toMatch(/^[0-9a-f-]{36}$/);
    const me = await b.json('/api/session');
    expect(me.data).toEqual({ session_id: made.data.session_id, tester_name: 'Asha Rao' });
  });

  it('stores special characters safely and returns them unchanged', async () => {
    const b = new Browser(env);
    const name = `Robert'); DROP TABLE playground.sessions;-- <img src=x onerror=alert(1)> 🙏 राहुल`;
    expect((await b.json('/api/session', { body: { name } })).status).toBe(201);
    expect((await b.json('/api/session')).data.tester_name).toBe(name);
    expect((await sql`select count(*)::int as n from playground.sessions`)[0]!.n).toBe(1);
  });

  it('survives a refresh (same cookie) and a new browser session asks again', async () => {
    const first = new Browser(env);
    const { session_id } = await first.start('Asha');
    expect((await first.json('/api/session')).data.session_id).toBe(session_id); // refresh
    const fresh = new Browser(env); // browser restarted: no cookie
    expect((await fresh.json('/api/session')).status).toBe(401);
  });

  it('uses a browser-session cookie: HttpOnly, SameSite, scoped, no expiry, Secure over https', async () => {
    const b = new Browser(env);
    await b.json('/api/session', { body: { name: 'Asha' } });
    const cookie = b.lastSetCookie.find((c) => c.startsWith('pg_session='))!.toLowerCase();
    expect(cookie).toContain('httponly');
    expect(cookie).toContain('samesite=lax');
    expect(cookie).toContain('path=/api');
    expect(cookie).toContain('secure');
    expect(cookie).not.toMatch(/max-age|expires/);
  });

  it('only stores a hash of the session token', async () => {
    const b = new Browser(env);
    await b.start('Asha');
    const token = b.jar.get('pg_session')!.value;
    const [row] = await sql`select token_hash from playground.sessions`;
    expect(row!.token_hash).not.toBe(token);
    expect(JSON.stringify(await sql`select * from playground.sessions`)).not.toContain(token);
  });

  it('gives each tester a distinct session even with the same name', async () => {
    const a = new Browser(env), c = new Browser(env);
    expect((await a.start('Asha')).session_id).not.toBe((await c.start('Asha')).session_id);
  });

  it('refuses forged, ended and idle-expired sessions', async () => {
    const forged = new Browser(env);
    forged.jar.set('pg_session', { value: 'x'.repeat(43), path: '/api' });
    expect((await forged.json('/api/session')).status).toBe(401);

    const b = new Browser(env);
    await b.start('Asha');
    await sql`update playground.sessions set last_seen_at = now() - interval '13 hours'`;
    expect((await b.json('/api/session')).status).toBe(401);

    const ended = new Browser(env);
    await ended.start('Ben');
    expect((await ended.json('/api/session', { method: 'DELETE' })).status).toBe(200);
    expect((await ended.json('/api/session')).status).toBe(401);
  });

  it('refuses cross-origin writes', async () => {
    const b = new Browser(env);
    const r = await b.json('/api/session', { body: { name: 'Eve' }, headers: { Origin: 'https://evil.example' } });
    expect(r.status).toBe(403);
    const ok = await b.json('/api/session', { body: { name: 'Eve' }, headers: { Origin: 'https://playground.test' } });
    expect(ok.status).toBe(201);
  });

  it('never trusts a tester name or session id sent by the client', async () => {
    mockUpstream((c) => (c.url.endsWith('/chat/completions') ? openAiReply() : undefined));
    const b = new Browser(env);
    const real = await b.start('Asha');
    const other = new Browser(env);
    const victim = await other.start('Victim');
    const res = await directChat(b, { tester_name: 'Mallory', session_id: victim.session_id, ip_hash: 'x' });
    expect(res.status).toBe(200);
    expect(res.data.tester_name).toBe('Asha');
    expect(res.data.session_id).toBe(real.session_id);
    const [row] = await sql`select tester_name, session_id from playground.experiments`;
    expect(row).toMatchObject({ tester_name: 'Asha', session_id: real.session_id });
  });

  it('keeps each tester\'s experiments private to their own session', async () => {
    mockUpstream((c) => (c.url.endsWith('/chat/completions') ? openAiReply() : undefined));
    const a = new Browser(env), c = new Browser(env);
    await a.start('Asha');
    await c.start('Ben');
    const exp = (await directChat(a)).data;
    expect((await c.json(`/api/playground/experiments/${exp.id}`)).status).toBe(404);
    expect((await c.json('/api/playground/experiments')).data).toEqual([]);
    expect((await c.json('/api/playground/feedback', { body: { experiment_id: exp.id, rating: 'bad' } })).status).toBe(404);
    expect((await a.json('/api/playground/experiments')).data.map((e: any) => e.id)).toEqual([exp.id]);
  });
});

describe('the Kundli chatbot is the default mode', () => {
  it('lists Kundli Chatbot first, never exposes ai-core as a provider, and hides secrets', async () => {
    mockUpstream((c) => (c.url.endsWith('/response-styles')
      ? Response.json([{ id: 'whatsapp_hinglish', name: 'WhatsApp Hinglish' }, { id: 'english', name: 'English' }]) : undefined));
    const b = new Browser(env);
    await b.start('Asha');
    const { data, text } = await b.json('/api/playground/models');
    expect(data.modes[0]).toMatchObject({ id: 'kundli_chatbot', label: 'Kundli Chatbot', available: true });
    expect(data.providers.map((p: any) => p.id)).toEqual(['nvidia', 'gemini', 'openrouter']);
    expect(JSON.stringify(data.providers)).not.toMatch(/ai-core|aicore|Tredev Chatbot/i);
    expect(data.chatbot.styles.map((s: any) => s.id)).toEqual(['whatsapp_hinglish', 'english']);
    expect(data.providers.find((p: any) => p.id === 'openrouter').configured).toBe(false);
    expect(text).not.toMatch(/SECRET|ai-core\.test|api_key/);
  });

  it('falls back to a built-in style list when the chatbot cannot be reached', async () => {
    mockUpstream(() => { throw new Error('offline'); });
    const b = new Browser(env);
    await b.start('Asha');
    const { data } = await b.json('/api/playground/models');
    expect(data.chatbot.styles.length).toBeGreaterThan(1);
  });
});

describe('serving and health', () => {
  it('serves the app shell for non-API routes and a JSON 404 for unknown API routes', async () => {
    const b = new Browser(env);
    const page = await b.call('/admin/experiments');
    expect(await page.text()).toContain('id="root"');
    expect((await b.json('/api/nope')).status).toBe(404);
  });

  it('reports health with a real database check, and degrades without one', async () => {
    const b = new Browser(env);
    expect((await b.json('/api/health')).data).toEqual({ status: 'ok', database: 'ok' });
    const broken = new Browser(await makeEnv({ DATABASE_URL: 'postgres://postgres:wrong@localhost:55432/pgtest' }));
    const r = await broken.json('/api/health');
    expect(r.status).toBe(503);
    expect(r.data.status).toBe('degraded');
  });

  it('sets security headers and refuses oversized bodies', async () => {
    const b = new Browser(env);
    const r = await b.json('/api/health');
    expect(r.headers.get('x-frame-options')).toBe('DENY');
    expect(r.headers.get('content-security-policy')).toContain("frame-ancestors 'none'");
    expect(r.headers.get('strict-transport-security')).toBeTruthy();
    expect(r.headers.get('cache-control')).toBe('no-store');
    await b.start('Asha');
    const big = await b.json('/api/playground/chat', { rawBody: JSON.stringify({ message: 'x'.repeat(1_100_000) }) });
    expect(big.status).toBe(413);
  });

  it('fails closed when SESSION_SECRET is missing', async () => {
    const b = new Browser(await makeEnv({ SESSION_SECRET: '' }));
    expect((await b.json('/api/session', { body: { name: 'Asha' } })).status).toBe(503);
  });
});
