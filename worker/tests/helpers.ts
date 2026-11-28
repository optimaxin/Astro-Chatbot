import { argon2idAsync } from '@noble/hashes/argon2.js';
import postgres from 'postgres';
import { vi } from 'vitest';
import { app } from '../src/app';
import type { Env } from '../src/env';

export const DB_URL = process.env.TEST_DATABASE_URL ?? 'postgres://postgres:postgres@localhost:55432/pgtest';
export const ADMIN_PASSWORD = 'correct horse battery staple';

const b64 = (bytes: Uint8Array) => Buffer.from(bytes).toString('base64').replace(/=+$/, '');

/** A real Argon2id PHC hash with small parameters so the tests stay fast. */
export async function adminHash(password = ADMIN_PASSWORD) {
  const salt = new Uint8Array(16).fill(7);
  const hash = await argon2idAsync(password, salt, { m: 1024, t: 1, p: 1, dkLen: 32 });
  return `$argon2id$v=19$m=1024,t=1,p=1$${b64(salt)}$${b64(hash)}`;
}

export const sql = postgres(DB_URL, { max: 4, onnotice: () => {} });

export async function resetDb() {
  await sql`truncate playground.claim_verifications, playground.reference_kundlis,
            playground.experiments, playground.sessions, playground.admin_sessions, playground.audit_logs
            restart identity cascade`;
}

export async function makeEnv(over: Partial<Env> = {}): Promise<Env> {
  return {
    DATABASE_URL: DB_URL,
    SESSION_SECRET: 'test-session-secret-0123456789',
    ADMIN_PASSWORD_HASH: await adminHash(),
    AI_CORE_URL: 'http://ai-core.test',
    TREDEV_BACKEND_URL: 'http://tredev-backend.test',
    NVIDIA_API_KEY: 'nvidia-test-key-SECRET',
    GEMINI_API_KEY: 'gemini-test-key-SECRET',
    OPENROUTER_API_KEY: '',
    AWS_LOCATION_ENABLED: 'true',
    ASSETS: { fetch: async () => new Response('<!doctype html><div id="root"></div>', { headers: { 'content-type': 'text/html' } }) } as unknown as Fetcher,
    ...over,
  };
}

interface Cookie { value: string; path: string }

/** A tiny browser: keeps cookies (respecting Path) and calls the Worker in-process. */
export class Browser {
  jar = new Map<string, Cookie>();
  pending: Promise<unknown>[] = [];
  lastSetCookie: string[] = [];
  constructor(public env: Env, public origin = 'https://playground.test') {}

  async call(path: string, init: { method?: string; body?: unknown; headers?: Record<string, string>; rawBody?: string } = {}) {
    const url = new URL(path, this.origin);
    const headers = new Headers(init.headers);
    const cookie = [...this.jar].filter(([, c]) => url.pathname.startsWith(c.path)).map(([k, c]) => `${k}=${c.value}`).join('; ');
    if (cookie) headers.set('Cookie', cookie);
    const method = init.method ?? (init.body !== undefined || init.rawBody !== undefined ? 'POST' : 'GET');
    let body: string | undefined;
    if (init.rawBody !== undefined) body = init.rawBody;
    else if (init.body !== undefined) { body = JSON.stringify(init.body); headers.set('Content-Type', 'application/json'); }
    const ctx = { waitUntil: (p: Promise<unknown>) => { this.pending.push(Promise.resolve(p).catch(() => undefined)); }, passThroughOnException() {} };
    const res = await app.fetch(new Request(url, { method, headers, body }), this.env, ctx as unknown as ExecutionContext);
    await Promise.all(this.pending.splice(0));
    this.lastSetCookie = res.headers.getSetCookie();
    for (const sc of this.lastSetCookie) {
      const [pair, ...attrs] = sc.split(';').map((s) => s.trim());
      const [name, value = ''] = pair!.split('=');
      const path = attrs.find((a) => a.toLowerCase().startsWith('path='))?.slice(5) ?? '/';
      const expired = /max-age=0/i.test(sc) || /expires=thu, 01 jan 1970/i.test(sc);
      if (expired || value === '') this.jar.delete(name!);
      else this.jar.set(name!, { value, path });
    }
    return res;
  }

  async json(path: string, init?: Parameters<Browser['call']>[1]) {
    const res = await this.call(path, init);
    const text = await res.text();
    let data: any = null;
    try { data = JSON.parse(text); } catch { /* not JSON */ }
    return { status: res.status, data, text, headers: res.headers };
  }

  /** Enters a name, exactly like the welcome screen. */
  async start(name: string) {
    const r = await this.json('/api/session', { body: { name } });
    if (r.status !== 201) throw new Error(`could not start session for ${name}: ${r.status} ${r.text}`);
    return r.data as { session_id: string; tester_name: string };
  }

  async adminLogin(password = ADMIN_PASSWORD) {
    const r = await this.json('/api/admin/login', { body: { password } });
    return r;
  }
}

export interface UpstreamCall { url: string; method: string; headers: Record<string, string>; body: any }

/**
 * Replaces the global fetch (used only for upstream providers; Postgres is a socket).
 * Returns the recorded calls. A handler returns a Response, or undefined to fall through to 404.
 */
export function mockUpstream(handler: (call: UpstreamCall) => Response | undefined | Promise<Response | undefined>) {
  const calls: UpstreamCall[] = [];
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
    let body: any = undefined;
    if (typeof init.body === 'string') { try { body = JSON.parse(init.body); } catch { body = init.body; } }
    const call: UpstreamCall = { url, method: init.method ?? 'GET', headers: Object.fromEntries(new Headers(init.headers).entries()), body };
    calls.push(call);
    return (await handler(call)) ?? new Response('no mock for ' + url, { status: 404 });
  });
  return calls;
}

export const chatbotReply = (over: Record<string, unknown> = {}) => ({
  answer: 'Aapka 10th house Capricorn hai aur uska lord Saturn 11th house mein hai.',
  messages: ['Aapka 10th house Capricorn hai.', 'Uska lord Saturn 11th house mein hai.'],
  persona: 'tredev', response_style: 'whatsapp_hinglish', session_id: 's', request_id: 'req_abc', intent: 'career',
  sources: [{ id: 'a' }], citations: [{ id: 'k1', title: '15 — Effects' }], degraded: false, notices: [], latency_ms: 4200,
  chart_facts_used: ['lagna=Aries', 'Saturn=Aquarius/H11', 'mahadasha=Moon'], session_token: 'anon-token-1',
  usage: {
    input_tokens: 5206, output_tokens: 150, total_tokens: 5356, tokens_available: true,
    model: 'nvidia/nemotron-3.5-lightning-30b-a3b', provider: 'nvidia', estimated_cost_usd: 0, turn_total_tokens: 5900,
    turn_cost_usd: 0, llm_calls: 2, fallback_used: false, token_source: 'provider', estimated_total_tokens: null,
  },
  ...over,
});

export const openAiReply = (text = 'Hello there!', usage: Record<string, unknown> | null = { prompt_tokens: 1000, completion_tokens: 500, total_tokens: 1500 }) =>
  Response.json({ model: 'reported-model', choices: [{ message: { content: text }, finish_reason: 'stop' }], ...(usage ? { usage } : {}) });

export const kundliChat = (b: Browser, message = 'Mera career kaisa rahega?', extra: Record<string, unknown> = {}) =>
  b.json('/api/playground/chat', {
    body: { mode: 'kundli_chatbot', message, request_id: `r-${Math.random().toString(36).slice(2, 12)}`,
      conversation_id: 'conv-0000000001', ...extra },
  });

export const directChat = (b: Browser, extra: Record<string, unknown> = {}) =>
  b.json('/api/playground/chat', {
    body: { mode: 'direct_provider', provider: 'gemini', model: 'gemini-2.5-flash', message: 'Tell me about my career.',
      request_id: `r-${Math.random().toString(36).slice(2, 12)}`, conversation_id: 'conv-0000000001', ...extra },
  });

/** `POST {TREDEV_BACKEND_URL}/api/calculators/kundli-full`'s real response shape (only the fields this playground reads). */
export const kundliFullReply = (over: Record<string, unknown> = {}) => ({
  success: true,
  data: {
    kundli: {
      ascendant: { rashi: 'Leo', degreeInSign: 12.5, longitude: 132.5 },
      planets: [
        { id: 'sun', rashi: 'Aries', house: 9, nakshatra: 'Ashwini', nakshatraPada: 2, nakshatraLord: 'Ketu', retrograde: false, degreeInSign: 5 },
        { id: 'moon', rashi: 'Taurus', house: 10, nakshatra: 'Rohini', nakshatraPada: 1, nakshatraLord: 'Moon', retrograde: false, degreeInSign: 3 },
        { id: 'mars', rashi: 'Scorpio', house: 4, nakshatra: 'Anuradha', nakshatraPada: 3, nakshatraLord: 'Saturn', retrograde: false, degreeInSign: 18 },
        { id: 'mercury', rashi: 'Pisces', house: 8, nakshatra: 'Revati', nakshatraPada: 4, nakshatraLord: 'Mercury', retrograde: true, degreeInSign: 22 },
        { id: 'jupiter', rashi: 'Sagittarius', house: 5, nakshatra: 'Purva Ashadha', nakshatraPada: 2, nakshatraLord: 'Venus', retrograde: false, degreeInSign: 9 },
        { id: 'venus', rashi: 'Libra', house: 3, nakshatra: 'Swati', nakshatraPada: 1, nakshatraLord: 'Rahu', retrograde: false, degreeInSign: 14 },
        { id: 'saturn', rashi: 'Aquarius', house: 7, nakshatra: 'Shatabhisha', nakshatraPada: 3, nakshatraLord: 'Rahu', retrograde: true, degreeInSign: 27 },
        { id: 'rahu', rashi: 'Gemini', house: 11, nakshatra: 'Ardra', nakshatraPada: 2, nakshatraLord: 'Rahu', retrograde: true, degreeInSign: 1 },
        { id: 'ketu', rashi: 'Sagittarius', house: 5, nakshatra: 'Mula', nakshatraPada: 2, nakshatraLord: 'Ketu', retrograde: true, degreeInSign: 1 },
      ],
      moonNakshatra: { name: 'Rohini', pada: 1, lord: 'Moon' },
    },
    mahadashaTimeline: [
      { lord: 'Moon', active: true, antardashas: [{ lord: 'Mars', active: false }, { lord: 'Rahu', active: true }] },
      { lord: 'Mars', active: false, antardashas: [] },
    ],
    doshas: { kaalSarp: { present: false }, sadeSati: { active: false } },
    yogas: [],
    navamsaChart: { ascendant: { rashi: 'Capricorn' }, planets: [] },
  },
  ...over,
});

export const generateReferenceKundli = (b: Browser, over: Record<string, unknown> = {}) =>
  b.json('/api/playground/kundli/generate', {
    body: { date: '1995-04-12', time: '06:30', latitude: 28.6139, longitude: 77.209, placeLabel: 'New Delhi', ...over },
  });

/** `GET {TREDEV_BACKEND_URL}/api/calculators/place-suggest`'s real response shape. */
export const placeSuggestReply = (items: Record<string, unknown>[] = [{ label: 'New Delhi, India', placeId: 'place-new-delhi' }]) =>
  Response.json({ success: true, data: items });

/** `GET {TREDEV_BACKEND_URL}/api/calculators/place-resolve`'s real response shape. */
export const placeResolveReply = (over: Record<string, unknown> = {}) =>
  Response.json({ success: true, data: { latitude: 28.6139, longitude: 77.209, displayName: 'New Delhi, India', ...over } });

export const placeNotFoundReply = () =>
  Response.json({ success: false, error: { code: 'NOT_FOUND', message: 'Could not resolve that place.' } }, { status: 404 });
