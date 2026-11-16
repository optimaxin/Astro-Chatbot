import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { beforeEach, describe, expect, it } from 'vitest';
import { calculateCost, catalog, findModel } from '../src/catalog';
import { hmacHex, safeEqual, verifyPassword } from '../src/crypto';
import { houseTable, parseJudgeOutput } from '../src/evaluator';
import { friendlyReason } from '../src/providers';
import { FilterError, parseFilters } from '../src/reports';
import * as store from '../src/store';
import { addDays, dayKey, dayStart, startOfToday, wallClock } from '../src/time';
import { adminHash, ADMIN_PASSWORD, resetDb, sql } from './helpers';

describe('password verification', () => {
  it('verifies the hash the project\'s own tooling produces', async () => {
    const hash = await adminHash();
    expect(await verifyPassword(ADMIN_PASSWORD, hash)).toBe('ok');
    expect(await verifyPassword('wrong', hash)).toBe('mismatch');
  });

  it('verifies an Argon2id hash made by a different library (argon2-cffi)', async () => {
    const hash = '$argon2id$v=19$m=19456,t=2,p=1$tXkgJ1ZUyeKyNINr/VmzTw$ncIvFJ074BQEG6CbWWzj2l74hKs5m/8sbGzU+KqtKz4';
    expect(await verifyPassword('fixture-password-123', hash)).toBe('ok');
    expect(await verifyPassword('fixture-password-124', hash)).toBe('mismatch');
  });

  it('supports the PBKDF2 fallback format for plans with a small CPU budget', async () => {
    const salt = new Uint8Array(16).fill(3);
    const key = await crypto.subtle.importKey('raw', new TextEncoder().encode('pw-123456'), 'PBKDF2', false, ['deriveBits']);
    const bits = new Uint8Array(await crypto.subtle.deriveBits({ name: 'PBKDF2', hash: 'SHA-256', salt, iterations: 50_000 }, key, 256));
    const b64 = (u: Uint8Array) => Buffer.from(u).toString('base64').replace(/=+$/, '');
    const stored = `pbkdf2$sha256$50000$${b64(salt)}$${b64(bits)}`;
    expect(await verifyPassword('pw-123456', stored)).toBe('ok');
    expect(await verifyPassword('pw-123457', stored)).toBe('mismatch');
  });

  it('reports malformed or dangerous hashes instead of throwing', async () => {
    for (const bad of ['', 'plain-text', '$argon2id$v=19$m=999999999,t=1,p=1$AAAA$AAAA', '$argon2id$garbage',
      'pbkdf2$sha256$99999999$AAAA$AAAA', 'pbkdf2$md5$1000$AAAA$AAAA']) {
      expect(await verifyPassword('x', bad), bad).toBe('invalid-hash');
    }
  });

  it('compares secrets in constant time and signs deterministically', async () => {
    expect(await safeEqual('abc', 'abc')).toBe(true);
    expect(await safeEqual('abc', 'abd')).toBe(false);
    expect(await safeEqual('abc', 'abcd')).toBe(false);
    expect(await hmacHex('k', 'm')).toBe(await hmacHex('k', 'm'));
    expect(await hmacHex('k', 'm')).not.toBe(await hmacHex('k2', 'm'));
  });
});

describe('time zones and filters', () => {
  it('buckets days in Asia/Kolkata and finds the start of a day', () => {
    expect(dayKey(Date.parse('2026-11-15T19:00:00Z') / 1000, 'Asia/Kolkata')).toBe('2026-11-16');
    expect(dayKey(Date.parse('2026-11-15T17:00:00Z') / 1000, 'Asia/Kolkata')).toBe('2026-11-15');
    expect(dayStart('2026-11-16', 'Asia/Kolkata')).toBe(Date.parse('2026-11-15T18:30:00Z') / 1000);
    expect(startOfToday('Asia/Kolkata', Date.parse('2026-11-16T03:00:00Z'))).toBe(Date.parse('2026-11-15T18:30:00Z') / 1000);
    expect(dayStart('2026-07-01', 'America/New_York')).toBe(Date.parse('2026-07-01T04:00:00Z') / 1000); // daylight time
    expect(addDays('2026-12-31', 1)).toBe('2027-01-01');
    expect(wallClock(Date.parse('2026-11-15T19:00:00Z') / 1000, 'Asia/Kolkata').toISOString()).toBe('2026-11-16T00:30:00.000Z');
  });

  it('turns inclusive from/to dates into a half-open instant range', () => {
    const f = parseFilters({ from: '2026-11-10', to: '2026-11-10' }, 'Asia/Kolkata');
    expect(f.until! - f.since!).toBe(86400);
    expect(() => parseFilters({ from: '2026-02-30' }, 'Asia/Kolkata')).toThrow(FilterError);
    expect(() => parseFilters({ to: '2026-1-1' }, 'Asia/Kolkata')).toThrow(FilterError);
    expect(() => parseFilters({ mode: 'x' }, 'Asia/Kolkata')).toThrow(FilterError);
    expect(parseFilters({ tester: '  Asha  ', provider: '' }, 'Asia/Kolkata')).toEqual({ tester: 'Asha' });
  });
});

describe('cost, providers and the judge', () => {
  it('calculates cost from actual tokens and never invents one', () => {
    const c = calculateCost(findModel('gemini', 'gemini-2.5-flash')!.pricing, 1000, 500);
    expect(c).toMatchObject({ input_cost: 0.0003, output_cost: 0.00125, currency: 'USD' });
    expect(c.total_cost).toBeCloseTo(0.00155, 12);
    expect(calculateCost(null, 1000, 500).total_cost).toBeNull();
    expect(calculateCost(findModel('gemini', 'gemini-2.5-flash')!.pricing, null, 5).total_cost).toBeNull();
  });

  it('keeps the catalogue to the three direct providers, with the NVIDIA evaluator', () => {
    expect(catalog.providers.map((p) => p.id)).toEqual(['nvidia', 'gemini', 'openrouter']);
    expect(catalog.evaluator).toEqual({ provider: 'nvidia', model: 'nvidia/nemotron-3.5-bolt-e4b' });
    expect(JSON.stringify(catalog)).not.toMatch(/ai-core|aicore/i);
  });

  it('maps provider failures to safe, readable reasons', () => {
    expect(friendlyReason(429, '')).toBe('Rate limit exceeded.');
    expect(friendlyReason(401, 'key=abc')).toBe('The provider rejected the API key.');
    expect(friendlyReason(500, JSON.stringify({ error: { message: 'boom\n  now' } }))).toBe('HTTP 500: boom now');
  });

  it('parses and clamps the judge\'s verdict, tolerating fences and prose', () => {
    const v = parseJudgeOutput('Sure!\n```json\n{"scores":{"relevance":9,"accuracy":12,"tone":-3,"emoji_usage":null,"chart_consistency":"7"},"feedback":" ok ","strengths":["a"],"improvements":[]}\n```');
    expect(v.scores).toMatchObject({ relevance: 9, accuracy: 10, tone: 0, emoji_usage: null, chart_consistency: 7, language_preservation: null });
    expect(v.overall).toBeCloseTo(6.5, 1);
    expect(v.feedback).toBe('ok');
    expect(() => parseJudgeOutput('no json here')).toThrow();
  });
});

describe('chart table for the judge', () => {
  it('derives houses, signs, lords and occupants from the chart facts', () => {
    const table = houseTable(['lagna=Aries', 'moonSign=Leo', 'Saturn=Aquarius/H11', 'Venus=Aquarius/H11', 'Ketu=Aries/H1'])!;
    const lines = table.split('\n');
    expect(lines).toHaveLength(12);
    expect(lines[0]).toBe('House 1: Aries (lord Mars) — contains Ketu');
    expect(lines[9]).toBe('House 10: Capricorn (lord Saturn)');
    expect(lines[10]).toBe('House 11: Aquarius (lord Saturn) — contains Saturn, Venus');
    expect(houseTable(['lagna=Virgo'])!.split('\n')[9]).toBe('House 10: Gemini (lord Mercury)');
  });
  it('returns nothing without a usable lagna', () => {
    expect(houseTable([])).toBeNull();
    expect(houseTable(['lagna=Nowhere', 'Sun=Leo/H5'])).toBeNull();
  });
});

describe('database', () => {
  beforeEach(resetDb);

  it('keeps every object in the dedicated playground schema and never touches other tables', () => {
    const sqlText = readFileSync(join(__dirname, '..', 'migrations', '0001_playground.sql'), 'utf8')
      .replace(/--.*$/gm, '').replace(/\$\$[\s\S]*?\$\$/g, ' ');
    const statements = sqlText.split(';').map((s) => s.trim()).filter(Boolean);
    expect(statements.length).toBeGreaterThan(10);
    for (const s of statements) {
      if (/^create (table|index|unique index)/i.test(s) || /^alter table/i.test(s)) expect(s, s).toMatch(/playground\./i);
      expect(s, s).not.toMatch(/\b(drop|truncate|delete\s+from|update\s+\w)/i);
    }
    expect(sqlText).not.toMatch(/\bpublic\./i);
  });

  it('creates exactly one row when the same request id is inserted concurrently', async () => {
    const [s] = await sql`insert into playground.sessions (token_hash, tester_name) values ('h', 'Asha') returning id`;
    const base = { request_id: 'req-concurrent-1', session_id: s!.id as string, tester_name: 'Asha', ip_hash: null, mode: 'direct_provider' as const,
      provider: 'gemini', model: 'm', model_requested: 'm', system_prompt: null, user_prompt: 'hi', context: null,
      params: { temperature: 0.5 }, effective_request: { messages: [{ role: 'user', content: 'hi' }] }, pricing_snapshot: null };
    const results = await Promise.all(Array.from({ length: 6 }, () => store.insertPending(sql, base)));
    expect(results.filter((r) => r.created)).toHaveLength(1);
    expect(new Set(results.map((r) => r.row.id)).size).toBe(1);
    expect(results[0]!.row.params).toEqual({ temperature: 0.5 });
    expect((await sql`select count(*)::int as n from playground.experiments`)[0]!.n).toBe(1);
  });

  it('round-trips JSON, unicode and timestamps, and finalises a pending record', async () => {
    const [s] = await sql`insert into playground.sessions (token_hash, tester_name) values ('h2', 'राहुल') returning id`;
    const { row } = await store.insertPending(sql, { request_id: 'req-roundtrip-1', session_id: s!.id as string, tester_name: 'राहुल',
      ip_hash: null, mode: 'kundli_chatbot', provider: 'unknown', model: 'unknown', model_requested: 'auto', system_prompt: null,
      user_prompt: 'नमस्ते\nworld "quoted" \\ backslash', context: '{}', params: null, effective_request: { a: [1, { b: 'ü' }] }, pricing_snapshot: null });
    expect(row.status).toBe('pending');
    expect(typeof row.created_at).toBe('number');
    const done = await store.finalizeExperiment(sql, row.id, {
      status: 'success', provider: 'nvidia', model: 'm', response: 'ok', input_tokens: 5, output_tokens: 3, total_tokens: 8,
      total_cost: 0.5, currency: 'USD', metadata: { ai_core: { chart_facts_used: ['lagna=Aries'] } },
    });
    expect(done).toMatchObject({ status: 'success', provider: 'nvidia', user_prompt: 'नमस्ते\nworld "quoted" \\ backslash', total_cost: 0.5 });
    expect(done.effective_request).toEqual({ a: [1, { b: 'ü' }] });
    expect(done.metadata.ai_core.chart_facts_used).toEqual(['lagna=Aries']);
  });

  it('survives a restart: data is read back through a fresh connection', async () => {
    const [s] = await sql`insert into playground.sessions (token_hash, tester_name) values ('h3', 'Asha') returning id`;
    await sql`insert into playground.experiments (request_id, session_id, tester_name, mode, provider, model, user_prompt, status)
              values ('req-persist-1', ${s!.id}, 'Asha', 'direct_provider', 'gemini', 'm', 'hi', 'success')`;
    const { default: postgres } = await import('postgres');
    const fresh = postgres(process.env.TEST_DATABASE_URL ?? 'postgres://postgres:postgres@localhost:55432/pgtest', { max: 1 });
    expect((await fresh`select tester_name from playground.experiments where request_id = 'req-persist-1'`)[0]!.tester_name).toBe('Asha');
    await fresh.end();
  });

  it('enforces integrity: an experiment needs a real session, and statuses are constrained', async () => {
    await expect(sql`insert into playground.experiments (request_id, session_id, tester_name, mode, provider, model, user_prompt, status)
      values ('x', gen_random_uuid(), 'n', 'direct_provider', 'p', 'm', 'u', 'success')`).rejects.toThrow(/foreign key/i);
    const [s] = await sql`insert into playground.sessions (token_hash, tester_name) values ('h4', 'A') returning id`;
    await expect(sql`insert into playground.experiments (request_id, session_id, tester_name, mode, provider, model, user_prompt, status)
      values ('y', ${s!.id}, 'n', 'direct_provider', 'p', 'm', 'u', 'weird')`).rejects.toThrow(/check/i);
    await expect(sql`insert into playground.sessions (token_hash, tester_name) values ('h5', ${'x'.repeat(81)})`).rejects.toThrow(/check/i);
  });
});
