import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { resolveTimezoneOffsetMinutes } from '../src/kundli';
import { Browser, generateReferenceKundli, kundliFullReply, makeEnv, mockUpstream, resetDb, sql, type UpstreamCall } from './helpers';

describe('resolveTimezoneOffsetMinutes (historical DST, not today\'s rules)', () => {
  it('resolves India\'s fixed +5:30 regardless of date', () => {
    expect(resolveTimezoneOffsetMinutes(28.6139, 77.209, '1995-04-12', '06:30')).toBe(330);
    expect(resolveTimezoneOffsetMinutes(28.6139, 77.209, '2026-01-01', '00:00')).toBe(330);
  });

  it('applies the actual 1995 US daylight-saving boundary (first Sunday of April, pre-2007 rule), not the modern one', () => {
    // 1995: DST began 1995-04-02. Using today's rule (second Sunday of March) here would be wrong.
    expect(resolveTimezoneOffsetMinutes(40.7128, -74.006, '1995-04-01', '12:00')).toBe(-300); // still EST
    expect(resolveTimezoneOffsetMinutes(40.7128, -74.006, '1995-04-03', '12:00')).toBe(-240); // already EDT
  });

  it('applies the modern (post-2007) US boundary for a recent date', () => {
    expect(resolveTimezoneOffsetMinutes(40.7128, -74.006, '2026-07-01', '12:00')).toBe(-240); // EDT
    expect(resolveTimezoneOffsetMinutes(40.7128, -74.006, '2026-01-01', '12:00')).toBe(-300); // EST
  });
});

let env: Awaited<ReturnType<typeof makeEnv>>;
let tester: Browser;
beforeEach(async () => {
  await resetDb();
  env = await makeEnv();
  tester = new Browser(env);
  await tester.start('Asha');
});
afterEach(() => vi.unstubAllGlobals());

const isKundliFull = (c: UpstreamCall) => c.url.endsWith('/api/calculators/kundli-full');

describe('reference Kundli generation', () => {
  it('resolves the timezone, calls the real calculator, and stores the chart as current', async () => {
    const calls = mockUpstream((c) => (isKundliFull(c) ? Response.json(kundliFullReply()) : undefined));
    const res = await generateReferenceKundli(tester);
    expect(res.status).toBe(201);
    expect(res.data).toMatchObject({ status: 'current', location_provider: 'manual', engine_source: 'tredev-backend:kundli-full' });
    expect(res.data.birth).toMatchObject({ date: '1995-04-12', time: '06:30', timezoneOffsetMinutes: 330, latitude: 28.6139, longitude: 77.209 });
    expect(res.data.chart.kundli.ascendant.rashi).toBe('Leo');

    expect(calls).toHaveLength(1);
    expect(calls[0]!.body).toMatchObject({ date: '1995-04-12', time: '06:30', timezoneOffsetMinutes: 330, latitude: 28.6139, longitude: 77.209 });

    const [row] = await sql`select status from playground.reference_kundlis where session_id = (select id from playground.sessions limit 1)`;
    expect(row!.status).toBe('current');
  });

  it('marks the AWS place-sourced chart distinctly from a manual one', async () => {
    mockUpstream((c) => (isKundliFull(c) ? Response.json(kundliFullReply()) : undefined));
    const res = await generateReferenceKundli(tester, { placeId: 'place-new-delhi' });
    expect(res.data.location_provider).toBe('aws');
    expect(res.data.location_place_id).toBe('place-new-delhi');
  });

  it('flips the previous chart to outdated when a new one is generated for the same session', async () => {
    mockUpstream((c) => (isKundliFull(c) ? Response.json(kundliFullReply()) : undefined));
    const first = await generateReferenceKundli(tester);
    const second = await generateReferenceKundli(tester, { date: '2000-01-01' });
    expect(second.status).toBe(201);
    const rows = await sql`select id, status from playground.reference_kundlis order by created_at`;
    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({ id: first.data.id, status: 'outdated' });
    expect(rows[1]).toMatchObject({ id: second.data.id, status: 'current' });
  });

  it('surfaces a backend failure as 502 and stores nothing', async () => {
    mockUpstream(() => Response.json({ success: false, error: { message: 'boom' } }, { status: 500 }));
    const res = await generateReferenceKundli(tester);
    expect(res.status).toBe(502);
    expect((await sql`select count(*)::int as n from playground.reference_kundlis`)[0]!.n).toBe(0);
  });

  it('is unavailable (503) when TREDEV_BACKEND_URL is not configured', async () => {
    const b = new Browser(await makeEnv({ TREDEV_BACKEND_URL: '' }));
    await b.start('Ben');
    expect((await generateReferenceKundli(b)).status).toBe(503);
  });

  it('reports no chart yet, then the generated one, via /kundli/current', async () => {
    expect((await tester.json('/api/playground/kundli/current')).data).toBeNull();
    mockUpstream((c) => (isKundliFull(c) ? Response.json(kundliFullReply()) : undefined));
    const generated = await generateReferenceKundli(tester);
    const current = await tester.json('/api/playground/kundli/current');
    expect(current.data.id).toBe(generated.data.id);
  });
});
