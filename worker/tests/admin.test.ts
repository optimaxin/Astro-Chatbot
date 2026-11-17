import ExcelJS from 'exceljs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CELL_LIMIT, safeText } from '../src/excel';
import { ADMIN_PASSWORD, Browser, makeEnv, resetDb, sql } from './helpers';

let env: Awaited<ReturnType<typeof makeEnv>>;
let admin: Browser;
beforeEach(async () => {
  await resetDb();
  env = await makeEnv();
  admin = new Browser(env);
});
afterEach(() => vi.unstubAllGlobals());

let seq = 0;
async function tester(name: string) {
  const b = new Browser(env);
  return { ...(await b.start(name)), browser: b };
}

async function seed(who: { session_id: string; tester_name: string }, over: Record<string, unknown> = {}) {
  const row = {
    request_id: `seed-${++seq}-${Math.random().toString(36).slice(2, 8)}`, session_id: who.session_id, tester_name: who.tester_name,
    mode: 'direct_provider', provider: 'gemini', model: 'gemini-2.5-flash', user_prompt: 'Hi', response: 'Hello',
    original_response: 'Hello', status: 'success', input_tokens: 100, output_tokens: 50, total_tokens: 150,
    input_cost: 0.00003, output_cost: 0.000125, total_cost: 0.000155, currency: 'USD', latency_ms: 800, usage_source: 'provider',
    ...over,
  };
  const [inserted] = await sql`insert into playground.experiments ${sql(row as never)} returning id`;
  return inserted!.id as number;
}

const evalJson = (scores: Record<string, number | null>, overall: number) =>
  sql.json({ label: 'AI Evaluation', scores, overall, feedback: 'ok', evaluator: { provider: 'nvidia', model: 'judge' } } as never);

async function login() {
  const r = await admin.adminLogin();
  expect(r.status).toBe(200);
  return r.data.csrf_token as string;
}

describe('admin authentication', () => {
  it('protects every admin and export endpoint', async () => {
    for (const path of ['/api/admin/me', '/api/admin/analytics', '/api/admin/experiments', '/api/admin/experiments/1',
      '/api/admin/models', '/api/admin/audit-logs', '/api/admin/export/excel']) {
      expect((await admin.json(path)).status, path).toBe(401);
    }
    expect((await admin.json('/api/admin/logout', { method: 'POST' })).status).toBe(401);
  });

  it('is separate from tester access: a tester session opens nothing in admin', async () => {
    const t = await tester('Asha');
    expect((await t.browser.json('/api/admin/analytics')).status).toBe(401);
    expect((await t.browser.json('/api/admin/export/excel')).status).toBe(401);
    // and an admin session is not a tester session
    await login();
    expect((await admin.json('/api/session')).status).toBe(401);
  });

  it('rejects a wrong password, audits it, and accepts the right one with a hardened cookie', async () => {
    expect((await admin.adminLogin('wrong')).status).toBe(401);
    expect((await admin.adminLogin('')).status).toBe(422);
    await login();
    const cookie = admin.lastSetCookie.find((c) => c.startsWith('pg_admin='))!.toLowerCase();
    expect(cookie).toContain('httponly');
    expect(cookie).toContain('samesite=strict');
    expect(cookie).toContain('path=/api/admin');
    expect(cookie).toContain('secure');
    expect((await admin.json('/api/admin/me')).data).toMatchObject({ authenticated: true });
    const actions = (await admin.json('/api/admin/audit-logs')).data.items.map((i: any) => i.action);
    expect(actions).toEqual(expect.arrayContaining(['admin.login', 'admin.login_failed']));
  });

  it('locks out after repeated failures, even for the right password', async () => {
    for (let i = 0; i < 5; i++) expect((await admin.adminLogin('wrong')).status).toBe(401);
    expect((await admin.adminLogin()).status).toBe(429);
    const [row] = await sql`select count(*)::int as n from playground.audit_logs where action = 'admin.login_blocked'`;
    expect(row!.n).toBe(1);
  });

  it('also stops a guesser who rotates addresses (global ceiling)', async () => {
    const from = (n: number, password: string) => admin.json('/api/admin/login', {
      body: { password }, headers: { 'CF-Connecting-IP': `198.51.100.${n}` },
    });
    // 25 wrong guesses, each from a different address: no single address ever reaches its own limit of 5
    for (let i = 1; i <= 25; i++) expect((await from(i, 'wrong')).status).toBe(401);
    // a brand-new address with the RIGHT password is still refused until the window passes
    const blocked = await from(99, ADMIN_PASSWORD);
    expect(blocked.status).toBe(429);
    expect(admin.jar.has('pg_admin')).toBe(false);
  });

  it('requires the CSRF token to sign out, and the session ends', async () => {
    const csrf = await login();
    expect((await admin.json('/api/admin/logout', { method: 'POST' })).status).toBe(403);
    expect((await admin.json('/api/admin/logout', { method: 'POST', headers: { 'X-CSRF-Token': 'nope' } })).status).toBe(403);
    expect((await admin.json('/api/admin/logout', { method: 'POST', headers: { 'X-CSRF-Token': csrf } })).status).toBe(200);
    expect((await admin.json('/api/admin/me')).status).toBe(401);
  });

  it('expires sessions server-side and never stores the raw token', async () => {
    await login();
    const token = admin.jar.get('pg_admin')!.value;
    expect(JSON.stringify(await sql`select * from playground.admin_sessions`)).not.toContain(token);
    await sql`update playground.admin_sessions set expires_at = now() - interval '1 second'`;
    expect((await admin.json('/api/admin/me')).status).toBe(401);
  });

  it('fails closed when the admin hash is missing or malformed', async () => {
    expect((await new Browser(await makeEnv({ ADMIN_PASSWORD_HASH: '' })).adminLogin()).status).toBe(503);
    expect((await new Browser(await makeEnv({ ADMIN_PASSWORD_HASH: 'not-a-hash' })).adminLogin()).status).toBe(503);
  });

  it('refuses cross-origin admin writes', async () => {
    const r = await admin.json('/api/admin/login', { body: { password: ADMIN_PASSWORD }, headers: { Origin: 'https://evil.example' } });
    expect(r.status).toBe(403);
  });
});

describe('analytics, tester-wise', () => {
  it('matches the database, per tester, mode and quality metric', async () => {
    const asha = await tester('Asha'), ben = await tester('Ben');
    await seed(asha, { evaluation: evalJson({ relevance: 9, language_preservation: 10, chart_consistency: 8 }, 9), evaluation_overall: 9, human_rating: 'good' });
    await seed(asha, { mode: 'kundli_chatbot', provider: 'nvidia', model: 'nvidia/nemotron-3.5-lightning-30b-a3b', total_cost: 0, input_cost: null, output_cost: null,
      evaluation: evalJson({ relevance: 7, language_preservation: 8, chart_consistency: null }, 7.5), evaluation_overall: 7.5 });
    await seed(ben, { status: 'failed', error: 'Rate limit exceeded.', response: null, input_tokens: null, output_tokens: null, total_tokens: null,
      input_cost: null, output_cost: null, total_cost: null, currency: null, human_rating: 'bad' });
    await login();
    const a = (await admin.json('/api/admin/analytics')).data;
    expect(a.totals).toMatchObject({ requests: 3, success: 2, failed: 1, input_tokens: 200, output_tokens: 100, total_tokens: 300, unique_sessions: 2, good: 1, bad: 1 });
    expect(a.totals.cost.USD).toBeCloseTo(0.000155, 9);
    expect(a.totals.avg_score).toBeCloseTo(8.3, 1);

    const byTester = Object.fromEntries(a.by_tester.map((t: any) => [t.tester, t]));
    expect(byTester.Asha).toMatchObject({ requests: 2, success: 2, total_tokens: 300, scored: 2, good: 1 });
    expect(byTester.Asha.avg_score).toBeCloseTo(8.3, 1);
    expect(byTester.Asha.quality).toMatchObject({ relevance: 8, language_preservation: 9, chart_consistency: 8 });
    expect(byTester.Ben).toMatchObject({ requests: 1, failed: 1, total_tokens: 0, bad: 1, avg_score: null });
    expect(Object.fromEntries(a.by_mode.map((m: any) => [m.mode, m.requests]))).toEqual({ direct_provider: 2, kundli_chatbot: 1 });
    expect(a.failures).toEqual([{ error: 'Rate limit exceeded.', count: 1 }]);
    expect(a.sessions).toBe(2);
  });

  it('buckets days in the report timezone (Asia/Kolkata)', async () => {
    const t = await tester('Asha');
    await seed(t, { created_at: new Date('2026-11-15T19:00:00Z') }); // 00:30 on the 16th in India
    await seed(t, { created_at: new Date('2026-11-15T17:00:00Z') }); // 22:30 on the 15th
    await login();
    const days = (await admin.json('/api/admin/analytics')).data.daily.map((d: any) => [d.date, d.requests]);
    expect(days).toEqual([['2026-11-15', 1], ['2026-11-16', 1]]);
  });
});

describe('experiment search and filters', () => {
  it('filters by tester, session, provider, model, mode, status, date and text', async () => {
    const asha = await tester('Asha Rao'), ben = await tester('Ben');
    await seed(asha, { user_prompt: 'career question', created_at: new Date('2026-11-10T10:00:00Z') });
    await seed(asha, { mode: 'kundli_chatbot', provider: 'nvidia', model: 'nvidia/nemotron-3.5-lightning-30b-a3b', user_prompt: 'marriage 100% sure?' });
    await seed(ben, { status: 'failed', error: 'x', provider: 'openrouter', model: 'openai/gpt-4o-mini', user_prompt: 'plain' });
    await login();
    const total = async (qs: string) => (await admin.json(`/api/admin/experiments?${qs}`)).data.total;
    expect(await total('')).toBe(3);
    expect(await total('tester=asha')).toBe(2); // case-insensitive, partial
    expect(await total('tester=Ben')).toBe(1);
    expect(await total(`session_id=${asha.session_id}`)).toBe(2);
    expect(await total(`session_id=${ben.session_id.slice(0, 8)}`)).toBe(1); // prefix
    expect(await total('provider=nvidia')).toBe(1);
    expect(await total('model=openai/gpt-4o-mini')).toBe(1);
    expect(await total('mode=kundli_chatbot')).toBe(1);
    expect(await total('mode=direct_provider&status=success')).toBe(1);
    expect(await total('status=failed')).toBe(1);
    expect(await total('from=2026-11-10&to=2026-11-10')).toBe(1);
    expect(await total('from=2026-11-11')).toBe(2);
    expect(await total('q=career')).toBe(1);
    expect(await total('q=100%25')).toBe(1); // % is literal, not a wildcard
    expect(await total('q=%25')).toBe(1);
    expect(await total('tester=asha&mode=direct_provider')).toBe(1);
  });

  it('validates filters and paginates', async () => {
    const t = await tester('Asha');
    for (let i = 0; i < 5; i++) await seed(t, { user_prompt: `p${i}` });
    await login();
    expect((await admin.json('/api/admin/experiments?from=2026-13-01')).status).toBe(400);
    expect((await admin.json('/api/admin/experiments?from=yesterday')).status).toBe(400);
    expect((await admin.json('/api/admin/experiments?mode=other')).status).toBe(400);
    expect((await admin.json('/api/admin/experiments?status=weird')).status).toBe(400);
    const page = (await admin.json('/api/admin/experiments?page_size=2&page=3')).data;
    expect(page).toMatchObject({ total: 5, page: 3, page_size: 2 });
    expect(page.items).toHaveLength(1);
    expect(page.items[0]).toMatchObject({ tester_name: 'Asha', prompt_preview: 'p0' });
  });

  it('inspects a full experiment (and audits the view) and lists models and filter options', async () => {
    const t = await tester('Asha');
    const id = await seed(t, { system_prompt: 'sys', context: 'ctx', user_prompt: 'q', response: 'a' });
    await login();
    const exp = (await admin.json(`/api/admin/experiments/${id}`)).data;
    expect(exp).toMatchObject({ id, tester_name: 'Asha', session_id: t.session_id, system_prompt: 'sys', context: 'ctx', response: 'a' });
    expect((await admin.json('/api/admin/experiments/99999')).status).toBe(404);
    const logs = (await admin.json('/api/admin/audit-logs')).data.items;
    expect(logs[0]).toMatchObject({ action: 'admin.view_experiment', details: { experiment_id: id } });
    const models = (await admin.json('/api/admin/models')).data;
    expect(models.seen).toMatchObject({ testers: ['Asha'], providers: ['gemini'], modes: ['direct_provider'] });
    expect(models.catalog.find((m: any) => m.provider === 'openrouter').configured).toBe(false);
  });
});

// ── Excel ────────────────────────────────────────────────────────────
const SHEETS = ['All Test Logs', 'Model Summary', 'Token Analysis', 'Cost Analysis', 'Daily Statistics', 'Failed Requests',
  'Feedback & Evaluation', 'Session Summary', 'Tester Summary'];

async function workbook(qs = '') {
  const res = await admin.call(`/api/admin/export/excel${qs}`);
  expect(res.status).toBe(200);
  expect(res.headers.get('content-disposition')).toContain('AI_Playground_Company_Testing_Logs.xlsx');
  const wb = new ExcelJS.Workbook();
  await wb.xlsx.load(await res.arrayBuffer());
  return wb;
}

function table(wb: ExcelJS.Workbook, sheet: string) {
  const ws = wb.getWorksheet(sheet)!;
  const headers = (ws.getRow(4).values as unknown[]).slice(1) as string[];
  const rows: Record<string, unknown>[] = [];
  for (let r = 5; r <= ws.rowCount; r++) {
    const first = ws.getRow(r).getCell(1).value;
    if (first == null || first === 'Total') break;
    rows.push(Object.fromEntries(headers.map((h, i) => [h, ws.getRow(r).getCell(i + 1).value])));
  }
  return { ws, headers, rows };
}

describe('Excel export', () => {
  it('has all nine sheets, every record, and tester names with session ids', async () => {
    const asha = await tester('Asha'), ben = await tester('Ben');
    await seed(asha); await seed(asha, { mode: 'kundli_chatbot', provider: 'nvidia', model: 'nvidia/nemotron-3.5-lightning-30b-a3b' });
    await seed(ben, { status: 'failed', error: 'boom', response: null, input_tokens: null, output_tokens: null, total_tokens: null, total_cost: null, currency: null });
    await login();
    const wb = await workbook();
    expect(wb.worksheets.map((w) => w.name)).toEqual(SHEETS);

    const logs = table(wb, 'All Test Logs');
    expect(logs.rows).toHaveLength(3);
    for (const r of logs.rows) {
      const owner = r.Tester === 'Asha' ? asha : ben;
      expect(r['Session ID']).toBe(owner.session_id);
    }
    expect(logs.rows.map((r) => r['Testing mode'])).toEqual(['Direct provider', 'Kundli Chatbot', 'Direct provider']);
    expect(logs.ws.views[0]).toMatchObject({ state: 'frozen', ySplit: 4 });
    expect(logs.ws.autoFilter).toBeTruthy();

    for (const sheet of ['Failed Requests', 'Feedback & Evaluation', 'Session Summary']) {
      expect(table(wb, sheet).headers, sheet).toEqual(expect.arrayContaining(['Tester', 'Session ID']));
    }
    const failed = table(wb, 'Failed Requests').rows;
    expect(failed).toHaveLength(1);
    expect(failed[0]).toMatchObject({ Tester: 'Ben', 'Session ID': ben.session_id, 'Error details': 'boom' });
  });

  it('summarises per tester and per session with totals that match the database', async () => {
    const asha = await tester('Asha'), ben = await tester('Ben');
    await seed(asha, { evaluation: evalJson({ relevance: 9 }, 9), evaluation_overall: 9, human_rating: 'good' });
    await seed(asha); await seed(ben);
    await login();
    const wb = await workbook();
    const testers = Object.fromEntries(table(wb, 'Tester Summary').rows.map((r) => [r.Tester as string, r]));
    expect(testers.Asha).toMatchObject({ 'Total experiments': 2, 'Total tokens': 300, 'Avg AI score': 9, 'Good ratings': 1, Sessions: 1 });
    expect(testers.Ben).toMatchObject({ 'Total experiments': 1, 'Total tokens': 150 });
    const sessions = Object.fromEntries(table(wb, 'Session Summary').rows.map((r) => [r['Session ID'] as string, r]));
    expect(sessions[asha.session_id]).toMatchObject({ Tester: 'Asha', 'Total experiments': 2, 'Total tokens': 300 });

    const [sum] = await sql`select sum(total_tokens)::int as tokens from playground.experiments`;
    const tokens = sum!.tokens as number;
    const logs = table(wb, 'All Test Logs');
    expect(logs.rows.reduce((n, r) => n + (r['Total tokens'] as number), 0)).toBe(tokens);
    const totalRow = logs.ws.getRow(logs.rows.length + 4 + 2).getCell(18).value as { formula: string; result: number };
    expect(totalRow.formula).toMatch(/^SUBTOTAL\(109,R5:R7\)$/);
    expect(totalRow.result).toBe(tokens);
  });

  it('filters the export by tester, session, mode, model, status and date', async () => {
    const asha = await tester('Asha'), ben = await tester('Ben');
    await seed(asha, { created_at: new Date('2026-11-10T10:00:00Z') });
    await seed(asha, { mode: 'kundli_chatbot', provider: 'nvidia', model: 'nvidia/nemotron-3.5-lightning-30b-a3b' });
    await seed(ben, { status: 'failed', error: 'x' });
    await login();
    const count = async (qs: string) => table(await workbook(qs), 'All Test Logs').rows.length;
    expect(await count('')).toBe(3);
    expect(await count('?tester=Asha')).toBe(2);
    expect(await count(`?session_id=${ben.session_id}`)).toBe(1);
    expect(await count('?mode=kundli_chatbot')).toBe(1);
    expect(await count('?model=gemini-2.5-flash')).toBe(2);
    expect(await count('?status=failed')).toBe(1);
    expect(await count('?from=2026-11-10&to=2026-11-10')).toBe(1);
    const only = table(await workbook('?tester=Ben'), 'Tester Summary').rows;
    expect(only.map((r) => r.Tester)).toEqual(['Ben']);
    expect((await admin.json('/api/admin/export/excel?from=bad')).status).toBe(400);
    expect((await admin.json('/api/admin/audit-logs')).data.items.some((i: any) => i.action === 'admin.export_excel')).toBe(true);
  });

  it('writes untrusted text as plain strings and preserves special characters', async () => {
    const t = await tester('=SUM(1+1)');
    const nasty = '=HYPERLINK("http://evil.example","click")';
    const multiline = 'line one\nline two — नमस्ते 🙏 "quoted" <b>&</b>';
    await seed(t, { user_prompt: nasty, response: multiline, human_feedback: '@cmd|calc', edited_response: '+1+1' });
    await login();
    const wb = await workbook();
    const row = table(wb, 'All Test Logs').rows[0]!;
    expect(row['User prompt']).toBe(nasty);
    expect(row['Generated response']).toBe(multiline);
    expect(row.Tester).toBe('=SUM(1+1)');
    const ws = wb.getWorksheet('All Test Logs')!;
    const cell = ws.getRow(5).getCell(11);
    expect(typeof cell.value).toBe('string');
    expect(cell.type).not.toBe(ExcelJS.ValueType.Formula);
    const fb = table(wb, 'Feedback & Evaluation').rows[0]!;
    expect(fb['Human feedback']).toBe('@cmd|calc');
    expect(fb['Edited response']).toBe('+1+1');
  });

  it('keeps currencies apart and shows missing values as blanks, not zeros', async () => {
    const t = await tester('Asha');
    await seed(t); await seed(t, { currency: 'EUR', total_cost: 1, input_cost: 0.4, output_cost: 0.6 });
    await seed(t, { total_cost: null, input_cost: null, output_cost: null, currency: null, input_tokens: null, output_tokens: null, total_tokens: null });
    await login();
    const wb = await workbook();
    const cost = table(wb, 'Cost Analysis').rows;
    expect(cost.map((r) => r.Currency).sort()).toEqual(['EUR', 'USD']);
    const logs = table(wb, 'All Test Logs').rows;
    expect(logs[2]!['Total tokens']).toBeNull();
    expect(logs[2]!['Total cost']).toBeNull();
  });

  it('strips characters XML cannot hold and caps cell length', async () => {
    expect(safeText('a\u0000b\u001fc')).toBe('abc');
    expect(safeText(null)).toBeNull();
    const long = safeText('x'.repeat(CELL_LIMIT + 500))!;
    expect(long.length).toBe(CELL_LIMIT);
    expect(long.endsWith('exceeds Excel\'s cell limit]')).toBe(true);
    const t = await tester('Asha');
    await seed(t, { response: 'ok\u0007bell' });
    await login();
    expect(table(await workbook(), 'All Test Logs').rows[0]!['Generated response']).toBe('okbell');
  });

  it('is valid and well-formed with no data at all', async () => {
    await login();
    const wb = await workbook();
    expect(wb.worksheets).toHaveLength(9);
    expect(table(wb, 'All Test Logs').rows).toEqual([]);
  });
});
