/**
 * Aggregations shared by the admin analytics API and the Excel export, so both
 * always show identical numbers. Days are bucketed in the report timezone, and
 * costs are kept per currency; different currencies are never summed.
 */
import { METRICS } from './evaluator';
import type { ExperimentFilters, Row } from './store';
import { addDays, dayKey, dayStart } from './time';

export const STATUSES = ['success', 'failed', 'pending'] as const;
export const MODES = ['kundli_chatbot', 'direct_provider'] as const;
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

export class FilterError extends Error {}

export interface RawFilters {
  from?: string; to?: string; provider?: string; model?: string; status?: string; mode?: string;
  tester?: string; session_id?: string; q?: string;
}

export function parseFilters(raw: RawFilters, tz: string): ExperimentFilters {
  const f: ExperimentFilters = {};
  for (const [label, value, key, shift] of [['from', raw.from, 'since', 0], ['to', raw.to, 'until', 1]] as const) {
    if (!value) continue;
    if (!DATE_RE.test(value) || Number.isNaN(Date.parse(value + 'T00:00:00Z')) ||
        new Date(value + 'T00:00:00Z').toISOString().slice(0, 10) !== value) {
      throw new FilterError(`'${label}' must be a valid YYYY-MM-DD date.`);
    }
    f[key] = dayStart(addDays(value, shift), tz);
  }
  if (raw.status && !(STATUSES as readonly string[]).includes(raw.status)) throw new FilterError("status must be 'success', 'failed' or 'pending'.");
  if (raw.mode && !(MODES as readonly string[]).includes(raw.mode)) throw new FilterError("mode must be 'kundli_chatbot' or 'direct_provider'.");
  for (const key of ['provider', 'model', 'status', 'mode', 'tester', 'session_id', 'q'] as const) {
    const v = raw[key]?.trim();
    if (v) f[key] = v.slice(0, 200);
  }
  return f;
}

export function describeFilters(raw: RawFilters): string {
  const shown = Object.entries(raw).filter(([, v]) => v);
  return 'filters: ' + (shown.length ? shown.map(([k, v]) => `${k}=${v}`).join(', ') : 'none (all records)');
}

type Costs = Record<string, number>;
interface Acc {
  requests: number; success: number; failed: number; pending: number;
  input_tokens: number; output_tokens: number; total_tokens: number;
  cost: Costs; latency_sum: number; latency_n: number; eval_sum: number; eval_n: number;
  good: number; bad: number; sessions: Set<string>; models: Set<string>; metrics: Record<string, number[]>;
}

const newAcc = (): Acc => ({
  requests: 0, success: 0, failed: 0, pending: 0, input_tokens: 0, output_tokens: 0, total_tokens: 0, cost: {},
  latency_sum: 0, latency_n: 0, eval_sum: 0, eval_n: 0, good: 0, bad: 0, sessions: new Set(), models: new Set(), metrics: {},
});

const addCost = (bucket: Costs, row: Row, field = 'total_cost') => {
  if (row[field] != null && row.currency) bucket[row.currency] = (bucket[row.currency] ?? 0) + Number(row[field]);
};

function feed(a: Acc, r: Row) {
  a.requests++;
  if (r.status === 'success') a.success++;
  else if (r.status === 'pending') a.pending++;
  else a.failed++;
  a.input_tokens += r.input_tokens ?? 0;
  a.output_tokens += r.output_tokens ?? 0;
  a.total_tokens += r.total_tokens ?? 0;
  addCost(a.cost, r);
  if (r.latency_ms != null) { a.latency_sum += r.latency_ms; a.latency_n++; }
  if (r.evaluation_overall != null) { a.eval_sum += Number(r.evaluation_overall); a.eval_n++; }
  if (r.human_rating === 'good') a.good++;
  if (r.human_rating === 'bad') a.bad++;
  a.sessions.add(r.session_id);
  a.models.add(`${r.provider}/${r.model}`);
  const scores = r.evaluation?.scores ?? {};
  for (const m of METRICS) if (typeof scores[m] === 'number') (a.metrics[m] ??= []).push(scores[m]);
}

const round = (n: number, d = 1) => Math.round(n * 10 ** d) / 10 ** d;

function finish(a: Acc) {
  const costs: Costs = {};
  for (const [k, v] of Object.entries(a.cost)) costs[k] = round(v, 8);
  const quality: Record<string, number> = {};
  for (const [m, vals] of Object.entries(a.metrics)) quality[m] = round(vals.reduce((x, y) => x + y, 0) / vals.length);
  return {
    requests: a.requests, success: a.success, failed: a.failed, pending: a.pending,
    input_tokens: a.input_tokens, output_tokens: a.output_tokens, total_tokens: a.total_tokens, cost: costs,
    avg_latency_ms: a.latency_n ? Math.round(a.latency_sum / a.latency_n) : null,
    avg_score: a.eval_n ? round(a.eval_sum / a.eval_n) : null, scored: a.eval_n,
    good: a.good, bad: a.bad, unique_sessions: a.sessions.size, quality,
  };
}

export type Summary = ReturnType<typeof finish>;

export function aggregate(rows: Row[], tz: string) {
  const total = newAcc();
  const byModel = new Map<string, Acc & { provider: string; model: string }>();
  const byProvider = new Map<string, Acc>();
  const byMode = new Map<string, Acc>();
  const byTester = new Map<string, Acc>();
  const daily = new Map<string, Acc>();
  const dailyModel = new Map<string, Acc & { input_cost: Costs; output_cost: Costs; date: string; provider: string; model: string }>();
  const sessions = new Map<string, Acc & { tester_name: string | null; first: number; last: number }>();
  const failures = new Map<string, number>();

  const get = <T>(map: Map<string, T>, key: string, make: () => T): T => {
    let v = map.get(key);
    if (!v) { v = make(); map.set(key, v); }
    return v;
  };

  for (const r of rows) {
    const day = dayKey(r.created_at, tz);
    const targets = [
      total,
      get(byModel, `${r.provider}\u0000${r.model}`, () => ({ ...newAcc(), provider: r.provider, model: r.model })),
      get(byProvider, r.provider, newAcc),
      get(byMode, r.mode, newAcc),
      get(byTester, r.tester_name, newAcc),
      get(daily, day, newAcc),
    ];
    for (const t of targets) feed(t, r);

    const dm = get(dailyModel, `${day}\u0000${r.provider}\u0000${r.model}`,
      () => ({ ...newAcc(), input_cost: {}, output_cost: {}, date: day, provider: r.provider, model: r.model }));
    feed(dm, r);
    addCost(dm.input_cost, r, 'input_cost');
    addCost(dm.output_cost, r, 'output_cost');

    const s = get(sessions, r.session_id, () => ({ ...newAcc(), tester_name: r.tester_name as string | null, first: r.created_at as number, last: r.created_at as number }));
    feed(s, r);
    s.first = Math.min(s.first, r.created_at);
    s.last = Math.max(s.last, r.created_at);

    if (r.status === 'failed') {
      const reason = String(r.error ?? 'Unknown error').slice(0, 160);
      failures.set(reason, (failures.get(reason) ?? 0) + 1);
    }
  }

  const byRequests = <T extends { requests: number }>(a: T[]) => a.sort((x, y) => y.requests - x.requests);
  const costMap = (c: Costs) => Object.fromEntries(Object.entries(c).map(([k, v]) => [k, round(v, 8)]));
  return {
    totals: finish(total),
    by_model: byRequests([...byModel.values()].map((a) => ({ provider: a.provider, model: a.model, ...finish(a) }))),
    by_provider: byRequests([...byProvider].map(([provider, a]) => ({ provider, ...finish(a) }))),
    by_mode: byRequests([...byMode].map(([mode, a]) => ({ mode, ...finish(a) }))),
    by_tester: byRequests([...byTester].map(([tester, a]) => ({ tester, ...finish(a) }))),
    daily: [...daily].sort(([a], [b]) => a.localeCompare(b)).map(([date, a]) => ({ date, ...finish(a) })),
    daily_model: [...dailyModel.values()].sort((a, b) => (a.date + a.model).localeCompare(b.date + b.model))
      .map((a) => ({ date: a.date, provider: a.provider, model: a.model, ...finish(a),
        input_cost: costMap(a.input_cost), output_cost: costMap(a.output_cost) })),
    failures: [...failures].sort((a, b) => b[1] - a[1]).slice(0, 20).map(([error, count]) => ({ error, count })),
    sessions: [...sessions].sort(([, a], [, b]) => a.first - b.first).map(([session_id, a]) => ({
      session_id, tester_name: a.tester_name, ...finish(a), models: [...a.models].sort(), first: a.first, last: a.last,
    })),
  };
}
