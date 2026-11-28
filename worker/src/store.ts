/** Every SQL statement the Worker runs. All objects live in the `playground` schema. */
import type { Sql } from './db';

export type Row = Record<string, any>;

export interface SessionRow {
  id: string;
  tester_name: string;
  aicore_session_token: string | null;
  last_seen_at: Date;
}

const SESSION_REFRESH_MS = 5 * 60 * 1000;
const DATE_FIELDS = ['created_at', 'feedback_at'];

/** Timestamps leave this module as epoch seconds, the shape the frontend expects. */
function normalize(row: Row): Row {
  const out: Row = { ...row };
  for (const key of DATE_FIELDS) {
    if (out[key] instanceof Date) out[key] = out[key].getTime() / 1000;
  }
  return out;
}

const json = (sql: Sql, value: unknown) => (value == null ? null : sql.json(value as never));

// ── tester sessions ──────────────────────────────────────────────────
export async function createSession(sql: Sql, tokenHash: string, name: string, ipHash: string | null) {
  const [row] = await sql`
    insert into playground.sessions (token_hash, tester_name, ip_hash)
    values (${tokenHash}, ${name}, ${ipHash})
    returning id, tester_name`;
  return row as { id: string; tester_name: string };
}

export async function getSession(sql: Sql, tokenHash: string, idleHours: number, maxHours: number) {
  const [row] = await sql`
    select id, tester_name, aicore_session_token, last_seen_at
    from playground.sessions
    where token_hash = ${tokenHash} and ended_at is null
      and last_seen_at > now() - (${idleHours}::float8 * interval '1 hour')
      and created_at > now() - (${maxHours}::float8 * interval '1 hour')`;
  if (!row) return null;
  if (Date.now() - (row.last_seen_at as Date).getTime() > SESSION_REFRESH_MS) {
    await sql`update playground.sessions set last_seen_at = now() where id = ${row.id}`;
  }
  return row as unknown as SessionRow;
}

export async function endSession(sql: Sql, id: string) {
  await sql`update playground.sessions set ended_at = now() where id = ${id}`;
}

export async function setAicoreToken(sql: Sql, id: string, token: string) {
  await sql`update playground.sessions set aicore_session_token = ${token} where id = ${id}`;
}

export async function countSessionsSince(sql: Sql, ipHash: string, sinceSec: number) {
  const [row] = await sql`
    select count(*)::int as n from playground.sessions
    where ip_hash = ${ipHash} and created_at >= to_timestamp(${sinceSec})`;
  return row!.n as number;
}

// ── experiments ──────────────────────────────────────────────────────
export interface PendingExperiment {
  request_id: string;
  session_id: string;
  tester_name: string;
  ip_hash: string | null;
  mode: 'kundli_chatbot' | 'direct_provider';
  provider: string;
  model: string;
  model_requested: string | null;
  system_prompt: string | null;
  user_prompt: string;
  context: string | null;
  params: unknown;
  effective_request: unknown;
  pricing_snapshot: unknown;
}

/** Idempotent on request_id: a retry gets the first record back and `created` is false. */
export async function insertPending(sql: Sql, e: PendingExperiment): Promise<{ created: boolean; row: Row }> {
  const values = {
    ...e,
    params: json(sql, e.params),
    effective_request: json(sql, e.effective_request),
    pricing_snapshot: json(sql, e.pricing_snapshot),
    status: 'pending',
  };
  const inserted = await sql`
    insert into playground.experiments ${sql(values as never)}
    on conflict (request_id) do nothing returning *`;
  if (inserted.length) return { created: true, row: normalize(inserted[0]!) };
  const [existing] = await sql`select * from playground.experiments where request_id = ${e.request_id}`;
  return { created: false, row: normalize(existing!) };
}

export async function findByRequestId(sql: Sql, requestId: string): Promise<Row | null> {
  const [row] = await sql`select * from playground.experiments where request_id = ${requestId}`;
  return row ? normalize(row) : null;
}

const FINAL_COLUMNS = [
  'status', 'error', 'provider', 'model', 'response', 'original_response', 'input_tokens', 'output_tokens',
  'total_tokens', 'usage_source', 'input_cost', 'output_cost', 'total_cost', 'currency', 'latency_ms',
  'provider_latency_ms', 'first_token_latency_ms', 'metadata',
] as const;

export async function finalizeExperiment(sql: Sql, id: number, patch: Row): Promise<Row> {
  const values: Row = {};
  for (const key of FINAL_COLUMNS) {
    if (key in patch) values[key] = key === 'metadata' ? json(sql, patch[key]) : patch[key];
  }
  const [row] = await sql`
    update playground.experiments set ${sql(values as never)}
    where id = ${id} returning *`;
  return normalize(row!);
}

/** A pending record that never completed (the Worker was recycled mid-request). */
export async function failStale(sql: Sql, id: number, olderThanSec: number): Promise<Row | null> {
  const [row] = await sql`
    update playground.experiments
    set status = 'failed', error = 'The generation did not complete.', usage_source = 'unavailable'
    where id = ${id} and status = 'pending' and created_at < now() - (${olderThanSec}::float8 * interval '1 second')
    returning *`;
  return row ? normalize(row) : null;
}

export async function setEvaluation(sql: Sql, id: number, evaluation: unknown, overall: number | null) {
  const [row] = await sql`
    update playground.experiments
    set evaluation = ${json(sql, evaluation)}, evaluation_overall = ${overall}
    where id = ${id} returning *`;
  return normalize(row!);
}

export async function setFeedback(
  sql: Sql, id: number, f: { rating: 'good' | 'bad' | null; feedback: string | null; edited: string | null },
) {
  const [row] = await sql`
    update playground.experiments
    set human_rating = ${f.rating}, human_feedback = ${f.feedback}, edited_response = ${f.edited}, feedback_at = now()
    where id = ${id} returning *`;
  return normalize(row!);
}

export async function getExperiment(sql: Sql, id: number): Promise<Row | null> {
  const [row] = await sql`select * from playground.experiments where id = ${id}`;
  return row ? normalize(row) : null;
}

export async function getOwnedExperiment(sql: Sql, id: number, sessionId: string): Promise<Row | null> {
  const [row] = await sql`select * from playground.experiments where id = ${id} and session_id = ${sessionId}`;
  return row ? normalize(row) : null;
}

export async function listForSession(sql: Sql, sessionId: string, limit: number, offset: number) {
  const rows = await sql`
    select id, created_at, session_id, tester_name, mode, provider, model, user_prompt, status, total_tokens,
           total_cost, currency, latency_ms, evaluation_overall, human_rating
    from playground.experiments where session_id = ${sessionId}
    order by id desc limit ${limit} offset ${offset}`;
  return rows.map(normalize);
}

export async function usageSince(sql: Sql, sinceSec: number) {
  const [row] = await sql`
    select count(*)::int as requests,
           coalesce(sum(case when currency = 'USD' then total_cost end), 0)::float8 as cost_usd
    from playground.experiments where created_at >= to_timestamp(${sinceSec})`;
  return { requests: row!.requests as number, costUsd: row!.cost_usd as number };
}

export async function countGenerationsSince(sql: Sql, ipHash: string, sinceSec: number) {
  const [row] = await sql`
    select count(*)::int as n from playground.experiments
    where ip_hash = ${ipHash} and created_at >= to_timestamp(${sinceSec})`;
  return row!.n as number;
}

// ── reference kundlis ────────────────────────────────────────────────
export async function createReferenceKundli(
  sql: Sql, sessionId: string,
  e: { birth: unknown; location_provider: 'aws' | 'manual'; location_place_id: string | null; chart: unknown; engine_source: string },
) {
  // One "current" chart per session: flip any existing one to outdated first
  // (spec: a birth-detail change requires regeneration, the old chart is
  // kept, not deleted) — same transaction-less two-step the rest of this
  // module uses, safe because only the owning session's own requests touch it.
  await sql`update playground.reference_kundlis set status = 'outdated' where session_id = ${sessionId} and status = 'current'`;
  const [row] = await sql`
    insert into playground.reference_kundlis (session_id, birth, location_provider, location_place_id, chart, engine_source)
    values (${sessionId}, ${json(sql, e.birth)}, ${e.location_provider}, ${e.location_place_id}, ${json(sql, e.chart)}, ${e.engine_source})
    returning id, session_id, birth, location_provider, location_place_id, chart, engine_source, status, created_at`;
  return normalize(row!);
}

export async function currentReferenceKundli(sql: Sql, sessionId: string): Promise<Row | null> {
  const [row] = await sql`
    select id, session_id, birth, location_provider, location_place_id, chart, engine_source, status, created_at
    from playground.reference_kundlis where session_id = ${sessionId} and status = 'current'`;
  return row ? normalize(row) : null;
}

export async function getReferenceKundli(sql: Sql, id: string, sessionId: string): Promise<Row | null> {
  const [row] = await sql`
    select id, session_id, birth, location_provider, location_place_id, chart, engine_source, status, created_at
    from playground.reference_kundlis where id = ${id} and session_id = ${sessionId}`;
  return row ? normalize(row) : null;
}

// ── claim verifications ─────────────────────────────────────────────
export async function createClaimVerification(
  sql: Sql,
  v: {
    experiment_id: number; reference_kundli_id: string; claims: unknown;
    supported_count: number; contradicted_count: number; unverifiable_count: number;
    consistency_percent: number | null; evaluator: unknown;
  },
) {
  const [row] = await sql`
    insert into playground.claim_verifications
      (experiment_id, reference_kundli_id, claims, supported_count, contradicted_count, unverifiable_count, consistency_percent, evaluator)
    values (${v.experiment_id}, ${v.reference_kundli_id}, ${json(sql, v.claims)}, ${v.supported_count},
            ${v.contradicted_count}, ${v.unverifiable_count}, ${v.consistency_percent}, ${json(sql, v.evaluator)})
    returning *`;
  return normalize(row!);
}

export async function getClaimVerificationForExperiment(sql: Sql, experimentId: number): Promise<Row | null> {
  const [row] = await sql`select * from playground.claim_verifications where experiment_id = ${experimentId}`;
  return row ? normalize(row) : null;
}

export async function verificationSummary(sql: Sql, since?: number, until?: number) {
  const from = since ?? 0;
  const to = until ?? 9_999_999_999;
  const [refs] = await sql`
    select count(*)::int as total_charts from playground.reference_kundlis
    where created_at >= to_timestamp(${from}) and created_at < to_timestamp(${to})`;
  const [verifications] = await sql`
    select count(*)::int as total_verifications,
           coalesce(sum(supported_count), 0)::int as supported,
           coalesce(sum(contradicted_count), 0)::int as contradicted,
           coalesce(sum(unverifiable_count), 0)::int as unverifiable,
           avg(consistency_percent)::float8 as avg_consistency
    from playground.claim_verifications
    where created_at >= to_timestamp(${from}) and created_at < to_timestamp(${to})`;
  const byModel = await sql`
    select e.provider, e.model, count(*)::int as verifications,
           coalesce(sum(cv.supported_count), 0)::int as supported,
           coalesce(sum(cv.contradicted_count), 0)::int as contradicted,
           avg(cv.consistency_percent)::float8 as avg_consistency
    from playground.claim_verifications cv
    join playground.experiments e on e.id = cv.experiment_id
    where cv.created_at >= to_timestamp(${from}) and cv.created_at < to_timestamp(${to})
    group by e.provider, e.model order by e.provider, e.model`;
  return {
    totalCharts: refs!.total_charts as number,
    totalVerifications: verifications!.total_verifications as number,
    supported: verifications!.supported as number,
    contradicted: verifications!.contradicted as number,
    unverifiable: verifications!.unverifiable as number,
    avgConsistency: verifications!.avg_consistency as number | null,
    byModel: byModel.map((r) => ({
      provider: r.provider as string, model: r.model as string, verifications: r.verifications as number,
      supported: r.supported as number, contradicted: r.contradicted as number, avgConsistency: r.avg_consistency as number | null,
    })),
  };
}

export async function referenceKundlisForExport(sql: Sql, f: { since?: number; until?: number }): Promise<Row[]> {
  const parts = [];
  if (f.since != null) parts.push(sql`rk.created_at >= to_timestamp(${f.since})`);
  if (f.until != null) parts.push(sql`rk.created_at < to_timestamp(${f.until})`);
  const where = parts.length ? sql`where ${parts.reduce((a, b) => sql`${a} and ${b}`)}` : sql``;
  const rows = await sql`
    select rk.id, rk.session_id, s.tester_name, rk.birth, rk.location_provider, rk.engine_source, rk.status, rk.created_at
    from playground.reference_kundlis rk
    join playground.sessions s on s.id = rk.session_id
    ${where}
    order by rk.created_at asc`;
  return rows.map(normalize);
}

export async function claimVerificationsForExport(sql: Sql, f: { since?: number; until?: number }): Promise<Row[]> {
  const parts = [];
  if (f.since != null) parts.push(sql`cv.created_at >= to_timestamp(${f.since})`);
  if (f.until != null) parts.push(sql`cv.created_at < to_timestamp(${f.until})`);
  const where = parts.length ? sql`where ${parts.reduce((a, b) => sql`${a} and ${b}`)}` : sql``;
  const rows = await sql`
    select cv.id, cv.experiment_id, cv.reference_kundli_id, e.tester_name, e.provider, e.model, cv.claims,
           cv.supported_count, cv.contradicted_count, cv.unverifiable_count, cv.consistency_percent,
           cv.evaluator, cv.created_at
    from playground.claim_verifications cv
    join playground.experiments e on e.id = cv.experiment_id
    ${where}
    order by cv.created_at asc`;
  return rows.map(normalize);
}

// ── admin queries ────────────────────────────────────────────────────
export interface ExperimentFilters {
  since?: number;
  until?: number;
  provider?: string;
  model?: string;
  status?: string;
  mode?: string;
  tester?: string;
  session_id?: string;
  q?: string;
}

const escapeLike = (s: string) => s.replace(/[\\%_]/g, (c) => '\\' + c);

function whereClause(sql: Sql, f: ExperimentFilters) {
  const parts = [];
  if (f.since != null) parts.push(sql`created_at >= to_timestamp(${f.since})`);
  if (f.until != null) parts.push(sql`created_at < to_timestamp(${f.until})`);
  if (f.provider) parts.push(sql`provider = ${f.provider}`);
  if (f.model) parts.push(sql`model = ${f.model}`);
  if (f.status) parts.push(sql`status = ${f.status}`);
  if (f.mode) parts.push(sql`mode = ${f.mode}`);
  if (f.tester) parts.push(sql`tester_name ilike ${'%' + escapeLike(f.tester) + '%'} escape '\\'`);
  if (f.session_id) parts.push(sql`session_id::text ilike ${escapeLike(f.session_id) + '%'} escape '\\'`);
  if (f.q) {
    const needle = '%' + escapeLike(f.q) + '%';
    parts.push(sql`(user_prompt ilike ${needle} escape '\\' or response ilike ${needle} escape '\\'
                    or tester_name ilike ${needle} escape '\\')`);
  }
  if (!parts.length) return sql``;
  return sql`where ${parts.reduce((a, b) => sql`${a} and ${b}`)}`;
}

export async function searchExperiments(sql: Sql, f: ExperimentFilters, limit: number, offset: number) {
  const where = whereClause(sql, f);
  const [counted] = await sql`select count(*)::int as total from playground.experiments ${where}`;
  const total = counted!.total as number;
  const rows = await sql`
    select id, created_at, request_id, session_id, tester_name, mode, provider, model,
           left(user_prompt, 160) as prompt_preview, left(response, 160) as response_preview,
           input_tokens, output_tokens, total_tokens, total_cost, currency, latency_ms, status, error,
           evaluation_overall, human_rating
    from playground.experiments ${where} order by id desc limit ${limit} offset ${offset}`;
  return { items: rows.map(normalize), total };
}

export async function allExperiments(sql: Sql, f: ExperimentFilters): Promise<Row[]> {
  const rows = await sql`select * from playground.experiments ${whereClause(sql, f)} order by id asc`;
  return rows.map(normalize);
}

export async function distinctValues(sql: Sql) {
  const [providers, models, testers, modes] = await Promise.all([
    sql`select distinct provider as v from playground.experiments order by 1`,
    sql`select distinct model as v from playground.experiments order by 1`,
    sql`select distinct tester_name as v from playground.experiments order by 1 limit 500`,
    sql`select distinct mode as v from playground.experiments order by 1`,
  ]);
  const vals = (r: Row[]) => r.map((x) => x.v as string);
  return { providers: vals(providers), models: vals(models), testers: vals(testers), modes: vals(modes) };
}

// ── admin sessions and audit log ─────────────────────────────────────
export async function createAdminSession(sql: Sql, tokenHash: string, expiresAtSec: number) {
  await sql`delete from playground.admin_sessions where expires_at < now()`;
  await sql`insert into playground.admin_sessions (session_token_hash, expires_at)
            values (${tokenHash}, to_timestamp(${expiresAtSec}))`;
}

export async function adminSessionValid(sql: Sql, tokenHash: string) {
  const rows = await sql`select 1 from playground.admin_sessions
                         where session_token_hash = ${tokenHash} and expires_at > now()`;
  return rows.length > 0;
}

export async function deleteAdminSession(sql: Sql, tokenHash: string) {
  await sql`delete from playground.admin_sessions where session_token_hash = ${tokenHash}`;
}

export async function audit(sql: Sql, action: string, ip: string | null, details: Row = {}) {
  await sql`insert into playground.audit_logs (action, ip, details)
            values (${action}, ${ip}, ${sql.json(details as never)})`;
}

export async function auditLogs(sql: Sql, limit: number, offset: number) {
  const [counted] = await sql`select count(*)::int as total from playground.audit_logs`;
  const total = counted!.total as number;
  const rows = await sql`select id, action, extract(epoch from timestamp)::float8 as timestamp, ip, details
                         from playground.audit_logs order by id desc limit ${limit} offset ${offset}`;
  return { items: rows as Row[], total };
}

/** Failed admin logins since a time; from one address, or (ip = null) from anywhere. */
export async function countRecentFailedLogins(sql: Sql, ip: string | null, sinceSec: number) {
  const [row] = await sql`select count(*)::int as n from playground.audit_logs
                          where action = 'admin.login_failed' and (${ip}::text is null or ip = ${ip})
                            and timestamp >= to_timestamp(${sinceSec})`;
  return row!.n as number;
}

export async function ping(sql: Sql) {
  await sql`select 1`;
}
