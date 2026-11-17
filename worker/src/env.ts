import type { Sql } from './db';

export interface Env {
  // Cloudflare only: static assets binding. On Node (Render) Express serves the built frontend instead.
  ASSETS?: Fetcher;
  // Node only: one shared connection pool created at startup. Without it each request opens its own connection.
  SQL_POOL?: Sql;
  // Node only: size of that pool (default 5; Supabase's free pooler has a small connection budget).
  DB_POOL_MAX?: string;
  // Cloudflare Hyperdrive in front of Supabase (wrangler.jsonc "hyperdrive"). Workers verify TLS
  // certificates and Supabase's pooler uses its own CA, so a deployed Worker must go through Hyperdrive.
  HYPERDRIVE?: Hyperdrive;
  // Direct connection string: tests and local tooling only.
  DATABASE_URL?: string;
  // Secrets (wrangler secret put ...)
  SESSION_SECRET: string;
  ADMIN_PASSWORD_HASH?: string;
  AI_CORE_URL?: string;
  NVIDIA_API_KEY?: string;
  GEMINI_API_KEY?: string;
  OPENROUTER_API_KEY?: string;
  // Plain variables (wrangler.jsonc "vars"); every one is optional.
  PLAYGROUND_RATE_LIMIT_PER_MINUTE?: string;
  PLAYGROUND_DAILY_REQUEST_LIMIT?: string;
  PLAYGROUND_DAILY_COST_LIMIT_USD?: string;
  PLAYGROUND_MAX_PROMPT_CHARS?: string;
  PLAYGROUND_REPORT_TIMEZONE?: string;
  PLAYGROUND_LLM_TIMEOUT_SECONDS?: string;
  TESTER_SESSION_IDLE_HOURS?: string;
  TESTER_SESSION_MAX_HOURS?: string;
  ADMIN_SESSION_HOURS?: string;
}

export interface Settings {
  rateLimitPerMinute: number;
  dailyRequestLimit: number;
  dailyCostLimitUsd: number;
  maxPromptChars: number;
  reportTimezone: string;
  llmTimeoutMs: number;
  testerIdleHours: number;
  testerMaxHours: number;
  adminSessionHours: number;
  maxBodyBytes: number;
}

function num(raw: string | undefined, fallback: number): number {
  const n = Number(raw);
  return raw !== undefined && raw.trim() !== '' && Number.isFinite(n) ? n : fallback;
}

export function loadSettings(env: Env): Settings {
  return {
    rateLimitPerMinute: num(env.PLAYGROUND_RATE_LIMIT_PER_MINUTE, 20),
    dailyRequestLimit: num(env.PLAYGROUND_DAILY_REQUEST_LIMIT, 1000),
    dailyCostLimitUsd: num(env.PLAYGROUND_DAILY_COST_LIMIT_USD, 5),
    maxPromptChars: num(env.PLAYGROUND_MAX_PROMPT_CHARS, 20000),
    reportTimezone: env.PLAYGROUND_REPORT_TIMEZONE?.trim() || 'Asia/Kolkata',
    llmTimeoutMs: num(env.PLAYGROUND_LLM_TIMEOUT_SECONDS, 90) * 1000,
    testerIdleHours: num(env.TESTER_SESSION_IDLE_HOURS, 12),
    testerMaxHours: num(env.TESTER_SESSION_MAX_HOURS, 24),
    adminSessionHours: num(env.ADMIN_SESSION_HOURS, 12),
    maxBodyBytes: 1_000_000,
  };
}
