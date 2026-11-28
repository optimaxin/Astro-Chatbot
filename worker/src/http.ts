import type { Context, MiddlewareHandler } from 'hono';
import { getCookie } from 'hono/cookie';
import { HTTPException } from 'hono/http-exception';
import { z } from 'zod';
import { hmacHex, sha256Hex } from './crypto';
import type { Sql } from './db';
import type { Env, Settings } from './env';
import * as store from './store';

export type Vars = { sql: Sql; settings: Settings; session: store.SessionRow };
export type AppEnv = { Bindings: Env; Variables: Vars };
export type Ctx = Context<AppEnv>;

export const SESSION_COOKIE = 'pg_session';

export const fail = (status: 400 | 401 | 403 | 404 | 409 | 413 | 422 | 429 | 502 | 503, detail: string) =>
  new HTTPException(status, { message: detail });

/** True on https, including behind a TLS-terminating proxy (Render sets X-Forwarded-Proto). */
export const isHttps = (c: Ctx) =>
  new URL(c.req.url).protocol === 'https:' || c.req.header('X-Forwarded-Proto')?.split(',')[0]?.trim() === 'https';
/** The origin the browser used to reach us, for the cross-origin check. */
export const requestOrigin = (c: Ctx) => `${isHttps(c) ? 'https' : 'http'}://${new URL(c.req.url).host}`;
/**
 * The caller's address, as reported by the hosting edge. These headers are only as trustworthy as the proxy in
 * front of the service, so a forged one can at most dodge the per-address limits; the daily caps and the global
 * admin-login ceiling do not depend on it.
 */
export const clientIp = (c: Ctx) =>
  c.req.header('CF-Connecting-IP') ?? c.req.header('True-Client-IP') ??
  c.req.header('X-Forwarded-For')?.split(',')[0]?.trim() ?? 'unknown';
export const ipHash = (c: Ctx) => hmacHex(c.env.SESSION_SECRET, 'ip:' + clientIp(c));

/** Reads a JSON body without trusting Content-Length alone. */
export async function readJson(c: Ctx): Promise<unknown> {
  const max = c.get('settings').maxBodyBytes;
  const declared = Number(c.req.header('content-length') ?? 0);
  if (declared > max) throw fail(413, 'Request body too large.');
  const text = await c.req.text();
  if (text.length > max) throw fail(413, 'Request body too large.');
  try {
    return JSON.parse(text);
  } catch {
    throw fail(400, 'Request body must be valid JSON.');
  }
}

export function parse<T extends z.ZodType>(schema: T, value: unknown): z.infer<T> {
  const result = schema.safeParse(value);
  if (!result.success) {
    const issue = result.error.issues[0]!;
    throw fail(422, `${issue.path.join('.') || 'body'}: ${issue.message}`);
  }
  return result.data;
}

/** The tester's server-side session. Identity comes from the database, never from the request body. */
export const requireSession: MiddlewareHandler<AppEnv> = async (c, next) => {
  const token = getCookie(c, SESSION_COOKIE);
  if (token) {
    const s = c.get('settings');
    const session = await store.getSession(c.get('sql'), await sha256Hex(token), s.testerIdleHours, s.testerMaxHours);
    if (session) {
      c.set('session', session);
      return next();
    }
  }
  throw fail(401, 'Please enter your name to start testing.');
};

// ── validation ───────────────────────────────────────────────────────
export const NAME_MAX = 80;

export type NameCheck = { ok: true; name: string } | { ok: false; error: string };

export function validateName(raw: unknown): NameCheck {
  if (typeof raw !== 'string') return { ok: false, error: 'Enter your name.' };
  // Trim, normalise and collapse inner whitespace so "  Asha   Rao " is one stable name.
  const name = raw.normalize('NFC').trim().replace(/\s+/gu, ' ');
  if (!name) return { ok: false, error: 'Enter your name.' };
  if ([...name].length > NAME_MAX) return { ok: false, error: `Use at most ${NAME_MAX} characters.` };
  // Control characters only: joiners used by Indic scripts are legitimate in names.
  if (/[\p{Cc}\u2028\u2029]/u.test(name)) return { ok: false, error: 'The name contains characters that are not allowed.' };
  if (!/[\p{L}\p{N}]/u.test(name)) return { ok: false, error: 'Use at least one letter or number.' };
  return { ok: true, name };
}

const noLineBreaks = (s: string) => s.replace(/[\r\n]+/g, ' ').trim();
const ID = /^[A-Za-z0-9_-]{8,64}$/;

export const birthSchema = z.object({
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, 'date must be YYYY-MM-DD'),
  time: z.string().regex(/^([01]\d|2[0-3]):[0-5]\d$/, 'time must be HH:MM (24h)'),
  timezoneOffsetMinutes: z.number().int().min(-900).max(900),
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
  name: z.string().max(80).transform(noLineBreaks).optional(),
  placeLabel: z.string().max(120).transform(noLineBreaks).optional(),
});

export const paramsSchema = z.object({
  temperature: z.number().min(0).max(2).optional(),
  top_p: z.number().gt(0).max(1).optional(),
  top_k: z.number().int().min(1).max(500).optional(),
  max_tokens: z.number().int().min(1).max(32000).optional(),
}).strict().default({});

export function chatSchema(maxChars: number) {
  const text = z.string().max(maxChars);
  return z.object({
    mode: z.enum(['kundli_chatbot', 'direct_provider']),
    message: text.refine((v) => v.trim().length > 0, 'message must not be blank'),
    request_id: z.string().regex(ID, 'request_id is malformed'),
    conversation_id: z.string().regex(ID, 'conversation_id is malformed'),
    style: z.string().regex(/^[a-z0-9_]{1,40}$/).nullish(),
    birth: birthSchema.optional(),
    provider: z.string().max(64).optional(),
    model: z.string().max(200).optional(),
    system_prompt: text.nullish(),
    context: text.nullish(),
    history: z.array(z.object({ role: z.enum(['user', 'assistant']), content: text })).max(60).default([]),
    params: paramsSchema,
  });
}

export const locationResolveSchema = z.object({ placeId: z.string().min(1).max(500) });

export const kundliGenerateSchema = z.object({
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, 'date must be YYYY-MM-DD'),
  time: z.string().regex(/^([01]\d|2[0-3]):[0-5]\d$/, 'time must be HH:MM (24h)'),
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
  placeLabel: z.string().max(120).transform(noLineBreaks).optional(),
  placeId: z.string().max(500).optional(),
});

export const evaluateSchema = z.object({ experiment_id: z.number().int().min(1) });

export const feedbackSchema = (maxChars: number) => z.object({
  experiment_id: z.number().int().min(1),
  rating: z.enum(['good', 'bad']).nullable().optional(),
  feedback: z.string().max(4000).nullish(),
  edited_response: z.string().max(maxChars).nullish(),
});
