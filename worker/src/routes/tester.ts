import { Hono } from 'hono';
import { deleteCookie, getCookie, setCookie } from 'hono/cookie';
import { z } from 'zod';
import { calculateCost, catalog, findModel, findProvider, providerConfigured, type Birth } from '../catalog';
import { randomToken, sha256Hex } from '../crypto';
import { connect, type Sql } from '../db';
import type { Env, Settings } from '../env';
import { evaluate } from '../evaluator';
import {
  chatSchema, evaluateSchema, fail, feedbackSchema, ipHash, isHttps, parse, readJson, requireSession,
  SESSION_COOKIE, validateName, type AppEnv, type Ctx,
} from '../http';
import {
  chatbotStyles, generateChatbot, generateDirect, ProviderError, type Generation, type Message,
} from '../providers';
import * as store from '../store';
import { startOfToday } from '../time';

export const session = new Hono<AppEnv>();
export const playground = new Hono<AppEnv>();

// ── session: name entry ──────────────────────────────────────────────
const SESSIONS_PER_IP_PER_HOUR = 60;

session.post('/', async (c) => {
  const body = parse(z.object({ name: z.unknown() }), await readJson(c));
  const check = validateName(body.name);
  if (!check.ok) throw fail(422, check.error);

  const sql = c.get('sql');
  const hash = await ipHash(c);
  if (await store.countSessionsSince(sql, hash, Date.now() / 1000 - 3600) >= SESSIONS_PER_IP_PER_HOUR) {
    throw fail(429, 'Too many new sessions from this connection. Try again later.');
  }
  const token = randomToken();
  const created = await store.createSession(sql, await sha256Hex(token), check.name, hash);
  // No Max-Age/Expires: a browser-session cookie, so closing the browser ends the session.
  setCookie(c, SESSION_COOKIE, token, { httpOnly: true, secure: isHttps(c), sameSite: 'Lax', path: '/api' });
  return c.json({ session_id: created.id, tester_name: created.tester_name }, 201);
});

session.get('/', requireSession, (c) => {
  const s = c.get('session');
  return c.json({ session_id: s.id, tester_name: s.tester_name });
});

session.delete('/', async (c) => {
  const token = getCookie(c, SESSION_COOKIE);
  if (token) {
    const s = await store.getSession(c.get('sql'), await sha256Hex(token), 1e6, 1e6);
    if (s) await store.endSession(c.get('sql'), s.id);
  }
  deleteCookie(c, SESSION_COOKIE, { path: '/api', secure: isHttps(c) });
  return c.json({ ok: true });
});

// ── limits ───────────────────────────────────────────────────────────
async function enforceLimits(c: Ctx, hash: string) {
  const sql = c.get('sql');
  const s = c.get('settings');
  if (s.rateLimitPerMinute > 0 && await store.countGenerationsSince(sql, hash, Date.now() / 1000 - 60) >= s.rateLimitPerMinute) {
    throw fail(429, `Playground rate limit reached (${s.rateLimitPerMinute} generations/minute).`);
  }
  await spendGuard(sql, s);
}

async function spendGuard(sql: Sql, s: Settings) {
  if (s.dailyRequestLimit <= 0 && s.dailyCostLimitUsd <= 0) return;
  const used = await store.usageSince(sql, startOfToday(s.reportTimezone) );
  if (s.dailyRequestLimit > 0 && used.requests >= s.dailyRequestLimit) {
    throw fail(429, "The playground's daily request limit has been reached. Try again tomorrow.");
  }
  if (s.dailyCostLimitUsd > 0 && used.costUsd >= s.dailyCostLimitUsd) {
    throw fail(429, "The playground's daily spending limit has been reached. Try again tomorrow.");
  }
}

// ── catalogue ────────────────────────────────────────────────────────
playground.use('*', requireSession);

playground.get('/models', async (c) => {
  const env = c.env;
  const s = c.get('settings');
  return c.json({
    modes: [
      { id: 'kundli_chatbot', label: catalog.chatbot.label, available: Boolean(env.AI_CORE_URL?.trim()) },
      { id: 'direct_provider', label: 'Direct model' },
    ],
    chatbot: { styles: await chatbotStyles(env, catalog.chatbot.fallback_styles), sample_birth: catalog.chatbot.sample_birth },
    providers: catalog.providers.map((p) => ({
      id: p.id, name: p.name, configured: providerConfigured(env, p), supported_params: p.supported_params,
    })),
    models: catalog.models.map((m) => ({ id: m.id, provider: m.provider, label: m.label, pricing: m.pricing })),
    evaluator: catalog.evaluator,
    limits: { rate_limit_per_minute: s.rateLimitPerMinute, max_prompt_chars: s.maxPromptChars },
  });
});

// ── chat: one experiment per generation ──────────────────────────────
const estimateTokens = (t: string | null | undefined) => Math.round((t ?? '').length / 4);
const STALE_PENDING_SECONDS = 150;

playground.post('/chat', async (c) => {
  const started = Date.now();
  const s = c.get('settings');
  const sess = c.get('session');
  const sql = c.get('sql');
  const body = parse(chatSchema(s.maxPromptChars), await readJson(c));
  const hash = await ipHash(c);

  // A retry of a request already recorded never generates again.
  const prior = await store.findByRequestId(sql, body.request_id);
  if (prior) return c.json(await resolvePrior(sql, prior, sess.id));

  await enforceLimits(c, hash);

  const base = {
    request_id: body.request_id, session_id: sess.id, tester_name: sess.tester_name, ip_hash: hash,
    mode: body.mode, user_prompt: body.message,
  } as const;
  let pending: store.PendingExperiment;
  let run: (sql: Sql) => Promise<Partial<Record<string, unknown>>>;

  if (body.mode === 'kundli_chatbot') {
    if (!c.env.AI_CORE_URL?.trim()) throw fail(503, 'The Kundli chatbot is not configured on this playground.');
    const birth: Birth = body.birth ?? catalog.chatbot.sample_birth;
    const style = body.style && body.style !== 'auto' ? body.style : null;
    pending = {
      ...base, provider: 'unknown', model: 'unknown', model_requested: style ?? 'auto', system_prompt: null,
      context: JSON.stringify(birth), params: { response_style: style ?? 'auto' }, pricing_snapshot: null,
      effective_request: {
        mode: 'kundli_chatbot', endpoint: 'POST /chat', response_style: style ?? 'auto', chat_bubbles: true, birth,
        conversation: `playground-${body.conversation_id}`,
        messages: [{ role: 'user', content: body.message }],
      },
    };
    run = async (db) => {
      const g = await generateChatbot(c.env, {
        question: body.message, birth, style, conversationId: body.conversation_id, sessionToken: sess.aicore_session_token,
      }, s.llmTimeoutMs);
      if (g.newSessionToken) await store.setAicoreToken(db, sess.id, g.newSessionToken);
      const degraded = Boolean(g.extra.degraded);
      return {
        ...success(g, started),
        error: !g.text ? 'The chatbot returned an empty response.'
          : degraded ? 'The chatbot could not reach its model and returned its fallback reply.' : null,
        status: !g.text || degraded ? 'failed' : 'success',
        provider: String(g.extra.underlying_provider), model: g.model,
        // The chatbot prices its own answer generation; it exposes no input/output split.
        input_cost: null, output_cost: null, total_cost: g.providerCost, currency: g.providerCost != null ? 'USD' : null,
        first_token_latency_ms: null,
        metadata: {
          provider_name: 'Kundli Chatbot', model_reported: g.model, cost_source: 'chatbot estimate (answer generation)',
          ttft_note: 'Not available: the chatbot validates the whole reply before sending it.',
          overhead_ms: Date.now() - started - g.providerLatencyMs,
          ai_core: { ...g.extra, requested_style: style, birth },
        },
      };
    };
  } else {
    const provider = body.provider ? findProvider(body.provider) : undefined;
    const model = provider && body.model ? findModel(provider.id, body.model) : undefined;
    if (!provider || !model) throw fail(400, `Unknown model ${body.model ?? ''} for provider ${body.provider ?? ''}.`);
    if (!providerConfigured(c.env, provider)) throw fail(400, `${provider.name} is not configured on this playground.`);

    const parts = [body.system_prompt?.trim(), body.context?.trim() ? `CONTEXT:\n${body.context.trim()}` : ''].filter(Boolean);
    const messages: Message[] = [
      ...(parts.length ? [{ role: 'system' as const, content: parts.join('\n\n') }] : []),
      ...body.history,
      { role: 'user', content: body.message },
    ];
    const requested = Object.fromEntries(Object.entries(body.params).filter(([, v]) => v !== undefined)) as Record<string, number>;
    const sent = Object.fromEntries(Object.entries(requested).filter(([k]) => provider.supported_params.includes(k as never)));
    pending = {
      ...base, provider: provider.id, model: model.id, model_requested: model.id, system_prompt: body.system_prompt ?? null,
      context: body.context ?? null, params: requested, pricing_snapshot: model.pricing,
      effective_request: {
        mode: 'direct_provider', provider: provider.id, model: model.id,
        params_sent: { ...sent, ...model.request_overrides }, params_ignored: Object.keys(requested).filter((k) => !(k in sent)).sort(),
        messages,
      },
    };
    const estimate = {
      system_prompt: estimateTokens(body.system_prompt), context: estimateTokens(body.context),
      history: body.history.reduce((n, t) => n + estimateTokens(t.content), 0), user_message: estimateTokens(body.message),
    };
    run = async () => {
      const g = await generateDirect(c.env, provider, model, messages, requested, s.llmTimeoutMs);
      const cost = calculateCost(model.pricing, g.inputTokens, g.outputTokens);
      return {
        ...success(g, started), ...cost,
        error: g.text ? null : 'The model returned an empty response.', status: g.text ? 'success' : 'failed',
        first_token_latency_ms: null,
        metadata: {
          provider_name: provider.name, model_reported: g.model, finish_reason: g.finishReason,
          reasoning_tokens: g.reasoningTokens, cached_tokens: g.cachedTokens, provider_reported_cost: g.providerCost,
          cost_source: cost.total_cost != null ? 'playground pricing' : null,
          prompt_breakdown_estimate: estimate, overhead_ms: Date.now() - started - g.providerLatencyMs,
        },
      };
    };
  }

  const { created, row } = await store.insertPending(sql, pending);
  if (!created) return c.json(await resolvePrior(sql, row, sess.id));

  // The generation finishes and is logged even if the browser disconnects mid-request.
  const work = finish(c.env, row.id as number, started, run, pending);
  c.executionCtx.waitUntil(work.catch(() => undefined));
  return c.json(await work);
});

function success(g: Generation, started: number) {
  return {
    response: g.text, original_response: g.text, input_tokens: g.inputTokens, output_tokens: g.outputTokens,
    total_tokens: g.totalTokens, usage_source: g.usageSource, latency_ms: Date.now() - started,
    provider_latency_ms: g.providerLatencyMs,
  };
}

/** Runs the generation on its own connection and always writes the outcome, success or failure. */
async function finish(
  env: Env, id: number, started: number, run: (db: Sql) => Promise<Partial<Record<string, unknown>>>,
  pending: store.PendingExperiment,
) {
  const db = connect(env);
  try {
    let patch: Record<string, unknown>;
    try {
      patch = (await run(db)) as Record<string, unknown>;
    } catch (err) {
      const known = err instanceof ProviderError;
      if (!known) console.error('generation failed unexpectedly:', err instanceof Error ? err.name : 'unknown');
      patch = {
        status: 'failed', error: known ? err.reason : 'Unexpected error while generating.', usage_source: 'unavailable',
        latency_ms: Date.now() - started,
        metadata: { provider_name: known ? err.provider : null, status_code: known ? err.status : null,
          ...(pending.mode === 'kundli_chatbot' ? { ai_core: { requested_style: pending.model_requested } } : {}) },
      };
    }
    return await store.finalizeExperiment(db, id, patch);
  } finally {
    await db.end({ timeout: 5 });
  }
}

async function resolvePrior(sql: Sql, row: store.Row, sessionId: string) {
  if (row.session_id !== sessionId) throw fail(409, 'request_id already used.');
  if (row.status === 'pending') {
    const failed = await store.failStale(sql, row.id, STALE_PENDING_SECONDS);
    if (!failed) throw fail(409, 'This request is already being processed.');
    return failed;
  }
  return row;
}

// ── evaluation, feedback and history ─────────────────────────────────
playground.post('/evaluate', async (c) => {
  const body = parse(evaluateSchema, await readJson(c));
  const sql = c.get('sql');
  const s = c.get('settings');
  const exp = await store.getOwnedExperiment(sql, body.experiment_id, c.get('session').id);
  if (!exp) throw fail(404, 'Experiment not found.');
  if (exp.status !== 'success' || !exp.response) throw fail(400, 'Only a successful response can be evaluated.');
  await spendGuard(sql, s);
  try {
    const verdict = await evaluate(c.env, exp, s.llmTimeoutMs);
    return c.json(await store.setEvaluation(sql, exp.id, verdict, verdict.overall));
  } catch (err) {
    if (err instanceof ProviderError) throw fail(502, `Evaluation failed (${err.provider}): ${err.reason}`);
    throw err;
  }
});

playground.post('/feedback', async (c) => {
  const s = c.get('settings');
  const body = parse(feedbackSchema(s.maxPromptChars), await readJson(c));
  const sql = c.get('sql');
  if (!(await store.getOwnedExperiment(sql, body.experiment_id, c.get('session').id))) throw fail(404, 'Experiment not found.');
  return c.json(await store.setFeedback(sql, body.experiment_id, {
    rating: body.rating ?? null, feedback: body.feedback?.trim() || null, edited: body.edited_response?.trim() || null,
  }));
});

playground.get('/experiments', async (c) => {
  const limit = Math.min(200, Math.max(1, Number(c.req.query('limit') ?? 50) || 50));
  const offset = Math.max(0, Number(c.req.query('offset') ?? 0) || 0);
  return c.json(await store.listForSession(c.get('sql'), c.get('session').id, limit, offset));
});

playground.get('/experiments/:id', async (c) => {
  const id = Number(c.req.param('id'));
  const exp = Number.isInteger(id) ? await store.getOwnedExperiment(c.get('sql'), id, c.get('session').id) : null;
  if (!exp) throw fail(404, 'Experiment not found.');
  return c.json(exp);
});
