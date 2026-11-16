import { providerKey, type Birth, type ModelConfig, type ProviderConfig } from './catalog';
import type { Env } from './env';

export class ProviderError extends Error {
  constructor(public provider: string, public reason: string, public status: number | null = null) {
    super(reason);
  }
}

export interface Generation {
  text: string;
  model: string;
  inputTokens: number | null;
  outputTokens: number | null;
  totalTokens: number | null;
  providerLatencyMs: number;
  finishReason: string | null;
  reasoningTokens: number | null;
  cachedTokens: number | null;
  providerCost: number | null;
  usageSource: 'provider' | 'estimated' | 'unavailable';
  extra: Record<string, unknown>;
}

export type Message = { role: 'system' | 'user' | 'assistant'; content: string };
type Fetch = typeof fetch;

const asInt = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? Math.round(v) : null);

export function friendlyReason(status: number | null, body: string): string {
  if (status === 429) return 'Rate limit exceeded.';
  if (status === 401 || status === 403) return 'The provider rejected the API key.';
  if (status === 402) return 'The provider account has insufficient credits.';
  if (status === 404) return 'The provider does not recognise this model id.';
  if (status === 410) return 'The provider has retired this model; it is no longer available.';
  if (status === 408 || status === 504) return 'The provider did not respond in time.';
  let detail = body;
  try {
    const parsed = JSON.parse(body);
    detail = parsed?.error?.message ?? parsed?.detail ?? parsed?.message ?? body;
  } catch { /* not JSON */ }
  return `HTTP ${status}: ${String(detail).replace(/\s+/g, ' ').slice(0, 300)}`;
}

async function post(name: string, url: string, init: RequestInit, timeoutMs: number, doFetch: Fetch): Promise<Response> {
  try {
    return await doFetch(url, { ...init, signal: AbortSignal.timeout(timeoutMs) });
  } catch (err) {
    const timedOut = err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError');
    throw new ProviderError(name, timedOut ? 'The provider did not respond in time.' : 'Could not connect to the provider.');
  }
}

/** Any provider that speaks the OpenAI chat-completions protocol (NVIDIA, Gemini, OpenRouter). */
export async function generateDirect(
  env: Env, provider: ProviderConfig, model: ModelConfig, messages: Message[],
  params: Record<string, number>, timeoutMs: number, doFetch: Fetch = fetch,
): Promise<Generation> {
  const key = providerKey(env, provider);
  if (!key) throw new ProviderError(provider.name, `${provider.api_key_env} is not set on the playground.`);

  const body: Record<string, unknown> = { model: model.id, messages };
  const extra: Record<string, unknown> = { ...provider.extra_body, ...model.extra_body };
  for (const [k, v] of Object.entries(params)) {
    if (!provider.supported_params.includes(k as never)) continue;
    // top_k is not part of the OpenAI schema, so it travels as an extra body field.
    if (k === 'top_k') extra.top_k = v;
    else body[k] = v;
  }
  Object.assign(body, model.request_overrides, extra);

  const started = Date.now();
  const res = await post(provider.name, `${provider.base_url}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    body: JSON.stringify(body),
  }, timeoutMs, doFetch);
  const providerLatencyMs = Date.now() - started;
  const raw = await res.text();
  if (!res.ok) throw new ProviderError(provider.name, friendlyReason(res.status, raw), res.status);

  let data: any;
  try { data = JSON.parse(raw); } catch { throw new ProviderError(provider.name, 'The provider returned an unreadable response.'); }
  const choice = data?.choices?.[0];
  const usage = data?.usage ?? {};
  const content = choice?.message?.content;
  const input = asInt(usage.prompt_tokens);
  return {
    text: (typeof content === 'string' ? content : '').trim(),
    model: typeof data?.model === 'string' ? data.model : model.id,
    inputTokens: input,
    outputTokens: asInt(usage.completion_tokens),
    totalTokens: asInt(usage.total_tokens),
    providerLatencyMs,
    finishReason: choice?.finish_reason ?? null,
    reasoningTokens: asInt(usage.completion_tokens_details?.reasoning_tokens),
    cachedTokens: asInt(usage.prompt_tokens_details?.cached_tokens),
    providerCost: typeof usage.cost === 'number' ? usage.cost : null,
    usageSource: input != null ? 'provider' : 'unavailable',
    extra: {},
  };
}

export interface ChatbotTurn {
  question: string;
  birth: Birth;
  style: string | null; // null = automatic, exactly like the production widget
  conversationId: string;
  sessionToken: string | null;
}

export interface ChatbotResult extends Generation {
  newSessionToken: string | null;
}

/**
 * One turn through the real Kundli chatbot (ai-core `POST /chat`, the endpoint the
 * Tredev website's chat widget uses). The chatbot builds its own prompt, retrieves
 * its own knowledge and validates its own reply, so nothing is generated or
 * re-processed here; usage, provider and model are read from its response.
 */
export async function generateChatbot(env: Env, turn: ChatbotTurn, timeoutMs: number, doFetch: Fetch = fetch): Promise<ChatbotResult> {
  const base = (env.AI_CORE_URL ?? '').trim().replace(/\/+$/, '');
  if (!base) throw new ProviderError('Kundli Chatbot', 'AI_CORE_URL is not set on the playground.');
  const session = `playground-${turn.conversationId}`;
  const body: Record<string, unknown> = {
    user_id: session, session_id: session, question: turn.question, birth: turn.birth, chat_bubbles: true,
  };
  if (turn.style) body.response_style = turn.style;
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (turn.sessionToken) headers['X-Chat-Session-Token'] = turn.sessionToken;

  const started = Date.now();
  const res = await post('Kundli Chatbot', `${base}/chat`, { method: 'POST', headers, body: JSON.stringify(body) }, timeoutMs, doFetch);
  const providerLatencyMs = Date.now() - started;
  const raw = await res.text();
  if (!res.ok) {
    let detail = '';
    try { detail = String(JSON.parse(raw)?.detail ?? '').slice(0, 200); } catch { /* not JSON */ }
    const why = res.status === 429 ? 'The chatbot is rate limiting requests.' : `The chatbot returned HTTP ${res.status}${detail ? `: ${detail}` : '.'}`;
    throw new ProviderError('Kundli Chatbot', why, res.status);
  }
  let data: any;
  try { data = JSON.parse(raw); } catch { throw new ProviderError('Kundli Chatbot', 'The chatbot returned an unreadable response.'); }

  const u = data?.usage ?? {};
  const tokensKnown = u.tokens_available === true;
  const cost = typeof u.estimated_cost_usd === 'number' ? u.estimated_cost_usd : null;
  return {
    text: String(data?.answer ?? '').trim(),
    // Underlying provider/model as reported by the chatbot itself; never assumed.
    model: typeof u.model === 'string' && u.model ? u.model : 'unknown',
    inputTokens: tokensKnown ? asInt(u.input_tokens) : null,
    outputTokens: tokensKnown ? asInt(u.output_tokens) : null,
    totalTokens: tokensKnown ? asInt(u.total_tokens) : null,
    providerLatencyMs,
    finishReason: null,
    reasoningTokens: null,
    cachedTokens: null,
    providerCost: cost,
    usageSource: tokensKnown ? 'provider' : u.token_source === 'estimated' ? 'estimated' : 'unavailable',
    newSessionToken: typeof data?.session_token === 'string' ? data.session_token : null,
    extra: {
      underlying_provider: typeof u.provider === 'string' && u.provider ? u.provider : 'unknown',
      fallback_used: Boolean(u.fallback_used),
      turn_total_tokens: asInt(u.turn_total_tokens),
      turn_cost_usd: typeof u.turn_cost_usd === 'number' ? u.turn_cost_usd : null,
      llm_calls: asInt(u.llm_calls),
      estimated_total_tokens: asInt(u.estimated_total_tokens),
      token_source: u.token_source ?? null,
      request_id: data?.request_id ?? null,
      intent: data?.intent ?? null,
      persona: data?.persona ?? null,
      response_style: data?.response_style ?? null,
      degraded: Boolean(data?.degraded),
      notices: Array.isArray(data?.notices) ? data.notices : [],
      citations: Array.isArray(data?.citations) ? data.citations : [],
      sources: Array.isArray(data?.sources) ? data.sources.length : 0,
      chart_facts_used: Array.isArray(data?.chart_facts_used) ? data.chart_facts_used : [],
      messages: Array.isArray(data?.messages) ? data.messages : [],
      chatbot_latency_ms: asInt(data?.latency_ms),
    },
  };
}

/** Reply styles the chatbot offers, read from the chatbot itself (cached briefly). */
let stylesCache: { at: number; base: string; styles: { id: string; name: string }[] } | null = null;
export async function chatbotStyles(env: Env, fallback: { id: string; name: string }[], doFetch: Fetch = fetch) {
  const base = (env.AI_CORE_URL ?? '').trim().replace(/\/+$/, '');
  if (!base) return fallback;
  if (stylesCache && stylesCache.base === base && Date.now() - stylesCache.at < 300_000) return stylesCache.styles;
  try {
    const res = await doFetch(`${base}/response-styles`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return fallback;
    const list = (await res.json()) as { id?: unknown; name?: unknown }[];
    const styles = list.filter((s) => typeof s.id === 'string').map((s) => ({ id: s.id as string, name: String(s.name ?? s.id) }));
    if (!styles.length) return fallback;
    stylesCache = { at: Date.now(), base, styles };
    return styles;
  } catch {
    return fallback;
  }
}
