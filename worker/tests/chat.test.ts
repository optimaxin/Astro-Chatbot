import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Browser, chatbotReply, directChat, kundliChat, makeEnv, mockUpstream, openAiReply, resetDb, sql, type UpstreamCall } from './helpers';

let env: Awaited<ReturnType<typeof makeEnv>>;
let tester: Browser;
beforeEach(async () => {
  await resetDb();
  env = await makeEnv();
  tester = new Browser(env);
  await tester.start('Asha');
});
afterEach(() => vi.unstubAllGlobals());

const isChatbot = (c: UpstreamCall) => c.url === 'http://ai-core.test/chat';
const isDirect = (c: UpstreamCall) => c.url.endsWith('/chat/completions');
const judge = JSON.stringify({
  scores: { relevance: 9, accuracy: 8, human_likeness: 9, tone: 8, conciseness: 9, context_awareness: 8, emoji_usage: null,
    language_preservation: 10, chart_consistency: 7 },
  feedback: 'Clear.', strengths: ['Natural'], improvements: ['Shorter'],
});

describe('Kundli Chatbot mode (the real chatbot endpoint)', () => {
  it('sends one request to the chatbot with birth details and records its real metrics', async () => {
    const calls = mockUpstream((c) => (isChatbot(c) ? Response.json(chatbotReply()) : undefined));
    const res = await kundliChat(tester, 'Mera career kaisa rahega?', { request_id: 'req-kundli-0001' });
    expect(res.status).toBe(200);

    // exactly one upstream generation, through the production endpoint, with chart context and bubbles like the website
    expect(calls).toHaveLength(1);
    expect(calls[0]!.method).toBe('POST');
    expect(calls[0]!.body).toMatchObject({
      question: 'Mera career kaisa rahega?', chat_bubbles: true,
      birth: { date: '1995-04-12', time: '06:30', timezoneOffsetMinutes: 330 },
    });
    expect(calls[0]!.body.session_id).toBe('playground-conv-0000000001');
    expect(calls[0]!.body).not.toHaveProperty('response_style'); // automatic, as in production

    const e = res.data;
    expect(e).toMatchObject({
      mode: 'kundli_chatbot', status: 'success', tester_name: 'Asha', request_id: 'req-kundli-0001',
      // underlying provider and model come from the chatbot's own response
      provider: 'nvidia', model: 'nvidia/nemotron-3.5-lightning-30b-a3b',
      input_tokens: 5206, output_tokens: 150, total_tokens: 5356, usage_source: 'provider',
      total_cost: 0, currency: 'USD', input_cost: null, output_cost: null, first_token_latency_ms: null,
      response: 'Aapka 10th house Capricorn hai aur uska lord Saturn 11th house mein hai.', system_prompt: null,
    });
    expect(e.original_response).toBe(e.response);
    expect(e.session_id).toMatch(/^[0-9a-f-]{36}$/);
    expect(e.latency_ms).toBeGreaterThanOrEqual(0);
    expect(e.metadata.ai_core).toMatchObject({
      underlying_provider: 'nvidia', fallback_used: false, turn_total_tokens: 5900, llm_calls: 2, intent: 'career',
      chart_facts_used: ['lagna=Aries', 'Saturn=Aquarius/H11', 'mahadasha=Moon'],
    });
    expect(e.metadata.ttft_note).toMatch(/not available/i);
    expect(e.effective_request.birth.date).toBe('1995-04-12');
  });

  it('forwards the selected reply style, the tester\'s own birth details, and keeps the chatbot conversation going', async () => {
    const calls = mockUpstream((c) => (isChatbot(c) ? Response.json(chatbotReply()) : undefined));
    const birth = { date: '1990-01-02', time: '23:45', timezoneOffsetMinutes: -300, latitude: 40.7, longitude: -74, placeLabel: 'New York' };
    await kundliChat(tester, 'first', { style: 'english', birth });
    await kundliChat(tester, 'second', { style: 'english', birth });
    expect(calls[0]!.body.response_style).toBe('english');
    expect(calls[0]!.body.birth).toMatchObject({ date: '1990-01-02', placeLabel: 'New York' });
    expect(calls[0]!.headers['x-chat-session-token']).toBeUndefined();
    expect(calls[1]!.headers['x-chat-session-token']).toBe('anon-token-1'); // the chatbot's own session continues
    const [row] = await sql`select aicore_session_token from playground.sessions`;
    expect(row!.aicore_session_token).toBe('anon-token-1');
  });

  it('marks unavailable metrics instead of inventing them', async () => {
    mockUpstream((c) => (isChatbot(c) ? Response.json(chatbotReply({
      usage: { tokens_available: false, token_source: 'unavailable', provider: 'nvidia', model: 'm', estimated_cost_usd: null },
    })) : undefined));
    const e = (await kundliChat(tester)).data;
    expect(e).toMatchObject({ status: 'success', input_tokens: null, output_tokens: null, total_tokens: null,
      usage_source: 'unavailable', total_cost: null, first_token_latency_ms: null });
  });

  it('records an unreachable chatbot as a failed experiment, with no invented provider or model', async () => {
    mockUpstream(() => { throw new TypeError('fetch failed'); });
    const res = await kundliChat(tester);
    expect(res.status).toBe(200);
    expect(res.data).toMatchObject({ status: 'failed', error: 'Could not connect to the provider.', provider: 'unknown', model: 'unknown', usage_source: 'unavailable' });
    expect((await sql`select count(*)::int as n from playground.experiments where status = 'failed'`)[0]!.n).toBe(1);
  });

  it('records the chatbot\'s fallback reply as a failure, and its HTTP errors', async () => {
    mockUpstream(() => Response.json(chatbotReply({ degraded: true, answer: 'Sorry, try again.' })));
    expect((await kundliChat(tester)).data).toMatchObject({ status: 'failed', response: 'Sorry, try again.' });
    mockUpstream(() => Response.json({ detail: 'Too many requests.' }, { status: 429 }));
    expect((await kundliChat(tester)).data).toMatchObject({ status: 'failed', error: 'The chatbot is rate limiting requests.' });
  });

  it('is unavailable (503) when AI_CORE_URL is not configured, and records nothing', async () => {
    const b = new Browser(await makeEnv({ AI_CORE_URL: '' }));
    await b.start('Ben');
    expect((await kundliChat(b)).status).toBe(503);
    expect((await sql`select count(*)::int as n from playground.experiments`)[0]!.n).toBe(0);
  });

  it('validates the birth form', async () => {
    mockUpstream(() => Response.json(chatbotReply()));
    const bad = { date: '12-04-1995', time: '25:00', timezoneOffsetMinutes: 330, latitude: 95, longitude: 0 };
    expect((await kundliChat(tester, 'x', { birth: bad })).status).toBe(422);
  });
});

describe('Direct provider mode', () => {
  it('calls the selected provider with only the supported parameters and the server-side key', async () => {
    const calls = mockUpstream((c) => (isDirect(c) ? openAiReply() : undefined));
    const res = await directChat(tester, {
      provider: 'nvidia', model: 'nvidia/nemotron-3.5-bolt-e4b', system_prompt: 'Be brief.', context: 'Lagna: Virgo',
      history: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'hello' }],
      params: { temperature: 0.4, top_k: 20, max_tokens: 100 },
    });
    expect(res.status).toBe(200);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe('https://integrate.api.nvidia.com/v1/chat/completions');
    expect(calls[0]!.headers.authorization).toBe('Bearer nvidia-test-key-SECRET');
    expect(calls[0]!.body).toMatchObject({ model: 'nvidia/nemotron-3.5-bolt-e4b', temperature: 0.4, max_tokens: 100 });
    expect(calls[0]!.body).not.toHaveProperty('top_k'); // NVIDIA does not support it
    expect(calls[0]!.body.messages.map((m: any) => m.role)).toEqual(['system', 'user', 'assistant', 'user']);
    expect(calls[0]!.body.messages[0].content).toContain('Be brief.');
    expect(calls[0]!.body.messages[0].content).toContain('Lagna: Virgo');
    expect(res.data.effective_request.params_ignored).toEqual(['top_k']);
    // the key never reaches the browser
    expect(res.text).not.toContain('SECRET');
  });

  it('computes cost from the provider-reported tokens and saves the pricing snapshot', async () => {
    mockUpstream((c) => (isDirect(c) ? openAiReply() : undefined));
    const e = (await directChat(tester)).data;
    expect(e).toMatchObject({ mode: 'direct_provider', provider: 'gemini', model: 'gemini-2.5-flash', status: 'success',
      input_tokens: 1000, output_tokens: 500, total_tokens: 1500, usage_source: 'provider', currency: 'USD' });
    expect(e.total_cost).toBeCloseTo(0.00155, 10);
    expect(e.pricing_snapshot).toMatchObject({ input_per_1m: 0.3, output_per_1m: 2.5, currency: 'USD' });
    expect(e.metadata.model_reported).toBe('reported-model');
  });

  it('shows N/A (null) rather than inventing a cost for unpriced models or missing usage', async () => {
    mockUpstream((c) => (isDirect(c) ? openAiReply('hi', null) : undefined));
    const e = (await directChat(tester)).data;
    expect(e).toMatchObject({ input_tokens: null, total_tokens: null, total_cost: null, usage_source: 'unavailable' });
    const free = mockUpstream((c) => (isDirect(c) ? openAiReply() : undefined));
    const u = (await directChat(tester, { provider: 'nvidia', model: 'nvidia/nemotron-3.5-bolt-e4b' })).data;
    expect(free).toHaveLength(1);
    expect(u.total_cost).toBeNull();
  });

  it('saves provider failures as failed experiments without leaking the key', async () => {
    mockUpstream(() => Response.json({ error: { message: 'Bad key nvidia-test-key-SECRET' } }, { status: 401 }));
    const res = await directChat(tester);
    expect(res.data).toMatchObject({ status: 'failed', error: 'The provider rejected the API key.' });
    expect(res.text).not.toContain('SECRET');
    mockUpstream(() => Response.json({ error: { message: 'gone' } }, { status: 410 }));
    expect((await directChat(tester)).data.error).toMatch(/retired/);
    mockUpstream(() => Response.json({ error: { message: 'x' } }, { status: 402 }));
    expect((await directChat(tester)).data.error).toMatch(/insufficient credits/);
  });

  it('rejects unknown models, unconfigured providers and bad parameters before spending anything', async () => {
    const calls = mockUpstream(() => openAiReply());
    expect((await directChat(tester, { model: 'nope' })).status).toBe(400);
    expect((await directChat(tester, { provider: 'openrouter', model: 'openai/gpt-4o-mini' })).status).toBe(400); // no key
    expect((await directChat(tester, { provider: 'ai-core', model: 'x' })).status).toBe(400);
    expect((await directChat(tester, { params: { temperature: 5 } })).status).toBe(422);
    expect((await directChat(tester, { message: '   ' })).status).toBe(422);
    expect((await directChat(tester, { message: 'x'.repeat(20001) })).status).toBe(422);
    expect(calls).toHaveLength(0);
    expect((await sql`select count(*)::int as n from playground.experiments`)[0]!.n).toBe(0);
  });
});

describe('logging, idempotency and limits', () => {
  it('never generates twice for a retried request id', async () => {
    const calls = mockUpstream((c) => (isChatbot(c) ? Response.json(chatbotReply()) : undefined));
    const body = { request_id: 'req-retry-0001' };
    const first = await kundliChat(tester, 'hello', body);
    const retry = await kundliChat(tester, 'hello', body);
    expect(retry.data.id).toBe(first.data.id);
    expect(calls).toHaveLength(1);
    expect((await sql`select count(*)::int as n from playground.experiments`)[0]!.n).toBe(1);
    const other = new Browser(env);
    await other.start('Ben');
    expect((await kundliChat(other, 'hello', body)).status).toBe(409); // another tester cannot reuse it
  });

  it('refuses a duplicate while the first is still running, then recovers a stuck one', async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => { release = r; });
    const calls = mockUpstream(async (c) => { if (isChatbot(c)) { await gate; return Response.json(chatbotReply()); } });
    const first = kundliChat(tester, 'slow', { request_id: 'req-inflight-01' });
    await vi.waitFor(async () => expect((await sql`select count(*)::int as n from playground.experiments`)[0]!.n).toBe(1));
    const sameTesterTab = new Browser(env); // same cookies, its own background-work list
    sameTesterTab.jar = tester.jar;
    const dup = await kundliChat(sameTesterTab, 'slow', { request_id: 'req-inflight-01' });
    expect(dup.status).toBe(409);
    expect(calls).toHaveLength(1);
    release();
    expect((await first).data.status).toBe('success');

    await sql`insert into playground.experiments (request_id, session_id, tester_name, mode, provider, model, user_prompt, status, created_at)
              select 'req-stuck-0001', id, tester_name, 'direct_provider', 'gemini', 'gemini-2.5-flash', 'x', 'pending', now() - interval '10 minutes'
              from playground.sessions limit 1`;
    const recovered = await directChat(tester, { request_id: 'req-stuck-0001' });
    expect(recovered.data).toMatchObject({ status: 'failed', error: 'The generation did not complete.' });
  });

  it('links every experiment to the right tester across concurrent sessions', async () => {
    mockUpstream((c) => (isChatbot(c) ? Response.json(chatbotReply()) : isDirect(c) ? openAiReply() : undefined));
    const names = Array.from({ length: 10 }, (_, i) => `Tester ${i}`);
    const browsers = await Promise.all(names.map(async (n) => { const b = new Browser(env); await b.start(n); return b; }));
    const results = await Promise.all(browsers.map((b, i) => (i % 2 ? kundliChat(b, `q${i}`) : directChat(b, { message: `q${i}` }))));
    expect(results.map((r) => r.status)).toEqual(Array(10).fill(200));
    const rows = await sql`select e.tester_name, e.user_prompt, e.session_id, s.tester_name as session_name
                           from playground.experiments e join playground.sessions s on s.id = e.session_id order by e.id`;
    expect(rows).toHaveLength(10);
    for (const r of rows) {
      expect(r.tester_name).toBe(r.session_name);
      expect(r.tester_name).toBe(`Tester ${String(r.user_prompt).slice(1)}`);
    }
    expect(new Set(rows.map((r) => r.session_id)).size).toBe(10);
  });

  it('rate limits generations per connection and stops at the daily caps', async () => {
    mockUpstream((c) => (isDirect(c) ? openAiReply() : undefined));
    const limited = new Browser(await makeEnv({ PLAYGROUND_RATE_LIMIT_PER_MINUTE: '2' }));
    await limited.start('Rita');
    expect((await directChat(limited)).status).toBe(200);
    expect((await directChat(limited)).status).toBe(200);
    expect((await directChat(limited)).status).toBe(429);

    await resetDb();
    const capped = new Browser(await makeEnv({ PLAYGROUND_DAILY_REQUEST_LIMIT: '1' }));
    await capped.start('Cal');
    expect((await directChat(capped)).status).toBe(200);
    expect((await directChat(capped)).status).toBe(429);

    await resetDb();
    const spend = new Browser(await makeEnv({ PLAYGROUND_DAILY_COST_LIMIT_USD: '0.002' }));
    await spend.start('Sam');
    expect((await directChat(spend)).status).toBe(200); // costs 0.00155
    expect((await directChat(spend)).status).toBe(200); // now over the cap
    expect((await directChat(spend)).status).toBe(429);
  });

  it('keeps the generation running via waitUntil so it is logged even if the browser disconnects', async () => {
    mockUpstream((c) => (isDirect(c) ? openAiReply() : undefined));
    const spy = vi.fn();
    const ctx = { waitUntil: (p: Promise<unknown>) => { spy(); void p; }, passThroughOnException() {} };
    const { app } = await import('../src/app');
    const cookie = `pg_session=${tester.jar.get('pg_session')!.value}`;
    const res = await app.fetch(new Request('https://playground.test/api/playground/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Cookie: cookie },
      body: JSON.stringify({ mode: 'direct_provider', provider: 'gemini', model: 'gemini-2.5-flash', message: 'hi',
        request_id: 'req-wait-00001', conversation_id: 'conv-0000000001' }),
    }), env, ctx as unknown as ExecutionContext);
    expect(res.status).toBe(200);
    expect(spy).toHaveBeenCalled();
  });
});

describe('evaluation and feedback', () => {
  it('judges a Kundli reply against the chart facts the chatbot used, as a separate call', async () => {
    const calls = mockUpstream((c) => (isChatbot(c) ? Response.json(chatbotReply())
      : isDirect(c) ? openAiReply(judge, { prompt_tokens: 400, completion_tokens: 120, total_tokens: 520 }) : undefined));
    const exp = (await kundliChat(tester, 'Mera career kaisa rahega?', { style: 'english' })).data;
    const res = await tester.json('/api/playground/evaluate', { body: { experiment_id: exp.id } });
    expect(res.status).toBe(200);
    expect(res.data.evaluation_overall).toBeCloseTo(8.5, 1);
    expect(res.data.evaluation.scores).toMatchObject({ language_preservation: 10, chart_consistency: 7, emoji_usage: null });
    expect(res.data.evaluation.label).toBe('AI Evaluation');

    // the judge is its own call and its usage is kept apart from the chatbot's tokens
    expect(calls.filter(isChatbot)).toHaveLength(1);
    const judgeCall = calls.find(isDirect)!;
    const prompt = judgeCall.body.messages[1].content as string;
    expect(prompt).toContain('Saturn=Aquarius/H11');
    expect(prompt).toContain('House 10: Capricorn (lord Saturn)');
    expect(prompt).toContain('REQUESTED REPLY STYLE: english');
    expect(res.data.total_tokens).toBe(5356);
    expect(res.data.evaluation.evaluator_usage.total_tokens).toBe(520);
  });

  it('only evaluates your own successful responses', async () => {
    mockUpstream((c) => (isChatbot(c) ? Response.json(chatbotReply()) : openAiReply(judge)));
    const exp = (await kundliChat(tester)).data;
    const other = new Browser(env);
    await other.start('Ben');
    expect((await other.json('/api/playground/evaluate', { body: { experiment_id: exp.id } })).status).toBe(404);
    mockUpstream(() => { throw new TypeError('x'); });
    const failed = (await kundliChat(tester)).data;
    expect((await tester.json('/api/playground/evaluate', { body: { experiment_id: failed.id } })).status).toBe(400);
  });

  it('reports an unreadable or failing judge as 502 without corrupting the record', async () => {
    mockUpstream((c) => (isChatbot(c) ? Response.json(chatbotReply()) : openAiReply('not json at all')));
    const exp = (await kundliChat(tester)).data;
    expect((await tester.json('/api/playground/evaluate', { body: { experiment_id: exp.id } })).status).toBe(502);
    expect((await tester.json(`/api/playground/experiments/${exp.id}`)).data.evaluation).toBeNull();
  });

  it('stores feedback on the experiment and keeps the original response', async () => {
    mockUpstream((c) => (isChatbot(c) ? Response.json(chatbotReply()) : undefined));
    const exp = (await kundliChat(tester)).data;
    const res = await tester.json('/api/playground/feedback', {
      body: { experiment_id: exp.id, rating: 'bad', feedback: 'Too long', edited_response: 'Shorter.' } });
    expect(res.data).toMatchObject({ human_rating: 'bad', human_feedback: 'Too long', edited_response: 'Shorter.', response: exp.response });
    expect(res.data.feedback_at).toBeGreaterThan(0);
  });

  it('lists a tester\'s history newest first', async () => {
    mockUpstream((c) => (isDirect(c) ? openAiReply() : undefined));
    const a = (await directChat(tester)).data.id;
    const b = (await directChat(tester, { message: 'second' })).data.id;
    expect((await tester.json('/api/playground/experiments')).data.map((e: any) => e.id)).toEqual([b, a]);
  });
});
