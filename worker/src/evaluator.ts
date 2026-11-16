/**
 * Automated (LLM-as-judge) evaluation of one experiment's response.
 *
 * Scores are an AI estimate, labelled as such in the UI, never ground truth. The
 * judge is a separate call: its own tokens and cost are reported separately and
 * never mixed into the tested chatbot's or model's numbers.
 */
import { calculateCost, catalog, findModel, findProvider } from './catalog';
import type { Env } from './env';
import { generateDirect, ProviderError, type Message } from './providers';
import type { Row } from './store';

export const METRICS = [
  'relevance', 'accuracy', 'human_likeness', 'tone', 'conciseness', 'context_awareness', 'emoji_usage',
  'language_preservation', 'chart_consistency',
] as const;
export type Metric = (typeof METRICS)[number];

const JUDGE_SYSTEM = `You are a strict reviewer of chatbot replies. You score one reply to one user message.

Score each metric from 0 to 10 (decimals allowed), or null where stated:
- relevance: does it answer what was actually asked?
- accuracy: is it consistent with the provided context and free of invented facts? With no context, judge plausibility and hedging.
- human_likeness: does it read like a person texting, not a report or a template?
- tone: warm, respectful, appropriate for the user?
- conciseness: no padding, no repetition, no over-explaining?
- context_awareness: does it use the system prompt, context and conversation history well?
- emoji_usage: are emojis used appropriately? null if the reply has no emojis and none were expected.
- language_preservation: is the reply in the language and script the user wrote in (English, Hindi in Devanagari, or Hinglish in Roman letters)? If a REQUESTED REPLY STYLE is given, judge against that style instead.
- chart_consistency: check every placement, sign, house, nakshatra and dasha claim in the reply against CHART FACTS. 10 = no claim contradicts the facts; lower for contradictions or placements that are not in the facts. null if no CHART FACTS are given or the reply makes no chart claims.

Return ONLY a JSON object, no markdown:
{"scores": {"relevance": n, "accuracy": n, "human_likeness": n, "tone": n, "conciseness": n, "context_awareness": n, "emoji_usage": n or null, "language_preservation": n, "chart_consistency": n or null},
 "feedback": "2-3 sentences of concrete feedback",
 "strengths": ["short item", "..."],
 "improvements": ["short item", "..."]}`;

const SIGNS = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'];
const LORDS = ['Mars', 'Venus', 'Mercury', 'Moon', 'Sun', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Saturn', 'Jupiter'];

/**
 * Whole-sign houses worked out from the lagna, so the judge compares the reply with a
 * ready table instead of having to derive "10th house = Capricorn, lord Saturn" itself.
 * Derived only from the chart facts the chatbot reported; null when there is no lagna.
 */
export function houseTable(facts: string[]): string | null {
  const lagna = facts.map((f) => f.match(/^lagna=(\w+)$/i)?.[1]).find(Boolean);
  const start = SIGNS.findIndex((x) => x.toLowerCase() === lagna?.toLowerCase());
  if (start < 0) return null;
  const planets: Record<number, string[]> = {};
  for (const f of facts) {
    const m = f.match(/^(\w+)=\w+\/H(\d{1,2})$/);
    if (m) (planets[Number(m[2])] ??= []).push(m[1]!);
  }
  return Array.from({ length: 12 }, (_, i) => {
    const sign = (start + i) % 12;
    const here = planets[i + 1]?.join(', ');
    return `House ${i + 1}: ${SIGNS[sign]} (lord ${LORDS[sign]})${here ? ` — contains ${here}` : ''}`;
  }).join('\n');
}

function judgePrompt(e: Row): string {
  const parts: string[] = [];
  const meta = e.metadata ?? {};
  const core = meta.ai_core;
  if (e.system_prompt) parts.push(`SYSTEM PROMPT GIVEN TO THE MODEL:\n${e.system_prompt}`);
  if (core) {
    parts.push('The reply comes from a Kundli (Vedic astrology) chatbot that builds its own prompt and knowledge retrieval.');
    if (core.requested_style) parts.push(`REQUESTED REPLY STYLE: ${core.requested_style}`);
    else parts.push('REQUESTED REPLY STYLE: automatic (follow the user\'s own language)');
    if (core.chart_facts_used?.length) {
      parts.push(`CHART FACTS (the chart the chatbot used):\n${core.chart_facts_used.join('\n')}`);
      const table = houseTable(core.chart_facts_used);
      if (table) parts.push(`HOUSES DERIVED FROM THE LAGNA (whole-sign; the lord of a house is the planet that rules its sign):\n${table}`);
    }
  } else if (e.context) {
    parts.push(`CONTEXT GIVEN TO THE MODEL:\n${e.context}`);
  }
  const history = ((e.effective_request?.messages ?? []) as Message[]).slice(0, -1)
    .filter((m) => m.role === 'user' || m.role === 'assistant');
  if (history.length) parts.push('EARLIER CONVERSATION:\n' + history.slice(-10).map((m) => `${m.role}: ${m.content}`).join('\n'));
  parts.push(`USER MESSAGE:\n${e.user_prompt}`);
  parts.push(`REPLY TO EVALUATE:\n${e.response}`);
  return parts.join('\n\n');
}

export interface Verdict {
  scores: Record<Metric, number | null>;
  overall: number | null;
  feedback: string;
  strengths: string[];
  improvements: string[];
}

/** Extracts the judge's JSON, tolerating code fences and stray prose. */
export function parseJudgeOutput(text: string): Verdict {
  const cleaned = text.trim().replace(/^```(?:json)?\s*|\s*```$/g, '');
  const match = cleaned.match(/\{[\s\S]*\}/);
  if (!match) throw new Error('judge did not return JSON');
  const data = JSON.parse(match[0]);
  const raw = data.scores ?? {};
  const scores = {} as Record<Metric, number | null>;
  for (const metric of METRICS) {
    const v = raw[metric];
    const n = v === null || v === undefined || v === '' ? NaN : Number(v);
    scores[metric] = Number.isFinite(n) ? Math.round(Math.max(0, Math.min(10, n)) * 10) / 10 : null;
  }
  const present = Object.values(scores).filter((v): v is number => v !== null);
  return {
    scores,
    overall: present.length ? Math.round((present.reduce((a, b) => a + b, 0) / present.length) * 10) / 10 : null,
    feedback: String(data.feedback ?? '').trim(),
    strengths: (Array.isArray(data.strengths) ? data.strengths : []).slice(0, 6).map(String),
    improvements: (Array.isArray(data.improvements) ? data.improvements : []).slice(0, 6).map(String),
  };
}

export async function evaluate(env: Env, experiment: Row, timeoutMs: number, doFetch: typeof fetch = fetch) {
  const target = catalog.evaluator;
  const model = target && findModel(target.provider, target.model);
  const provider = model && findProvider(model.provider);
  if (!model || !provider) throw new ProviderError('evaluator', 'No evaluator model is configured.');

  const started = Date.now();
  const result = await generateDirect(env, provider, model, [
    { role: 'system', content: JUDGE_SYSTEM },
    { role: 'user', content: judgePrompt(experiment) },
  ], { temperature: 0, max_tokens: 800 }, timeoutMs, doFetch);

  let verdict: Verdict;
  try {
    verdict = parseJudgeOutput(result.text);
  } catch (err) {
    throw new ProviderError(provider.name, `The evaluator returned an unreadable verdict (${(err as Error).message}).`);
  }
  return {
    ...verdict,
    label: 'AI Evaluation',
    evaluator: { provider: provider.id, model: model.id },
    evaluator_usage: {
      input_tokens: result.inputTokens,
      output_tokens: result.outputTokens,
      total_tokens: result.totalTokens,
      latency_ms: Date.now() - started,
      ...calculateCost(model.pricing, result.inputTokens, result.outputTokens),
    },
  };
}
