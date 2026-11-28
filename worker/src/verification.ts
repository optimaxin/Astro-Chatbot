/**
 * Chatbot response verification: extract the factual chart claims a reply
 * makes, then compare each one against the reference Kundli's own structured
 * facts. Extraction may use an LLM (natural-language variation is real), but
 * the pass/fail decision is always this module's own deterministic
 * comparison — an LLM never overrides a contradiction (spec requirement).
 */
import { catalog, findModel, findProvider } from './catalog';
import type { Env } from './env';
import { chartFactTable, type KundliFullChart } from './kundli';
import { generateDirect, ProviderError, type Message } from './providers';

const PLANETS = ['sun', 'moon', 'mars', 'mercury', 'jupiter', 'venus', 'saturn', 'rahu', 'ketu'] as const;

const ATTRIBUTE_PATTERN = new RegExp(
  `^(lagna|moonSign|nakshatra|nakshatraPada|mahadasha|antardasha|planet\\.(${PLANETS.join('|')})\\.(sign|house|retrograde|nakshatra))$`,
);

const CLAIM_SYSTEM = `You extract objectively verifiable Vedic-astrology chart claims from one chatbot reply. You do not judge whether they are true.

Only extract a claim if it states one of these exact chart facts (use the attribute key shown, substituting the planet's name):
- lagna (ascendant sign, e.g. "your ascendant/lagna is Leo")
- moonSign (the sign the Moon is in)
- nakshatra (the Moon's nakshatra/birth star)
- nakshatraPada (the Moon's nakshatra pada, 1-4)
- mahadasha (the current running Mahadasha lord)
- antardasha (the current running Antardasha lord)
- planet.<sun|moon|mars|mercury|jupiter|venus|saturn|rahu|ketu>.sign (a planet's zodiac sign)
- planet.<...>.house (a planet's house number, 1-12)
- planet.<...>.retrograde (whether a planet is retrograde — statedValue "true" or "false")
- planet.<...>.nakshatra (a planet's nakshatra)

Do NOT extract: predictions, personality descriptions, advice, remedies, life-event forecasts, yoga/dosha interpretations, or anything about divisional charts, dashas other than the current Mahadasha/Antardasha, or houses/signs the reply does not explicitly name. If a sentence mixes a chart fact with a prediction, extract only the factual part.

Return ONLY a JSON array, no markdown, no prose:
[{"claim": "the exact sentence or clause making the claim", "attribute": "one of the attribute keys above, planet name lowercase", "statedValue": "the value the reply states, as plain text"}]
Return [] if the reply makes no such claims.`;

export interface ExtractedClaim {
  claim: string;
  attribute: string;
  statedValue: string;
}

export type ClaimStatus = 'SUPPORTED' | 'CONTRADICTED' | 'UNVERIFIABLE' | 'NOT_CHART_FACT';

export interface VerifiedClaim extends ExtractedClaim {
  referenceValue: string | number | boolean | null;
  status: ClaimStatus;
  explanation: string;
}

export interface VerificationResult {
  claims: VerifiedClaim[];
  supportedCount: number;
  contradictedCount: number;
  unverifiableCount: number;
  consistencyPercent: number | null;
  evaluator: { provider: string; model: string };
}

function parseClaimsOutput(text: string): ExtractedClaim[] {
  const cleaned = text.trim().replace(/^```(?:json)?\s*|\s*```$/g, '');
  const match = cleaned.match(/\[[\s\S]*\]/);
  if (!match) return [];
  let parsed: unknown;
  try { parsed = JSON.parse(match[0]); } catch { return []; }
  if (!Array.isArray(parsed)) return [];
  return parsed
    .filter((c): c is Record<string, unknown> => c != null && typeof c === 'object')
    .map((c) => ({
      claim: String(c.claim ?? '').trim(),
      attribute: String(c.attribute ?? '').trim(),
      statedValue: String(c.statedValue ?? '').trim(),
    }))
    .filter((c) => c.claim && c.attribute && c.statedValue);
}

export async function extractClaims(env: Env, replyText: string, timeoutMs: number, doFetch: typeof fetch = fetch): Promise<ExtractedClaim[]> {
  const target = catalog.evaluator;
  const model = target && findModel(target.provider, target.model);
  const provider = model && findProvider(model.provider);
  if (!model || !provider) throw new ProviderError('verification', 'No evaluator model is configured.');

  const messages: Message[] = [
    { role: 'system', content: CLAIM_SYSTEM },
    { role: 'user', content: `CHATBOT REPLY:\n${replyText}` },
  ];
  const result = await generateDirect(env, provider, model, messages, { temperature: 0, max_tokens: 1200 }, timeoutMs, doFetch);
  return parseClaimsOutput(result.text);
}

function normalize(v: string): string {
  return v.trim().toLowerCase();
}

function valuesMatch(reference: string | number | boolean, stated: string): boolean {
  if (typeof reference === 'boolean') {
    const s = normalize(stated);
    const truthy = s === 'true' || s === 'retrograde' || s === 'yes';
    const falsy = s === 'false' || s === 'direct' || s === 'not retrograde' || s === 'no';
    if (truthy) return reference === true;
    if (falsy) return reference === false;
    return false;
  }
  if (typeof reference === 'number') {
    const n = Number(stated.replace(/[^\d.-]/g, ''));
    return Number.isFinite(n) && n === reference;
  }
  return normalize(String(reference)) === normalize(stated);
}

/** The deterministic half: never lets an extraction artifact turn a real contradiction into a pass. */
export function compareClaims(claims: ExtractedClaim[], chart: KundliFullChart): VerifiedClaim[] {
  const facts = chartFactTable(chart);
  return claims.map((c) => {
    if (!ATTRIBUTE_PATTERN.test(c.attribute)) {
      return { ...c, referenceValue: null, status: 'NOT_CHART_FACT', explanation: 'Not a recognised chart-fact attribute.' };
    }
    const reference = facts[c.attribute];
    if (reference === undefined) {
      return { ...c, referenceValue: null, status: 'UNVERIFIABLE', explanation: 'This chart does not have that detail available.' };
    }
    const match = valuesMatch(reference, c.statedValue);
    return {
      ...c,
      referenceValue: reference,
      status: match ? 'SUPPORTED' : 'CONTRADICTED',
      explanation: match
        ? `Matches the reference chart (${reference}).`
        : `Reference chart says ${reference}, reply says ${c.statedValue}.`,
    };
  });
}

export function summarize(claims: VerifiedClaim[]): Pick<VerificationResult, 'supportedCount' | 'contradictedCount' | 'unverifiableCount' | 'consistencyPercent'> {
  const supportedCount = claims.filter((c) => c.status === 'SUPPORTED').length;
  const contradictedCount = claims.filter((c) => c.status === 'CONTRADICTED').length;
  const unverifiableCount = claims.filter((c) => c.status === 'UNVERIFIABLE').length;
  const denominator = supportedCount + contradictedCount;
  return {
    supportedCount, contradictedCount, unverifiableCount,
    consistencyPercent: denominator > 0 ? Math.round((supportedCount / denominator) * 1000) / 10 : null,
  };
}

export async function verifyReply(env: Env, replyText: string, chart: KundliFullChart, timeoutMs: number, doFetch: typeof fetch = fetch): Promise<VerificationResult> {
  const target = catalog.evaluator!;
  const extracted = await extractClaims(env, replyText, timeoutMs, doFetch);
  const claims = compareClaims(extracted, chart);
  return { claims, ...summarize(claims), evaluator: { provider: target.provider, model: target.model } };
}
