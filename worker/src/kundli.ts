/**
 * Reference Kundli generation. No calculation engine lives here: this calls
 * the Tredev backend's own public `/api/calculators/kundli-full` — the exact
 * same calculator ai-core itself calls for the production chatbot's chart
 * facts — so the playground never risks a second, divergent implementation.
 */
import tzLookup from 'tz-lookup';
import type { Env } from './env';
import { ProviderError } from './providers';

export function resolveTimezoneOffsetMinutes(latitude: number, longitude: number, date: string, time: string): number {
  let zone: string;
  try {
    zone = tzLookup(latitude, longitude);
  } catch {
    throw new ProviderError('Timezone', 'Could not determine a timezone for that location.');
  }
  const [year, month, day] = date.split('-').map(Number) as [number, number, number];
  const [hour, minute] = time.split(':').map(Number) as [number, number];
  // Format the birth wall-clock instant (read as UTC, which is wrong by
  // exactly the offset we're solving for) in the target zone with a
  // long UTC offset — the zone's rules for that calendar date, historical
  // DST included, give back the real offset without a second dependency.
  const asOf = new Date(Date.UTC(year, month - 1, day, hour, minute));
  const parts = new Intl.DateTimeFormat('en-US', { timeZone: zone, timeZoneName: 'longOffset', hour: '2-digit' }).formatToParts(asOf);
  const offsetPart = parts.find((p) => p.type === 'timeZoneName')?.value;
  const match = offsetPart?.match(/GMT([+-])(\d{2}):?(\d{2})?/);
  if (!match) throw new ProviderError('Timezone', 'Could not compute a UTC offset for that location.');
  const sign = match[1] === '-' ? -1 : 1;
  return sign * (Number(match[2]) * 60 + Number(match[3] ?? '0'));
}

export interface ReferenceBirth {
  date: string;
  time: string;
  timezoneOffsetMinutes: number;
  latitude: number;
  longitude: number;
}

/** Raw `/api/calculators/kundli-full` payload — only the fields this playground reads are typed. */
export interface KundliFullChart {
  kundli: {
    ascendant: { rashi: string; degreeInSign: number; longitude: number };
    planets: Array<{ id: string; rashi: string; house: number; nakshatra: string; nakshatraPada: number; nakshatraLord: string; retrograde: boolean; degreeInSign: number }>;
    moonNakshatra: { name: string; pada: number; lord: string };
  };
  mahadashaTimeline: Array<{ lord: string; active: boolean; antardashas: Array<{ lord: string; active: boolean }> }>;
  doshas: unknown;
  yogas: unknown;
  navamsaChart: unknown;
  [key: string]: unknown;
}

export async function fetchKundliFull(env: Env, birth: ReferenceBirth, doFetch: typeof fetch = fetch): Promise<KundliFullChart> {
  const base = (env.TREDEV_BACKEND_URL ?? '').trim().replace(/\/+$/, '');
  if (!base) throw new ProviderError('Reference Kundli', 'TREDEV_BACKEND_URL is not set on the playground.');

  let res: Response;
  try {
    res = await doFetch(`${base}/api/calculators/kundli-full`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        date: birth.date, time: birth.time, timezoneOffsetMinutes: birth.timezoneOffsetMinutes,
        latitude: birth.latitude, longitude: birth.longitude,
      }),
      signal: AbortSignal.timeout(15000),
    });
  } catch (err) {
    const timedOut = err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError');
    throw new ProviderError('Reference Kundli', timedOut ? 'The Tredev backend did not respond in time.' : 'Could not connect to the Tredev backend.');
  }
  const raw = await res.text();
  if (!res.ok) {
    let detail = '';
    try { detail = String(JSON.parse(raw)?.error?.message ?? '').slice(0, 200); } catch { /* not JSON */ }
    throw new ProviderError('Reference Kundli', `The Tredev backend returned HTTP ${res.status}${detail ? `: ${detail}` : '.'}`, res.status);
  }
  let data: { success?: boolean; data?: KundliFullChart; error?: { message?: string } };
  try { data = JSON.parse(raw); } catch { throw new ProviderError('Reference Kundli', 'The Tredev backend returned an unreadable response.'); }
  if (!data.success || !data.data) throw new ProviderError('Reference Kundli', data.error?.message ?? 'The Tredev backend reported a failure.');
  return data.data;
}

/** Flat lookup table the verifier compares extracted claims against — one value per attribute key. */
export function chartFactTable(chart: KundliFullChart): Record<string, string | number | boolean> {
  const k = chart.kundli;
  const activeMahadasha = chart.mahadashaTimeline.find((p) => p.active);
  const activeAntardasha = activeMahadasha?.antardashas.find((a) => a.active);
  const facts: Record<string, string | number | boolean> = {
    lagna: k.ascendant.rashi,
    moonSign: k.planets.find((p) => p.id === 'moon')?.rashi ?? '',
    nakshatra: k.moonNakshatra.name,
    nakshatraPada: k.moonNakshatra.pada,
    mahadasha: activeMahadasha?.lord ?? '',
    antardasha: activeAntardasha?.lord ?? '',
  };
  for (const p of k.planets) {
    facts[`planet.${p.id}.sign`] = p.rashi;
    facts[`planet.${p.id}.house`] = p.house;
    facts[`planet.${p.id}.retrograde`] = p.retrograde;
    facts[`planet.${p.id}.nakshatra`] = p.nakshatra;
  }
  return facts;
}
