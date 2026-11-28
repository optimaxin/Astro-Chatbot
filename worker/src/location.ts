/**
 * Birthplace search: proxies the Tredev backend's own, already-deployed
 * geocoding endpoints (`/api/calculators/place-suggest`, `/place-resolve`),
 * which hold the real AWS Location Service key server-side (with a Nominatim
 * fallback). The playground never holds an AWS key of its own — it reuses
 * `TREDEV_BACKEND_URL`, the same secret already used for Reference Kundli.
 */
import type { Env } from './env';
import { ProviderError } from './providers';

export interface PlaceSuggestion {
  label: string;
  /** Present for AWS-sourced suggestions — resolve via resolvePlace(). */
  placeId?: string;
  /** Present for Nominatim-sourced suggestions (AWS not configured on the backend) — already final. */
  latitude?: number;
  longitude?: number;
}

export interface ResolvedPlace {
  latitude: number;
  longitude: number;
  displayName: string;
}

type Fetch = typeof fetch;

export function locationEnabled(env: Env): boolean {
  return (env.AWS_LOCATION_ENABLED ?? '').trim().toLowerCase() === 'true' && Boolean(env.TREDEV_BACKEND_URL?.trim());
}

function backendBase(env: Env): string {
  const base = (env.TREDEV_BACKEND_URL ?? '').trim().replace(/\/+$/, '');
  if (!base) throw new ProviderError('Location search', 'TREDEV_BACKEND_URL is not set on this playground.');
  return base;
}

async function call(url: string, doFetch: Fetch): Promise<unknown> {
  let res: Response;
  try {
    res = await doFetch(url, { signal: AbortSignal.timeout(6000) });
  } catch (err) {
    const timedOut = err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError');
    throw new ProviderError('Location search', timedOut ? 'The Tredev backend did not respond in time.' : 'Could not connect to the Tredev backend.');
  }
  const raw = await res.text();
  let data: { success?: boolean; data?: unknown; error?: { message?: string } };
  try { data = JSON.parse(raw); } catch { throw new ProviderError('Location search', 'The Tredev backend returned an unreadable response.'); }
  if (!res.ok || !data.success) {
    throw new ProviderError('Location search', data.error?.message ?? `The Tredev backend returned HTTP ${res.status}.`, res.status);
  }
  return data.data;
}

export async function suggestPlaces(env: Env, query: string, doFetch: Fetch = fetch): Promise<PlaceSuggestion[]> {
  const base = backendBase(env);
  const items = await call(`${base}/api/calculators/place-suggest?q=${encodeURIComponent(query)}`, doFetch) as PlaceSuggestion[];
  return items.map((i) => ({ label: i.label, placeId: i.placeId, latitude: i.latitude, longitude: i.longitude }));
}

export async function resolvePlace(env: Env, placeId: string, doFetch: Fetch = fetch): Promise<ResolvedPlace> {
  const base = backendBase(env);
  return await call(`${base}/api/calculators/place-resolve?placeId=${encodeURIComponent(placeId)}`, doFetch) as ResolvedPlace;
}
