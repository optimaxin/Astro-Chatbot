/**
 * Birthplace search via Amazon Location Service's Geo Places API (Suggest +
 * GetPlace), authenticated with an API key (no SigV4/IAM signing needed —
 * the key travels as a query param). A separate, playground-only
 * integration: it does not call or import the Tredev backend's own
 * geocoding service.
 */
import type { Env } from './env';
import { ProviderError } from './providers';

export interface PlaceSuggestion {
  label: string;
  placeId: string;
}

export interface ResolvedPlace {
  latitude: number;
  longitude: number;
  displayName: string;
}

type Fetch = typeof fetch;

// Suggest requires exactly one of BiasPosition/Filter.BoundingBox/Filter.Circle
// (a 400 otherwise); this playground's testers are predominantly asking about
// Indian birthplaces, so biasing toward India's geographic center improves
// ranking for short/ambiguous queries without excluding anywhere else.
const INDIA_CENTER: [number, number] = [78.9629, 20.5937];

export function locationEnabled(env: Env): boolean {
  return (env.AWS_LOCATION_ENABLED ?? '').trim().toLowerCase() === 'true' && Boolean(env.AWS_LOCATION_API_KEY?.trim());
}

function region(env: Env): string {
  return (env.AWS_LOCATION_REGION ?? '').trim() || 'ap-south-1';
}

function requireKey(env: Env): string {
  const key = (env.AWS_LOCATION_API_KEY ?? '').trim();
  if (!key) throw new ProviderError('AWS Location', 'Location search is not configured on this playground.');
  return key;
}

async function call(url: string, init: RequestInit, doFetch: Fetch): Promise<Response> {
  let res: Response;
  try {
    res = await doFetch(url, { ...init, signal: AbortSignal.timeout(5000) });
  } catch (err) {
    const timedOut = err instanceof Error && (err.name === 'TimeoutError' || err.name === 'AbortError');
    throw new ProviderError('AWS Location', timedOut ? 'AWS Location Service did not respond in time.' : 'Could not connect to AWS Location Service.');
  }
  if (!res.ok) throw new ProviderError('AWS Location', `AWS Location Service request failed (HTTP ${res.status}).`, res.status);
  return res;
}

export async function suggestPlaces(env: Env, query: string, doFetch: Fetch = fetch): Promise<PlaceSuggestion[]> {
  if (!locationEnabled(env)) return [];
  const key = requireKey(env);
  const url = `https://places.geo.${region(env)}.api.aws/v2/suggest?key=${encodeURIComponent(key)}`;
  const res = await call(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ QueryText: query, MaxResults: 6, BiasPosition: INDIA_CENTER }),
  }, doFetch);
  const body = (await res.json()) as {
    ResultItems?: Array<{
      Title?: string;
      SuggestResultItemType?: string;
      Place?: { PlaceId?: string; Address?: { Label?: string } };
    }>;
  };
  // Suggest also returns bare "query refinement" items (a corrected search
  // term with no place attached) alongside real place candidates — only the
  // latter (type "Place", with a PlaceId) are selectable here.
  return (body.ResultItems ?? [])
    .filter((item) => item.SuggestResultItemType === 'Place' && item.Place?.PlaceId)
    .map((item) => ({ label: item.Place!.Address?.Label || item.Title || query, placeId: item.Place!.PlaceId! }));
}

export async function resolvePlace(env: Env, placeId: string, doFetch: Fetch = fetch): Promise<ResolvedPlace> {
  const key = requireKey(env);
  const url = `https://places.geo.${region(env)}.api.aws/v2/place/${encodeURIComponent(placeId)}?key=${encodeURIComponent(key)}`;
  const res = await call(url, { method: 'GET' }, doFetch);
  const body = (await res.json()) as { Position?: [number, number]; Address?: { Label?: string }; Title?: string };
  if (!body.Position) throw new ProviderError('AWS Location', 'That place could not be resolved to coordinates.');
  const [longitude, latitude] = body.Position; // AWS returns [lng, lat] (WGS 84), the opposite order from most other providers
  return { latitude, longitude, displayName: body.Address?.Label || body.Title || '' };
}
