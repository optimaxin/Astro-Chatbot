import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Browser, makeEnv, mockUpstream, placeNotFoundReply, placeResolveReply, placeSuggestReply, resetDb, type UpstreamCall } from './helpers';

let env: Awaited<ReturnType<typeof makeEnv>>;
let tester: Browser;
beforeEach(async () => {
  await resetDb();
  env = await makeEnv();
  tester = new Browser(env);
  await tester.start('Asha');
});
afterEach(() => vi.unstubAllGlobals());

const isSuggest = (c: UpstreamCall) => c.url.includes('/api/calculators/place-suggest');
const isResolve = (c: UpstreamCall) => c.url.includes('/api/calculators/place-resolve');

describe('birthplace search (proxied through the Tredev backend)', () => {
  it('suggests places and resolves an AWS-sourced pick to coordinates', async () => {
    const calls = mockUpstream((c) => (isSuggest(c) ? placeSuggestReply() : isResolve(c) ? placeResolveReply() : undefined));
    const suggest = await tester.json('/api/playground/locations/suggest?q=new+delhi');
    expect(suggest.status).toBe(200);
    expect(suggest.data).toEqual([{ label: 'New Delhi, India', placeId: 'place-new-delhi' }]);
    expect(calls[0]!.url).toBe('http://tredev-backend.test/api/calculators/place-suggest?q=new%20delhi');

    const resolve = await tester.json('/api/playground/locations/resolve', { body: { placeId: 'place-new-delhi' } });
    expect(resolve.status).toBe(200);
    expect(resolve.data).toEqual({ latitude: 28.6139, longitude: 77.209, displayName: 'New Delhi, India' });
    expect(calls[1]!.url).toBe('http://tredev-backend.test/api/calculators/place-resolve?placeId=place-new-delhi');
  });

  it('passes through a Nominatim-fallback suggestion that already carries coordinates (no placeId)', async () => {
    mockUpstream((c) => (isSuggest(c) ? placeSuggestReply([{ label: 'Paris, France', latitude: 48.8566, longitude: 2.3522 }]) : undefined));
    const res = await tester.json('/api/playground/locations/suggest?q=paris');
    expect(res.data).toEqual([{ label: 'Paris, France', latitude: 48.8566, longitude: 2.3522 }]);
  });

  it('returns an empty list for a too-short query, without calling the backend', async () => {
    const calls = mockUpstream(() => placeSuggestReply());
    const res = await tester.json('/api/playground/locations/suggest?q=a');
    expect(res.data).toEqual([]);
    expect(calls).toHaveLength(0);
  });

  it('surfaces a backend failure as a clear 502, never silently empty on resolve', async () => {
    mockUpstream(() => placeNotFoundReply());
    const res = await tester.json('/api/playground/locations/resolve', { body: { placeId: 'nowhere' } });
    expect(res.status).toBe(502);
    expect(res.data.detail).toMatch(/resolve that place/i);
  });

  it('is unavailable (503) when location search is not configured', async () => {
    const b = new Browser(await makeEnv({ AWS_LOCATION_ENABLED: '' }));
    await b.start('Ben');
    expect((await b.json('/api/playground/locations/suggest?q=delhi')).status).toBe(503);
    expect((await b.json('/api/playground/locations/resolve', { body: { placeId: 'x' } })).status).toBe(503);
    const models = await b.json('/api/playground/models');
    expect(models.data.location).toEqual({ enabled: false });

    const noBackend = new Browser(await makeEnv({ TREDEV_BACKEND_URL: '' }));
    await noBackend.start('Cal');
    expect((await noBackend.json('/api/playground/locations/suggest?q=delhi')).status).toBe(503);
  });
});
