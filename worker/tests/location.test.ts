import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { awsGetPlaceReply, awsSuggestReply, Browser, makeEnv, mockUpstream, resetDb, type UpstreamCall } from './helpers';

let env: Awaited<ReturnType<typeof makeEnv>>;
let tester: Browser;
beforeEach(async () => {
  await resetDb();
  env = await makeEnv();
  tester = new Browser(env);
  await tester.start('Asha');
});
afterEach(() => vi.unstubAllGlobals());

const isSuggest = (c: UpstreamCall) => c.url.includes('/v2/suggest');
const isGetPlace = (c: UpstreamCall) => c.url.includes('/v2/place/');

describe('birthplace search (AWS Location)', () => {
  it('suggests places and resolves a pick to coordinates', async () => {
    const calls = mockUpstream((c) => (isSuggest(c) ? awsSuggestReply() : isGetPlace(c) ? awsGetPlaceReply() : undefined));
    const suggest = await tester.json('/api/playground/locations/suggest?q=new+delhi');
    expect(suggest.status).toBe(200);
    expect(suggest.data).toEqual([{ label: 'New Delhi, India', placeId: 'place-new-delhi' }]);
    expect(calls[0]!.body).toMatchObject({ QueryText: 'new delhi', BiasPosition: [78.9629, 20.5937] });
    expect(calls[0]!.url).toContain('key=aws-location-test-key-SECRET');

    const resolve = await tester.json('/api/playground/locations/resolve', { body: { placeId: 'place-new-delhi' } });
    expect(resolve.status).toBe(200);
    expect(resolve.data).toEqual({ latitude: 28.6139, longitude: 77.209, displayName: 'New Delhi, India' });
    expect(calls[1]!.url).toContain('/v2/place/place-new-delhi');
  });

  it('drops bare query-refinement items and keeps only real places', async () => {
    mockUpstream((c) => (isSuggest(c) ? awsSuggestReply() : undefined));
    const res = await tester.json('/api/playground/locations/suggest?q=new+delhi');
    expect(res.data).toHaveLength(1); // the "Did you mean" refinement item is filtered out
  });

  it('returns an empty list for a too-short query, without calling AWS', async () => {
    const calls = mockUpstream(() => awsSuggestReply());
    const res = await tester.json('/api/playground/locations/suggest?q=a');
    expect(res.data).toEqual([]);
    expect(calls).toHaveLength(0);
  });

  it('surfaces an AWS failure as a clear 502, never silently empty on resolve', async () => {
    mockUpstream(() => Response.json({ message: 'bad key' }, { status: 403 }));
    const res = await tester.json('/api/playground/locations/resolve', { body: { placeId: 'x' } });
    expect(res.status).toBe(502);
    expect(res.data.detail).toMatch(/AWS Location/);
  });

  it('is unavailable (503) when AWS Location is not configured', async () => {
    const b = new Browser(await makeEnv({ AWS_LOCATION_ENABLED: '' }));
    await b.start('Ben');
    expect((await b.json('/api/playground/locations/suggest?q=delhi')).status).toBe(503);
    expect((await b.json('/api/playground/locations/resolve', { body: { placeId: 'x' } })).status).toBe(503);
    const models = await b.json('/api/playground/models');
    expect(models.data.location).toEqual({ enabled: false });
  });
});
