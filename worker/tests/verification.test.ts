import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { KundliFullChart } from '../src/kundli';
import { compareClaims, summarize, type ExtractedClaim } from '../src/verification';
import {
  Browser, chatbotReply, generateReferenceKundli, kundliChat, kundliFullReply, makeEnv, mockUpstream, resetDb, sql,
  type UpstreamCall,
} from './helpers';

const CHART = kundliFullReply().data as unknown as KundliFullChart;

describe('compareClaims (deterministic — never lets an extraction artifact mask a real contradiction)', () => {
  it('supports a claim that matches the reference chart exactly', () => {
    const claims: ExtractedClaim[] = [{ claim: 'Your ascendant is Leo', attribute: 'lagna', statedValue: 'Leo' }];
    expect(compareClaims(claims, CHART)[0]).toMatchObject({ status: 'SUPPORTED', referenceValue: 'Leo' });
  });

  it('contradicts an incorrect planetary claim, even phrased casually', () => {
    const claims: ExtractedClaim[] = [{ claim: 'Your Moon is in Gemini', attribute: 'moonSign', statedValue: 'gemini' }];
    expect(compareClaims(claims, CHART)[0]).toMatchObject({ status: 'CONTRADICTED', referenceValue: 'Taurus' });
  });

  it('contradicts an incorrect house placement', () => {
    const claims: ExtractedClaim[] = [{ claim: 'Mars sits in your 7th house', attribute: 'planet.mars.house', statedValue: '7' }];
    expect(compareClaims(claims, CHART)[0]).toMatchObject({ status: 'CONTRADICTED', referenceValue: 4 });
  });

  it('supports a correct house placement', () => {
    const claims: ExtractedClaim[] = [{ claim: 'Mars sits in your 4th house', attribute: 'planet.mars.house', statedValue: '4' }];
    expect(compareClaims(claims, CHART)[0]).toMatchObject({ status: 'SUPPORTED' });
  });

  it('supports and contradicts retrograde claims correctly', () => {
    const claims: ExtractedClaim[] = [
      { claim: 'Mercury is retrograde', attribute: 'planet.mercury.retrograde', statedValue: 'true' },
      { claim: 'Venus is retrograde', attribute: 'planet.venus.retrograde', statedValue: 'retrograde' },
    ];
    const [mercury, venus] = compareClaims(claims, CHART);
    expect(mercury).toMatchObject({ status: 'SUPPORTED' }); // Mercury really is retrograde in the fixture
    expect(venus).toMatchObject({ status: 'CONTRADICTED' }); // Venus is not
  });

  it('marks a claim about a chart detail this chart does not have as unverifiable', () => {
    const noKetu: KundliFullChart = { ...CHART, kundli: { ...CHART.kundli, planets: CHART.kundli.planets.filter(p => p.id !== 'ketu') } };
    const claims: ExtractedClaim[] = [{ claim: 'Ketu is in your 5th house', attribute: 'planet.ketu.house', statedValue: '5' }];
    expect(compareClaims(claims, noKetu)[0]).toMatchObject({ status: 'UNVERIFIABLE' });
  });

  it('marks an unrecognised attribute as not a chart fact, never as a pass', () => {
    const claims: ExtractedClaim[] = [{ claim: 'You will have a great year', attribute: 'yearly_prediction', statedValue: 'great' }];
    expect(compareClaims(claims, CHART)[0]).toMatchObject({ status: 'NOT_CHART_FACT' });
  });

  it('checks every claim in a multi-claim paragraph independently', () => {
    const claims: ExtractedClaim[] = [
      { claim: 'ascendant Leo', attribute: 'lagna', statedValue: 'Leo' },
      { claim: 'moon Gemini', attribute: 'moonSign', statedValue: 'Gemini' },
      { claim: 'mahadasha Moon', attribute: 'mahadasha', statedValue: 'Moon' },
    ];
    const results = compareClaims(claims, CHART);
    expect(results.map(r => r.status)).toEqual(['SUPPORTED', 'CONTRADICTED', 'SUPPORTED']);
  });

  it('summarizes consistency as supported / (supported + contradicted), excluding unverifiable and non-chart-fact claims', () => {
    const claims: ExtractedClaim[] = [
      { claim: 'a', attribute: 'lagna', statedValue: 'Leo' }, // supported
      { claim: 'b', attribute: 'moonSign', statedValue: 'Gemini' }, // contradicted
      { claim: 'c', attribute: 'planet.ketu.house', statedValue: '5' }, // unverifiable once ketu is missing
      { claim: 'd', attribute: 'yearly_prediction', statedValue: 'x' }, // not a chart fact
    ];
    const noKetu: KundliFullChart = { ...CHART, kundli: { ...CHART.kundli, planets: CHART.kundli.planets.filter(p => p.id !== 'ketu') } };
    const verified = compareClaims(claims, noKetu);
    const s = summarize(verified);
    expect(s).toMatchObject({ supportedCount: 1, contradictedCount: 1, unverifiableCount: 1, consistencyPercent: 50 });
  });

  it('reports insufficient evidence, never a fake 100%, when there is nothing verifiable to score', () => {
    const verified = compareClaims([{ claim: 'a', attribute: 'yearly_prediction', statedValue: 'x' }], CHART);
    expect(summarize(verified).consistencyPercent).toBeNull();
  });
});

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
const isKundliFull = (c: UpstreamCall) => c.url.endsWith('/api/calculators/kundli-full');
const isExtractor = (c: UpstreamCall) => c.url.endsWith('/chat/completions');

describe('end-to-end: a chat turn is verified against the session\'s current reference chart', () => {
  it('attaches a verification result to the chat response and stores it', async () => {
    mockUpstream((c) => isKundliFull(c) ? Response.json(kundliFullReply()) : undefined);
    const chart = await generateReferenceKundli(tester);
    expect(chart.status).toBe(201);

    const claimsJson = JSON.stringify([
      { claim: 'Your ascendant is Leo', attribute: 'lagna', statedValue: 'Leo' },
      { claim: 'Your Moon is in Gemini', attribute: 'moonSign', statedValue: 'Gemini' },
    ]);
    mockUpstream((c) => {
      if (isChatbot(c)) return Response.json(chatbotReply({ answer: 'Your ascendant is Leo, and your Moon is in Gemini.' }));
      if (isExtractor(c)) return Response.json({ model: 'reported-model', choices: [{ message: { content: claimsJson } }] });
      return undefined;
    });
    const chat = await kundliChat(tester, 'Tell me about my chart');
    expect(chat.status).toBe(200);
    expect(chat.data.verification).toMatchObject({ supported_count: 1, contradicted_count: 1, consistency_percent: 50 });
    expect(chat.data.verification.claims).toHaveLength(2);

    const [row] = await sql`select * from playground.claim_verifications where experiment_id = ${chat.data.id}`;
    expect(row).toMatchObject({ supported_count: 1, contradicted_count: 1 });

    // a retry of the same request_id returns the same stored verification, not a fresh (costly) one
    const retryCalls = mockUpstream(() => { throw new Error('should not be called again'); });
    const retry = await kundliChat(tester, 'Tell me about my chart', { request_id: chat.data.request_id });
    expect(retry.data.verification.id).toBe(chat.data.verification.id);
    expect(retryCalls).toHaveLength(0);
  });

  it('never fails the chat turn when there is no reference chart yet', async () => {
    mockUpstream((c) => (isChatbot(c) ? Response.json(chatbotReply()) : undefined));
    const chat = await kundliChat(tester);
    expect(chat.status).toBe(200);
    expect(chat.data.status).toBe('success');
    expect(chat.data.verification).toBeUndefined();
  });

  it('never fails the chat turn when the claim extractor itself errors', async () => {
    mockUpstream((c) => {
      if (isChatbot(c)) return Response.json(chatbotReply());
      if (isKundliFull(c)) return Response.json(kundliFullReply());
      return undefined; // the extractor call falls through to a 404, simulating its failure
    });
    expect((await generateReferenceKundli(tester)).status).toBe(201);
    const chat = await kundliChat(tester);
    expect(chat.status).toBe(200);
    expect(chat.data.status).toBe('success');
    expect(chat.data.verification).toBeUndefined();
  });
});
