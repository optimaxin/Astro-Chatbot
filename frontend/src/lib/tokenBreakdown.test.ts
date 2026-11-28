import { describe, expect, it } from 'vitest';
import { computeTurnBreakdown } from './tokenBreakdown';

// Real shapes only — the first fixture is the exact numbers from the reported
// observation (M2): main reply 5,366 in / 88 out / 5,454 total, whole turn
// ~5,981 (2 llm_calls: the main reply + one auxiliary call). The second
// mirrors the worker test suite's own captured chatbotReply() fixture
// (turn_total_tokens: 5900, total_tokens: 5356, llm_calls: 2).
const base = {
  input_tokens: null as number | null, output_tokens: null as number | null, total_tokens: null as number | null,
  usage_source: null as string | null, metadata: null as any,
};

describe('computeTurnBreakdown', () => {
  it('separates the main reply from the auxiliary-call aggregate using the reported observation\'s own numbers', () => {
    const e = { ...base, input_tokens: 5366, output_tokens: 88, total_tokens: 5454, usage_source: 'provider',
      metadata: { ai_core: { turn_total_tokens: 5981, llm_calls: 2 } } };
    const b = computeTurnBreakdown(e);
    expect(b.main).toEqual({ input: 5366, output: 88, total: 5454, source: 'provider' });
    expect(b.auxiliary).toEqual({ calls: 1, tokens: 527 });
    expect(b.wholeTurn.tokens).toBe(5981);
  });

  it('matches the worker suite\'s own chatbotReply() fixture', () => {
    const e = { ...base, total_tokens: 5356, usage_source: 'provider',
      metadata: { ai_core: { turn_total_tokens: 5900, llm_calls: 2, turn_cost_usd: 0 } } };
    const b = computeTurnBreakdown(e);
    expect(b.auxiliary).toEqual({ calls: 1, tokens: 544 });
    expect(b.wholeTurn).toEqual({ tokens: 5900, costUsd: 0 });
  });

  it('reports zero auxiliary calls and no auxiliary tokens when only the main call ran', () => {
    const e = { ...base, total_tokens: 100, metadata: { ai_core: { turn_total_tokens: 100, llm_calls: 1 } } };
    const b = computeTurnBreakdown(e);
    expect(b.auxiliary).toEqual({ calls: 0, tokens: null });
  });

  it('never invents an auxiliary total when the turn total is missing (degraded reply, old record)', () => {
    const e = { ...base, total_tokens: 5454, metadata: { ai_core: { llm_calls: 2 } } };
    const b = computeTurnBreakdown(e);
    expect(b.auxiliary).toEqual({ calls: 1, tokens: null });
    expect(b.wholeTurn.tokens).toBeNull();
  });

  it('never invents an auxiliary total when the main total is unavailable', () => {
    const e = { ...base, total_tokens: null, usage_source: 'unavailable', metadata: { ai_core: { turn_total_tokens: 5981, llm_calls: 2 } } };
    const b = computeTurnBreakdown(e);
    expect(b.main.source).toBe('unavailable');
    expect(b.auxiliary.tokens).toBeNull();
  });

  it('keeps estimated and real sources distinct, never blending them into the aggregate', () => {
    const e = { ...base, total_tokens: 1200, usage_source: 'estimated', metadata: { ai_core: { turn_total_tokens: 1500, llm_calls: 2 } } };
    const b = computeTurnBreakdown(e);
    expect(b.main.source).toBe('estimated');
    expect(b.auxiliary).toEqual({ calls: 1, tokens: 300 });
  });

  it('handles no ai_core metadata at all (direct_provider mode) without throwing', () => {
    const e = { ...base, input_tokens: 10, output_tokens: 5, total_tokens: 15, usage_source: 'provider' };
    const b = computeTurnBreakdown(e);
    expect(b.main).toEqual({ input: 10, output: 5, total: 15, source: 'provider' });
    expect(b.auxiliary).toEqual({ calls: 0, tokens: null });
    expect(b.wholeTurn).toEqual({ tokens: null, costUsd: null });
  });
});
