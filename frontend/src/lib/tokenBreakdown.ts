// Token-meter accuracy: separates the main chatbot reply's own tokens from
// the aggregate of whatever auxiliary calls (intent classifier, quality
// check, humanizer, ...) ran in the same turn, instead of blurring both into
// one "whole turn" figure. ai-core reports only an aggregate for those
// auxiliary calls today (not broken down per call type) — this never
// invents a split it doesn't have; it stays null/unavailable rather than
// guessing.
import type { Experiment } from '../types';

export interface MainBreakdown {
  input: number | null;
  output: number | null;
  total: number | null;
  source: string | null;
}

export interface AuxiliaryBreakdown {
  /** Calls other than the main reply (classifier, quality check, ...), as an aggregate — ai-core does not report which call used how many tokens. */
  calls: number;
  tokens: number | null;
}

export interface WholeTurnBreakdown {
  tokens: number | null;
  costUsd: number | null;
}

export interface TurnBreakdown {
  main: MainBreakdown;
  auxiliary: AuxiliaryBreakdown;
  wholeTurn: WholeTurnBreakdown;
}

export function computeTurnBreakdown(e: Pick<Experiment, 'input_tokens' | 'output_tokens' | 'total_tokens' | 'usage_source' | 'metadata'>): TurnBreakdown {
  const core = e.metadata?.ai_core;
  const main: MainBreakdown = { input: e.input_tokens, output: e.output_tokens, total: e.total_tokens, source: e.usage_source ?? null };

  const llmCalls = core?.llm_calls ?? null;
  const calls = llmCalls != null ? Math.max(0, llmCalls - 1) : 0;
  const tokens = calls > 0 && core?.turn_total_tokens != null && e.total_tokens != null
    ? Math.max(0, core.turn_total_tokens - e.total_tokens)
    : null;

  return {
    main,
    auxiliary: { calls, tokens },
    wholeTurn: { tokens: core?.turn_total_tokens ?? null, costUsd: core?.turn_cost_usd ?? null },
  };
}
