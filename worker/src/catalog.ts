import raw from './models.json';
import type { Env } from './env';

export const PARAM_KEYS = ['temperature', 'top_p', 'top_k', 'max_tokens'] as const;
export type ParamKey = (typeof PARAM_KEYS)[number];

export interface Pricing {
  input_per_1m: number;
  output_per_1m: number;
  currency: string;
  effective_from?: string | null;
}

export interface ProviderConfig {
  id: string;
  name: string;
  base_url: string;
  api_key_env: 'NVIDIA_API_KEY' | 'GEMINI_API_KEY' | 'OPENROUTER_API_KEY';
  supported_params: ParamKey[];
  extra_body?: Record<string, unknown>;
}

export interface ModelConfig {
  id: string;
  provider: string;
  label: string;
  pricing: Pricing | null;
  request_overrides?: Record<string, unknown>;
  extra_body?: Record<string, unknown>;
}

export interface Birth {
  date: string;
  time: string;
  timezoneOffsetMinutes: number;
  latitude: number;
  longitude: number;
  name?: string;
  placeLabel?: string;
}

export interface Catalog {
  chatbot: { label: string; fallback_styles: { id: string; name: string }[]; sample_birth: Birth };
  providers: ProviderConfig[];
  models: ModelConfig[];
  evaluator: { provider: string; model: string } | null;
}

export const catalog = raw as unknown as Catalog;

export function findProvider(id: string): ProviderConfig | undefined {
  return catalog.providers.find((p) => p.id === id);
}

export function findModel(provider: string, id: string): ModelConfig | undefined {
  return catalog.models.find((m) => m.provider === provider && m.id === id);
}

export function providerKey(env: Env, provider: ProviderConfig): string {
  return (env[provider.api_key_env] ?? '').trim();
}

export function providerConfigured(env: Env, provider: ProviderConfig): boolean {
  return providerKey(env, provider) !== '';
}

export interface Cost {
  input_cost: number | null;
  output_cost: number | null;
  total_cost: number | null;
  currency: string | null;
}

/** Cost from actual token counts. Any unknown input yields null, never 0. */
export function calculateCost(pricing: Pricing | null, input: number | null, output: number | null): Cost {
  if (!pricing || input == null || output == null) {
    return { input_cost: null, output_cost: null, total_cost: null, currency: pricing?.currency ?? null };
  }
  const inputCost = (input * pricing.input_per_1m) / 1_000_000;
  const outputCost = (output * pricing.output_per_1m) / 1_000_000;
  const round = (n: number) => Math.round(n * 1e10) / 1e10;
  return {
    input_cost: round(inputCost),
    output_cost: round(outputCost),
    total_cost: round(inputCost + outputCost),
    currency: pricing.currency,
  };
}
