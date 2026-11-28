export type ParamKey = 'temperature' | 'top_p' | 'top_k' | 'max_tokens';
export type Mode = 'kundli_chatbot' | 'direct_provider';

export interface ProviderInfo {
  id: string;
  name: string;
  configured: boolean;
  supported_params: ParamKey[];
}

export interface ModelInfo {
  id: string;
  provider: string;
  label: string;
  pricing: { input_per_1m: number; output_per_1m: number; currency: string } | null;
}

export interface Birth {
  date: string;
  time: string;
  timezoneOffsetMinutes: number;
  latitude: number;
  longitude: number;
  placeLabel?: string;
}

export interface Catalog {
  modes: { id: Mode; label: string; available?: boolean }[];
  chatbot: { styles: { id: string; name: string }[]; sample_birth: Birth };
  providers: ProviderInfo[];
  models: ModelInfo[];
  evaluator: { provider: string; model: string } | null;
  location?: { enabled: boolean };
  limits?: { rate_limit_per_minute: number; max_prompt_chars: number };
}

export interface PlaceSuggestion {
  label: string;
  placeId: string;
}

export interface ResolvedPlace {
  latitude: number;
  longitude: number;
  displayName: string;
}

export interface ReferenceKundliPlanet {
  id: string;
  rashi: string;
  house: number;
  nakshatra: string;
  nakshatraPada: number;
  retrograde: boolean;
  degreeInSign: number;
}

export interface ReferenceKundliChart {
  kundli: {
    ascendant: { rashi: string; degreeInSign: number };
    planets: ReferenceKundliPlanet[];
    moonNakshatra: { name: string; pada: number; lord: string };
  };
  mahadashaTimeline: Array<{ lord: string; active: boolean; antardashas: Array<{ lord: string; active: boolean }> }>;
  doshas?: Record<string, unknown>;
  yogas?: unknown;
  navamsaChart?: { ascendant: { rashi: string }; planets: { id: string; rashi: string; house: number }[] };
}

export interface ReferenceKundli {
  id: string;
  session_id: string;
  birth: Birth;
  location_provider: 'aws' | 'manual';
  location_place_id: string | null;
  chart: ReferenceKundliChart;
  engine_source: string;
  status: 'current' | 'outdated';
  created_at: number;
}

export type ClaimStatus = 'SUPPORTED' | 'CONTRADICTED' | 'UNVERIFIABLE' | 'NOT_CHART_FACT';

export interface VerifiedClaim {
  claim: string;
  attribute: string;
  statedValue: string;
  referenceValue: string | number | boolean | null;
  status: ClaimStatus;
  explanation: string;
}

export interface VerificationResult {
  id: number;
  experiment_id: number;
  reference_kundli_id: string;
  claims: VerifiedClaim[];
  supported_count: number;
  contradicted_count: number;
  unverifiable_count: number;
  consistency_percent: number | null;
  evaluator: { provider: string; model: string };
  created_at: number;
}

export interface Tester {
  session_id: string;
  tester_name: string;
}

export interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface Evaluation {
  label: string;
  scores: Record<string, number | null>;
  overall: number | null;
  feedback: string;
  strengths: string[];
  improvements: string[];
  evaluator: { provider: string; model: string };
  evaluator_usage: {
    input_tokens: number | null;
    output_tokens: number | null;
    total_tokens: number | null;
    latency_ms: number;
    total_cost: number | null;
    currency: string | null;
  };
}

export interface AiCoreMeta {
  underlying_provider?: string;
  fallback_used?: boolean;
  turn_total_tokens?: number | null;
  turn_cost_usd?: number | null;
  llm_calls?: number | null;
  intent?: string | null;
  response_style?: string | null;
  requested_style?: string | null;
  degraded?: boolean;
  notices?: string[];
  citations?: { id: string; title: string }[];
  sources?: number;
  chart_facts_used?: string[];
  messages?: string[];
  birth?: Birth;
}

export interface Experiment {
  id: number;
  created_at: number;
  request_id: string;
  session_id: string;
  tester_name: string;
  mode: Mode;
  provider: string;
  model: string;
  model_requested: string | null;
  system_prompt: string | null;
  user_prompt: string;
  context: string | null;
  params: Partial<Record<ParamKey, number>> & { response_style?: string } | null;
  effective_request: {
    mode?: Mode;
    endpoint?: string;
    provider?: string;
    model?: string;
    params_sent?: Record<string, unknown>;
    params_ignored?: string[];
    response_style?: string;
    birth?: Birth;
    messages: { role: string; content: string }[];
  } | null;
  status: 'pending' | 'success' | 'failed';
  error: string | null;
  response: string | null;
  original_response: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
  usage_source: string | null;
  input_cost: number | null;
  output_cost: number | null;
  total_cost: number | null;
  currency: string | null;
  latency_ms: number | null;
  provider_latency_ms: number | null;
  first_token_latency_ms: number | null;
  evaluation: Evaluation | null;
  evaluation_overall: number | null;
  verification?: VerificationResult | null;
  human_rating: 'good' | 'bad' | null;
  human_feedback: string | null;
  edited_response: string | null;
  pricing_snapshot: Record<string, unknown> | null;
  metadata: {
    provider_name?: string;
    model_reported?: string;
    finish_reason?: string | null;
    reasoning_tokens?: number | null;
    cached_tokens?: number | null;
    provider_reported_cost?: number | null;
    status_code?: number | null;
    prompt_breakdown_estimate?: Record<string, number>;
    overhead_ms?: number;
    cost_source?: string | null;
    ttft_note?: string;
    ai_core?: AiCoreMeta;
  } | null;
}

export interface ExperimentSummary {
  id: number;
  created_at: number;
  mode: Mode;
  provider: string;
  model: string;
  user_prompt: string;
  status: 'pending' | 'success' | 'failed';
  total_tokens: number | null;
  total_cost: number | null;
  currency: string | null;
  latency_ms: number | null;
  evaluation_overall: number | null;
  human_rating: 'good' | 'bad' | null;
}

export interface PlaygroundConfig {
  mode: Mode;
  // Kundli Chatbot
  style: string; // 'auto' = decided by the chatbot, as on the website
  birth: Birth;
  placeId?: string; // AWS-resolved birthplace, carried into Reference Kundli generation only
  // Direct provider
  provider: string;
  model: string;
  systemPrompt: string;
  context: string;
  params: Record<ParamKey, number>;
  enabled: Record<ParamKey, boolean>;
  autoEvaluate: boolean;
}
