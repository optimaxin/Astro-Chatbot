export type ParamKey = 'temperature' | 'top_p' | 'top_k' | 'max_tokens';

export interface ProviderInfo {
  id: string;
  name: string;
  configured: boolean;
  kind: 'openai' | 'aicore';
  supported_params: ParamKey[];
}

export interface ModelInfo {
  id: string;
  provider: string;
  label: string;
  pricing: { input_per_1m: number; output_per_1m: number; currency: string } | null;
}

export interface Catalog {
  providers: ProviderInfo[];
  models: ModelInfo[];
  evaluator: { provider: string; model: string } | null;
  limits?: { rate_limit_per_minute: number; max_prompt_chars: number };
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

export interface Experiment {
  id: number;
  created_at: number;
  session_id: string;
  provider: string;
  model: string;
  system_prompt: string | null;
  user_prompt: string;
  context: string | null;
  params: Partial<Record<ParamKey, number>> | null;
  effective_request: {
    provider: string;
    base_url: string;
    model: string;
    params_sent: Record<string, unknown>;
    params_ignored: string[];
    messages: { role: string; content: string }[];
  } | null;
  status: 'success' | 'failed';
  error: string | null;
  response: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
  input_cost: number | null;
  output_cost: number | null;
  total_cost: number | null;
  currency: string | null;
  latency_ms: number | null;
  provider_latency_ms: number | null;
  usage_source?: string | null;
  tester_name?: string | null;
  request_id?: string | null;
  evaluation: Evaluation | null;
  evaluation_overall: number | null;
  human_rating: 'good' | 'bad' | null;
  human_feedback: string | null;
  edited_response: string | null;
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
    ai_core?: {
      intent?: string | null;
      response_style?: string | null;
      chart?: string | null;
      degraded?: boolean | null;
      citations?: { id: string; title: string }[] | null;
      notices?: string[] | null;
    };
  } | null;
}

export interface ExperimentSummary {
  id: number;
  created_at: number;
  provider: string;
  model: string;
  user_prompt: string;
  status: 'success' | 'failed';
  total_tokens: number | null;
  total_cost: number | null;
  currency: string | null;
  latency_ms: number | null;
  evaluation_overall: number | null;
  human_rating: 'good' | 'bad' | null;
}

export interface PlaygroundConfig {
  provider: string;
  model: string;
  systemPrompt: string;
  context: string;
  params: Record<ParamKey, number>;
  enabled: Record<ParamKey, boolean>;
  autoEvaluate: boolean;
}
