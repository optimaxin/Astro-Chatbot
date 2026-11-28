import { resilientFetch } from './net';
import type {
  Birth, Catalog, ChatTurn, Experiment, ExperimentSummary, Mode, ParamKey, PlaceSuggestion, ReferenceKundli,
  ResolvedPlace, Tester,
} from './types';

// Same origin in production (one Worker). In dev, Vite proxies /api to `wrangler dev`.
const BASE = (import.meta.env.VITE_PLAYGROUND_API_URL || '').replace(/\/+$/, '');

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

export const randomId = (prefix: string) =>
  `${prefix}-${(crypto.randomUUID?.() ?? `${Date.now()}${Math.random()}`).replace(/[^a-zA-Z0-9]/g, '').slice(0, 24)}`;

async function request<T>(path: string, init: RequestInit = {}, watch = false): Promise<T> {
  let res: Response;
  try {
    // Identity is a server-side session in an HttpOnly cookie; nothing about the tester is sent from here.
    // Calls retry through hosting cold starts; a retried chat is safe because its request_id makes it idempotent.
    res = await resilientFetch(`${BASE}${path}`, {
      credentials: 'same-origin',
      ...init,
      headers: { 'Content-Type': 'application/json', ...init.headers },
    }, { watch });
  } catch {
    throw new ApiError('Cannot reach the playground backend.', 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }
  return res.json() as Promise<T>;
}

export const session = {
  current: () => request<Tester>('/api/session', {}, true),
  start: (name: string) => request<Tester>('/api/session', { method: 'POST', body: JSON.stringify({ name }) }, true),
  end: () => request<{ ok: boolean }>('/api/session', { method: 'DELETE' }),
};

export interface ChatPayload {
  mode: Mode;
  message: string;
  request_id: string;
  conversation_id: string;
  style?: string | null;
  birth?: Birth;
  provider?: string;
  model?: string;
  system_prompt?: string;
  context?: string;
  history?: ChatTurn[];
  params?: Partial<Record<ParamKey, number>>;
}

const P = '/api/playground';

export const api = {
  models: () => request<Catalog>(`${P}/models`, {}, true),
  chat: (payload: ChatPayload) => request<Experiment>(`${P}/chat`, { method: 'POST', body: JSON.stringify(payload) }),
  evaluate: (experimentId: number) =>
    request<Experiment>(`${P}/evaluate`, { method: 'POST', body: JSON.stringify({ experiment_id: experimentId }) }),
  feedback: (experimentId: number, rating: 'good' | 'bad' | null, feedback: string, editedResponse: string) =>
    request<Experiment>(`${P}/feedback`, {
      method: 'POST',
      body: JSON.stringify({ experiment_id: experimentId, rating, feedback, edited_response: editedResponse }),
    }),
  experiments: () => request<ExperimentSummary[]>(`${P}/experiments?limit=100`),
  experiment: (id: number) => request<Experiment>(`${P}/experiments/${id}`),
  locations: {
    suggest: (q: string) => request<PlaceSuggestion[]>(`${P}/locations/suggest?q=${encodeURIComponent(q)}`),
    resolve: (placeId: string) =>
      request<ResolvedPlace>(`${P}/locations/resolve`, { method: 'POST', body: JSON.stringify({ placeId }) }),
  },
  kundli: {
    generate: (birth: { date: string; time: string; latitude: number; longitude: number; placeLabel?: string; placeId?: string }) =>
      request<ReferenceKundli>(`${P}/kundli/generate`, { method: 'POST', body: JSON.stringify(birth) }),
    current: () => request<ReferenceKundli | null>(`${P}/kundli/current`),
  },
};
