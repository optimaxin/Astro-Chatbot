import type { Catalog, ChatTurn, Experiment, ExperimentSummary, ParamKey } from './types';

// Same origin in production (one service). In dev, Vite proxies /api to the backend.
const BASE = (import.meta.env.VITE_PLAYGROUND_API_URL || '').replace(/\/+$/, '');
const TOKEN_KEY = 'playground.token';
const CLIENT_KEY = 'playground.client';
const NAME_KEY = 'playground.tester';

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

function read(store: Storage | undefined, key: string): string {
  try { return store?.getItem(key) || ''; } catch { return ''; }
}

function write(store: Storage | undefined, key: string, value: string | null): void {
  try {
    if (value) store?.setItem(key, value);
    else store?.removeItem(key);
  } catch { /* storage unavailable: lives for this page load only */ }
}

const local = typeof localStorage === 'undefined' ? undefined : localStorage;
const session = typeof sessionStorage === 'undefined' ? undefined : sessionStorage;

export const randomId = (prefix: string) =>
  `${prefix}-${(crypto.randomUUID?.() ?? `${Date.now()}${Math.random()}`).replace(/[^a-zA-Z0-9]/g, '').slice(0, 24)}`;

let memoryClient = '';
// Anonymous, random, per browser. Scopes history; it is not an identity.
export function clientId(): string {
  let id = read(local, CLIENT_KEY) || memoryClient;
  if (!id) {
    id = randomId('c');
    memoryClient = id;
    write(local, CLIENT_KEY, id);
  }
  return id;
}

export const getToken = () => read(session, TOKEN_KEY);
export const setToken = (token: string | null) => write(session, TOKEN_KEY, token);
export const getTesterName = () => read(local, NAME_KEY);
export const setTesterName = (name: string) => write(local, NAME_KEY, name.trim() || null);

async function request<T>(path: string, init: RequestInit = {}, token = getToken()): Promise<T> {
  let res: Response;
  const headers: Record<string, string> = { 'Content-Type': 'application/json', 'X-Playground-Client': clientId() };
  if (token) headers.Authorization = `Bearer ${token}`;
  try {
    res = await fetch(`${BASE}/api/playground${path}`, { ...init, headers: { ...headers, ...init.headers } });
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

export interface ChatPayload {
  provider: string;
  model: string;
  message: string;
  system_prompt?: string;
  context?: string;
  history: ChatTurn[];
  params: Partial<Record<ParamKey, number>>;
  session_id: string;
  request_id: string;
  tester_name?: string;
}

export const api = {
  models: (token?: string) => request<Catalog>('/models', {}, token),
  chat: (payload: ChatPayload) => request<Experiment>('/chat', { method: 'POST', body: JSON.stringify(payload) }),
  evaluate: (experimentId: number) =>
    request<Experiment>('/evaluate', { method: 'POST', body: JSON.stringify({ experiment_id: experimentId }) }),
  feedback: (experimentId: number, rating: 'good' | 'bad' | null, feedback: string, editedResponse: string) =>
    request<Experiment>('/feedback', {
      method: 'POST',
      body: JSON.stringify({ experiment_id: experimentId, rating, feedback, edited_response: editedResponse }),
    }),
  experiments: () => request<ExperimentSummary[]>('/experiments?limit=100'),
  experiment: (id: number) => request<Experiment>(`/experiments/${id}`),
};
