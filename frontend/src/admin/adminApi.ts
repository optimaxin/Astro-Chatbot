const BASE = (import.meta.env.VITE_PLAYGROUND_API_URL || '').replace(/\/+$/, '');
let csrf = '';

export class AdminError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

export type Costs = Record<string, number>;

export interface Totals {
  requests: number; success: number; failed: number;
  input_tokens: number; output_tokens: number; total_tokens: number;
  cost: Costs; avg_latency_ms: number | null; unique_sessions: number; unique_clients?: number;
}

export interface Analytics {
  totals: Totals;
  by_model: (Totals & { provider: string; model: string })[];
  by_provider: (Totals & { provider: string })[];
  daily: (Totals & { date: string })[];
  failures: { error: string; count: number }[];
  sessions: number;
}

export interface Row {
  id: number; created_at: number; request_id: string | null; session_id: string | null;
  client_id: string | null; tester_name: string | null; provider: string; model: string;
  prompt_preview: string; response_preview: string | null;
  input_tokens: number | null; output_tokens: number | null; total_tokens: number | null;
  total_cost: number | null; currency: string | null; latency_ms: number | null;
  status: 'success' | 'failed'; error: string | null; evaluation_overall: number | null; human_rating: string | null;
}

export interface Page<T> { items: T[]; total: number; page: number; page_size: number }

export interface AuditEntry { id: number; action: string; timestamp: number; ip: string | null; details: Record<string, unknown> }

export type Filters = Partial<Record<'from' | 'to' | 'provider' | 'model' | 'status' | 'session_id' | 'q', string>>;

export const query = (params: Record<string, string | number | undefined>) => {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== '') q.set(k, String(v)); });
  const s = q.toString();
  return s ? `?${s}` : '';
};

async function call(path: string, init: RequestInit = {}): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/api/admin${path}`, {
      credentials: 'same-origin',
      ...init,
      headers: { 'Content-Type': 'application/json', ...(csrf ? { 'X-CSRF-Token': csrf } : {}), ...init.headers },
    });
  } catch {
    throw new AdminError('Cannot reach the playground backend.', 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { const b = await res.json(); detail = typeof b.detail === 'string' ? b.detail : detail; } catch { /* */ }
    throw new AdminError(detail || `Request failed (${res.status})`, res.status);
  }
  return res;
}

const json = async <T,>(path: string, init?: RequestInit) => (await call(path, init)).json() as Promise<T>;

async function download(path: string, fallbackName: string): Promise<void> {
  const res = await call(path);
  const name = /filename="([^"]+)"/.exec(res.headers.get('content-disposition') || '')?.[1] || fallbackName;
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export const adminApi = {
  me: async () => { csrf = (await json<{ csrf_token: string }>('/me')).csrf_token; },
  login: async (password: string) => {
    csrf = (await json<{ csrf_token: string }>('/login', { method: 'POST', body: JSON.stringify({ password }) })).csrf_token;
  },
  logout: async () => { await call('/logout', { method: 'POST' }); csrf = ''; },
  analytics: (f: Filters) => json<Analytics>(`/analytics${query(f)}`),
  experiments: (f: Filters, page: number, pageSize: number) =>
    json<Page<Row>>(`/experiments${query({ ...f, page, page_size: pageSize })}`),
  experiment: (id: number) => json<Record<string, unknown>>(`/experiments/${id}`),
  models: () => json<{ seen: { providers: string[]; models: string[] } }>('/models'),
  audit: (page: number) => json<Page<AuditEntry>>(`/audit-logs${query({ page, page_size: 50 })}`),
  exportExcel: (f: Filters) => download(`/export/excel${query(f)}`, 'AI_Playground_Company_Testing_Logs.xlsx'),
  backup: () => download('/backup', 'playground-backup.db'),
};
