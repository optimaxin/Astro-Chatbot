// Free hosting sleeps when idle: the first request after a quiet period can take up to a minute while the
// service starts, and can briefly fail while it (or the database) wakes up. Every call goes through here so the
// page waits and retries instead of showing an error, and tells the person what is happening.

type Listener = (waking: boolean) => void;
const listeners = new Set<Listener>();
let waking = false;
let wakingCalls = 0; // calls currently showing the notice, so one finishing does not hide it for the others
const set = (v: boolean) => {
  if (v !== waking) { waking = v; listeners.forEach((l) => l(v)); }
};

export const onWaking = (l: Listener) => { listeners.add(l); l(waking); return () => { listeners.delete(l); }; };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const RETRY_DELAYS_MS = [1500, 3000, 6000, 10000];
const SLOW_AFTER_MS = 4000;

/** A hosting-side or temporary-database failure: worth waiting for. Real app errors (400, 401, 429...) are not. */
function transient(res: Response): boolean {
  if (res.status !== 502 && res.status !== 503 && res.status !== 504) return false;
  // The app answers in JSON: a 502 there means an AI provider failed (retrying would just repeat the failure), and a
  // 503 for missing configuration is permanent. Only the database-unreachable 503 (Retry-After) is worth waiting for.
  // A non-JSON 502/503/504 is the hosting platform's own page while the service starts up.
  if (!(res.headers.get('content-type') ?? '').includes('json')) return true;
  return res.status === 503 && res.headers.has('Retry-After');
}

export interface Options {
  /** Show the "waking up" notice if this call is slow (short calls only; a chat reply is legitimately slow). */
  watch?: boolean;
}

export async function resilientFetch(url: string, init: RequestInit, opts: Options = {}): Promise<Response> {
  let counted = false;
  const show = () => { if (!counted) { counted = true; wakingCalls++; set(true); } };
  const timer = opts.watch ? setTimeout(show, SLOW_AFTER_MS) : undefined;
  try {
    for (let attempt = 0; ; attempt++) {
      let res: Response | undefined;
      try {
        res = await fetch(url, init);
      } catch (err) {
        if (attempt >= RETRY_DELAYS_MS.length) throw err;
      }
      if (res && !transient(res)) return res;
      if (res && attempt >= RETRY_DELAYS_MS.length) return res;
      if (opts.watch) show();
      await sleep(RETRY_DELAYS_MS[attempt]!);
    }
  } finally {
    clearTimeout(timer);
    if (counted) { wakingCalls--; set(wakingCalls > 0); }
  }
}
