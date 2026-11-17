import postgres from 'postgres';
import type { Env } from './env';

export type Sql = ReturnType<typeof postgres>;

function connectionString(env: Env): { url: string; viaHyperdrive: boolean } {
  const viaHyperdrive = env.HYPERDRIVE?.connectionString;
  const url = viaHyperdrive ?? env.DATABASE_URL;
  if (!url) throw new Error('No database configured (DATABASE_URL, or a Hyperdrive binding on Cloudflare).');
  return { url, viaHyperdrive: Boolean(viaHyperdrive) };
}

function options(url: string, viaHyperdrive: boolean, max: number): postgres.Options<Record<string, never>> {
  const parsed = new URL(url);
  const local = ['localhost', '127.0.0.1', '::1', '[::1]'].includes(parsed.hostname);
  // An explicit ?sslmode=... in the connection string wins; otherwise remote databases are always encrypted.
  const sslOption = parsed.searchParams.has('sslmode') && !viaHyperdrive ? {} : { ssl: viaHyperdrive || local ? (false as const) : ('require' as const) };
  return {
    max,
    // Supabase's pooler runs in transaction mode on port 6543, which has no prepared statements.
    prepare: false,
    fetch_types: false,
    // `require` encrypts the connection without verifying the certificate: Supabase's pooler certificate
    // is issued by Supabase's own CA, which Node does not trust by default.
    ...sslOption,
    connect_timeout: 15,
    onnotice: () => {},
  };
}

/**
 * One short-lived connection per request (Cloudflare Workers, tests). In production on a Worker the
 * connection string comes from Hyperdrive, which pools connections and handles TLS to Supabase.
 */
export function connect(env: Env): Sql {
  const { url, viaHyperdrive } = connectionString(env);
  return postgres(url, { ...options(url, viaHyperdrive, 1), idle_timeout: 5 });
}

/**
 * A shared pool for a long-running Node process (Render). Connections open lazily on first use, so
 * starting the server never waits on the database. Idle connections are closed after 20 s so a free
 * Supabase pooler is not held open, and every connection is recycled after 30 minutes. A connection
 * the server dropped is discarded and the next query opens a new one.
 */
export function createPool(env: Env): Sql {
  const { url, viaHyperdrive } = connectionString(env);
  const max = Math.min(Math.max(Math.trunc(Number(env.DB_POOL_MAX)) || 5, 1), 20);
  return postgres(url, { ...options(url, viaHyperdrive, max), idle_timeout: 20, max_lifetime: 60 * 30 });
}
