import postgres from 'postgres';
import type { Env } from './env';

export type Sql = ReturnType<typeof postgres>;

/**
 * One short-lived connection per request. In production the connection string comes from
 * Hyperdrive, which pools connections and handles TLS to Supabase (a Worker cannot connect
 * to Supabase's pooler directly: its certificate is issued by Supabase's own CA). Tests and
 * local tooling use DATABASE_URL instead.
 */
export function connect(env: Env): Sql {
  const viaHyperdrive = env.HYPERDRIVE?.connectionString;
  const url = viaHyperdrive ?? env.DATABASE_URL;
  if (!url) throw new Error('No database configured (Hyperdrive binding or DATABASE_URL).');
  const local = ['localhost', '127.0.0.1', '::1', '[::1]'].includes(new URL(url).hostname);
  return postgres(url, {
    max: 1,
    prepare: false,
    fetch_types: false,
    ssl: viaHyperdrive || local ? false : 'require',
    connect_timeout: 10,
    idle_timeout: 5,
    onnotice: () => {},
  });
}
