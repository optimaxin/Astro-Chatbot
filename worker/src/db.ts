import postgres from 'postgres';
import type { Env } from './env';

export type Sql = ReturnType<typeof postgres>;

/**
 * One short-lived connection per request (a Worker has no long-lived process to
 * pool in). `prepare: false` keeps it compatible with Supabase's pooler.
 */
export function connect(env: Env): Sql {
  const url = env.DATABASE_URL;
  if (!url) throw new Error('DATABASE_URL is not configured.');
  const local = ['localhost', '127.0.0.1', '::1', '[::1]'].includes(new URL(url).hostname);
  return postgres(url, {
    max: 1,
    prepare: false,
    fetch_types: false,
    ssl: local ? false : 'require',
    connect_timeout: 10,
    idle_timeout: 5,
    onnotice: () => {},
  });
}
