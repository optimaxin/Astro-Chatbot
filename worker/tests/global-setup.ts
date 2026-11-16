import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import postgres from 'postgres';

export const DB_URL = process.env.TEST_DATABASE_URL ?? 'postgres://postgres:postgres@localhost:55432/pgtest';

/** Applies the real migration to the throwaway test database before any test runs. */
export default async function setup() {
  const sql = postgres(DB_URL, { max: 1, onnotice: () => {} });
  try {
    await sql`select 1`;
  } catch {
    await sql.end();
    throw new Error(
      `Cannot reach the test database at ${DB_URL}.\n` +
      'Start one with:  docker run -d --name pgplay-test -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=pgtest -p 55432:5432 postgres:16-alpine\n' +
      'or set TEST_DATABASE_URL. Never point it at a real database: tests truncate the playground tables.');
  }
  const dir = join(__dirname, '..', 'migrations');
  for (const file of readdirSync(dir).filter((f) => f.endsWith('.sql')).sort()) {
    await sql.unsafe(readFileSync(join(dir, file), 'utf8'));
  }
  await sql.end();
}
