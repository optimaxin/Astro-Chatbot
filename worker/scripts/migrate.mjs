// Applies worker/migrations/*.sql to the database in DATABASE_URL, once each,
// in order. Everything it runs lives in the `playground` schema.
//
//   DATABASE_URL=postgres://... node scripts/migrate.mjs
//
// Use the same Supabase connection string the Tredev backend uses. DDL needs the
// direct connection or the session pooler, not the transaction pooler.
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import postgres from 'postgres';

const url = process.env.DATABASE_URL;
if (!url) {
  console.error('Set DATABASE_URL first.');
  process.exit(1);
}
const local = ['localhost', '127.0.0.1', '::1'].includes(new URL(url).hostname);
const sql = postgres(url, { max: 1, ssl: local ? false : 'require', onnotice: () => {} });
const dir = join(dirname(fileURLToPath(import.meta.url)), '..', 'migrations');

try {
  await sql`create schema if not exists playground`;
  await sql`create table if not exists playground.schema_migrations (
    name text primary key, applied_at timestamptz not null default now())`;
  const done = new Set((await sql`select name from playground.schema_migrations`).map((r) => r.name));
  for (const file of readdirSync(dir).filter((f) => f.endsWith('.sql')).sort()) {
    if (done.has(file)) {
      console.log(`skip   ${file}`);
      continue;
    }
    await sql.begin(async (tx) => {
      await tx.unsafe(readFileSync(join(dir, file), 'utf8'));
      await tx`insert into playground.schema_migrations (name) values (${file})`;
    });
    console.log(`applied ${file}`);
  }
} finally {
  await sql.end({ timeout: 5 });
}
