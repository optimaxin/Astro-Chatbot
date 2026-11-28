-- Reference Kundli generation + chatbot response verification. Additive only;
-- nothing here touches playground.sessions/experiments/admin_sessions/audit_logs
-- beyond new foreign keys, and nothing outside the `playground` schema.

create table if not exists playground.reference_kundlis (
  id                 uuid primary key default gen_random_uuid(),
  session_id         uuid not null references playground.sessions (id),
  birth              jsonb not null,
  location_provider  text not null check (location_provider in ('aws', 'manual')),
  location_place_id  text,
  chart              jsonb not null,
  engine_source      text not null,
  status             text not null default 'current' check (status in ('current', 'outdated')),
  created_at         timestamptz not null default now()
);
create index if not exists idx_pg_refkundli_session on playground.reference_kundlis (session_id, created_at desc);
-- One "current" chart per session: a new generation flips the old one to
-- 'outdated' first (application-level, see store.ts), this just guards it.
create unique index if not exists idx_pg_refkundli_one_current
  on playground.reference_kundlis (session_id) where status = 'current';

create table if not exists playground.claim_verifications (
  id                   integer generated always as identity primary key,
  experiment_id        integer not null references playground.experiments (id),
  reference_kundli_id  uuid not null references playground.reference_kundlis (id),
  claims               jsonb not null,
  supported_count      integer not null default 0,
  contradicted_count   integer not null default 0,
  unverifiable_count   integer not null default 0,
  consistency_percent  double precision,
  evaluator            jsonb,
  created_at           timestamptz not null default now()
);
create index if not exists idx_pg_claimver_experiment on playground.claim_verifications (experiment_id);
create index if not exists idx_pg_claimver_refkundli on playground.claim_verifications (reference_kundli_id);
create index if not exists idx_pg_claimver_created on playground.claim_verifications (created_at desc);

alter table playground.reference_kundlis enable row level security;
alter table playground.claim_verifications enable row level security;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    revoke all on playground.reference_kundlis from anon;
    revoke all on playground.claim_verifications from anon;
  end if;
  if exists (select 1 from pg_roles where rolname = 'authenticated') then
    revoke all on playground.reference_kundlis from authenticated;
    revoke all on playground.claim_verifications from authenticated;
  end if;
end
$$;
