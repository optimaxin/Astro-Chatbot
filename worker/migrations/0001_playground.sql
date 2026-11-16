-- AI Model Playground — dedicated schema. Everything lives in `playground`;
-- nothing here reads or changes any existing Tredev Astro table.
-- Safe to run more than once.

create schema if not exists playground;

-- One row per tester visit. `token_hash` is the SHA-256 of the random token held
-- in the tester's session cookie; the token itself is never stored.
create table if not exists playground.sessions (
  id                    uuid primary key default gen_random_uuid(),
  token_hash            text not null unique,
  tester_name           text not null check (char_length(tester_name) between 1 and 80),
  ip_hash               text,
  -- Anonymous identity issued by the Kundli chatbot, so one tester's
  -- conversations continue across requests.
  aicore_session_token  text,
  created_at            timestamptz not null default now(),
  last_seen_at          timestamptz not null default now(),
  ended_at              timestamptz
);
create index if not exists idx_pg_sessions_created on playground.sessions (created_at desc);
create index if not exists idx_pg_sessions_ip on playground.sessions (ip_hash, created_at);

create table if not exists playground.experiments (
  id                     integer generated always as identity primary key,
  request_id             text not null unique,
  session_id             uuid not null references playground.sessions (id),
  tester_name            text not null,
  ip_hash                text,
  mode                   text not null check (mode in ('kundli_chatbot', 'direct_provider')),
  -- For Kundli Chatbot runs these are the provider and model the chatbot
  -- actually used internally, as reported by its own response.
  provider               text not null,
  model                  text not null,
  model_requested        text,
  system_prompt          text,
  user_prompt            text not null,
  context                text,
  params                 jsonb,
  effective_request      jsonb,
  status                 text not null check (status in ('pending', 'success', 'failed')),
  error                  text,
  response               text,
  original_response      text,
  humanized_response     text,
  input_tokens           integer,
  output_tokens          integer,
  total_tokens           integer,
  usage_source           text,
  input_cost             double precision,
  output_cost            double precision,
  total_cost             double precision,
  currency               text,
  latency_ms             integer,
  provider_latency_ms    integer,
  first_token_latency_ms integer,
  evaluation             jsonb,
  evaluation_overall     double precision,
  human_rating           text check (human_rating in ('good', 'bad')),
  human_feedback         text,
  edited_response        text,
  feedback_at            timestamptz,
  pricing_snapshot       jsonb,
  metadata               jsonb,
  created_at             timestamptz not null default now()
);
create index if not exists idx_pg_exp_session on playground.experiments (session_id, id desc);
create index if not exists idx_pg_exp_created on playground.experiments (created_at desc);
create index if not exists idx_pg_exp_tester on playground.experiments (tester_name);
create index if not exists idx_pg_exp_model on playground.experiments (provider, model);
create index if not exists idx_pg_exp_mode on playground.experiments (mode);
create index if not exists idx_pg_exp_status on playground.experiments (status);
create index if not exists idx_pg_exp_ip on playground.experiments (ip_hash, created_at);

create table if not exists playground.admin_sessions (
  id                 integer generated always as identity primary key,
  session_token_hash text not null unique,
  expires_at         timestamptz not null,
  created_at         timestamptz not null default now()
);

create table if not exists playground.audit_logs (
  id        integer generated always as identity primary key,
  action    text not null,
  timestamp timestamptz not null default now(),
  ip        text,
  details   jsonb not null default '{}'::jsonb
);
create index if not exists idx_pg_audit_time on playground.audit_logs (timestamp desc);
create index if not exists idx_pg_audit_login on playground.audit_logs (action, ip, timestamp);

-- Defence in depth. The Worker connects with a database role that bypasses
-- RLS; enabling it with no policies means the Supabase REST/GraphQL APIs can
-- never expose these tables, even if the schema is later added to the API.
alter table playground.sessions enable row level security;
alter table playground.experiments enable row level security;
alter table playground.admin_sessions enable row level security;
alter table playground.audit_logs enable row level security;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    revoke all on schema playground from anon;
    revoke all on all tables in schema playground from anon;
  end if;
  if exists (select 1 from pg_roles where rolname = 'authenticated') then
    revoke all on schema playground from authenticated;
    revoke all on all tables in schema playground from authenticated;
  end if;
end
$$;
