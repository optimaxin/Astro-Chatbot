import type { Experiment } from '../types';
import { fmtCost, fmtInt, fmtLatency } from '../format';
import { computeTurnBreakdown } from '../lib/tokenBreakdown';

interface Props {
  experiment: Experiment | null;
  loading: boolean;
}

const BREAKDOWN_LABELS: Record<string, string> = {
  system_prompt: 'System',
  context: 'Context',
  history: 'History',
  user_message: 'Message',
};

function Bar({ value, max, tone }: { value: number | null; max: number; tone: string }) {
  const pct = value == null || max <= 0 ? 0 : Math.max(2, Math.round((value / max) * 100));
  return (
    <div className="bar" aria-hidden="true">
      <div className={`bar-fill ${tone}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

const SOURCE_LABEL: Record<string, string> = {
  provider: 'reported by the provider',
  estimated: 'estimated (~4 characters per token), not provider-reported',
  unavailable: 'not reported',
};

export default function TokenMeter({ experiment: e, loading }: Props) {
  if (!e) {
    return (
      <section className="section meter">
        <h2 className="section-title">Token meter</h2>
        <p className="empty">{loading ? 'Generating…' : 'Send a message to see token usage, cost and latency.'}</p>
      </section>
    );
  }

  const meta = e.metadata ?? {};
  const core = meta.ai_core;
  const kundli = e.mode === 'kundli_chatbot';
  const tokensKnown = e.input_tokens != null && e.output_tokens != null;
  const max = Math.max(e.input_tokens ?? 0, e.output_tokens ?? 0);
  const breakdown = meta.prompt_breakdown_estimate ?? {};
  const breakdownTotal = Object.values(breakdown).reduce((a, b) => a + b, 0);

  return (
    <section className={`section meter ${loading ? 'is-stale' : ''}`}>
      <div className="section-head">
        <h2 className="section-title">Token meter</h2>
        <span className="chip mono">#{e.id}</span>
      </div>
      <p className="meter-model mono">
        {kundli ? 'Kundli Chatbot' : meta.provider_name ?? e.provider}
        {e.provider !== 'unknown' ? ` · ${kundli ? `${e.provider} / ` : ''}${e.model}` : ''}
      </p>
      {kundli && <p className="hint">Provider and model are what the chatbot itself reports it used for this reply.</p>}

      {e.status === 'failed' ? (
        <div className="notice error">
          <strong>Generation failed</strong>
          <span>Source: {kundli ? 'Kundli Chatbot' : meta.provider_name ?? e.provider}</span>
          <span>Reason: {e.error}</span>
          <span className="muted">
            {tokensKnown ? 'The reply is saved, but it was flagged as a failure.' : 'Token data unavailable. Try again or switch mode.'}
          </span>
        </div>
      ) : !tokensKnown ? (
        <div className="notice">The {kundli ? 'chatbot' : 'provider'} did not report token usage for this request. Nothing is estimated here.</div>
      ) : null}

      <div className="metric-grid">
        <div className="metric" title="Includes the system prompt, chart facts, conversation history and your message — everything the model reads before it answers, which is usually far more text than the reply itself.">
          <span className="metric-label">Input tokens</span>
          <span className="metric-value mono">{fmtInt(e.input_tokens)}</span>
          <Bar value={e.input_tokens} max={max} tone="tone-in" />
        </div>
        <div className="metric">
          <span className="metric-label">Output tokens</span>
          <span className="metric-value mono">{fmtInt(e.output_tokens)}</span>
          <Bar value={e.output_tokens} max={max} tone="tone-out" />
        </div>
        <div className="metric wide">
          <span className="metric-label">{kundli ? 'Main chatbot response' : 'Total tokens'}</span>
          <span className="metric-value big mono">{fmtInt(e.total_tokens)}</span>
          {meta.reasoning_tokens ? (
            <span className="hint mono">incl. {fmtInt(meta.reasoning_tokens)} reasoning tokens</span>
          ) : null}
        </div>
      </div>
      {tokensKnown && (
        <p className="hint">Token source: {SOURCE_LABEL[e.usage_source ?? 'provider'] ?? e.usage_source}</p>
      )}

      {kundli && core && <TurnBreakdown experiment={e} />}

      {kundli && core && (
        <div className="kv-block">
          <h3 className="sub-title">Chatbot details</h3>
          {core.intent && <div className="kv"><span>Intent</span><span className="mono">{core.intent}</span></div>}
          <div className="kv"><span>Reply style</span>
            <span className="mono">{core.response_style ?? '—'}{core.requested_style ? '' : ' (auto)'}</span></div>
          {core.fallback_used && <div className="kv"><span>Fallback model</span><span className="mono">used</span></div>}
          <div className="kv"><span>Chart facts used</span><span className="mono">{core.chart_facts_used?.length ?? 0}</span></div>
          <div className="kv"><span>Knowledge citations</span><span className="mono">{core.citations?.length ?? 0}</span></div>
          {core.citations?.map(c => <p key={c.id} className="hint">· {c.title}</p>)}
          {core.notices?.map(n => <p key={n} className="hint warn">{n}</p>)}
        </div>
      )}

      {breakdownTotal > 0 && (
        <div className="breakdown">
          <span className="metric-label">Prompt composition <em>(estimated, ~4 chars/token)</em></span>
          <div className="stack" aria-hidden="true">
            {Object.entries(breakdown).map(([k, v]) =>
              v > 0 ? <div key={k} className={`stack-seg seg-${k}`} style={{ flexGrow: v }} title={`${BREAKDOWN_LABELS[k]} ≈ ${v}`} /> : null,
            )}
          </div>
          <div className="legend">
            {Object.entries(breakdown).map(([k, v]) => (
              <span key={k} className="legend-item">
                <i className={`dot seg-${k}`} />
                {BREAKDOWN_LABELS[k] ?? k} <span className="mono">≈{fmtInt(v)}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="kv-block">
        <h3 className="sub-title">Estimated cost{meta.cost_source ? ` · ${meta.cost_source}` : ''}</h3>
        {!kundli && <div className="kv"><span>Input</span><span className="mono">{fmtCost(e.input_cost, e.currency)}</span></div>}
        {!kundli && <div className="kv"><span>Output</span><span className="mono">{fmtCost(e.output_cost, e.currency)}</span></div>}
        <div className="kv total"><span>Total</span><span className="mono">{fmtCost(e.total_cost, e.currency)}</span></div>
        {meta.provider_reported_cost != null && (
          <div className="kv"><span>Provider-reported</span><span className="mono">{fmtCost(meta.provider_reported_cost)}</span></div>
        )}
      </div>

      <div className="kv-block">
        <h3 className="sub-title">Latency</h3>
        <div className="kv total"><span>Total request</span><span className="mono">{fmtLatency(e.latency_ms)}</span></div>
        <div className="kv"><span>{kundli ? 'Chatbot call' : 'Provider call'}</span><span className="mono">{fmtLatency(e.provider_latency_ms)}</span></div>
        <div className="kv" title={meta.ttft_note ?? undefined}>
          <span>Time to first token</span>
          <span className="mono">{e.first_token_latency_ms != null ? fmtLatency(e.first_token_latency_ms) : 'N/A'}</span>
        </div>
        {e.first_token_latency_ms == null && (
          <p className="hint">{meta.ttft_note ?? 'Not measured: replies are returned whole, not streamed.'}</p>
        )}
        {meta.overhead_ms != null && (
          <div className="kv"><span>Playground overhead</span><span className="mono">{fmtLatency(meta.overhead_ms)}</span></div>
        )}
        {meta.finish_reason && (
          <div className="kv"><span>Finish reason</span><span className="mono">{meta.finish_reason}</span></div>
        )}
      </div>
    </section>
  );
}

/** Separates the main reply's own tokens from whatever auxiliary calls ran in the same turn, and the whole-turn total — never blending a real number with a guess. */
function TurnBreakdown({ experiment: e }: { experiment: Experiment }) {
  const b = computeTurnBreakdown(e);
  return (
    <div className="kv-block">
      <h3 className="sub-title">Auxiliary AI processing</h3>
      <p className="hint" title="ai-core runs extra calls alongside the main reply (an intent classifier, a quality check, and sometimes more) — their combined token usage, not the reply's own.">
        {b.auxiliary.calls === 0
          ? 'No auxiliary calls reported for this turn.'
          : `${b.auxiliary.calls} call${b.auxiliary.calls === 1 ? '' : 's'} alongside the main reply (classifier, quality check, etc.)`}
      </p>
      <div className="kv">
        <span>Combined tokens</span>
        <span className="mono">{b.auxiliary.tokens != null ? `${fmtInt(b.auxiliary.tokens)} (aggregate, not per call)` : 'Not available'}</span>
      </div>

      <h3 className="sub-title">Complete chatbot turn</h3>
      <div className="kv total">
        <span>Whole-turn tokens</span>
        <span className="mono">{b.wholeTurn.tokens != null ? fmtInt(b.wholeTurn.tokens) : 'Not available'}</span>
      </div>
      <div className="kv"><span>Whole-turn cost</span><span className="mono">{fmtCost(b.wholeTurn.costUsd)}</span></div>
      <p className="hint">Main reply + auxiliary calls above, as ai-core reports it — never recomputed by summing the two, so it can't drift from the chatbot's own figure.</p>
    </div>
  );
}
