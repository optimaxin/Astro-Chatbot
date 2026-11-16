import type { Experiment } from '../types';
import { fmtCost, fmtInt, fmtLatency } from '../format';

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
        {meta.provider_name ?? e.provider} · {e.model}
      </p>

      {e.status === 'failed' ? (
        <div className="notice error">
          <strong>Generation failed</strong>
          <span>Provider: {meta.provider_name ?? e.provider}</span>
          <span>Reason: {e.error}</span>
          <span className="muted">Try another model or retry the request. Token data unavailable.</span>
        </div>
      ) : !tokensKnown ? (
        <div className="notice">The provider did not report token usage for this request.</div>
      ) : null}
      {tokensKnown && (
        <p className="hint">
          Token source: {e.usage_source === 'estimated' ? 'estimated (~4 chars/token)' : 'reported by the provider'}
        </p>
      )}
      {meta.ai_core && (
        <div className="kv-block">
          <h3 className="sub-title">Chatbot</h3>
          {meta.ai_core.intent && <div className="kv"><span>Intent</span><span className="mono">{meta.ai_core.intent}</span></div>}
          {meta.ai_core.response_style && <div className="kv"><span>Style</span><span className="mono">{meta.ai_core.response_style}</span></div>}
          {meta.ai_core.chart && <p className="hint">Chart: {meta.ai_core.chart}</p>}
          <div className="kv"><span>Knowledge citations</span><span className="mono">{meta.ai_core.citations?.length ?? 0}</span></div>
          {meta.ai_core.citations?.map(c => <p key={c.id} className="hint">· {c.title}</p>)}
          {meta.ai_core.notices?.map(n => <p key={n} className="hint warn">{n}</p>)}
        </div>
      )}

      <div className="metric-grid">
        <div className="metric">
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
          <span className="metric-label">Total tokens</span>
          <span className="metric-value big mono">{fmtInt(e.total_tokens)}</span>
          {meta.reasoning_tokens ? (
            <span className="hint mono">incl. {fmtInt(meta.reasoning_tokens)} reasoning tokens</span>
          ) : null}
        </div>
      </div>

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
        <div className="kv"><span>Input</span><span className="mono">{fmtCost(e.input_cost, e.currency)}</span></div>
        <div className="kv"><span>Output</span><span className="mono">{fmtCost(e.output_cost, e.currency)}</span></div>
        <div className="kv total"><span>Total</span><span className="mono">{fmtCost(e.total_cost, e.currency)}</span></div>
        {meta.provider_reported_cost != null && (
          <div className="kv"><span>Provider-reported</span><span className="mono">{fmtCost(meta.provider_reported_cost)}</span></div>
        )}
      </div>

      <div className="kv-block">
        <h3 className="sub-title">Latency</h3>
        <div className="kv total"><span>Total request</span><span className="mono">{fmtLatency(e.latency_ms)}</span></div>
        <div className="kv"><span>Provider call</span><span className="mono">{fmtLatency(e.provider_latency_ms)}</span></div>
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
