import { useEffect, useState } from 'react';
import { fmtCost, fmtInt, fmtLatency, fmtScore } from '../format';
import { adminApi, type Analytics, type Costs, type Filters } from './adminApi';
import { DayBars, HBars } from './charts';

const usd = (c: Costs) => c.USD ?? 0;
const costText = (c: Costs) =>
  Object.keys(c).length ? Object.entries(c).map(([cur, v]) => fmtCost(v, cur)).join(' + ') : 'N/A';
const MODE_LABEL: Record<string, string> = { kundli_chatbot: 'Kundli Chatbot', direct_provider: 'Direct provider' };
const METRIC_LABEL: Record<string, string> = {
  relevance: 'Relevance', accuracy: 'Accuracy', human_likeness: 'Human-likeness', tone: 'Tone', conciseness: 'Conciseness',
  context_awareness: 'Context awareness', emoji_usage: 'Emoji usage', language_preservation: 'Language kept',
  chart_consistency: 'Chart consistency',
};

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="stat">
      <span className="metric-label">{label}</span>
      <span className="stat-value mono">{value}</span>
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
}

export default function Dashboard({ onAuthError }: { onAuthError: (e: unknown) => boolean }) {
  const [filters, setFilters] = useState<Filters>({});
  const [options, setOptions] = useState<{ testers: string[] }>({ testers: [] });
  const [data, setData] = useState<Analytics | null>(null);
  const [error, setError] = useState('');
  const set = (k: keyof Filters, v: string) => setFilters(f => ({ ...f, [k]: v }));

  useEffect(() => { adminApi.models().then(m => setOptions(m.seen)).catch(onAuthError); }, [onAuthError]);
  useEffect(() => {
    let live = true;
    setError('');
    adminApi.analytics(filters)
      .then(d => { if (live) setData(d); })
      .catch(e => { if (live && !onAuthError(e)) setError(e.message); });
    return () => { live = false; };
  }, [filters, onAuthError]);

  const t = data?.totals;
  const top = data?.by_model[0];
  const multiCurrency = data ? Object.keys(t!.cost).some(c => c !== 'USD') : false;
  const active = Object.values(filters).some(Boolean);

  return (
    <div className="admin-page">
      <div className="filter-bar wrap">
        <label className="field inline"><span>From</span>
          <input type="date" value={filters.from ?? ''} onChange={e => set('from', e.target.value)} /></label>
        <label className="field inline"><span>To</span>
          <input type="date" value={filters.to ?? ''} onChange={e => set('to', e.target.value)} /></label>
        <label className="field inline"><span>Tester</span>
          <input list="dash-testers" value={filters.tester ?? ''} placeholder="Any" onChange={e => set('tester', e.target.value)} />
          <datalist id="dash-testers">{options.testers.map(n => <option key={n} value={n} />)}</datalist></label>
        <label className="field inline"><span>Testing mode</span>
          <select value={filters.mode ?? ''} onChange={e => set('mode', e.target.value)}>
            <option value="">All</option>
            <option value="kundli_chatbot">Kundli Chatbot</option>
            <option value="direct_provider">Direct provider</option>
          </select></label>
        {active && <button className="btn small" onClick={() => setFilters({})}>Clear</button>}
      </div>
      {error && <div className="notice error">{error}</div>}
      {!data || !t ? (!error && <p className="empty">Loading analytics…</p>) : (
        <>
          <div className="stat-grid">
            <Stat label="Total experiments" value={fmtInt(t.requests)} />
            <Stat label="Successful" value={fmtInt(t.success)}
              hint={t.requests ? `${Math.round((t.success / t.requests) * 100)}% success rate` : undefined} />
            <Stat label="Failed" value={fmtInt(t.failed)} hint={t.pending ? `${t.pending} still running` : undefined} />
            <Stat label="Testers" value={fmtInt(data.by_tester.length)} hint={`${fmtInt(t.unique_sessions)} sessions`} />
            <Stat label="Input tokens" value={fmtInt(t.input_tokens)} />
            <Stat label="Output tokens" value={fmtInt(t.output_tokens)} />
            <Stat label="Total tokens" value={fmtInt(t.total_tokens)} />
            <Stat label="Total estimated cost" value={costText(t.cost)} hint="from stored per-request pricing snapshots" />
            <Stat label="Average latency" value={fmtLatency(t.avg_latency_ms)} />
            <Stat label="Average AI score" value={t.avg_score == null ? '—' : `${fmtScore(t.avg_score)} / 10`}
              hint={`${t.scored} evaluated · ${t.good} good / ${t.bad} bad ratings`} />
            <Stat label="Most-used model" value={top ? top.model : '—'}
              hint={top ? `${fmtInt(top.requests)} requests · ${top.provider}` : undefined} />
          </div>

          <div className="chart-grid">
            <section className="card wide">
              <h2 className="section-title">Tester-wise results</h2>
              <div className="table-wrap">
                <table className="data">
                  <thead><tr><th>Tester</th><th>Sessions</th><th>Experiments</th><th>Failed</th><th>Tokens</th><th>Cost</th>
                    <th>Avg latency</th><th>Avg AI score</th><th>Ratings 👍/👎</th></tr></thead>
                  <tbody>
                    {data.by_tester.map(r => (
                      <tr key={r.tester}>
                        <td>{r.tester}</td><td className="mono">{fmtInt(r.unique_sessions)}</td>
                        <td className="mono">{fmtInt(r.requests)}</td><td className="mono">{fmtInt(r.failed)}</td>
                        <td className="mono">{fmtInt(r.total_tokens)}</td><td className="mono">{costText(r.cost)}</td>
                        <td className="mono">{fmtLatency(r.avg_latency_ms)}</td>
                        <td className="mono">{fmtScore(r.avg_score)}</td>
                        <td className="mono">{r.good} / {r.bad}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {data.by_tester.length === 0 && <p className="empty center-text">No testers match.</p>}
              </div>
            </section>

            <section className="card">
              <h2 className="section-title">Response quality (AI evaluation, average of {t.scored})</h2>
              <HBars items={Object.entries(t.quality).map(([k, v]) => ({ label: METRIC_LABEL[k] ?? k, value: v }))}
                format={n => `${fmtScore(n)} / 10`} empty="No evaluated responses yet." />
              <p className="hint">Automated estimates from a judge model, not ground truth.</p>
            </section>
            <section className="card">
              <h2 className="section-title">Requests by testing mode</h2>
              <HBars items={data.by_mode.map(m => ({ label: MODE_LABEL[m.mode] ?? m.mode, value: m.requests,
                note: `· ${fmtInt(m.total_tokens)} tok` }))} />
            </section>

            <section className="card wide">
              <h2 className="section-title">Daily testing activity</h2>
              <DayBars days={data.daily.map(d => ({ date: d.date, value: d.success, secondary: d.failed }))}
                primaryLabel="Successful" secondaryLabel="Failed" />
            </section>
            <section className="card">
              <h2 className="section-title">Requests by model</h2>
              <HBars items={data.by_model.map(m => ({ label: `${m.provider} · ${m.model}`, value: m.requests }))} />
            </section>
            <section className="card">
              <h2 className="section-title">Usage by provider</h2>
              <HBars items={data.by_provider.map(p => ({ label: p.provider, value: p.requests, note: `· ${fmtInt(p.total_tokens)} tok` }))} />
            </section>
            <section className="card wide">
              <h2 className="section-title">Token consumption per day</h2>
              <DayBars days={data.daily.map(d => ({ date: d.date, value: d.total_tokens }))} primaryLabel="Total tokens" />
            </section>
            <section className="card">
              <h2 className="section-title">Cost by model (USD)</h2>
              <HBars items={data.by_model.filter(m => usd(m.cost) > 0).map(m => ({ label: m.model, value: usd(m.cost) }))}
                format={n => fmtCost(n)} empty="No priced requests yet." />
              {multiCurrency && <p className="hint warn">Some costs are in other currencies and are not shown here.</p>}
            </section>
            <section className="card">
              <h2 className="section-title">Daily API expenditure (USD)</h2>
              <DayBars days={data.daily.map(d => ({ date: d.date, value: usd(d.cost) }))} format={n => fmtCost(n)} primaryLabel="Estimated USD" />
            </section>
            <section className="card wide">
              <h2 className="section-title">Failed requests by reason</h2>
              <HBars items={data.failures.map(f => ({ label: f.error, value: f.count }))} empty="No failures." />
            </section>
            <section className="card wide">
              <h2 className="section-title">Model summary</h2>
              <div className="table-wrap">
                <table className="data">
                  <thead><tr><th>Provider</th><th>Model</th><th>Requests</th><th>Failed</th><th>Tokens</th><th>Cost</th><th>Avg latency</th></tr></thead>
                  <tbody>
                    {data.by_model.map(m => (
                      <tr key={`${m.provider}/${m.model}`}>
                        <td>{m.provider}</td><td className="mono">{m.model}</td><td className="mono">{fmtInt(m.requests)}</td>
                        <td className="mono">{fmtInt(m.failed)}</td><td className="mono">{fmtInt(m.total_tokens)}</td>
                        <td className="mono">{costText(m.cost)}</td><td className="mono">{fmtLatency(m.avg_latency_ms)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
