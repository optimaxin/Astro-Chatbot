import { useEffect, useState } from 'react';
import { fmtCost, fmtInt, fmtLatency, fmtScore, fmtTime } from '../format';
import { adminApi, type Filters, type Page, type Row } from './adminApi';

const PAGE_SIZE = 25;

function Detail({ id, onClose, onAuthError }: { id: number; onClose: () => void; onAuthError: (e: unknown) => boolean }) {
  const [exp, setExp] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    adminApi.experiment(id).then(setExp).catch(e => { if (!onAuthError(e)) setError(e.message); });
  }, [id, onAuthError]);

  const block = (title: string, text: unknown) =>
    text ? (<><h3 className="sub-title">{title}</h3><pre className="pre">{String(text)}</pre></>) : null;

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <div className="drawer" role="dialog" aria-label={`Experiment ${id}`} onClick={e => e.stopPropagation()}>
        <div className="section-head">
          <h2 className="section-title">Experiment #{id}</h2>
          <button className="btn small" onClick={onClose}>Close</button>
        </div>
        {error && <div className="notice error">{error}</div>}
        {!exp ? (!error && <p className="empty">Loading…</p>) : (
          <>
            <div className="kv-grid">
              {([
                ['Time', fmtTime(exp.created_at)], ['Status', exp.status],
                ['Testing mode', exp.mode === 'kundli_chatbot' ? 'Kundli Chatbot' : 'Direct provider'], ['Provider', exp.provider],
                ['Model', exp.model], ['Model reported', exp.metadata?.model_reported ?? '—'], ['Tester', exp.tester_name ?? '—'],
                ['Session', exp.session_id ?? '—'], ['Request ID', exp.request_id ?? '—'],
                ['Input tokens', fmtInt(exp.input_tokens)], ['Output tokens', fmtInt(exp.output_tokens)],
                ['Total tokens', fmtInt(exp.total_tokens)], ['Token source', exp.usage_source ?? '—'],
                ['Total cost', fmtCost(exp.total_cost, exp.currency)], ['Latency', fmtLatency(exp.latency_ms)],
                ['AI score', fmtScore(exp.evaluation_overall)], ['Human rating', exp.human_rating ?? '—'],
              ] as [string, string][]).map(([k, v]) => (
                <div key={k} className="kv"><span>{k}</span><span className="mono">{v}</span></div>
              ))}
            </div>
            {exp.error && <div className="notice error">{exp.error}</div>}
            {block('System prompt', exp.system_prompt)}
            {block('Context', exp.context)}
            {block('User prompt', exp.user_prompt)}
            {block('Response', exp.response)}
            {block('Edited response (tester)', exp.edited_response)}
            {block('Tester feedback', exp.human_feedback)}
            {exp.evaluation && block('AI evaluation', JSON.stringify(exp.evaluation, null, 2))}
            {block('Pricing snapshot', exp.pricing_snapshot && JSON.stringify(exp.pricing_snapshot, null, 2))}
            {block('Exact request sent', JSON.stringify(exp.effective_request, null, 2))}
            {block('Metadata', JSON.stringify(exp.metadata, null, 2))}
          </>
        )}
      </div>
    </div>
  );
}

export default function Experiments({ onAuthError }: { onAuthError: (e: unknown) => boolean }) {
  const [filters, setFilters] = useState<Filters>({});
  const [draft, setDraft] = useState<Filters>({});
  const [page, setPage] = useState(1);
  const [data, setData] = useState<Page<Row> | null>(null);
  const [options, setOptions] = useState<{ providers: string[]; models: string[]; testers: string[] }>({ providers: [], models: [], testers: [] });
  const [openId, setOpenId] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState('');

  useEffect(() => { adminApi.models().then(m => setOptions(m.seen)).catch(onAuthError); }, [onAuthError]);
  useEffect(() => {
    let live = true;
    adminApi.experiments(filters, page, PAGE_SIZE)
      .then(d => { if (live) { setData(d); setError(''); } })
      .catch(e => { if (live && !onAuthError(e)) setError(e.message); });
    return () => { live = false; };
  }, [filters, page, onAuthError]);

  const apply = () => { setFilters(draft); setPage(1); };
  const set = (k: keyof Filters, v: string) => setDraft(d => ({ ...d, [k]: v }));
  const exportExcel = async (f: Filters, label: string) => {
    setBusy(label);
    try { await adminApi.exportExcel(f); } catch (e) { if (!onAuthError(e)) setError((e as Error).message); }
    finally { setBusy(''); }
  };
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;
  const hasDate = Boolean(filters.from || filters.to);

  return (
    <div className="admin-page">
      <div className="filter-bar wrap">
        <label className="field inline"><span>From</span><input type="date" value={draft.from ?? ''} onChange={e => set('from', e.target.value)} /></label>
        <label className="field inline"><span>To</span><input type="date" value={draft.to ?? ''} onChange={e => set('to', e.target.value)} /></label>
        <label className="field inline"><span>Tester</span>
          <input list="exp-testers" value={draft.tester ?? ''} placeholder="Any" onChange={e => set('tester', e.target.value)}
            onKeyDown={e => e.key === 'Enter' && apply()} />
          <datalist id="exp-testers">{options.testers.map(n => <option key={n} value={n} />)}</datalist>
        </label>
        <label className="field inline"><span>Testing mode</span>
          <select value={draft.mode ?? ''} onChange={e => set('mode', e.target.value)}>
            <option value="">All</option><option value="kundli_chatbot">Kundli Chatbot</option><option value="direct_provider">Direct provider</option>
          </select>
        </label>
        <label className="field inline"><span>Provider</span>
          <select value={draft.provider ?? ''} onChange={e => set('provider', e.target.value)}>
            <option value="">All</option>{options.providers.map(p => <option key={p}>{p}</option>)}
          </select>
        </label>
        <label className="field inline"><span>Model</span>
          <select value={draft.model ?? ''} onChange={e => set('model', e.target.value)}>
            <option value="">All</option>{options.models.map(m => <option key={m}>{m}</option>)}
          </select>
        </label>
        <label className="field inline"><span>Status</span>
          <select value={draft.status ?? ''} onChange={e => set('status', e.target.value)}>
            <option value="">All</option><option value="success">Success</option><option value="failed">Failed</option>
          </select>
        </label>
        <label className="field inline"><span>Session ID</span><input value={draft.session_id ?? ''} onChange={e => set('session_id', e.target.value)} /></label>
        <label className="field inline grow"><span>Search prompt / response / tester</span>
          <input value={draft.q ?? ''} onChange={e => set('q', e.target.value)} onKeyDown={e => e.key === 'Enter' && apply()} />
        </label>
        <button className="btn primary small" onClick={apply}>Apply</button>
        <button className="btn small" onClick={() => { setDraft({}); setFilters({}); setPage(1); }}>Reset</button>
      </div>

      <div className="export-bar">
        <span className="metric-label">Excel reports</span>
        <button className="btn small" disabled={!!busy} onClick={() => exportExcel({}, 'all')}>
          {busy === 'all' ? 'Preparing…' : 'Complete'}</button>
        <button className="btn small" disabled={!!busy || !hasDate}
          title={hasDate ? '' : 'Set From/To and Apply first'}
          onClick={() => exportExcel({ from: filters.from, to: filters.to }, 'date')}>
          {busy === 'date' ? 'Preparing…' : 'Date-filtered'}</button>
        <button className="btn small" disabled={!!busy || !filters.model} title={filters.model ? '' : 'Choose a model and Apply first'}
          onClick={() => exportExcel({ model: filters.model, provider: filters.provider }, 'model')}>
          {busy === 'model' ? 'Preparing…' : 'Model-wise'}</button>
        <button className="btn small" disabled={!!busy || !filters.tester} title={filters.tester ? '' : 'Enter a Tester and Apply first'}
          onClick={() => exportExcel({ tester: filters.tester }, 'tester')}>
          {busy === 'tester' ? 'Preparing…' : 'Tester-wise'}</button>
        <button className="btn small" disabled={!!busy} onClick={() => exportExcel({ status: 'failed' }, 'failed')}>
          {busy === 'failed' ? 'Preparing…' : 'Failed requests'}</button>
        <button className="btn small" disabled={!!busy || !filters.session_id} title={filters.session_id ? '' : 'Enter a Session ID and Apply first'}
          onClick={() => exportExcel({ session_id: filters.session_id }, 'session')}>
          {busy === 'session' ? 'Preparing…' : 'Session-wise'}</button>
        <button className="btn small primary" disabled={!!busy} onClick={() => exportExcel(filters, 'current')}>
          {busy === 'current' ? 'Preparing…' : 'Current filters'}</button>
      </div>

      {error && <div className="notice error">{error}</div>}
      <div className="table-wrap">
        <table className="data clickable">
          <thead>
            <tr>
              <th>ID</th><th>Time</th><th>Tester</th><th>Session</th><th>Mode</th><th>Provider</th><th>Model</th><th>Prompt</th><th>Response</th>
              <th>In</th><th>Out</th><th>Total</th><th>Cost</th><th>Latency</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map(r => (
              <tr key={r.id} onClick={() => setOpenId(r.id)} tabIndex={0} onKeyDown={e => e.key === 'Enter' && setOpenId(r.id)}>
                <td className="mono">{r.id}</td>
                <td className="nowrap">{fmtTime(r.created_at)}</td>
                <td>{r.tester_name}</td>
                <td className="mono" title={r.session_id}>{r.session_id.slice(0, 8)}</td>
                <td>{r.mode === 'kundli_chatbot' ? 'Kundli bot' : 'Direct'}</td>
                <td>{r.provider}</td>
                <td className="mono">{r.model}</td>
                <td className="preview">{r.prompt_preview}</td>
                <td className="preview">{r.response_preview ?? <span className="muted">{r.error}</span>}</td>
                <td className="mono">{fmtInt(r.input_tokens)}</td>
                <td className="mono">{fmtInt(r.output_tokens)}</td>
                <td className="mono">{fmtInt(r.total_tokens)}</td>
                <td className="mono">{fmtCost(r.total_cost, r.currency)}</td>
                <td className="mono">{fmtLatency(r.latency_ms)}</td>
                <td><span className={`status ${r.status}`}>{r.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
        {data && data.items.length === 0 && <p className="empty center-text">No experiments match.</p>}
      </div>
      <div className="pager">
        <span className="muted">{data ? `${fmtInt(data.total)} experiments` : ''}</span>
        <button className="btn small" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>Previous</button>
        <span className="mono">{page} / {pages}</span>
        <button className="btn small" disabled={page >= pages} onClick={() => setPage(p => p + 1)}>Next</button>
      </div>
      {openId != null && <Detail id={openId} onClose={() => setOpenId(null)} onAuthError={onAuthError} />}
    </div>
  );
}
