import type { ExperimentSummary } from '../types';
import { fmtCost, fmtInt, fmtLatency, fmtScore, fmtTime } from '../format';

interface Props {
  open: boolean;
  items: ExperimentSummary[];
  loading: boolean;
  onClose: () => void;
  onOpen: (id: number) => void;
}

export default function HistoryDrawer({ open, items, loading, onClose, onOpen }: Props) {
  if (!open) return null;
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={e => e.stopPropagation()} aria-label="Experiment history">
        <div className="section-head">
          <h2 className="section-title">Experiment history</h2>
          <button className="btn ghost small" onClick={onClose} aria-label="Close history">✕</button>
        </div>
        {loading && <p className="empty">Loading…</p>}
        {!loading && items.length === 0 && <p className="empty">No experiments yet.</p>}
        <div className="history-table" role="table">
          <div className="history-row head" role="row">
            <span>#</span><span>Model</span><span>Tokens</span><span>Cost</span><span>Latency</span><span>Eval</span><span>👍/👎</span>
          </div>
          {items.map(x => (
            <button key={x.id} className={`history-row ${x.status}`} role="row" onClick={() => onOpen(x.id)}>
              <span className="mono">#{x.id}</span>
              <span className="history-model">
                <span className="truncate">{x.mode === 'kundli_chatbot' ? 'Kundli chatbot · ' : ''}{x.model}</span>
                <span className="hint truncate">{fmtTime(x.created_at)} · {x.user_prompt}</span>
              </span>
              <span className="mono">{x.status === 'failed' ? 'failed' : fmtInt(x.total_tokens)}</span>
              <span className="mono">{fmtCost(x.total_cost, x.currency)}</span>
              <span className="mono">{fmtLatency(x.latency_ms)}</span>
              <span className="mono">{fmtScore(x.evaluation_overall)}</span>
              <span>{x.human_rating === 'good' ? '👍' : x.human_rating === 'bad' ? '👎' : ''}</span>
            </button>
          ))}
        </div>
      </aside>
    </div>
  );
}
