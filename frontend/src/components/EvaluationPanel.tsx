import type { Experiment } from '../types';
import { fmtCost, fmtInt, fmtScore } from '../format';

interface Props {
  experiment: Experiment | null;
  evaluating: boolean;
  error: string;
  onEvaluate: () => void;
}

const LABELS: Record<string, string> = {
  relevance: 'Relevance',
  accuracy: 'Accuracy',
  human_likeness: 'Human-like',
  tone: 'Tone',
  conciseness: 'Conciseness',
  context_awareness: 'Context awareness',
  emoji_usage: 'Emoji usage',
  language_preservation: 'Language kept',
  chart_consistency: 'Chart consistency',
};

export default function EvaluationPanel({ experiment, evaluating, error, onEvaluate }: Props) {
  if (!experiment || experiment.status !== 'success') return null;
  const ev = experiment.evaluation;

  return (
    <section className="section">
      <div className="section-head">
        <h2 className="section-title">AI Evaluation</h2>
        <button className="btn ghost small" onClick={onEvaluate} disabled={evaluating}>
          {evaluating ? 'Evaluating…' : ev ? 'Re-evaluate' : 'Evaluate'}
        </button>
      </div>
      <p className="hint">Automated estimate by a judge model — not ground truth. Chart consistency is checked against the chart facts the chatbot used.</p>

      {error && <div className="notice error">{error}</div>}
      {!ev && !evaluating && !error && <p className="empty">Not evaluated yet.</p>}
      {evaluating && !ev && <p className="empty">Asking the judge model…</p>}

      {ev && (
        <div className={evaluating ? 'is-stale' : ''}>
          <div className="scores">
            {Object.entries(ev.scores).map(([k, v]) => (
              <div key={k} className="score-row">
                <span>{LABELS[k] ?? k}</span>
                <div className="bar small" aria-hidden="true">
                  <div className="bar-fill tone-score" style={{ width: `${v == null ? 0 : v * 10}%` }} />
                </div>
                <span className="mono score-num">{v == null ? 'n/a' : `${fmtScore(v)}`}</span>
              </div>
            ))}
            <div className="score-row overall">
              <span>Overall</span>
              <span />
              <span className="mono score-num">{fmtScore(ev.overall)} / 10</span>
            </div>
          </div>

          {ev.feedback && <p className="feedback-text">{ev.feedback}</p>}
          {ev.strengths.length > 0 && (
            <div className="list-block">
              <h3 className="sub-title">Strengths</h3>
              <ul>{ev.strengths.map((s, i) => <li key={i}>{s}</li>)}</ul>
            </div>
          )}
          {ev.improvements.length > 0 && (
            <div className="list-block">
              <h3 className="sub-title">Improvements</h3>
              <ul>{ev.improvements.map((s, i) => <li key={i}>{s}</li>)}</ul>
            </div>
          )}
          <p className="hint mono">
            Judge: {ev.evaluator.model} · {fmtInt(ev.evaluator_usage.total_tokens)} tokens ·{' '}
            {fmtCost(ev.evaluator_usage.total_cost, ev.evaluator_usage.currency)} (not included above)
          </p>
        </div>
      )}
    </section>
  );
}
