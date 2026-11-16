import { useEffect, useState } from 'react';
import type { Experiment } from '../types';

interface Props {
  experiment: Experiment | null;
  onSave: (rating: 'good' | 'bad' | null, feedback: string, edited: string) => Promise<void>;
}

export default function FeedbackPanel({ experiment, onSave }: Props) {
  const [rating, setRating] = useState<'good' | 'bad' | null>(null);
  const [text, setText] = useState('');
  const [editing, setEditing] = useState(false);
  const [edited, setEdited] = useState('');
  const [state, setState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  useEffect(() => {
    setRating(experiment?.human_rating ?? null);
    setText(experiment?.human_feedback ?? '');
    setEdited(experiment?.edited_response ?? '');
    setEditing(Boolean(experiment?.edited_response));
    setState('idle');
  }, [experiment?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!experiment || experiment.status !== 'success') return null;

  const save = async (nextRating = rating) => {
    setState('saving');
    try {
      await onSave(nextRating, text, editing ? edited : '');
      setState('saved');
    } catch {
      setState('error');
    }
  };

  const vote = (value: 'good' | 'bad') => {
    const next = rating === value ? null : value;
    setRating(next);
    void save(next);
  };

  return (
    <section className="section">
      <div className="section-head">
        <h2 className="section-title">Human feedback</h2>
        <span className="hint">
          {state === 'saving' ? 'Saving…' : state === 'saved' ? 'Saved' : state === 'error' ? 'Save failed' : ''}
        </span>
      </div>
      <div className="vote-row">
        <button className={`btn vote ${rating === 'good' ? 'active-good' : ''}`} onClick={() => vote('good')}>
          👍 Good
        </button>
        <button className={`btn vote ${rating === 'bad' ? 'active-bad' : ''}`} onClick={() => vote('bad')}>
          👎 Bad
        </button>
        <button
          className={`btn vote ${editing ? 'active-edit' : ''}`}
          onClick={() => {
            setEditing(e => !e);
            if (!edited) setEdited(experiment.response ?? '');
          }}
        >
          ✏️ Edit
        </button>
      </div>
      {editing && (
        <label className="field">
          <span>Preferred response</span>
          <textarea rows={5} value={edited} onChange={e => setEdited(e.target.value)} />
        </label>
      )}
      <label className="field">
        <span>What should be improved?</span>
        <textarea rows={3} value={text} onChange={e => setText(e.target.value)} placeholder="Optional" />
      </label>
      <button className="btn primary small" onClick={() => void save()} disabled={state === 'saving'}>
        Save feedback
      </button>
    </section>
  );
}
