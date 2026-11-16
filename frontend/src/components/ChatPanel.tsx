import { useEffect, useRef, useState } from 'react';
import type { Experiment } from '../types';

export interface Turn {
  key: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
  experimentId?: number;
}

interface Props {
  turns: Turn[];
  loading: boolean;
  selectedId: number | null;
  selected: Experiment | null;
  onSelect: (experimentId: number) => void;
  onSend: (message: string) => void;
  onNewSession: () => void;
}

export default function ChatPanel({ turns, loading, selectedId, selected, onSelect, onSend, onNewSession }: Props) {
  const [draft, setDraft] = useState('');
  const [showRequest, setShowRequest] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [turns, loading]);

  const send = () => {
    const text = draft.trim();
    if (!text || loading) return;
    onSend(text);
    setDraft('');
    inputRef.current?.focus();
  };

  return (
    <main className="panel chat" aria-label="Chat test area">
      <div className="chat-toolbar">
        <span className="section-title">Conversation</span>
        <div className="toolbar-actions">
          <button className="btn ghost" onClick={() => setShowRequest(s => !s)} disabled={!selected}>
            {showRequest ? 'Hide request' : 'Inspect request'}
          </button>
          <button className="btn ghost" onClick={onNewSession} disabled={loading} title="Ctrl+Shift+N">
            New session
          </button>
        </div>
      </div>

      {showRequest && selected?.effective_request && (
        <div className="inspector">
          <div className="inspector-head mono">
            #{selected.id} · POST {selected.effective_request.base_url} · {selected.effective_request.model}
          </div>
          {selected.effective_request.params_ignored.length > 0 && (
            <p className="hint warn">
              Not sent (unsupported by provider): {selected.effective_request.params_ignored.join(', ')}
            </p>
          )}
          <pre className="mono">
            {JSON.stringify(
              { params: selected.effective_request.params_sent, messages: selected.effective_request.messages },
              null,
              2,
            )}
          </pre>
        </div>
      )}

      <div className="messages" ref={listRef}>
        {turns.length === 0 && (
          <div className="empty-chat">
            <p className="empty-title">Start an experiment</p>
            <p className="empty">Pick a model, adjust the prompt on the left, and send a message.</p>
          </div>
        )}
        {turns.map(t => {
          const clickable = t.experimentId != null && t.role !== 'user';
          return (
            <div key={t.key} className={`msg msg-${t.role}`}>
              <div className="msg-meta">
                {t.role === 'user' ? 'User' : t.role === 'assistant' ? 'Assistant' : 'Error'}
                {t.experimentId != null && t.role !== 'user' && <span className="mono"> · #{t.experimentId}</span>}
              </div>
              <div
                className={`bubble ${clickable ? 'clickable' : ''} ${selectedId === t.experimentId && clickable ? 'selected' : ''}`}
                onClick={clickable ? () => onSelect(t.experimentId!) : undefined}
                role={clickable ? 'button' : undefined}
                tabIndex={clickable ? 0 : undefined}
                onKeyDown={clickable ? e => { if (e.key === 'Enter') onSelect(t.experimentId!); } : undefined}
              >
                {t.content}
              </div>
            </div>
          );
        })}
        {loading && (
          <div className="msg msg-assistant">
            <div className="msg-meta">Assistant</div>
            <div className="bubble typing" aria-label="Generating"><span /><span /><span /></div>
          </div>
        )}
      </div>

      <div className="composer">
        <textarea
          ref={inputRef}
          value={draft}
          rows={2}
          placeholder="Type your message…  (Enter to send, Shift+Enter for a new line)"
          onChange={e => setDraft(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          disabled={loading}
        />
        <button className="btn primary send" onClick={send} disabled={loading || !draft.trim()} aria-label="Send">
          {loading ? '…' : '➤'}
        </button>
      </div>
    </main>
  );
}
