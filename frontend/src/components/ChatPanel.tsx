import { useEffect, useRef, useState } from 'react';
import type { Experiment, Mode } from '../types';

export interface Turn {
  key: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
  experimentId?: number;
  /** Groups the bubbles of one multi-bubble reply together (same experiment, several Turn entries). */
  turnId?: string;
}

interface Props {
  turns: Turn[];
  loading: boolean;
  /** A reply's own bubbles are still being revealed one at a time (separate from `loading`, which ends once the server responds). */
  pacing: boolean;
  selectedId: number | null;
  selected: Experiment | null;
  mode: Mode;
  onSelect: (experimentId: number) => void;
  onSend: (message: string) => void;
  onNewConversation: () => void;
}

export default function ChatPanel({ turns, loading, pacing, selectedId, selected, mode, onSelect, onSend, onNewConversation }: Props) {
  const [draft, setDraft] = useState('');
  const [showRequest, setShowRequest] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
  }, [turns, loading, pacing]);

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
          <button className="btn ghost" onClick={onNewConversation} disabled={loading} title="Ctrl+Shift+N">
            New conversation
          </button>
        </div>
      </div>

      {showRequest && selected?.effective_request && (
        <div className="inspector">
          <div className="inspector-head mono">
            #{selected.id} ·{' '}
            {selected.effective_request.mode === 'kundli_chatbot'
              ? `${selected.effective_request.endpoint ?? 'POST /chat'} (Kundli chatbot) · answered by ${selected.provider} / ${selected.model}`
              : `${selected.provider} · ${selected.effective_request.model}`}
          </div>
          {(selected.effective_request.params_ignored?.length ?? 0) > 0 && (
            <p className="hint warn">
              Not sent (unsupported by provider): {selected.effective_request.params_ignored!.join(', ')}
            </p>
          )}
          <pre className="mono">
            {JSON.stringify(
              selected.effective_request.mode === 'kundli_chatbot'
                ? {
                  response_style: selected.effective_request.response_style,
                  birth: selected.effective_request.birth,
                  messages: selected.effective_request.messages,
                  note: 'The chatbot adds its own system instructions, chart data, knowledge and memory.',
                }
                : { params: selected.effective_request.params_sent, messages: selected.effective_request.messages },
              null,
              2,
            )}
          </pre>
        </div>
      )}

      <div className="messages" ref={listRef}>
        {turns.length === 0 && (
          <div className="empty-chat">
            <p className="empty-title">{mode === 'kundli_chatbot' ? 'Test the Kundli chatbot' : 'Start an experiment'}</p>
            <p className="empty">
              {mode === 'kundli_chatbot'
                ? 'Check the birth details on the left, then ask a question as a website visitor would.'
                : 'Pick a model, adjust the prompt on the left, and send a message.'}
            </p>
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
        {(loading || pacing) && (
          <div className="msg msg-assistant">
            <div className="msg-meta">Assistant</div>
            <div className="bubble typing" aria-label={loading ? 'Generating' : 'Typing'}><span /><span /><span /></div>
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
