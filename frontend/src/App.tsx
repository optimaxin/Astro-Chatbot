import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError, randomId, session } from './api';
import type { Catalog, ChatTurn, Experiment, ExperimentSummary, ParamKey, PlaygroundConfig, Tester } from './types';
import ConfigPanel from './components/ConfigPanel';
import ChatPanel, { type Turn } from './components/ChatPanel';
import TokenMeter from './components/TokenMeter';
import EvaluationPanel from './components/EvaluationPanel';
import FeedbackPanel from './components/FeedbackPanel';
import HistoryDrawer from './components/HistoryDrawer';
import VerificationPanel from './components/VerificationPanel';
import Welcome from './components/Welcome';
import { deliverBubbles, prefersReducedMotion, type DeliveryHandle } from './lib/bubbleDelivery';

const PARAM_KEYS: ParamKey[] = ['temperature', 'top_p', 'top_k', 'max_tokens'];
let turnSeq = 0;
const turnKey = () => `t${++turnSeq}`;

function defaultConfig(catalog: Catalog): PlaygroundConfig {
  const chatbotUp = catalog.modes.find(m => m.id === 'kundli_chatbot')?.available !== false;
  const provider = catalog.providers.find(p => p.configured) ?? catalog.providers[0];
  const model = catalog.models.find(m => m.provider === provider?.id);
  return {
    // The Kundli chatbot is what this playground exists to test, so it is the default.
    mode: chatbotUp ? 'kundli_chatbot' : 'direct_provider',
    style: 'auto',
    birth: catalog.chatbot.sample_birth,
    provider: provider?.id ?? '',
    model: model?.id ?? '',
    systemPrompt: 'You are a helpful, friendly conversational assistant. Keep replies short and natural.',
    context: '',
    params: { temperature: 0.7, top_p: 0.95, top_k: 40, max_tokens: 1000 },
    enabled: { temperature: true, top_p: false, top_k: false, max_tokens: true },
    autoEvaluate: true,
  };
}

// A multi-bubble reply is several consecutive Turn entries sharing one turnId;
// rejoin them into one logical message before building conversation history.
function collapseBubbles(turns: Turn[]): Turn[] {
  const out: Turn[] = [];
  for (const t of turns) {
    const last = out[out.length - 1];
    if (t.role === 'assistant' && last?.role === 'assistant' && last.turnId && t.turnId === last.turnId) {
      out[out.length - 1] = { ...last, content: `${last.content}\n\n${t.content}` };
    } else {
      out.push(t);
    }
  }
  return out;
}

// Only answered exchanges are sent back as history; a failed turn never reached the model.
function historyFrom(turns: Turn[]): ChatTurn[] {
  const collapsed = collapseBubbles(turns);
  const out: ChatTurn[] = [];
  collapsed.forEach((t, i) => {
    const next = collapsed[i + 1];
    if (t.role === 'user' && next?.role === 'assistant') {
      out.push({ role: 'user', content: t.content }, { role: 'assistant', content: next.content });
    }
  });
  return out;
}

/** The chatbot's own pre-split bubbles (ai-core's `messages`), falling back to one bubble. Direct-model replies are always one bubble. */
function bubbleTextsFor(exp: Experiment): string[] {
  if (exp.mode === 'kundli_chatbot') {
    const msgs = exp.metadata?.ai_core?.messages;
    if (msgs && msgs.length) return msgs;
  }
  return [exp.response ?? ''];
}

function readTheme(): 'light' | 'dark' {
  try {
    const saved = localStorage.getItem('playground.theme');
    if (saved === 'light' || saved === 'dark') return saved;
  } catch { /* storage unavailable */ }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const [tester, setTester] = useState<Tester | null | undefined>(undefined); // undefined = still checking
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [config, setConfig] = useState<PlaygroundConfig | null>(null);
  const [fatal, setFatal] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [conversationId, setConversationId] = useState(() => randomId('c'));
  const [experiments, setExperiments] = useState<Record<number, Experiment>>({});
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [evaluatingId, setEvaluatingId] = useState<number | null>(null);
  const [evalError, setEvalError] = useState('');
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState<ExperimentSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [theme, setTheme] = useState(readTheme);
  const [pacing, setPacing] = useState(false);
  const deliveryRef = useRef<DeliveryHandle | null>(null);

  useEffect(() => () => deliveryRef.current?.cancel(), []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('playground.theme', theme); } catch { /* ignore */ }
  }, [theme]);

  const reset = useCallback(() => {
    setCatalog(null);
    setConfig(null);
    setTurns([]);
    setExperiments({});
    setSelectedId(null);
    setHistory([]);
    setFatal('');
    setEvalError('');
    setConversationId(randomId('c'));
  }, []);

  // A 401 anywhere means the server-side session is gone: ask for the name again.
  const handleAuthError = useCallback((err: unknown) => {
    if (err instanceof ApiError && err.status === 401) {
      reset();
      setTester(null);
      return true;
    }
    return false;
  }, [reset]);

  // On load (and on every refresh) the server tells us whether this browser session already has a tester.
  useEffect(() => {
    session.current().then(setTester).catch(() => setTester(null));
  }, []);

  useEffect(() => {
    if (!tester) return;
    api.models()
      .then(c => { setCatalog(c); setConfig(prev => prev ?? defaultConfig(c)); })
      .catch(err => { if (!handleAuthError(err)) setFatal(err.message); });
  }, [tester, handleAuthError]);

  const remember = (e: Experiment) => setExperiments(prev => ({ ...prev, [e.id]: e }));
  const selected = selectedId != null ? experiments[selectedId] ?? null : null;

  const runEvaluation = useCallback(async (id: number) => {
    setEvaluatingId(id);
    setEvalError('');
    try {
      remember(await api.evaluate(id));
    } catch (err) {
      if (!handleAuthError(err)) setEvalError(err instanceof Error ? err.message : 'Evaluation failed.');
    } finally {
      setEvaluatingId(null);
    }
  }, [handleAuthError]);

  const send = async (message: string) => {
    if (!config) return;
    const kundli = config.mode === 'kundli_chatbot';
    const params = Object.fromEntries(
      PARAM_KEYS.filter(k => config.enabled[k]).map(k => [k, config.params[k]]),
    ) as Partial<Record<ParamKey, number>>;
    const history = historyFrom(turns);
    setTurns(prev => [...prev, { key: turnKey(), role: 'user', content: message }]);
    setLoading(true);
    setEvalError('');
    // A new message while the previous reply is still revealing its bubbles: show the rest of it at once.
    deliveryRef.current?.flush();
    try {
      const base = { message, request_id: randomId('r'), conversation_id: conversationId };
      const exp = await api.chat(kundli
        ? { ...base, mode: 'kundli_chatbot', style: config.style === 'auto' ? null : config.style, birth: config.birth }
        : {
          ...base, mode: 'direct_provider', provider: config.provider, model: config.model,
          system_prompt: config.systemPrompt || undefined, context: config.context || undefined, history, params,
        });
      remember(exp);
      setSelectedId(exp.id);
      if (exp.status === 'success') {
        const turnId = randomId('tg');
        // `let`, not `const`: a one-bubble reply finishes synchronously, before deliverBubbles has returned.
        let handle: DeliveryHandle | null = null;
        handle = deliverBubbles(bubbleTextsFor(exp), {
          onShow: text => setTurns(prev => [...prev, { key: turnKey(), role: 'assistant', content: text, experimentId: exp.id, turnId }]),
          onTyping: setPacing,
          onDone: () => { if (deliveryRef.current === handle) deliveryRef.current = null; },
        }, { reducedMotion: prefersReducedMotion() });
        if (!handle.finished) deliveryRef.current = handle;
      } else {
        setTurns(prev => [...prev, { key: turnKey(), role: 'error', content: `Generation failed — ${exp.error}`, experimentId: exp.id }]);
      }
      if (exp.status === 'success' && config.autoEvaluate) void runEvaluation(exp.id);
    } catch (err) {
      if (!handleAuthError(err)) {
        const text = err instanceof Error ? err.message : 'Request failed.';
        setTurns(prev => [...prev, { key: turnKey(), role: 'error', content: text }]);
      }
    } finally {
      setLoading(false);
    }
  };

  const newConversation = useCallback(() => {
    deliveryRef.current?.cancel();
    deliveryRef.current = null;
    setPacing(false);
    setTurns([]);
    setSelectedId(null);
    setConversationId(randomId('c'));
    setEvalError('');
  }, []);

  const openHistory = useCallback(async () => {
    setHistoryOpen(true);
    setHistoryLoading(true);
    try {
      setHistory(await api.experiments());
    } catch (err) {
      handleAuthError(err);
    } finally {
      setHistoryLoading(false);
    }
  }, [handleAuthError]);

  const reopen = async (id: number) => {
    try {
      deliveryRef.current?.cancel();
      deliveryRef.current = null;
      setPacing(false);
      const exp = await api.experiment(id);
      remember(exp);
      const prior = exp.mode === 'direct_provider'
        ? (exp.effective_request?.messages ?? []).slice(0, -1).filter(m => m.role !== 'system')
        : [];
      const turnId = randomId('tg');
      setTurns([
        ...prior.map(m => ({ key: turnKey(), role: m.role as 'user' | 'assistant', content: m.content })),
        { key: turnKey(), role: 'user', content: exp.user_prompt },
        // Reopening shows every bubble of a past reply at once — the staggered
        // reveal only ever plays for a reply as it first arrives (send(), above).
        ...(exp.status === 'success'
          ? bubbleTextsFor(exp).map(text => ({ key: turnKey(), role: 'assistant' as const, content: text, experimentId: exp.id, turnId }))
          : [{ key: turnKey(), role: 'error' as const, content: `Generation failed — ${exp.error}`, experimentId: exp.id }]),
      ]);
      setSelectedId(exp.id);
      setConfig(prev => prev && (exp.mode === 'kundli_chatbot'
        ? {
          ...prev, mode: 'kundli_chatbot',
          style: exp.params?.response_style ?? 'auto',
          birth: exp.effective_request?.birth ?? prev.birth,
        }
        : {
          ...prev, mode: 'direct_provider', provider: exp.provider, model: exp.model,
          systemPrompt: exp.system_prompt ?? '', context: exp.context ?? '',
          params: { ...prev.params, ...(exp.params ?? {}) } as Record<ParamKey, number>,
          enabled: Object.fromEntries(PARAM_KEYS.map(k => [k, exp.params?.[k] != null])) as Record<ParamKey, boolean>,
        }));
      setHistoryOpen(false);
    } catch (err) {
      handleAuthError(err);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setHistoryOpen(false);
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'n') { e.preventDefault(); newConversation(); }
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'h') { e.preventDefault(); void openHistory(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [newConversation, openHistory]);

  const saveFeedback = async (rating: 'good' | 'bad' | null, feedback: string, edited: string) => {
    if (!selected) return;
    remember(await api.feedback(selected.id, rating, feedback, edited));
  };

  const changeTester = async () => {
    try { await session.end(); } catch { /* the cookie is dropped either way */ }
    reset();
    setTester(null);
  };

  if (tester === undefined) return <p className="empty center">Loading…</p>;
  if (tester === null) {
    return <Welcome onStart={async name => { setTester(await session.start(name)); }} />;
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">◆</span>
          <h1>AI Model Playground</h1>
          <span className="chip">testing</span>
        </div>
        <div className="toolbar-actions">
          <span className="tester-badge" title={`Session ${tester.session_id}`}>Testing as <strong>{tester.tester_name}</strong></span>
          <button className="btn ghost" onClick={() => void changeTester()} title="Start again as a different tester">Change</button>
          <button className="btn ghost" onClick={() => void openHistory()} title="Ctrl+Shift+H">History</button>
          <button className="btn ghost" onClick={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))}
            aria-label="Toggle theme">{theme === 'dark' ? '☀' : '☾'}</button>
        </div>
      </header>

      {fatal && <div className="notice error banner">{fatal}</div>}
      {!catalog || !config ? (
        !fatal && <p className="empty center">Loading models…</p>
      ) : (
        <div className="layout">
          <ConfigPanel catalog={catalog} config={config} onChange={setConfig} disabled={loading} onAuthError={handleAuthError} />
          <ChatPanel
            turns={turns}
            loading={loading}
            pacing={pacing}
            selectedId={selectedId}
            selected={selected}
            mode={config.mode}
            onSelect={id => { setSelectedId(id); setEvalError(''); }}
            onSend={send}
            onNewConversation={newConversation}
          />
          <aside className="panel side" aria-label="Metrics">
            <TokenMeter experiment={selected} loading={loading} />
            <EvaluationPanel
              experiment={selected}
              evaluating={evaluatingId != null && evaluatingId === selectedId}
              error={evalError}
              onEvaluate={() => selected && void runEvaluation(selected.id)}
            />
            <VerificationPanel experiment={selected} />
            <FeedbackPanel experiment={selected} onSave={saveFeedback} />
          </aside>
        </div>
      )}

      <HistoryDrawer open={historyOpen} items={history} loading={historyLoading}
        onClose={() => setHistoryOpen(false)} onOpen={id => void reopen(id)} />
      <footer className="privacy">
        Your name, prompts, generated responses and testing metrics are recorded for internal evaluation.
        Do not enter personal, confidential or customer data.
      </footer>
    </div>
  );
}
