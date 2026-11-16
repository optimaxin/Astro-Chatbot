import { useCallback, useEffect, useState } from 'react';
import { api, ApiError, getTesterName, getToken, randomId, setTesterName, setToken } from './api';
import type { Catalog, ChatTurn, Experiment, ExperimentSummary, ParamKey, PlaygroundConfig } from './types';
import ConfigPanel from './components/ConfigPanel';
import ChatPanel, { type Turn } from './components/ChatPanel';
import TokenMeter from './components/TokenMeter';
import EvaluationPanel from './components/EvaluationPanel';
import FeedbackPanel from './components/FeedbackPanel';
import HistoryDrawer from './components/HistoryDrawer';
import Login from './components/Login';

const PARAM_KEYS: ParamKey[] = ['temperature', 'top_p', 'top_k', 'max_tokens'];
const newSessionId = () => `s-${Math.random().toString(36).slice(2, 12)}`;
let turnSeq = 0;
const turnKey = () => `t${++turnSeq}`;

function defaultConfig(catalog: Catalog): PlaygroundConfig {
  const provider = catalog.providers.find(p => p.configured) ?? catalog.providers[0];
  const model = catalog.models.find(m => m.provider === provider?.id);
  return {
    provider: provider?.id ?? '',
    model: model?.id ?? '',
    systemPrompt: 'You are a helpful, friendly conversational assistant. Keep replies short and natural.',
    context: '',
    params: { temperature: 0.7, top_p: 0.95, top_k: 40, max_tokens: 1000 },
    enabled: { temperature: true, top_p: false, top_k: false, max_tokens: true },
    autoEvaluate: true,
  };
}

// Only answered exchanges are sent back as history; a failed turn never reached the model.
function historyFrom(turns: Turn[]): ChatTurn[] {
  const out: ChatTurn[] = [];
  turns.forEach((t, i) => {
    const next = turns[i + 1];
    if (t.role === 'user' && next?.role === 'assistant') {
      out.push({ role: 'user', content: t.content }, { role: 'assistant', content: next.content });
    }
  });
  return out;
}

function readTheme(): 'light' | 'dark' {
  try {
    const saved = localStorage.getItem('playground.theme');
    if (saved === 'light' || saved === 'dark') return saved;
  } catch { /* storage unavailable */ }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  // Testers need no login. Only a deployment with a shared token asks for one.
  const [needsToken, setNeedsToken] = useState(false);
  const [authVersion, setAuthVersion] = useState(0);
  const [testerName, setName] = useState(getTesterName);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [config, setConfig] = useState<PlaygroundConfig | null>(null);
  const [fatal, setFatal] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [sessionId, setSessionId] = useState(newSessionId);
  const [experiments, setExperiments] = useState<Record<number, Experiment>>({});
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [evaluatingId, setEvaluatingId] = useState<number | null>(null);
  const [evalError, setEvalError] = useState('');
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState<ExperimentSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [theme, setTheme] = useState(readTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('playground.theme', theme); } catch { /* ignore */ }
  }, [theme]);

  const handleAuthError = useCallback((err: unknown) => {
    if (err instanceof ApiError && err.status === 401) {
      setToken(null);
      setNeedsToken(true);
      return true;
    }
    return false;
  }, []);

  useEffect(() => {
    if (needsToken) return;
    api.models()
      .then(c => {
        setFatal('');
        setCatalog(c);
        setConfig(prev => prev ?? defaultConfig(c));
      })
      .catch(err => { if (!handleAuthError(err)) setFatal(err.message); });
  }, [needsToken, authVersion, handleAuthError]);

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
    const params = Object.fromEntries(
      PARAM_KEYS.filter(k => config.enabled[k]).map(k => [k, config.params[k]]),
    ) as Partial<Record<ParamKey, number>>;
    const history = historyFrom(turns);
    setTurns(prev => [...prev, { key: turnKey(), role: 'user', content: message }]);
    setLoading(true);
    setEvalError('');
    try {
      const exp = await api.chat({
        provider: config.provider,
        model: config.model,
        message,
        system_prompt: config.systemPrompt || undefined,
        context: config.context || undefined,
        history,
        params,
        session_id: sessionId,
        request_id: randomId('r'),
        tester_name: testerName.trim() || undefined,
      });
      remember(exp);
      setSelectedId(exp.id);
      setTurns(prev => [...prev, exp.status === 'success'
        ? { key: turnKey(), role: 'assistant', content: exp.response ?? '', experimentId: exp.id }
        : { key: turnKey(), role: 'error', content: `Generation failed — ${exp.error}`, experimentId: exp.id }]);
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

  const newSession = useCallback(() => {
    setTurns([]);
    setSelectedId(null);
    setSessionId(newSessionId());
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
      const exp = await api.experiment(id);
      remember(exp);
      const prior = (exp.effective_request?.messages ?? []).slice(0, -1).filter(m => m.role !== 'system');
      setTurns([
        ...prior.map(m => ({ key: turnKey(), role: m.role as 'user' | 'assistant', content: m.content })),
        { key: turnKey(), role: 'user', content: exp.user_prompt },
        exp.status === 'success'
          ? { key: turnKey(), role: 'assistant', content: exp.response ?? '', experimentId: exp.id }
          : { key: turnKey(), role: 'error', content: `Generation failed — ${exp.error}`, experimentId: exp.id },
      ]);
      setSessionId(exp.session_id);
      setSelectedId(exp.id);
      setConfig(prev => prev && ({
        ...prev,
        provider: exp.provider,
        model: exp.model,
        systemPrompt: exp.system_prompt ?? '',
        context: exp.context ?? '',
        params: { ...prev.params, ...(exp.params ?? {}) },
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
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'n') { e.preventDefault(); newSession(); }
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'h') { e.preventDefault(); void openHistory(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [newSession, openHistory]);

  const saveFeedback = async (rating: 'good' | 'bad' | null, feedback: string, edited: string) => {
    if (!selected) return;
    remember(await api.feedback(selected.id, rating, feedback, edited));
  };

  if (needsToken) {
    return (
      <Login onSubmit={async token => {
        await api.models(token);
        setToken(token);
        setNeedsToken(false);
        setAuthVersion(v => v + 1);
      }} />
    );
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
          <input
            className="tester-name"
            value={testerName}
            maxLength={60}
            onChange={e => { setName(e.target.value); setTesterName(e.target.value); }}
            placeholder="Your name (optional)"
            aria-label="Tester name (optional)"
          />
          <button className="btn ghost" onClick={() => void openHistory()} title="Ctrl+Shift+H">History</button>
          <button className="btn ghost" onClick={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))}
            aria-label="Toggle theme">{theme === 'dark' ? '☀' : '☾'}</button>
          {getToken() && (
            <button className="btn ghost" onClick={() => { setToken(null); setNeedsToken(true); }}>Sign out</button>
          )}
        </div>
      </header>

      {fatal && <div className="notice error banner">{fatal}</div>}
      {!catalog || !config ? (
        !fatal && <p className="empty center">Loading models…</p>
      ) : (
        <div className="layout">
          <ConfigPanel catalog={catalog} config={config} onChange={setConfig} disabled={loading} />
          <ChatPanel
            turns={turns}
            loading={loading}
            selectedId={selectedId}
            selected={selected}
            onSelect={id => { setSelectedId(id); setEvalError(''); }}
            onSend={send}
            onNewSession={newSession}
          />
          <aside className="panel side" aria-label="Metrics">
            <TokenMeter experiment={selected} loading={loading} />
            <EvaluationPanel
              experiment={selected}
              evaluating={evaluatingId != null && evaluatingId === selectedId}
              error={evalError}
              onEvaluate={() => selected && void runEvaluation(selected.id)}
            />
            <FeedbackPanel experiment={selected} onSave={saveFeedback} />
          </aside>
        </div>
      )}

      <HistoryDrawer open={historyOpen} items={history} loading={historyLoading}
        onClose={() => setHistoryOpen(false)} onOpen={id => void reopen(id)} />
      <footer className="privacy">
        Every prompt and response you send here is logged and reviewed internally to evaluate the models.
        Do not enter personal, confidential or customer data.
      </footer>
    </div>
  );
}
