import { useCallback, useEffect, useState } from 'react';
import { fmtTime } from '../format';
import { adminApi, AdminError, type AuditEntry, type Page } from './adminApi';
import Dashboard from './Dashboard';
import Experiments from './Experiments';
import './admin.css';

type Tab = 'dashboard' | 'experiments' | 'audit';

function readTheme(): 'light' | 'dark' {
  try {
    const saved = localStorage.getItem('playground.theme');
    if (saved === 'light' || saved === 'dark') return saved;
  } catch { /* storage unavailable */ }
  return 'dark';
}

function AdminLogin({ onDone }: { onDone: () => void }) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      await adminApi.login(password);
      setPassword('');
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not sign in.');
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="login">
      <form className="login-card" onSubmit={submit}>
        <h1>Playground admin</h1>
        <p className="hint">Analytics, experiment logs and Excel reports. Testers do not need this page.</p>
        <input type="password" autoFocus autoComplete="current-password" value={password}
          onChange={e => setPassword(e.target.value)} placeholder="Admin password" aria-label="Admin password" />
        {error && <p className="notice error">{error}</p>}
        <button className="btn primary" disabled={busy || !password}>{busy ? 'Checking…' : 'Sign in'}</button>
        <a className="hint" href="/">← Back to the playground</a>
      </form>
    </div>
  );
}

function AuditLog({ onAuthError }: { onAuthError: (e: unknown) => boolean }) {
  const [page, setPage] = useState(1);
  const [data, setData] = useState<Page<AuditEntry> | null>(null);
  useEffect(() => { adminApi.audit(page).then(setData).catch(onAuthError); }, [page, onAuthError]);
  const pages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;
  return (
    <div className="admin-page">
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>Time</th><th>Action</th><th>IP</th><th>Details</th></tr></thead>
          <tbody>
            {data?.items.map(a => (
              <tr key={a.id}>
                <td className="nowrap">{fmtTime(a.timestamp)}</td><td className="mono">{a.action}</td>
                <td className="mono">{a.ip ?? '—'}</td>
                <td className="mono preview">{Object.keys(a.details).length ? JSON.stringify(a.details) : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pager">
        <button className="btn small" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>Previous</button>
        <span className="mono">{page} / {pages}</span>
        <button className="btn small" disabled={page >= pages} onClick={() => setPage(p => p + 1)}>Next</button>
      </div>
    </div>
  );
}

export default function AdminApp() {
  const [state, setState] = useState<'checking' | 'out' | 'in'>('checking');
  const [tab, setTab] = useState<Tab>('dashboard');
  const [theme, setTheme] = useState(readTheme);
  const [notice, setNotice] = useState('');

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('playground.theme', theme); } catch { /* ignore */ }
  }, [theme]);

  useEffect(() => {
    document.title = 'Playground Admin';
    adminApi.me().then(() => setState('in')).catch(() => setState('out'));
  }, []);

  const onAuthError = useCallback((err: unknown) => {
    if (err instanceof AdminError && err.status === 401) {
      setState('out');
      return true;
    }
    return false;
  }, []);

  if (state === 'checking') return <p className="empty center">Loading…</p>;
  if (state === 'out') return <AdminLogin onDone={() => setState('in')} />;

  const backup = async () => {
    setNotice('');
    try { await adminApi.backup(); } catch (e) { if (!onAuthError(e)) setNotice((e as Error).message); }
  };

  return (
    <div className="app admin">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">◆</span>
          <h1>Playground Admin</h1>
        </div>
        <nav className="tabs" aria-label="Admin sections">
          {(['dashboard', 'experiments', 'audit'] as Tab[]).map(t => (
            <button key={t} className={`tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
              {t === 'audit' ? 'Audit log' : t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
        <div className="toolbar-actions">
          <button className="btn ghost" onClick={() => void backup()} title="Download a consistent copy of the SQLite database">
            Backup DB</button>
          <button className="btn ghost" onClick={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))}
            aria-label="Toggle theme">{theme === 'dark' ? '☀' : '☾'}</button>
          <button className="btn ghost" onClick={async () => { try { await adminApi.logout(); } finally { setState('out'); } }}>
            Sign out</button>
        </div>
      </header>
      {notice && <div className="notice error banner">{notice}</div>}
      <main className="admin-main">
        {tab === 'dashboard' && <Dashboard onAuthError={onAuthError} />}
        {tab === 'experiments' && <Experiments onAuthError={onAuthError} />}
        {tab === 'audit' && <AuditLog onAuthError={onAuthError} />}
      </main>
    </div>
  );
}
