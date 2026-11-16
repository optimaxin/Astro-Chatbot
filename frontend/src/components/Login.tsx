import { useState } from 'react';

interface Props {
  onSubmit: (token: string) => Promise<void>;
}

export default function Login({ onSubmit }: Props) {
  const [token, setToken] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token.trim()) return;
    setBusy(true);
    setError('');
    try {
      await onSubmit(token.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not sign in.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login">
      <form className="login-card" onSubmit={submit}>
        <h1>AI Model Playground</h1>
        <p className="hint">Internal testing tool. Enter the playground access token.</p>
        <input
          type="password"
          autoFocus
          value={token}
          onChange={e => setToken(e.target.value)}
          placeholder="Access token"
          aria-label="Access token"
        />
        {error && <p className="notice error">{error}</p>}
        <button className="btn primary" disabled={busy || !token.trim()}>
          {busy ? 'Checking…' : 'Enter playground'}
        </button>
      </form>
    </div>
  );
}
