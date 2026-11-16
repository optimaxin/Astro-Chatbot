import { useState } from 'react';

interface Props {
  onStart: (name: string) => Promise<void>;
}

const MAX = 80;

// Mirrors the server's rules; the server is what actually enforces them.
export function checkName(raw: string): { ok: true; name: string } | { ok: false; error: string } {
  const name = raw.normalize('NFC').trim().replace(/\s+/gu, ' ');
  if (!name) return { ok: false, error: 'Enter your name to continue.' };
  if ([...name].length > MAX) return { ok: false, error: `Use at most ${MAX} characters.` };
  if (/\p{Cc}/u.test(name)) return { ok: false, error: 'The name contains characters that are not allowed.' };
  if (!/[\p{L}\p{N}]/u.test(name)) return { ok: false, error: 'Use at least one letter or number.' };
  return { ok: true, name };
}

export default function Welcome({ onStart }: Props) {
  const [value, setValue] = useState('');
  const [touched, setTouched] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const check = checkName(value);
  const shown = error || (touched && !check.ok ? check.error : '');

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setTouched(true);
    if (!check.ok || busy) return;
    setBusy(true);
    setError('');
    try {
      await onStart(check.name);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start the session.');
      setBusy(false);
    }
  };

  return (
    <div className="login">
      <form className="login-card welcome" onSubmit={submit} noValidate>
        <span className="brand-mark" aria-hidden="true">◆</span>
        <h1>Welcome to the AI Model Playground</h1>
        <p className="hint">
          Test the Tredev Kundli chatbot and compare AI models. Tell us who you are so your results can be
          matched to you. There is nothing to sign up for.
        </p>
        <label className="field">
          <span>Your name</span>
          <input
            type="text"
            autoFocus
            autoComplete="name"
            value={value}
            maxLength={MAX * 2}
            onChange={e => { setValue(e.target.value); setError(''); }}
            onBlur={() => setTouched(true)}
            placeholder="e.g. Asha Rao"
            aria-invalid={Boolean(shown)}
            aria-describedby="name-help"
          />
        </label>
        <p id="name-help" className={`hint ${shown ? 'error-text' : ''}`} role={shown ? 'alert' : undefined}>
          {shown || `Up to ${MAX} characters. No email, phone number or password needed.`}
        </p>
        <button className="btn primary" disabled={!check.ok || busy}>
          {busy ? 'Starting…' : 'Start testing'}
        </button>
        <p className="hint privacy-note">
          <strong>Privacy:</strong> your name, your prompts, the generated responses and testing metrics (tokens, cost,
          latency, ratings) are recorded for internal evaluation of the chatbot and models. Please do not enter
          personal, confidential or customer information.
        </p>
      </form>
    </div>
  );
}
