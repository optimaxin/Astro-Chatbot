import type { Catalog, ParamKey, PlaygroundConfig } from '../types';

interface Props {
  catalog: Catalog;
  config: PlaygroundConfig;
  onChange: (next: PlaygroundConfig) => void;
  disabled: boolean;
}

const PARAMS: { key: ParamKey; label: string; min: number; max: number; step: number; slider: boolean }[] = [
  { key: 'temperature', label: 'Temperature', min: 0, max: 2, step: 0.05, slider: true },
  { key: 'top_p', label: 'Top P', min: 0.05, max: 1, step: 0.01, slider: true },
  { key: 'top_k', label: 'Top K', min: 1, max: 500, step: 1, slider: false },
  { key: 'max_tokens', label: 'Max output tokens', min: 1, max: 32000, step: 1, slider: false },
];

export default function ConfigPanel({ catalog, config, onChange, disabled }: Props) {
  const provider = catalog.providers.find(p => p.id === config.provider);
  const models = catalog.models.filter(m => m.provider === config.provider);
  const model = models.find(m => m.id === config.model);
  const set = (patch: Partial<PlaygroundConfig>) => onChange({ ...config, ...patch });

  const selectProvider = (id: string) => {
    const first = catalog.models.find(m => m.provider === id);
    set({ provider: id, model: first?.id ?? '' });
  };

  return (
    <aside className="panel config" aria-label="Model configuration">
      <section className="section">
        <h2 className="section-title">Model</h2>
        <label className="field">
          <span>Provider</span>
          <select value={config.provider} onChange={e => selectProvider(e.target.value)} disabled={disabled}>
            {catalog.providers.map(p => (
              <option key={p.id} value={p.id}>
                {p.name}{p.configured ? '' : ' (no API key)'}
              </option>
            ))}
          </select>
        </label>
        {provider && !provider.configured && (
          <p className="hint warn">This provider has no API key on the backend. Requests will fail.</p>
        )}
        <label className="field">
          <span>Model</span>
          <select value={config.model} onChange={e => set({ model: e.target.value })} disabled={disabled}>
            {models.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </label>
        <p className="hint mono">
          {model?.pricing
            ? `$${model.pricing.input_per_1m} in / $${model.pricing.output_per_1m} out per 1M tokens`
            : provider?.kind === 'aicore' ? 'Cost: as reported by the chatbot' : 'Pricing: N/A'}
        </p>
        {provider?.kind === 'aicore' && (
          <div className="notice">
            <strong>Testing the live chatbot</strong>
            <span>
              Messages go to the Tredev chatbot itself (knowledge base, guardrails, reply checks). It keeps its
              own conversation memory, so the system prompt and parameters below do not apply.
            </span>
            <span className="muted">
              A sample chart (12 Apr 1995, 06:30, New Delhi) is used unless you paste birth details as JSON in
              Context, e.g.{' '}
              <code className="mono">{'{"date":"1995-04-12","time":"06:30","timezoneOffsetMinutes":330,"latitude":28.61,"longitude":77.21}'}</code>
            </span>
          </div>
        )}
      </section>

      <section className="section">
        <h2 className="section-title">Parameters</h2>
        {PARAMS.map(p => {
          const supported = provider?.supported_params.includes(p.key) ?? false;
          const on = config.enabled[p.key] && supported;
          return (
            <div key={p.key} className={`param ${supported ? '' : 'param-off'}`}>
              <label className="param-head">
                <input
                  type="checkbox"
                  checked={on}
                  disabled={!supported || disabled}
                  onChange={e => set({ enabled: { ...config.enabled, [p.key]: e.target.checked } })}
                />
                <span>{p.label}</span>
                <span className="mono param-value">{supported ? config.params[p.key] : 'not supported'}</span>
              </label>
              {supported && (
                <input
                  className={p.slider ? 'slider' : 'number'}
                  type={p.slider ? 'range' : 'number'}
                  min={p.min}
                  max={p.max}
                  step={p.step}
                  value={config.params[p.key]}
                  disabled={!on || disabled}
                  onChange={e => set({ params: { ...config.params, [p.key]: Number(e.target.value) } })}
                />
              )}
            </div>
          );
        })}
        <p className="hint">Unchecked or unsupported parameters are not sent; the provider default applies.</p>
      </section>

      <section className="section grow">
        <h2 className="section-title">Prompt</h2>
        <label className="field">
          <span>System prompt</span>
          <textarea
            rows={6}
            value={config.systemPrompt}
            onChange={e => set({ systemPrompt: e.target.value })}
            placeholder="You are a helpful conversational AI..."
            disabled={disabled || provider?.kind === 'aicore'}
            title={provider?.kind === 'aicore' ? 'The chatbot uses its own system prompt' : undefined}
          />
        </label>
        <label className="field">
          <span>Context (optional)</span>
          <textarea
            rows={5}
            value={config.context}
            onChange={e => set({ context: e.target.value })}
            placeholder="User / Kundli / context data..."
            disabled={disabled}
          />
        </label>
        <label className="toggle">
          <input
            type="checkbox"
            checked={config.autoEvaluate}
            onChange={e => set({ autoEvaluate: e.target.checked })}
          />
          <span>Run AI evaluation after each response</span>
        </label>
      </section>
    </aside>
  );
}
