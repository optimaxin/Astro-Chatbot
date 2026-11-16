import type { Catalog, Mode, ParamKey, PlaygroundConfig } from '../types';

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
  const set = (patch: Partial<PlaygroundConfig>) => onChange({ ...config, ...patch });
  const kundli = config.mode === 'kundli_chatbot';
  const chatbotUp = catalog.modes.find(m => m.id === 'kundli_chatbot')?.available !== false;

  return (
    <aside className="panel config" aria-label="Test configuration">
      <section className="section">
        <h2 className="section-title">Testing mode</h2>
        <div className="segmented" role="radiogroup" aria-label="Testing mode">
          {catalog.modes.map(m => (
            <button
              key={m.id}
              type="button"
              role="radio"
              aria-checked={config.mode === m.id}
              className={`seg ${config.mode === m.id ? 'active' : ''}`}
              disabled={disabled || m.available === false}
              onClick={() => set({ mode: m.id as Mode })}
            >
              {m.label}
            </button>
          ))}
        </div>
        {!chatbotUp && <p className="hint warn">The Kundli chatbot is not connected on this deployment.</p>}
      </section>

      {kundli ? <KundliSettings catalog={catalog} config={config} set={set} disabled={disabled} />
        : <DirectSettings catalog={catalog} config={config} set={set} disabled={disabled} />}

      <section className="section">
        <label className="toggle">
          <input type="checkbox" checked={config.autoEvaluate} onChange={e => set({ autoEvaluate: e.target.checked })} />
          <span>Run AI evaluation after each response</span>
        </label>
      </section>
    </aside>
  );
}

type SectionProps = { catalog: Catalog; config: PlaygroundConfig; set: (p: Partial<PlaygroundConfig>) => void; disabled: boolean };

function KundliSettings({ catalog, config, set, disabled }: SectionProps) {
  const birth = config.birth;
  const setBirth = (patch: Partial<typeof birth>) => set({ birth: { ...birth, ...patch } });
  const num = (v: string) => (v.trim() === '' ? NaN : Number(v));
  return (
    <>
      <section className="section">
        <h2 className="section-title">Kundli Chatbot</h2>
        <div className="notice">
          <strong>Testing the real chatbot</strong>
          <span>
            Your messages go to the same Kundli chatbot the website uses, with its own chart data, knowledge base,
            instructions and conversation memory. Nothing is bypassed or re-generated here.
          </span>
        </div>
        <label className="field">
          <span>Reply style</span>
          <select value={config.style} onChange={e => set({ style: e.target.value })} disabled={disabled}>
            <option value="auto">Automatic (as on the website)</option>
            {catalog.chatbot.styles.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
        </label>
        <p className="hint">Automatic follows the language you write in.</p>
      </section>

      <section className="section">
        <h2 className="section-title">Birth details</h2>
        <p className="hint">The chatbot builds the Kundli from these, exactly as for a website visitor.</p>
        <div className="grid2">
          <label className="field"><span>Date of birth</span>
            <input type="date" value={birth.date} disabled={disabled} onChange={e => setBirth({ date: e.target.value })} /></label>
          <label className="field"><span>Time (24h)</span>
            <input type="time" value={birth.time} disabled={disabled} onChange={e => setBirth({ time: e.target.value })} /></label>
          <label className="field"><span>Latitude</span>
            <input type="number" step="0.0001" min={-90} max={90} value={Number.isNaN(birth.latitude) ? '' : birth.latitude}
              disabled={disabled} onChange={e => setBirth({ latitude: num(e.target.value) })} /></label>
          <label className="field"><span>Longitude</span>
            <input type="number" step="0.0001" min={-180} max={180} value={Number.isNaN(birth.longitude) ? '' : birth.longitude}
              disabled={disabled} onChange={e => setBirth({ longitude: num(e.target.value) })} /></label>
          <label className="field"><span>UTC offset (min)</span>
            <input type="number" step="1" min={-900} max={900} value={Number.isNaN(birth.timezoneOffsetMinutes) ? '' : birth.timezoneOffsetMinutes}
              disabled={disabled} onChange={e => setBirth({ timezoneOffsetMinutes: num(e.target.value) })} /></label>
          <label className="field"><span>Place (label)</span>
            <input type="text" maxLength={120} value={birth.placeLabel ?? ''} disabled={disabled}
              onChange={e => setBirth({ placeLabel: e.target.value })} /></label>
        </div>
        <button className="btn small" type="button" disabled={disabled} onClick={() => set({ birth: catalog.chatbot.sample_birth })}>
          Use sample chart
        </button>
        <p className="hint">Changing the chart mid-conversation is fine; start a new conversation for a clean test.</p>
      </section>
    </>
  );
}

function DirectSettings({ catalog, config, set, disabled }: SectionProps) {
  const provider = catalog.providers.find(p => p.id === config.provider);
  const models = catalog.models.filter(m => m.provider === config.provider);
  const model = models.find(m => m.id === config.model);
  const selectProvider = (id: string) => {
    const first = catalog.models.find(m => m.provider === id);
    set({ provider: id, model: first?.id ?? '' });
  };

  return (
    <>
      <section className="section">
        <h2 className="section-title">Model</h2>
        <label className="field">
          <span>Provider</span>
          <select value={config.provider} onChange={e => selectProvider(e.target.value)} disabled={disabled}>
            {catalog.providers.map(p => (
              <option key={p.id} value={p.id}>{p.name}{p.configured ? '' : ' (no API key)'}</option>
            ))}
          </select>
        </label>
        {provider && !provider.configured && (
          <p className="hint warn">This provider has no API key on the backend. Requests will be refused.</p>
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
            : 'Pricing: N/A'}
        </p>
        <p className="hint">Calls this provider directly with your own prompt. It does not use the Kundli chatbot.</p>
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
          <textarea rows={6} value={config.systemPrompt} onChange={e => set({ systemPrompt: e.target.value })}
            placeholder="You are a helpful conversational AI..." disabled={disabled} />
        </label>
        <label className="field">
          <span>Context (optional)</span>
          <textarea rows={5} value={config.context} onChange={e => set({ context: e.target.value })}
            placeholder="User / Kundli / context data..." disabled={disabled} />
        </label>
      </section>
    </>
  );
}
