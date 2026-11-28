import LocationSearch from './LocationSearch';
import ReferenceKundli from './ReferenceKundli';
import type { Catalog, PlaygroundConfig } from '../types';

interface Props {
  catalog: Catalog;
  config: PlaygroundConfig;
  onChange: (next: PlaygroundConfig) => void;
  disabled: boolean;
  onAuthError: (err: unknown) => boolean;
}

export default function ConfigPanel({ catalog, config, onChange, disabled, onAuthError }: Props) {
  const set = (patch: Partial<PlaygroundConfig>) => onChange({ ...config, ...patch });
  const chatbotUp = catalog.modes.find(m => m.id === 'kundli_chatbot')?.available !== false;

  return (
    <aside className="panel config" aria-label="Test configuration">
      <section className="section">
        <h2 className="section-title">Kundli Chatbot Testing</h2>
        {!chatbotUp && <p className="hint warn">The Kundli chatbot is not connected on this deployment.</p>}
      </section>

      <KundliSettings catalog={catalog} config={config} set={set} disabled={disabled} onAuthError={onAuthError} />

      <section className="section">
        <label className="toggle">
          <input type="checkbox" checked={config.autoEvaluate} onChange={e => set({ autoEvaluate: e.target.checked })} />
          <span>Run AI evaluation after each response</span>
        </label>
      </section>
    </aside>
  );
}

type SectionProps = {
  catalog: Catalog; config: PlaygroundConfig; set: (p: Partial<PlaygroundConfig>) => void; disabled: boolean;
  onAuthError: (err: unknown) => boolean;
};

function KundliSettings({ catalog, config, set, disabled, onAuthError }: SectionProps) {
  const birth = config.birth;
  const setBirth = (patch: Partial<typeof birth>) => set({ birth: { ...birth, ...patch } });
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
        </div>
        {catalog.location?.enabled ? (
          <LocationSearch
            placeLabel={birth.placeLabel ?? ''}
            disabled={disabled}
            onResolved={({ latitude, longitude, placeLabel, placeId }) =>
              set({ birth: { ...birth, latitude, longitude, placeLabel }, placeId })}
          />
        ) : (
          <p className="hint warn">Birth location search is currently unavailable. Please try again later.</p>
        )}
        <button className="btn small" type="button" disabled={disabled} onClick={() => set({ birth: catalog.chatbot.sample_birth, placeId: undefined })}>
          Use sample chart (demo)
        </button>
      </section>

      <ReferenceKundli
        birth={birth}
        placeId={config.placeId}
        disabled={disabled}
        onAuthError={onAuthError}
        onChanged={chart => { if (chart) set({ birth: { ...birth, timezoneOffsetMinutes: chart.birth.timezoneOffsetMinutes } }); }}
      />
    </>
  );
}
