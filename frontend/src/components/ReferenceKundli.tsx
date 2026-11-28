import { useEffect, useState } from 'react';
import { api, ApiError } from '../api';
import type { Birth, ReferenceKundli as ReferenceKundliData } from '../types';

interface Props {
  birth: Birth;
  placeId?: string;
  disabled: boolean;
  onAuthError: (err: unknown) => boolean;
  onChanged: (chart: ReferenceKundliData | null) => void;
}

function sameBirth(a: Birth, b: Birth): boolean {
  return a.date === b.date && a.time === b.time && a.latitude === b.latitude && a.longitude === b.longitude;
}

/** Generates and displays the reference Kundli the chatbot's replies are verified against — the same calculator production uses, not a second engine. */
export default function ReferenceKundli({ birth, placeId, disabled, onAuthError, onChanged }: Props) {
  const [chart, setChart] = useState<ReferenceKundliData | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    api.kundli.current()
      .then(c => { setChart(c); onChanged(c); })
      .catch(err => { if (!onAuthError(err)) setError(err instanceof Error ? err.message : 'Could not load the reference chart.'); })
      .finally(() => setLoading(false));
    // Only the first load reads the current chart; after that this component owns the state it set.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const generate = async () => {
    setGenerating(true);
    setError('');
    try {
      const next = await api.kundli.generate({
        date: birth.date, time: birth.time, latitude: birth.latitude, longitude: birth.longitude,
        placeLabel: birth.placeLabel, placeId,
      });
      setChart(next);
      onChanged(next);
    } catch (err) {
      if (!onAuthError(err)) setError(err instanceof ApiError ? err.message : 'Could not generate the reference Kundli.');
    } finally {
      setGenerating(false);
    }
  };

  const outdated = chart != null && !sameBirth(chart.birth, birth);
  const hasCoordinates = Number.isFinite(birth.latitude) && Number.isFinite(birth.longitude);
  const k = chart?.chart.kundli;
  const activeMahadasha = chart?.chart.mahadashaTimeline.find(p => p.active);
  const activeAntardasha = activeMahadasha?.antardashas.find(a => a.active);

  return (
    <section className="section">
      <div className="section-head">
        <h2 className="section-title">Reference Kundli</h2>
        <button className="btn ghost small" onClick={() => void generate()} disabled={disabled || generating || !hasCoordinates}>
          {generating ? 'Generating…' : chart ? 'Regenerate' : 'Generate Kundli'}
        </button>
      </div>
      <p className="hint">
        The same chart production uses, computed from these birth details — used to verify the chatbot's factual
        chart claims below and in the chat.
      </p>
      {error && <div className="notice error">{error}</div>}
      {!hasCoordinates && !error && <p className="hint">Search and select a birthplace above (or use the sample chart) to generate a reference Kundli.</p>}
      {loading && <p className="empty">Loading…</p>}
      {!loading && !chart && !error && hasCoordinates && <p className="empty">No reference chart yet for this session.</p>}
      {outdated && (
        <p className="hint warn">Birth details changed since this chart was generated — regenerate for accurate verification.</p>
      )}

      {k && (
        <div className={outdated ? 'is-stale' : ''}>
          <div className="scores">
            <div className="score-row"><span>Lagna</span><span /><span className="mono score-num">{k.ascendant.rashi}</span></div>
            <div className="score-row"><span>Moon sign</span><span /><span className="mono score-num">{k.planets.find(p => p.id === 'moon')?.rashi}</span></div>
            <div className="score-row"><span>Nakshatra</span><span /><span className="mono score-num">{k.moonNakshatra.name} ({k.moonNakshatra.pada})</span></div>
            <div className="score-row"><span>Mahadasha</span><span /><span className="mono score-num">{activeMahadasha?.lord ?? '—'}</span></div>
            <div className="score-row"><span>Antardasha</span><span /><span className="mono score-num">{activeAntardasha?.lord ?? '—'}</span></div>
          </div>

          <button className="btn ghost small" style={{ marginTop: 10 }} onClick={() => setExpanded(e => !e)}>
            {expanded ? 'Hide planet positions' : 'Show planet positions'}
          </button>
          {expanded && (
            <div className="list-block">
              <ul className="claim-list">
                {k.planets.map(p => (
                  <li key={p.id} className="claim-row">
                    <span className="chip tone-neutral">{p.id}</span>
                    <div>
                      <p className="claim-text">
                        {p.rashi}, house {p.house}{p.retrograde ? ' (retrograde)' : ''}
                      </p>
                      <p className="hint mono">{p.nakshatra} pada {p.nakshatraPada}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <p className="hint mono">Engine: {chart!.engine_source}</p>
        </div>
      )}
    </section>
  );
}
