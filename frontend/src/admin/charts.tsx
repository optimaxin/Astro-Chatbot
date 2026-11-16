import { fmtInt } from '../format';

export interface BarItem { label: string; value: number; note?: string }

export function HBars({ items, format = fmtInt, empty = 'No data yet.' }: {
  items: BarItem[]; format?: (n: number) => string; empty?: string;
}) {
  const max = Math.max(0, ...items.map(i => i.value));
  if (!items.length || max === 0) return <p className="empty">{empty}</p>;
  return (
    <div className="hbars">
      {items.map(i => (
        <div key={i.label} className="hbar-row" title={`${i.label}: ${format(i.value)}`}>
          <span className="hbar-label truncate">{i.label}</span>
          <div className="hbar-track"><div className="hbar-fill" style={{ width: `${Math.max(2, (i.value / max) * 100)}%` }} /></div>
          <span className="hbar-value mono">{format(i.value)}{i.note ? <em> {i.note}</em> : null}</span>
        </div>
      ))}
    </div>
  );
}

export interface DayItem { date: string; value: number; secondary?: number }

/** Vertical bars per day; `secondary` (e.g. failures) is stacked on top in red. */
export function DayBars({ days, format = fmtInt, primaryLabel, secondaryLabel }: {
  days: DayItem[]; format?: (n: number) => string; primaryLabel: string; secondaryLabel?: string;
}) {
  const shown = days.slice(-30);
  const max = Math.max(0, ...shown.map(d => d.value + (d.secondary ?? 0)));
  if (!shown.length || max === 0) return <p className="empty">No data yet.</p>;
  return (
    <div>
      <div className="daybars" role="img" aria-label={`${primaryLabel} per day`}>
        {shown.map(d => (
          <div key={d.date} className="daybar"
            title={`${d.date} — ${primaryLabel}: ${format(d.value)}${secondaryLabel ? `, ${secondaryLabel}: ${format(d.secondary ?? 0)}` : ''}`}>
            <div className="daybar-stack">
              {d.secondary ? <div className="daybar-seg bad" style={{ height: `${(d.secondary / max) * 100}%` }} /> : null}
              <div className="daybar-seg" style={{ height: `${(d.value / max) * 100}%` }} />
            </div>
            <span className="daybar-label">{d.date.slice(5)}</span>
          </div>
        ))}
      </div>
      <div className="legend">
        <span className="legend-item"><i className="dot accent" />{primaryLabel}</span>
        {secondaryLabel && <span className="legend-item"><i className="dot bad" />{secondaryLabel}</span>}
      </div>
    </div>
  );
}
