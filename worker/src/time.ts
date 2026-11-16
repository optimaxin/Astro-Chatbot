/** Time helpers for the report timezone (default Asia/Kolkata). */

const formatters = new Map<string, Intl.DateTimeFormat>();

function parts(epochMs: number, tz: string) {
  let f = formatters.get(tz);
  if (!f) {
    f = new Intl.DateTimeFormat('en-GB', {
      timeZone: tz, hourCycle: 'h23', year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
    formatters.set(tz, f);
  }
  const p = Object.fromEntries(f.formatToParts(new Date(epochMs)).map((x) => [x.type, x.value]));
  return { y: +p.year!, mo: +p.month!, d: +p.day!, h: +p.hour!, mi: +p.minute!, s: +p.second! };
}

/** 'YYYY-MM-DD' of an instant in the report timezone. */
export function dayKey(epochSec: number, tz: string): string {
  const { y, mo, d } = parts(epochSec * 1000, tz);
  return `${y}-${String(mo).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
}

/** Wall-clock time in the report timezone, as a UTC-based Date (what Excel needs to show it unchanged). */
export function wallClock(epochSec: number, tz: string): Date {
  const { y, mo, d, h, mi, s } = parts(epochSec * 1000, tz);
  return new Date(Date.UTC(y, mo - 1, d, h, mi, s));
}

/** The instant (epoch seconds) at which a calendar day starts in the report timezone. */
export function dayStart(ymd: string, tz: string): number {
  const [y, m, d] = ymd.split('-').map(Number) as [number, number, number];
  const guess = Date.UTC(y, m - 1, d);
  // Correct the guess by the zone's offset at that moment (twice, to settle across DST changes).
  let t = guess;
  for (let i = 0; i < 2; i++) {
    const w = parts(t, tz);
    const asUtc = Date.UTC(w.y, w.mo - 1, w.d, w.h, w.mi, w.s);
    t -= asUtc - guess;
  }
  return t / 1000;
}

export const startOfToday = (tz: string, nowMs = Date.now()): number => dayStart(dayKey(nowMs / 1000, tz), tz);

export function addDays(ymd: string, days: number): string {
  const [y, m, d] = ymd.split('-').map(Number) as [number, number, number];
  return new Date(Date.UTC(y, m - 1, d + days)).toISOString().slice(0, 10);
}
