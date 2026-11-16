export const fmtInt = (n: number | null | undefined) => (n == null ? '—' : n.toLocaleString());

export function fmtCost(n: number | null | undefined, currency: string | null | undefined = 'USD'): string {
  if (n == null) return 'N/A';
  const symbol = currency === 'USD' || !currency ? '$' : `${currency} `;
  if (n === 0) return `${symbol}0`;
  return `${symbol}${n < 0.01 ? n.toFixed(6) : n.toFixed(4)}`;
}

export const fmtLatency = (ms: number | null | undefined) =>
  ms == null ? '—' : ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(2)} s`;

export const fmtTime = (epochSeconds: number) =>
  new Date(epochSeconds * 1000).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  });

export const fmtScore = (n: number | null | undefined) => (n == null ? '—' : n.toFixed(1));
