import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { PlaceSuggestion } from '../types';

interface Props {
  placeLabel: string;
  disabled: boolean;
  onResolved: (place: { latitude: number; longitude: number; placeLabel: string; placeId?: string }) => void;
}

/** Birthplace search (proxied through the Tredev backend): type → suggestions → resolve to coordinates on pick. */
export default function LocationSearch({ placeLabel, disabled, onResolved }: Props) {
  const [query, setQuery] = useState(placeLabel ?? '');
  const [suggestions, setSuggestions] = useState<PlaceSuggestion[]>([]);
  const [open, setOpen] = useState(false);
  const [searching, setSearching] = useState(false);
  const [resolving, setResolving] = useState(false);
  const [error, setError] = useState('');
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => setQuery(placeLabel ?? ''), [placeLabel]);

  useEffect(() => {
    clearTimeout(debounceRef.current);
    const q = query.trim();
    if (q.length < 2 || q === placeLabel) {
      setSuggestions([]);
      return;
    }
    setSearching(true);
    debounceRef.current = setTimeout(() => {
      api.locations.suggest(q)
        .then(setSuggestions)
        .catch(() => setSuggestions([]))
        .finally(() => setSearching(false));
    }, 300);
    return () => clearTimeout(debounceRef.current);
    // placeLabel only decides when to skip searching; it never drives a re-search itself.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  const pick = async (s: PlaceSuggestion) => {
    // A Nominatim-fallback suggestion already carries coordinates — only an AWS-sourced one (placeId) needs resolving.
    if (!s.placeId) {
      if (s.latitude == null || s.longitude == null) return;
      onResolved({ latitude: s.latitude, longitude: s.longitude, placeLabel: s.label, placeId: undefined });
      setQuery(s.label);
      setSuggestions([]);
      setOpen(false);
      return;
    }
    setResolving(true);
    setError('');
    try {
      const place = await api.locations.resolve(s.placeId);
      const label = place.displayName || s.label;
      onResolved({ latitude: place.latitude, longitude: place.longitude, placeLabel: label, placeId: s.placeId });
      setQuery(label);
      setSuggestions([]);
      setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not resolve that place.');
    } finally {
      setResolving(false);
    }
  };

  return (
    <label className="field location-search">
      <span>Place of birth</span>
      <input
        type="text"
        value={query}
        disabled={disabled || resolving}
        placeholder="Search city, town or place..."
        onChange={e => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
      />
      {open && (searching || suggestions.length > 0) && (
        <ul className="suggestions" role="listbox">
          {searching && <li className="suggestion-item muted">Searching…</li>}
          {!searching && suggestions.map(s => (
            <li key={s.placeId} className="suggestion-item" role="option" aria-selected={false}
              onMouseDown={() => void pick(s)}>
              {s.label}
            </li>
          ))}
        </ul>
      )}
      {resolving && <p className="hint">Resolving coordinates…</p>}
      {error && <p className="hint warn">{error}</p>}
    </label>
  );
}
