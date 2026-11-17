import { useEffect, useState } from 'react';
import { onWaking } from '../net';

/** Shown while the free host is starting up, so a slow first load does not look broken. */
export default function WakeBanner() {
  const [waking, setWaking] = useState(false);
  useEffect(() => onWaking(setWaking), []);
  if (!waking) return null;
  return (
    <div className="wake-banner" role="status" aria-live="polite">
      Waking up the server… The free host sleeps when idle, so the first load can take up to a minute. Nothing is lost; this page will continue by itself.
    </div>
  );
}
