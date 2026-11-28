// Sequential, WhatsApp-style delivery of one reply's chat bubbles.
//
// The reply is generated at full speed; only its PRESENTATION is paced:
//   show bubble 1 now
//   -> wait for its entrance animation to finish
//   -> typing dots for a natural pause (delayFor)
//   -> show bubble 2, and so on; no dots after the last bubble.
// Pure and timer-injectable so it is tested with a fake clock
// (bubbleDelivery.test.ts). Ported verbatim from the production Kundli chat
// widget (web/frontend/src/sections/KundliSection/bubbleDelivery.ts) — same
// pacing, same accessibility behavior, so a tester sees exactly what a real
// site visitor sees.

export const BUBBLE_DELIVERY = {
  // Must match the CSS entrance animation; the widget passes it to CSS as
  // --bubble-in-ms so the two cannot drift apart.
  animationMs: 280,
  // Pause AFTER the previous bubble's animation ends. A fixed cycle rather
  // than random jitter: varied enough not to feel mechanical, same feel on
  // every reply. Mean 1.8 s, range 1.6–2.0 s.
  delaysMs: [1600, 1900, 1700, 2000, 1800] as const,
  // prefers-reduced-motion: no entrance animation and a shorter fixed pause,
  // so messages still arrive one by one but without movement or long waits.
  reducedMotionDelayMs: 900,
} as const;

export function delayFor(index: number, reducedMotion = false): number {
  if (reducedMotion) return BUBBLE_DELIVERY.reducedMotionDelayMs;
  const cycle = BUBBLE_DELIVERY.delaysMs;
  return cycle[((index % cycle.length) + cycle.length) % cycle.length];
}

export function entranceMs(reducedMotion = false): number {
  return reducedMotion ? 0 : BUBBLE_DELIVERY.animationMs;
}

export function prefersReducedMotion(): boolean {
  try {
    return typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  } catch {
    return false;
  }
}

export interface DeliveryCallbacks {
  // `isLast` lets the caller attach per-reply extras only to the final
  // bubble, so they appear once the whole reply is shown.
  onShow: (text: string, index: number, isLast: boolean) => void;
  onTyping: (visible: boolean) => void;
  onDone: () => void;
}

export interface Timers {
  setTimeout: (fn: () => void, ms: number) => unknown;
  clearTimeout: (id: unknown) => void;
}

export interface DeliveryHandle {
  // Stop silently: no callback of any kind fires afterwards (unmount,
  // conversation switch).
  cancel(): void;
  // Show every remaining bubble at once, hide the dots, finish. Used when a new
  // message is sent mid-sequence, so no content of the previous reply is lost.
  flush(): void;
  readonly finished: boolean;
}

const realTimers: Timers = {
  setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms),
  clearTimeout: id => globalThis.clearTimeout(id as ReturnType<typeof setTimeout>),
};

export function deliverBubbles(
  texts: readonly string[],
  callbacks: DeliveryCallbacks,
  options: { reducedMotion?: boolean; timers?: Timers } = {},
): DeliveryHandle {
  const timers = options.timers ?? realTimers;
  const reduced = !!options.reducedMotion;
  let next = 0;
  let finished = false;
  let typing = false;
  let pending: unknown = null;

  const clear = () => {
    if (pending !== null) timers.clearTimeout(pending);
    pending = null;
  };
  const setTyping = (visible: boolean) => {
    if (typing !== visible) {
      typing = visible;
      callbacks.onTyping(visible);
    }
  };
  const finish = () => {
    finished = true;
    clear();
    setTyping(false);
    callbacks.onDone();
  };
  const showNext = () => {
    const index = next++;
    setTyping(false);
    callbacks.onShow(texts[index]!, index, index === texts.length - 1);
    if (next >= texts.length) {
      finish();
      return;
    }
    // The pause is measured from the END of this bubble's entrance animation.
    pending = timers.setTimeout(() => {
      setTyping(true);
      pending = timers.setTimeout(() => {
        pending = null;
        showNext();
      }, delayFor(index, reduced));
    }, entranceMs(reduced));
  };

  if (texts.length === 0) finish();
  else showNext();

  return {
    cancel() {
      if (finished) return;
      finished = true;
      clear();
    },
    flush() {
      if (finished) return;
      clear();
      setTyping(false);
      while (next < texts.length) {
        const index = next++;
        callbacks.onShow(texts[index]!, index, index === texts.length - 1);
      }
      finish();
    },
    get finished() {
      return finished;
    },
  };
}
