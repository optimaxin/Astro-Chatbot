// Sequential bubble delivery against a fake clock: one bubble at a time, the
// pause measured from the end of the previous bubble's entrance animation,
// typing dots only between bubbles, and clean cancel/flush. Mirrors the
// production widget's own test (web/frontend/.../bubbleDelivery.test.ts),
// rewritten for vitest.
import { describe, expect, it } from 'vitest';
import { BUBBLE_DELIVERY, delayFor, deliverBubbles, entranceMs, type Timers } from './bubbleDelivery';

class FakeClock implements Timers {
  now = 0;
  private seq = 0;
  private queue = new Map<number, { at: number; fn: () => void }>();
  setTimeout = (fn: () => void, ms: number) => {
    const id = ++this.seq;
    this.queue.set(id, { at: this.now + ms, fn });
    return id;
  };
  clearTimeout = (id: unknown) => {
    this.queue.delete(id as number);
  };
  get pending() {
    return this.queue.size;
  }
  // Run every timer due up to `now + ms`, in time order.
  advance(ms: number) {
    const end = this.now + ms;
    for (;;) {
      const due = [...this.queue.entries()].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
      if (!due) break;
      this.queue.delete(due[0]);
      this.now = due[1].at;
      due[1].fn();
    }
    this.now = end;
  }
  runAll() {
    this.advance(60_000);
  }
}

interface Event { t: number; kind: 'show' | 'typing-on' | 'typing-off' | 'done'; index?: number; last?: boolean }

function run(texts: string[], reducedMotion = false) {
  const clock = new FakeClock();
  const events: Event[] = [];
  const handle = deliverBubbles(texts, {
    onShow: (_text, index, last) => events.push({ t: clock.now, kind: 'show', index, last }),
    onTyping: on => events.push({ t: clock.now, kind: on ? 'typing-on' : 'typing-off' }),
    onDone: () => events.push({ t: clock.now, kind: 'done' }),
  }, { reducedMotion, timers: clock });
  return { clock, events, handle, shows: () => events.filter(e => e.kind === 'show') };
}

const texts = (n: number) => Array.from({ length: n }, (_, i) => `Bubble ${i + 1}.`);

describe('deliverBubbles', () => {
  it('shows a single bubble immediately, with no timer and no typing dots', () => {
    const { clock, events, handle } = run(texts(1));
    expect(events).toEqual([{ t: 0, kind: 'show', index: 0, last: true }, { t: 0, kind: 'done' }]);
    expect(clock.pending).toBe(0);
    expect(handle.finished).toBe(true);
  });

  it.each([2, 3, 4, 5])('delivers %i bubbles one at a time, never two at the same instant', (n) => {
    const { clock, events, shows } = run(texts(n));
    expect(shows()).toHaveLength(1);
    expect(shows()[0]!.t).toBe(0);
    clock.runAll();
    const s = shows();
    expect(s.map(e => e.index)).toEqual([...Array(n).keys()]);
    expect(new Set(s.map(e => e.t)).size).toBe(n);
    expect(s.map(e => e.last)).toEqual(s.map((_, i) => i === n - 1));
    // Each gap = previous bubble's entrance animation + its pause.
    for (let i = 1; i < n; i++) {
      expect(s[i]!.t - s[i - 1]!.t).toBe(BUBBLE_DELIVERY.animationMs + delayFor(i - 1));
      const on = events.filter(e => e.kind === 'typing-on' && e.t > s[i - 1]!.t && e.t < s[i]!.t);
      expect(on).toHaveLength(1);
      expect(on[0]!.t - s[i - 1]!.t).toBe(BUBBLE_DELIVERY.animationMs);
    }
    // The indicator ends hidden and nothing remains scheduled.
    const typing = events.filter(e => e.kind.startsWith('typing'));
    expect(typing.at(-1)?.kind).toBe('typing-off');
    expect(events.at(-1)?.kind).toBe('done');
    expect(clock.pending).toBe(0);
  });

  it('paces with a varied but bounded cycle, averaging ~1.8s, not random', () => {
    const pauses = [0, 1, 2, 3, 4].map(i => delayFor(i));
    expect(new Set(pauses).size).toBeGreaterThan(1);
    expect(pauses.every(p => p >= 1500 && p <= 2000)).toBe(true);
    const mean = pauses.reduce((a, b) => a + b, 0) / pauses.length;
    expect(Math.abs(mean - 1800)).toBeLessThanOrEqual(50);
    expect(delayFor(5)).toBe(delayFor(0));
  });

  it('cancel(): no callback of any kind afterwards, no timer left', () => {
    const { clock, events, handle } = run(texts(4));
    clock.advance(BUBBLE_DELIVERY.animationMs + 100); // dots are showing
    const before = events.length;
    handle.cancel();
    clock.runAll();
    expect(events).toHaveLength(before);
    expect(clock.pending).toBe(0);
    expect(handle.finished).toBe(true);
  });

  it('flush(): every remaining bubble exactly once, dots hidden, done, no timer left; a second flush/cancel is a no-op', () => {
    const { clock, events, handle, shows } = run(texts(5));
    clock.advance(BUBBLE_DELIVERY.animationMs + delayFor(0) + 10); // two bubbles shown
    expect(shows()).toHaveLength(2);
    handle.flush();
    expect(shows().map(e => e.index)).toEqual([0, 1, 2, 3, 4]);
    expect(events.filter(e => e.kind === 'done')).toHaveLength(1);
    expect(events.filter(e => e.kind.startsWith('typing')).at(-1)?.kind).toBe('typing-off');
    handle.flush();
    handle.cancel();
    clock.runAll();
    expect(shows()).toHaveLength(5); // no duplicates after a second flush
    expect(clock.pending).toBe(0);
  });

  it('flushes an unfinished reply then starts a new one with no overlapping timers', () => {
    const clock = new FakeClock();
    const shown: string[] = [];
    const cb = (tag: string) => ({ onShow: (t: string) => shown.push(`${tag}:${t}`), onTyping: () => {}, onDone: () => {} });
    const first = deliverBubbles(['a1.', 'a2.', 'a3.'], cb('A'), { timers: clock });
    clock.advance(500);
    first.flush();
    deliverBubbles(['b1.', 'b2.'], cb('B'), { timers: clock });
    clock.runAll();
    expect(shown).toEqual(['A:a1.', 'A:a2.', 'A:a3.', 'B:b1.', 'B:b2.']);
  });

  it('respects reduced motion: no entrance wait, a short fixed pause, still sequential', () => {
    expect(entranceMs(true)).toBe(0);
    const { clock, shows } = run(texts(3), true);
    clock.runAll();
    expect(shows().map(e => e.t)).toEqual([0, BUBBLE_DELIVERY.reducedMotionDelayMs, 2 * BUBBLE_DELIVERY.reducedMotionDelayMs]);
  });

  it('delivers Hindi, Hinglish and English text verbatim', () => {
    const replies = ['आपकी कुंडली में शुक्र सातवें भाव में है।', 'Aapke career mein growth ke chances hain.',
      'Your 10th lord sits in the 6th house.'];
    const got: string[] = [];
    const clock = new FakeClock();
    deliverBubbles(replies, { onShow: t => got.push(t), onTyping: () => {}, onDone: () => {} }, { timers: clock });
    clock.runAll();
    expect(got).toEqual(replies);
  });

  it('finishes an empty reply at once', () => {
    const { events, clock } = run([]);
    expect(events).toEqual([{ t: 0, kind: 'done' }]);
    expect(clock.pending).toBe(0);
  });
});
