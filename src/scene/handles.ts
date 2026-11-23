// Drag semantics shared by the 2-D and 3-D views.

import { DOMAIN } from '../data/presets';
import type { Vec2 } from '../math/estimators';
import { pointOnPath } from '../path/profile';
import type { State, Store } from '../state/store';

export type HandleKind = 'x0' | 'point' | 'pathA' | 'pathB';

export interface DragTarget {
  kind: HandleKind;
  index?: number;
}

export function clampToDomain(x: Vec2): Vec2 {
  return [Math.min(DOMAIN.x1, Math.max(DOMAIN.x0, x[0])), Math.min(DOMAIN.y1, Math.max(DOMAIN.y0, x[1]))];
}

/** Feature-space position of a handle. */
export function handlePosition(s: State, t: DragTarget): Vec2 {
  switch (t.kind) {
    case 'x0':
      return s.x0;
    case 'point':
      return s.X[t.index!];
    case 'pathA':
      return s.pathA;
    case 'pathB':
      return s.pathB;
  }
}

/**
 * Start a drag. Any manual drag ends a running experiment (the runner and the
 * hand never write x0 at the same time); grabbing x0 also pauses autoplay.
 */
export function beginDrag(store: Store, t: DragTarget): void {
  const patch: Partial<State> = { drag: { kind: t.kind, index: t.index }, experiment: null };
  if (t.kind === 'x0') {
    patch.playing = false;
    patch.onPath = false;
  }
  store.set(patch);
}

export function applyDrag(store: Store, t: DragTarget, pos: Vec2): void {
  const p = clampToDomain(pos);
  const s = store.get();
  switch (t.kind) {
    case 'x0':
      store.set({ x0: p, onPath: false, playing: false });
      break;
    case 'point': {
      // Only the input moves; the response Y_i is unchanged.
      const X = s.X.map((x, i) => (i === t.index ? p : x));
      store.set({ X });
      break;
    }
    case 'pathA':
    case 'pathB': {
      const A = t.kind === 'pathA' ? p : s.pathA;
      const B = t.kind === 'pathB' ? p : s.pathB;
      const patch: Partial<State> = { pathA: A, pathB: B };
      if (s.onPath) patch.x0 = pointOnPath(A, B, s.s);
      store.set(patch);
      break;
    }
  }
}

export function endDrag(store: Store): void {
  if (store.get().drag) store.set({ drag: null });
}
