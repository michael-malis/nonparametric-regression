// Derived, memoized quantities. Everything drawn on screen comes from here.

import { DOMAIN, PRESETS, valueScaleFor } from '../data/presets';
import type { ValueScale } from '../math/colormap';
import { type CellDiagram, computeCells } from '../geometry/cells';
import { type KnnPrediction, type NwPrediction, type Vec2, predictKNN, predictNW } from '../math/estimators';
import { type Profile, nwProfile, pointOnPath, stepProfile } from '../path/profile';
import type { Method, State } from './store';

export type AnyPrediction = (KnnPrediction & { kind: 'knn' }) | (NwPrediction & { kind: 'nw' });

function memo<A extends unknown[], R>(fn: (...a: A) => R, keyOf: (...a: A) => string): (...a: A) => R {
  let lastKey: string | null = null;
  let last: R;
  return (...a: A) => {
    const key = keyOf(...a);
    if (key !== lastKey) {
      last = fn(...a);
      lastKey = key;
    }
    return last;
  };
}

const dataScaleMemo = memo(
  (s: State): ValueScale => {
    const p = PRESETS[s.presetIndex].scale;
    const d = valueScaleFor(s.Y);
    // Y changes only through the one-response experiment; widen, never clip.
    return d.lo >= p.lo && d.hi <= p.hi ? p : { lo: Math.min(p.lo, d.lo), hi: Math.max(p.hi, d.hi) };
  },
  (s) => `${s.presetIndex}|${s.dataVersion}`,
);

/**
 * Colour domain shared by all methods and views: the dataset's, unless an
 * experiment has fixed it (so a colour change always means a value change).
 */
export function valueScale(s: State): ValueScale {
  return s.scaleOverride ?? dataScaleMemo(s);
}

/** k of the estimator currently shown (1 on the 1NN stage). */
export function effectiveK(s: State): number {
  return s.method === '1nn' ? 1 : Math.min(s.kKnn, s.X.length);
}

// Geometry depends on X and k. A response edit changes only plateau heights.
const layouts = new WeakMap<Vec2[], Map<number, CellDiagram>>();
let lastLayout: CellDiagram | undefined;
let lastResponses: number[] | undefined;
let lastDiagram: CellDiagram;
export function cells(s: State, k = effectiveK(s)): CellDiagram {
  let byK = layouts.get(s.X);
  if (!byK) { byK = new Map(); layouts.set(s.X, byK); }
  let layout = byK.get(k);
  if (!layout) { layout = computeCells(s,k,DOMAIN); byK.set(k,layout); }
  if (layout === lastLayout && s.Y === lastResponses) return lastDiagram;
  const heights = new Map(layout.cells.map(c=>[c.key,c.set.reduce((sum,i)=>sum+s.Y[i],0)/k]));
  lastDiagram = {...layout,cells:layout.cells.map(c=>({...c,value:heights.get(c.key)!,
    edges:c.edges.map(e=>({...e,neighborValue:e.neighborKey===null?null:heights.get(e.neighborKey)!}))}))};
  lastLayout = layout;
  lastResponses = s.Y;
  return lastDiagram;
}

export function predictAt(s: State, x: Vec2, method: Method = s.method): AnyPrediction {
  if (method === 'nw') return { ...predictNW(s, x, s.h), kind: 'nw' };
  return { ...predictKNN(s, x, method === '1nn' ? 1 : s.kKnn), kind: 'knn' };
}

const predictionMemo = memo(
  (s: State): AnyPrediction => predictAt(s, s.x0),
  (s) => `${s.dataVersion}|${s.method}|${s.kKnn}|${s.h}|${s.x0[0]}|${s.x0[1]}`,
);

export function prediction(s: State): AnyPrediction {
  return predictionMemo(s);
}

function profileFor(s: State, method: Method, k: number): Profile {
  return method === 'nw' ? nwProfile(s, s.h, s.pathA, s.pathB) : stepProfile(s, k, s.pathA, s.pathB);
}

const profileMemo = memo(
  (s: State): Profile => profileFor(s, s.method, effectiveK(s)),
  (s) => `${s.dataVersion}|${s.method}|${effectiveK(s)}|${s.h}|${s.pathA}|${s.pathB}`,
);

export function profile(s: State): Profile {
  return profileMemo(s);
}

/** Mark the next exact crossing only when the query is within 0.8 feature units. */
export function upcomingCrossing(s: State): Vec2 | null {
  if (s.experiment?.id !== 'segment' || s.method === 'nw') return null;
  const slice = profile(s);
  if (slice.kind !== 'step') return null;
  const t = slice.setChanges.find(t => t > s.s + 1e-9);
  const length = Math.hypot(s.pathB[0]-s.pathA[0],s.pathB[1]-s.pathA[1]);
  return t !== undefined && (t-s.s)*length < 0.8 ? pointOnPath(s.pathA,s.pathB,t) : null;
}

/** Comparison profile drawn faintly: 1NN under kNN, kNN (stage k) under NW. */
export interface GhostProfile {
  label: string;
  profile: Profile;
}

const ghostMemo = memo(
  (s: State): GhostProfile | null => {
    if (s.method === 'knn') return { label: '1NN', profile: stepProfile(s, 1, s.pathA, s.pathB) };
    if (s.method === 'nw') {
      const k = Math.min(s.kKnn, s.X.length);
      return { label: `kNN, k = ${k}`, profile: stepProfile(s, k, s.pathA, s.pathB) };
    }
    return null;
  },
  (s) => `${s.dataVersion}|${s.method}|${s.kKnn}|${s.pathA}|${s.pathB}`,
);

export function ghostProfile(s: State): GhostProfile | null {
  return ghostMemo(s);
}

/** Value of the current profile at s (exact: evaluates the estimator at γ(s)). */
export function profileValueAt(s: State, t: number): number {
  const x: Vec2 = [s.pathA[0] + t * (s.pathB[0] - s.pathA[0]), s.pathA[1] + t * (s.pathB[1] - s.pathA[1])];
  return predictAt(s, x).value;
}
