// Cross-sections g(s) = f̂(γ(s)) of the 2-D estimators along a straight path
// γ(s) = (1-s)A + sB, s in [0, 1]. This is a slice of the 2-D surface, not a
// separate 1-D regression.

import { type Dataset, type Vec2, clampK, nwValue, orderByDistance, pack, sanitizeH } from '../math/estimators';

export function pointOnPath(A: Vec2, B: Vec2, s: number): Vec2 {
  return [A[0] + s * (B[0] - A[0]), A[1] + s * (B[1] - A[1])];
}

export function pathLength(A: Vec2, B: Vec2): number {
  return Math.hypot(B[0] - A[0], B[1] - A[1]);
}

/** Parameter s of the orthogonal projection of x onto the line AB, clamped to [0,1]. */
export function projectOnPath(A: Vec2, B: Vec2, x: Vec2): { s: number; dist: number } {
  const dx = B[0] - A[0];
  const dy = B[1] - A[1];
  const L2 = dx * dx + dy * dy;
  const s = L2 > 0 ? Math.min(1, Math.max(0, ((x[0] - A[0]) * dx + (x[1] - A[1]) * dy) / L2)) : 0;
  const p = pointOnPath(A, B, s);
  return { s, dist: Math.hypot(x[0] - p[0], x[1] - p[1]) };
}

export interface StepSegment {
  s0: number;
  s1: number;
  value: number;
}

export interface StepProfile {
  kind: 'step';
  k: number;
  segments: StepSegment[];
  /** Parameters where the neighbour set changes (whether or not the value jumps). */
  setChanges: number[];
}

/**
 * Exact piecewise-constant profile of kNN along a straight path.
 * Along γ(s), |γ(s)-X_i|^2 - |γ(s)-X_j|^2 is linear in s, so each pair (i, j)
 * swaps order at most once. Between consecutive swap parameters the full
 * distance order, hence N_k, is constant.
 */
export function stepProfile(data: Dataset, kIn: number, A: Vec2, B: Vec2): StepProfile {
  const n = data.X.length;
  const k = clampK(kIn, n);
  const d: Vec2 = [B[0] - A[0], B[1] - A[1]];
  const cuts: number[] = [];
  for (let i = 0; i < n; i++) {
    const Xi = data.X[i];
    for (let j = i + 1; j < n; j++) {
      const Xj = data.X[j];
      const ex = Xj[0] - Xi[0];
      const ey = Xj[1] - Xi[1];
      const den = 2 * (d[0] * ex + d[1] * ey);
      if (Math.abs(den) < 1e-14) continue;
      const num = Xj[0] * Xj[0] + Xj[1] * Xj[1] - (Xi[0] * Xi[0] + Xi[1] * Xi[1]) - 2 * (A[0] * ex + A[1] * ey);
      const s = num / den;
      if (s > 0 && s < 1) cuts.push(s);
    }
  }
  cuts.sort((a, b) => a - b);
  const bounds = [0];
  for (const s of cuts) if (s - bounds[bounds.length - 1] > 1e-12) bounds.push(s);
  if (1 - bounds[bounds.length - 1] > 1e-12) bounds.push(1);
  else bounds[bounds.length - 1] = 1;

  const segments: StepSegment[] = [];
  const setChanges: number[] = [];
  let prevKey = '';
  for (let m = 0; m + 1 < bounds.length; m++) {
    const s0 = bounds[m];
    const s1 = bounds[m + 1];
    const mid = pointOnPath(A, B, (s0 + s1) / 2);
    const set = orderByDistance(data.X, mid).order.slice(0, k);
    const key = [...set].sort((a, b) => a - b).join(',');
    let sum = 0;
    for (const i of set) sum += data.Y[i];
    const value = sum / k;
    if (m > 0 && key !== prevKey) setChanges.push(s0);
    prevKey = key;
    const last = segments[segments.length - 1];
    if (last && Math.abs(last.value - value) <= 1e-12 * Math.max(1, Math.abs(value))) {
      last.s1 = s1;
    } else {
      segments.push({ s0, s1, value });
    }
  }
  return { kind: 'step', k, segments, setChanges };
}

export interface SmoothProfile {
  kind: 'smooth';
  h: number;
  s: Float64Array;
  v: Float64Array;
}

/**
 * Gaussian NW profile g(s). Starts from a uniform grid (at least 8 samples per
 * bandwidth) and bisects every interval whose two end values differ by more
 * than `tolValue`, so steep but continuous transitions at small h are resolved
 * instead of being drawn as long straight chords.
 */
export function nwProfile(data: Dataset, hIn: number, A: Vec2, B: Vec2, tolValue = 0.02): SmoothProfile {
  const h = sanitizeH(hIn);
  const L = pathLength(A, B);
  const N0 = Math.min(4000, Math.max(400, Math.ceil((8 * L) / h)));
  const p = pack(data);
  const scratch = new Float64Array(p.n);
  const g = (t: number) => nwValue(p, A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]), h, scratch);
  const S: number[] = [];
  const V: number[] = [];
  const MAX = 40000;
  const minDs = 1e-6;
  let t0 = 0;
  let v0 = g(0);
  S.push(t0);
  V.push(v0);
  for (let i = 1; i <= N0; i++) {
    const t1 = i / N0;
    const v1 = g(t1);
    // Depth-first refinement of [t0, t1].
    const stack: [number, number, number, number][] = [[t0, v0, t1, v1]];
    while (stack.length) {
      const [a, va, b, vb] = stack.pop()!;
      if (Math.abs(vb - va) > tolValue && b - a > minDs && S.length < MAX) {
        const m = (a + b) / 2;
        const vm = g(m);
        stack.push([m, vm, b, vb]); // processed after the left half
        stack.push([a, va, m, vm]);
      } else {
        S.push(b);
        V.push(vb);
      }
    }
    t0 = t1;
    v0 = v1;
  }
  return { kind: 'smooth', h, s: Float64Array.from(S), v: Float64Array.from(V) };
}

export type Profile = StepProfile | SmoothProfile;

export function profileRange(p: Profile): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  if (p.kind === 'step') {
    for (const seg of p.segments) {
      lo = Math.min(lo, seg.value);
      hi = Math.max(hi, seg.value);
    }
  } else {
    for (const x of p.v) {
      lo = Math.min(lo, x);
      hi = Math.max(hi, x);
    }
  }
  return [lo, hi];
}
