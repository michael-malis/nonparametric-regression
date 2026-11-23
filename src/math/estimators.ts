// Pure estimators for regression on two features. No rendering dependencies.
//
// All three estimators are linear smoothers  f̂(x) = Σ w_i(x) Y_i  with
// w_i(x) >= 0 and Σ w_i(x) = 1. The weights depend on x, the design X and
// the tuning parameter (k or h), never on the responses Y.

export type Vec2 = [number, number];

export interface Dataset {
  X: Vec2[];
  Y: number[];
}

/** Squared Euclidean distance. */
export function sqDist(a: Vec2, b: Vec2): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
}

/** Relative tolerance under which two squared distances count as a tie. */
const TIE_RTOL = 1e-12;

/**
 * Compare (distance, index) pairs. Ties in distance are broken by the smaller
 * index first. This rule never looks at Y, as the lecture notes require.
 */
export function compareByDistance(d: ArrayLike<number>, i: number, j: number): number {
  const di = d[i];
  const dj = d[j];
  const tol = TIE_RTOL * Math.max(1, di, dj);
  if (di < dj - tol) return -1;
  if (di > dj + tol) return 1;
  return i - j;
}

/** Indices of all design points sorted by distance to x (ties: smaller index first). */
export function orderByDistance(X: Vec2[], x: Vec2): { order: number[]; d2: Float64Array } {
  const n = X.length;
  const d2 = new Float64Array(n);
  for (let i = 0; i < n; i++) d2[i] = sqDist(X[i], x);
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((i, j) => compareByDistance(d2, i, j));
  return { order, d2 };
}

export function clampK(k: number, n: number): number {
  if (!Number.isFinite(k)) return 1;
  return Math.min(Math.max(1, Math.round(k)), Math.max(1, n));
}

export interface Prediction {
  value: number;
  /** Normalized weights, one per observation; they sum to one. */
  weights: Float64Array;
  /** Squared distances from x to every X_i. */
  d2: Float64Array;
  /** Observation indices sorted by distance to x. */
  order: number[];
}

export interface KnnPrediction extends Prediction {
  k: number;
  /** N_k(x), in order of increasing distance. */
  neighbors: number[];
  /** R_k(x): distance to the k-th nearest design point. */
  radius: number;
}

export interface NwPrediction extends Prediction {
  h: number;
}

/** k-nearest-neighbour average. k = 1 gives the 1NN (Voronoi) estimate. */
export function predictKNN(data: Dataset, x: Vec2, kIn: number): KnnPrediction {
  const n = data.X.length;
  if (n === 0) throw new Error('predictKNN: empty dataset');
  const k = clampK(kIn, n);
  const { order, d2 } = orderByDistance(data.X, x);
  const neighbors = order.slice(0, k);
  const weights = new Float64Array(n);
  let sum = 0;
  for (const i of neighbors) {
    weights[i] = 1 / k;
    sum += data.Y[i];
  }
  return {
    value: sum / k,
    weights,
    d2,
    order,
    k,
    neighbors,
    radius: Math.sqrt(d2[neighbors[k - 1]]),
  };
}

export function predict1NN(data: Dataset, x: Vec2): KnnPrediction {
  return predictKNN(data, x, 1);
}

/** Smallest bandwidth the engine accepts; guards against h <= 0 and NaN. */
export const H_MIN = 1e-6;

export function sanitizeH(h: number): number {
  if (!Number.isFinite(h)) return h === Infinity ? 1e12 : 1;
  return Math.max(H_MIN, h);
}

/**
 * Gaussian Nadaraya–Watson weights, computed stably: the smallest squared
 * distance is subtracted before exponentiation, so the largest raw weight is
 * exactly 1 and the normalizing sum is always >= 1 (never 0/0).
 */
export function nwWeights(X: Vec2[], x: Vec2, hIn: number, d2In?: Float64Array): Float64Array {
  const n = X.length;
  const h = sanitizeH(hIn);
  const d2 = d2In ?? Float64Array.from(X, (p) => sqDist(p, x));
  let m = Infinity;
  for (let i = 0; i < n; i++) if (d2[i] < m) m = d2[i];
  const inv = 1 / (2 * h * h);
  const w = new Float64Array(n);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const a = Math.exp(-(d2[i] - m) * inv);
    w[i] = a;
    s += a;
  }
  for (let i = 0; i < n; i++) w[i] /= s;
  return w;
}

export function predictNW(data: Dataset, x: Vec2, hIn: number): NwPrediction {
  const n = data.X.length;
  if (n === 0) throw new Error('predictNW: empty dataset');
  const h = sanitizeH(hIn);
  const { order, d2 } = orderByDistance(data.X, x);
  const weights = nwWeights(data.X, x, h, d2);
  let v = 0;
  for (let i = 0; i < n; i++) v += weights[i] * data.Y[i];
  return { value: v, weights, d2, order, h };
}

// ---------------------------------------------------------------------------
// Fast scalar evaluators for dense grids (no allocation per call).

export interface PackedData {
  n: number;
  xs: Float64Array;
  ys: Float64Array;
  Y: Float64Array;
}

export function pack(data: Dataset): PackedData {
  const n = data.X.length;
  return {
    n,
    xs: Float64Array.from(data.X, (p) => p[0]),
    ys: Float64Array.from(data.X, (p) => p[1]),
    Y: Float64Array.from(data.Y),
  };
}

/** Gaussian NW value at (x1, x2); same stable formula as predictNW. */
export function nwValue(p: PackedData, x1: number, x2: number, h: number, scratch: Float64Array): number {
  const { n, xs, ys, Y } = p;
  let m = Infinity;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - x1;
    const dy = ys[i] - x2;
    const d = dx * dx + dy * dy;
    scratch[i] = d;
    if (d < m) m = d;
  }
  const inv = 1 / (2 * h * h);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const a = Math.exp(-(scratch[i] - m) * inv);
    num += a * Y[i];
    den += a;
  }
  return num / den;
}

export function mean(values: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < values.length; i++) s += values[i];
  return s / values.length;
}

/** Gaussian products factor along the two axes of a rectangular sampling grid. */
export function nwGrid(p: PackedData, xs: Float64Array, ys: Float64Array, h: number): Float64Array {
  const inv = 1 / (2 * h * h);
  function factors(axis: Float64Array, sites: Float64Array) {
    const weights = new Float64Array(axis.length * p.n);
    for (let j = 0; j < axis.length; j++) {
      for (let k = 0; k < p.n; k++) {
        const delta = axis[j] - sites[k];
        weights[j * p.n + k] = Math.exp(-delta * delta * inv);
      }
    }
    return weights;
  }
  const wx = factors(xs, p.xs), wy = factors(ys, p.ys);
  const grid = new Float64Array(xs.length * ys.length);
  const scratch = new Float64Array(p.n);
  for (let j = 0; j < ys.length; j++) {
    for (let i = 0; i < xs.length; i++) {
      let sum = 0, total = 0;
      for (let k = 0; k < p.n; k++) {
        const weight = wx[i*p.n+k] * wy[j*p.n+k];
        sum += weight;
        total += weight * p.Y[k];
      }
      grid[j*xs.length+i] = sum > 1e-280 ? total / sum : nwValue(p, xs[i], ys[j], h, scratch);
    }
  }
  return grid;
}
