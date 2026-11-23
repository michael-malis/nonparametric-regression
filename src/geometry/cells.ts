// Exact order-k Voronoi cells, clipped to the rectangular display domain.
//
// For k = 1 these are the ordinary Voronoi cells. For k > 1 the region where
// the neighbour set equals S is the convex polygon
//     { x : |x - X_s| <= |x - X_t|  for all s in S, t not in S },
// an intersection of k(n-k) half-planes bounded by perpendicular bisectors.
// The cells tile the domain, so a breadth-first walk across cell edges, starting
// from one cell, reaches every non-empty cell. Each cell is clipped exactly; the
// only approximation is floating-point arithmetic.

import { type Dataset, type Vec2, clampK, orderByDistance } from '../math/estimators';

export interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface CellEdge {
  p: Vec2;
  q: Vec2;
  /** True when the edge lies on the domain boundary. */
  boundary: boolean;
  /** Prediction value of the cell across this edge (null on the boundary). */
  neighborValue: number | null;
  neighborKey: string | null;
}

export interface Cell {
  /** Sorted neighbour indices joined by ',' (identifies N_k on this cell). */
  key: string;
  set: number[];
  value: number;
  /** Counter-clockwise convex polygon. */
  polygon: Vec2[];
  edges: CellEdge[];
  area: number;
}

export interface CellDiagram {
  k: number;
  cells: Cell[];
  /** Sum of cell areas divided by the domain area; 1 when the tiling is complete. */
  coverage: number;
}

interface Poly {
  pts: Vec2[];
  /** labels[i] tags the edge pts[i] -> pts[i+1]: < 0 boundary, >= 0 bisector. */
  labels: number[];
}

/** Clip a convex polygon against a*x + b*y <= c. New edges get `label`. */
function clipHalfPlane(poly: Poly, a: number, b: number, c: number, label: number, tol: number): Poly {
  const { pts, labels } = poly;
  const m = pts.length;
  const out: Vec2[] = [];
  const outLabels: number[] = [];
  for (let i = 0; i < m; i++) {
    const P = pts[i];
    const Q = pts[(i + 1) % m];
    const fP = a * P[0] + b * P[1] - c;
    const fQ = a * Q[0] + b * Q[1] - c;
    const inP = fP <= tol;
    const inQ = fQ <= tol;
    if (inP) {
      out.push(P);
      if (inQ) {
        outLabels.push(labels[i]);
      } else {
        outLabels.push(labels[i]);
        const t = fP / (fP - fQ);
        out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]);
        outLabels.push(label);
      }
    } else if (inQ) {
      const t = fP / (fP - fQ);
      out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]);
      outLabels.push(labels[i]);
    }
  }
  return dedupe({ pts: out, labels: outLabels }, tol);
}

function dedupe(poly: Poly, tol: number): Poly {
  const pts: Vec2[] = [];
  const labels: number[] = [];
  const m = poly.pts.length;
  for (let i = 0; i < m; i++) {
    const P = poly.pts[i];
    const Q = poly.pts[(i + 1) % m];
    if (m > 1 && Math.abs(P[0] - Q[0]) <= tol && Math.abs(P[1] - Q[1]) <= tol) continue;
    pts.push(P);
    labels.push(poly.labels[i]);
  }
  return { pts, labels };
}

export function polygonArea(pts: Vec2[]): number {
  let s = 0;
  for (let i = 0; i < pts.length; i++) {
    const P = pts[i];
    const Q = pts[(i + 1) % pts.length];
    s += P[0] * Q[1] - Q[0] * P[1];
  }
  return s / 2;
}

export function setKey(set: number[]): string {
  return [...set].sort((a, b) => a - b).join(',');
}

function neighborSet(data: Dataset, x: Vec2, k: number): number[] {
  return orderByDistance(data.X, x).order.slice(0, k);
}

function meanOf(Y: number[], set: number[]): number {
  let s = 0;
  for (const i of set) s += Y[i];
  return s / set.length;
}

/** Polygon of the cell where the k-nearest set is exactly `set`. */
function cellPolygon(data: Dataset, set: number[], rect: Rect, tol: number): Poly {
  const n = data.X.length;
  const inSet = new Uint8Array(n);
  for (const i of set) inSet[i] = 1;
  let poly: Poly = {
    pts: [
      [rect.x0, rect.y0],
      [rect.x1, rect.y0],
      [rect.x1, rect.y1],
      [rect.x0, rect.y1],
    ],
    labels: [-1, -2, -3, -4],
  };
  for (const s of set) {
    const Xs = data.X[s];
    for (let t = 0; t < n; t++) {
      if (inSet[t]) continue;
      const Xt = data.X[t];
      // |x-Xs|^2 <= |x-Xt|^2  <=>  2 x.(Xt-Xs) <= |Xt|^2 - |Xs|^2
      const a = 2 * (Xt[0] - Xs[0]);
      const b = 2 * (Xt[1] - Xs[1]);
      const c = Xt[0] * Xt[0] + Xt[1] * Xt[1] - (Xs[0] * Xs[0] + Xs[1] * Xs[1]);
      if (Math.abs(a) + Math.abs(b) < 1e-14) continue; // coincident points: tie rule decides, no edge
      poly = clipHalfPlane(poly, a, b, c, s * n + t, tol);
      if (poly.pts.length < 3) return poly;
    }
  }
  return poly;
}

/**
 * Compute every non-empty cell of the order-k Voronoi diagram on `rect`.
 * Neighbouring cells are found by probing just outside the midpoint of each
 * edge with the same kNN rule (and tie-break) the estimator uses.
 */
export function computeCells(data: Dataset, kIn: number, rect: Rect): CellDiagram {
  const n = data.X.length;
  const k = clampK(kIn, n);
  const size = Math.max(rect.x1 - rect.x0, rect.y1 - rect.y0);
  const tol = 1e-12 * size;
  const probe = 1e-7 * size;
  const domainArea = (rect.x1 - rect.x0) * (rect.y1 - rect.y0);

  const seen = new Set<string>();
  const cells: Cell[] = [];
  const queue: number[][] = [];

  const enqueue = (set: number[]) => {
    const key = setKey(set);
    if (seen.has(key)) return;
    seen.add(key);
    queue.push(set);
  };

  const inside = (x: Vec2) => x[0] > rect.x0 && x[0] < rect.x1 && x[1] > rect.y0 && x[1] < rect.y1;

  const walk = () => {
    while (queue.length) {
      const set = queue.shift()!;
      const key = setKey(set);
      const poly = cellPolygon(data, set, rect, tol);
      if (poly.pts.length < 3) continue;
      const area = polygonArea(poly.pts);
      if (area <= tol * size) continue;
      const value = meanOf(data.Y, set);
      const edges: CellEdge[] = [];
      const m = poly.pts.length;
      for (let i = 0; i < m; i++) {
        const p = poly.pts[i];
        const q = poly.pts[(i + 1) % m];
        const dx = q[0] - p[0];
        const dy = q[1] - p[1];
        const len = Math.hypot(dx, dy);
        if (len <= tol) continue;
        const mid: Vec2 = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
        // Outward normal of a CCW polygon edge.
        const probePt: Vec2 = [mid[0] + (dy / len) * probe, mid[1] - (dx / len) * probe];
        if (poly.labels[i] < 0 || !inside(probePt)) {
          edges.push({ p, q, boundary: true, neighborValue: null, neighborKey: null });
          continue;
        }
        const nb = neighborSet(data, probePt, k);
        const nbKey = setKey(nb);
        edges.push({ p, q, boundary: false, neighborValue: meanOf(data.Y, nb), neighborKey: nbKey });
        enqueue(nb);
      }
      cells.push({ key, set: [...set].sort((a, b) => a - b), value, polygon: poly.pts, edges, area });
    }
  };

  enqueue(neighborSet(data, [(rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2], k));
  walk();

  let covered = cells.reduce((s, c) => s + c.area, 0);
  if (Math.abs(covered - domainArea) > 1e-9 * domainArea) {
    // Degenerate configurations (e.g. collinear bisectors) can hide a neighbour
    // behind a vertex. Seed the walk from a grid scan until the tiling closes.
    const G = 96;
    for (let gy = 0; gy < G; gy++) {
      for (let gx = 0; gx < G; gx++) {
        const x: Vec2 = [
          rect.x0 + ((gx + 0.5) / G) * (rect.x1 - rect.x0),
          rect.y0 + ((gy + 0.5) / G) * (rect.y1 - rect.y0),
        ];
        enqueue(neighborSet(data, x, k));
      }
    }
    walk();
    covered = cells.reduce((s, c) => s + c.area, 0);
  }

  return { k, cells, coverage: covered / domainArea };
}

/** Index of the cell containing x (by the same kNN rule), or -1. */
export function findCell(diagram: CellDiagram, data: Dataset, x: Vec2): number {
  const key = setKey(neighborSet(data, x, diagram.k));
  return diagram.cells.findIndex((c) => c.key === key);
}

export function pointInConvex(poly: Vec2[], x: Vec2, tol = 1e-9): boolean {
  for (let i = 0; i < poly.length; i++) {
    const P = poly[i];
    const Q = poly[(i + 1) % poly.length];
    const cross = (Q[0] - P[0]) * (x[1] - P[1]) - (Q[1] - P[1]) * (x[0] - P[0]);
    if (cross < -tol) return false;
  }
  return true;
}
