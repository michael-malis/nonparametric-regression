import { Delaunay } from 'd3-delaunay';
import { describe, expect, it } from 'vitest';
import { DOMAIN, PRESETS } from '../../src/data/presets';
import { computeCells, pointInConvex } from '../../src/geometry/cells';
import { type Dataset, type Vec2, predictKNN } from '../../src/math/estimators';

function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

describe('V10 / V12 exact order-k cells', () => {
  for (const preset of PRESETS) {
    for (const k of [1, 2, 3, 5, preset.X.length - 1, preset.X.length]) {
      it(`${preset.id}, k=${k}: cells tile the domain and match the estimator`, () => {
        const diag = computeCells(preset, k, DOMAIN);
        expect(diag.coverage).toBeCloseTo(1, 9);
        const r = rng(k * 101 + preset.X.length);
        for (let t = 0; t < 400; t++) {
          const x: Vec2 = [10 * r(), 10 * r()];
          const p = predictKNN(preset, x, k);
          const key = [...p.neighbors].sort((a, b) => a - b).join(',');
          const cell = diag.cells.find((c) => c.key === key);
          expect(cell, `no cell for set ${key}`).toBeDefined();
          expect(pointInConvex(cell!.polygon, x, 1e-7)).toBe(true);
          expect(cell!.value).toBeCloseTo(p.value, 13);
        }
        if (k === preset.X.length) expect(diag.cells.length).toBe(1);
      });
    }
  }

  it('k=1 cells agree with d3-delaunay Voronoi areas (independent implementation)', () => {
    for (const preset of PRESETS) {
      const vor = new Delaunay(Float64Array.from(preset.X.flat())).voronoi([DOMAIN.x0, DOMAIN.y0, DOMAIN.x1, DOMAIN.y1]);
      const diag = computeCells(preset, 1, DOMAIN);
      expect(diag.cells.length).toBe(preset.X.length);
      for (const cell of diag.cells) {
        const poly = vor.cellPolygon(cell.set[0]) as unknown as [number, number][];
        let a = 0;
        for (let j = 0; j + 1 < poly.length; j++) a += poly[j][0] * poly[j + 1][1] - poly[j + 1][0] * poly[j][1];
        expect(Math.abs(a / 2)).toBeCloseTo(cell.area, 9);
      }
    }
  });

  it('k>1 cells are not the 1NN cells (V12)', () => {
    const p = PRESETS[0];
    expect(computeCells(p, 3, DOMAIN).cells.length).not.toBe(computeCells(p, 1, DOMAIN).cells.length);
  });

  it('edge neighbour values are the values across the edge', () => {
    const p = PRESETS[0];
    const diag = computeCells(p, 3, DOMAIN);
    for (const c of diag.cells)
      for (const e of c.edges) {
        if (e.boundary) continue;
        const mid: Vec2 = [(e.p[0] + e.q[0]) / 2, (e.p[1] + e.q[1]) / 2];
        const dx = e.q[0] - e.p[0];
        const dy = e.q[1] - e.p[1];
        const L = Math.hypot(dx, dy);
        const out: Vec2 = [mid[0] + (dy / L) * 1e-6, mid[1] - (dx / L) * 1e-6];
        expect(e.neighborValue).toBeCloseTo(predictKNN(p, out, 3).value, 12);
      }
  });

  it('symmetric (degenerate) design still tiles the domain', () => {
    const d: Dataset = {
      X: [[2, 2], [8, 2], [2, 8], [8, 8], [5, 5], [5, 2], [5, 8], [2, 5], [8, 5]],
      Y: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    };
    for (const k of [1, 2, 3, 4]) expect(computeCells(d, k, DOMAIN).coverage).toBeCloseTo(1, 9);
  });

  it('random designs tile the domain for many k', () => {
    for (let seed = 1; seed <= 6; seed++) {
      const r = rng(seed);
      const n = 6 + seed * 3;
      const d: Dataset = {
        X: Array.from({ length: n }, () => [10 * r(), 10 * r()] as Vec2),
        Y: Array.from({ length: n }, () => 10 * r()),
      };
      for (const k of [1, 2, 4, Math.floor(n / 2)]) expect(computeCells(d, k, DOMAIN).coverage).toBeCloseTo(1, 9);
    }
  });
});
