import { describe, expect, it } from 'vitest';
import { type Dataset, type Vec2, mean, nwGrid, nwValue, pack, predict1NN, predictKNN, predictNW } from '../../src/math/estimators';
import { PRESETS } from '../../src/data/presets';

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}
function randomData(seed: number, n: number): Dataset {
  const r = rng(seed);
  const X: Vec2[] = [];
  const Y: number[] = [];
  for (let i = 0; i < n; i++) {
    X.push([10 * r(), 10 * r()]);
    Y.push(Math.round(100 * (1 + 8 * r())) / 100);
  }
  return { X, Y };
}
const sum = (w: ArrayLike<number>) => Array.from(w).reduce((a, b) => a + b, 0);

describe('V01 1NN', () => {
  it('returns the response of the nearest design point', () => {
    const data: Dataset = { X: [[0, 0], [4, 0], [0, 3]], Y: [10, 20, 30] };
    expect(predict1NN(data, [0.4, 0.2]).value).toBe(10);
    expect(predict1NN(data, [3.1, 0.1]).value).toBe(20);
    expect(predict1NN(data, [0.2, 2.2]).value).toBe(30);
    expect(predict1NN(data, [0.4, 0.2]).neighbors).toEqual([0]);
  });
  it('reproduces every observation at its own design point (distinct X)', () => {
    for (const p of PRESETS) p.X.forEach((x, i) => expect(predict1NN(p, x).value).toBe(p.Y[i]));
  });
});

describe('V02 kNN weights', () => {
  const data = randomData(7, 12);
  for (const k of [1, 2, 3, 12]) {
    it(`k=${k}: k weights equal 1/k, the rest 0, sum 1`, () => {
      const x: Vec2 = [4.2, 5.1];
      const p = predictKNN(data, x, k);
      const nonzero = Array.from(p.weights).filter((w) => w > 0);
      expect(nonzero.length).toBe(k);
      nonzero.forEach((w) => expect(w).toBeCloseTo(1 / k, 15));
      expect(sum(p.weights)).toBeCloseTo(1, 14);
      const dmax = Math.max(...p.neighbors.map((i) => p.d2[i]));
      for (let i = 0; i < data.X.length; i++) if (!p.neighbors.includes(i)) expect(p.d2[i]).toBeGreaterThanOrEqual(dmax);
      expect(p.value).toBeCloseTo(mean(p.neighbors.map((i) => data.Y[i])), 14);
      expect(p.radius).toBeCloseTo(Math.sqrt(dmax), 14);
    });
  }
});

describe('V03 deterministic ties', () => {
  const data: Dataset = { X: [[-1, 0], [1, 0], [0, 5]], Y: [1, 3, 9] };
  it('on the exact bisector the smaller index wins', () => {
    expect(predict1NN(data, [0, 0]).neighbors).toEqual([0]);
    expect(predict1NN(data, [0, -2]).value).toBe(1);
  });
  it('is stable under repetition', () => {
    const a = predictKNN(data, [0, 1], 2).neighbors;
    for (let t = 0; t < 20; t++) expect(predictKNN(data, [0, 1], 2).neighbors).toEqual(a);
  });
  it('coincident design points: smaller index first', () => {
    const d: Dataset = { X: [[2, 2], [2, 2], [5, 5]], Y: [1, 7, 4] };
    expect(predict1NN(d, [2.1, 2]).neighbors).toEqual([0]);
  });
});

describe('V04 k = n gives the global mean', () => {
  it('is constant on a grid', () => {
    const data = randomData(3, 9);
    const ybar = mean(data.Y);
    for (let i = 0; i <= 10; i++) for (let j = 0; j <= 10; j++) expect(predictKNN(data, [i, j], 9).value).toBeCloseTo(ybar, 13);
  });
  it('k outside [1, n] is clamped', () => {
    const data = randomData(3, 9);
    expect(predictKNN(data, [1, 1], 50).k).toBe(9);
    expect(predictKNN(data, [1, 1], 0).k).toBe(1);
    expect(predictKNN(data, [1, 1], NaN).k).toBe(1);
  });
});

describe('V05 Gaussian NW weights', () => {
  const data = randomData(11, 10);
  for (const h of [0.3, 1, 3]) {
    it(`h=${h}: positive, sum to 1, equal the textbook formula`, () => {
      const x: Vec2 = [5, 5];
      const p = predictNW(data, x, h);
      const raw = data.X.map((Xi) => Math.exp(-((Xi[0] - x[0]) ** 2 + (Xi[1] - x[1]) ** 2) / (2 * h * h)));
      const tot = raw.reduce((a, b) => a + b, 0);
      p.weights.forEach((w, i) => {
        expect(w).toBeGreaterThan(0);
        expect(w).toBeCloseTo(raw[i] / tot, 12);
      });
      expect(sum(p.weights)).toBeCloseTo(1, 14);
      expect(p.value).toBeCloseTo(raw.reduce((a, r, i) => a + r * data.Y[i], 0) / tot, 12);
    });
  }
  it('hand computation with three points, h = 1', () => {
    const d: Dataset = { X: [[0, 0], [1, 0], [0, 2]], Y: [10, 20, 30] };
    const a = [1, Math.exp(-0.5), Math.exp(-2)];
    const t = a[0] + a[1] + a[2];
    expect(predictNW(d, [0, 0], 1).value).toBeCloseTo((10 * a[0] + 20 * a[1] + 30 * a[2]) / t, 14);
  });
});

describe('V06 extreme bandwidths', () => {
  const data = randomData(5, 14);
  it('no NaN / Infinity for tiny and huge h, far query points', () => {
    for (const h of [1e-9, 1e-4, 0.01, 0.15, 6, 1e6, 0, -1, NaN, Infinity]) {
      for (const x of [[5, 5], [1e4, -1e4], [-300, 2]] as Vec2[]) {
        const p = predictNW(data, x, h);
        expect(Number.isFinite(p.value)).toBe(true);
        expect(sum(p.weights)).toBeCloseTo(1, 12);
      }
    }
  });
  it('the packed grid evaluator agrees with predictNW', () => {
    const pk = pack(data);
    const s = new Float64Array(pk.n);
    for (const h of [0.05, 0.7, 4])
      for (const x of [[2, 3], [7.7, 0.4]] as Vec2[]) expect(nwValue(pk, x[0], x[1], h, s)).toBeCloseTo(predictNW(data, x, h).value, 12);
  });
});

describe('V07 convex combination bounds', () => {
  it('min Y <= f̂ <= max Y for all estimators on random data', () => {
    for (let seed = 1; seed <= 20; seed++) {
      const data = randomData(seed, 3 + (seed % 15));
      const lo = Math.min(...data.Y);
      const hi = Math.max(...data.Y);
      const r = rng(seed * 31);
      for (let t = 0; t < 30; t++) {
        const x: Vec2 = [12 * r() - 1, 12 * r() - 1];
        const vals = [
          predict1NN(data, x).value,
          predictKNN(data, x, 1 + (t % data.X.length)).value,
          predictNW(data, x, 0.05 + 4 * r()).value,
        ];
        for (const v of vals) {
          expect(v).toBeGreaterThanOrEqual(lo - 1e-12);
          expect(v).toBeLessThanOrEqual(hi + 1e-12);
        }
      }
    }
  });
});

describe('V08 NW limits', () => {
  const data = randomData(21, 12);
  it('h -> 0 approaches 1NN, h -> ∞ approaches the mean', () => {
    const r = rng(9);
    for (let t = 0; t < 30; t++) {
      const x: Vec2 = [10 * r(), 10 * r()];
      expect(predictNW(data, x, 1e-4).value).toBeCloseTo(predict1NN(data, x).value, 8);
      expect(predictNW(data, x, 1e5).value).toBeCloseTo(mean(data.Y), 6);
    }
  });
  it('h -> 0 at an exact tie averages the tied points', () => {
    const d: Dataset = { X: [[-1, 0], [1, 0], [0, 6]], Y: [2, 6, 9] };
    expect(predictNW(d, [0, 0], 1e-4).value).toBeCloseTo(4, 10);
  });
});

describe('V35 degenerate data', () => {
  it('constant responses give a constant estimate everywhere', () => {
    const d: Dataset = { X: [[1, 1], [5, 2], [3, 8]], Y: [4, 4, 4] };
    for (const x of [[0, 0], [9, 9], [3, 3]] as Vec2[]) {
      expect(predict1NN(d, x).value).toBe(4);
      expect(predictKNN(d, x, 2).value).toBe(4);
      expect(predictNW(d, x, 0.5).value).toBeCloseTo(4, 14);
    }
  });
  it('coincident design points do not crash any estimator', () => {
    const d: Dataset = { X: [[2, 2], [2, 2], [2, 2]], Y: [1, 5, 9] };
    expect(predict1NN(d, [0, 0]).value).toBe(1);
    expect(predictKNN(d, [0, 0], 3).value).toBe(5);
    expect(predictNW(d, [0, 0], 0.01).value).toBeCloseTo(5, 12);
  });
  it('empty data throws a clear error', () => {
    expect(() => predictKNN({ X: [], Y: [] }, [0, 0], 1)).toThrow(/empty/);
    expect(() => predictNW({ X: [], Y: [] }, [0, 0], 1)).toThrow(/empty/);
  });
});

it('separable NW grid matches the point estimator, including distant points at narrow bandwidths', () => {
  const d = PRESETS[2];
  const xs = Float64Array.of(0, 2.45, 5, 10, 50);
  const ys = Float64Array.of(10, 7.12, 2, 0);
  for (const h of [0.15, 0.8, 3, 6]) {
    const grid = nwGrid(pack(d), xs, ys, h);
    ys.forEach((y, j) => xs.forEach((x, i) => {
      expect(grid[j * xs.length + i]).toBeCloseTo(predictNW(d, [x, y], h).value, 11);
    }));
  }
});
