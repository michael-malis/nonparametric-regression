// Export predictions of the TypeScript engine for an independent NumPy check.
//   npx tsx tests/numerical/export_cases.ts  ->  .artifacts/numerical/cases.json

import { mkdirSync, writeFileSync } from 'node:fs';
import { DOMAIN, PRESETS } from '../../src/data/presets';
import { computeCells } from '../../src/geometry/cells';
import { type Vec2, predictKNN, predictNW } from '../../src/math/estimators';
import { nwProfile, stepProfile } from '../../src/path/profile';

const grid: Vec2[] = [];
for (let i = 0; i <= 24; i++) for (let j = 0; j <= 24; j++) grid.push([(10 * i) / 24 + 0.013, (10 * j) / 24 - 0.007]);

const out = PRESETS.map((p) => ({
  id: p.id,
  X: p.X,
  Y: p.Y,
  grid,
  knn: [1, 3, 5, p.X.length].map((k) => ({
    k,
    values: grid.map((x) => predictKNN(p, x, k).value),
    weights: grid.slice(0, 40).map((x) => Array.from(predictKNN(p, x, k).weights)),
    cellAreas: computeCells(p, k, DOMAIN).cells.map((c) => ({ set: c.set, area: c.area, value: c.value })),
    profile: stepProfile(p, k, p.pathA, p.pathB).segments,
  })),
  nw: [0.15, 0.5, 1, 3].map((h) => ({
    h,
    values: grid.map((x) => predictNW(p, x, h).value),
    weights: grid.slice(0, 40).map((x) => Array.from(predictNW(p, x, h).weights)),
    profile: (() => {
      const pr = nwProfile(p, h, p.pathA, p.pathB);
      return { s: Array.from(pr.s).filter((_, i) => i % 7 === 0), v: Array.from(pr.v).filter((_, i) => i % 7 === 0) };
    })(),
  })),
  pathA: p.pathA,
  pathB: p.pathB,
}));

const outputDir = new URL('../../.artifacts/numerical/', import.meta.url);
mkdirSync(outputDir, { recursive: true });
writeFileSync(new URL('cases.json', outputDir), JSON.stringify(out));
console.log(`exported ${out.length} datasets, ${grid.length} grid points each`);
