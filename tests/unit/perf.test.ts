import { expect, it } from 'vitest';
import { DOMAIN, PRESETS } from '../../src/data/presets';
import { computeCells } from '../../src/geometry/cells';
import { nwProfile, stepProfile } from '../../src/path/profile';

// Geometry is rebuilt on every pointer move while an observation is dragged,
// so the worst case (16 points, any k) must stay well inside a frame budget.
it('worst-case geometry rebuild stays interactive', () => {
  const dense = PRESETS[2];
  const rows: string[] = [];
  for (const k of [1, 3, 5, 8, 12, 15, 16]) {
    const reps = 3;
    const t0 = performance.now();
    let cells = 0;
    for (let r = 0; r < reps; r++) cells = computeCells(dense, k, DOMAIN).cells.length;
    const ms = (performance.now() - t0) / reps;
    rows.push(`k=${k}: ${cells} cells, ${ms.toFixed(1)} ms`);
    expect(ms).toBeLessThan(80);
  }
  const t1 = performance.now();
  stepProfile(dense, 5, dense.pathA, dense.pathB);
  nwProfile(dense, 0.15, dense.pathA, dense.pathB);
  const tp = performance.now() - t1;
  rows.push(`profiles: ${tp.toFixed(1)} ms`);
  expect(tp).toBeLessThan(80);
  console.log(rows.join('\n'));
});
