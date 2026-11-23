import { describe, expect, it } from 'vitest';
import { PRESETS } from '../../src/data/presets';
import { predictKNN, predictNW } from '../../src/math/estimators';
import { nwProfile, pointOnPath, stepProfile } from '../../src/path/profile';

describe('V24 / V25 exact step profile along γ', () => {
  for (const preset of PRESETS)
    for (const k of [1, 3, 5]) {
      it(`${preset.id}, k=${k}: matches dense direct evaluation away from breakpoints`, () => {
        const prof = stepProfile(preset, k, preset.pathA, preset.pathB);
        expect(prof.segments[0].s0).toBe(0);
        expect(prof.segments[prof.segments.length - 1].s1).toBe(1);
        for (let i = 1; i < prof.segments.length; i++) {
          expect(prof.segments[i].s0).toBe(prof.segments[i - 1].s1);
          expect(Math.abs(prof.segments[i].value - prof.segments[i - 1].value)).toBeGreaterThan(1e-12);
        }
        const N = 5000;
        let jumps = 0;
        let prev = NaN;
        for (let j = 0; j <= N; j++) {
          const s = j / N;
          const direct = predictKNN(preset, pointOnPath(preset.pathA, preset.pathB, s), k).value;
          const seg = prof.segments.find((g) => s >= g.s0 && s <= g.s1)!;
          const nearCut = prof.segments.some((g) => Math.abs(s - g.s0) < 1e-6 || Math.abs(s - g.s1) < 1e-6);
          if (!nearCut) expect(direct).toBeCloseTo(seg.value, 12);
          if (j > 0 && Math.abs(direct - prev) > 1e-12) jumps++;
          prev = direct;
        }
        // Every jump that dense sampling sees is a boundary of the exact profile.
        expect(prof.segments.length - 1).toBeGreaterThanOrEqual(jumps);
      });
    }
});

describe('V26 NW profile', () => {
  it('samples equal the engine and the profile has no jumps', () => {
    const p = PRESETS[0];
    for (const h of [0.2, 1, 4]) {
      const prof = nwProfile(p, h, p.pathA, p.pathB);
      for (let i = 0; i < prof.s.length; i += 37) {
        expect(prof.v[i]).toBeCloseTo(predictNW(p, pointOnPath(p.pathA, p.pathB, prof.s[i]), h).value, 12);
      }
      let maxStep = 0;
      for (let i = 1; i < prof.v.length; i++) maxStep = Math.max(maxStep, Math.abs(prof.v[i] - prof.v[i - 1]));
      expect(maxStep).toBeLessThanOrEqual(0.02 + 1e-12);
      for (let i = 1; i < prof.s.length; i++) expect(prof.s[i]).toBeGreaterThan(prof.s[i - 1]);
    }
  });
});
