import { describe, expect, it } from 'vitest';
import { PRESETS, valueScaleFor } from '../../src/data/presets';
const VALUE_SCALE = PRESETS[0].scale;
import { LUT, LUT_SIZE, SCALE_STOPS_HEX, STOP_POS, colorBytes, rgbToOklab } from '../../src/math/colormap';

describe('V15 colour scale', () => {
  it('stops are ordered by lightness', () => {
    for (let i = 1; i < STOP_POS.length; i++) expect(STOP_POS[i]).toBeGreaterThan(STOP_POS[i - 1]);
  });
  it('OKLab lightness is strictly increasing along the LUT', () => {
    let prev = -1;
    let minStep = Infinity;
    for (let i = 0; i < LUT_SIZE; i += 8) {
      const L = rgbToOklab([LUT[4 * i] / 255, LUT[4 * i + 1] / 255, LUT[4 * i + 2] / 255])[0];
      if (prev >= 0) minStep = Math.min(minStep, L - prev);
      prev = L;
    }
    expect(minStep).toBeGreaterThan(0);
  });
  it('ends are the palette colours and the mapping is fixed', () => {
    expect(colorBytes(VALUE_SCALE.lo, VALUE_SCALE)).toEqual([0x4a, 0x0c, 0x1c]);
    expect(colorBytes(VALUE_SCALE.hi, VALUE_SCALE)).toEqual([0xf7, 0xc3, 0xcc]);
    expect(SCALE_STOPS_HEX.length).toBe(6);
  });
  it('per-dataset domain contains every response, constant Y gets a valid domain', () => {
    for (const p of PRESETS) {
      expect(p.scale.lo).toBeLessThanOrEqual(Math.min(...p.Y));
      expect(p.scale.hi).toBeGreaterThanOrEqual(Math.max(...p.Y));
    }
    const c = valueScaleFor([4, 4, 4]);
    expect(c.hi).toBeGreaterThan(c.lo);
  });
});
