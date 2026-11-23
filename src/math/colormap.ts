// One fixed value scale for the map, surface, observations, legend and profile.
// Increasing prediction corresponds to increasing OKLab lightness.
export const SCALE_STOPS_HEX = ['#4A0C1C', '#78142B', '#A81B3C', '#D92D4F', '#EF8090', '#F7C3CC'];

type Lab = [number, number, number];

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
function linearToSrgb(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
}

export function rgbToOklab([r, g, b]: [number, number, number]): Lab {
  const R = srgbToLinear(r);
  const G = srgbToLinear(g);
  const B = srgbToLinear(b);
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B);
  const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
  const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

export function oklabToRgb([L, a, b]: Lab): [number, number, number] {
  const l = Math.pow(L + 0.3963377774 * a + 0.2158037573 * b, 3);
  const m = Math.pow(L - 0.1055613458 * a - 0.0638541728 * b, 3);
  const s = Math.pow(L - 0.0894841775 * a - 1.291485548 * b, 3);
  const R = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  const G = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  const B = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s;
  const clamp = (x: number) => Math.min(1, Math.max(0, x));
  return [clamp(linearToSrgb(R)), clamp(linearToSrgb(G)), clamp(linearToSrgb(B))];
}

const STOP_LAB = SCALE_STOPS_HEX.map((h) => rgbToOklab(hexToRgb(h)));
const L0 = STOP_LAB[0][0];
const L1 = STOP_LAB[STOP_LAB.length - 1][0];
/** Stop positions in [0,1]: proportional to OKLab lightness. */
export const STOP_POS = STOP_LAB.map((lab) => (lab[0] - L0) / (L1 - L0));

/** Colour at normalized position t in [0, 1]. */
export function scaleAt(t: number): [number, number, number] {
  const u = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  let i = 0;
  while (i < STOP_POS.length - 2 && u > STOP_POS[i + 1]) i++;
  const a = STOP_POS[i];
  const b = STOP_POS[i + 1];
  const w = b > a ? (u - a) / (b - a) : 0;
  const A = STOP_LAB[i];
  const B = STOP_LAB[i + 1];
  return oklabToRgb([A[0] + w * (B[0] - A[0]), A[1] + w * (B[1] - A[1]), A[2] + w * (B[2] - A[2])]);
}

/** The fixed numeric domain of the colour (and height) scale. */
export interface ValueScale {
  lo: number;
  hi: number;
}

export const LUT_SIZE = 1024;

/** RGBA lookup table, sRGB bytes. */
export function buildLut(): Uint8Array {
  const lut = new Uint8Array(LUT_SIZE * 4);
  for (let i = 0; i < LUT_SIZE; i++) {
    const [r, g, b] = scaleAt(i / (LUT_SIZE - 1));
    lut[4 * i] = Math.round(r * 255);
    lut[4 * i + 1] = Math.round(g * 255);
    lut[4 * i + 2] = Math.round(b * 255);
    lut[4 * i + 3] = 255;
  }
  return lut;
}

export const LUT = buildLut();

export function normalize(v: number, scale: ValueScale): number {
  if (scale.hi <= scale.lo) return 0.5;
  return (v - scale.lo) / (scale.hi - scale.lo);
}

export function colorBytes(v: number, scale: ValueScale): [number, number, number] {
  const t = Math.min(1, Math.max(0, normalize(v, scale)));
  const i = Math.round(t * (LUT_SIZE - 1));
  return [LUT[4 * i], LUT[4 * i + 1], LUT[4 * i + 2]];
}

export function colorCss(v: number, scale: ValueScale): string {
  const [r, g, b] = colorBytes(v, scale);
  return `rgb(${r},${g},${b})`;
}

export function colorHex(v: number, scale: ValueScale): number {
  const [r, g, b] = colorBytes(v, scale);
  return (r << 16) | (g << 8) | b;
}
