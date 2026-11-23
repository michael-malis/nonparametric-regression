// Shared drawing colours for Canvas, WebGL and the path profile.
export const THEME = {
  bg: '#FFFAFB',
  bgRaised: '#FDEEF0',
  ink: '#4A0C1C',
  inkMuted: '#784554',
  inkFaint: 'rgba(120,69,84,0.38)',
  accent: '#A81B3C',
  path: '#4A0C1C',
  ring: '#4A0C1C',
  wall: '#AB7081',
  // The pale outer stroke keeps paths and query markers visible on dark plateaus.
  outline: '#FFFAFB',
  label: 'rgba(255,250,251,0.94)',
  font: '"Atkinson Hyperlegible", "Segoe UI", system-ui, sans-serif',
};

/** Draw "base" with a lowered subscript, e.g. Y₁₂ = 6.2, on a 2-D canvas. */
export function drawSubText(
  ctx: CanvasRenderingContext2D,
  base: string,
  sub: string,
  rest: string,
  x: number,
  y: number,
  size: number,
  align: 'left' | 'center' | 'right' = 'left',
  halo = 0,
): number {
  ctx.save();
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  const big = `${size}px ${THEME.font}`;
  const small = `${Math.round(size * 0.68)}px ${THEME.font}`;
  ctx.font = big;
  const wb = ctx.measureText(base).width;
  const wr = ctx.measureText(rest).width;
  ctx.font = small;
  const ws = sub ? ctx.measureText(sub).width + 1 : 0;
  const total = wb + ws + wr;
  let cx = align === 'left' ? x : align === 'center' ? x - total / 2 : x - total;
  // Optional halo: the caller sets ctx.strokeStyle and passes halo > 0.
  const put = (t: string, px: number, py: number) => {
    if (halo > 0) {
      ctx.lineWidth = halo;
      ctx.lineJoin = 'round';
      ctx.strokeText(t, px, py);
    }
    ctx.fillText(t, px, py);
  };
  ctx.font = big;
  put(base, cx, y);
  cx += wb;
  if (sub) {
    ctx.font = small;
    put(sub, cx + 0.5, y + size * 0.22);
    cx += ws;
  }
  ctx.font = big;
  put(rest, cx, y);
  ctx.restore();
  return total;
}

export function fmt(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return '—';
  const s = v.toFixed(digits);
  return s === `-${(0).toFixed(digits)}` ? (0).toFixed(digits) : s;
}

/**
 * One format for every displayed weight. Gaussian weights are never 0, so a
 * tiny one prints as "<0.001" and one just below 1 as ">0.999" (never "0.000"
 * or "1.000" for a value that is not exactly that). kNN zeros print as "0".
 */
export function fmtW(w: number): string {
  if (!Number.isFinite(w)) return '—';
  if (w === 0) return '0';
  if (w === 1) return '1';
  if (w < 0.0005) return '<0.001';
  if (w >= 0.9995) return '>0.999';
  return w.toFixed(3);
}

/** Positive number in a readable form: 3 decimals, or a·10⁻ⁿ when tiny. */
export function fmtSmall(v: number): string {
  if (v === 0) return '0';
  if (v >= 0.001) return v.toFixed(3);
  const e = Math.floor(Math.log10(v));
  const m = v / Math.pow(10, e);
  return `${m.toFixed(1)}·10${sup(e)}`;
}

const SUB = '₀₁₂₃₄₅₆₇₈₉';
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
export function sub(i: number): string {
  return String(i)
    .split('')
    .map((d) => SUB[+d])
    .join('');
}
export function sup(i: number): string {
  return String(i)
    .split('')
    .map((d) => (d === '-' ? '⁻' : SUP[+d]))
    .join('');
}

/** "= 0.351", "< 0.001" or "> 0.999": a weight as a relation, for labels like "w₁₂ > 0.999". */
export function relW(w: number): string {
  const s = fmtW(w);
  return s[0] === '<' || s[0] === '>' ? `${s[0]} ${s.slice(1)}` : `= ${s}`;
}

/**
 * A weight as a number inside the live arithmetic: 3 decimals, or as many as
 * needed (up to 6) to show that a weight close to 1 is not 1.
 */
export function fmtWTerm(w: number): string {
  if (w < 0.9995 || w === 1) return w === 1 ? '1' : w.toFixed(3);
  for (let d = 4; d <= 6; d++) if (w.toFixed(d) !== (1).toFixed(d)) return w.toFixed(d);
  return '>0.999999';
}
