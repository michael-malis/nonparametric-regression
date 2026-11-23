import type { Rect } from '../geometry/cells';
import type { ValueScale } from '../math/colormap';
import type { Vec2 } from '../math/estimators';

export const DOMAIN: Rect = { x0: 0, y0: 0, x1: 10, y1: 10 };

/** Top of the 3-D value axis (heights are drawn from f̂ = 0 upwards). */
export const HEIGHT_MAX = 10;

/**
 * Colour domain of a dataset: [min Y, max Y] rounded outwards to 0.5. With
 * non-negative weights every 1NN, kNN and NW prediction lies in [min Y, max Y],
 * so one fixed domain serves all methods and both views. Dragging X_i never
 * changes Y, hence never changes the scale.
 */
export function valueScaleFor(Y: number[]): ValueScale {
  let lo = Math.floor(2 * Math.min(...Y)) / 2;
  let hi = Math.ceil(2 * Math.max(...Y)) / 2;
  if (hi - lo < 1e-9) {
    lo -= 0.5;
    hi += 0.5;
  }
  return { lo, hi };
}

export interface Preset {
  id: string;
  label: string;
  X: Vec2[];
  Y: number[];
  x0: Vec2;
  k: number;
  h: number;
  pathA: Vec2;
  pathB: Vec2;
  scale: ValueScale;
}

// The first eight sites are shared. The interleaved sites fill the open spaces.
export const OUTLIER = 6;
export const RESPONSE_SITE = 2;
const sites: Vec2[] = [
  [1.1, 1.8], [4.7, 1.1], [8.7, 2.3], [2.8, 4.8],
  [6.6, 5.4], [1.3, 8.3], [5.2, 8.7], [8.9, 8.1],
  [2.9, 1.2], [6.7, 2.1], [1.0, 5.4], [4.5, 3.7],
  [8.8, 5.0], [3.4, 7.1], [6.9, 7.9], [5.0, 6.2],
];

export function routeEndpoints(X: Vec2[]): [Vec2, Vec2] {
  const west = X.reduce((a,b) => a[0] < b[0] ? a : b);
  const east = X.reduce((a,b) => a[0] > b[0] ? a : b);
  return [[west[0] * 0.55, west[1]], [(10 + east[0]) / 2, east[1]]];
}

function preset(n: number, outlier: boolean): Preset {
  const X = sites.slice(0, n);
  const Y = X.map(([x,y],i) => i === OUTLIER && outlier ? 9.7 :
    Math.round(10 * (8.4 - 0.50*x + 0.08*(y-5) + 0.10*Math.sin(i*3))) / 10);
  const [pathA,pathB] = routeEndpoints(X);
  return {id: String(n) + (outlier ? '-outlier' : ''), label: n+' points', X,Y,
    pathA,pathB, x0:[4.6,4.1], k:3, h:1.2, scale:{lo:0,hi:10}};
}
export const PRESETS = [preset(8,false),preset(8,true),preset(16,false),preset(16,true)];
