// 2-D view: the feature plane (x1, x2) with the prediction map f̂(x1, x2).
// 1NN / kNN: exact order-k cells, each filled with the single colour C(f̂).
// NW: the continuous map f̂_h, evaluated per sample with the same formula.

import { DOMAIN, OUTLIER } from '../data/presets';
import { LUT, LUT_SIZE, colorCss, normalize } from '../math/colormap';
import { type Vec2, nwGrid, pack } from '../math/estimators';
import { cells, effectiveK, prediction, upcomingCrossing, valueScale } from '../state/model';
import type { Key, State, Store } from '../state/store';
import { THEME, drawSubText, fmt, relW, sub } from '../ui/theme';
import { type DragTarget, applyDrag, beginDrag, endDrag, handlePosition } from './handles';

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}
const overlaps = (a: Rect, b: Rect) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/** True when the NW kernel halo K(|x - x0|/h) should be visible. */
export function haloOn(s: State): boolean {
  return s.method === 'nw' && (s.haloPinned || s.haloActive || s.experiment?.id === 'hsweep');
}

export interface Square {
  x: number;
  y: number;
  side: number;
}

export class View2D {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private map = document.createElement('canvas');
  private mapKey = '';
  private dpr = 1;
  private w = 0;
  private h = 0;
  sq: Square = { x: 0, y: 0, side: 1 };
  private drag: { target: DragTarget; pointerId: number; offset: Vec2 } | null = null;
  private hoverTarget: DragTarget | null = null;
  visible = true;

  constructor(
    private root: HTMLElement,
    private store: Store,
    private onPointEdit: (index:number) => void,
  ) {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'view2d';
    this.canvas.tabIndex = 0;
    this.canvas.addEventListener('keydown', e => {
      if (document.body.dataset.mode !== 'explore') return;
      const steps: Record<string,Vec2> = {ArrowLeft:[-0.1,0], ArrowRight:[0.1,0], ArrowUp:[0,0.1], ArrowDown:[0,-0.1]};
      const step = steps[e.key];
      if (!step) return;
      const [x,y] = this.store.get().x0;
      applyDrag(this.store,{kind:'x0'},[x+step[0],y+step[1]]);
      e.preventDefault();
    });
    this.canvas.setAttribute('aria-label', 'Feature plane with the prediction map; drag the query or observations in Explore, or use arrow keys to move the query');
    root.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;
    this.canvas.addEventListener('pointerdown', this.onDown);
    this.canvas.addEventListener('pointerleave', () => {
      if (!this.drag && this.hoverTarget) {this.hoverTarget = null; this.draw();}
    });
    this.canvas.addEventListener('pointermove', this.onMove);
    this.canvas.addEventListener('pointerup', this.onUp);
    this.canvas.addEventListener('pointercancel', this.onUp);
    this.canvas.addEventListener('lostpointercapture', this.onUp);
    new ResizeObserver(() => this.resize()).observe(root);
    this.resize();
  }

  setVisible(v: boolean): void {
    this.visible = v;
    this.canvas.style.display = v ? 'block' : 'none';
    if (v) {
      this.resize();
    }
  }

  resize(): void {
    const r = this.root.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = r.width;
    this.h = r.height;
    this.canvas.width = Math.round(r.width * this.dpr);
    this.canvas.height = Math.round(r.height * this.dpr);
    this.canvas.style.width = `${r.width}px`;
    this.canvas.style.height = `${r.height}px`;
    // Square plot: equal scale on both axes keeps bisectors and circles Euclidean.
    const padTop = 26;
    const padBottom = 52;
    const free = this.w - 20;
    const side = Math.max(120, Math.min(this.h - padTop - padBottom, free - 40));
    this.sq = {
      x: Math.round(10 + 40 + (free - 40 - side) / 2),
      y: Math.round(padTop + (this.h - padTop - padBottom - side) / 2),
      side: Math.round(side),
    };
    this.draw();
  }

  update(_s: State, _changed: Set<Key>): void {
    if (this.visible) this.draw();
  }

  // --- coordinates ---------------------------------------------------------
  toPx(x: Vec2): [number, number] {
    const sc = this.sq.side / (DOMAIN.x1 - DOMAIN.x0);
    return [this.sq.x + (x[0] - DOMAIN.x0) * sc, this.sq.y + (DOMAIN.y1 - x[1]) * sc];
  }

  toFeature(px: number, py: number): Vec2 {
    const sc = this.sq.side / (DOMAIN.x1 - DOMAIN.x0);
    return [DOMAIN.x0 + (px - this.sq.x) / sc, DOMAIN.y1 - (py - this.sq.y) / sc];
  }

  private local(e: PointerEvent): [number, number] {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  // --- picking -------------------------------------------------------------
  pick(px: number, py: number): DragTarget | null {
    const s = this.store.get();
    const d = (x: Vec2) => {
      const [qx, qy] = this.toPx(x);
      return Math.hypot(qx - px, qy - py);
    };
    let best: DragTarget | null = null;
    let bestD = Infinity;
    // Compare visible anchors in CSS pixels. On an exact tie an observation wins.
    if (document.body.dataset.mode === 'explore') {
      s.X.forEach((x, i) => {
        const dd = d(x);
        if (dd <= 15 && dd < bestD) {
          best = { kind: 'point', index: i };
          bestD = dd;
        }
      });
    }
    const queryDistance = d(s.x0);
    if (queryDistance <= 16 && queryDistance < bestD) {
      best = { kind: 'x0' };
      bestD = queryDistance;
    }
    if (s.profileOpen && !s.experiment) {
      for (const kind of ['pathA', 'pathB'] as const) {
        const dd = d(kind === 'pathA' ? s.pathA : s.pathB);
        if (dd <= 15 && dd < bestD) {
          best = { kind };
          bestD = dd;
        }
      }
    }
    return best;
  }

  private onDown = (e: PointerEvent) => {
    if (this.drag || e.button !== 0 || document.body.dataset.mode !== 'explore') return;
    const [px, py] = this.local(e);
    let target = this.pick(px, py);
    const f = this.toFeature(px, py);
    const inside = px >= this.sq.x && px <= this.sq.x + this.sq.side && py >= this.sq.y && py <= this.sq.y + this.sq.side;
    if (!target && inside) {
      // A press on empty map places x0 there and keeps dragging it.
      target = { kind: 'x0' };
      beginDrag(this.store, target);
      applyDrag(this.store, target, f);
    } else if (target) {
      if (target.kind === 'point') this.onPointEdit(target.index!);
      beginDrag(this.store, target);
    } else {
      return;
    }
    const hp = handlePosition(this.store.get(), target);
    this.drag = { target, pointerId: e.pointerId, offset: [hp[0] - f[0], hp[1] - f[1]] };
    this.canvas.setPointerCapture(e.pointerId);
    this.canvas.style.cursor = 'grabbing';
    e.preventDefault();
  };

  private onMove = (e: PointerEvent) => {
    const [px, py] = this.local(e);
    if (this.drag && e.pointerId === this.drag.pointerId) {
      const f = this.toFeature(px, py);
      applyDrag(this.store, this.drag.target, [f[0] + this.drag.offset[0], f[1] + this.drag.offset[1]]);
      return;
    }
    const state = this.store.get();
    const site = state.X.findIndex(p=> {const [x,y]=this.toPx(p);return Math.hypot(x-px,y-py)<14;});
    this.canvas.title = site<0 ? '' : `${state.includeOutlier && site===OUTLIER?'Outlier · ':''}Y${sub(site+1)} = ${state.Y[site].toFixed(1)}`;
    const t = this.pick(px, py);
    if ((t?.kind ?? null) !== (this.hoverTarget?.kind ?? null) || t?.index !== this.hoverTarget?.index) {
      this.hoverTarget = t;
      this.canvas.style.cursor = t ? 'grab' : 'crosshair';
      this.draw();
    }
  };

  private onUp = (e: PointerEvent) => {
    if (!this.drag || e.pointerId !== this.drag.pointerId) return;
    this.drag = null;
    if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
    this.canvas.style.cursor = 'grab';
    endDrag(this.store);
  };

  // --- drawing -------------------------------------------------------------
  private renderMap(s: State): void {
    const k = effectiveK(s);
    const VS = valueScale(s);
    // While an experiment animates NW, a coarser grid keeps the frame rate up;
    // every stop is drawn again at full resolution.
    const moving = s.experiment?.phase === 'running';
    const size = Math.round(this.sq.side * this.dpr);
    const res = Math.min(moving ? 220 : 460, Math.max(120, Math.round(size / 2)));
    const key = `${s.dataVersion}|${s.presetIndex}|${s.method}|${k}|${s.method === 'nw' ? `${s.h}|${res}` : s.highlightObs}|${VS.lo}|${VS.hi}|${this.sq.side}|${this.dpr}|${s.showRegions}`;
    if (key === this.mapKey) return;
    this.mapKey = key;
    this.map.width = size;
    this.map.height = size;
    const m = this.map.getContext('2d')!;
    m.clearRect(0, 0, size, size);
    const sc = size / (DOMAIN.x1 - DOMAIN.x0);
    const P = (x: Vec2): [number, number] => [(x[0] - DOMAIN.x0) * sc, (DOMAIN.y1 - x[1]) * sc];
    const poly = (pts: Vec2[]) => {
      m.beginPath();
      pts.forEach((v, i) => {
        const [x, y] = P(v);
        if (i === 0) m.moveTo(x, y);
        else m.lineTo(x, y);
      });
      m.closePath();
    };

    if (s.method !== 'nw') {
      const diag = cells(s, k);
      for (const c of diag.cells) {
        poly(c.polygon);
        m.fillStyle = colorCss(c.value, VS);
        m.fill();
        // Same-colour stroke closes anti-aliasing seams between cells.
        m.strokeStyle = m.fillStyle;
        m.lineWidth = 1;
        m.stroke();
      }
      // Cell boundaries (where N_k(x) changes): faint, so the eye compares values, not polygons.
      m.strokeStyle = 'rgba(74,12,28,0.28)';
      m.lineWidth = Math.max(1, 1.1 * this.dpr);
      m.lineJoin = 'round';
      for (const c of diag.cells) {
        for (const e of c.edges) {
          if (e.boundary || (k>1 && !s.showRegions)) continue;
          const [ax, ay] = P(e.p);
          const [bx, by] = P(e.q);
          m.beginPath();
          m.moveTo(ax, ay);
          m.lineTo(bx, by);
          m.stroke();
        }
      }
      // Footprint of one observation: every cell whose neighbour set contains it.
      if (s.highlightObs !== null) {
        const j = s.highlightObs;
        m.lineJoin = 'round';
        for (const c of diag.cells) {
          if (!c.set.includes(j)) continue;
          poly(c.polygon);
          m.strokeStyle = THEME.outline;
          m.lineWidth = 4.5 * this.dpr;
          m.stroke();
          m.strokeStyle = THEME.ring;
          m.lineWidth = 2 * this.dpr;
          m.stroke();
        }
      }
      return;
    }

    // Nadaraya–Watson: the exact estimator at every sample centre, coloured by
    // the value scale. No level lines: they would read as region borders that
    // NW does not have, and they darkened flat areas at small h.
    const pk = pack(s);

    const img = new ImageData(res, res);
    const D = img.data;
    const step = (DOMAIN.x1 - DOMAIN.x0) / res;
    const xs = Float64Array.from({length:res}, (_,i)=>DOMAIN.x0+(i+0.5)*step);
    const ys = Float64Array.from({length:res}, (_,i)=>DOMAIN.y1-(i+0.5)*step);
    const grid = nwGrid(pk,xs,ys,s.h);
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const val = grid[j*res+i];
        const t = Math.min(1, Math.max(0, normalize(val, VS)));
        const li = Math.round(t * (LUT_SIZE - 1)) * 4;
        const idx = 4 * (j * res + i);
        D[idx] = LUT[li];
        D[idx + 1] = LUT[li + 1];
        D[idx + 2] = LUT[li + 2];
        D[idx + 3] = 255;
      }
    }
    const tmp = document.createElement('canvas');
    tmp.width = res;
    tmp.height = res;
    tmp.getContext('2d')!.putImageData(img, 0, 0);
    m.imageSmoothingEnabled = true;
    m.imageSmoothingQuality = 'high';
    m.drawImage(tmp, 0, 0, size, size);
  }

  draw(): void {
    if (!this.visible || this.w < 2) return;
    const s = this.store.get();
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = THEME.bg;
    ctx.fillRect(0, 0, this.w, this.h);
    this.renderMap(s);
    const { x, y, side } = this.sq;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.map, x, y, side, side);
    if (haloOn(s)) this.drawHalo(s);
    this.drawAxes();
    this.obstacles = [];
    if (s.profileOpen) this.drawPath(s);
    this.drawOverlays(s);
  }

  private drawAxes(): void {
    const ctx = this.ctx;
    const { x, y, side } = this.sq;
    ctx.strokeStyle = THEME.inkFaint;
    ctx.lineWidth = 1;
    ctx.strokeRect(x - 0.5, y - 0.5, side + 1, side + 1);
    ctx.fillStyle = THEME.inkMuted;
    ctx.font = `14px ${THEME.font}`;
    for (let t = 0; t <= 10; t += 2) {
      const [px] = this.toPx([t, 0]);
      const [, py] = this.toPx([0, t]);
      ctx.strokeStyle = THEME.inkFaint;
      ctx.beginPath();
      ctx.moveTo(px, y + side);
      ctx.lineTo(px, y + side + 6);
      ctx.moveTo(x, py);
      ctx.lineTo(x - 6, py);
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(String(t), px, y + side + 9);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(t), x - 10, py);
    }
    ctx.fillStyle = THEME.ink;
    drawSubText(ctx, 'x', '1', '', x + side / 2, y + side + 42, 17, 'center');
    ctx.save();
    ctx.translate(x - 40, y + side / 2);
    ctx.rotate(-Math.PI / 2);
    drawSubText(ctx, 'x', '2', '', 0, 6, 17, 'center');
    ctx.restore();
  }

  /**
   * Soft kernel halo for NW (2-D only, while h changes or when pinned with K):
   * the map is dimmed in proportion to 1 - K(|x - x0|/h), so brightness follows
   * the Gaussian kernel. It has no edge: K is never 0. In 3-D the surface keeps
   * its exact colours; the weights show in the links, rings and weight panel.
   */
  private drawHalo(s: State): void {
    const ctx = this.ctx;
    const [qx, qy] = this.toPx(s.x0);
    const sc = this.sq.side / (DOMAIN.x1 - DOMAIN.x0);
    const R = 3.6 * s.h * sc;
    const g = ctx.createRadialGradient(qx, qy, 0, qx, qy, R);
    for (let i = 0; i <= 12; i++) {
      const t = i / 12;
      const K = Math.exp(-((3.6 * t) ** 2) / 2);
      g.addColorStop(t, `rgba(74,12,28,${(0.5 * (1 - K)).toFixed(3)})`);
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(this.sq.x, this.sq.y, this.sq.side, this.sq.side);
    ctx.clip();
    ctx.fillStyle = g;
    ctx.fillRect(this.sq.x, this.sq.y, this.sq.side, this.sq.side);
    ctx.restore();
  }

  private drawPath(s: State): void {
    const ctx = this.ctx;
    const [ax, ay] = this.toPx(s.pathA);
    const [bx, by] = this.toPx(s.pathB);
    ctx.lineCap = 'round';
    ctx.strokeStyle = THEME.outline;
    ctx.lineWidth = 6;
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.globalAlpha = 0.85;
    ctx.strokeStyle = THEME.path;
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 7]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    if (s.onPath) {
      const [px,py] = this.toPx(s.x0);
      ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(px,py); ctx.stroke();
    }
    const crossing = upcomingCrossing(s);
    if (crossing) {
      const [x,y] = this.toPx(crossing);
      ctx.lineWidth=3; ctx.strokeStyle=THEME.outline; ctx.strokeRect(x-6,y-6,12,12);
      ctx.lineWidth=1.5; ctx.strokeStyle=THEME.path; ctx.strokeRect(x-6,y-6,12,12);
    }
    // The dashed line is a soft obstacle: labels prefer not to sit on it.
    const len = Math.hypot(bx - ax, by - ay);
    for (let d = 0; d <= len; d += 12) {
      const u = len > 0 ? d / len : 0;
      this.obstacles.push({ x: ax + u * (bx - ax) - 4, y: ay + u * (by - ay) - 4, w: 8, h: 8 });
    }
    // direction arrow at 55 %
    const ang = Math.atan2(by - ay, bx - ax);
    const mx = ax + 0.55 * (bx - ax);
    const my = ay + 0.55 * (by - ay);
    ctx.fillStyle = THEME.path;
    ctx.strokeStyle = THEME.outline;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(mx + 8 * Math.cos(ang), my + 8 * Math.sin(ang));
    ctx.lineTo(mx + 8 * Math.cos(ang + 2.5), my + 8 * Math.sin(ang + 2.5));
    ctx.lineTo(mx + 8 * Math.cos(ang - 2.5), my + 8 * Math.sin(ang - 2.5));
    ctx.closePath();
    ctx.stroke();
    ctx.fill();
    for (const [px, py, label] of [
      [ax, ay, 'A'],
      [bx, by, 'B'],
    ] as const) {
      ctx.fillStyle = THEME.path;
      ctx.strokeStyle = THEME.outline;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.rect(px - 6, py - 6, 12, 12);
      ctx.fill();
      ctx.stroke();
      this.obstacles.push({ x: px - 8, y: py - 8, w: 16, h: 16 }, { x: px - 7, y: py + 9, w: 14, h: 18 });
      ctx.fillStyle = THEME.ink;
      ctx.strokeStyle = THEME.outline;
      ctx.lineWidth = 4;
      ctx.lineJoin = 'round';
      ctx.font = `bold 15px ${THEME.font}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.strokeText(label, px, py + 18);
      ctx.fillText(label, px, py + 18);
    }
  }

  // --- labels: placed so that none covers x0, its label or another label ----
  private obstacles: Rect[] = [];
  private placed: Rect[] = [];

  private measure(text: [string, string, string], size: number): number {
    const ctx = this.ctx;
    ctx.font = `${size}px ${THEME.font}`;
    const wb = ctx.measureText(text[0]).width + ctx.measureText(text[2]).width;
    ctx.font = `${Math.round(size * 0.68)}px ${THEME.font}`;
    return wb + (text[1] ? ctx.measureText(text[1]).width + 1 : 0);
  }

  private inSquare(r: Rect): boolean {
    return r.x >= this.sq.x + 2 && r.y >= this.sq.y + 2 && r.x + r.w <= this.sq.x + this.sq.side - 2 && r.y + r.h <= this.sq.y + this.sq.side - 2;
  }

  /** First candidate that collides with nothing; else the first that avoids x0 and other labels; else null. */
  private place(cands: Rect[], soft: Rect[] = this.obstacles): Rect | null {
    const ok = cands.filter((r) => this.inSquare(r) && !this.placed.some((p) => overlaps(p, r)));
    const best = ok.find((r) => !soft.some((o) => overlaps(o, r))) ?? ok[0] ?? null;
    if (best) this.placed.push(best);
    return best;
  }

  /** Boxed label next to an observation (four candidate corners). */
  private boxLabel(text: [string, string, string], px: number, py: number, strong = false): boolean {
    const size = 15;
    const w = this.measure(text, size) + 14;
    const h = 22;
    const cands: Rect[] = [
      { x: px + 13, y: py - 31, w, h },
      { x: px - 13 - w, y: py - 31, w, h },
      { x: px + 13, y: py + 9, w, h },
      { x: px - 13 - w, y: py + 9, w, h },
    ];
    const r = this.place(cands);
    if (!r) return false;
    const ctx = this.ctx;
    ctx.fillStyle = THEME.label;
    ctx.beginPath();
    ctx.roundRect(r.x, r.y, r.w, r.h, 5);
    ctx.fill();
    if (strong) {
      ctx.strokeStyle = THEME.ring;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    ctx.fillStyle = THEME.ink;
    drawSubText(ctx, text[0], text[1], text[2], r.x + 7, r.y + 16, size);
    return true;
  }

  private drawOverlays(s: State): void {
    const ctx = this.ctx;
    const pred = prediction(s);
    const VS = valueScale(s);
    const [qx, qy] = this.toPx(s.x0);
    const sc = this.sq.side / (DOMAIN.x1 - DOMAIN.x0);
    const n = s.X.length;
    let wmax = 0;
    for (let i = 0; i < n; i++) wmax = Math.max(wmax, pred.weights[i]);
    const nw = pred.kind === 'nw';
    const emph = s.emphasis === 'radius';

    ctx.save();
    ctx.beginPath();
    ctx.rect(this.sq.x, this.sq.y, this.sq.side, this.sq.side);
    ctx.clip();
    // R_k(x0): the exact neighbourhood of kNN, drawn thin and neutral. NW has no
    // such circle: a Gaussian kernel has no radius beyond which weights vanish.
    if (pred.kind === 'knn') {
      const r = pred.radius * sc;
      ctx.setLineDash(emph ? [9, 5] : [5, 6]);
      ctx.strokeStyle = THEME.outline;
      ctx.globalAlpha = emph ? 0.9 : 0.55;
      ctx.lineWidth = emph ? 5 : 3;
      ctx.beginPath();
      ctx.arc(qx, qy, r, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.strokeStyle = THEME.ring;
      ctx.globalAlpha = emph ? 1 : 0.5;
      ctx.lineWidth = emph ? 2.2 : 1.2;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }
    // Influence lines from x0 to every observation with positive weight
    // (NW: all of them; opacity and width follow the weight).
    for (let i = 0; i < n; i++) {
      const w = pred.weights[i];
      if (w <= 0 || (pred.kind==='knn' && pred.k===n)) continue;
      const rel = w / wmax;
      const [px, py] = this.toPx(s.X[i]);
      ctx.strokeStyle = THEME.outline;
      ctx.globalAlpha = nw ? 0.5 * rel : 0.8;
      ctx.lineWidth = (nw ? 1 + 3.5 * rel : 3) + 3;
      ctx.beginPath();
      ctx.moveTo(qx, qy);
      ctx.lineTo(px, py);
      ctx.stroke();
      ctx.strokeStyle = THEME.accent;
      ctx.globalAlpha = nw ? 0.1 + 0.9 * rel : 1;
      ctx.lineWidth = nw ? 0.8 + 3.7 * rel : 3;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.restore();

    // Observations: the data (X_i, Y_i), filled with the colour of Y_i.
    this.placed = [{ x: qx - 26, y: qy - 26, w: 52, h: 52 }];
    for (let i = 0; i < n; i++) {
      const [px, py] = this.toPx(s.X[i]);
      const active = (s.drag?.kind === 'point' && s.drag.index === i) ||
        (this.hoverTarget?.kind === 'point' && this.hoverTarget.index === i);
      // A solid ink centre stays visible even where the map equals Y_i.
      // The narrow response-coloured edge retains the site's value encoding.
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, 2 * Math.PI);
      ctx.fillStyle = THEME.ink;
      ctx.fill();
      ctx.strokeStyle = active || s.highlightObs === i ? THEME.accent : THEME.outline;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.strokeStyle = colorCss(s.Y[i], VS);
      ctx.lineWidth = 1;
      ctx.stroke();
      if (s.includeOutlier && i === OUTLIER) {
        ctx.strokeRect(px - 6, py - 6, 12, 12);
      }
      this.obstacles.push({ x: px - 11, y: py - 11, w: 22, h: 22 });
    }

    // x0's label is placed first (highest priority) on the side away from its
    // neighbours, and drawn last; no other label may cover it.
    let vx = 0;
    let vy = 0;
    for (let i = 0; i < n; i++) {
      const w = pred.weights[i];
      if (w <= 0) continue;
      const [px, py] = this.toPx(s.X[i]);
      const L = Math.hypot(px - qx, py - qy) || 1;
      vx += (w * (px - qx)) / L;
      vy += (w * (py - qy)) / L;
    }
    ctx.font = `bold 19px ${THEME.font}`;
    const lw = ctx.measureText('x₀').width + 4;
    const lh = 22;
    const x0Cands = [
      { dx: -1, dy: -1 },
      { dx: 1, dy: -1 },
      { dx: -1, dy: 1 },
      { dx: 1, dy: 1 },
    ]
      .sort((a, b) => a.dx * vx + a.dy * vy - (b.dx * vx + b.dy * vy))
      .map(({ dx, dy }) => ({ x: dx < 0 ? qx - 17 - lw : qx + 17, y: dy < 0 ? qy - 17 - lh : qy + 17, w: lw, h: lh }));
    const x0Rect = this.place(x0Cands) ?? x0Cands[0];

    // Feedback next to the hand while an observation is dragged (Edit data).
    if (s.drag?.kind === 'point') {
      const i = s.drag.index!;
      const [px, py] = this.toPx(s.X[i]);
      this.boxLabel(['X', String(i + 1), ` moves; Y${sub(i + 1)} = ${fmt(s.Y[i], 1)} fixed`], px, py, true);
    }
    // The moving response of the one-response experiment.
    if (s.highlightObs !== null) {
      const i = s.highlightObs;
      const [px, py] = this.toPx(s.X[i]);
      this.boxLabel(['Y', String(i + 1), ` = ${fmt(s.Y[i], 2)}`], px, py, true);
    }
    // Responses of the selected neighbours, or the largest NW weights.
    if (pred.kind === 'knn') {
      if (pred.k <= 6) {
        for (const i of pred.neighbors.slice(0,1)) {
          if (i === s.highlightObs) continue;
          const [px, py] = this.toPx(s.X[i]);
          this.boxLabel(['Y', String(i + 1), ` = ${fmt(s.Y[i], 1)}`], px, py);
        }
      }
    } else {
      const top = [...pred.order].sort((a, b) => pred.weights[b] - pred.weights[a]).slice(0, 1);
      for (const i of top) {
        if (i === s.highlightObs) continue;
        const [px, py] = this.toPx(s.X[i]);
        this.boxLabel(['w', String(i + 1), ` ${relW(pred.weights[i])}`], px, py);
      }
    }
    // R_k label: lowest priority; shown only where it covers nothing.
    if (pred.kind === 'knn') {
      const r = pred.radius * sc;
      const text: [string, string, string] = ['R', String(pred.k), '(x₀)'];
      const size = emph ? 17 : 15;
      const w = this.measure(text, size) + 4;
      const h = size + 6;
      const cands: Rect[] = [];
      for (const a of [-Math.PI / 4, (-3 * Math.PI) / 4, Math.PI / 4, (3 * Math.PI) / 4, -Math.PI / 2, Math.PI / 2]) {
        const cx = qx + (r + 4) * Math.cos(a);
        const cy = qy + (r + 4) * Math.sin(a);
        cands.push({ x: Math.cos(a) >= -1e-9 ? cx + 2 : cx - w - 2, y: Math.sin(a) >= -1e-9 ? cy + 2 : cy - h - 2, w, h });
      }
      const rr = this.place(cands);
      if (rr && (emph || !this.obstacles.some((o) => overlaps(o, rr)))) {
        ctx.fillStyle = emph ? THEME.ink : THEME.inkMuted;
        ctx.strokeStyle = THEME.outline;
        drawSubText(ctx, text[0], text[1], text[2], rr.x + 2, rr.y + size, size, 'left', 4);
      }
    }

    this.drawQuery(qx, qy, colorCss(pred.value, VS), x0Rect);
  }

  private drawQuery(qx: number, qy: number, fill: string, label: Rect): void {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(qx - 8, qy);
    ctx.lineTo(qx + 8, qy);
    ctx.moveTo(qx, qy - 8);
    ctx.lineTo(qx, qy + 8);
    ctx.lineCap = 'round';
    ctx.strokeStyle = THEME.outline;
    ctx.lineWidth = 4;
    ctx.stroke();
    ctx.strokeStyle = THEME.accent;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = fill;
    ctx.fillRect(qx - 1, qy - 1, 2, 2);
    ctx.fillStyle = THEME.accent;
    ctx.strokeStyle = THEME.outline;
    ctx.lineWidth = 5;
    ctx.lineJoin = 'round';
    ctx.font = `bold 19px ${THEME.font}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.strokeText('x₀', label.x + 2, label.y + 17);
    ctx.fillText('x₀', label.x + 2, label.y + 17);
  }

  /** Test hook: rectangles of the labels placed in the last frame (x0's label first). */
  debugLabels(): Rect[] {
    return this.placed.slice(1);
  }
}
