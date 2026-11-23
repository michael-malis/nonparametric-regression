// Lower panel: the cross-section g(s) = f̂(γ(s)) over the whole path, with a
// cursor at the current s. Step profiles are drawn as horizontal pieces only;
// a jump is marked by a faint dotted vertical, never by a sloped line.

import katex from 'katex';
import { colorCss } from '../math/colormap';
import type { Vec2 } from '../math/estimators';
import { type Profile, pointOnPath } from '../path/profile';
import { ghostProfile, profile, profileValueAt, valueScale } from '../state/model';
import type { Key, State, Store } from '../state/store';
import { THEME, fmt } from './theme';

export class ProfileChart {
  readonly el: HTMLElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private legend: HTMLElement;
  private dpr = 1;
  private w = 0;
  private h = 0;
  private scrubbing: number | null = null;
  private pad = { l: 44, r: 16, t: 10, b: 24 };

  constructor(
    parent: HTMLElement,
    private store: Store,
  ) {
    this.el = document.createElement('section');
    this.el.className = 'drawer';
    this.el.setAttribute('aria-label', 'Profile of the prediction along the path');
    this.el.innerHTML = `<header><span class="ptitle"></span><span class="plegend"></span></header><canvas data-testid="profile-canvas"></canvas>`;
    parent.appendChild(this.el);
    katex.render(String.raw`g(s)=\hat f\big(\gamma(s)\big)`, this.el.querySelector('.ptitle') as HTMLElement, { throwOnError: false });
    this.legend = this.el.querySelector('.plegend')!;
    this.canvas = this.el.querySelector('canvas')!;
    this.ctx = this.canvas.getContext('2d')!;
    new ResizeObserver(() => this.resize()).observe(this.canvas);
    this.canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || document.body.dataset.mode !== 'explore') return;
      this.scrubbing = e.pointerId;
      this.canvas.setPointerCapture(e.pointerId);
      this.scrub(e);
    });
    this.canvas.addEventListener('pointermove', (e) => {
      if (this.scrubbing === e.pointerId) this.scrub(e);
    });
    const stop = (e: PointerEvent) => {
      if (this.scrubbing === e.pointerId) this.scrubbing = null;
    };
    this.canvas.addEventListener('pointerup', stop);
    this.canvas.addEventListener('pointercancel', stop);
  }

  private scrub(e: PointerEvent): void {
    const r = this.canvas.getBoundingClientRect();
    const t = Math.min(1, Math.max(0, (e.clientX - r.left - this.pad.l) / (this.w - this.pad.l - this.pad.r)));
    const s = this.store.get();
    // Scrubbing is a manual change of x0: it ends a running experiment.
    this.store.set({ s: t, x0: pointOnPath(s.pathA, s.pathB, t), onPath: true, playing: false, experiment: null });
  }

  private resize(): void {
    const r = this.canvas.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = r.width;
    this.h = r.height;
    this.canvas.width = Math.round(r.width * this.dpr);
    this.canvas.height = Math.round(r.height * this.dpr);
    this.draw();
  }

  update(s: State, _changed: Set<Key>): void {
    if (s.profileOpen) this.draw();
  }

  private X(t: number): number {
    return this.pad.l + t * (this.w - this.pad.l - this.pad.r);
  }

  private VS = { lo: 0, hi: 10 };

  private Y(v: number): number {
    const u = (v - this.VS.lo) / (this.VS.hi - this.VS.lo);
    return this.h - this.pad.b - u * (this.h - this.pad.t - this.pad.b);
  }

  private drawProfile(p: Profile, ghost: boolean): void {
    const ctx = this.ctx;
    if (p.kind === 'step') {
      p.segments.forEach((g, i) => {
        const y = this.Y(g.value);
        if (ghost) {
          ctx.strokeStyle = 'rgba(74,12,28,0.38)';
          ctx.lineWidth = 1.6;
          ctx.setLineDash([5, 4]);
        } else {
          ctx.setLineDash([]);
          ctx.strokeStyle = THEME.ink;
          ctx.lineWidth = 8;
          ctx.beginPath();
          ctx.moveTo(this.X(g.s0), y);
          ctx.lineTo(this.X(g.s1), y);
          ctx.stroke();
          ctx.strokeStyle = colorCss(g.value, this.VS);
          ctx.lineWidth = 4.5;
        }
        ctx.lineCap = 'butt';
        ctx.beginPath();
        ctx.moveTo(this.X(g.s0), y);
        ctx.lineTo(this.X(g.s1), y);
        ctx.stroke();
        const nx = p.segments[i + 1];
        if (nx && !ghost) {
          ctx.setLineDash([2, 4]);
          ctx.strokeStyle = 'rgba(74,12,28,0.35)';
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(this.X(g.s1), y);
          ctx.lineTo(this.X(g.s1), this.Y(nx.value));
          ctx.stroke();
        }
      });
      ctx.setLineDash([]);
      return;
    }
    if (ghost) return;
    ctx.lineJoin = 'round';
    ctx.strokeStyle = THEME.ink;
    ctx.lineWidth = 8;
    ctx.beginPath();
    for (let i = 0; i < p.s.length; i++) {
      const x = this.X(p.s[i]);
      const y = this.Y(p.v[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.lineWidth = 4.5;
    ctx.lineCap = 'round';
    // Colour each short piece by its own value.
    const stride = Math.max(1, Math.floor(p.s.length / 600));
    for (let i = 0; i < p.s.length - 1; i += stride) {
      const j = Math.min(p.s.length - 1, i + stride);
      ctx.strokeStyle = colorCss((p.v[i] + p.v[j]) / 2, this.VS);
      ctx.beginPath();
      ctx.moveTo(this.X(p.s[i]), this.Y(p.v[i]));
      for (let m = i + 1; m <= j; m++) ctx.lineTo(this.X(p.s[m]), this.Y(p.v[m]));
      ctx.stroke();
    }
  }

  draw(): void {
    const s = this.store.get();
    if (!s.profileOpen || this.w < 2) return;
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.font = `12px ${THEME.font}`;
    this.VS = valueScale(s);
    const tickStep = this.VS.hi - this.VS.lo > 6 ? 2 : 1;
    // Value grid on the dataset's fixed colour domain (same as the legend).
    for (let v = Math.ceil(this.VS.lo); v <= this.VS.hi + 1e-9; v += 1) {
      const y = this.Y(v);
      ctx.strokeStyle = 'rgba(74,12,28,0.09)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(this.pad.l, y);
      ctx.lineTo(this.w - this.pad.r, y);
      ctx.stroke();
      if ((v - Math.ceil(this.VS.lo)) % tickStep !== 0) continue;
      ctx.fillStyle = THEME.inkMuted;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(v), this.pad.l - 8, y);
    }
    ctx.textBaseline = 'top';
    ctx.font = `12px ${THEME.font}`;
    for (const t of [0.25, 0.5, 0.75]) {
      ctx.fillStyle = THEME.inkMuted;
      ctx.textAlign = 'center';
      ctx.fillText(String(t), this.X(t), this.h - this.pad.b + 6);
    }
    ctx.fillStyle = THEME.ink;
    ctx.font = `bold 13px ${THEME.font}`;
    ctx.textAlign = 'left';
    ctx.fillText('A', this.X(0), this.h - this.pad.b + 6);
    ctx.textAlign = 'right';
    ctx.fillText('B', this.X(1), this.h - this.pad.b + 6);

    const ghost = ghostProfile(s);
    if (ghost) this.drawProfile(ghost.profile, true);
    this.drawProfile(profile(s), false);
    this.legend.innerHTML = ghost ? `<span class="ghost-key"></span>${ghost.label}, same path` : '';

    // Cursor at the current s, value evaluated exactly at γ(s). The dot is x0's accent.
    if (s.onPath) {
      const x = this.X(s.s);
      const v = profileValueAt(s, s.s);
      const y = this.Y(v);
      ctx.strokeStyle = 'rgba(74,12,28,0.45)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, this.pad.t);
      ctx.lineTo(x, this.h - this.pad.b);
      ctx.stroke();
      ctx.fillStyle = THEME.outline;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = THEME.accent;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
      ctx.fill();
      const label = `g(${fmt(s.s, 2)}) = ${fmt(v, 2)}`;
      ctx.font = `bold 14px ${THEME.font}`;
      const tw = ctx.measureText(label).width;
      const lx = x + 14 + tw > this.w - this.pad.r ? x - 14 - tw : x + 14;
      const ly = Math.max(this.pad.t + 2, Math.min(y - 24, this.h - this.pad.b - 22));
      ctx.fillStyle = THEME.label;
      ctx.beginPath();
      ctx.roundRect(lx - 6, ly - 2, tw + 12, 20, 5);
      ctx.fill();
      ctx.fillStyle = THEME.ink;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(label, lx, ly + 2);
    } else {
      ctx.fillStyle = THEME.inkMuted;
      ctx.font = `13px ${THEME.font}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText('x₀ is off the path: click here to put it on γ(s).', (this.pad.l + this.w - this.pad.r) / 2, this.pad.t + 2);
    }
  }
}

export type { Vec2 };
