// Compact overlays: stage track, formula, live computation, prediction readout
// (top left) and the colour legend with the current weights (top right).

import katex from 'katex';
import { LUT, LUT_SIZE, colorCss } from '../math/colormap';
import type { RunnerInfo } from '../experiments/runner';
import { type AnyPrediction, prediction, valueScale } from '../state/model';
import { type Method, type State, type View } from '../state/store';
import { THEME, fmt, fmtW, sub } from './theme';

const FORMULA: Record<Method, string> = {
  '1nn': String.raw`\begin{gathered}\hat f_1(x)=Y_{j^*}\\j^*=\arg\min_i\lVert x-X_i\rVert\end{gathered}`,
  knn: String.raw`\hat f_k(x)=\frac1k\sum_{i\in N_k(x)}Y_i`,
  nw: String.raw`\begin{gathered}\hat f_h(x)=\frac{\sum_i a_iY_i}{\sum_j a_j}\\a_i=e^{-\lVert x-X_i\rVert^2/(2h^2)}\end{gathered}`,
};

const HINT: Record<`${Method}-${View}`, string> = {
  '1nn-2d': 'Move x₀ across a border to change the nearest site. Drag a dot to move its input while keeping its response.',
  '1nn-3d': 'Each plateau has the height of its nearest response. Drag a dot vertically to change its response, or drag empty space to rotate.',
  'knn-2d': 'Move x₀ or a dot. The dashed circle reaches the kth neighbour. Increasing k averages more responses and reduces noise variance to σ²/k under independent, equal-variance errors.',
  'knn-3d': 'Drag a dot vertically. Each plateau that includes it changes by 1/k of the response change.',
  'nw-2d': 'Move x₀ or increase h. Each Gaussian weight is positive. A larger bandwidth gives distant sites more influence relative to the nearest site.',
  'nw-3d': 'Drag a dot vertically, then increase h. The response influences a wider area as the Gaussian weights spread.',
};

const lastTex = new WeakMap<HTMLElement,string>();
function tex(el: HTMLElement, src: string, display = false): void {
  if (lastTex.get(el) === src) return;
  lastTex.set(el, src);
  katex.render(src, el, { throwOnError: false, displayMode: display, output: 'htmlAndMathml' });
}

export class Hud {
  private root: HTMLElement;
  private title: HTMLElement;
  private formula: HTMLElement;
  private calcLine: HTMLElement;
  private calcNote: HTMLElement;
  private readout: HTMLElement;
  private coords: HTMLElement;
  private hint: HTMLElement;
  private side: HTMLElement;
  private legendCanvas: HTMLCanvasElement;
  private legendMarker: HTMLElement;
  private weights: HTMLElement;
  private lastFormula = '';
  private details: HTMLDetailsElement;
  private currentState: State | null = null;

  constructor(
    stage: HTMLElement,
    private experimentInfo: (s: State) => RunnerInfo | null = () => null,
  ) {
    this.root = document.createElement('section');
    this.root.className = 'hud';
    this.root.innerHTML = `
      <h1></h1>
      <div class="readout" data-testid="readout"><span class="swatch"></span><span class="value"></span></div>
      <div class="coords" data-testid="coords"></div>
      <p class="hint" data-testid="hint"></p>
      <details class="meaning"><summary>Why these weights?</summary><p></p><p class="moments"></p></details>
      <details class="calculation"><summary>Calculation &amp; weights</summary>
      <div class="formula" data-testid="formula"></div>
      <div class="calc" data-testid="calc"><div class="calc-line"></div><div class="calc-note"></div></div>
      <div class="weights" data-testid="weights"></div></details>`;
    stage.appendChild(this.root);
    this.details = this.root.querySelector('.calculation')!;
    this.details.addEventListener('toggle',()=>{
      if(this.details.open && this.currentState)this.updateDetails(this.currentState, prediction(this.currentState));
    });
    this.title = this.root.querySelector('h1')!;
    this.formula = this.root.querySelector('.formula')!;
    this.calcLine = this.root.querySelector('.calc-line')!;
    this.calcNote = this.root.querySelector('.calc-note')!;
    this.readout = this.root.querySelector('.readout')!;
    this.coords = this.root.querySelector('.coords')!;
    this.hint = this.root.querySelector('.hint')!;

    this.side = document.createElement('aside');
    this.side.className = 'side';
    this.side.innerHTML = `
      <div class="legend" aria-label="Colour and height scale of the prediction">
        <div class="legend-title"></div>
        <div class="legend-body">
          <canvas width="22" height="240"></canvas>
          <div class="legend-ticks"></div>
          <div class="legend-marker"><span></span></div>
        </div>
        <div class="legend-note" data-testid="legend-note"></div>
      </div>`;
    stage.parentElement!.appendChild(this.side);
    tex(this.side.querySelector('.legend-title')!, String.raw`\hat f`);
    this.legendCanvas = this.side.querySelector('canvas')!;
    this.legendMarker = this.side.querySelector('.legend-marker')!;
    this.weights = this.root.querySelector('.weights')!;
    this.drawLegendBar();
  }

  private legendKey = '';

  private drawLegendBar(): void {
    const c = this.legendCanvas;
    const g = c.getContext('2d')!;
    const img = g.createImageData(c.width, c.height);
    for (let y = 0; y < c.height; y++) {
      const t = 1 - y / (c.height - 1);
      const i = Math.round(t * (LUT_SIZE - 1)) * 4;
      for (let x = 0; x < c.width; x++) {
        const o = 4 * (y * c.width + x);
        img.data[o] = LUT[i];
        img.data[o + 1] = LUT[i + 1];
        img.data[o + 2] = LUT[i + 2];
        img.data[o + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
  }

  /** Tick labels of the dataset's colour domain (rebuilt only when the dataset changes). */
  private drawLegendTicks(VS: { lo: number; hi: number }): void {
    const key = `${VS.lo}|${VS.hi}`;
    if (key === this.legendKey) return;
    this.legendKey = key;
    const ticks = this.side.querySelector('.legend-ticks')!;
    ticks.innerHTML = '';
    for (let v = Math.ceil(VS.lo); v <= VS.hi + 1e-9; v += 1) {
      const s = document.createElement('span');
      s.textContent = String(v);
      s.style.top = `${(1 - (v - VS.lo) / (VS.hi - VS.lo)) * 100}%`;
      ticks.appendChild(s);
    }
  }

  update(s: State): void {
    this.currentState = s;
    const pred = prediction(s);
    const VS = valueScale(s);
    this.drawLegendTicks(VS);
    const key = `${s.method}`;
    if (key !== this.lastFormula) {
      this.lastFormula = key;
      this.root.querySelector('.meaning p')!.textContent = s.method === 'nw'
        ? 'Gaussian scores decay with distance on the scale h. Dividing by their sum gives weights that add to one. The prediction is a weighted average of the responses.'
        : s.method === '1nn'
        ? 'The nearest input has weight one. Every other weight is zero. A Voronoi edge marks a tie in distance. Crossing it changes the selected site, even when the two responses happen to agree.'
        : 'The nearest k inputs each have weight 1/k. For an L-Lipschitz regression function, the magnitude of conditional bias is at most L × Rₖ. Increasing k can enlarge this bound. The actual bias can shrink when signed deviations cancel.';
      tex(this.formula, FORMULA[s.method], true);
    }
    this.title.textContent = s.method === "1nn" ? "Nearest neighbour" : s.method === "knn" ? "Neighbour average" : "Kernel average";

    const valueEl = this.readout.querySelector('.value') as HTMLElement;
    const sym = s.method === '1nn' ? String.raw`\hat f_1` : s.method === 'knn' ? String.raw`\hat f_{${pred.kind === 'knn' ? pred.k : ''}}` : String.raw`\hat f_h`;
    tex(valueEl, String.raw`${sym}(x_0)=${fmt(pred.value, 2)}`);
    (this.readout.querySelector('.swatch') as HTMLElement).style.background = colorCss(pred.value, VS);
    tex(this.coords, String.raw`x_0=(${fmt(s.x0[0], 2)},\ ${fmt(s.x0[1], 2)})` + (s.method === 'nw' ? String.raw`,\quad h=${fmt(s.h, 2)}` : s.method === 'knn' ? String.raw`,\quad k=${pred.kind === 'knn' ? pred.k : ''}` : ''));

    let hint = HINT[`${s.method}-${s.view}`];
    if (s.drag?.kind === 'point') hint = `Moving X${sub(s.drag.index! + 1)} changes only its position; Y${sub(s.drag.index! + 1)} = ${fmt(s.Y[s.drag.index!], 1)} stays.`;
    else if (s.drag?.kind === 'pathA' || s.drag?.kind === 'pathB') hint = 'Moving an end of the path γ recomputes the whole slice g(s).';
    else if (s.editData) hint = 'Edit data: drag an observation Xᵢ to a new place; its response Yᵢ stays. U undoes, E leaves Edit data.';
    else if (s.playing) hint = 'x₀ = γ(s) moves along the path; the profile shows g(s) = f̂(γ(s)).';
    const info = this.experimentInfo(s);
    if (!info && document.body.dataset.mode === 'demo') hint = s.method==='1nn' ? 'Each dot marks an input Xᵢ. Its region is coloured by the response Yᵢ. At the cross x₀, read the prediction from the nearest site.' : s.method==='knn' ? 'Each plateau averages the same set of neighbours.' : 'Closer observations carry more weight.';
    this.hint.classList.toggle('exp', !!info);
    if (info) {
      this.hint.replaceChildren();
      const body = document.createElement('span');
      body.className = 'exp-body';
      body.textContent = info.caption;
      this.hint.append(body);
    } else {
      this.hint.textContent = hint;
    }

    const t = (pred.value - VS.lo) / (VS.hi - VS.lo);
    this.legendMarker.style.top = `${(1 - Math.min(1, Math.max(0, t))) * 100}%`;
    (this.legendMarker.querySelector('span') as HTMLElement).textContent = fmt(pred.value, 2);
    (this.side.querySelector('.legend-note') as HTMLElement).textContent = s.scaleOverride
      ? `Colour scale fixed at [${fmt(VS.lo, 1)}, ${fmt(VS.hi, 1)}]`
      : '';

    const factor = pred.weights.reduce((sum,w) => sum+w*w,0);
    this.root.querySelector('.moments')!.textContent = `Fix the inputs and imagine drawing new responses with independent, mean-zero errors of variance σ². The prediction variance is σ² Σᵢ wᵢ² ≈ ${factor.toFixed(3)}σ². Conditional MSE for estimating f(x₀) adds squared bias, where bias is Σᵢ wᵢ f(Xᵢ) − f(x₀).`;
    if (this.details.open) this.updateDetails(s, pred);
  }

  private updateDetails(s: State, pred: AnyPrediction): void {
    const [line, note] = this.calcTex(s, pred);
    tex(this.calcLine, line);
    this.calcNote.textContent = [note, s.emphasis==='radius'?'At equal distances, the smaller site index wins.':''].filter(Boolean).join(' ');
    this.renderWeights(s, pred);
  }

  /** One KaTeX line with the live computation, plus an optional plain note. */
  private calcTex(s: State, pred: AnyPrediction): [string, string] {
    if (pred.kind === 'knn') {
      const sum = pred.neighbors.reduce((total,i)=>total+s.Y[i],0);
      const list = pred.neighbors.map(i=>i+1).join(', ');
      return [String.raw`\frac{${fmt(sum,1)}}{${pred.k}}\approx ${fmt(pred.value,2)}`,
        `Nearest sites: ${list}.`];
    }
    const j = pred.weights.indexOf(Math.max(...pred.weights));
    const local = pred.weights[j] * s.Y[j];
    return [String.raw`\hat f_h(x_0)\approx ${fmt(pred.value,2)}`,
      `Contribution from the largest weight: w${sub(j+1)}Y${sub(j+1)} ≈ ${fmt(local,3)}. Other sites together contribute ≈ ${fmt(pred.value-local,3)}.`];
  }

  private renderWeights(s: State, pred: AnyPrediction): void {
    const n = s.X.length;
    const VS = valueScale(s);
    const dot = (i: number) => `<span class="wi"><span class="dot" style="background:${colorCss(s.Y[i], VS)}"></span>${i + 1}</span>`;
    if (pred.kind === 'knn') {
      // Neighbours in order of distance, then the first outsider: a change of
      // order (rows swap, f̂ stays) is visibly different from a change of set.
      const shown = pred.order.slice(0, Math.min(pred.k, 8));
      const next = pred.k < n ? pred.order[pred.k] : null;
      const row = (i: number, out: boolean) => `<div class="wrow knn${out ? ' out' : ''}">
          ${dot(i)}
          <span class="wy">${fmt(s.Y[i], 1)}</span>
          <span class="wd">${fmt(Math.sqrt(pred.d2[i]), 2)}</span>
          <span class="ww">${out ? '0' : fracLabel(pred.k)}</span>
        </div>`;
      const more = pred.k - shown.length;
      const footer = `${more > 0 ? `<div class="wfoot">${more} more at 1/${pred.k}</div>` : ''}<div class="wfoot">${n - pred.k === 0 ? 'No' : n - pred.k} other${n - pred.k === 1 ? '' : 's'} at w = 0; Σw = 1</div>`;
      this.weights.innerHTML = `
        <div class="whead knn"><span>i</span><span>Y<sub>i</sub></span><span>‖x<sub>0</sub>−X<sub>i</sub>‖</span><span>w<sub>i</sub></span></div>
        ${shown.map((i) => row(i, false)).join('')}${next !== null ? `<div class="wsep">next outside R<sub>${pred.k}</sub>:</div>${row(next, true)}` : ''}${footer}`;
      return;
    }
    const order = [...Array(n).keys()].sort((a, b) => pred.weights[b] - pred.weights[a] || a - b);
    const shown = order.slice(0, 6);
    const rest = order.slice(shown.length);
    const restW = rest.reduce((a, i) => a + pred.weights[i], 0);
    const wmax = pred.weights[order[0]];
    const rows = shown
      .map((i) => {
        const w = pred.weights[i];
        return `<div class="wrow">
          ${dot(i)}
          <span class="wy">${fmt(s.Y[i], 1)}</span>
          <span class="wbar" title="relative weight wᵢ / max w"><span style="width:${(100 * w) / Math.max(wmax, 1e-300)}%"></span></span>
          <span class="ww">${fmtW(w)}</span>
        </div>`;
      })
      .join('');
    this.weights.innerHTML = `
      <div class="whead"><span>i</span><span>Y<sub>i</sub></span><span>relative</span><span>w<sub>i</sub>(x<sub>0</sub>)</span></div>
      ${rows}<div class="wfoot">${rest.length} more: Σw ${restW < 0.0005 ? '< 0.001' : `= ${fmtW(restW)}`}</div><div class="wfoot">All ${n} weights &gt; 0; Σw = 1</div>`;
  }
}

function fracLabel(k: number): string {
  return k === 1 ? '1' : `1/${k}`;
}

export { sub };
export { THEME };
