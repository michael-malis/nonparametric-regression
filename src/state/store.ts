// Single application state. Updates are batched into one animation frame.

import { PRESETS, type Preset } from '../data/presets';
import type { ExperimentId } from '../experiments/plans';
import type { ValueScale } from '../math/colormap';
import type { Vec2 } from '../math/estimators';

export type Method = '1nn' | 'knn' | 'nw';
export type View = '2d' | '3d';

export const METHODS: Method[] = ['1nn', 'knn', 'nw'];
export const METHOD_LABEL: Record<Method, string> = { '1nn': '1NN', knn: 'kNN', nw: 'Nadaraya–Watson' };

export interface State {
  presetIndex: number;
  includeOutlier: boolean;
  showRegions: boolean;
  X: Vec2[];
  Y: number[];
  /** Bumped whenever X or Y changes. */
  dataVersion: number;
  x0: Vec2;
  method: Method;
  view: View;
  /** k used on the kNN stage (1NN always uses k = 1). */
  kKnn: number;
  h: number;
  pathA: Vec2;
  pathB: Vec2;
  profileOpen: boolean;
  playing: boolean;
  /** Path parameter of the playback cursor. */
  s: number;
  /** True while x0 = γ(s). */
  onPath: boolean;
  speed: number;
  uiHidden: boolean;
  helpOpen: boolean;
  /** What the pointer is dragging right now, if anything. */
  drag: null | { kind: 'x0' | 'point' | 'pathA' | 'pathB'; index?: number };
  /** Active pedagogical experiment, if any. */
  experiment: ExpRun | null;
  /** Fixed colour domain that overrides the dataset's (the one-response experiment). */
  scaleOverride: ValueScale | null;
  /** Observations X_i can be dragged only in this explicit mode. */
  editData: boolean;
  /** Observation whose footprint is outlined (one-response experiment). */
  highlightObs: number | null;
  /** Briefly emphasised helper geometry at an experiment stop. */
  emphasis: null | 'radius';
  /** NW: soft kernel halo K(|x - x0|/h) pinned on by the lecturer (key K). */
  haloPinned: boolean;
  /** NW: halo shown while h is being changed. */
  haloActive: boolean;
}

/**
 * A running pedagogical experiment. While it is set, the runner is the only
 * writer of its parameter (x0 via s, one Y_j, h or k); any manual change of
 * x0, X, k or h sets it back to null.
 */
export interface ExpRun {
  id: ExperimentId;
  /** Index of the last stop reached. */
  stop: number;
  /** Stop being moved to (equals `stop` when standing at a stop). */
  target: number;
  phase: 'paused' | 'running' | 'done';
}

export type Key = keyof State;
type Listener = (s: State, changed: Set<Key>) => void;

export function presetState(p: Preset, index: number): Partial<State> {
  return {
    presetIndex: index,
    includeOutlier: index % 2 === 1,
    X: p.X.map((x) => [x[0], x[1]] as Vec2),
    Y: [...p.Y],
    x0: [p.x0[0], p.x0[1]],
    kKnn: p.k,
    h: p.h,
    pathA: [p.pathA[0], p.pathA[1]],
    pathB: [p.pathB[0], p.pathB[1]],
    playing: false,
    s: 0,
    onPath: false,
    experiment: null,
    scaleOverride: null,
    highlightObs: null,
    emphasis: null,
  };
}

export class Store {
  private state: State;
  private listeners: Listener[] = [];
  private pending = new Set<Key>();
  private scheduled = false;

  constructor() {
    this.state = {
      ...(presetState(PRESETS[0], 0) as State),
      dataVersion: 0,
      showRegions: false,
      method: '1nn',
      view: '2d',
      profileOpen: false,
      speed: 1,
      uiHidden: false,
      helpOpen: false,
      drag: null,
      editData: false,
      haloPinned: false,
      haloActive: false,
    };
  }

  get(): State {
    return this.state;
  }

  set(patch: Partial<State>): void {
    const next = { ...this.state, ...patch };
    for (const k of Object.keys(patch) as Key[]) this.pending.add(k);
    if ('X' in patch || 'Y' in patch) {
      next.dataVersion = this.state.dataVersion + 1;
      this.pending.add('dataVersion');
    }
    this.state = next;
    this.schedule();
  }

  subscribe(fn: Listener): void {
    this.listeners.push(fn);
  }

  /** Run all listeners now with every key marked as changed. */
  flushAll(): void {
    const all = new Set(Object.keys(this.state) as Key[]);
    for (const fn of this.listeners) fn(this.state, all);
  }

  private schedule(): void {
    if (this.scheduled) return;
    this.scheduled = true;
    requestAnimationFrame(() => this.flush());
  }

  /** Deliver pending changes synchronously (used by the frame loop and tests). */
  flush(): void {
    this.scheduled = false;
    if (!this.pending.size) return;
    const changed = this.pending;
    this.pending = new Set();
    for (const fn of this.listeners) fn(this.state, changed);
  }
}
