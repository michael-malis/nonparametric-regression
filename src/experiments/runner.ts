// Runs one experiment plan against the store.
//
// While an experiment is running, the runner is the only writer of its
// parameter. Motion follows wall-clock time (dt in seconds, clamped by the
// frame loop), and a motion that would pass a stop is cut exactly at it, so a
// long frame can never skip an event. Discrete parameters (k) jump; nothing
// that is discontinuous in the estimator is ever interpolated.

import { pointOnPath } from '../path/profile';
import type { ExpRun, State, Store } from '../state/store';
import { EXPERIMENTS, METHOD_EXPERIMENTS, type ExperimentId, type Param, type Plan, buildPlan } from './plans';

/** Current value of the plan's parameter in the state. */
export function paramValue(plan: Plan, s: State): number {
  switch (plan.param) {
    case 'path':
      return s.s;
    case 'y':
      return s.Y[plan.index];
    case 'h':
      return s.h;
    case 'k':
      return s.kKnn;
  }
}

/** State patch that sets the plan's parameter to v (and everything derived from it). */
export function paramPatch(plan: Plan, s: State, v: number): Partial<State> {
  switch (plan.param) {
    case 'path':
      return { s: v, x0: pointOnPath(s.pathA, s.pathB, v), onPath: true };
    case 'y': {
      const Y = s.Y.slice();
      Y[plan.index] = v;
      return { Y };
    }
    case 'h':
      return { h: v };
    case 'k':
      return { kKnn: Math.round(v) };
  }
}

/**
 * Move `cur` toward `target` by at most `step` (in ln h for 'h'). Lands exactly
 * on `target` instead of passing it. 'k' is discrete and always lands at once.
 */
export function stepToward(param: Param, cur: number, target: number, step: number): { value: number; reached: boolean } {
  if (param === 'k' || !(step > 0)) return { value: target, reached: param === 'k' || cur === target };
  const log = param === 'h';
  const a = log ? Math.log(cur) : cur;
  const b = log ? Math.log(target) : target;
  const dir = Math.sign(b - a);
  const next = a + dir * step;
  if (dir === 0 || (next - b) * dir >= 0) return { value: target, reached: true };
  return { value: log ? Math.exp(next) : next, reached: false };
}

export interface RunnerInfo {
  id: ExperimentId;
  title: string;
  /** "start", "2/5", … */
  step: string;
  caption: string;
  phase: ExpRun['phase'];
  /** True while standing between two stops (moving or paused mid-way). */
  between: boolean;
  atEnd: boolean;
}

export class Runner {
  private plan: Plan | null = null;
  /** Was the profile open before the first experiment of a chain? Restored on exit. */
  private profileBefore = false;

  constructor(
    private store: Store,
    private reducedMotion: () => boolean = () => false,
  ) {
    // Whoever ends the experiment (Esc, a drag, a slider), clean up after it.
    store.subscribe((s, changed) => {
      if (changed.has('experiment') && !s.experiment && this.plan) this.cleanup();
    });
  }

  get current(): Plan | null {
    return this.store.get().experiment ? this.plan : null;
  }

  /** Load an experiment in its reproducible start state, standing at stop 0. */
  start(id: ExperimentId, view?: State['view']): void {
    const s = this.store.get();
    const method = s.method;
    if (!METHOD_EXPERIMENTS[method].includes(id)) return;
    const plan = buildPlan(id, s);
    this.plan = plan;
    if (!s.experiment) this.profileBefore = s.profileOpen;
    this.store.set({
      ...plan.start,
      method,
      view: view ?? s.view,
      emphasis: plan.stops[0].emphasis ?? null,
      experiment: { id, stop: 0, target: 0, phase: 'paused' },
    });
  }

  /** Page Down: continue to the next stop. Returns false when already at the end. */
  next(): boolean {
    const e = this.store.get().experiment;
    const plan = this.current;
    if (!e || !plan) return false;
    if (e.phase === 'running') {
      this.land(e.target); // a second press finishes the motion at once
      return true;
    }
    if (e.target > e.stop) {
      this.run(e.target);
      return true;
    }
    if (e.stop >= plan.stops.length - 1) return false;
    this.run(e.stop + 1);
    return true;
  }

  /** Page Up: back to the beginning of the current step (the previous stop). At stop 0: leave. */
  prev(): void {
    const e = this.store.get().experiment;
    if (!e || !this.current) return;
    if (e.target > e.stop || e.phase === 'running') this.land(e.stop);
    else if (e.stop > 0) this.land(e.stop - 1);

  }

  /** Space: pause a motion, resume it, go on to the next stop, or replay after the end. */
  toggle(): void {
    const e = this.store.get().experiment;
    if (!e || !this.current) return;
    if (e.phase === 'running') this.store.set({ experiment: { ...e, phase: 'paused' } });
    else if (e.phase === 'done') { this.restart(); this.next(); }
    else this.next();
  }

  /** Home: the reproducible start state again. */
  restart(): void {
    const e = this.store.get().experiment;
    const plan = this.current;
    if (!e || !plan) return;
    this.store.set({
      ...plan.start,
      emphasis: plan.stops[0].emphasis ?? null,
      experiment: { id: e.id, stop: 0, target: 0, phase: 'paused' },
    });
  }

  /** Esc: hand control back; the state stays exactly as it is. */
  exit(): void {
    if (this.store.get().experiment) this.store.set({ experiment: null });
  }

  /** Advance a running motion by dt seconds (called once per frame). */
  tick(dt: number): void {
    const s = this.store.get();
    const e = s.experiment;
    const plan = this.current;
    if (!e || !plan || e.phase !== 'running' || s.drag) return;
    const target = plan.stops[e.target].value;
    const { value, reached } = stepToward(plan.param, paramValue(plan, s), target, plan.rate * dt);
    if (reached) this.land(e.target);
    else this.store.set(paramPatch(plan, s, value));
  }

  info(s: State): RunnerInfo | null {
    const e = s.experiment;
    const plan = this.current;
    if (!e || !plan) return null;
    const last = plan.stops.length - 1;
    const between = e.target !== e.stop;
    return {
      id: e.id,
      title: EXPERIMENTS[e.id].title,
      step: e.stop === 0 && !between ? 'start' : `${between ? e.target : e.stop}/${last}`,
      caption: between ? plan.running(s) : plan.stops[e.stop].caption(s),
      phase: e.phase,
      between,
      atEnd: e.stop === last && !between,
    };
  }

  private run(i: number): void {
    const s = this.store.get();
    const e = s.experiment!;
    const plan = this.plan!;
    if (plan.param === 'k' || plan.rate <= 0 || this.reducedMotion()) {
      this.land(i);
      return;
    }
    this.store.set({ experiment: { ...e, target: i, phase: 'running' }, emphasis: null });
  }

  /** Stand exactly at stop i. */
  private land(i: number): void {
    const s = this.store.get();
    const e = s.experiment!;
    const plan = this.plan!;
    const last = plan.stops.length - 1;
    this.store.set({
      ...paramPatch(plan, s, plan.stops[i].value),
      emphasis: plan.stops[i].emphasis ?? null,
      experiment: { id: e.id, stop: i, target: i, phase: i === last ? 'done' : 'paused' },
    });
  }

  private cleanup(): void {
    const patch: Partial<State> = { highlightObs: null, emphasis: null, haloActive: false, profileOpen: this.profileBefore };
    this.plan = null;
    this.store.set(patch);
  }
}
