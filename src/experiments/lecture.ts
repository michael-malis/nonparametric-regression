import { PRESETS } from '../data/presets';
import type { Method, Store, View } from '../state/store';
import type { ExperimentId } from './plans';
import { Runner } from './runner';

interface Scene { method: Method; view: View; id: ExperimentId }

export const SCENES: readonly Scene[] = [
  {method:'1nn', view:'2d', id:'segment'},
  {method:'1nn', view:'3d', id:'response'},
  {method:'knn', view:'2d', id:'segment'},
  {method:'knn', view:'3d', id:'response'},
  {method:'knn', view:'3d', id:'ksteps'},
  {method:'nw', view:'2d', id:'segment'},
  {method:'nw', view:'3d', id:'hsweep'},
  {method:'nw', view:'3d', id:'response'},
];

/** Each press selects a scene. The runner visits its mathematical stops on its own. */
export class Lecture {
  index = -1;
  playing = false;
  ready = false;
  baselineY: number[] | null = null;
  private hold = 0;

  constructor(private store: Store, private runner: Runner) {}

  load(index: number, ready = false, view = SCENES[index].view) {
    const s = this.store.get();
    this.baselineY ??= [...s.Y];
    const preset = PRESETS[s.presetIndex];
    const scene = SCENES[index];
    this.index = index;
    this.ready = ready;
    this.playing = !ready;
    this.hold = 0.7;
    this.store.set({method:scene.method, x0:preset.x0, kKnn:preset.k, h:preset.h,
      ...(s.Y === this.baselineY ? {} : {Y:this.baselineY})});
    this.runner.start(scene.id, view);
  }

  bookmark(id: ExperimentId) {
    const s = this.store.get();
    const index = SCENES.findIndex(scene => scene.method === s.method && scene.id === id);
    if (index >= 0) this.load(index, true, s.view);
  }

  next() {
    if (this.ready) {
      this.ready = false;
      this.playing = true;
    } else if (this.index < SCENES.length - 1) {
      this.load(this.index + 1);
    } else {
      this.load(0);
    }
  }

  previous() {
    if (this.index > 0) this.load(this.index - 1);
  }

  replay() {
    if (this.index >= 0) this.load(this.index, false, this.store.get().view);
  }

  pause() {
    this.playing = false;
    if (this.store.get().experiment?.phase === 'running') this.runner.toggle();
  }

  togglePause() {
    const e = this.store.get().experiment;
    if (this.playing) this.pause();
    else if (e && e.phase !== 'done') {
      this.ready = false;
      this.playing = true;
      if (e.target > e.stop) this.runner.next();
    }
  }

  tick(dt: number) {
    if (!this.playing) return;
    const e = this.store.get().experiment;
    if (!e) return;
    if (e.phase === 'running') this.runner.tick(dt);
    else {
      this.hold -= dt;
      if (this.hold > 0) return;
      this.runner.next();
      this.hold = 1.3;
    }
    if (this.store.get().experiment?.phase === 'done') this.playing = false;
  }

  exit() {
    this.runner.exit();
    this.index = -1;
    this.playing = this.ready = false;
    this.baselineY = null;
  }
}
