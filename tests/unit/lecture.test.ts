import { beforeAll, expect, it } from 'vitest';
import { PRESETS } from '../../src/data/presets';
import { Lecture, SCENES } from '../../src/experiments/lecture';
import { Runner } from '../../src/experiments/runner';
import { Store, presetState } from '../../src/state/store';

beforeAll(()=>{globalThis.requestAnimationFrame=()=>0;});

for (const [index,preset] of PRESETS.entries()) {
  it(`plays all scenes for ${preset.id} from the same responses`,()=>{
    const store=new Store();
    store.set(presetState(preset,index));
    const runner=new Runner(store);
    const lecture=new Lecture(store,runner);
    for (const scene of SCENES) {
      lecture.next();
      expect(store.get().Y).toEqual(preset.Y);
      expect(store.get().method).toBe(scene.method);
      expect(store.get().view).toBe(scene.view);
      expect(store.get().experiment?.id).toBe(scene.id);
      const stops=new Set<number>([0]);
      for(let frame=0;lecture.playing && frame<4000;frame++) {
        lecture.tick(1/60);
        stops.add(store.get().experiment!.stop);
      }
      expect(store.get().experiment?.phase).toBe('done');
      expect(stops.size).toBe(runner.current!.stops.length);
      const end=store.get();
      lecture.tick(100);
      expect(store.get()).toBe(end);
    }
    lecture.next();
    expect(lecture.index).toBe(0);
    expect(store.get().Y).toEqual(preset.Y);
  });
}

it('keeps both the initial hold and mid-motion pause under presenter control',()=>{
  const store=new Store();
  const lecture=new Lecture(store,new Runner(store));
  lecture.next();
  lecture.pause();
  lecture.tick(5);
  expect(store.get().experiment?.target).toBe(0);
  lecture.togglePause();
  for(let i=0;i<55;i++)lecture.tick(1/60);
  expect(store.get().s).toBeGreaterThan(0);
  lecture.pause();
  const atPause=store.get().s;
  lecture.tick(5);
  expect(store.get().s).toBe(atPause);
  lecture.togglePause();
  lecture.tick(1/60);
  expect(store.get().s).toBeGreaterThan(atPause);
  lecture.next();
  expect(store.get().experiment?.id).toBe('response');
  lecture.previous();
  expect(store.get().s).toBe(0);
  expect(lecture.playing).toBe(true);
});
