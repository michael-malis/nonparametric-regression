import { beforeAll, expect, it } from 'vitest';
import { PRESETS, OUTLIER } from '../../src/data/presets';
import { Store, presetState, METHODS } from '../../src/state/store';
import { buildPlan, METHOD_EXPERIMENTS } from '../../src/experiments/plans';
import { Runner } from '../../src/experiments/runner';
import { predictAt, cells } from '../../src/state/model';
import { pathLength,stepProfile } from '../../src/path/profile';
beforeAll(()=>{globalThis.requestAnimationFrame=()=>0;});
it('toggling the outlier preserves every site and changes one response',()=>{
  for(const i of [0,2]){
    const a=PRESETS[i], b=PRESETS[i+1];
    expect(a.X).toEqual(b.X);
    expect(a.Y.map((y,j)=>y!==b.Y[j]?j:-1).filter(j=>j>=0)).toEqual([OUTLIER]);
  }
  expect(PRESETS[2].X.slice(0,8)).toEqual(PRESETS[0].X);
});
for (const [i,p] of PRESETS.entries()) for (const method of METHODS) {
  it(p.id+' '+method+' follows the selected dataset through every event',()=>{
    const store=new Store();
    const runner=new Runner(store);
    for(const id of METHOD_EXPERIMENTS[method]){
      store.set({...presetState(p,i),method});store.flush();
      const original=store.get(), plan=buildPlan(id,original);
      runner.start(id);
      for(let j=1;j<plan.stops.length;j++){
        runner.next();
        for(let frame=0;store.get().experiment!.phase==='running';frame++){
          expect(frame).toBeLessThan(600);runner.tick(0.1);
        }
        expect(store.get().experiment!.stop).toBe(j);
      }
      const end=store.get();
      expect(end.X).toEqual(original.X);
      expect(end.experiment?.phase).toBe('done');
      expect(runner.next()).toBe(false);
      if(id==='segment'){
        expect(pathLength(end.pathA,end.pathB)).toBeGreaterThan(6);
        if(method!=='nw'){
          const changes=stepProfile(end,method==='1nn'?1:end.kKnn,end.pathA,end.pathB).setChanges;
          expect(changes.length).toBeGreaterThanOrEqual(2);
          expect(plan.stops.slice(1,-1).map(t=>t.value)).toEqual(changes);
        }
      }else if(id==='response'){
        for(const x of [[2,4],[8,2],[5,8]] as [number,number][]){
          const a=predictAt(original,x), b=predictAt(end,x);
          expect(b.value-a.value).toBeCloseTo(a.weights[plan.index]*(end.Y[plan.index]-original.Y[plan.index]),11);
        }
        expect(cells(original).cells.map(c=>c.polygon)).toEqual(cells(end).cells.map(c=>c.polygon));
      }else if(id==='ksteps'){
        expect(end.kKnn).toBe(end.X.length);
        expect(predictAt(end,[1,3]).value).toBeCloseTo(end.Y.reduce((a,b)=>a+b)/end.Y.length,12);
      }else expect(end.h).toBe(3.6);
      runner.restart();expect(store.get().experiment?.stop).toBe(0);
      runner.exit();store.flush();
    }
  });
}
