import { mkdirSync, writeFileSync } from 'node:fs';
import { PRESETS } from '../../src/data/presets';
import { METHODS, Store,presetState } from '../../src/state/store';
import { buildPlan,METHOD_EXPERIMENTS } from '../../src/experiments/plans';
import { paramPatch } from '../../src/experiments/runner';
import { predictAt } from '../../src/state/model';

const cases=[];
for(const [i,p] of PRESETS.entries())for(const method of METHODS){
  const state={...new Store().get(),...presetState(p,i),method};
  for(const id of METHOD_EXPERIMENTS[method]){
    const plan=buildPlan(id,state),start={...state,...plan.start};
    const frames=plan.stops.map(stop=>{
      const s={...start,...paramPatch(plan,start,stop.value)};
      const result=predictAt(s,s.x0);
      return {position:stop.value,x:s.x0,Y:s.Y,k:method==='1nn'?1:s.kKnn,h:s.h,value:result.value,weights:[...result.weights]};
    });
    cases.push({id,method,dataset:p.id,X:p.X,Y:p.Y,A:start.pathA,B:start.pathB,frames});
  }
}
const outputDir = new URL('../../.artifacts/numerical/', import.meta.url);
mkdirSync(outputDir, { recursive: true });
writeFileSync(new URL('experiments.json',outputDir),JSON.stringify(cases));
