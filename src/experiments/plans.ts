import { RESPONSE_SITE, routeEndpoints } from '../data/presets';
import { stepProfile, pointOnPath } from '../path/profile';
import { predictKNN } from '../math/estimators';
import type { Method, State } from '../state/store';
import { predictAt } from '../state/model';
export type ExperimentId = 'segment' | 'response' | 'hsweep' | 'ksteps';
export type Param = 'path' | 'y' | 'h' | 'k';

export interface Stop {
  /** Value of the experiment's parameter at this stop (s, Y_j, h or k). */
  value: number;
  caption: (s: State) => string;
  emphasis?: 'radius';
}

export interface Plan {
  id: ExperimentId;
  method: Method;
  param: Param;
  /** Observation whose response moves (param 'y'). */
  index: number;
  /** Speed of continuous motion: s per second ('path'), Y per second ('y'), ln h per second ('h'). */
  rate: number;
  stops: Stop[];
  /** Reproducible start state; includes the parameter value of stop 0. */
  start: Partial<State>;
  /** Caption while moving between stops. */
  running: (s: State) => string;
}


export const EXPERIMENTS = {
  segment: {title:'Move across the map'}, response:{title:'Change one response'},
  ksteps:{title:'Increase k'}, hsweep:{title:'Change bandwidth h'},
};
export const METHOD_EXPERIMENTS: Record<Method,ExperimentId[]> = {
  '1nn':['segment','response'], knn:['segment','response','ksteps'], nw:['segment','response','hsweep']
};
export function buildPlan(id: ExperimentId, s: State): Plan {
  const n = s.X.length;
  const k = s.method === '1nn' ? 1 : s.kKnn;
  const start: Partial<State> = {profileOpen:false, highlightObs:null, haloActive:false, playing:false, onPath:false};
  const index = RESPONSE_SITE;
  const plan = {id, method:s.method, index, start};
  if (id === 'response') {
    const before = s.Y[index];
    const after = before <= 7 ? before + 3 : before - 3;
    const change = (state: State) => {
      const y = state.Y[index], dy = y-before;
      const effect = s.method==='1nn' ? 'Only this plateau changes height.' : s.method==='knn' ? `Selected plateaus shift by ΔY/${k}.` : 'Local change = weight × ΔY.';
      const w = predictAt(state, state.x0).weights[index];
      return `Only Y${index+1} changes. With weights fixed, Δf̂(x₀) = w${index+1} ΔY${index+1} ≈ ${w.toFixed(3)} × (${dy.toFixed(2)}) ≈ ${(w*dy).toFixed(2)}. ${effect}`;
    };
    Object.assign(start,{Y:[...s.Y],highlightObs:index,x0:[Math.max(0,s.X[index][0]-0.5),Math.min(10,s.X[index][1]+0.4)]});
    return {...plan,param:'y',rate:1.1,stops:[before,after].map(value=>({value,caption:change})),running:change};
  }
  if (id === 'ksteps') {
    start.kKnn = 1;
    const caption = (v: State) => v.kKnn === n ? 'Every location now predicts the sample mean.' :
      `Average the nearest ${v.kKnn}. Each carries weight 1/${v.kKnn}.`;
    return {...plan,param:'k',rate:0,stops:[...new Set([1,3,5,8,n])].map(value=>({value,caption})),running:caption};
  }
  if (id === 'hsweep') {
    Object.assign(start,{h:0.4,haloActive:true});
    const caption = (v:State)=>`h = ${v.h.toFixed(2)}. ${v.h<1?'Nearby sites dominate.':'Weights spread more evenly across sites.'}`;
    return {...plan,param:'h',rate:0.8,stops:[0.4,1.2,3.6].map(value=>({value,caption})),running:caption};
  }
  const [A,B] = routeEndpoints(s.X);
  Object.assign(start,{pathA:A,pathB:B,x0:A,s:0,onPath:true,profileOpen:true});
  const changes = s.method === 'nw' ? [] : stepProfile(s,k,A,B).setChanges;
  const bounds = [0,...changes,1];
  const events = s.method === 'nw' ? [0,0.33,0.67,1] : bounds;
  const stops:Stop[] = events.map((value,i) => {
    let message = i===0 ? 'Follow the query from A to B.' : 'The query has reached B.';
    if (s.method==='nw') message = 'Gaussian weights shift continuously along the route.';
    else if (k===n) message = 'All sites contribute everywhere. There is no boundary.';
    else if (i>0 && i<events.length-1) {
      const at = (t:number)=>predictKNN(s,pointOnPath(A,B,t),k).value.toFixed(2);
      message = `${k===1?'Nearest site changes':'One neighbour is exchanged'}. Across the boundary: ${at((bounds[i-1]+value)/2)} → ${at((bounds[i+1]+value)/2)}.`;
    }
    return {value,caption:()=>message,emphasis:i>0 && i<events.length-1 && s.method!=='nw'?'radius':undefined};
  });
  return {...plan,param:'path',rate:0.13,stops,running:()=> 'Follow the moving query along the marked path.'};
}
