import '@fontsource/atkinson-hyperlegible/latin-400.css';
import '@fontsource/atkinson-hyperlegible/latin-700.css';
import 'katex/dist/katex.min.css';
import './style.css';
import { PRESETS, OUTLIER, RESPONSE_SITE } from './data/presets';
import { EXPERIMENTS, METHOD_EXPERIMENTS, type ExperimentId } from './experiments/plans';
import { Lecture, SCENES } from './experiments/lecture';
import { Runner } from './experiments/runner';
import { View2D } from './scene/view2d';
import { View3D } from './scene/view3d';
import { cells, prediction } from './state/model';
import { METHOD_LABEL, type Method, Store, type View, type State, presetState } from './state/store';
import { lectureLayout } from './ui/controls';
import { Hud } from './ui/hud';
import { ProfileChart } from './ui/profileChart';
import { Voronoi } from './ui/voronoi';

const app = document.querySelector<HTMLElement>('#app')!;
app.innerHTML = lectureLayout;
const q = <T extends HTMLElement = HTMLElement>(sel:string) => app.querySelector<T>(sel)!;
const input = (id:string) => q<HTMLInputElement>('#'+id);
const button = (id:string) => q<HTMLButtonElement>('#'+id);
const select = (id:string) => q<HTMLSelectElement>('#'+id);
const store = new Store();
const runner = new Runner(store,()=>matchMedia('(prefers-reduced-motion: reduce)').matches);
const lecture = new Lecture(store, runner);
const viewport = q('.viewport');
const v2 = new View2D(viewport,store,index => {
  undoEdit = {X:store.get().X};
  site=index;
  store.set({highlightObs:index});
});
let v3: View3D | null = null;
let webglUnavailable = false;
function surface(): View3D | null {
  if (!v3 && !webglUnavailable) {
    try {
      v3 = new View3D(viewport,store,index=>{
        undoEdit={Y:store.get().Y};site=index;store.set({highlightObs:index});
      });
    } catch (error) {
      webglUnavailable = true;
      console.warn('3D renderer unavailable',error);
      q<HTMLButtonElement>('[data-view="3d"]').disabled = true;
      message.textContent='3D requires WebGL2. The 2D map is available.';
      store.set({view:'2d'});
    }
  }
  return v3;
}
const geometry = new Voronoi(app.querySelector('svg')!);
const hud = new Hud(q('.rail'),s=>runner.info(s));
const chart = new ProfileChart(q('.rail'),store);
type Mode = 'demo'|'explore'|'voronoi';
let mode: Mode = 'demo';
let site = RESPONSE_SITE;
let geometryData: State['X'] | null = null;
let undoEdit: Pick<State,'X'> | Pick<State,'Y'> | null = null;
let originalY = store.get().Y[OUTLIER];
let visibleView = '';
const message = document.createElement('p');
message.className = 'status';
message.setAttribute('role','status');
q('.rail').append(message);

function exitExample() {
  lecture.exit();
  store.flush();
  message.textContent='';
}
function setMode(next:Mode) {
  if(next===mode) return;
  if(next==='voronoi' && geometryData!==store.get().X){
    geometry.load(store.get().X); geometryData=store.get().X;
  }
  if(next==='explore'){exitExample();store.set({highlightObs:site});}
  if(next==='demo' && mode==='explore') store.set({highlightObs:null});
  mode=next; render();
  if (next === 'voronoi') geometry.sizeMarkers();
}
function setMethod(method:Method) {
  if(method===store.get().method) return;
  exitExample();store.set({method,highlightObs:mode==='explore'?site:null});
}
function setData(n:number,outlier:boolean,reset=false) {
  const s=store.get();
  const index=(n===16?2:0)+Number(outlier), p=PRESETS[index];
  if(!reset && s.presetIndex===index) return;
  const before = lecture.baselineY ?? s.Y;
  exitExample(); undoEdit=null;
  const Y=p.Y.map((y,i)=>reset?y:(before[i]??y));
  if(reset) originalY=PRESETS[index-Number(outlier)].Y[OUTLIER];
  else if(s.includeOutlier!==outlier){
    if(outlier) originalY=before[OUTLIER];
    Y[OUTLIER]=outlier?p.Y[OUTLIER]:originalY;
  }
  site=Math.min(site,n-1);
  store.set({...presetState(p,index),X:!reset && n===s.X.length?s.X:p.X,Y,
    kKnn:reset?p.k:Math.min(s.kKnn,n),h:reset?p.h:s.h,x0:reset?p.x0:s.x0,
    highlightObs:mode==='explore'?site:null,profileOpen:false});
  if(mode==='voronoi'){geometry.load(store.get().X);geometryData=store.get().X;}
  message.textContent=reset?'Selected dataset restored.':'Dataset updated.';
}
function play() {
  if(mode!=='demo') return;
  message.textContent='';
  lecture.next();
  render();
}
app.querySelectorAll<HTMLElement>('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode as Mode));
app.querySelectorAll<HTMLElement>('[data-method]').forEach(b=>b.onclick=()=>setMethod(b.dataset.method as Method));
app.querySelectorAll<HTMLElement>('[data-view]').forEach(b=>b.onclick=()=>store.set({view:b.dataset.view as View}));
app.querySelectorAll<HTMLInputElement>('[name=count]').forEach(r=>r.onchange=()=>setData(+r.value,store.get().includeOutlier));
input('outlier').onchange=()=>setData(store.get().X.length,input('outlier').checked);
button('play').onclick=play;
const experimentSelect = select('experiment');
SCENES.forEach((scene,index) => experimentSelect.add(new Option(
  `${index + 1}. ${METHOD_LABEL[scene.method]} · ${EXPERIMENTS[scene.id].title}`, String(index))));
experimentSelect.onchange = () => {lecture.load(Number(experimentSelect.value)); render();};
button('previous').onclick=()=>lecture.previous();
button('restart').onclick=()=>{
  lecture.replay();
  q('#options').hidePopover();
  button('play').focus();
};
button('pause').onclick=()=>{lecture.togglePause();render();};
button('reset-sites').onclick=()=>geometry.reset();
button('camera').onclick=()=>v3?.resetCamera();
button('reset').onclick=()=>setData(store.get().X.length,store.get().includeOutlier,true);
button('full').onclick=()=>document.fullscreenElement?void document.exitFullscreen():void document.documentElement.requestFullscreen();
input('regions').onchange=()=>store.set({showRegions:input('regions').checked});
input('profile').onchange=()=>store.set({profileOpen:input('profile').checked});
input('tuning').oninput=()=>{
  const value=+input('tuning').value;
  store.set(store.get().method==='nw'?{h:value}:{kKnn:value});
};
select('site').onchange=()=>{site=+select('site').value;store.set({highlightObs:site});};
const response=input('response');
let editing=false;
response.oninput=()=>{
  const Y=[...store.get().Y];
  if(!editing) undoEdit={Y:store.get().Y};
  editing=true;Y[site]=+response.value;store.set({Y,highlightObs:site});
};
response.onchange=()=>{editing=false;};
button('undo').onclick=()=>{if(undoEdit){store.set(undoEdit);undoEdit=null;}};
function pressed(attr:'mode'|'method'|'view',value:string) {
  app.querySelectorAll<HTMLElement>('[data-'+attr+']').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset[attr]===value)));
}
function setText(selector:string, text:string) {
  const node=q(selector);
  if(node.textContent!==text)node.textContent=text;
}
function render() {
  const s=store.get(), e=s.experiment;
  document.body.dataset.mode=mode;
  document.body.classList.toggle('profile-open',s.profileOpen && s.view==='2d');
  pressed('mode',mode);pressed('method',s.method);pressed('view',s.view);
  const view=mode==='voronoi'?'geometry':s.view;
  if(view!==visibleView){
    if(view==='3d' && !surface()){if(store.get().view==='3d')store.set({view:'2d'});return;}
    v2.setVisible(view==='2d');v3?.setVisible(view==='3d');visibleView=view;
  }
  q('#outlier-option').hidden=mode==='voronoi';
  q('#config').textContent=mode==='voronoi' ? `${s.X.length} sites` : `${s.X.length} observations · ${s.includeOutlier?'Outlier on':'No added outlier'}`;
  input('outlier').checked=s.includeOutlier;
  q<HTMLInputElement>(`[name=count][value="${s.X.length}"]`).checked=true;
  const active=lecture.index>=0 && e;
  experimentSelect.value = String(lecture.index);
  const lastScene=lecture.index===SCENES.length-1;
  const primary=button('play');
  primary.textContent=!active?'Start':lecture.ready?'Start scene':lastScene?'Restart lecture':'Next';
  button('previous').hidden=!active || lecture.index===0;
  button('pause').hidden=!active || lecture.ready || e.phase==='done';
  button('pause').textContent=lecture.playing?'Pause':'Resume';
  button('restart').disabled=!active;
  setText('#scene-title',active?EXPERIMENTS[e.id].title:'Guided exploration');
  setText('#scene-context',active
    ? `Scene ${lecture.index+1} of ${SCENES.length} · ${METHOD_LABEL[s.method]} · ${s.view.toUpperCase()}`
    : 'Start the sequence or choose an experiment.');
  setText('#progress',!active?`${SCENES.length} scenes`:e.phase==='done'
    ? lastScene?'Lecture complete':'Scene complete'
    : !lecture.playing && !lecture.ready?'Paused':'');
  q('#tuning-label').hidden=s.method==='1nn';
  const nw=s.method==='nw', tune=input('tuning');
  tune.min=nw?'0.15':'1';tune.max=nw?'6':String(s.X.length);tune.step=nw?'0.05':'1';tune.value=String(nw?s.h:s.kKnn);
  q('#tuning-name').textContent=nw?'Bandwidth h':'Neighbours k';q('#tuning-value').textContent=nw?s.h.toFixed(2):String(s.kKnn);
  const sites=select('site');
  if(sites.options.length!==s.X.length)sites.replaceChildren(...s.X.map((_,i)=>new Option('Y'+String(i+1),String(i))));
  sites.value=String(site);response.value=String(s.Y[site]);q('#response-value').textContent=s.Y[site].toFixed(1);
  button('undo').disabled=!undoEdit;
  input('regions').checked=s.showRegions;input('regions').disabled=s.method!=='knn' || s.view!=='2d';
  input('profile').checked=s.profileOpen;input('profile').disabled=e?.id==='segment';
  button('camera').disabled=s.view!=='3d';
}
store.subscribe((s,changed)=>{
  if(mode!=='demo' || ['experiment','view','method','presetIndex','X','profileOpen','showRegions'].some(k=>changed.has(k as keyof State)))render();
  if(mode==='voronoi')return;
  if(s.view==='2d')v2.update(s,changed);else v3?.update(s,changed);
  hud.update(s);chart.update(s,changed);
});
for(const b of app.querySelectorAll<HTMLButtonElement>('[data-mode]')){
  b.addEventListener('click',()=>{store.flushAll();});
}
window.addEventListener('keydown',e=>{
  if(document.querySelector(':popover-open') || e.ctrlKey || e.altKey || e.metaKey)return;
  if(e.key==='Escape'){if(mode==='demo'){lecture.pause();render();}return;}
  if(mode==='voronoi' || (e.target as HTMLElement).matches('input,select,summary'))return;
  if(mode==='explore' && (e.key==='2'||(e.key==='3' && !webglUnavailable)))store.set({view:e.key==='2'?'2d':'3d'});
  if(mode!=='demo' || e.repeat)return;
  if(e.key===' ' && !(e.target as HTMLElement).closest('button')){
    e.preventDefault();
    if(lecture.index<0 || lecture.ready) play();
    else {lecture.togglePause();render();}
  }
  if(e.key==='PageDown'||e.key==='ArrowRight'){e.preventDefault();play();}
  if(e.key==='PageUp'||e.key==='ArrowLeft'){e.preventDefault();lecture.previous();}

});
let last=performance.now();
function frame(now:number){
  if(mode==='demo' && !document.hidden)lecture.tick(Math.min(.15,(now-last)/1000));
  last=now;store.flush();if(mode!=='voronoi')v3?.frame(now);requestAnimationFrame(frame);
}
// A lecturer can bookmark an explicit starting scene. An empty address stays idle.
const params = new URLSearchParams(location.hash.slice(1));
const stageNumber = Number(params.get('stage'));
if (stageNumber >= 1 && stageNumber <= 6 && Number.isInteger(stageNumber)) {
  const methods: Method[] = ['1nn','knn','nw'];
  store.set({method:methods[Math.floor((stageNumber-1)/2)],view:stageNumber%2?'2d':'3d'});
}
if (params.has('count') || params.has('outlier') || params.has('preset')) {
  const dense = params.get('count')==='16' || params.get('preset')==='dense';
  const outlier = params.get('outlier')==='1' || params.get('preset')==='outlier';
  setData(dense?16:8,outlier);
}
if(params.get('profile')==='1')store.set({profileOpen:true});
const requested = params.get('exp') as ExperimentId | null;
if(requested && METHOD_EXPERIMENTS[store.get().method].includes(requested)){
  lecture.bookmark(requested);
} else {
  setMode('explore');
}
message.textContent='';
store.flushAll();document.fonts.ready.then(()=>{v2.draw();chart.draw();});requestAnimationFrame(frame);
Object.assign(window,{__demo:{state:()=>store.get(),prediction:()=>prediction(store.get()),cells:()=>cells(store.get()),v2,get v3(){return surface();},
  plan:()=>runner.current, mode:()=>mode}});
