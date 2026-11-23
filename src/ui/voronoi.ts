import { computeCells } from '../geometry/cells';
import { DOMAIN } from '../data/presets';
import type { Vec2 } from '../math/estimators';

// This scene owns its sites. Nothing in a geometry detour writes to the regression store.
export class Voronoi {
  private sites: Vec2[] = [];
  private baseline: Vec2[] = [];
  private selected: number | null = null;
  private cells: SVGGElement;
  private markers: SVGGElement;
  constructor(private svg: SVGSVGElement) {
    svg.innerHTML = '<g class="polygons"></g><g class="markers"></g>';
    this.cells = svg.querySelector('.polygons')!;
    this.markers = svg.querySelector('.markers')!;
    let drag: {pointer: number; dx: number; dy: number} | null = null;
    const local = (e: PointerEvent) => new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM()!.inverse());
    svg.addEventListener('pointerdown', e => {
      if (drag || e.button !== 0) return;
      const p = local(e);
      let distance = 15 / svg.getScreenCTM()!.a;
      this.sites.forEach(([x,y], i) => {
        const d = Math.hypot(10*x-p.x, 100-10*y-p.y);
        if (d < distance) { distance = d; this.selected = i; }
      });
      if (this.selected === null) return;
      const [x,y] = this.sites[this.selected];
      drag = {pointer:e.pointerId, dx:x-p.x/10, dy:y-(10-p.y/10)};
      e.preventDefault();
      (this.markers.children[this.selected] as SVGElement).focus({preventScroll:true});
      svg.setPointerCapture(e.pointerId);
    });
    svg.addEventListener('pointermove', e => {
      if (this.selected === null || drag?.pointer !== e.pointerId) return;
      const p = local(e);
      this.move(this.selected,p.x/10+drag.dx,10-p.y/10+drag.dy);
    });
    svg.addEventListener('lostpointercapture', e => {
      if (drag?.pointer === e.pointerId) {this.selected=null; drag=null;}
    });
    new ResizeObserver(() => this.sizeMarkers()).observe(svg);
    svg.addEventListener('keydown', e=>{
      const target = e.target as SVGElement;
      if (target.dataset.site === undefined) return;
      const i = Number(target.dataset.site), [x,y] = this.sites[i];
      const delta:Record<string,Vec2> = {ArrowUp:[0,.2],ArrowDown:[0,-.2],ArrowLeft:[-.2,0],ArrowRight:[.2,0]};
      if(delta[e.key]){e.preventDefault();e.stopPropagation();this.move(i,x+delta[e.key][0],y+delta[e.key][1]);}
    });
  }
  load(sites:Vec2[]) { this.baseline = sites.map(p=>[...p]); this.reset(); }
  reset() {
    this.sites = this.baseline.map(p=>[...p]);
    this.markers.innerHTML = this.sites.map((_,i)=>`<circle tabindex="0" role="button" aria-label="Move site ${i+1} with arrow keys" data-site="${i}" vector-effect="non-scaling-stroke"><title>Site ${i+1}</title></circle>`).join('');
    this.draw();
  }
  private move(i:number,x:number,y:number) {
    this.sites[i] = [Math.min(10,Math.max(0,x)),Math.min(10,Math.max(0,y))];
    this.draw();
  }
  sizeMarkers() {
    const scale = this.svg.getScreenCTM()?.a;
    if (scale) for (const marker of this.markers.children) marker.setAttribute('r', String(4 / scale));
  }
  private draw() {
    const diagram=computeCells({X:this.sites,Y:this.sites.map(()=>1)},1,DOMAIN);
    this.cells.innerHTML=diagram.cells.map(c=>`<polygon points="${c.polygon.map(([x,y])=>[10*x,100-10*y].join(',')).join(' ')}"/>`).join('');
    this.sizeMarkers();
    Array.from(this.markers.children).forEach((e,i)=>{
      e.setAttribute('cx',String(this.sites[i][0]*10)); e.setAttribute('cy',String(100-this.sites[i][1]*10));
    });
    this.svg.setAttribute('aria-label',`Voronoi diagram with ${this.sites.length} movable sites`);
  }
}
