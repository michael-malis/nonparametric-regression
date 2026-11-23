import { upcomingCrossing, predictAt } from '../state/model';
// 3-D view: the graph z = f̂(x1, x2) over the feature plane.
//
// 1NN / kNN: every exact order-k cell becomes a horizontal plateau at height
// f̂ = mean of its neighbours' Y. Plateaus never connect through sloped
// triangles; where the value jumps a neutral vertical wall is drawn instead.
// NW: a dense height-field mesh; its colour is evaluated per fragment from the
// same Gaussian NW formula, so colour is exact even between mesh vertices.
//
// World coordinates: X = x1 - 5, Y (up) = VZ * f̂, Z = -(x2 - 5).

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Line2 } from 'three/examples/jsm/lines/Line2.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js';
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { DOMAIN, OUTLIER, HEIGHT_MAX } from '../data/presets';
import type { CellDiagram } from '../geometry/cells';
import { LUT, LUT_SIZE, colorHex } from '../math/colormap';
import { type Vec2, nwGrid, nwValue, pack } from '../math/estimators';
import { pointOnPath } from '../path/profile';
import { cells, effectiveK, prediction, profile, valueScale } from '../state/model';
import type { Key, State, Store } from '../state/store';
import { THEME, fmt } from '../ui/theme';
import { type DragTarget, applyDrag, beginDrag, endDrag, handlePosition } from './handles';

/** World height per unit of f̂ (a declared geometric scale, fixed for all methods). */
export const VZ = 0.5;
const CX = (DOMAIN.x0 + DOMAIN.x1) / 2;
const CY = (DOMAIN.y0 + DOMAIN.y1) / 2;
const MAX_POINTS = 64;
const NW_RES = 200;

export function world(x: Vec2, f = 0): THREE.Vector3 {
  return new THREE.Vector3(x[0] - CX, VZ * f, -(x[1] - CY));
}

const lutTexture = (() => {
  const t = new THREE.DataTexture(LUT, LUT_SIZE, 1, THREE.RGBAFormat);
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  return t;
})();

const LUT_GLSL = /* glsl */ `
  uniform sampler2D uLut;
  uniform float uLo;
  uniform float uHi;
  vec3 scaleColor(float f) {
    float t = clamp((f - uLo) / (uHi - uLo), 0.0, 1.0);
    return texture2D(uLut, vec2((t * ${LUT_SIZE - 1}.0 + 0.5) / ${LUT_SIZE}.0, 0.5)).rgb;
  }
`;

function lutUniforms() {
  return { uLut: { value: lutTexture }, uLo: { value: 0 }, uHi: { value: 10 } };
}

// Shaders write raw sRGB values: the colour on screen is exactly C(f̂), the same
// bytes as in the 2-D view. Flat plateaus get no shading at all.
function plateauMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: lutUniforms(),
    vertexShader: /* glsl */ `
      attribute float aValue;
      varying float vValue;
      void main() {
        vValue = aValue;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      ${LUT_GLSL}
      varying float vValue;
      void main() { gl_FragColor = vec4(scaleColor(vValue), 1.0); }`,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
}

function hatchedWallMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
      uniforms:lutUniforms(),side:THREE.DoubleSide,
      vertexShader:`attribute float aValue; attribute float aStrength; varying float strength; varying float vValue; varying vec2 face;
      void main(){strength=aStrength; vValue=aValue; face=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
      fragmentShader:`${LUT_GLSL}
      varying float vValue; varying vec2 face; varying float strength;
      void main(){
        float phase=(face.y-face.x)*2.0;
        float width=max(fwidth(phase),0.01);
        float line=1.0-smoothstep(0.04,0.04+width,abs(fract(phase+0.5)-0.5));
        line*=1.0-smoothstep(0.25,0.5,width);
        gl_FragColor=vec4(scaleColor(vValue)*(0.64+0.23*strength*line),1.0);
      }`
    });
}

function wallMaterial(): THREE.ShaderMaterial {
  const c = new THREE.Color(THEME.wall);
  // THREE.Color stores linear values; convert back to the sRGB bytes we want on screen.
  c.convertLinearToSRGB();
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Vector3(c.r, c.g, c.b) } },
    vertexShader: /* glsl */ `
      varying vec3 vN;
      void main() {
        vN = normal;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      varying vec3 vN;
      void main() {
        float l = 0.7 + 0.45 * max(dot(normalize(vN), normalize(vec3(0.55, 0.2, 0.8))), 0.0);
        gl_FragColor = vec4(uColor * l, 1.0);
      }`,
    side: THREE.DoubleSide,
  });
}

function nwMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      ...lutUniforms(),
      uPts: { value: Array.from({ length: MAX_POINTS }, () => new THREE.Vector2()) },
      uYs: { value: new Array(MAX_POINTS).fill(0) },
      uN: { value: 0 },
      uH: { value: 1 },
    },
    vertexShader: /* glsl */ `
      attribute vec2 aFeat;
      varying vec2 vFeat;
      varying vec3 vN;
      void main() {
        vFeat = aFeat;
        vN = normal;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      ${LUT_GLSL}
      uniform vec2 uPts[${MAX_POINTS}];
      uniform float uYs[${MAX_POINTS}];
      uniform int uN;
      uniform float uH;
      varying vec2 vFeat;
      varying vec3 vN;
      void main() {
        // Stable Gaussian NW: subtract the smallest squared distance first.
        float m = 1e30;
        for (int i = 0; i < ${MAX_POINTS}; i++) {
          if (i >= uN) break;
          vec2 d = vFeat - uPts[i];
          m = min(m, dot(d, d));
        }
        float inv = 1.0 / (2.0 * uH * uH);
        float num = 0.0;
        float den = 0.0;
        for (int i = 0; i < ${MAX_POINTS}; i++) {
          if (i >= uN) break;
          vec2 d = vFeat - uPts[i];
          float a = exp(-(dot(d, d) - m) * inv);
          num += a * uYs[i];
          den += a;
        }
        float f = num / den;
        vec3 c = scaleColor(f);
        // Very faint level lines f̂ = 1, 2, ... as a reading aid for the relief.
        // Where the surface is (nearly) flat there is no line: a flat area whose
        // value happens to be an integer keeps its exact colour.
        float q = f;
        float w = fwidth(q);
        float line = 1.0 - clamp(abs(q - floor(q + 0.5)) / max(w, 1e-6) - 0.25, 0.0, 1.0);
        line *= smoothstep(0.0008, 0.004, w);
        c *= 1.0 - 0.12 * line;
        // Mild shading only on slopes; a horizontal surface keeps its exact colour.
        vec3 L = normalize(vec3(0.45, 1.0, 0.35));
        float lam = clamp(dot(normalize(vN), L) / L.y, 0.0, 1.0);
        c *= mix(0.8, 1.0, lam);
        gl_FragColor = vec4(c, 1.0);
      }`,
    side: THREE.DoubleSide,
  });
}

function ghostify<T extends THREE.Object3D>(obj: T, opacity: number): T {
  obj.traverse((o) => {
    const m = (o as THREE.Mesh).material as THREE.Material | undefined;
    if (!m) return;
    const g = m.clone();
    g.transparent = true;
    g.opacity = opacity;
    g.depthTest = false;
    g.depthWrite = false;
    (o as THREE.Mesh).material = g;
    o.renderOrder = 20;
  });
  return obj;
}

/** CSS2D label; the visible text sits in an inner span so CSS can offset it. */
function makeLabel(className: string, html = ''): CSS2DObject {
  const el = document.createElement('div');
  el.className = className;
  const inner = document.createElement('span');
  inner.className = 'lbl';
  inner.innerHTML = html;
  el.appendChild(inner);
  return new CSS2DObject(el);
}

function setLabel(o: CSS2DObject, html: string): void {
  if (o.userData.labelText === html) return;
  o.userData.labelText = html;
  (o.element.firstElementChild as HTMLElement).innerHTML = html;
}

// Both endpoints share one GPU buffer. Update it in place as the prediction moves.
function moveSegment(line: Line2, from: THREE.Vector3, to: THREE.Vector3) {
  const g=line.geometry;
  const start=g.getAttribute('instanceStart');
  start.setXYZ(0,from.x,from.y,from.z);
  g.getAttribute('instanceEnd').setXYZ(0,to.x,to.y,to.z);
  start.needsUpdate=true;
  g.computeBoundingBox();
  g.computeBoundingSphere();
}

interface PointViz {
  sphere: THREE.Mesh;
  outline: THREE.Mesh;
  ghost: THREE.Mesh;
  floor: THREE.Mesh;
  stem: THREE.Line;
  stemGhost: THREE.Line;
  link: Line2;
  linkGhost: Line2;
  label: CSS2DObject;
}

export class View3D {
  readonly renderer: THREE.WebGLRenderer;
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private labels: CSS2DRenderer;
  private scene = new THREE.Scene();
  private surface = new THREE.Group();
  private surfaceKey = '';
  private nwMesh: THREE.Mesh | null = null;
  private nwSkirt: THREE.Mesh | null = null;
  private points = new THREE.Group();
  private pointViz: PointViz[] = [];
  private pointKey = '';
  private pathGroup = new THREE.Group();
  private crossing = makeLabel("crossing","◇");
  private pathKey = '';
  private backdrop: Record<'left' | 'right' | 'front' | 'back', THREE.Group> = {} as never;
  private vAxes: Record<string, THREE.Group> = {};
  private lineMats: LineMaterial[] = [];
  private x0Marker!: THREE.Group;
  private x0Ghost!: THREE.Group;
  private x0Guide!: Line2;
  private x0GuideGhost!: Line2;
  private x0Floor!: THREE.Group;
  private x0Label!: CSS2DObject;
  private abHandles: THREE.Mesh[] = [];
  private labelInfo: { text: string; prio: number }[] = [];
  private abLabels: CSS2DObject[] = [];
  private dirty = true;
  private tween: null | { from: THREE.Spherical; to: THREE.Spherical; t0: number; dur: number; target: THREE.Vector3 } = null;
  private drag: null | { target: DragTarget; pointerId: number; plane: THREE.Plane; offset: Vec2 } = null;
  private raycaster = new THREE.Raycaster();
  private sharedSphere = new THREE.SphereGeometry(4.7, 24, 16);
  private sharedOutline = new THREE.SphereGeometry(5.5, 24, 16);
  private verticalDrag: {index:number;pointer:number;startY:number;initial:number;pixelsPerUnit:number} | null = null;
  visible = false;
  private w = 1;
  private h = 1;
  private stepMaterials = {
    top:plateauMaterial(),
    wall:hatchedWallMaterial(),
    rim:this.lineMat(THEME.outline, 1, { transparent:true, opacity:0.32 }),
    halo:this.lineMat(THEME.outline, 5, { transparent:true, opacity:0.85 }),
    selected:this.lineMat(THEME.ring, 2.2),
  };

  constructor(
    private root: HTMLElement,
    private store: Store,
    private onResponseEdit: (index:number)=>void,
  ) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(new THREE.Color(THEME.bg), 1);
    this.renderer.domElement.className = 'view3d';
    this.renderer.domElement.setAttribute('aria-label', '3-D prediction surface; drag x0 or observations, drag elsewhere to rotate');
    root.appendChild(this.renderer.domElement);
    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = 'labels3d';
    root.appendChild(this.labels.domElement);

    this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.14;
    this.controls.minDistance = 6;
    this.controls.maxDistance = 48;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02; // never look from below the feature plane
    this.controls.addEventListener('change', () => (this.dirty = true));
    this.controls.addEventListener('start', () => (this.tween = null));

    this.scene.add(this.surface, this.points, this.pathGroup, this.crossing);
    this.buildStatic();
    this.buildX0();

    // Capture phase on the container runs before OrbitControls' own listener:
    // a press on a handle suspends the camera for the whole gesture.
    root.addEventListener('pointerdown', this.onDown, { capture: true });
    this.renderer.domElement.addEventListener('pointermove', this.onMove);
    this.renderer.domElement.addEventListener('pointerup', this.onUp);
    this.renderer.domElement.addEventListener('pointercancel', this.onUp);
    this.renderer.domElement.addEventListener('lostpointercapture', this.onUp);
    new ResizeObserver(() => this.resize()).observe(root);
    this.setVisible(false);
  }

  setVisible(v: boolean): void {
    this.visible = v;
    this.renderer.domElement.style.display = v ? 'block' : 'none';
    this.labels.domElement.style.display = v ? 'block' : 'none';
    if (v) {
      this.resize();
      this.dirty = true;
    }
  }

  private resize(): void {
    const r = this.root.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return;
    this.w = r.width;
    this.h = r.height;
    this.renderer.setSize(r.width, r.height);
    this.labels.setSize(r.width, r.height);
    const orbit = new THREE.Spherical().setFromVector3(this.camera.position.clone().sub(this.controls.target));
    const previousFit = this.cameraPlaced ? this.fittedView(orbit.phi, orbit.theta).radius : 1;
    const aspectChanged = Math.abs(this.camera.aspect - r.width/r.height) > 0.001;
    this.camera.aspect = r.width / r.height;
    this.camera.updateProjectionMatrix();
    for (const m of this.lineMats) m.resolution.set(r.width, r.height);
    if (!this.cameraPlaced) {
      // The default view depends on the viewport, so place it on the first real size.
      this.cameraPlaced = true;
      this.resetCamera(false);
    } else if (aspectChanged) {
      const fit = this.fittedView(orbit.phi, orbit.theta).radius;
      this.controls.maxDistance = Math.max(48, fit * 1.5);
      orbit.radius *= fit / previousFit;
      this.camera.position.setFromSpherical(orbit).add(this.controls.target);
      this.controls.update();
    }
    this.dirty = true;
  }

  private cameraPlaced = false;

  private lineMat(color: string, width: number, opts: Partial<THREE.MaterialParameters> & { dashed?: boolean } = {}): LineMaterial {
    const m = new LineMaterial({
      color: new THREE.Color(color).getHex(),
      linewidth: width,
      worldUnits: false,
      transparent: opts.transparent ?? false,
      opacity: opts.opacity ?? 1,
      depthTest: opts.depthTest ?? true,
      depthWrite: opts.depthWrite ?? true,
      dashed: opts.dashed ?? false,
    });
    m.resolution.set(this.w, this.h);
    this.lineMats.push(m);
    return m;
  }

  // --- static scene: floor, grid, axes ------------------------------------
  private buildStatic(): void {
    const margin = 1.1;
    const size = DOMAIN.x1 - DOMAIN.x0 + 2 * margin;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshBasicMaterial({ color: THEME.bgRaised }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.004;
    this.scene.add(floor);

    const grid: number[] = [];
    for (let t = DOMAIN.x0; t <= DOMAIN.x1 + 1e-9; t += 1) {
      grid.push(...world([t, DOMAIN.y0 - margin]).toArray(), ...world([t, DOMAIN.y1 + margin]).toArray());
      grid.push(...world([DOMAIN.x0 - margin, t]).toArray(), ...world([DOMAIN.x1 + margin, t]).toArray());
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(grid, 3));
    this.scene.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: THEME.ink, transparent: true, opacity: 0.08 })));

    // Domain outline on the floor.
    const corners: Vec2[] = [
      [DOMAIN.x0, DOMAIN.y0],
      [DOMAIN.x1, DOMAIN.y0],
      [DOMAIN.x1, DOMAIN.y1],
      [DOMAIN.x0, DOMAIN.y1],
      [DOMAIN.x0, DOMAIN.y0],
    ];
    const og = new LineGeometry();
    og.setPositions(corners.flatMap((c) => world(c, 0).toArray()));
    this.scene.add(new Line2(og, this.lineMat(THEME.ink, 1.5, { transparent: true, opacity: 0.4 })));

    // Tick labels on the floor edges.
    for (let t = 0; t <= 10; t += 2) {
      const a = makeLabel('tick3d', String(t));
      a.position.copy(world([t, DOMAIN.y0 - 0.55]));
      this.scene.add(a);
      const b = makeLabel('tick3d', String(t));
      b.position.copy(world([DOMAIN.x0 - 0.55, t]));
      this.scene.add(b);
    }
    const ax1 = makeLabel('axis3d', 'x<sub>1</sub>');
    ax1.position.copy(world([DOMAIN.x1 + 0.9, DOMAIN.y0 - 0.6]));
    const ax2 = makeLabel('axis3d', 'x<sub>2</sub>');
    ax2.position.copy(world([DOMAIN.x0 - 1.25, DOMAIN.y1 + 1.4]));
    this.scene.add(ax1, ax2);

    // Backdrop: horizontal value lines on the two far sides + a vertical f̂ axis
    // in their shared corner. Sides are chosen from the camera position.
    const sides: Record<'left' | 'right' | 'front' | 'back', [Vec2, Vec2]> = {
      left: [[DOMAIN.x0, DOMAIN.y0], [DOMAIN.x0, DOMAIN.y1]],
      right: [[DOMAIN.x1, DOMAIN.y0], [DOMAIN.x1, DOMAIN.y1]],
      front: [[DOMAIN.x0, DOMAIN.y0], [DOMAIN.x1, DOMAIN.y0]],
      back: [[DOMAIN.x0, DOMAIN.y1], [DOMAIN.x1, DOMAIN.y1]],
    };
    for (const [name, [p, q]] of Object.entries(sides) as [keyof typeof sides, [Vec2, Vec2]][]) {
      const grp = new THREE.Group();
      const pos: number[] = [];
      for (let f = 2; f <= HEIGHT_MAX + 1e-9; f += 2) pos.push(...world(p, f).toArray(), ...world(q, f).toArray());
      pos.push(...world(p, 0).toArray(), ...world(p, HEIGHT_MAX).toArray(), ...world(q, 0).toArray(), ...world(q, HEIGHT_MAX).toArray());
      const bg = new THREE.BufferGeometry();
      bg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      grp.add(new THREE.LineSegments(bg, new THREE.LineBasicMaterial({ color: THEME.ink, transparent: true, opacity: 0.14 })));
      this.backdrop[name] = grp;
      this.scene.add(grp);
    }
    for (const corner of ['lf', 'lb', 'rf', 'rb']) {
      const c: Vec2 = [corner[0] === 'l' ? DOMAIN.x0 : DOMAIN.x1, corner[1] === 'f' ? DOMAIN.y0 : DOMAIN.y1];
      const grp = new THREE.Group();
      const lg = new LineGeometry();
      lg.setPositions([...world(c, 0).toArray(), ...world(c, HEIGHT_MAX).toArray()]);
      grp.add(new Line2(lg, this.lineMat(THEME.ink, 2, { transparent: true, opacity: 0.7 })));
      for (let f = 0; f <= HEIGHT_MAX + 1e-9; f += 2) {
        const l = makeLabel('tick3d vtick', String(f));
        l.position.copy(world(c, f));
        grp.add(l);
      }
      const title = makeLabel('axis3d', 'f&#770;');
      title.position.copy(world(c, HEIGHT_MAX + 0.9));
      grp.add(title);
      this.vAxes[corner] = grp;
      this.scene.add(grp);
    }
  }

  private updateBackdrop(): void {
    const p = this.camera.position;
    const left = p.x > 0; // camera on the right: the left side is far
    const back = p.z > 0;
    this.backdrop.left.visible = left;
    this.backdrop.right.visible = !left;
    this.backdrop.back.visible = back;
    this.backdrop.front.visible = !back;
    // The value axis stands at the outer end of a far side that appears
    // leftmost on screen: a side corner, never the occluded far corner.
    const xs = left ? DOMAIN.x0 : DOMAIN.x1;
    const ys = back ? DOMAIN.y1 : DOMAIN.y0;
    const c1: Vec2 = [xs, back ? DOMAIN.y0 : DOMAIN.y1];
    const c2: Vec2 = [left ? DOMAIN.x1 : DOMAIN.x0, ys];
    const pick = this.screenOf(world(c1, 0))[0] < this.screenOf(world(c2, 0))[0] ? c1 : c2;
    const key = `${pick[0] === DOMAIN.x0 ? 'l' : 'r'}${pick[1] === DOMAIN.y0 ? 'f' : 'b'}`;
    for (const [k, g] of Object.entries(this.vAxes)) {
      g.visible = k === key;
      g.traverse((o) => {
        if (o instanceof CSS2DObject) o.visible = k === key;
      });
    }
  }

  // --- x0 marker ------------------------------------------------------------
  private buildX0(): void {
    // A diamond, so x0 never looks like one of the round observations.
    const core = new THREE.Mesh(new THREE.OctahedronGeometry(7), new THREE.MeshBasicMaterial({ color: THEME.accent }));
    const outline = new THREE.Mesh(new THREE.OctahedronGeometry(8), new THREE.MeshBasicMaterial({ color: THEME.outline, side: THREE.BackSide }));
    this.x0Marker = new THREE.Group();
    this.x0Marker.add(core, outline);
    this.x0Ghost = ghostify(new THREE.Group().add(core.clone()), 0.45);
    const gg = new LineGeometry();
    gg.setPositions([0, 0, 0, 0, 1, 0]);
    this.x0Guide = new Line2(gg, this.lineMat(THEME.accent, 3));
    this.x0GuideGhost = new Line2(gg, this.lineMat(THEME.accent, 2, { transparent: true, opacity: 0.45, depthTest: false, depthWrite: false }));
    this.x0GuideGhost.renderOrder = 20;
    // Floor handle: ring + crosshair, drawn see-through (the surface covers it).
    const ring = new THREE.Mesh(new THREE.RingGeometry(3, 4, 48), new THREE.MeshBasicMaterial({ color: THEME.accent, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    this.x0Floor = ghostify(new THREE.Group().add(ring), 0.45);
    this.x0Label = makeLabel('x0label', 'x<sub>0</sub>');
    this.scene.add(this.x0Marker, this.x0Ghost, this.x0Guide, this.x0GuideGhost, this.x0Floor, this.x0Label);
  }

  // Dispose resources owned by an old observation set or path once, even if shared inside the group.
  private clearGroup(group: THREE.Group) {
    const geometries = new Set<THREE.BufferGeometry>();
    const materials = new Set<THREE.Material>();
    group.traverse(object=>{
      if(object instanceof CSS2DObject) object.element.remove();
      const {material, geometry}=object as THREE.Mesh;
      if(material) for(const m of Array.isArray(material)?material:[material]) materials.add(m);
      if(geometry && !(object instanceof THREE.Sprite)) geometries.add(geometry);
    });
    for(const g of [this.sharedSphere,this.sharedOutline]) geometries.delete(g);
    for(const g of geometries) g.dispose();
    for(const material of materials) material.dispose();
    this.lineMats=this.lineMats.filter(m=>!materials.has(m));
    group.clear();
  }

  // --- observations ------------------------------------------------------------
  private buildPoints(s: State): void {
    this.clearGroup(this.points);
    this.pointViz = [];
    const outlineMat = new THREE.MeshBasicMaterial({ color: THEME.outline, side: THREE.BackSide });
    const ghostOutlineMat = new THREE.MeshBasicMaterial({ color: THEME.ink, side: THREE.BackSide, transparent: true, opacity: 0.35, depthTest: false, depthWrite: false });
    const stemMat = new THREE.LineBasicMaterial({ color: THEME.ink, transparent: true, opacity: 0.5 });
    const stemGhostMat = new THREE.LineBasicMaterial({ color: THEME.ink, transparent: true, opacity: 0.22, depthTest: false, depthWrite: false });
    const floorGeo = new THREE.CircleGeometry(2.5, 24);
    s.X.forEach((_, i) => {
      const col = colorHex(s.Y[i], valueScale(s));
      const sphere = new THREE.Mesh(this.sharedSphere, new THREE.MeshBasicMaterial({ color: col }));
      const outline = new THREE.Mesh(this.sharedOutline, outlineMat);
      const ghost = new THREE.Mesh(this.sharedSphere, new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.5, depthTest: false, depthWrite: false }));
      ghost.renderOrder = 18;
      const ghostRing = new THREE.Mesh(this.sharedOutline, ghostOutlineMat);
      ghostRing.renderOrder = 17;
      ghost.add(ghostRing);
      const floor = new THREE.Mesh(floorGeo, new THREE.MeshBasicMaterial({ color: THEME.ink, transparent: true, opacity: 0.45, depthTest: false, depthWrite: false, side: THREE.DoubleSide }));
      floor.rotation.x = -Math.PI / 2;
      floor.renderOrder = 18;
      const sg = new THREE.BufferGeometry();
      sg.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0, 1, 0], 3));
      const stem = new THREE.Line(sg, stemMat);
      const stemGhost = new THREE.Line(sg, stemGhostMat);
      stemGhost.renderOrder = 17;
      const lg = new LineGeometry();
      lg.setPositions([0, 0, 0, 1, 1, 1]);
      const link = new Line2(lg, this.lineMat(THEME.accent, 3, { transparent: true }));
      const linkGhost = new Line2(lg, this.lineMat(THEME.accent, 2, { transparent: true, opacity: 0.3, depthTest: false, depthWrite: false }));
      linkGhost.renderOrder = 19;
      const label = makeLabel('obslabel');
      this.points.add(stem, stemGhost, floor, sphere, outline, ghost, link, linkGhost, label);
      this.pointViz.push({ sphere, outline, ghost, floor, stem, stemGhost, link, linkGhost, label });
    });
  }

  private updatePoints(s: State): void {
    const key = `${s.presetIndex}|${s.X.length}`;
    if (key !== this.pointKey) {
      this.pointKey = key;
      this.buildPoints(s);
    }
    const pred = prediction(s);
    const VS = valueScale(s);
    const top = world(s.x0, pred.value);
    let wmax = 0;
    for (const w of pred.weights) wmax = Math.max(wmax, w);
    s.X.forEach((x, i) => {
      const v = this.pointViz[i];
      const obs = world(x, s.Y[i]);
      const col = colorHex(s.Y[i], VS);
      (v.sphere.material as THREE.MeshBasicMaterial).color.setHex(col);
      (v.ghost.material as THREE.MeshBasicMaterial).color.setHex(col);
      v.sphere.position.copy(obs);
      v.outline.position.copy(obs);
      v.ghost.position.copy(obs);
      v.floor.position.copy(world(x, 0)).setY(0.01);
      v.stem.position.copy(world(x, 0));
      v.stem.scale.set(1, Math.max(1e-3, VZ * s.Y[i]), 1);
      v.stemGhost.position.copy(v.stem.position);
      v.stemGhost.scale.copy(v.stem.scale);
      const w = pred.weights[i];
      const rel = wmax > 0 ? w / wmax : 0;
      const on = w > 0;
      v.link.visible = on && !(pred.kind === "knn" && pred.k===s.X.length);
      v.linkGhost.visible = v.link.visible;
      if (on) {
        moveSegment(v.link,top,obs);
        const lm = v.link.material as LineMaterial;
        lm.opacity = pred.kind === 'nw' ? 0.2 + 0.8 * rel : 1;
        lm.linewidth = pred.kind === 'nw' ? 1 + 3 * rel : 3;
        (v.linkGhost.material as LineMaterial).opacity = pred.kind === 'nw' ? 0.25 * rel : 0.3;
      }
      let html = '';
      let prio = 0;
      if (s.drag?.kind === 'point' && s.drag.index === i) {
        html = `X<sub>${i + 1}</sub> moves; Y<sub>${i + 1}</sub> = ${fmt(s.Y[i], 1)} fixed`;
        prio = 3;
      } else if (s.includeOutlier && i === OUTLIER) {
        html = `◇ Outlier · Y<sub>${i+1}</sub> = ${fmt(s.Y[i],1)}`;
        prio = 2;
      } else if (s.highlightObs === i) {
        html = `Y<sub>${i + 1}</sub> = ${fmt(s.Y[i], 2)}`;
        prio = 2;
      } else if (pred.kind === 'knn' && i === pred.neighbors[0] && pred.k <= 6) {
        html = `Y<sub>${i + 1}</sub> = ${fmt(s.Y[i], 1)}`;
        prio = 1;
      }
      setLabel(v.label, html);
      v.label.element.classList.toggle('strong', prio >= 2);
      this.labelInfo[i] = { text: html.replace(/<[^>]+>/g, ''), prio };
      v.label.position.copy(obs);
    });

    // x0: base on the feature plane, marker exactly at f̂(x0), vertical guide.
    const base = world(s.x0, 0);
    this.x0Marker.position.copy(top);
    this.x0Ghost.position.copy(top);
    this.x0Floor.position.copy(base).setY(0.015);
    moveSegment(this.x0Guide,base,top);
    this.x0Label.position.copy(top).add(new THREE.Vector3(0, 0.55, 0));
  }

  // --- surfaces -----------------------------------------------------------------
  private disposeSurface(): void {
    this.surface.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.geometry || m === this.nwMesh || m === this.nwSkirt) return;
      m.geometry.dispose();
    });
    this.surface.clear();
  }

  private buildStepSurface(diag: CellDiagram, VS: { lo: number; hi: number }, highlight: number | null): void {
    const tops: number[] = [];
    const marks: number[] = [];
    const vals: number[] = [];
    const walls: number[] = [];
    const wallUV: number[] = [];
    const wallValues: number[] = [];
    const hatchStrength: number[] = [];
    const rims: number[] = [];
    for (const c of diag.cells) {
      const z = VZ * c.value;
      const P = c.polygon.map((v) => world(v, c.value));
      for (let i = 1; i + 1 < P.length; i++) {
        tops.push(...P[0].toArray(), ...P[i].toArray(), ...P[i + 1].toArray());
        vals.push(c.value, c.value, c.value);
      }
      const marked = highlight !== null && c.set.includes(highlight);
      for (const e of c.edges) {
        rims.push(...world(e.p, c.value).toArray(), ...world(e.q, c.value).toArray());
        if (marked) marks.push(...world(e.p, c.value).toArray(), ...world(e.q, c.value).toArray());
        // Vertical wall down to the floor, owned by the higher side of a jump
        // (or the domain boundary). No sloped surface ever bridges two plateaus.
        if (!e.boundary && (e.neighborValue as number) >= c.value - 1e-12) continue;
        const base = e.neighborValue ?? 0;
        const a0 = world(e.p, base);
        const b0 = world(e.q, base);
        const a1 = a0.clone().setY(z);
        const b1 = b0.clone().setY(z);
        walls.push(...a1.toArray(), ...a0.toArray(), ...b0.toArray(), ...a1.toArray(), ...b0.toArray(), ...b1.toArray());
        const dx = e.q[0] - e.p[0];
        const dy = e.q[1] - e.p[1];
        const L = Math.hypot(dx, dy) || 1;
        const low = base * VZ;
        wallUV.push(0,z, 0,low, L,low, 0,z, L,low, L,z);
        for (let i=0;i<6;i++) {wallValues.push(c.value);hatchStrength.push(e.boundary ? 0.38 : 1);}
      }
    }
    const tg = new THREE.BufferGeometry();
    tg.setAttribute('position', new THREE.Float32BufferAttribute(tops, 3));
    tg.setAttribute('aValue', new THREE.Float32BufferAttribute(vals, 1));
    const pm = this.stepMaterials.top;
    pm.uniforms.uLo.value = VS.lo;
    pm.uniforms.uHi.value = VS.hi;
    const topMesh = new THREE.Mesh(tg, pm);
    const wg = new THREE.BufferGeometry();
    wg.setAttribute('position', new THREE.Float32BufferAttribute(walls, 3));
    wg.setAttribute('uv',new THREE.Float32BufferAttribute(wallUV,2));
    wg.setAttribute('aValue',new THREE.Float32BufferAttribute(wallValues,1));
    wg.setAttribute('aStrength',new THREE.Float32BufferAttribute(hatchStrength,1));
    const hatch = this.stepMaterials.wall;
    hatch.uniforms.uLo.value=VS.lo;hatch.uniforms.uHi.value=VS.hi;
    const wallMesh=new THREE.Mesh(wg,hatch);
    const rg = new LineSegmentsGeometry();
    rg.setPositions(rims);
    const rimLines = new LineSegments2(rg, this.stepMaterials.rim);
    this.surface.add(topMesh, wallMesh, rimLines);
    if (marks.length) {
      // Footprint of one observation: light outline of every plateau it takes part in.
      const mg = new LineSegmentsGeometry();
      mg.setPositions(marks);
      const halo = new LineSegments2(mg, this.stepMaterials.halo);
      const ink = new LineSegments2(mg, this.stepMaterials.selected);
      halo.renderOrder = 2;
      ink.renderOrder = 3;
      this.surface.add(halo, ink);
    }
  }

  private ensureNwMesh(): void {
    if (this.nwMesh) return;
    const R = NW_RES;
    const pos = new Float32Array((R + 1) * (R + 1) * 3);
    const feat = new Float32Array((R + 1) * (R + 1) * 2);
    const idx: number[] = [];
    for (let j = 0; j <= R; j++)
      for (let i = 0; i <= R; i++) {
        const k = j * (R + 1) + i;
        const x1 = DOMAIN.x0 + (i / R) * (DOMAIN.x1 - DOMAIN.x0);
        const x2 = DOMAIN.y0 + (j / R) * (DOMAIN.y1 - DOMAIN.y0);
        feat[2 * k] = x1;
        feat[2 * k + 1] = x2;
        pos[3 * k] = x1 - CX;
        pos[3 * k + 2] = -(x2 - CY);
      }
    for (let j = 0; j < R; j++)
      for (let i = 0; i < R; i++) {
        const a = j * (R + 1) + i;
        const b = a + 1;
        const c = a + R + 1;
        const d = c + 1;
        idx.push(a, c, b, b, c, d);
      }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aFeat', new THREE.BufferAttribute(feat, 2));
    g.setIndex(idx);
    this.nwMesh = new THREE.Mesh(g, nwMaterial());
    // Boundary skirt (neutral wall) from the surface edge down to the floor.
    const n = 4 * R;
    const sp = new Float32Array((n + 1) * 2 * 3);
    const sn = new Float32Array((n + 1) * 2 * 3);
    const si: number[] = [];
    for (let m = 0; m < n; m++) si.push(2 * m, 2 * m + 1, 2 * m + 2, 2 * m + 1, 2 * m + 3, 2 * m + 2);
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(sp, 3));
    sg.setAttribute('normal', new THREE.BufferAttribute(sn, 3));
    sg.setIndex(si);
    this.nwSkirt = new THREE.Mesh(sg, wallMaterial());
  }

  private boundaryLoop(): Vec2[] {
    const R = NW_RES;
    const pts: Vec2[] = [];
    const L = DOMAIN.x1 - DOMAIN.x0;
    for (let i = 0; i < R; i++) pts.push([DOMAIN.x0 + (i / R) * L, DOMAIN.y0]);
    for (let i = 0; i < R; i++) pts.push([DOMAIN.x1, DOMAIN.y0 + (i / R) * L]);
    for (let i = 0; i < R; i++) pts.push([DOMAIN.x1 - (i / R) * L, DOMAIN.y1]);
    for (let i = 0; i <= R; i++) pts.push([DOMAIN.x0, DOMAIN.y1 - (i / R) * L]);
    return pts;
  }

  private buildNwSurface(s: State): void {
    this.ensureNwMesh();
    const mesh = this.nwMesh!;
    const pk = pack(s);
    const scratch = new Float64Array(pk.n);
    const pos = mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
    const feat = mesh.geometry.getAttribute('aFeat') as THREE.BufferAttribute;
    const stride = NW_RES + 1;
    const xs = Float64Array.from({length:stride}, (_,i)=>feat.getX(i));
    const ys = Float64Array.from({length:stride}, (_,j)=>feat.getY(j*stride));
    const heights = nwGrid(pk,xs,ys,s.h);
    for (let k = 0; k < pos.count; k++) pos.setY(k, VZ * heights[k]);
    pos.needsUpdate = true;
    mesh.geometry.computeVertexNormals();
    mesh.geometry.computeBoundingSphere();
    const u = (mesh.material as THREE.ShaderMaterial).uniforms;
    for (let i = 0; i < MAX_POINTS; i++) {
      if (i < s.X.length) {
        u.uPts.value[i].set(s.X[i][0], s.X[i][1]);
        u.uYs.value[i] = s.Y[i];
      }
    }
    u.uN.value = s.X.length;
    u.uH.value = s.h;
    u.uLo.value = valueScale(s).lo;
    u.uHi.value = valueScale(s).hi;
    const skirt = this.nwSkirt!;
    const sp = skirt.geometry.getAttribute('position') as THREE.BufferAttribute;
    const sn = skirt.geometry.getAttribute('normal') as THREE.BufferAttribute;
    const loop = this.boundaryLoop();
    loop.forEach((x, m) => {
      const top = world(x, nwValue(pk, x[0], x[1], s.h, scratch));
      sp.setXYZ(2 * m, top.x, top.y, top.z);
      sp.setXYZ(2 * m + 1, top.x, 0, top.z);
      const nx = x[0] <= DOMAIN.x0 + 1e-9 ? -1 : x[0] >= DOMAIN.x1 - 1e-9 ? 1 : 0;
      const nz = nx !== 0 ? 0 : x[1] <= DOMAIN.y0 + 1e-9 ? 1 : -1;
      sn.setXYZ(2 * m, nx, 0, nz);
      sn.setXYZ(2 * m + 1, nx, 0, nz);
    });
    sp.needsUpdate = true;
    sn.needsUpdate = true;
    skirt.geometry.computeBoundingSphere();
    this.surface.add(mesh, skirt);
  }

  private updateSurface(s: State): void {
    const k = effectiveK(s);
    const VS = valueScale(s);
    const key = s.method === 'nw' ? `nw|${s.dataVersion}|${s.presetIndex}|${s.h}|${VS.lo}|${VS.hi}` : `step|${s.dataVersion}|${s.presetIndex}|${k}|${VS.lo}|${VS.hi}|${s.highlightObs}`;
    if (key !== this.surfaceKey) {
      this.surfaceKey = key;
      this.disposeSurface();
      if (s.method === 'nw') this.buildNwSurface(s);
      else this.buildStepSurface(cells(s, k), VS, s.highlightObs);
    }
  }

  // --- path and slice -------------------------------------------------------------
  private updatePath(s: State): void {
    const next = upcomingCrossing(s);
    this.crossing.visible = !!next;
    if (next) this.crossing.position.copy(world(next,predictAt(s,next).value));
    this.pathGroup.visible = s.profileOpen;
    if (!s.profileOpen) return;
    const prof = profile(s);
    const key = `${s.dataVersion}|${s.method}|${effectiveK(s)}|${s.h}|${s.pathA}|${s.pathB}`;
    if (key === this.pathKey) return;
    this.pathKey = key;
    this.clearGroup(this.pathGroup);
    this.abHandles = [];
    this.abLabels = [];

    const fg = new LineGeometry();
    fg.setPositions([...world(s.pathA, 0).toArray(), ...world(s.pathB, 0).toArray()]);
    const floorLine = new Line2(fg, this.lineMat(THEME.path, 2, { transparent: true, opacity: 0.7, depthTest: false, depthWrite: false }));
    floorLine.renderOrder = 16;
    this.pathGroup.add(floorLine);
    for (const [p, name] of [
      [s.pathA, 'A'],
      [s.pathB, 'B'],
    ] as const) {
      const box = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.06, 0.28), new THREE.MeshBasicMaterial({ color: THEME.path, depthTest: false, transparent: true }));
      box.renderOrder = 21;
      box.position.copy(world(p, 0)).setY(0.03);
      this.abHandles.push(box);
      const lab = makeLabel('ablabel', name);
      lab.position.copy(world(p, 0)).add(new THREE.Vector3(0, 0.35, 0));
      this.abLabels.push(lab);
      this.pathGroup.add(box, lab);
    }

    // The slice g(s) = f̂(γ(s)) drawn on the surface itself.
    const lift = 0.03;
    const at = (t: number, v: number) => world(pointOnPath(s.pathA, s.pathB, t), v).add(new THREE.Vector3(0, lift, 0));
    if (prof.kind === 'step') {
      const segs: number[] = [];
      const jumps: number[] = [];
      prof.segments.forEach((g, i) => {
        segs.push(...at(g.s0, g.value).toArray(), ...at(g.s1, g.value).toArray());
        const nx = prof.segments[i + 1];
        if (nx) jumps.push(...at(g.s1, g.value).toArray(), ...at(g.s1, nx.value).toArray());
      });
      const halo = new LineSegmentsGeometry();
      halo.setPositions(segs);
      this.pathGroup.add(new LineSegments2(halo, this.lineMat(THEME.outline, 7)));
      this.pathGroup.add(new LineSegments2(halo, this.lineMat(THEME.ink, 3.5)));
      const ghost = new LineSegments2(halo, this.lineMat(THEME.ink, 2, { transparent: true, opacity: 0.28, depthTest: false, depthWrite: false }));
      ghost.renderOrder = 15;
      this.pathGroup.add(ghost);
      if (jumps.length) {
        const jg = new LineSegmentsGeometry();
        jg.setPositions(jumps);
        const jl = new LineSegments2(jg, this.lineMat(THEME.ink, 1.5, { transparent: true, opacity: 0.55, dashed: true }));
        const jm = jl.material as LineMaterial;
        jm.dashSize = 0.08;
        jm.gapSize = 0.07;
        jl.computeLineDistances();
        this.pathGroup.add(jl);
      }
    } else {
      const pts: number[] = [];
      const stride = Math.max(1, Math.floor(prof.s.length / 1500));
      for (let i = 0; i < prof.s.length; i += stride) pts.push(...at(prof.s[i], prof.v[i]).toArray());
      const last = prof.s.length - 1;
      if (last % stride !== 0) pts.push(...at(prof.s[last], prof.v[last]).toArray());
      const lg = new LineGeometry();
      lg.setPositions(pts);
      this.pathGroup.add(new Line2(lg, this.lineMat(THEME.outline, 7)));
      this.pathGroup.add(new Line2(lg, this.lineMat(THEME.ink, 3.5)));
      const ghost = new Line2(lg, this.lineMat(THEME.ink, 2, { transparent: true, opacity: 0.28, depthTest: false, depthWrite: false }));
      ghost.renderOrder = 15;
      this.pathGroup.add(ghost);
    }
  }

  update(s: State, _changed: Set<Key>): void {
    if (!this.visible) return;
    this.updateSurface(s);
    this.updatePoints(s);
    this.updatePath(s);
    this.dirty = true;
  }

  // --- camera ----------------------------------------------------------------------
  private target(): THREE.Vector3 {
    return new THREE.Vector3(0, VZ * 3.2, 0);
  }

  /** Fit projected floor and height-axis corners, including room for tick labels. */
  private fittedView(phi: number, theta: number): THREE.Spherical {
    const forward = new THREE.Vector3().setFromSphericalCoords(1, phi, theta);
    const right = new THREE.Vector3().crossVectors(this.camera.up, forward).normalize();
    const up = new THREE.Vector3().crossVectors(forward, right);
    const slope = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const target = this.target();
    let radius = this.controls.minDistance;
    for (const height of [0, VZ * (HEIGHT_MAX + 0.9)]) {
      const extent = height === 0 ? 6.1 : 5;
      for (const x of [-extent, extent]) {
        for (const z of [-extent, extent]) {
          const offset = new THREE.Vector3(x, height, z).sub(target);
          const width = Math.abs(offset.dot(right)) / this.camera.aspect;
          const depth = offset.dot(forward);
          radius = Math.max(radius, depth + 1.10 * Math.max(width, Math.abs(offset.dot(up))) / slope);
        }
      }
    }
    return new THREE.Spherical(radius, phi, theta);
  }

  resetCamera(animate = true): void {
    this.flyTo(this.fittedView(0.98, -0.62), animate);
  }

  topView(animate = true): void {
    this.flyTo(this.fittedView(1e-4, 0), animate);
  }

  /** Drop any leftover orbit/zoom inertia so a camera move ends exactly where asked. */
  private killInertia(): void {
    this.controls.enableDamping = false;
    this.controls.update();
    this.controls.enableDamping = true;
  }

  private flyTo(to: THREE.Spherical, animate: boolean): void {
    const target = this.target();
    this.controls.maxDistance = Math.max(48, to.radius * 1.5);
    this.killInertia();
    if (!animate) {
      this.controls.target.copy(target);
      this.camera.position.copy(new THREE.Vector3().setFromSpherical(to).add(target));
      this.controls.update();
      this.dirty = true;
      return;
    }
    const from = new THREE.Spherical().setFromVector3(this.camera.position.clone().sub(this.controls.target));
    // Rotate the short way round.
    let dTheta = to.theta - from.theta;
    while (dTheta > Math.PI) dTheta -= 2 * Math.PI;
    while (dTheta < -Math.PI) dTheta += 2 * Math.PI;
    const t = to.clone();
    t.theta = from.theta + dTheta;
    this.tween = { from, to: t, t0: performance.now(), dur: 700, target };
  }

  // --- picking & dragging ------------------------------------------------------------
  private ndc(e: PointerEvent): THREE.Vector2 {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, 1 - ((e.clientY - r.top) / r.height) * 2);
  }

  /** Screen position (CSS px, canvas-relative) of a world point. */
  screenOf(v: THREE.Vector3): [number, number] {
    const p = v.clone().project(this.camera);
    return [((p.x + 1) / 2) * this.w, ((1 - p.y) / 2) * this.h];
  }

  /** Handles with their world anchors; a handle may be grabbed at either anchor. */
  private anchors(s: State): { target: DragTarget; at: THREE.Vector3; radius: number }[] {
    const pred = prediction(s);
    const out: { target: DragTarget; at: THREE.Vector3; radius: number }[] = [
      { target: { kind: 'x0' }, at: world(s.x0, pred.value), radius: 16 },
      { target: { kind: 'x0' }, at: world(s.x0, 0), radius: 16 },
    ];
    if (s.profileOpen && !s.experiment) {
      out.push({ target: { kind: 'pathA' }, at: world(s.pathA, 0), radius: 20 });
      out.push({ target: { kind: 'pathB' }, at: world(s.pathB, 0), radius: 20 });
    }
    if (s.editData) {
      s.X.forEach((x, i) => {
        out.push({ target: { kind: 'point', index: i }, at: world(x, s.Y[i]), radius: 20 });
        out.push({ target: { kind: 'point', index: i }, at: world(x, 0), radius: 14 });
      });
    }
    return out;
  }

  pickAt(px: number, py: number): { target: DragTarget; at: THREE.Vector3 } | null {
    const s = this.store.get();
    let best: { target: DragTarget; at: THREE.Vector3 } | null = null;
    let bestScore = Infinity;
    for (const a of this.anchors(s)) {
      const [sx, sy] = this.screenOf(a.at);
      const d = Math.hypot(sx - px, sy - py);
      if (d > a.radius) continue;
      // x0 wins ties (it is what the lecture is about), except in Edit data.
      const score = d;
      if (score < bestScore) {
        bestScore = score;
        best = { target: a.target, at: a.at };
      }
    }
    return best;
  }

  private planeHit(ndc: THREE.Vector2, plane: THREE.Plane): THREE.Vector3 | null {
    this.raycaster.setFromCamera(ndc, this.camera);
    const hit = new THREE.Vector3();
    return this.raycaster.ray.intersectPlane(plane, hit);
  }

  private onDown = (e: PointerEvent) => {
    if (!this.visible || this.drag || this.verticalDrag || e.button !== 0 || document.body.dataset.mode !== 'explore') return;
    if (e.target !== this.renderer.domElement) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const state = this.store.get();
    const px = e.clientX-r.left, py = e.clientY-r.top;
    const hit = this.pickAt(px, py);
    const index = this.siteAt(px, py);
    const distanceTo = (p: THREE.Vector3) => {
      const [x,y] = this.screenOf(p);
      return Math.hypot(px-x,py-y);
    };
    if(index>=0 && (!hit || distanceTo(world(state.X[index],state.Y[index])) <= distanceTo(hit.at))){
      const initial=state.Y[index], x=state.X[index];
      const dy=this.screenOf(world(x,initial))[1]-this.screenOf(world(x,initial+1))[1];
      if(dy>2){
        this.killInertia();this.controls.enabled=false;this.tween=null;
        this.verticalDrag={index,pointer:e.pointerId,startY:e.clientY,initial,pixelsPerUnit:dy};
        this.onResponseEdit(index);
        this.renderer.domElement.setPointerCapture(e.pointerId);
        e.preventDefault();return;
      }
    }
    if (!hit) return; // the camera gets the gesture
    this.controls.enabled = false;
    this.tween = null;
    // Drag on the horizontal plane through the grabbed anchor (ray casting).
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -hit.at.y);
    const p = this.planeHit(this.ndc(e), plane) ?? hit.at.clone();
    const hp = handlePosition(this.store.get(), hit.target);
    const f: Vec2 = [p.x + CX, -p.z + CY];
    this.drag = { target: hit.target, pointerId: e.pointerId, plane, offset: [hp[0] - f[0], hp[1] - f[1]] };
    beginDrag(this.store, hit.target);
    this.renderer.domElement.setPointerCapture(e.pointerId);
    this.renderer.domElement.style.cursor = 'grabbing';
    e.preventDefault();
  };

  private siteAt(px:number,py:number):number {
    const s=this.store.get();
    let distance=14, index=-1;
    s.X.forEach((x,i)=>{
      const [u,v]=this.screenOf(world(x,s.Y[i])),d=Math.hypot(u-px,v-py);
      if(d<distance){distance=d;index=i;}
    });
    return index;
  }

  private onMove = (e: PointerEvent) => {
    const response=this.verticalDrag;
    if(response && response.pointer===e.pointerId){
      const Y=[...this.store.get().Y];
      Y[response.index]=Math.round(10*Math.max(0,Math.min(10,response.initial+(response.startY-e.clientY)/response.pixelsPerUnit)))/10;
      this.store.set({Y});return;
    }
    if (this.drag && e.pointerId === this.drag.pointerId) {
      const p = this.planeHit(this.ndc(e), this.drag.plane);
      if (!p) return;
      applyDrag(this.store, this.drag.target, [p.x + CX + this.drag.offset[0], -p.z + CY + this.drag.offset[1]]);
      return;
    }
    if (e.buttons) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const hit = this.pickAt(e.clientX - r.left, e.clientY - r.top);
    const s=this.store.get();
    const i=this.siteAt(e.clientX-r.left,e.clientY-r.top);
    this.renderer.domElement.title=i<0?'':`${s.includeOutlier && i===OUTLIER?'Outlier · ':''}Y${i+1} = ${s.Y[i].toFixed(1)}`;
    this.renderer.domElement.style.cursor = i>=0 && document.body.dataset.mode==='explore' ? 'ns-resize' : hit ? 'grab' : 'default';
  };

  private onUp = (e: PointerEvent) => {
    if(this.verticalDrag?.pointer===e.pointerId){
      this.verticalDrag=null;this.controls.enabled=true;
      if(this.renderer.domElement.hasPointerCapture(e.pointerId))this.renderer.domElement.releasePointerCapture(e.pointerId);
      return;
    }
    if (!this.drag || e.pointerId !== this.drag.pointerId) return;
    this.drag = null;
    this.controls.enabled = true;
    if (this.renderer.domElement.hasPointerCapture(e.pointerId)) this.renderer.domElement.releasePointerCapture(e.pointerId);
    this.renderer.domElement.style.cursor = 'grab';
    endDrag(this.store);
  };

  get isDragging(): boolean {
    return this.drag !== null;
  }

  // --- frame ------------------------------------------------------------------------------
  frame(now: number): void {
    if (!this.visible) return;
    if (this.tween) {
      const { from, to, t0, dur, target } = this.tween;
      const u = Math.min(1, (now - t0) / dur);
      const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      const sp = new THREE.Spherical(
        from.radius + (to.radius - from.radius) * e,
        from.phi + (to.phi - from.phi) * e,
        from.theta + (to.theta - from.theta) * e,
      );
      this.controls.target.lerp(target, e);
      this.camera.position.copy(new THREE.Vector3().setFromSpherical(sp).add(this.controls.target));
      this.camera.lookAt(this.controls.target);
      if (u >= 1) {
        this.tween = null;
        this.killInertia();
      }
      this.dirty = true;
    } else if (this.controls.update()) this.dirty = true;
    if (!this.dirty) return;
    this.dirty = false;
    this.updateBackdrop();
    this.declutterLabels();
    // A camera-space depth converts one CSS pixel to world units at each anchor.
    this.camera.updateMatrixWorld();
    const pixelAngle = 2 * Math.tan(THREE.MathUtils.degToRad(this.camera.getEffectiveFOV()) / 2) / this.h;
    const depthAt = (p: THREE.Vector3) => -p.clone().applyMatrix4(this.camera.matrixWorldInverse).z * pixelAngle;
    for (const v of this.pointViz) {
      const unit = depthAt(v.sphere.position);
      for (const mesh of [v.sphere, v.outline, v.ghost]) mesh.scale.setScalar(unit);
      v.floor.scale.setScalar(depthAt(v.floor.position));
    }
    for (const mesh of [this.x0Marker, this.x0Ghost, this.x0Floor]) mesh.scale.setScalar(depthAt(mesh.position));
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  }

  /**
   * Observation labels sit up and to the right of their point. A label is shown
   * only if its box misses x0, x0's label and every label of higher priority.
   */
  private declutterLabels(): void {
    const s = this.store.get();
    const pred = prediction(s);
    type R = { x: number; y: number; w: number; h: number };
    const hit = (a: R, b: R) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
    const [tx, ty] = this.screenOf(world(s.x0, pred.value));
    const [lx, ly] = this.screenOf(this.x0Label.position);
    const taken: R[] = [
      { x: tx - 16, y: ty - 16, w: 32, h: 32 },
      { x: lx - 14, y: ly - 13, w: 28, h: 26 },
    ];
    const order = this.labelInfo
      .map((info, i) => ({ ...info, i }))
      .filter((o) => o.text)
      .sort((a, b) => b.prio - a.prio || a.i - b.i);
    const shown = new Set<number>();
    for (const o of order) {
      const [ax, ay] = this.screenOf(this.pointViz[o.i].label.position);
      const tw = 7.6 * o.text.length + 14;
      const W = 62 + tw;
      const H = 34 + 22;
      const r: R = { x: ax - W / 2 + 62, y: ay - H / 2, w: tw, h: 22 };
      const label = this.pointViz[o.i].label.element.firstElementChild as HTMLElement;
      const offsets = [[0,0],[-tw-6,0],[0,38],[-tw-6,38]];
      const offset = offsets.find(([dx,dy])=>{
        const box={...r,x:r.x+dx,y:r.y+dy};
        return box.x>=0 && box.y>=0 && box.x+tw<this.w && box.y+22<this.h && !taken.some(q=>hit(q,box));
      });
      if (!offset) continue;
      label.style.transform = `translate(${offset[0]}px, ${offset[1]}px)`;
      taken.push({...r,x:r.x+offset[0],y:r.y+offset[1]});
      shown.add(o.i);
    }
    this.pointViz.forEach((v, i) => {
      const on = shown.has(i);
      v.label.visible = on;
      v.label.element.style.display = on ? '' : 'none';
    });
  }

  /** Test hook: indices of observation labels shown in 3-D. */
  debugLabels(): number[] {
    return this.pointViz.flatMap((v, i) => (v.label.visible ? [i] : []));
  }

  /** Debug/test hook: screen coordinates of x0's surface marker and floor base. */
  debugAnchors(): Record<string, [number, number]> {
    const s = this.store.get();
    const pred = prediction(s);
    const out: Record<string, [number, number]> = {
      x0Top: this.screenOf(world(s.x0, pred.value)),
      x0Base: this.screenOf(world(s.x0, 0)),
    };
    s.X.forEach((x, i) => (out[`obs${i}`] = this.screenOf(world(x, s.Y[i]))));
    return out;
  }
}
