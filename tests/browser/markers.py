"""Exercise the built page under a repository subpath with real pointer input."""
from functools import partial
from io import BytesIO
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from itertools import product
from math import dist, exp, isclose
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from shutil import copytree
from PIL import Image, ImageChops, ImageDraw
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
LIVE_URL = os.environ.get('DEMO_TEST_URL')
OUT = ROOT / '.artifacts' / ('browser-public' if LIVE_URL else 'browser')


def check_marker_size(page, view):
    # Read the radius actually drawn on Canvas or projected by the WebGL camera.
    diameter=page.evaluate("""v => {
      if (v === '2d') {
        const map=__demo.v2, ctx=map.canvas.getContext('2d');
        const [cx,cy]=map.toPx(__demo.state().X[2]), radii=[];
        const arc=ctx.arc;
        ctx.arc=function(x,y,r,...args) {
          if (Math.hypot(x-cx,y-cy)<0.01) radii.push(r);
          return arc.call(this,x,y,r,...args);
        };
        try {map.draw();} finally {ctx.arc=arc;}
        return 2*Math.max(...radii);
      }
      const scene=__demo.v3, mesh=scene.pointViz[2].outline;
      const right=scene.camera.position.clone().setFromMatrixColumn(scene.camera.matrixWorld,0);
      const edge=mesh.position.clone().addScaledVector(right,mesh.geometry.parameters.radius*mesh.scale.x);
      return 2*Math.abs(scene.screenOf(edge)[0]-scene.screenOf(mesh.position)[0]);
    }""",view)
    assert 9.9 <= diameter <= 11.5, diameter


def check_site_pixels(page, node, phase):
    box = node.bounding_box()
    x, y = (round(box[c] + box[d] / 2) for c, d in [('x','width'), ('y','height')])
    # Capture the actual paint, including browser focus decorations outside the SVG bounds.
    crop = dict(x=x-24, y=y-24, width=48, height=48)
    shot = Image.open(BytesIO(page.screenshot(clip=crop, scale='css'))).convert('RGB')
    shot.save(OUT / f'voronoi-{page.viewport_size["width"]}-{phase}.png')
    cell = (247, 195, 204)
    mask = ImageChops.difference(shot, Image.new('RGB', shot.size, cell)).convert('L').point(lambda v: 255 if v else 0)
    ImageDraw.floodfill(mask, (24,24), 128)
    bounds = mask.point(lambda v: 255 if v == 128 else 0).getbbox()
    assert bounds and all(7 <= size <= 13 for size in (bounds[2]-bounds[0], bounds[3]-bounds[1])), (phase, bounds)
    # A detached outline can extend well beyond the connected central dot.
    probes = [(17,24),(31,24),(24,17),(24,31),(12,24),(36,24),(24,12),(24,36)]
    assert all(dist(shot.getpixel(point), cell) < 10 for point in probes), phase
    return shot.getpixel((24,24))


def inspect(page, width, height):
    page.set_viewport_size(dict(width=width, height=height))
    page.locator('[data-mode="explore"]').click()
    for method in ['1nn', 'knn', 'nw']:
        page.locator(f'[data-method="{method}"]').click()
        for view in ['2d', '3d']:
            page.locator(f'[data-view="{view}"]').click()
            page.wait_for_function("v => __demo.state().view === v", arg=view)
            page.evaluate('new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))')
            anchor = page.evaluate("""v => {
              const s=__demo.state(), canvas=document.querySelector('.view'+v);
              const rect=canvas.getBoundingClientRect();
              const a=v==='2d'?__demo.v2.toPx(s.X[2]):__demo.v3.debugAnchors().obs2;
              return [rect.x+a[0],rect.y+a[1]];
            }""", view)
            x,y=anchor
            before=page.evaluate('__demo.state()')
            check_marker_size(page,view)
            page.mouse.move(x+6,y)
            check_marker_size(page,view)
            page.mouse.down()
            check_marker_size(page,view)
            page.mouse.move(x+6+(12 if view=='2d' else 0), y-15, steps=4)
            after=page.evaluate('__demo.state()')
            if view=='2d':
                assert after['drag']=={'kind':'point','index':2}
                assert before['Y']==after['Y'] and before['X'][2]!=after['X'][2]
            else:
                assert before['X']==after['X'] and before['Y'][2]!=after['Y'][2]
            check_marker_size(page,view)
            page.mouse.up()
            check_marker_size(page,view)
            assert page.evaluate('__demo.state().drag') is None
            page.locator('#undo').click()
            assert page.evaluate('__demo.state().X')==before['X']
            assert page.evaluate('__demo.state().Y')==before['Y']
            print(f'PASS {width}x{height} {method} {view} pointer capture, move, release and undo',flush=True)
    page.screenshot(path=OUT/f'explore-{width}.png')
    page.locator('[data-mode="voronoi"]').click()
    node=page.locator('[data-site="0"]')
    box=node.bounding_box()
    assert 7 <= box['width'] <= 10, box
    x,y=box['x']+box['width']/2,box['y']+box['height']/2
    normal = check_site_pixels(page,node,'normal')
    page.mouse.move(x,y)
    check_site_pixels(page,node,'hover')
    page.mouse.down()
    check_site_pixels(page,node,'down')
    page.mouse.move(x+4,y-3,steps=3)
    check_site_pixels(page,node,'drag')
    page.mouse.up()
    check_site_pixels(page,node,'released')
    page.mouse.move(0,0)
    page.keyboard.press('Tab')
    page.keyboard.press('Shift+Tab')
    assert node.evaluate("e => e.matches(':focus-visible')")
    assert check_site_pixels(page,node,'keyboard') != normal
    left = node.get_attribute('cx')
    page.keyboard.press('ArrowRight')
    assert float(node.get_attribute('cx')) > float(left)
    page.locator('#reset-sites').click()
    near=page.locator('[data-site="1"]').bounding_box()
    end=near['x']+near['width']/2-7, near['y']+near['height']/2
    page.mouse.move(x-6,y)
    page.mouse.down()
    page.mouse.move(end[0]-6,end[1],steps=8)
    assert abs(node.bounding_box()['width']-box['width']) < .1
    page.mouse.up()
    # Both sites are inside the pick tolerance. Grab the closer one.
    page.mouse.move(end[0]+1,end[1])
    page.mouse.down()
    page.mouse.move(x,y,steps=5)
    page.mouse.up()
    assert page.locator('[data-site="1"]').bounding_box()==near
    assert abs(node.bounding_box()['x']+box['width']/2-x)<2, (box, near, end, node.bounding_box())
    page.locator('#reset-sites').click()
    print(f'PASS {width}x{height} Voronoi painted pixels, keyboard focus and nearest-site dragging',flush=True)
    page.locator('[data-mode="demo"]').click()
    page.locator('#experiment').select_option('6')
    page.wait_for_function("__demo.state().experiment?.id === 'hsweep'")


def check_moments(page):
    page.locator('[data-mode="explore"]').click()
    page.locator('[data-view="2d"]').click()
    page.locator('.meaning summary').click()
    for method, setting in [('knn','1'),('knn','8'),('nw','0.15'),('nw','6')]:
        page.locator(f'[data-method="{method}"]').click()
        page.wait_for_function("m => document.querySelector('#tuning').min === (m === 'nw' ? '0.15' : '1')",arg=method)
        page.locator('#tuning').fill(setting)
        page.wait_for_function("""([method,value]) => {
          const s=__demo.state();
          return s.method===method && (method==='nw'?s.h:s.kKnn)===Number(value);
        }""",arg=[method,setting])
        s=page.evaluate('__demo.state()')
        distances=[dist(s['x0'],point) for point in s['X']]
        if method=='knn':
            indices=sorted(range(8),key=distances.__getitem__)[:int(setting)]
            weights=[1/len(indices) if i in indices else 0 for i in range(8)]
        else:
            scores=[exp(-(d*d-min(distances)**2)/(2*float(setting)**2)) for d in distances]
            weights=[v/sum(scores) for v in scores]
        actual=page.evaluate('__demo.prediction().weights')
        assert all(isclose(a,b,abs_tol=1e-12) for a,b in zip(weights,actual))
        # Known f(x)=x1 and all 256 configurations of independent ±1 errors.
        truth=s['x0'][0]
        draws=[sum(w*(x[0]+e) for w,x,e in zip(weights,s['X'],errors))
               for errors in product((-1,1),repeat=8)]
        mean=sum(draws)/len(draws)
        bias=mean-truth
        variance=sum((v-mean)**2 for v in draws)/len(draws)
        mse=sum((v-truth)**2 for v in draws)/len(draws)
        assert isclose(mse,bias*bias+variance,abs_tol=1e-10)
        assert isclose(variance,sum(w*w for w in weights),abs_tol=1e-10)
        page.wait_for_function("v => document.querySelector('.moments').textContent.includes(v)",arg=f'{variance:.3f}σ²')
        if method=='knn':
            assert abs(bias)<=max(distances[i] for i in indices)+1e-10
    assert page.locator('.katex-error').count()==0
    print('PASS parameter limits, Gaussian normalization, conditional moments and radius bound',flush=True)
    # Move x0 from just beyond its visible cross, then use the keyboard.
    pos=page.evaluate("""() => {
      const v=__demo.v2, r=v.canvas.getBoundingClientRect(), a=v.toPx(__demo.state().x0);
      return [r.x+a[0],r.y+a[1]];
    }""")
    x,y=pos
    page.mouse.move(x-9,y)
    page.mouse.down()
    page.mouse.move(x-29,y-12,steps=3)
    assert page.evaluate('__demo.state().drag.kind')=='x0'
    page.mouse.up()
    query=page.evaluate('__demo.state().x0')
    page.locator('.view2d').press('ArrowRight')
    assert isclose(page.evaluate('__demo.state().x0[0]'),query[0]+0.1)
    page.locator('#more').click()
    page.locator('#reset').click()
    page.keyboard.press('Escape')
    assert page.evaluate('__demo.state().x0')==s['x0']
    print('PASS query hit area, keyboard movement and dataset reset',flush=True)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory() as tmp, sync_playwright() as pw:
        copytree(ROOT/'dist',Path(tmp)/'sample-project')
        server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=tmp))
        Thread(target=server.serve_forever,daemon=True).start()
        try:
            with pw.chromium.launch(args=['--use-angle=swiftshader','--enable-unsafe-swiftshader']) as browser:
                page=browser.new_page(reduced_motion='reduce')
                errors=[]
                page.on('pageerror',lambda e: errors.append(str(e)))
                page.on('console',lambda m: errors.append(m.text) if m.type=='error' else None)
                url=LIVE_URL or f'http://127.0.0.1:{server.server_port}/sample-project/'
                for size in [(1366,768),(390,844)]:
                    page.goto(url+'#count=16')
                    page.evaluate('document.fonts.ready')
                    inspect(page,*size)
                page.goto('about:blank')
                page.goto(url)
                check_moments(page)
                assert not errors,errors
        finally:
            server.shutdown()
            server.server_close()


if __name__=='__main__':
    run()
