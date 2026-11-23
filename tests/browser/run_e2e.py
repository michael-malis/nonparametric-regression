"""Exercise the offline release, or the deployment specified by DEMO_TEST_URL."""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
LIVE_URL = os.environ.get('DEMO_TEST_URL')
OUT = ROOT/'.artifacts'/('browser-public' if LIVE_URL else 'browser')
OUT.mkdir(parents=True, exist_ok=True)
URL = LIVE_URL or (ROOT/'dist/index.html').as_uri()
RESULTS = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    RESULTS.append(name)
    print('PASS',name,flush=True)

def state(p):
    return p.evaluate('__demo.state()')
def click(p,sel):
    p.locator(sel).click()
    p.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
def mode(p,m):
    click(p,f'button[data-mode="{m}"]')
def method(p,m):
    if p.evaluate("__demo.mode()")!="explore": mode(p,"explore")
    click(p,f'[data-method="{m}"]')
def view(p,v):
    click(p,f'[data-view="{v}"]')
def choose(p,id):
    s=state(p)
    stage=1+2*['1nn','knn','nw'].index(s['method'])+int(s['view']=='3d')
    url=f"{URL}#stage={stage}&exp={id}&count={len(s['X'])}&outlier={int(s['includeOutlier'])}"
    p.goto('about:blank')
    p.goto(url)
    p.wait_for_function('id => window.__demo?.state().experiment?.id === id',arg=id)
def restart(p):
    click(p,'#more');click(p,'#restart')
def config(p,n,flag):
    click(p,'#data-toggle')
    p.locator(f'[name=count][value="{n}"]').check()
    p.locator('#outlier').set_checked(flag)
    p.keyboard.press('Escape')
    p.wait_for_function('a=>__demo.state().X.length===a[0] && __demo.state().includeOutlier===a[1]',arg=[n,flag])

def layout(p):
    return p.evaluate("""() => {
      const stage=document.querySelector('.stage').getBoundingClientRect(),foot=document.querySelector('footer').getBoundingClientRect();
      const visible=[...document.querySelectorAll('header button,.scene-head button,footer button,footer select')].filter(e=>e.checkVisibility()&&e.offsetWidth);
      const boxes=visible.map(e=>e.getBoundingClientRect());
      return foot.top>=stage.bottom-1 && foot.bottom<=innerHeight+1 &&
        boxes.every(r=>r.left>=-1&&r.right<=innerWidth+1&&r.top>=0&&r.bottom<=innerHeight+1) &&
        boxes.every((r,i)=>boxes.slice(i+1).every(b=>r.right<=b.left+1||r.left>=b.right-1||r.bottom<=b.top+1||r.top>=b.bottom-1));
    }""")

def luminance(hexcolor):
    v=[int(hexcolor[i:i+2],16)/255 for i in (0,2,4)]
    vals=[x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in v]
    return sum(a*b for a,b in zip(vals,[.2126,.7152,.0722]))
def contrast(a,b):
    x,y=sorted([luminance(a),luminance(b)])
    return (y+.05)/(x+.05)

def run():
    with sync_playwright() as pw:
        browser=pw.chromium.launch(args=['--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        errors=[];external=[]
        def open_page(ctx):
            def guard(route):
                if route.request.url.startswith(('blob:','data:')) or route.request.url.partition('#')[0] == URL:route.continue_()
                else:external.append(route.request.url);route.abort()
            ctx.route('**/*',guard)
            ctx.set_offline(not LIVE_URL)
            p=ctx.new_page()
            p.on('pageerror',lambda e: errors.append(str(e)))
            p.on('response',lambda r: errors.append(f'{r.status} {r.url}') if r.status>=400 else None)
            p.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
            p.goto(URL);p.wait_for_function('!!window.__demo')
            return p
        ctx=browser.new_context(viewport=dict(width=1366,height=768),reduced_motion='reduce')
        p=open_page(ctx)
        check('Explore opens with visible method controls',p.evaluate('__demo.mode()')=='explore' and p.locator('.methods').is_visible())
        mode(p,'demo')
        check('Lecture opens ready for Start',len(state(p)['X'])==8 and not state(p)['includeOutlier'] and not state(p)['experiment'] and p.get_by_role('button',name='Start',exact=True).is_enabled())
        check('Lecture needs no method, view or example selection',p.locator('.scene-head button:visible,footer select:visible').count()==0)
        p.keyboard.press('PageDown')
        check('Presenter remote starts the first scene',state(p)['experiment']['id']=='segment' and p.locator('#play').inner_text()=='Next')
        check('Real WebGL2 context',p.evaluate('__demo.v3.renderer.getContext() instanceof WebGL2RenderingContext'))
        route=[('1nn','2d','segment'),('1nn','3d','response'),('knn','2d','segment'),('knn','3d','response'),
               ('knn','3d','ksteps'),('nw','2d','segment'),('nw','3d','hsweep'),('nw','3d','response')]
        mode(p,'explore')
        for n in [8,16]:
            for flag in [False,True]:
                config(p,n,flag)
                snapshot=state(p)
                if flag:
                    check(f'{n} outlier changes exactly one Y',X==snapshot['X'] and sum(a!=b for a,b in zip(Y,snapshot['Y']))==1)
                X,Y=snapshot['X'],snapshot['Y']
                for m in ['1nn','knn','nw']:
                    method(p,m)
                    for v in ['2d','3d']:
                        view(p,v)
                        s=state(p)
                        prediction=p.evaluate('__demo.prediction()')
                        w=prediction['weights']
                        check(f'{n}/{int(flag)} {m} {v}',s['X']==X and s['Y']==Y and s['method']==m and s['view']==v and s['kKnn']<=n and abs(sum(w)-1)<1e-10 and abs(sum(a*b for a,b in zip(w,Y))-prediction['value'])<1e-10)
                        p.screenshot(path=OUT/f'{n}-{int(flag)}-{m}-{v}.png')
                mode(p,'demo')
                for index,(m,v,ex) in enumerate(route,1):
                    click(p,'#play')
                    s=state(p)
                    check(f'{n}/{int(flag)} scene {index} starts {m} {v} {ex}',
                          (s['method'],s['view'],s['experiment']['id'])==(m,v,ex)
                          and s['X']==X and s['includeOutlier']==flag
                          and p.locator('#scene-context').inner_text().startswith(f'Scene {index} of 8'))
                    if ex=='response':
                        check(f'{n}/{int(flag)} {m} response uses same starting data',p.evaluate('__demo.plan().start.Y')==Y)
                p.wait_for_function("__demo.state().experiment.phase==='done'",timeout=12000)
                p.wait_for_timeout(250)
                check(f'{n}/{int(flag)} final result waits for presenter',p.locator('#progress').inner_text()=='Lecture complete')
                click(p,'#play')
                check(f'{n}/{int(flag)} restart restores the lecture baseline',state(p)['Y']==Y and state(p)['method']=='1nn')
                mode(p,'explore')
        for width,height in [(1024,768),(1366,768),(1920,1080)]:
            p.set_viewport_size(dict(width=width,height=height))
            for tab in ['demo','explore','voronoi']:
                mode(p,tab);check(f'{width} {tab} controls fit',layout(p))
            mode(p,'demo');method(p,'knn');view(p,'3d');choose(p,'response')
            click(p,'#play');p.wait_for_function("__demo.state().experiment.phase==='done'")
            check(f'{width} WebGL labels stay apart',p.evaluate("""() => {
              const nodes=[...document.querySelectorAll('.obslabel .lbl,.x0label')].filter(e=>e.textContent && e.checkVisibility() && e.getBoundingClientRect().width);
              const r=nodes.map(e=>e.getBoundingClientRect());
              return r.every((a,i)=>r.slice(i+1).every(b=>a.right<=b.left||a.left>=b.right||a.bottom<=b.top||a.top>=b.bottom));
            }"""))
            check(f'{width} scene receives at least 70% of width',p.locator('.viewport').bounding_box()['width']>=width*.70)
            p.screenshot(path=OUT/f'wall-{width}.png')
            mode(p,'explore')
        p.set_viewport_size(dict(width=1366,height=768))
        config(p,8,False);method(p,'knn');view(p,'2d');choose(p,'segment');click(p,'#play')
        p.keyboard.press('Escape');saved=state(p); mode(p,'voronoi')
        site=p.locator('[data-site="0"]');a=site.bounding_box()
        before=p.locator('.polygons').inner_html()
        cx=a['x']+a['width']/2;cy=a['y']+a['height']/2
        p.mouse.move(cx,cy);p.mouse.down();p.mouse.move(cx+48,cy-30,steps=6);p.mouse.up()
        check('Drag recomputes Voronoi geometry',p.locator('.polygons').inner_html()!=before)
        p.screenshot(path=OUT/'voronoi.png');click(p,'#reset-sites')
        check('Independent site reset',p.locator('.polygons').inner_html()==before)
        mode(p,'demo');check('Voronoi detour preserves whole regression state',state(p)==saved)
        check('Long route keeps endpoints on screen at equal feature scales',p.evaluate("""()=>{
          const s=__demo.state(), v=__demo.v2, a=v.toPx(s.pathA),b=v.toPx(s.pathB),q=v.sq;
          return (b[0]-a[0])/q.side>=0.6 && [a,b].every(([x,y])=>x>=q.x&&x<=q.x+q.side&&y>=q.y&&y<=q.y+q.side)
            && Math.abs(v.toPx([1,0])[0]-v.toPx([0,0])[0]-(v.toPx([0,0])[1]-v.toPx([0,1])[1]))<1e-7;
        }"""))
        check('Map labels do not intersect',p.evaluate("""()=>{
          const r=__demo.v2.debugLabels();
          return r.every((a,i)=>r.slice(i+1).every(b=>a.x+a.w<=b.x||b.x+b.w<=a.x||a.y+a.h<=b.y||b.y+b.h<=a.y));
        }"""))
        val=p.evaluate('__demo.prediction().value')
        click(p,'#more');p.locator('#regions').check();p.keyboard.press('Escape')
        check('Exact regions toggle preserves prediction',state(p)['showRegions'] and p.evaluate('__demo.prediction().value')==val)
        click(p,'#previous');check('Back reopens the preceding scene',state(p)['method']=='1nn' and state(p)['experiment']['id']=='response')
        click(p,'#play')
        mode(p,'explore')
        y=state(p)['Y'];x=state(p)['X']
        p.locator('#response').focus();p.keyboard.press('ArrowUp');p.wait_for_timeout(100)
        check('Explore changes one response only',state(p)['X']==x and sum(a!=b for a,b in zip(y,state(p)['Y']))==1)
        click(p,'#undo');check('Undo response edit',state(p)['Y']==y)
        view(p,'3d')
        anchor=p.evaluate("""()=>{let v=__demo.v3,r=v.renderer.domElement.getBoundingClientRect(),a=v.debugAnchors().obs2;return [r.x+a[0],r.y+a[1]]}""")
        camera=p.evaluate('__demo.v3.camera.position.toArray()')
        p.mouse.move(*anchor);p.mouse.down();p.mouse.move(anchor[0],anchor[1]-38,steps=8);p.mouse.up()
        check('Vertical WebGL drag moves only Y with fixed camera',state(p)['X']==x and state(p)['Y'][2]>y[2] and sum(a!=b for a,b in zip(y,state(p)['Y']))==1 and p.evaluate('__demo.v3.camera.position.toArray()')==camera)
        click(p,'#undo')
        view(p,'2d');p.locator('#tuning').fill('8')
        check('Explore k=n gives the sample mean',abs(p.evaluate('__demo.prediction().value')-sum(y)/len(y))<1e-10)
        choose(p,'segment')
        check('Bookmarked segment waits at A',state(p)['s']==0 and p.locator('#play').inner_text()=='Start scene')
        click(p,'#play');p.keyboard.press('PageDown')
        check('Next starts the following scene and selects its view',state(p)['method']=='knn' and state(p)['view']=='3d' and state(p)['experiment']['id']=='response')
        restart(p)
        check('Replay uses the same starting responses',p.evaluate('__demo.plan().start.Y')==state(p)['Y'])
        check('Replay closes options and returns focus to Next',p.locator('#play').evaluate('el=>el===document.activeElement') and not p.locator('#options').is_visible())
        p.keyboard.press('Tab')
        check('Keyboard focus visible',p.evaluate("getComputedStyle(document.querySelector(':focus-visible')).outlineStyle==='solid'"))
        p.locator('.calculation summary').click()
        p.wait_for_selector('.calculation[open] .weights .wrow')
        check('Expanded details show current weights',p.locator('.weights .wrow').count()>0 and p.locator('.calc-line').inner_text().strip())
        p.locator('.calculation summary').click()
        for a,b,minratio in [('4a0c1c','fdeef0',4.5),('78142b','fffafb',4.5),('ffffff','78142b',4.5),('ffffff','a81b3c',4.5),('a87782','fdeef0',3),('d92d4f','fffafb',3)]:
            ratio=contrast(a,b);check(f'Contrast #{a}/#{b} = {ratio:.2f}',ratio>=minratio)
        p.screenshot(path=OUT/'lecture-controls.png')
        ctx.close()
        fallback=browser.new_context(reduced_motion='reduce')
        fallback.add_init_script('''const original=HTMLCanvasElement.prototype.getContext;
          HTMLCanvasElement.prototype.getContext=function(type,...args){
            return type==='webgl2'?null:original.call(this,type,...args);
          }''')
        f=fallback.new_page();f.goto(URL);f.wait_for_function('!!window.__demo')
        mode(f,'demo')
        for i in range(8):
            click(f,'#play')
            assert state(f)['view']=='2d' and f.locator('#scene-context').inner_text().startswith(f'Scene {i+1} of 8')
        check('All eight scenes work with the 2D fallback',state(f)['method']=='nw')
        fallback.close()
        video=browser.new_context(viewport=dict(width=1366,height=768),record_video_dir=OUT,record_video_size=dict(width=1366,height=768))
        p=open_page(video)
        mode(p,'demo')
        for index,(m,v,ex) in enumerate(route,1):
            click(p,'#play')
            if m=='knn' and ex=='response':
                surface_material=p.evaluate_handle('__demo.v3.surface.children[0].material')
                link_buffer=p.evaluate_handle("__demo.v3.x0Guide.geometry.getAttribute('instanceStart').data")
            if index==1:
                p.wait_for_function("__demo.state().s>0 && __demo.state().experiment.phase==='running'")
                click(p,'#pause')
                rest=state(p)['s'];p.wait_for_timeout(350)
                check('Pause holds the query',state(p)['s']==rest and p.locator('#pause').inner_text()=='Resume')
                click(p,'#pause')
            p.wait_for_function("__demo.state().experiment.phase==='done'",timeout=60000)
            if m=='knn' and ex=='response':
                check('Response animation reuses its WebGL material and guide buffer',
                      p.evaluate('m=>m===__demo.v3.surface.children[0].material',surface_material)
                      and p.evaluate("b=>b===__demo.v3.x0Guide.geometry.getAttribute('instanceStart').data",link_buffer))
            if (m,ex) in [('1nn','segment'),('knn','response'),('knn','ksteps'),('nw','hsweep')]:
                p.screenshot(path=OUT/f'finished-{ex}.png')
            s=state(p)
            check(f'Recorded full scene {index}: {m}, {v}, {ex}',(s['method'],s['view'],s['experiment']['id'])==(m,v,ex))
        check('Filmed lecture ends on its final result',p.locator('#progress').inner_text()=='Lecture complete')
        recording=p.video;video.close();recording.save_as(OUT/'walkthrough.webm');recording.delete()
        check('No external resources or browser errors',not errors and not external)
        browser.close()
    target = LIVE_URL or 'Local file, blocked network'
    report=f'# Browser checks\n\n{len(RESULTS)} passed. {target}, Chromium WebGL2.\n\n'+'\n'.join('- '+v for v in RESULTS)+'\n'
    (OUT/'results.md').write_text(report,encoding='utf8')
    (OUT/'results.json').write_text(json.dumps(RESULTS,indent=2),encoding='utf8')
if __name__=='__main__':
    run()
