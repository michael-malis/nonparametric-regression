export const lectureLayout = `
<header class="mast"><strong>Nonparametric regression</strong>
  <nav aria-label="Activity"><button data-mode="demo">Guided scenes</button><button data-mode="explore">Explore</button><button data-mode="voronoi">Voronoi</button></nav>
  <button id="data-toggle" popovertarget="data-panel">Dataset <small id="config"></small></button>
  <section id="data-panel" popover><fieldset><legend>Number of observations</legend>
  <label><input type="radio" name="count" value="8"> 8 points</label>
  <label><input type="radio" name="count" value="16"> 16 points</label></fieldset>
  <label id="outlier-option"><input id="outlier" type="checkbox"> Include response outlier</label></section>
</header>
<section class="scene-head"><div class="demo-only scene-caption" aria-live="polite"><strong id="scene-title">Guided lecture</strong><span id="scene-context"></span></div>
  <label class="demo-only scene-picker">Experiment <select id="experiment"><option value="-1" disabled>Choose a scene</option></select></label>
  <div class="explore-only methods" role="group" aria-label="Regression method">
  <button data-method="1nn">1NN</button><button data-method="knn">kNN</button><button data-method="nw">Nadaraya–Watson</button></div>
  <div class="explore-only views" role="group" aria-label="Representation"><button data-view="2d">2D</button><button data-view="3d">3D</button></div>
  <strong class="geometry-only">2D geometry</strong></section>
<main class="stage"><aside class="rail"></aside><div class="viewport" data-testid="viewport"></div>
<section class="geometry-only geometry"><h2>Voronoi regions</h2><p>Drag a site, or Tab to it and use the arrow keys. Along a shared edge, both sites are equally near.</p>
<svg viewBox="-3 -3 106 106" aria-label="Movable Voronoi sites"></svg></section></main>
<footer><div class="demo-only transport">
<button class="primary" id="play">Start</button><button id="pause" hidden>Pause</button>
<button id="previous" hidden>Back</button><output id="progress"></output></div>
<div class="explore-only explore-controls"><label id="tuning-label"><span id="tuning-name"></span> <input id="tuning" type="range"><output id="tuning-value"></output></label>
<label>Response <select id="site"></select></label><label>Y <input id="response" type="range" min="0" max="10" step="0.1"><output id="response-value"></output></label><button id="undo">Undo</button></div>
<div class="geometry-only"><button id="reset-sites">Reset sites</button></div>
<button id="more" popovertarget="options">View options</button>
<section id="options" popover><button class="demo-only" id="restart">Replay this scene</button><label><input id="regions" type="checkbox"> Neighbour regions (kNN map)</label>
<label><input id="profile" type="checkbox"> Show path profile</label><button id="camera">Reset camera</button>
<button id="reset">Restore selected dataset</button><button id="full">Full screen</button>
<small class="demo-only">← / → or Page Up / Down · previous / next scene<br>Space · start or pause / resume<br>Esc · pause or close</small><small class="explore-only">2 / 3 · change view<br>Arrow keys on the map · move x₀</small></section><small class="credit">Developed by Michael Malis</small></footer>
`;
