"""Projector and deuteranopia simulations of the palette (method of the first UI/UX audit).

Projector in a lit room: black level raised by 6 %, saturation reduced by 18 %.
Deuteranopia: Machado et al. (2009), severity 1.0, in linear RGB.
These are approximations; they do not replace a check on the real projector.

    python -I tools/palette_sim.py   ->  .artifacts/palette/
"""

import math
import pathlib
import re

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHOTS = ROOT / ".artifacts" / "browser"
OUT = ROOT / ".artifacts" / "palette"
OUT.mkdir(parents=True, exist_ok=True)

scale_source = (ROOT / "src/math/colormap.ts").read_text(encoding="utf8")
SCALE = re.findall(r"#[0-9A-F]{6}", next(line for line in scale_source.splitlines() if line.startswith("export const SCALE_STOPS_HEX")))
theme = dict(re.findall(r"(\w+): '(#[0-9A-F]{6})'", (ROOT / "src/ui/theme.ts").read_text(encoding="utf8")))
BG, INK, ACCENT, WALL = (theme[k] for k in ("bg", "ink", "accent", "wall"))
MACHADO_D = np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]])


def to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def projector(rgb):  # rgb in [0,1] sRGB
    grey = rgb @ np.array([0.2126, 0.7152, 0.0722])
    rgb = grey[..., None] + (rgb - grey[..., None]) * 0.82
    return 0.06 + 0.94 * rgb


def deutan(rgb):
    return to_srgb(to_lin(rgb) @ MACHADO_D.T)


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def lum(rgb):
    return float(to_lin(rgb) @ np.array([0.2126, 0.7152, 0.0722]))


def contrast(a, b):
    la, lb = sorted([lum(a), lum(b)])
    return (lb + 0.05) / (la + 0.05)


def oklab(rgb):
    r, g, b = to_lin(rgb)
    l, m, s = (np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566], [0.0883024619, 0.2817188376, 0.6299787005]]) @ np.array([r, g, b])) ** (1 / 3)
    return np.array([0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s])


lines = ["# Palette simulations (Crimson Unboxing Light)", ""]
for name, f in (("normal", lambda x: x), ("projector", projector), ("deuteranopia", deutan)):
    bg = f(hexrgb(BG))
    sc = [f(hexrgb(h)) for h in SCALE]
    acc = f(hexrgb(ACCENT))
    dE = [100 * math.dist(oklab(sc[i]), oklab(sc[i + 1])) for i in range(len(sc) - 1)]
    acc_min = min(100 * math.dist(oklab(acc), oklab(c)) for c in sc)
    lines.append(f"## {name}")
    lines.append(f"- lowest scale step vs background: {contrast(sc[0], bg):.2f}:1; wall vs background: {contrast(f(hexrgb(WALL)), bg):.2f}:1; ink vs background: {contrast(f(hexrgb(INK)), bg):.2f}:1")
    lines.append(f"- adjacent scale steps ΔE·100: {', '.join(f'{x:.1f}' for x in dE)} (min {min(dE):.1f})")
    lines.append(f"- accent (x0) vs nearest scale colour ΔE·100: {acc_min:.1f}")
    lines.append("")

for fname in ("8-0-knn-2d.png", "wall-1366.png", "finished-hsweep.png", "finished-segment.png"):
    src = SHOTS / fname
    if not src.exists():
        continue
    img = np.asarray(Image.open(src).convert("RGB")).astype(float) / 255
    for tag, f in (("projector", projector), ("deutan", deutan)):
        out = (np.clip(f(img), 0, 1) * 255).round().astype(np.uint8)
        Image.fromarray(out).save(OUT / f"sim-{tag}-{src.name}")
    lines.append(f"- simulated: {fname}")

(OUT / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
