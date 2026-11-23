"""Independent NumPy re-implementation of 1NN / kNN / Gaussian NW (check V09).

Reads .artifacts/numerical/cases.json produced by the TypeScript engine and recomputes every
number from X, Y alone. Cell areas of the order-k diagrams are checked by
Monte Carlo (each cell's area must match the share of random points whose
k-nearest set is that cell's set).

    npm run verify:numpy
"""

import json
import pathlib
import sys

import numpy as np

ARTIFACTS = pathlib.Path(__file__).resolve().parents[2] / ".artifacts" / "numerical"
cases = json.loads((ARTIFACTS / "cases.json").read_text())
TOL = 1e-10
failures = []
report = []


def knn_sets(X, pts, k):
    d = ((pts[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    idx = np.arange(len(X))
    # ties broken by the smaller index: stable sort on distance
    order = np.argsort(d, axis=1, kind="stable")
    return order[:, :k], d


def knn_predict(X, Y, pts, k):
    sets, _ = knn_sets(X, pts, k)
    return Y[sets].mean(1), sets


def nw_weights(X, pts, h):
    d = ((pts[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    a = np.exp(-(d - d.min(1, keepdims=True)) / (2 * h * h))
    return a / a.sum(1, keepdims=True)


def expect(name, ok, detail=""):
    report.append(f"{'pass' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


rng = np.random.default_rng(0)
for c in cases:
    X = np.array(c["X"], float)
    Y = np.array(c["Y"], float)
    G = np.array(c["grid"], float)
    n = len(X)
    for block in c["knn"]:
        k = block["k"]
        ref, sets = knn_predict(X, Y, G, k)
        err = np.abs(ref - np.array(block["values"])).max()
        expect(f"{c['id']} kNN k={k}: values on 625 grid points", err < TOL, f"max |Δ| = {err:.2e}")
        W = np.zeros((40, n))
        W[np.arange(40)[:, None], sets[:40]] = 1 / k
        werr = np.abs(W - np.array(block["weights"])).max()
        expect(f"{c['id']} kNN k={k}: weights", werr < TOL, f"max |Δ| = {werr:.2e}")
        # Monte Carlo areas of the exact cells
        M = 200_000
        P = rng.uniform(0, 10, (M, 2))
        s_mc, _ = knn_sets(X, P, k)
        keys = np.sort(s_mc, axis=1)
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        mc = {tuple(row): cnt / M * 100 for row, cnt in zip(np.unique(keys, axis=0), counts)}
        worst = 0.0
        for cell in block["cellAreas"]:
            a_mc = mc.get(tuple(cell["set"]), 0.0)
            se = np.sqrt(max(cell["area"], 0.01) / 100 * (1 - cell["area"] / 100) / M) * 100
            worst = max(worst, abs(a_mc - cell["area"]) / max(se, 1e-9))
            v = Y[cell["set"]].mean()
            if abs(v - cell["value"]) > TOL:
                failures.append(f"cell value {c['id']} k={k}")
        total = sum(cell["area"] for cell in block["cellAreas"])
        expect(f"{c['id']} kNN k={k}: {len(block['cellAreas'])} exact cells vs Monte Carlo", worst < 5 and abs(total - 100) < 1e-8, f"worst z = {worst:.2f}, total area = {total:.10f}")
        # profile: piecewise constant values equal direct evaluation at segment midpoints
        A, B = np.array(c["pathA"]), np.array(c["pathB"])
        segs = block["profile"]
        mids = np.array([(s["s0"] + s["s1"]) / 2 for s in segs])
        pv, _ = knn_predict(X, Y, A + mids[:, None] * (B - A), k)
        perr = np.abs(pv - np.array([s["value"] for s in segs])).max()
        # and dense sampling finds no jump that the exact profile misses
        ss = np.linspace(0, 1, 20001)
        dv, _ = knn_predict(X, Y, A + ss[:, None] * (B - A), k)
        jumps = np.count_nonzero(np.abs(np.diff(dv)) > 1e-12)
        expect(f"{c['id']} kNN k={k}: path profile ({len(segs)} pieces)", perr < TOL and jumps <= len(segs) - 1, f"max |Δ| = {perr:.2e}, sampled jumps = {jumps}")
    for block in c["nw"]:
        h = block["h"]
        W = nw_weights(X, G, h)
        ref = W @ Y
        err = np.abs(ref - np.array(block["values"])).max()
        expect(f"{c['id']} NW h={h}: values on 625 grid points", err < TOL, f"max |Δ| = {err:.2e}")
        werr = np.abs(W[:40] - np.array(block["weights"])).max()
        # Mathematically every Gaussian weight is > 0. In float64, exp(t) underflows
        # to 0 for t < -745, so positivity is required wherever it is representable.
        d = ((G[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        expo = -(d - d.min(1, keepdims=True)) / (2 * h * h)
        representable = expo > -700
        pos_ok = (W[representable] > 0).all() and (W >= 0).all()
        n_under = int((~representable).sum())
        expect(f"{c['id']} NW h={h}: weights (> 0 where representable, sum 1)", werr < TOL and pos_ok and np.allclose(W.sum(1), 1),
               f"max |Δ| = {werr:.2e}, float64-underflowed weights: {n_under}")
        s = np.array(block["profile"]["s"])
        A, B = np.array(c["pathA"]), np.array(c["pathB"])
        pv = nw_weights(X, A + s[:, None] * (B - A), h) @ Y
        perr = np.abs(pv - np.array(block["profile"]["v"])).max()
        expect(f"{c['id']} NW h={h}: path profile samples", perr < TOL, f"max |Δ| = {perr:.2e}")

print("\n".join(report))
print(f"\n{len(report) - len(failures)}/{len(report)} checks passed")
(ARTIFACTS / "numpy_results.txt").write_text("\n".join(report) + f"\n\n{len(report) - len(failures)}/{len(report)} checks passed\n", encoding="utf-8")
sys.exit(1 if failures else 0)
