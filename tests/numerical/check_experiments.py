"""Recompute the stop predictions, weights and crossing locations from raw points."""
import json
from functools import cmp_to_key
from pathlib import Path
import numpy as np

artifacts = Path(__file__).resolve().parents[2] / '.artifacts' / 'numerical'
cases = json.loads((artifacts / 'experiments.json').read_text())
results = []
for c in cases:
    X, base = np.array(c['X']), np.array(c['Y'])
    for f in c['frames']:
        d = ((X - f['x'])**2).sum(axis=1)
        Y = np.array(f['Y'])
        if c['method'] == 'nw':
            w = np.exp(-(d - d.min()) / (2*f['h']**2))
            w /= w.sum()
        else:
            def compare(i, j):
                if abs(d[i]-d[j]) <= 1e-12*max(1, d[i], d[j]):
                    return i-j
                return 1 if d[i] > d[j] else -1
            nearest = sorted(range(len(X)), key=cmp_to_key(compare))[:f['k']]
            w = np.zeros(len(X))
            w[nearest] = 1/f['k']
        assert np.allclose(w, f['weights'], atol=1e-12, rtol=0)
        assert abs(w@Y-f['value']) < 1e-11
        if c['id'] == 'response':
            assert np.count_nonzero(Y-base) <= 1
            assert abs(w@(Y-base) - (f['value']-w@base)) < 1e-11
    if c['id']=='segment' and c['method']!='nw':
        A,B = np.array(c['A']), np.array(c['B'])
        t = np.linspace(0,1,20001)
        points = A+t[:,None]*(B-A)
        nearest = np.sort(np.argsort(((points[:,None]-X)**2).sum(2),axis=1)[:,:f['k']],axis=1)
        idx = np.flatnonzero(np.any(nearest[:-1]!=nearest[1:],axis=1))
        events = np.array([q['position'] for q in c['frames'][1:-1]])
        assert len(events)==len(idx) and np.all(abs(events-t[idx]) <= 1/20000+1e-12)
    results.append(f"PASS {c['dataset']} {c['method']} {c['id']} ({len(c['frames'])} stops)")
results.append(f'{len(cases)}/{len(cases)} scenario comparisons passed')
(artifacts / 'experiments_results.txt').write_text('\n'.join(results)+'\n',encoding='utf8')
print(results[-1])
