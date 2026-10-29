"""Check of the weight bound for Cases 2 and 3: l synapses, f inputs each inside a window of length Delta, then a gap G.
w >= (k pi / (1 - e^{-G}))^2 e^{2 Delta} / (l f)  =>  at least k outputs in the gap (when no output falls inside the window).
w <= ((k pi / (1 - e^{-G}))^2 - L) / (l f), L = current left from before the window  =>  at most k outputs in the gap."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'figs'))
from spikegd.qif_code.figs.generate_figs import simulate, groups, PI
rng = np.random.default_rng(7)
for (l, f, Delta, G) in [(1, 1, 0.0, np.inf), (2, 3, 0.02, 2.0), (4, 2, 0.05, 1.0), (3, 4, 0.1, 0.5)]:
    for k in [3, 8, 15]:
        X = (k*PI/(1-np.exp(-G)))**2
        w_lo = X*np.exp(2*Delta)/(l*f); w_hi = X/(l*f)
        res = []
        for _ in range(200):
            t = np.sort(rng.uniform(0, Delta, l*f)) + 0.1; syn = np.repeat(np.arange(l), f); rng.shuffle(syn)
            T = t[-1] + (G if np.isfinite(G) else 60.0)
            for w, side in [(w_lo, 'lo'), (w_hi, 'hi')]:
                ev = simulate(t, syn, np.full(l, w), V0=rng.uniform(-1, 1), b0=0.0, T_end=T)
                gs = groups(ev, t, T)
                inside = len(gs) > 1                       # an output fell inside the window: not this arrangement
                n_last = gs[-1]['n_out']
                res.append((side, inside, n_last))
        lo = [(i, n) for s, i, n in res if s == 'lo']; hi = [(i, n) for s, i, n in res if s == 'hi']
        ok_lo = all(n >= k for i, n in lo if not i); ok_hi = all(n <= k for i, n in hi if not i)
        print('l=%d f=%d Delta=%.2f G=%s k=%2d | w_lo=%8.1f: >=k holds %s (window spikes %3d%%) | w_hi=%8.1f: <=k holds %s (window spikes %3d%%)'
              % (l, f, Delta, G, k, w_lo, ok_lo, 100*np.mean([i for i, n in lo]), w_hi, ok_hi, 100*np.mean([i for i, n in hi])))