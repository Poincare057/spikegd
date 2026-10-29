import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'figs'))
from spikegd.qif_code.figs.generate_figs import H, PI, groups, jac, map_out_out, simulate, tsp
rng = np.random.default_rng(0)
# 1. complex-step out->out vs the closed form of eq. (c-Joo)
for b in [12.0, 50.0, 400.0]:
    c = np.sqrt(b); J = jac(map_out_out, np.array([1.0, b]))
    Jc = np.array([[1, -PI/(2*c**2*(c-PI))], [0, (c-PI)/c]])
    print('out->out  b=%5.0f  max|diff| = %.2e' % (b, np.abs(J-Jc).max()))
# 2. time-to-event identity on all branches
for V0, b in [(0.3, 5.0), (2.0, -1.0), (1.5, 0.0+1e-9), (-0.5, 30.0)]:
    g = np.array([np.imag(tsp(V0+1j*H, b))/H, np.imag(tsp(V0, b+1j*H))/H])
    f = np.array([V0**2-V0+b, -2*b])
    print('grad t_sp . f at (%.1f,%.1g) = %.12f' % (V0, b, g@f))
# 3. RTRL trace vs finite differences of whole runs (Case 3, Poisson inputs)
T = 4.0; n = 6; l = 2
t_in = np.sort(rng.uniform(0, T, n)); syn = np.arange(n) % l; w = np.array([150.0, 90.0])
ev = simulate(t_in, syn, w, V0=0.2, b0=0.0, T_end=T)
outs = np.array(ev['outputs']); Mout = np.array([M[0] for M, k in zip(ev['M'], ev['kind']) if k.endswith('out')])
eps = 1e-6
for p in range(1+l):
    def run(d):
        ww = w.copy(); V = 0.2
        if p == 0: V += d
        else: ww[p-1] += d
        return np.array(simulate(t_in, syn, ww, V0=V, b0=0.0, T_end=T)['outputs'])
    fd = (run(eps)-run(-eps))/(2*eps)
    print('param %d: %d outputs, max rel err RTRL vs FD = %.2e' % (p, len(outs), np.max(np.abs(fd-Mout[:, p])/np.maximum(1, np.abs(fd)))))
# 4. telescoping c2/c_{i+1} after the last input (Case 1b)
ev = simulate([0.1], [0], [600.0], V0=0.0, b0=0.0, T_end=50)
g = groups(ev, [0.1])[0]; c2 = g['c2']; k = g['n_out']
Mt = np.array([M[0, 0] for M, kk in zip(ev['M'], ev['kind']) if kk.endswith('out')])
pred = c2/(c2-np.arange(k)*PI)
print('Case 1b: %d outputs, ceil(c2/pi)=%d, max |M_t(i)/M_t(1) - c2/c_{i+1}| = %.2e' % (k, int(np.ceil(c2/PI)), np.abs(Mt/Mt[0]-pred).max()))
print('last amplification c2/c_{k+1} = %.3f  (>= k = %d)' % (pred[-1], k))
