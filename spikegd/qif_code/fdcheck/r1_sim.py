"""Independent R1 simulator (values only) on the generate_figs closed forms, for finite differences."""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs'))
from spikegd.qif_code.figs.generate_figs import flow, flow_reset, tsp, tsp_reset, PI
def r1_outputs(w_ext, t_ext, wj, d, T, V0=0.0, b0=0.0):
    V, b = flow(V0, b0, t_ext), b0*np.exp(-2*t_ext)+w_ext; t = t_ext; outs = []
    while True:
        Ts = tsp(V, b)
        if not np.isfinite(Ts) or t+Ts >= T: return np.array(outs)
        t = t+Ts; b = b*np.exp(-2*Ts); outs.append(t)
        if wj == 0:                                   # no re-entry: the train continues from the reset
            while b > PI**2 and t + tsp_reset(b) < T:
                Ts = tsp_reset(b); t += Ts; b *= np.exp(-2*Ts); outs.append(t)
            return np.array(outs)
        if t+d >= T: return np.array(outs)
        if b > PI**2 and tsp_reset(b) < d: raise ValueError('output before its re-entry: outside (R3)')
        V, b = flow_reset(b, d), b*np.exp(-2*d)+wj; t = t+d
