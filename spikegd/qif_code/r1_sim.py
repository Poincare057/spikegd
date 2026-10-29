"""Independent R1 simulator (values only) on the generate_figs closed forms, for finite
differences.
"""

from __future__ import annotations

import numpy as np
from generate_figs import PI, flow, flow_reset, tsp, tsp_reset

# %%
############################
### Simulation
############################


def r1_outputs(
    w_ext: float,
    t_ext: float,
    wj: float,
    d: float,
    T: float,
    V0: float = 0.0,
    b0: float = 0.0,
) -> np.ndarray:
    """Return output times with delayed recurrence, for finite differences."""
    ### Initialize the neuron
    V, b = flow(V0, b0, t_ext), b0 * np.exp(-2 * t_ext) + w_ext
    t = t_ext
    outs = []

    ### Advance outputs and delayed inputs
    while True:
        Ts = tsp(V, b)
        if not np.isfinite(Ts) or t + Ts >= T:
            return np.array(outs)
        t = t + Ts
        b = b * np.exp(-2 * Ts)
        outs.append(t)
        if wj == 0:  # no re-entry: the train continues from the reset
            while b > PI**2 and t + tsp_reset(b) < T:
                Ts = tsp_reset(b)
                t += Ts
                b *= np.exp(-2 * Ts)
                outs.append(t)
            return np.array(outs)
        if t + d >= T:
            return np.array(outs)
        if b > PI**2 and tsp_reset(b) < d:
            raise ValueError("output before its re-entry: outside (R3)")
        V, b = flow_reset(b, d), b * np.exp(-2 * d) + wj
        t = t + d
