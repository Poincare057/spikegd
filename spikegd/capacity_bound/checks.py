"""Simulator, gradient, and weight-bound checks; results go to output/RUN_RESULTS.md."""

from __future__ import annotations

import numpy as np
from generate_figs import PI, H, groups, jac, map_out_out, simulate, tsp
from results import capture_results

# %%
############################
### Simulator checks
############################


def check_simulator() -> None:
    rng = np.random.default_rng(0)
    # 1. complex-step out->out vs the closed form of eq. (c-Joo)
    for b in [12.0, 50.0, 400.0]:
        c = np.sqrt(b)
        J = jac(map_out_out, np.array([1.0, b]))
        Jc = np.array([[1, -PI / (2 * c**2 * (c - PI))], [0, (c - PI) / c]])
        print("out->out  b=%5.0f  max|diff| = %.2e" % (b, np.abs(J - Jc).max()))
    # 2. time-to-event identity on all branches
    for V0, b in [(0.3, 5.0), (2.0, -1.0), (1.5, 0.0 + 1e-9), (-0.5, 30.0)]:
        g = np.array(
            [np.imag(tsp(V0 + 1j * H, b)) / H, np.imag(tsp(V0, b + 1j * H)) / H]
        )
        f = np.array([V0**2 - V0 + b, -2 * b])
        print("grad t_sp . f at (%.1f,%.1g) = %.12f" % (V0, b, g @ f))
    # 3. RTRL trace vs finite differences of whole runs (Case 3, Poisson inputs)
    T = 4.0
    n = 6
    n_weights = 2
    t_in = np.sort(rng.uniform(0, T, n))
    syn = np.arange(n) % n_weights
    w = np.array([150.0, 90.0])
    ev = simulate(t_in, syn, w, V0=0.2, b0=0.0, T_end=T)
    outs = np.array(ev["outputs"])
    Mout = np.array([M[0] for M, k in zip(ev["M"], ev["kind"]) if k.endswith("out")])
    eps = 1e-6
    for p in range(1 + n_weights):

        def run(d):
            ww = w.copy()
            V = 0.2
            if p == 0:
                V += d
            else:
                ww[p - 1] += d
            return np.array(simulate(t_in, syn, ww, V0=V, b0=0.0, T_end=T)["outputs"])

        fd = (run(eps) - run(-eps)) / (2 * eps)
        print(
            "param %d: %d outputs, max rel err RTRL vs FD = %.2e"
            % (
                p,
                len(outs),
                np.max(np.abs(fd - Mout[:, p]) / np.maximum(1, np.abs(fd))),
            )
        )
    # 4. telescoping c2/c_{i+1} after the last input (Case 1b)
    ev = simulate([0.1], [0], [600.0], V0=0.0, b0=0.0, T_end=50)
    g = groups(ev, [0.1])[0]
    c2 = g["c2"]
    k = g["n_out"]
    Mt = np.array([M[0, 0] for M, kk in zip(ev["M"], ev["kind"]) if kk.endswith("out")])
    pred = c2 / (c2 - np.arange(k) * PI)
    print(
        "Case 1b: %d outputs, ceil(c2/pi)=%d, max |M_t(i)/M_t(1) - c2/c_{i+1}| = %.2e"
        % (k, int(np.ceil(c2 / PI)), np.abs(Mt / Mt[0] - pred).max())
    )
    print("last amplification c2/c_{k+1} = %.3f  (>= k = %d)" % (pred[-1], k))


# %%
############################
### Weight-bound checks
############################


def check_weight_bounds() -> None:
    """Check weight bounds for Cases 2 and 3, excluding spikes in the input window.

    For l synapses and f inputs per synapse in a window Delta followed by gap G:
    w >= (k*pi / (1-exp(-G)))**2 * exp(2*Delta) / (l*f) guarantees at least k outputs;
    w <= ((k*pi / (1-exp(-G)))**2 - L) / (l*f) guarantees at most k outputs,
    where L is current left from before the window (zero in these trials).
    """
    rng = np.random.default_rng(7)
    for n_weights, f, Delta, G in [
        (1, 1, 0.0, np.inf),
        (2, 3, 0.02, 2.0),
        (4, 2, 0.05, 1.0),
        (3, 4, 0.1, 0.5),
    ]:
        for k in [3, 8, 15]:
            X = (k * PI / (1 - np.exp(-G))) ** 2
            w_lo = X * np.exp(2 * Delta) / (n_weights * f)
            w_hi = X / (n_weights * f)
            res = []
            for _ in range(200):
                t = np.sort(rng.uniform(0, Delta, n_weights * f)) + 0.1
                syn = np.repeat(np.arange(n_weights), f)
                rng.shuffle(syn)
                T = t[-1] + (G if np.isfinite(G) else 60.0)
                for w, side in [(w_lo, "lo"), (w_hi, "hi")]:
                    ev = simulate(
                        t,
                        syn,
                        np.full(n_weights, w),
                        V0=rng.uniform(-1, 1),
                        b0=0.0,
                        T_end=T,
                    )
                    gs = groups(ev, t, T)
                    inside = (
                        len(gs) > 1
                    )  # an output fell inside the window: not this arrangement
                    n_last = gs[-1]["n_out"]
                    res.append((side, inside, n_last))
            lo = [(i, n) for s, i, n in res if s == "lo"]
            hi = [(i, n) for s, i, n in res if s == "hi"]
            ok_lo = all(n >= k for i, n in lo if not i)
            ok_hi = all(n <= k for i, n in hi if not i)
            print(
                (
                    "l=%d f=%d Delta=%.2f G=%s k=%2d | w_lo=%8.1f: >=k holds "
                    "%s (window spikes %3d%%) | w_hi=%8.1f: <=k holds %s "
                    "(window spikes %3d%%)"
                )
                % (
                    n_weights,
                    f,
                    Delta,
                    G,
                    k,
                    w_lo,
                    ok_lo,
                    100 * np.mean([i for i, n in lo]),
                    w_hi,
                    ok_hi,
                    100 * np.mean([i for i, n in hi]),
                )
            )


# %%
############################
### Command line
############################


def main() -> None:
    with capture_results("NUMERICS", "Simulator checks"):
        check_simulator()
    with capture_results("WEIGHT_BOUNDS", "Weight-bound checks"):
        check_weight_bounds()


if __name__ == "__main__":
    main()
