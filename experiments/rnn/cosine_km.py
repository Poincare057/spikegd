"""
Cosine similarity between each rule's update and the exact gradient, along the
rule's own training trajectories from sensitivity_km.py.

At the saved parameters (every --every trials, every noise level and seed) of
the `local` and `matkc` runs, the exact gradient (autodiff of the full loss) and
the rule's gradient are evaluated on the clean inputs, and

    cos = <g_rule, g_exact> / (|g_rule| |g_exact|)

is computed over the recurrent weights, (a) on the target neurons' rows (the
only rows the local rules update) and (b) on all rows. Initial-state
gradients are not included. Writes cosine_km.npz and a figure.

Usage (from experiments/rnn/, after sensitivity_km.py run):

    python cosine_km.py             # -> cosine_km.npz, sensitivity_km_Akc_cosine.png
    python cosine_km.py --plot-only
"""

import argparse
import json
import time

import experiment as ex  # sets JAX_PLATFORMS / JAX_ENABLE_X64 on import
import jax
import jax.numpy as jnp
import numpy as np
from spikegd.qif import QIFNeuron

from fastgrad import make_fast_grad

OUT = "cosine_km.npz"


def compute(args):
    d = np.load(args.data)
    sig = d["sigmas"]
    meta = json.loads(str(d["args"]))
    rec = meta["record"]
    cfg = dict(ex.BASE_CONFIG)
    N, ntar = cfg["N"], cfg["Ntar"]
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = ex.make_trial(cfg, ntar, 0.0)
    ones = jnp.ones_like(trial["w0"]); one = jnp.array(1.0)
    full = ex.make_full_loss(neuron, trial, ones)
    fit = json.load(open(args.fit))
    rules = {
        "local": make_fast_grad(neuron, trial, ntar, "trace"),
        "matkc": make_fast_grad(neuron, trial, ntar, "trace",
                                spike_mix=np.array(fit["spike_mix"]),
                                cross=np.array(fit["cross"])),
    }
    nw = N * (N - 1)
    step = max(args.every // rec, 1)

    def cosines(theta, rule_fn):
        w = theta[:nw].reshape(N, N - 1); x0 = theta[nw:].reshape(2, N)
        gE = jax.grad(lambda ww: full(ww, x0, one))(w)
        _, (gR, _) = rule_fn(w, x0, trial["spikes_in"])
        def cos(a, b):
            return jnp.vdot(a, b) / jnp.maximum(jnp.linalg.norm(a) * jnp.linalg.norm(b), 1e-300)
        return jnp.stack([cos(gR[:ntar], gE[:ntar]), cos(gR, gE)])

    out = {"sigmas": sig, "trials": np.arange(0, d["exact_clean"].shape[-1], step) * rec}
    t0 = time.time()
    for r, fn in rules.items():
        f = jax.jit(jax.vmap(lambda th, fn=fn: cosines(th, fn)))
        th = d[f"{r}_theta"][:, :, ::step]                     # (S, R, B', P)
        S, R, B, P = th.shape
        res = np.zeros((S, R, B, 2))
        for s in range(S):
            seeds = range(1) if sig[s] == 0 else range(R)       # sigma = 0: one run
            for q in seeds:
                res[s, q] = np.asarray(f(jnp.asarray(th[s, q])))
            if sig[s] == 0:
                res[s, 1:] = res[s, 0]
        out[r] = res
        print(f"  {r}: done ({time.time() - t0:.0f}s)", flush=True)
    np.savez(OUT, **out)
    print(f"  -> {OUT}")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(OUT)
    sig, trials = d["sigmas"], d["trials"]
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#b5b4ad"
    ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    pos = [s for s in sig if s > 0]
    col = {float(s): (INK if s == 0 else ramp[pos.index(s)]) for s in sig}
    names = {"local": "local gradient", "matkc": "local + matrix A + cross-target"}
    style = {"local": dict(color="#2a78d6", marker="s"), "matkc": dict(color="#1baf7a", marker="^")}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": GRID,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6),
                             gridspec_kw={"width_ratios": [1, 1, 0.95], "wspace": 0.3})
    for c, r in enumerate(["local", "matkc"]):
        ax = axes[c]
        v = d[r][..., 0]                                        # target rows
        for si, s in enumerate(sig):
            q1, q3 = np.percentile(v[si], [25, 75], axis=0)
            if s > 0:
                ax.fill_between(trials, q1, q3, color=col[s], alpha=0.2, lw=0)
            ax.plot(trials, np.median(v[si], 0), color=col[s], lw=1.6 if s == 0 else 1.3,
                    label=f"σ = {s:g}")
        ax.set_xscale("symlog", linthresh=trials[1]); ax.set_ylim(-0.2, 1.02)
        ax.axhline(1, color=GRID, lw=0.8)
        ax.set_xlabel("trial"); ax.grid(alpha=0.3, lw=0.5, axis="y")
        ax.set_title(f"{'E' if c == 0 else ' '}   {names[r]}", loc="left", fontsize=9.5)
        if c == 0:
            ax.set_ylabel("cosine with exact gradient\n(target neurons' weights)")
            ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax = axes[2]
    for k, r in enumerate(["local", "matkc"]):
        for j, (part, ls) in enumerate([(0, "-"), (1, ":")]):
            v = d[r][..., part].reshape(len(sig), -1)            # all saved points
            med = np.median(v, 1); q1, q3 = np.percentile(v, [25, 75], axis=1)
            xi = np.arange(len(sig)) + (k - 0.5) * 0.1 + (j - 0.5) * 0.04
            ax.errorbar(xi, med, yerr=[med - q1, q3 - med], ls=ls, capsize=2.5, lw=1.2,
                        ms=5, elinewidth=1, label=f"{names[r]}, "
                        f"{'target rows' if part == 0 else 'all rows'}", **style[r])
    ax.set_xticks(np.arange(len(sig)), [f"{s:g}" for s in sig])
    ax.set_xlabel("input jitter σ"); ax.set_ylabel("cosine with exact gradient")
    ax.set_title("F   over all of training", loc="left", fontsize=9.5)
    ax.set_ylim(-0.2, 1.02); ax.grid(alpha=0.3, lw=0.5, axis="y")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Alignment of the update with the exact gradient along each rule's own training "
                 "(clean inputs; median ± interquartile range over 20 noise seeds)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"  figure -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="sensitivity_km.npz")
    ap.add_argument("--fit", default="spike_AC_fit_km.json")
    ap.add_argument("--every", type=int, default=250, help="trials between evaluations")
    ap.add_argument("--out", default="sensitivity_km_Akc_cosine.png")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        compute(args)
    plot(args)


if __name__ == "__main__":
    main()
