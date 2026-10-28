"""
Fit the low-rank non-locality correction to the local QIF gradient.

From the working notes, the exact weight gradient is approximated by

    dL/dw_ij  ~=  (1 + alpha_ij) * dL/dw_ij|local,     alpha_ij = a_i b_j

where the local part holds every presynaptic spike time fixed (the frozen
surround of experiment.make_frozen_loss). (1 + alpha_ij) does not depend on the
target spike index, so it factors out of the sum over target spikes and the
fit is element-wise on the gradient matrices.

Only target neurons have a non-zero local row (dL/dt_k = 0 elsewhere), so with
Ntar targets only a_0..a_{Ntar-1} are identifiable and the rule cannot
represent the exact gradient on non-target rows at all. Both are reported.

Usage (from experiments/rnn/):

    python lowrank.py collect --trials 400
    python lowrank.py fit
    python lowrank.py plot
"""

import argparse
from experiments.rnn.spikegd_exp.experiment import data_path, figure_path
import json
import time

import experiment as ex  # sets JAX_PLATFORMS / JAX_ENABLE_X64 on import
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import grad
from spikegd.qif import QIFNeuron

DATA = "lowrank_data.npz"
FIT = "lowrank_fit.json"


def stage_collect(args):
    """Train the intact network, recording exact and local gradients each step."""
    cfg = dict(ex.BASE_CONFIG)
    cfg["seed"] = args.seed
    N, ntar = cfg["N"], args.ntar
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = ex.make_trial(cfg, ntar, args.w_scale)
    ones = jnp.ones_like(trial["w0"])
    full = ex.make_full_loss(neuron, trial, ones)
    one = jnp.array(1.0)

    def local_row(w, x0, k):
        queue, rates, _ = ex.record_surround(neuron, trial, w, x0, ones, one, k)
        frozen = ex.make_frozen_loss(neuron, trial, queue, rates, k, ones, one)
        return grad(frozen, argnums=0)(w, x0)[k]

    optim = optax.adabelief(
        optax.exponential_decay(args.lr, max(args.trials // 5, 1), 1 / jnp.e)
    )

    @jax.jit
    def step(p, s):
        loss, g = jax.value_and_grad(lambda q: full(q[0], q[1], one))(p)
        g_loc = jnp.stack([local_row(p[0], p[1], k) for k in range(ntar)])
        u, s = optim.update(g, s)
        return optax.apply_updates(p, u), s, loss, g[0], g_loc

    p = (trial["w0"], trial["x0"])
    s = optim.init(p)
    E, G, L = [], [], []
    t0 = time.time()
    for t in range(args.trials):
        # gradients are taken at the parameters *before* update t
        p, s, loss, g_w, g_loc = step(p, s)
        Gt = np.zeros((N, N - 1))
        Gt[:ntar] = np.asarray(g_loc)
        E.append(np.asarray(ex.w_to_weights_net(g_w)))
        G.append(np.asarray(ex.w_to_weights_net(jnp.asarray(Gt))))
        L.append(float(loss))
        if t % 50 == 0:
            print(f"  step {t:4d}  loss {float(loss):.4e}  ({time.time() - t0:.0f}s)",
                  flush=True)
    np.savez(data_path(DATA), exact=np.array(E), local=np.array(G), loss=np.array(L),
             ntar=ntar, args=json.dumps(vars(args)))
    print(f"collected {args.trials} steps in {time.time() - t0:.0f}s -> {DATA}")


# ===========================================================================
# Fit
# ===========================================================================


def rel_err(E, Ghat, rows):
    """Mean over steps of ||E - Ghat|| / ||E|| restricted to the given rows."""
    num = np.linalg.norm((E - Ghat)[:, rows], axis=(1, 2))
    den = np.maximum(np.linalg.norm(E[:, rows], axis=(1, 2)), 1e-30)
    return float(np.mean(num / den))


def cosine(E, Ghat, rows):
    """Mean over steps of the cosine between the flattened update directions."""
    e = E[:, rows].reshape(len(E), -1)
    g = Ghat[:, rows].reshape(len(E), -1)
    c = np.sum(e * g, 1) / np.maximum(np.linalg.norm(e, axis=1)
                                       * np.linalg.norm(g, axis=1), 1e-30)
    return float(np.mean(c))


def fit_rank1(E, G, w, offdiag, iters=500, ridge=1e-12):
    """Minimise sum_t w_t sum_ij (E - G (1 + a_i b_j))^2 by alternating LS.

    With R = E - G this is sum w (R - G a_i b_j)^2, which is linear in a for
    fixed b and vice versa, so each half-step is closed form.
    """
    R = (E - G) * offdiag
    Gw = G * offdiag
    W = w[:, None, None]
    RG = np.sum(W * R * Gw, 0)          # (N, N)
    GG = np.sum(W * Gw * Gw, 0)
    alpha_full = RG / (GG + ridge)      # best unconstrained alpha per entry

    # initialise from the leading singular pair of the identifiable block
    u, sv, vt = np.linalg.svd(alpha_full)
    a, b = u[:, 0] * np.sqrt(sv[0]), vt[0] * np.sqrt(sv[0])
    for _ in range(iters):
        a = (RG @ b) / (GG @ (b * b) + ridge)
        b = (RG.T @ a) / (GG.T @ (a * a) + ridge)
    # fix the scale gauge a -> c a, b -> b / c
    c = np.sqrt(np.linalg.norm(b) / max(np.linalg.norm(a), 1e-30))
    return a * c, b / c, alpha_full


def stage_fit(args):
    d = np.load(data_path(DATA))
    E, G, ntar = d["exact"], d["local"], int(d["ntar"])
    T, N, _ = E.shape
    offdiag = 1.0 - np.eye(N)
    tar = list(range(ntar))
    allr = list(range(N))

    # alternating blocks so train and test both span the whole trajectory
    blk = (np.arange(T) // args.block) % 2
    tr, te = blk == 0, blk == 1
    w = 1.0 / np.maximum(np.linalg.norm(E[:, tar], axis=(1, 2)), 1e-30) ** 2

    a, b, alpha_full = fit_rank1(E[tr], G[tr], w[tr], offdiag)
    alpha1 = np.outer(a, b) * offdiag

    preds = {
        "local (alpha=0)": G,
        "rank-1 alpha": G * (1 + alpha1),
        "full alpha (ceiling)": G * (1 + alpha_full * offdiag),
    }
    print(f"  Ntar={ntar}  steps={T}  train={tr.sum()}  test={te.sum()}  "
          f"(blocks of {args.block})\n")
    print(f"  {'model':<22} {'split':<6} {'rel err tar':>12} {'cos tar':>8} "
          f"{'rel err all':>12} {'cos all':>8}")
    print("  " + "-" * 72)
    out = {"a": a.tolist(), "b": b.tolist(), "alpha_rank1": alpha1.tolist(),
           "alpha_full": (alpha_full * offdiag).tolist(), "metrics": {}}
    for name, P in preds.items():
        for split, m in (("train", tr), ("test", te)):
            r = {"rel_err_target_rows": rel_err(E[m], P[m], tar),
                 "cos_target_rows": cosine(E[m], P[m], tar),
                 "rel_err_all_rows": rel_err(E[m], P[m], allr),
                 "cos_all_rows": cosine(E[m], P[m], allr)}
            out["metrics"][f"{name}|{split}"] = r
            print(f"  {name:<22} {split:<6} {r['rel_err_target_rows']:12.4f} "
                  f"{r['cos_target_rows']:8.4f} {r['rel_err_all_rows']:12.4f} "
                  f"{r['cos_all_rows']:8.4f}")
    nt = np.linalg.norm(E[:, tar], axis=(1, 2)) ** 2
    na = np.linalg.norm(E, axis=(1, 2)) ** 2
    share = float(np.mean(nt / np.maximum(na, 1e-30)))
    out["target_row_energy_share"] = share
    print(f"\n  share of ||g_exact||^2 on target rows (mean over steps): {share:.3f}")
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    print(f"  a (rows 0..{ntar - 1} identifiable): {a[:ntar]}")
    print(f"  b: {b}")
    json.dump(out, open(data_path(FIT), "w"), indent=1)
    print(f"  -> {FIT}")


def stage_plot(args):
    """Exact vs local vs rank-1 fitted gradients over training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(data_path(DATA))
    E, G, ntar = d["exact"], d["local"], int(d["ntar"])
    T, N, _ = E.shape
    fit = json.load(open(data_path(FIT)))
    F = G * (1 + np.array(fit["alpha_rank1"]))
    steps = np.arange(T)
    tar = list(range(ntar))
    test = ((steps // args.block) % 2) == 1

    INK, MUTED = "#0b0b0b", "#52514e"
    C_LOC, C_FIT = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#b5b4ad",
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})

    from matplotlib.ticker import FixedLocator, NullLocator

    def symticks(axis, lin, vmax):
        """Symlog ticks at 0 and every other decade above linthresh only."""
        hi = int(np.ceil(np.log10(vmax)))
        lo = int(np.ceil(np.log10(lin))) + 1
        dec = [10.0 ** e for e in range(hi, lo - 1, -2)]
        axis.set_major_locator(FixedLocator(sorted([-v for v in dec] + [0] + dec)))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: "0" if v == 0 else f"{'−' if v < 0 else ''}1e{int(np.log10(abs(v)))}"))

    def shade(ax):
        for s0 in range(args.block, T, 2 * args.block):
            ax.axvspan(s0, min(s0 + args.block, T), color="#efeeea", lw=0, zorder=0)

    # the four target-row synapses carrying the most exact-gradient energy
    energy = np.sum(E[:, tar] ** 2, 0)
    for i in tar:
        energy[i, i] = -1
    top = np.dstack(np.unravel_index(np.argsort(energy.ravel())[::-1][:4],
                                     energy.shape))[0]

    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.45, wspace=0.32)

    # --- A: per-step relative error on target rows --------------------------
    ax = fig.add_subplot(gs[0, :3])
    shade(ax)
    den = np.linalg.norm(E[:, tar], axis=(1, 2))
    for P, col, ls, lab in [(G, C_LOC, "--", "local  (α = 0)"),
                            (F, C_FIT, "-", "rank-1 fitted")]:
        err = np.linalg.norm((E - P)[:, tar], axis=(1, 2)) / den
        ax.plot(steps, err, color=col, ls=ls, lw=1.4, label=lab)
    ax.set_yscale("log")
    ax.set_xlim(0, T)
    ax.set_xlabel("training step")
    ax.set_ylabel("‖g_exact − ĝ‖ / ‖g_exact‖   (target rows)")
    ax.set_title("A   relative error vs exact gradient   (grey = held-out steps)",
                 loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3, lw=0.5, axis="y")

    # --- B: exact vs fitted, every target-row entry on held-out steps -------
    ax = fig.add_subplot(gs[0, 3])
    off = ~np.eye(N, dtype=bool)
    sel = np.zeros((N, N), bool); sel[tar] = True; sel &= off
    ex_v = E[test][:, sel].ravel()
    lo_v, fi_v = G[test][:, sel].ravel(), F[test][:, sel].ravel()
    lin = np.percentile(np.abs(ex_v[ex_v != 0]), 20)
    ax.scatter(ex_v, lo_v, s=6, color=C_LOC, alpha=0.35, lw=0, label="local")
    ax.scatter(ex_v, fi_v, s=6, color=C_FIT, alpha=0.35, lw=0, label="rank-1")
    m = np.max(np.abs(ex_v))
    ax.plot([-m, m], [-m, m], color=MUTED, lw=0.8, ls=":")
    ax.set_xscale("symlog", linthresh=lin); ax.set_yscale("symlog", linthresh=lin)
    symticks(ax.xaxis, lin, m); symticks(ax.yaxis, lin, m)
    ax.set_xlabel("exact dL/dw"); ax.set_ylabel("approximate dL/dw")
    ax.set_title("B   held-out entries", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, markerscale=2.5, loc="upper left")

    # --- C: traces of the largest synapses ----------------------------------
    for n, (i, j) in enumerate(top):
        ax = fig.add_subplot(gs[1, n])
        shade(ax)
        ax.plot(steps, E[:, i, j], color=INK, lw=1.6, label="exact")
        ax.plot(steps, G[:, i, j], color=C_LOC, lw=1.1, ls="--", label="local")
        ax.plot(steps, F[:, i, j], color=C_FIT, lw=1.1, label="rank-1")
        lt = np.percentile(np.abs(E[:, i, j][E[:, i, j] != 0]), 25)
        ax.set_yscale("symlog", linthresh=lt)
        symticks(ax.yaxis, lt, np.max(np.abs(E[:, i, j])))
        ax.axhline(0, color="#b5b4ad", lw=0.6)
        ax.set_xlim(0, T)
        ax.set_xlabel("training step")
        a_ij = fit["alpha_rank1"][i][j]
        ax.set_title(f"{'C' if n == 0 else ' '}   w[{i}←{j}]   α = {a_ij:+.2f}",
                     loc="left", fontsize=9)
        if n == 0:
            ax.set_ylabel("dL/dw  (symlog)")
            ax.legend(frameon=False, fontsize=7)

    fig.savefig(figure_path(args.out), dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"figure written to {args.out}")


def main():
    global DATA, FIT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["collect", "fit", "plot"])
    ap.add_argument("--out", type=figure_path, default="lowrank_gradients.png")
    ap.add_argument("--data", type=data_path, default=DATA, help="gradient data file (collect/fit/plot)")
    ap.add_argument("--fit-file", type=data_path, default=FIT, help="fit output/input file (fit/plot)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ntar", type=int, default=2)
    ap.add_argument("--w-scale", type=float, default=3.0)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--block", type=int, default=25)
    args = ap.parse_args()
    DATA, FIT = args.data, args.fit_file
    {"collect": stage_collect, "fit": stage_fit, "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
