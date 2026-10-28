"""
Train the QIF RNN with the rank-1 non-locality rule instead of exact gradients.

Three update rules, all from the same initialisation and with the same optimiser
as rnn.py / experiment.stage_train (Adabelief, exponential lr decay):

  exact   full gradient through every spike (autodiff of the event simulation)
  local   dL/dw_ij|local, presynaptic spike times frozen      (alpha = 0)
  rank1   (1 + a_i b_j) * dL/dw_ij|local, alpha from lowrank_fit.json

The local and rank-1 rules only update the rows (and initial states) of target
neurons: every other neuron has dL/dt = 0, so its local gradient is zero.

Wall-clock time per step is measured after compilation. It reflects how the
local gradient is computed *here* (frozen re-simulation + autodiff), not the
cost of an online eligibility-trace implementation; see the notes printed at
the end and the discussion accompanying this file.

Usage (from experiments/rnn/, after `python lowrank.py collect && fit`):

    python train_lowrank.py --trials 400
"""

import argparse
import json
import time

from experiments.rnn.Spike_Experiments import experiment as ex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import grad
from spikegd.qif import QIFNeuron

OUT = "train_lowrank.json"


def net_to_w(M):
    """(N, N) matrix with zero diagonal -> spikegd's (N, N-1) layout."""
    N = M.shape[0]
    return np.stack([np.delete(M[i], i) for i in range(N)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ntar", type=int, default=2)
    ap.add_argument("--w-scale", type=float, default=3.0)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--fit", default="lowrank_fit.json")
    ap.add_argument("--rules", nargs="+", default=["exact", "local", "rank1"],
                    choices=["exact", "local", "rank1"])
    ap.add_argument("--optim", default="adabelief", choices=["adabelief", "sgd"])
    ap.add_argument("--timing-reps", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = "" if args.optim == "adabelief" else f"_{args.optim}"
    args.out = args.out or f"train_lowrank{tag}.png"
    global OUT
    OUT = f"train_lowrank{tag}.json"

    cfg = dict(ex.BASE_CONFIG)
    cfg["seed"] = args.seed
    N, ntar = cfg["N"], args.ntar
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = ex.make_trial(cfg, ntar, args.w_scale)
    ones = jnp.ones_like(trial["w0"])
    one = jnp.array(1.0)
    full = ex.make_full_loss(neuron, trial, ones)
    alpha = jnp.asarray(net_to_w(np.array(json.load(open(args.fit))["alpha_rank1"])))

    def exact_grad(p):
        return jax.value_and_grad(lambda q: full(q[0], q[1], one))(p)

    def rule_grad(p, scale):
        """Local gradient of every target row, rescaled by (1 + alpha) or 1."""
        w, x0 = p
        loss = full(w, x0, one)
        gw, gx = jnp.zeros_like(w), jnp.zeros_like(x0)
        for k in range(ntar):
            queue, rates, _ = ex.record_surround(neuron, trial, w, x0, ones, one, k)
            frozen = ex.make_frozen_loss(neuron, trial, queue, rates, k, ones, one)
            g_w, g_x = grad(frozen, argnums=(0, 1))(w, x0)
            gw = gw.at[k].set(g_w[k] * scale[k])
            gx = gx.at[:, k].set(g_x[:, k])
        return loss, (gw, gx)

    grads = {
        "exact": exact_grad,
        "local": lambda p: rule_grad(p, jnp.ones_like(alpha)),
        "rank1": lambda p: rule_grad(p, 1.0 + alpha),
    }

    # Adam-type optimisers (Adabelief included) divide each parameter's update by
    # a running estimate of that gradient's own scale, so a constant per-synapse
    # factor (1 + alpha_ij) > 0 cancels exactly: rank1 and local then give
    # identical trajectories. Plain SGD is the setting in which alpha can matter.
    sched = optax.exponential_decay(args.lr, max(args.trials // 5, 1), 1 / jnp.e)
    optim = optax.adabelief(sched) if args.optim == "adabelief" else optax.sgd(sched)
    results = {}
    for rule in args.rules:
        gfn = jax.jit(grads[rule])

        @jax.jit
        def step(p, s, gfn=gfn):
            loss, g = gfn(p)
            u, s = optim.update(g, s)
            return optax.apply_updates(p, u), s, loss

        p = (trial["w0"], trial["x0"])
        s = optim.init(p)
        t0 = time.time()
        jax.block_until_ready(step(p, s))          # compile
        compile_s = time.time() - t0

        # steady-state cost of one gradient evaluation (no optimiser update)
        jax.block_until_ready(gfn(p))
        t0 = time.time()
        for _ in range(args.timing_reps):
            jax.block_until_ready(gfn(p))
        grad_ms = 1e3 * (time.time() - t0) / args.timing_reps

        losses = []
        t0 = time.time()
        for _ in range(args.trials):
            p, s, loss = step(p, s)
            losses.append(float(loss))
        losses.append(float(full(p[0], p[1], one)))
        train_s = time.time() - t0
        results[rule] = {"loss": losses, "grad_ms": grad_ms,
                         "train_s": train_s, "compile_s": compile_s}
        print(f"  {rule:<6} final loss {losses[-1]:.4e}   min {min(losses):.4e}   "
              f"grad {grad_ms:6.2f} ms   {args.trials} steps {train_s:5.1f}s   "
              f"(compile {compile_s:.0f}s)", flush=True)

    # number of simulated events: what every cost scales with
    out = neuron.event(trial["x0"], ex.w_to_weights_net(trial["w0"]),
                       trial["weights_in"], trial["spikes_in"], trial["cfg"])
    times, is_in, neurons, _ = out
    n_in = int(jnp.sum(is_in & jnp.isfinite(times)))
    n_net = int(jnp.sum((~is_in) & (neurons >= 0)))
    results["events"] = {"input": n_in, "network": n_net, "buffer": int(times.shape[0])}
    print(f"\n  events per trial: {n_in} input + {n_net} network spikes "
          f"(simulation buffer {times.shape[0]})")
    json.dump(results, open(OUT, "w"), indent=1)

    # --- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED = "#0b0b0b", "#52514e"
    style = {"exact": dict(color=INK, ls="-", label="exact gradient"),
             "local": dict(color="#2a78d6", ls="--", label="local  (α = 0)"),
             "rank1": dict(color="#eb6834", ls="-", label="rank-1 rule")}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#b5b4ad",
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4),
                                 gridspec_kw={"width_ratios": [2.2, 1]})
    for rule in args.rules:
        a1.plot(results[rule]["loss"], lw=1.5, **style[rule])
    a1.set_yscale("log")
    a1.set_xlabel("training step"); a1.set_ylabel("loss")
    a1.set_title(f"A   training loss, same init, {args.optim}  (lr {args.lr:g})",
                 loc="left", fontsize=9.5)
    a1.legend(frameon=False); a1.grid(alpha=0.3, lw=0.5, axis="y")

    names = list(args.rules)
    ms = [results[r]["grad_ms"] for r in names]
    a2.barh(range(len(names)), ms, color=[style[r]["color"] for r in names],
            height=0.55)
    for y, v in enumerate(ms):
        a2.text(v, y, f"  {v:.1f} ms", va="center", color=INK, fontsize=8)
    a2.set_yticks(range(len(names)), [style[r]["label"] for r in names])
    a2.invert_yaxis(); a2.set_xlim(0, max(ms) * 1.35)
    a2.set_xlabel("wall-clock per gradient (CPU, jitted)")
    a2.set_title("B   cost in this implementation", loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"  figure -> {args.out}, results -> {OUT}")


if __name__ == "__main__":
    main()
