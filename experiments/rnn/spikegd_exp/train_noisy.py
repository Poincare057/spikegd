"""
Train with Gaussian jitter on the input spike times: exact vs local vs rank-1.

Every training step draws fresh noise, t_in -> max(t_in + sigma * xi, 0), and
re-sorts the input queue, so each step sees a different input realisation of
the same task (targets are unchanged). The three update rules are the ones in
train_lowrank.py:

  exact   autodiff through the full event simulation
  local   dL/dw_ij with presynaptic spike times frozen        (alpha = 0)
  rank1   (1 + a_i b_j) * local, alpha from lowrank_fit.json   (fitted on clean
          inputs; it is not refitted under noise)

Recorded per step: the loss on that step's noisy trial and the loss on the
clean inputs. After training, the loss is averaged over --n-eval fresh noisy
realisations at the same sigma (held-out noise).

The default optimiser is SGD because Adam-type optimisers cancel any constant
per-synapse factor (1 + alpha_ij) > 0, which makes rank1 identical to local.

Usage (from experiments/rnn/):

    python train_noisy.py --sigmas 0 0.05 0.1 0.2
"""

import argparse

from experiments.rnn.spikegd_exp.experiment import data_path, figure_path
import json
import time

from experiments.rnn.spikegd_exp import experiment as ex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import grad, random
from spikegd.qif import QIFNeuron

from train_lowrank import net_to_w

OUT = "train_noisy.json"
RULES = ["exact", "local", "rank1"]


def jitter(spikes_in, key, sigma):
    """Gaussian jitter on every finite input time, clipped at 0 and re-sorted."""
    times, neurons = spikes_in
    t = jnp.maximum(times + sigma * random.normal(key, times.shape), 0.0)
    t = jnp.where(jnp.isfinite(times), t, jnp.inf)
    order = jnp.argsort(t)
    return t[order], neurons[order]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--ntar", type=int, default=2)
    ap.add_argument("--w-scale", type=float, default=3.0)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--optim", default="sgd", choices=["sgd", "adabelief"])
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--fit", type=data_path, default="lowrank_fit.json")
    ap.add_argument("--out", type=figure_path, default="train_noisy.png")
    args = ap.parse_args()

    cfg = dict(ex.BASE_CONFIG)
    cfg["seed"] = args.seed
    ntar = args.ntar
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = ex.make_trial(cfg, ntar, args.w_scale)
    ones = jnp.ones_like(trial["w0"])
    one = jnp.array(1.0)
    alpha = jnp.asarray(net_to_w(np.array(json.load(open(data_path(args.fit)))["alpha_rank1"])))
    scales = {"local": jnp.ones_like(alpha), "rank1": 1.0 + alpha}

    def loss_on(p, spikes):
        tr = dict(trial, spikes_in=spikes)
        return ex.make_full_loss(neuron, tr, ones)(p[0], p[1], one)

    def gradient(rule, p, spikes):
        if rule == "exact":
            return jax.value_and_grad(loss_on)(p, spikes)
        tr = dict(trial, spikes_in=spikes)
        w, x0 = p
        loss = ex.make_full_loss(neuron, tr, ones)(w, x0, one)
        gw, gx = jnp.zeros_like(w), jnp.zeros_like(x0)
        for k in range(ntar):
            queue, rates, _ = ex.record_surround(neuron, tr, w, x0, ones, one, k)
            frozen = ex.make_frozen_loss(neuron, tr, queue, rates, k, ones, one)
            g_w, g_x = grad(frozen, argnums=(0, 1))(w, x0)
            gw = gw.at[k].set(g_w[k] * scales[rule][k])
            gx = gx.at[:, k].set(g_x[:, k])
        return loss, (gw, gx)

    sched = optax.exponential_decay(args.lr, max(args.trials // 5, 1), 1 / jnp.e)
    optim = optax.sgd(sched) if args.optim == "sgd" else optax.adabelief(sched)

    clean_loss = jax.jit(lambda p: loss_on(p, trial["spikes_in"]))

    @jax.jit
    def eval_noisy(p, keys, sigma):
        return jax.vmap(lambda k: loss_on(p, jitter(trial["spikes_in"], k, sigma)))(keys)

    results = {"args": vars(args), "runs": {}}
    for sigma in args.sigmas:
        for rule in RULES:
            @jax.jit
            def step(p, s, key, rule=rule, sigma=sigma):
                spikes = jitter(trial["spikes_in"], key, sigma)
                loss, g = gradient(rule, p, spikes)
                u, s = optim.update(g, s)
                return optax.apply_updates(p, u), s, loss

            # identical noise sequence for every rule at a given sigma
            key = random.PRNGKey(args.noise_seed)
            p = (trial["w0"], trial["x0"])
            s = optim.init(p)
            noisy, clean = [], [float(clean_loss(p))]
            t0 = time.time()
            for _ in range(args.trials):
                key, sub = random.split(key)
                p, s, loss = step(p, s, sub)
                noisy.append(float(loss))
                clean.append(float(clean_loss(p)))
            ekeys = random.split(random.PRNGKey(10_000 + args.noise_seed), args.n_eval)
            held = np.asarray(eval_noisy(p, ekeys, sigma))
            run = {"noisy_train_loss": noisy, "clean_loss": clean,
                   "heldout_noisy_mean": float(held.mean()),
                   "heldout_noisy_sem": float(held.std(ddof=1) / np.sqrt(len(held))),
                   "seconds": time.time() - t0}
            results["runs"][f"{sigma:g}|{rule}"] = run
            print(f"  sigma {sigma:<5g} {rule:<6} clean final {clean[-1]:.4f}   "
                  f"held-out noisy {run['heldout_noisy_mean']:.4f} "
                  f"± {run['heldout_noisy_sem']:.4f}   ({run['seconds']:.0f}s)", flush=True)
    json.dump(results, open(data_path(OUT), "w"), indent=1)

    # --- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED = "#0b0b0b", "#52514e"
    style = {"exact": dict(color=INK, ls="-", label="exact"),
             "local": dict(color="#2a78d6", ls="--", label="local  (α = 0)"),
             "rank1": dict(color="#eb6834", ls="-", label="rank-1")}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#b5b4ad",
                         "xtick.color": MUTED, "ytick.color": MUTED})
    S = len(args.sigmas)
    fig, axes = plt.subplots(1, S + 1, figsize=(3.1 * (S + 1), 3.2),
                             gridspec_kw={"width_ratios": [1] * S + [1.15]})
    lo = min(min(r["clean_loss"]) for r in results["runs"].values())
    hi = max(max(r["clean_loss"]) for r in results["runs"].values())
    for ax, sigma in zip(axes, args.sigmas):
        for rule in RULES:
            r = results["runs"][f"{sigma:g}|{rule}"]
            ax.plot(r["noisy_train_loss"], color=style[rule]["color"], lw=0.6, alpha=0.25)
            ax.plot(r["clean_loss"], lw=1.4, **style[rule])
        ax.set_yscale("log"); ax.set_ylim(lo * 0.9, hi * 1.1)
        ax.set_xlabel("training step")
        ax.set_title(f"σ = {sigma:g}", loc="left", fontsize=9.5)
        ax.grid(alpha=0.3, lw=0.5, axis="y")
    axes[0].set_ylabel("loss on clean inputs\n(faint: that step's noisy trial)")
    axes[0].legend(frameon=False, fontsize=7)

    ax = axes[-1]
    xs = np.arange(S)
    for d, rule in zip((-0.22, 0, 0.22), RULES):
        m = [results["runs"][f"{s:g}|{rule}"]["heldout_noisy_mean"] for s in args.sigmas]
        e = [results["runs"][f"{s:g}|{rule}"]["heldout_noisy_sem"] for s in args.sigmas]
        ax.errorbar(xs + d, m, yerr=e, fmt="o", ms=5, capsize=2, lw=1,
                    color=style[rule]["color"], label=style[rule]["label"])
    ax.set_xticks(xs, [f"{s:g}" for s in args.sigmas])
    ax.set_xlabel("input jitter σ")
    ax.set_ylabel(f"loss on {args.n_eval} held-out noisy inputs")
    ax.set_title("after training (mean ± s.e.m.)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3, lw=0.5, axis="y")
    fig.suptitle(f"Training under input spike-time jitter  ({args.optim}, lr {args.lr:g})",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(figure_path(args.out), dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"  figure -> {args.out}, results -> {OUT}")


if __name__ == "__main__":
    main()
