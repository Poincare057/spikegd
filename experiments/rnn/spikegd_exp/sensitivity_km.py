"""
Sensitivity of training to input spike-time jitter, in the published setting.

Training follows Klos & Memmesheimer's rnn.ipynb exactly (config_qif):
  seed 7, N = 10, Ntar = 2, r_targets = [1, 1/2], K = 500, T = 10,
  recurrent weights and initial states start at zero,
  AdaBelief with exponential_decay(1e-2, 2000, 1/e), 10,000 trials.
The only change is the update rule, and fresh Gaussian jitter on the input
spike times at every trial, t_in -> max(t_in + sigma * xi, 0).

Rules (same init, optimiser and noise draws):
  exact   autodiff through the full simulation (as in rnn.py)
  local   one-pass trace gradient, alpha = 0                     (fastgrad.py)
  rank1   one-pass trace gradient x (1 + a_i b_j), alpha fitted on the
          published exact run (lowrank_fit_km.json)

Note: AdaBelief divides every update by a running estimate of that
gradient's own scale, so a constant per-synapse factor (1 + alpha_ij) > 0
cancels up to the optimiser's 1e-16 epsilon, while a negative factor reverses
that synapse's update at full size. With the published-run fit, target
neuron 0 has 1 + alpha ~ 0.99 and target neuron 1 has |1 + alpha| ~ 0.06 with
4 of 9 factors negative, so rank1 equals local except on those 4 synapses,
which it trains in the opposite direction.

  rank1c  the corrected rank-1 rule: AdaBelief is given the plain local
          gradient, and (1 + a_i b_j) multiplies its normalised update
          before the learning rate, i.e.
              dw_ij = -lr(t) * (1 + alpha_ij) * AdaBelief(g_local)_ij,
          so the optimiser cannot cancel alpha (same b1, b2, eps, schedule).
  rank2c  the same corrected rule with a rank-2 alpha. With 2 target neurons
          only 2 rows of alpha are identifiable, so rank 2 is the unrestricted
          per-synapse optimum (lowrank.py's "alpha_full").
  rank1sgd  the rank-1 rule as written in the notes, with plain SGD:
              dw_ij = -lr(t) * (1 + alpha_ij) * g_local,ij,
          lr(t) = exponential_decay(--lr-sgd, 2000, 1/e); alpha is the
          raw-gradient fit (lowrank_fit_km.json), the right fit for SGD.
  betac   the neuron-specific, synapse-independent rule from the notes:
              dw_ij = -lr(t) * (1 + beta_i) * AdaBelief(g_local)_ij,
          AdaBelief normalises the plain local gradient first and the factor
          multiplies the normalised update, so it is not cancelled; beta_i
          from fit_beta.py (beta_fit_km.json).
  betat   the same with a time-dependent factor (1 + beta_i(t)): beta_i fitted
          on successive windows of the published exact run (fit_beta.py
          "schedule") and interpolated in log(trial) between window midpoints.
  matpre  per-neuron synapse-mixing matrix (fit_matrix.py, "A_per_neuron"):
          each target row of the local gradient is multiplied by its own
          (I + A^(i)) and the result is passed to AdaBelief.
  matk    spike-specific matrices (fit_spike_matrix.py): the eligibility of each
          output k of target i is multiplied by its own (I + A^(i,k)) as it is
          accumulated (fastgrad spike_mix), and the result is passed to AdaBelief.
  matkc   matk plus explicit cross-target terms (fit_spike_matrix.py fitcross):
          each output's eligibility also gets the learning signal
          sum_{m != i} sum_c' dL/dt_{c'_m} C^(m,i)_{c'c}; passed to AdaBelief.
  matpost the per-neuron matrix applied after AdaBelief's normalisation:
          dw = -lr(t) * AdaBelief(g_local) (I + A^(i)).
  localsgd  the local rule (alpha = 0) with the same plain SGD and learning
          rate, so it differs from rank1sgd only by the (1 + alpha) factor.
  feedback  learning-signal rule (fit_feedback.py): local term plus
          (B delta)_i * ebar_ij for every neuron i, B fitted on a published
          exact run; trained with plain AdaBelief (additive, so not cancelled).

All noise seeds of one (rule, sigma) are trained in parallel with vmap.
Parameters and the clean-input loss are recorded every --record trials.

Usage (from experiments/rnn/):

    python sensitivity_km.py run      # -> sensitivity_km.npz
    python sensitivity_km.py plot     # -> sensitivity_km.png
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
from jax import lax, random
from spikegd.qif import QIFNeuron

from experiments.rnn.spikegd_exp.fastgrad import make_fast_grad
from train_lowrank import net_to_w
from train_noisy import jitter

DATA = "sensitivity_km.npz"
RULES = ["exact", "local", "rank1", "rank1c", "rank2c", "rank1sgd", "localsgd", "betac",
         "betat", "matpre", "matpost", "matk", "matkc", "feedback"]


def build(args):
    cfg = dict(ex.BASE_CONFIG)          # identical to rnn.ipynb's config_qif
    ntar = cfg["Ntar"]
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = ex.make_trial(cfg, ntar, 0.0)   # w_scale 0: zero initial weights
    ones = jnp.ones_like(trial["w0"]); one = jnp.array(1.0)
    # alpha is only needed by the rank-1 / rank-2 rules; exact and local runs
    # must not depend on the fit file existing
    if any(r in args.rules for r in ("rank1", "rank1c", "rank2c", "rank1sgd")):
        fit = json.load(open(data_path(args.fit)))
        alpha = jnp.asarray(net_to_w(np.array(fit["alpha_rank1"])))
        alpha2 = jnp.asarray(net_to_w(np.array(fit["alpha_full"])))
    else:
        alpha = alpha2 = jnp.zeros_like(trial["w0"])
    trace = make_fast_grad(neuron, trial, ntar, "trace")

    def loss_on(p, spikes):
        return ex.make_full_loss(neuron, dict(trial, spikes_in=spikes), ones)(
            p[0], p[1], one)

    def rule(scale):
        def f(p, spikes):
            _, (gw, gx) = trace(p[0], p[1], spikes)
            return (gw * scale, gx)
        return f

    grads = {"exact": lambda p, s: jax.grad(loss_on)(p, s),
             "local": rule(jnp.ones_like(alpha)), "rank1": rule(1.0 + alpha),
             "rank1c": rule(jnp.ones_like(alpha)), "rank2c": rule(jnp.ones_like(alpha)),
             "rank1sgd": rule(1.0 + alpha), "localsgd": rule(jnp.ones_like(alpha)),
             "betac": rule(jnp.ones_like(alpha)), "betat": rule(jnp.ones_like(alpha)),
             "matpost": rule(jnp.ones_like(alpha))}
    if any(r in args.rules for r in ("matpre", "matpost")):
        N = trial["w0"].shape[0]
        Mix = jnp.eye(N)[None] + jnp.asarray(json.load(open(data_path(args.A_fit)))["A_per_neuron"])
        OFF = jnp.array([[j for j in range(N) if j != i] for i in range(N)])

        def mix(gw):
            """Multiply each row of a (N, N-1) weight array by its own (I + A^(i))."""
            out = jnp.einsum("ij,ijk->ik", ex.w_to_weights_net(gw), Mix)
            return jnp.take_along_axis(out, OFF, axis=1)

        def matpre(p, s):
            _, (gw, gx) = trace(p[0], p[1], s)
            return mix(gw), gx
        grads["matpre"] = matpre
    if "matk" in args.rules:
        spike_trace = make_fast_grad(
            neuron, trial, trial["cfg"]["Ntar"], "trace",
            spike_mix=np.array(json.load(open(data_path(args.spike_A_fit)))["spike_mix"]))

        def matk(p, s):
            return spike_trace(p[0], p[1], s)[1]
        grads["matk"] = matk
    if "matkc" in args.rules:
        fit_c = json.load(open(data_path(args.spike_AC_fit)))
        cross_trace = make_fast_grad(
            neuron, trial, trial["cfg"]["Ntar"], "trace",
            spike_mix=np.array(fit_c["spike_mix"]), cross=np.array(fit_c["cross"]))

        def matkc(p, s):
            return cross_trace(p[0], p[1], s)[1]
        grads["matkc"] = matkc
    if "feedback" in args.rules:
        from fit_feedback import make_rule as make_feedback
        fb = make_feedback(neuron, trial, json.load(open(data_path(args.fb_fit)))["B"])
        grads["feedback"] = lambda p, s: fb(p[0], p[1], s)
    sched = optax.exponential_decay(args.lr, args.decay_steps, 1 / jnp.e)
    adabelief = optax.adabelief(sched)

    # optax.adabelief(lr) == chain(scale_by_belief(), scale_by_learning_rate(lr));
    # the corrected rules insert the per-synapse factor between the two halves
    def corrected(scale):
        return optax.chain(
            optax.scale_by_belief(),
            optax.stateless(lambda u, params=None: (u[0] * scale, u[1])),
            optax.scale_by_learning_rate(sched))

    optims = {r: adabelief for r in grads}
    optims["rank1c"] = corrected(1.0 + alpha)
    optims["rank2c"] = corrected(1.0 + alpha2)
    optims["rank1sgd"] = optax.sgd(
        optax.exponential_decay(args.lr_sgd, args.decay_steps, 1 / jnp.e))
    optims["localsgd"] = optims["rank1sgd"]
    if "betac" in args.rules:
        beta = jnp.asarray(json.load(open(data_path(args.beta_fit)))["beta"])
        # one factor per postsynaptic neuron, shared by all of its input synapses
        optims["betac"] = corrected((1.0 + beta)[:, None] * jnp.ones_like(trial["w0"]))
    if "matpost" in args.rules:
        optims["matpost"] = optax.chain(
            optax.scale_by_belief(),
            optax.stateless(lambda u, params=None: (mix(u[0]), u[1])),
            optax.scale_by_learning_rate(sched))
    if "betat" in args.rules:
        sch = json.load(open(data_path(args.beta_fit)))["schedule"]
        xk = jnp.log1p(jnp.asarray(sch["knots"]))
        fk = 1.0 + jnp.asarray(sch["beta"])               # (windows, N)

        def by_trial():
            """Multiply neuron i's normalised update by 1 + beta_i(trial)."""
            def init(params):
                return jnp.zeros([], jnp.int32)

            def update(u, count, params=None):
                x = jnp.log1p(count.astype(fk.dtype))
                f = jax.vmap(lambda col: jnp.interp(x, xk, col), in_axes=1)(fk)   # (N,)
                return (u[0] * f[:, None], u[1]), count + 1
            return optax.GradientTransformation(init, update)

        optims["betat"] = optax.chain(optax.scale_by_belief(), by_trial(),
                                      optax.scale_by_learning_rate(sched))
    return trial, loss_on, grads, optims


def make_train(trial, loss_on, g, optim, args):
    """Train one noise seed; returns recorded (theta, clean loss) and end params."""
    n_blocks = args.trials // args.record

    def train(key, sigma):
        p0 = (trial["w0"], trial["x0"])
        s0 = optim.init(p0)

        def one_step(carry, k):
            p, s = carry
            gr = g(p, jitter(trial["spikes_in"], k, sigma))
            u, s = optim.update(gr, s)
            return (optax.apply_updates(p, u), s), None

        def block(carry, keys):
            carry, _ = lax.scan(one_step, carry, keys)
            p = carry[0]
            theta = jnp.concatenate([jnp.ravel(p[0]), jnp.ravel(p[1])])
            return carry, (theta, loss_on(p, trial["spikes_in"]))

        keys = random.split(key, n_blocks * args.record).reshape(
            n_blocks, args.record, 2)
        (p_end, _), (thetas, clean) = lax.scan(block, (p0, s0), keys)
        theta0 = jnp.concatenate([jnp.ravel(p0[0]), jnp.ravel(p0[1])])
        thetas = jnp.concatenate([theta0[None], thetas])
        clean = jnp.concatenate([loss_on(p0, trial["spikes_in"])[None], clean])
        return thetas, clean, p_end

    return train


def stage_run(args):
    trial, loss_on, grads, optims = build(args)
    sig = args.sigmas
    S, R = len(sig), args.n_seeds
    # keep results of rules not being rerun
    out = {}
    try:
        old = np.load(data_path(DATA))
        if np.array_equal(old["sigmas"], np.array(sig)):
            out = {k: old[k] for k in old.files
                   if k not in ("sigmas", "args") and k.split("_")[0] not in args.rules}
    except FileNotFoundError:
        pass
    t_start = time.time()
    eval_noisy = jax.jit(jax.vmap(
        lambda p, keys, s: jnp.mean(jax.vmap(
            lambda k: loss_on(p, jitter(trial["spikes_in"], k, s)))(keys))))

    # every (sigma, seed) pair; sigma = 0 is deterministic, so it runs once.
    # Seed r uses the same base key for every rule (paired comparison); the
    # sigma index is folded in so that noise levels are independent.
    jobs = [(si, r) for si, s in enumerate(sig) for r in range(1 if s == 0 else R)]
    keys = jnp.stack([random.fold_in(random.PRNGKey(1000 + r), si) for si, r in jobs])
    sigs = jnp.array([sig[si] for si, _ in jobs])
    ekeys = jnp.stack([random.split(random.fold_in(random.PRNGKey(5000 + r), si),
                                    args.n_eval) for si, r in jobs])
    for rule in args.rules:
        train = jax.jit(jax.vmap(make_train(trial, loss_on, grads[rule], optims[rule],
                                            args)))
        th, cl, p_end = train(keys, sigs)
        ho = np.asarray(eval_noisy(p_end, ekeys, sigs))
        th, cl = np.asarray(th), np.asarray(cl)
        B = cl.shape[-1]
        TH = np.zeros((S, R, B, th.shape[-1])); CL = np.zeros((S, R, B)); HO = np.zeros((S, R))
        for j, (si, r) in enumerate(jobs):
            rs = range(R) if sig[si] == 0 else [r]
            for rr in rs:
                TH[si, rr], CL[si, rr], HO[si, rr] = th[j], cl[j], ho[j]
        out[f"{rule}_theta"], out[f"{rule}_clean"], out[f"{rule}_heldout"] = TH, CL, HO
        for si, s in enumerate(sig):
            print(f"  {rule:<6} sigma {s:<6g} final clean median "
                  f"{np.median(CL[si, :, -1]):.4f}  held-out noisy median "
                  f"{np.median(HO[si]):.4f}", flush=True)
        print(f"  {rule} done ({time.time() - t_start:.0f}s)", flush=True)
        np.savez(data_path(DATA), sigmas=np.array(sig), args=json.dumps(vars(args)), **out)
    print(f"  -> {DATA}")


def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(data_path(DATA))
    sig = d["sigmas"]
    meta = json.loads(str(d["args"]))
    rules = [r for r in args.plot_rules if f"{r}_clean" in d]
    rec = meta["record"]
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#b5b4ad"
    ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b", "#06203f"]
    pos = [s for s in sig if s > 0]
    col = {float(s): (INK if s == 0 else ramp[pos.index(s)]) for s in sig}
    rule_style = {"exact": dict(color=INK, marker="o", label="exact"),
                  "local": dict(color="#2a78d6", marker="s", label="local"),
                  "rank1": dict(color="#eb6834", marker="^", label="rank-1 (inside AdaBelief)"),
                  "rank1c": dict(color="#eb6834", marker="^", label="rank-1"),
                  "rank2c": dict(color="#1baf7a", marker="D", label="rank-2"),
                  "rank1sgd": dict(color="#1baf7a", marker="D", label="rank-1, plain SGD"),
                  "localsgd": dict(color="#6250d6", marker="v", label="local, plain SGD"),
                  "betac": dict(color="#eb6834", marker="^", label="constant β"),
                  "betat": dict(color="#1baf7a", marker="D", label="β(t)"),
                  "matpre": dict(color="#6250d6", marker="D", label="local + per-neuron A"),
                  "matpost": dict(color="#e87ba4", marker="P", label="matrix A, after AdaBelief"),
                  "matk": dict(color="#eb6834", marker="v", label="local + matrix A"),
                  "matkc": dict(color="#1baf7a", marker="^", label="local + matrix A + cross-target"),
                  "feedback": dict(color="#6250d6", marker="v", label="learning signal (B)")}
    names = {"exact": "exact gradient", "local": "local gradient", "betac": "constant β",
             "betat": "β(t)", "matpre": "local + per-neuron A", "matk": "local + matrix A",
             "matkc": "A + cross-target",
             "matpost": "A, after AdaBelief",
             "rank1": "rank-1, α inside AdaBelief", "rank1c": "rank-1 rule",
             "rank2c": "rank-2 rule", "rank1sgd": "rank-1, SGD",
             "localsgd": "local, SGD",
             "feedback": "learning-signal rule (B)"}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": GRID,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    nc = len(rules)
    fig, axes = plt.subplots(2, nc + 1, figsize=(3.75 * (nc + 1), 6.6),
                             gridspec_kw={"width_ratios": [1] * nc + [0.95],
                                          "hspace": 0.42, "wspace": 0.3})
    R = d[f"{rules[0]}_clean"].shape[1]
    trials = np.arange(d[f"{rules[0]}_clean"].shape[-1]) * rec

    lo = min(np.percentile(d[f"{r}_clean"], 25, axis=1).min() for r in rules)
    hi = max(np.percentile(d[f"{r}_clean"], 75, axis=1).max() for r in rules)
    dev = {}
    for r in rules:
        th = d[f"{r}_theta"]
        dev[r] = np.linalg.norm(th - th[sig == 0][0][None, :1], axis=-1)
    dpos = min(np.percentile(v[sig > 0], 25, axis=1)[:, 1:].min() for v in dev.values())
    dpos = max(dpos, 1e-6)
    dmax = max(np.percentile(v, 75, axis=1).max() for v in dev.values())

    for c, r in enumerate(rules):
        ax = axes[0, c]
        cl = d[f"{r}_clean"]
        for si, s in enumerate(sig):
            if s > 0:
                q1, q3 = np.percentile(cl[si], [25, 75], axis=0)
                ax.fill_between(trials, q1, q3, color=col[s], alpha=0.2, lw=0)
            ax.plot(trials, np.median(cl[si], 0), color=col[s],
                    lw=1.6 if s == 0 else 1.3, label=f"σ = {s:g}")
        ax.set_xscale("symlog", linthresh=rec); ax.set_yscale("log")
        ax.set_ylim(lo * 0.8, hi * 1.2)
        ax.set_title(f"{'A' if c == 0 else ' '}   {names[r]}: loss",
                     loc="left", fontsize=9.5)
        ax.set_xlabel("trial"); ax.grid(alpha=0.3, lw=0.5, axis="y")
        if c == 0:
            ax.set_ylabel("loss (line: median, band: interquartile\n"
                          f"range over {R} noise seeds)")
            ax.legend(frameon=False, fontsize=7, loc="lower left")

        ax = axes[1, c]
        for si, s in enumerate(sig):
            if s == 0:
                continue
            v = dev[r][si]
            q1, q3 = np.percentile(v, [25, 75], axis=0)
            ax.fill_between(trials, np.maximum(q1, dpos), np.maximum(q3, dpos),
                            color=col[s], alpha=0.2, lw=0)
            ax.plot(trials, np.maximum(np.median(v, 0), dpos), color=col[s], lw=1.3,
                    label=f"σ = {s:g}")
        ax.set_xscale("symlog", linthresh=rec); ax.set_yscale("log")
        ax.set_ylim(dpos, dmax * 1.5)
        ax.set_title(f"{'B' if c == 0 else ' '}   {names[r]}: drift",
                     loc="left", fontsize=9.5)
        ax.set_xlabel("trial"); ax.grid(alpha=0.3, lw=0.5, axis="y")
        if c == 0:
            ax.set_ylabel("‖θ_σ(t) − θ_0(t)‖")
            ax.legend(frameon=False, fontsize=7, loc="lower right")

    for row, (key, ylab, title) in enumerate([
            ("heldout", f"loss on {meta['n_eval']} held-out noisy inputs",
             "C   final loss vs noise"),
            ("drift", "final ‖θ_σ − θ_0‖", "D   final drift vs noise")]):
        ax = axes[row, nc]
        for k, r in enumerate(rules):
            off = (k - (len(rules) - 1) / 2) * 0.08
            if key == "heldout":
                vals, xs = d[f"{r}_heldout"], sig
            else:
                vals, xs = dev[r][sig > 0][:, :, -1], np.array(pos)
            xi = np.arange(len(xs)) + off
            med = np.median(vals, 1)
            q1, q3 = np.percentile(vals, [25, 75], axis=1)
            ax.errorbar(xi, med, yerr=[med - q1, q3 - med], capsize=2.5, elinewidth=1,
                        lw=1.2, ms=5, **rule_style[r])
        ax.set_xticks(np.arange(len(xs)), [f"{x:g}" for x in xs])
        ax.set_xlabel("input jitter σ"); ax.set_ylabel(ylab)
        if key == "drift":
            ax.set_yscale("log")
        else:
            ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.grid(alpha=0.3, lw=0.5, axis="y")
        if row == 1:   # one legend for both summary panels, below them
            ax.legend(frameon=False, fontsize=7.5, loc="upper center",
                      bbox_to_anchor=(0.5, -0.2), ncol=2)

    opt = ("AdaBelief" if not any(r.endswith("sgd") for r in rules)
           else "AdaBelief unless marked SGD")
    fig.suptitle("Sensitivity of training to input spike-time jitter, published setting "
                 f"({opt}, {meta['trials']:,} trials, zero init; median ± "
                 f"interquartile range over {R} noise seeds)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.savefig(figure_path(args.out), dpi=170, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"  figure -> {args.out}")

    if args.separate:
        # each part of the same figure as its own image: <stem>_<part>.png
        from matplotlib.transforms import Bbox
        stem = args.out.rsplit(".", 1)[0]
        leg_c = axes[0, nc].legend(frameon=False, fontsize=7.5)   # C gets its own legend
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        parts = {"loss_curves": list(axes[0, :nc]), "drift_curves": list(axes[1, :nc]),
                 "final_loss": [axes[0, nc]], "final_drift": [axes[1, nc]]}
        title = fig._suptitle
        boxes = {name: Bbox.union([a.get_tightbbox(renderer) for a in axs])
                 .transformed(fig.dpi_scale_trans.inverted()).padded(0.08)
                 for name, axs in parts.items()}       # measured while all are visible
        for name, axs in parts.items():
            bb = boxes[name]
            for a in axes.flat:                  # show only this part's panels
                a.set_visible(any(a is b for b in axs))
            if title is not None:
                title.set_visible(False)
            path = f"{stem}_{name}.png"
            fig.savefig(figure_path(path), dpi=200, bbox_inches=bb, facecolor="#fcfcfb")
            print(f"  part   -> {path}")
        for a in axes.flat:
            a.set_visible(True)
        if title is not None:
            title.set_visible(True)
        leg_c.remove()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["run", "plot"])
    ap.add_argument("--trials", type=int, default=10_000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--decay-steps", type=int, default=2_000)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.025, 0.05, 0.1, 0.2])
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--record", type=int, default=50, help="record every N trials")
    ap.add_argument("--rules", nargs="+", default=RULES, choices=RULES)
    ap.add_argument("--plot-rules", nargs="+",
                    default=["exact", "local", "rank1c", "localsgd", "rank1sgd"],
                    choices=RULES)
    ap.add_argument("--fit", type=data_path, default="lowrank_fit_km.json")
    ap.add_argument("--fb-fit", type=data_path, default="feedback_fit.json")
    ap.add_argument("--separate", action="store_true",
                    help="plot: also save each part of the figure as its own image")
    ap.add_argument("--beta-fit", type=data_path, default="beta_fit_km.json")
    ap.add_argument("--A-fit", type=data_path, dest="A_fit", default="A_fit_km.json")
    ap.add_argument("--spike-A-fit", type=data_path, dest="spike_A_fit", default="spike_A_fit_km.json")
    ap.add_argument("--spike-AC-fit", type=data_path, dest="spike_AC_fit", default="spike_AC_fit_km.json")
    ap.add_argument("--lr-sgd", type=float, default=3.0,
                    help="initial SGD learning rate for rank1sgd")
    ap.add_argument("--out", type=figure_path, default="sensitivity_km.png")
    args = ap.parse_args()
    {"run": stage_run, "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
