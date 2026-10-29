"""
Local vs non-local influence on the exact gradient of a recurrent QIF network.

Fix a reference neuron K and look at the row dL/dW[K, :], the gradient of the
synapses feeding into K. Every three-factor rule (e-prop, RFLO) has to produce
this row from a synapse-local eligibility trace scaled by a per-neuron learning
signal, which means it can only represent influence that stays inside K. This
script measures how much of the exact gradient that leaves out.
  g_exact     full recurrent gradient of row K
  g_local     the same gradient with the surround frozen, so every path that
              leaves K is severed: the direct-path contribution alone
  g_nonlocal  g_exact - g_local, by definition (a residual, not an independent measurement)
  rho         ||g_nonlocal|| / ||g_exact||, reported as a percentage
Two graph conditions remove indirect paths, interpolated by eps in [0, 1]:
  sink      empties K's outgoing column, so nothing leaves K at all. Both
            indirect families (returning and non-returning) are gone.
  acyclic   partitions the other neurons into U and D; K receives only from U,
            sends only into D, and nothing in D reaches U or K. Cycles through
            K are gone but K still drives the network.
At eps = 0 either condition makes g_local and g_exact the same object, so
rho = 0 exactly. 

Usage:
    python experiment.py train        --trials 400
    python experiment.py decompose    --eps 0 0.25 0.5 1.0
    python experiment.py partitions   --n-partitions 30
    python experiment.py figure
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax   
import jax.numpy as jnp   
import numpy as np   
import optax   
from jax import grad, random   
from jaxtyping import Array
from experiments.rnn.rnn import init_input, init_target, init_weights, outfn   
from output_paths import DATA_DIR, FIGS_DIR, ensure_output_dirs, output_file, svg_path   
from spikegd.qif import QIFNeuron   

# %%
############################
### Configuration
############################


# Published config from rnn.ipynb.
BASE_CONFIG = {
    "seed": 7,
    "tau": 1.0,
    "eps": 1e-6,
    "alpha": 10.0,
    "N": 10,
    "w_in_exc": 5.0,
    "w_in_inh": -5.0,
    "r_in": 1.0,
    "T": 10.0,
    "K": 500,
    "dt": 0.01,
    "Ntar": 2,
    "r_targets": [1.0, 0.5],
}

SNAPS = DATA_DIR / "snaps.pkl"
DECOMP = DATA_DIR / "decomposition.json"
PARTS = DATA_DIR / "partitions.json"


# %%
############################
### Model
############################


def w_to_weights_net(w: Array) -> Array:
    """
    Vectorised rnn.w_to_weights_net, traceable under jit.
    """
    N = w.shape[0]
    wn = jnp.zeros((N, N))
    r, c = jnp.tril_indices(N, -1)
    wn = wn.at[r, c].set(w[r, c])
    r, c = jnp.triu_indices(N, 1)
    return wn.at[r, c].set(w[r, c - 1])


def _mask_from_pairs(N: int, k: int, pairs: list[tuple[int, int]]) -> Array:
    """
    Build a (N, N-1) mask that is 0 on the given (post, pre) synapses.
    """
    mask = np.ones((N, N - 1))
    for j, i in pairs:  # synapse j -> i
        if i == j: continue
        mask[i, j if j < i else j - 1] = 0.0
    return jnp.asarray(mask)


def sink_mask(N: int, k: int) -> Array:
    """
    Block every synapse leaving K: K receives from all, sends to none.
    """
    return _mask_from_pairs(N, k, [(k, i) for i in range(N) if i != k])


def acyclic_mask(N: int, k: int, D: Iterable[int]) -> Array:
    """
    Block the synapses that would close a cycle through K.

    D is the downstream set; U is everything else except K. Blocked blocks are
    D -> U, D -> K and K -> U.
    """
    D = list(D)
    U = [i for i in range(N) if i != k and i not in D]
    pairs = [(j, i) for j in D for i in U + [k]] + [(k, i) for i in U]
    return _mask_from_pairs(N, k, pairs)


def apply_eps(w: Array, mask: Array, eps: float | Array) -> Array:
    """
    W_ij(eps): blocked synapses scaled by eps, all others untouched.
    eps is part of the parameterisation rather than an edit applied before
    differentiating, so the chain rule carries the factor through every blocked
    entry. This matters whenever the mask touches row K itself, as the acyclic
    mask does: such a synapse has zero value but a non-zero raw gradient.
    """
    return jnp.where(mask == 0.0, eps * w, w)


# %%
############################
### Initialization
############################


def make_trial(cfg: dict, ntar: int, w_scale: float) -> dict:
    """
    Input spikes, targets, input weights and init, exactly as rnn.py.
    """

    ### Unpack arguments
    c = dict(cfg)
    c["Ntar"] = ntar
    c["r_targets"] = cfg["r_targets"][:ntar]

    ### Input and target spikes
    key = random.PRNGKey(c["seed"])
    key, spikes_in = init_input(key, c)
    weights_in = init_weights(c)
    key, t_targets = init_target(key, c)

    ### Initial parameters
    key, subkey = random.split(key)
    w0 = w_scale * random.normal(subkey, (c["N"], c["N"] - 1))
    x0 = jnp.zeros((2, c["N"]))
    denom = float(sum(len(t) for t in t_targets))
    return dict(
        cfg=c,
        spikes_in=spikes_in,
        weights_in=weights_in,
        t_targets=t_targets,
        w0=w0,
        x0=x0,
        denom=denom,
    )


# %%
############################
### Loss functions
############################


def lossfn_shared(t_outs: list[Array], t_targets: list[Array], denom: float) -> Array:
    """
    rnn.lossfn with the normaliser passed in.

    rnn.lossfn divides by the total number of targets, which would rescale the
    gradient purely as a side effect of Ntar and contaminate any comparison
    across loss conditions. Fixing the denominator also lets the frozen loss,
    which only ever sees K, use the same scale as the recurrent one.
    """
    total = 0.0
    for t_out, t_target in zip(t_outs, t_targets):
        total = total + jnp.sum(
            ((t_out[:-1] - t_target[:-1]) / (t_target[:-1] + 2)) ** 2
        )
        total = total + ((t_out[-1] - t_target[-1]) / (t_target[-1] + 2)) ** 2 * (
            t_out[-1] < t_target[-1]
        )
    return total / denom


def make_full_loss(neuron: QIFNeuron, trial: dict, mask: Array) -> Callable:
    """
    L(w, x0, eps) over the whole network.
    """
    cfg, denom = trial["cfg"], trial["denom"]

    def loss_fn(w: Array, x0: Array, eps: float | Array) -> Array:
        w_eff = apply_eps(w, mask, eps)
        out = neuron.event(
            x0, w_to_weights_net(w_eff), trial["weights_in"], trial["spikes_in"], cfg
        )
        t_outs = outfn(neuron, out, [w_eff, x0], trial["t_targets"], cfg)
        return lossfn_shared(t_outs, trial["t_targets"], denom)

    return loss_fn


def record_surround(
    neuron: QIFNeuron,
    trial: dict,
    w: Array,
    x0: Array,
    mask: Array,
    eps: float | Array,
    k: int,
) -> tuple:
    """
    Run the network once and freeze everything the surround produces.

    The recorded event queue is a constant, so when K is resimulated against it
    no gradient can flow out of K and back. That is what makes the resulting
    gradient direct-path-only without committing to a particular trace.

    K's own spikes are dropped from the queue. They would carry zero weight, but
    each lands at exactly the time K spikes in the resimulation, and that tie
    sends the gradient down the wrong branch (up to ~9% error at eps = 0, sink).
    """
    cfg = trial["cfg"]
    w_eff = apply_eps(w, mask, eps)
    out = neuron.event(
        x0, w_to_weights_net(w_eff), trial["weights_in"], trial["spikes_in"], cfg
    )
    times, is_in, neurons, xs = out

    N, Nin = cfg["N"], trial["weights_in"].shape[1]
    # event -> channel:  external input i -> i
    #                    network spike from j -> Nin + j
    #                    post-trial padding -> Nin + N  (weight pinned to zero)
    idx = jnp.where(is_in, neurons, jnp.where(neurons >= 0, Nin + neurons, Nin + N))
    own = (is_in == False) & (neurons == k)  # noqa: E712
    frozen_times = jnp.where((neurons >= 0) & ~own, times, jnp.inf)

    Kord = jnp.sum(neurons >= 0)
    pseudo_rates = neuron.construct_ratefn(xs[Kord])(input=-xs[Kord][1])
    return (frozen_times, idx.astype(int)), pseudo_rates, out


def make_frozen_loss(
    neuron: QIFNeuron,
    trial: dict,
    queue: tuple,
    pseudo_rates: Array,
    k: int,
    mask: Array,
    eps: float | Array,
) -> Callable:
    """
    L as a function of K's incoming weights, surround held fixed.
    """
    cfg, denom = trial["cfg"], trial["denom"]
    single_cfg = dict(cfg)
    single_cfg["Ntar"] = 1
    t_target_k = trial["t_targets"][k]
    Ntargets = len(t_target_k)

    def loss_fn(w: Array, x0: Array) -> Array:
        # Same parameterisation as the recurrent loss, so masked entries carry
        # the same eps factor in both gradients.
        weights_net = w_to_weights_net(apply_eps(w, mask, eps))
        weights_eff = jnp.concatenate(
            [trial["weights_in"][k], weights_net[k], jnp.zeros(1)]
        )[jnp.newaxis, :]
        out = neuron.event(
            x0[:, k][:, jnp.newaxis], jnp.zeros((1, 1)), weights_eff, queue, single_cfg
        )
        times, is_in, neurons, xs = out
        Kord = jnp.sum(neurons >= 0)
        Kout = jnp.sum((neurons == 0) & (is_in == False))  # noqa: E712
        t_out_ord = times[jnp.argsort(is_in)[:Ntargets]]
        inp = neuron.linear(pseudo_rates, weights_net[k])
        t_out_pseudo = neuron.t_pseudo(
            xs[Kord][:, 0], inp, jnp.arange(Ntargets) + 1 - Kout, cfg
        )
        t_out = jnp.where(jnp.arange(Ntargets) < Kout, t_out_ord, t_out_pseudo)
        return lossfn_shared([t_out], [t_target_k], denom)

    return loss_fn


# %%
############################
### Gradient decomposition
############################


def decompose(
    neuron: QIFNeuron,
    trial: dict,
    w: Array,
    x0: Array,
    eps: float,
    k: int,
    mask: Array,
) -> dict:
    """
    Split dL/dW[k, :] into local and non-local parts at these weights.
    """
    full = make_full_loss(neuron, trial, mask)
    g_exact = grad(full, argnums=0)(w, x0, jnp.array(eps))[k]
    loss_val = float(full(w, x0, jnp.array(eps)))

    queue, rates, out = record_surround(neuron, trial, w, x0, mask, eps, k)
    frozen = make_frozen_loss(neuron, trial, queue, rates, k, mask, jnp.array(eps))
    g_local = grad(frozen, argnums=0)(w, x0)[k]

    g_nl = g_exact - g_local
    n_ex = float(jnp.linalg.norm(g_exact))
    _, is_in, neurons, _ = out
    na, nb = n_ex, float(jnp.linalg.norm(g_local))
    return {
        "loss": loss_val,
        "norm_exact": n_ex,
        "norm_local": nb,
        "norm_nonlocal": float(jnp.linalg.norm(g_nl)),
        "rho_pct": 100.0 * float(jnp.linalg.norm(g_nl)) / max(n_ex, 1e-30),
        "cos": float(jnp.dot(g_exact, g_local) / max(na * nb, 1e-30)),
        # spike counts: a change means the comparison moved to another branch
        "Kout": int(jnp.sum((neurons == k) & (is_in == False))),  # noqa: E712
        "Ktot": int(jnp.sum(neurons >= 0)),
    }


# %%
############################
### Training and measurements
############################


def stage_train(args: argparse.Namespace) -> None:
    """
    Train the intact network and checkpoint along the way.

    Row K of the training gradient is exactly g_exact at eps = 1, so it is
    cached here rather than recomputed later.
    """
    cfg = dict(BASE_CONFIG)
    cfg["seed"] = args.seed
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = make_trial(cfg, args.ntar, args.w_scale)
    ones = jnp.ones_like(trial["w0"])
    loss_full = make_full_loss(neuron, trial, ones)

    schedule = optax.exponential_decay(args.lr, max(args.trials // 5, 1), 1 / jnp.e)
    optim = optax.adabelief(schedule)

    @jax.jit
    def step(p: tuple, opt_state: optax.OptState) -> tuple:
        loss, grads = jax.value_and_grad(
            lambda q: loss_full(q[0], q[1], jnp.array(1.0))
        )(p)
        updates, opt_state = optim.update(grads, opt_state)
        return optax.apply_updates(p, updates), opt_state, loss, grads[0][args.k]

    ### Initialize
    p = (trial["w0"], trial["x0"])
    opt_state = optim.init(p)
    checkpoints = set(args.checkpoints)
    snaps, trace = [], []

    ### Training
    t0 = time.time()
    for t in range(args.trials + 1):
        if t in checkpoints:
            snaps.append((t, p[0], p[1], float(loss_full(p[0], p[1], jnp.array(1.0)))))
        if t == args.trials:
            break
        p, opt_state, loss, g_k = step(p, opt_state)
        trace.append((t + 1, float(loss), float(jnp.linalg.norm(g_k))))

    ### Save checkpoints
    with open(output_file(SNAPS), "wb") as file:
        pickle.dump({"snaps": snaps, "trace": trace, "args": vars(args)}, file)
    print(f"trained {args.trials} steps in {time.time() - t0:.1f}s")
    for step_index, _, _, loss in snaps:
        print(f"  step {step_index:5d}   loss {loss:.5e}")


def stage_decompose(args: argparse.Namespace) -> None:
    """
    Sweep eps at each checkpoint. Run in chunks; results accumulate.
    """
    cfg = dict(BASE_CONFIG)
    cfg["seed"] = args.seed
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = make_trial(cfg, args.ntar, args.w_scale)
    with open(SNAPS, "rb") as file:
        snaps = pickle.load(file)["snaps"]
    mask = (
        sink_mask(cfg["N"], args.k)
        if args.arch == "sink"
        else acyclic_mask(cfg["N"], args.k, range(args.k + 1, args.k + 1 + args.n_down))
    )

    out = {}
    if DECOMP.exists():
        with open(DECOMP) as file:
            out = json.load(file)
    for eps in args.eps:
        t0 = time.time()
        rows = []
        for s, w, x0, tr_loss in snaps:
            m = decompose(neuron, trial, w, x0, eps, args.k, mask)
            m["step"], m["train_loss"] = s, tr_loss
            rows.append(m)
        out[f"{args.arch}:{eps}"] = rows
        print(
            f"  eps {eps:<5} {time.time() - t0:5.1f}s   "
            + "  ".join(f"{r['step']}:{r['rho_pct']:.2f}%" for r in rows)
        )
    with open(output_file(DECOMP), "w") as file:
        json.dump(out, file)


def stage_partitions(args: argparse.Namespace) -> None:
    """
    Vary which neurons sit downstream of K, holding |D| and weights fixed.
    """
    cfg = dict(BASE_CONFIG)
    cfg["seed"] = args.seed
    N, k = cfg["N"], args.k
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = make_trial(cfg, 2, args.w_scale)  # Ntar=2 so a loss neuron can be in D
    with open(SNAPS, "rb") as file:
        snaps = {s: (w, x) for s, w, x, _ in pickle.load(file)["snaps"]}
    last = max(snaps)

    rng = np.random.default_rng(args.part_seed)
    pool = [i for i in range(N) if i != k and i != 1]
    out = []
    if PARTS.exists():
        with open(PARTS) as file:
            out = json.load(file)
    for label, (w, x0) in [("init", snaps[0]), (f"trained({last})", snaps[last])]:
        for _ in range(args.n_partitions):
            # neuron 1 is the extra loss neuron and is always downstream, so a
            # non-returning path always exists; the rest of D is random.
            D = sorted(
                [1]
                + [
                    int(v)
                    for v in rng.choice(pool, size=args.n_down - 1, replace=False)
                ]
            )
            m = decompose(neuron, trial, w, x0, 0.0, k, acyclic_mask(N, k, D))
            m["weights"], m["D"] = label, D
            out.append(m)
        v = [r["rho_pct"] for r in out if r["weights"] == label]
        print(
            f"  {label:<14} min {min(v):6.3f}%   median "
            f"{float(np.median(v)):6.3f}%   max {max(v):7.3f}%"
        )
    with open(output_file(PARTS), "w") as file:
        json.dump(out, file)


# %%
############################
### Plotting
############################


def stage_figure(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ### Load measurements
    with open(DECOMP) as file:
        dec = json.load(file)
    with open(PARTS) as file:
        part = json.load(file)

    def curves(arch: str) -> dict:
        """{eps: rows} for one architecture, sorted by eps."""
        return {
            float(key.split(":")[1]): dec[key]
            for key in dec
            if key.startswith(arch + ":")
        }

    STYLE = {"sink": dict(ls="-", marker="o"), "acyclic": dict(ls="--", marker="s")}
    COL = {"sink": "#2b6cb0", "acyclic": "#c53030"}
    arches = [a for a in ("sink", "acyclic") if curves(a)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    # --- A: interpolation at initialisation --------------------------------
    ax = axes[0]
    for arch in arches:
        c = curves(arch)
        nz = sorted(e for e in c if e > 0)
        ax.plot(
            nz,
            [c[e][0]["rho_pct"] for e in nz],
            color=COL[arch],
            lw=1.6,
            ms=5,
            label=arch,
            **STYLE[arch],
        )
    ax.plot([0], [0], "o", color="k", ms=6)
    ax.annotate(
        r"$\epsilon=0$:  $\rho=0$",
        xy=(0, 0),
        xytext=(0.12, 0.35),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.set_xlabel(r"$\epsilon$   (0 = reduced, 1 = intact)")
    ax.set_ylabel(r"non-local share  $\rho$  (%)")
    ax.set_title("A   at initialisation", loc="left", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3, lw=0.5)

    # --- B: along training -------------------------------------------------
    ax = axes[1]
    for arch in arches:
        c = curves(arch)
        for e, alpha in [(0.25, 0.45), (0.5, 1.0)]:
            if e in c:
                ax.plot(
                    [r["step"] for r in c[e]],
                    [r["rho_pct"] for r in c[e]],
                    color=COL[arch],
                    alpha=alpha,
                    lw=1.6,
                    ms=4,
                    label=rf"{arch}, $\epsilon$={e:g}",
                    **STYLE[arch],
                )
    # at eps = 1 the mask is inert, so both conditions are the intact network
    intact = curves(arches[0]).get(1.0)
    if intact:
        ax.plot(
            [r["step"] for r in intact],
            [r["rho_pct"] for r in intact],
            color="k",
            lw=1.8,
            ms=4,
            marker="^",
            label=r"intact ($\epsilon$=1)",
        )
    ax.axhline(100, color="#666", ls=":", lw=1)
    ax.text(
        0.03,
        0.93,
        "local gradient fully wrong",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#666",
    )
    ax.set_yscale("log")
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1%", "10%", "100%"])
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\rho$  (%)")
    ax.set_title("B   along training", loc="left", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=1)
    ax.grid(alpha=0.3, lw=0.5)

    # --- C: partition dependence (acyclic only: D exists only there) -------
    ax = axes[2]
    labels = sorted({r["weights"] for r in part}, key=lambda s: s != "init")
    for j, lab in enumerate(labels):
        v = np.array(sorted(r["rho_pct"] for r in part if r["weights"] == lab))
        col = COL["acyclic"] if j else "#2b6cb0"
        x = np.full_like(v, j) + np.linspace(-0.14, 0.14, len(v))
        ax.scatter(x, v, s=22, color=col, zorder=3, edgecolor="w", lw=0.4)
        ax.plot([j - 0.22, j + 0.22], [np.median(v)] * 2, color=col, lw=2.2)
        ax.text(j + 0.28, v.max(), f"{v.max():.0f}%", fontsize=8, color=col)
        ax.text(j + 0.28, max(v.min(), 1e-3), f"{v.min():.2f}%", fontsize=8, color=col)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5 + 0.65)
    ax.set_ylabel(r"$\rho$  (%)")
    ax.set_title(
        r"C   acyclic, $\epsilon=0$: partitions at fixed $|\mathcal{D}|$",
        loc="left",
        fontsize=10,
    )
    ax.grid(alpha=0.3, lw=0.5, axis="y")

    fig.tight_layout()
    args.out = output_file(args.out)
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    print(f"figure written to {args.out}")


# %%
############################
### Masked training
############################


def train_masked(
    neuron: QIFNeuron,
    trial: dict,
    mask: Array,
    trials: int,
    lr: float,
    k: int,
    checkpoints: list[int] | None = None,
) -> tuple:
    """
    Train with the mask in force throughout (eps = 0), from the same init.

    This is a different optimisation problem from the intact one: the blocked
    synapses are absent for the whole trajectory, so the optimiser explores a
    restricted weight space rather than being perturbed at the end.
    """
    loss_full = make_full_loss(neuron, trial, mask)
    optim = optax.adabelief(optax.exponential_decay(lr, max(trials // 5, 1), 1 / jnp.e))

    @jax.jit
    def step(p: tuple, opt_state: optax.OptState) -> tuple:
        loss, g = jax.value_and_grad(lambda q: loss_full(q[0], q[1], jnp.array(0.0)))(p)
        updates, opt_state = optim.update(g, opt_state)
        return optax.apply_updates(p, updates), opt_state, loss

    p = (trial["w0"], trial["x0"])
    opt_state = optim.init(p)
    ckpt = set(checkpoints or [])
    snaps = []
    for t in range(trials + 1):
        if t in ckpt:
            snaps.append((t, p[0], p[1], float(loss_full(p[0], p[1], jnp.array(0.0)))))
        if t == trials:
            break
        p, opt_state, loss = step(p, opt_state)
    if checkpoints is not None:
        return p[0], p[1], float(loss), snaps
    return p[0], p[1], float(loss)


def stage_converge(args: argparse.Namespace) -> None:
    """
    Does the gradient become more local as the masked network trains?

    The mask is in force for the whole trajectory, and rho is measured at
    eps = 0 with that same mask at every checkpoint. For the sink condition rho
    is pinned at 0 by construction, so it serves as the control; for the
    acyclic condition rho is the non-returning family and is free to move.
    """
    cfg = dict(BASE_CONFIG)
    cfg["seed"] = args.seed
    N, k = cfg["N"], args.k
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = make_trial(cfg, args.ntar, args.w_scale)
    D = list(range(k + 1, k + 1 + args.n_down))
    masks = {"sink": sink_mask(N, k), "acyclic": acyclic_mask(N, k, D)}

    out = {}
    for name in args.conds:
        mask = masks[name]
        *_, snaps = train_masked(
            neuron, trial, mask, args.trials, args.lr, k, checkpoints=args.checkpoints
        )
        rows = []
        for t, w, x0, tr_loss in snaps:
            m = decompose(neuron, trial, w, x0, 0.0, k, mask)
            m["step"], m["train_loss"] = t, tr_loss
            rows.append(m)
        out[name] = rows
        print(f"\n  {name}   (mask in force during training, rho at eps=0)")
        print(
            f"  {'step':>6} {'train L':>11} | {'|g_exact|':>10} "
            f"{'|g_nonloc|':>11} {'rho':>8} {'cos':>10} {'Kout':>5}"
        )
        print("  " + "-" * 68)
        for m in rows:
            print(
                f"  {m['step']:6d} {m['train_loss']:11.4e} | "
                f"{m['norm_exact']:10.3e} {m['norm_nonlocal']:11.3e} "
                f"{m['rho_pct']:7.3f}% {m['cos']:10.6f} {m['Kout']:5d}"
            )
    with open(output_file(DATA_DIR / "converge.json"), "w") as file:
        json.dump(out, file)


# %%
############################
### Spike-train comparison
############################


def spike_trains(
    neuron: QIFNeuron,
    trial: dict,
    w: Array,
    x0: Array,
    mask: Array,
    eps: float,
) -> list[np.ndarray]:
    """
    Ordinary spike times of every neuron, as a list of sorted arrays.
    """
    cfg = trial["cfg"]
    out = neuron.event(
        x0,
        w_to_weights_net(apply_eps(w, mask, eps)),
        trial["weights_in"],
        trial["spikes_in"],
        cfg,
    )
    times, is_in, neurons, _ = out
    t = np.asarray(times)
    net = (~np.asarray(is_in)) & (np.asarray(neurons) >= 0)
    ids = np.asarray(neurons)
    return [np.sort(t[net & (ids == i)]) for i in range(cfg["N"])]


def compare_trains(a: list[np.ndarray], b: list[np.ndarray]) -> list[dict]:
    """
    Per-neuron comparison: spike count difference and worst timing shift.
    """
    rows = []
    for i, (x, y) in enumerate(zip(a, b)):
        if len(x) == len(y):
            dmax = float(np.max(np.abs(x - y))) if len(x) else 0.0
        else:
            dmax = float("nan")
        rows.append({"neuron": i, "n_a": len(x), "n_b": len(y), "dmax": dmax})
    return rows


def stage_structural(args: argparse.Namespace) -> None:
    """
    Fixed weights, acyclic mask: U must be blind to row K, D must not.

    This is the structural claim behind the acyclic condition, checked directly
    rather than through the gradient. Nothing is retrained, so the dynamics are
    identical apart from the perturbation itself.
    """
    cfg = dict(BASE_CONFIG)
    cfg["seed"] = args.seed
    N, k = cfg["N"], args.k
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = make_trial(cfg, args.ntar, args.w_scale)
    D = list(range(k + 1, k + 1 + args.n_down))
    U = [i for i in range(N) if i != k and i not in D]
    mask = acyclic_mask(N, k, D)

    w, x0 = trial["w0"], trial["x0"]
    base = spike_trains(neuron, trial, w, x0, mask, 0.0)

    rng = np.random.default_rng(args.part_seed)
    w_pert = np.asarray(w).copy()
    w_pert[k] += 0.25 * rng.standard_normal(w_pert.shape[1])  # row K only
    pert = spike_trains(neuron, trial, jnp.asarray(w_pert), x0, mask, 0.0)

    rows = compare_trains(base, pert)
    print("  perturbing row K under the acyclic mask (fixed weights)\n")
    print(f"  {'neuron':>6} {'set':>5} {'n base':>7} {'n pert':>7} {'max |dt|':>11}")
    print("  " + "-" * 42)
    for r in rows:
        s = "K" if r["neuron"] == k else ("D" if r["neuron"] in D else "U")
        print(
            f"  {r['neuron']:>6} {s:>5} {r['n_a']:>7} {r['n_b']:>7} {r['dmax']:>11.3e}"
        )
    bad = [
        r
        for r in rows
        if r["neuron"] in U and (r["n_a"] != r["n_b"] or r["dmax"] > 1e-10)
    ]
    print(f"\n  U unchanged: {'YES' if not bad else 'NO  <-- structural bug'}")


def stage_retrain(args: argparse.Namespace) -> None:
    """
    Train each condition to convergence and compare the spike trains.
    """
    cfg = dict(BASE_CONFIG)
    cfg["seed"] = args.seed
    N, k = cfg["N"], args.k
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = make_trial(cfg, args.ntar, args.w_scale)
    D = list(range(k + 1, k + 1 + args.n_down))
    U = [i for i in range(N) if i != k and i not in D]
    conds = {
        "intact": jnp.ones_like(trial["w0"]),
        "sink": sink_mask(N, k),
        "acyclic": acyclic_mask(N, k, D),
    }

    trained, out = {}, {}
    for name, mask in conds.items():
        t0 = time.time()
        w, x0, loss = train_masked(neuron, trial, mask, args.trials, args.lr, k)
        trained[name] = (w, x0, mask)
        print(f"  {name:<8} final loss {loss:.5e}   ({time.time() - t0:.0f}s)")

    ref = spike_trains(neuron, trial, *trained["intact"][:2], trained["intact"][2], 0.0)
    print(f"\n  D = {D}     U = {U}\n")
    for name in ("sink", "acyclic"):
        w, x0, mask = trained[name]
        rows = compare_trains(ref, spike_trains(neuron, trial, w, x0, mask, 0.0))
        out[name] = rows
        print(f"  {name} vs intact")
        print(
            f"  {'neuron':>6} {'set':>5} {'n intact':>9} {'n cond':>7} {'max |dt|':>11}"
        )
        print("  " + "-" * 44)
        for r in rows:
            s = "K" if r["neuron"] == k else ("D" if r["neuron"] in D else "U")
            print(
                f"  {r['neuron']:>6} {s:>5} {r['n_a']:>9} {r['n_b']:>7} "
                f"{r['dmax']:>11.3e}"
            )
        same_U = all(
            r["n_a"] == r["n_b"] and not (r["dmax"] > 1e-10)
            for r in rows
            if r["neuron"] in U
        )
        same_D = all(
            r["n_a"] == r["n_b"] and not (r["dmax"] > 1e-10)
            for r in rows
            if r["neuron"] in D
        )
        print(
            f"    U identical to intact: {same_U}     D identical to intact: {same_D}\n"
        )
    with open(output_file(DATA_DIR / "retrain.json"), "w") as file:
        json.dump(out, file)


# %%
############################
### Command line
############################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "train",
            "decompose",
            "partitions",
            "figure",
            "retrain",
            "structural",
            "converge",
        ],
    )
    parser.add_argument("--k", type=int, default=0, help="reference neuron")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ntar", type=int, default=1)
    parser.add_argument("--w-scale", type=float, default=3.0)
    parser.add_argument("--arch", default="sink", choices=["sink", "acyclic"])
    parser.add_argument("--n-down", type=int, default=5, help="|D|")
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--checkpoints", type=int, nargs="+", default=[0, 10, 25, 50, 100, 200, 400]
    )
    parser.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--n-partitions", type=int, default=6)
    parser.add_argument("--part-seed", type=int, default=0)
    parser.add_argument(
        "--out", type=svg_path, default=str(FIGS_DIR / "main_figure.svg")
    )
    parser.add_argument(
        "--conds", nargs="+", default=["sink", "acyclic"], choices=["sink", "acyclic"]
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    if args.stage == "figure":
        args.out = output_file(args.out)
    {
        "train": stage_train,
        "decompose": stage_decompose,
        "partitions": stage_partitions,
        "figure": stage_figure,
        "retrain": stage_retrain,
        "structural": stage_structural,
        "converge": stage_converge,
    }[args.stage](args)


if __name__ == "__main__":
    main()
