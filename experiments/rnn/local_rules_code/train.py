"""
Train the published RNN task with three update rules under input spike-time
jitter, t_in -> max(t_in + sigma * xi, 0), xi ~ N(0, 1), fresh every trial.

Rules (same zero init, optimiser, schedule and noise draws):
  exact    autodiff through the full event-based simulation
  local    one-pass eligibility traces, target rows only (A = C = 0)
  fitted   local + A + C on the target rows (Eq. UpdateAC) and the
           off-target rows (Eq. UpdateOff), coefficients from fit.py
All rules: AdaBelief, exponential_decay(1e-2, 2000, 1/e), 10,000 trials.

All (sigma, seed) jobs of one rule are trained in parallel (vmap). sigma = 0
is deterministic and runs once. Every --record trials the clean-input loss and
the parameters are saved; at the end, the loss on --n-eval held-out jittered
inputs. Running a subset of rules adds them to an existing results file with
the same sigmas and seeds.

Usage (from experiments/rnn/local_rules_code/, after fit.py):

    python train.py                       # -> output/data/train_results.npz
    python train.py --rules fitted        # rerun one rule only
"""

import argparse
import json
import time
from collections.abc import Callable

import experiment as ex  # sets JAX_PLATFORMS / JAX_ENABLE_X64 on import
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax, random
from jaxtyping import Array
from spikegd.qif import QIFNeuron

from fastgrad import make_fast_grad
from output_paths import DATA_DIR, ensure_output_dirs, output_file

# %%
############################
### Configuration
############################


OUT = str(DATA_DIR / "train_results.npz")
RULES = ["exact", "local", "fitted"]
# Noise stream of a sigma = its position in this grid (kept fixed so that runs
# with different sigma lists share the same draws for the same sigma and seed).
SIGMA_GRID = [0.0, 0.025, 0.05, 0.1, 0.2]


# %%
############################
### Input spikes
############################


def jitter(spikes_in: tuple, key: Array, sigma: float | Array) -> tuple:
    """
    Gaussian jitter on every finite input time, clipped at 0 and re-sorted.
    """
    times, neurons = spikes_in
    t = jnp.maximum(times + sigma * random.normal(key, times.shape), 0.0)
    t = jnp.where(jnp.isfinite(times), t, jnp.inf)
    order = jnp.argsort(t)
    return t[order], neurons[order]


# %%
############################
### Initialization
############################


def build(args: argparse.Namespace) -> tuple:
    """
    Creates the trial, loss, gradient rules and optimizer.
    """
    ### Unpack arguments
    cfg = dict(ex.BASE_CONFIG)
    ntar = cfg["Ntar"]
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = ex.make_trial(cfg, ntar, 0.0)
    ones = jnp.ones_like(trial["w0"])
    one = jnp.array(1.0)

    ### Loss function
    def loss_on(p: tuple, spikes: tuple) -> Array:
        return ex.make_full_loss(neuron, dict(trial, spikes_in=spikes), ones)(p[0], p[1], one)

    ### Gradient rules
    grads = {"exact": lambda p, s: jax.grad(loss_on)(p, s)}
    local = make_fast_grad(neuron, trial, ntar, "trace")
    grads["local"] = lambda p, s: local(p[0], p[1], s)[1]
    if "fitted" in args.rules:
        with open(args.fit) as file:
            f = json.load(file)
        fitted = make_fast_grad(neuron, trial, ntar, "trace",
                                spike_mix=np.array(f["spike_mix"]), cross=np.array(f["cross"]),
                                offtarget=np.array(f["offtarget"]))
        grads["fitted"] = lambda p, s: fitted(p[0], p[1], s)[1]
    ### Optimizer
    optim = optax.adabelief(optax.exponential_decay(args.lr, args.decay_steps, 1 / jnp.e))
    return trial, loss_on, grads, optim


# %%
############################
### Training
############################


def make_train(
    trial: dict,
    loss_on: Callable,
    g: Callable,
    optim: optax.GradientTransformation,
    args: argparse.Namespace,
) -> Callable:
    """
    Train one (sigma, seed); returns recorded parameters, clean losses, end params.
    """
    ### Unpack arguments
    n_blocks: int = args.trials // args.record

    def train(key: Array, sigma: float | Array) -> tuple:
        ### Initialize
        p0 = (trial["w0"], trial["x0"])

        # Update step
        def one_step(carry: tuple, key: Array) -> tuple:
            p, opt_state = carry
            grad = g(p, jitter(trial["spikes_in"], key, sigma))
            updates, opt_state = optim.update(grad, opt_state)
            return (optax.apply_updates(p, updates), opt_state), None

        # Record parameters and clean-input loss after each block
        def block(carry: tuple, keys: Array) -> tuple:
            carry, _ = lax.scan(one_step, carry, keys)
            p = carry[0]
            return carry, (jnp.concatenate([jnp.ravel(p[0]), jnp.ravel(p[1])]),
                           loss_on(p, trial["spikes_in"]))

        ### Training
        keys = random.split(key, n_blocks * args.record).reshape(n_blocks, args.record, 2)
        (p_end, _), (thetas, clean) = lax.scan(block, (p0, optim.init(p0)), keys)
        theta0 = jnp.concatenate([jnp.ravel(p0[0]), jnp.ravel(p0[1])])
        return (jnp.concatenate([theta0[None], thetas]),
                jnp.concatenate([loss_on(p0, trial["spikes_in"])[None], clean]), p_end)

    return train


def run(args: argparse.Namespace) -> dict:
    """
    Trains each rule under input jitter and saves parameters and loss metrics.
    """
    ### Set up the simulation
    trial, loss_on, grads, optim = build(args)
    sig, R = args.sigmas, args.n_seeds
    ### Keep results from other rules when updating an existing file
    out = {}
    try:  # keep other rules of an existing file with the same sigmas and seeds
        with np.load(args.out) as old:
            if (np.array_equal(old["sigmas"], sig)
                    and next(old[k].shape[1] for k in old.files if k.endswith("_loss")) == R):
                out = {k: old[k] for k in old.files if k.split("_")[0] not in args.rules}
    except (FileNotFoundError, StopIteration):
        pass

    ### Noise streams and held-out evaluation
    jobs = [(si, r) for si, s in enumerate(sig) for r in range(1 if s == 0 else R)]
    idx = [SIGMA_GRID.index(s) for s in sig]
    keys = jnp.stack([random.fold_in(random.PRNGKey(1000 + r), idx[si]) for si, r in jobs])
    ekeys = jnp.stack([random.split(random.fold_in(random.PRNGKey(5000 + r), idx[si]),
                                    args.n_eval) for si, r in jobs])
    sigs = jnp.array([sig[si] for si, _ in jobs])
    heldout = jax.jit(jax.vmap(lambda p, ks, s: jnp.mean(jax.vmap(
        lambda k: loss_on(p, jitter(trial["spikes_in"], k, s)))(ks))))
    ### Training
    t0 = time.time()
    for rule in args.rules:
        train = jax.jit(jax.vmap(make_train(trial, loss_on, grads[rule], optim, args)))
        th, cl, p_end = train(keys, sigs)
        ho, th, cl = np.asarray(heldout(p_end, ekeys, sigs)), np.asarray(th), np.asarray(cl)
        B = cl.shape[-1]
        TH = np.zeros((len(sig), R, B, th.shape[-1]))
        CL = np.zeros((len(sig), R, B))
        HO = np.zeros((len(sig), R))
        for j, (si, r) in enumerate(jobs):
            for rr in (range(R) if sig[si] == 0 else [r]):  # sigma = 0: copy the one run
                TH[si, rr], CL[si, rr], HO[si, rr] = th[j], cl[j], ho[j]
        out[f"{rule}_theta"], out[f"{rule}_loss"], out[f"{rule}_heldout"] = TH, CL, HO
        for si, s in enumerate(sig):
            print(f"  {rule:<7} sigma {s:<5g} final clean loss (median) "
                  f"{np.median(CL[si, :, -1]):.4f}   held-out jittered {np.median(HO[si]):.4f}",
                  flush=True)
        print(f"  {rule} done ({time.time() - t0:.0f}s)", flush=True)
        np.savez(args.out, sigmas=np.array(sig), trials=np.arange(B) * args.record,
                 args=json.dumps(vars(args)), **out)
    print(f"  -> {args.out}")

    return out


# %%
############################
### Command line
############################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rules", nargs="+", default=RULES, choices=RULES)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.05, 0.2],
                    help=f"subset of {SIGMA_GRID}")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--decay-steps", type=int, default=2_000)
    parser.add_argument("--record", type=int, default=50, help="save every N trials")
    parser.add_argument("--n-eval", type=int, default=30, help="held-out jittered inputs")
    parser.add_argument("--fit", default=str(DATA_DIR / "fit_rule.json"))
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    args.out = output_file(args.out)
    run(args)


if __name__ == "__main__":
    main()
