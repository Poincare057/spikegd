"""
Cosine similarity between each approximate rule's update and the exact
gradient, along the rule's own training runs from train.py.

At the parameters saved every --every trials (every sigma and seed), both the
rule's gradient and the exact gradient (autodiff) are evaluated on the inputs
without jitter, and

    cos = <g_rule, dL/dw> / (|g_rule| |dL/dw|)

is computed over the recurrent weights on (0) the target rows, (1) all rows,
(2) the off-target rows (only the fitted rule updates them; 0 for local).

Usage (from experiments/rnn/local_rules_code/, after train.py):

    python cosine.py                      # -> output/data/cosine_results.npz
"""

import argparse
import json
import time
import experiment as ex
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from fastgrad import make_fast_grad
from output_paths import DATA_DIR, ensure_output_dirs, output_file
from spikegd.qif import QIFNeuron

# %%
############################
### Configuration
############################


PARTS = ["target rows", "all rows", "off-target rows"]


# %%
############################
### Gradient comparison
############################


def cosine_similarity(a: Array, b: Array) -> Array:
    """
    Computes cosine similarity, returning zero for a zero update vector.
    """
    return jnp.vdot(a, b) / jnp.maximum(jnp.linalg.norm(a) * jnp.linalg.norm(b), 1e-300)


def run(args: argparse.Namespace) -> dict:
    """
    Compares each rule with the exact gradient along its recorded trajectory.
    """

    ### Unpack recordings and task configuration
    d = np.load(args.data)
    sig, trials = d["sigmas"], d["trials"]
    cfg = dict(ex.BASE_CONFIG)
    N, ntar = cfg["N"], cfg["Ntar"]
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = ex.make_trial(cfg, ntar, 0.0)
    one = jnp.array(1.0)
    full = ex.make_full_loss(neuron, trial, jnp.ones_like(trial["w0"]))

    ### Gradient rules
    with open(args.fit) as file:
        f = json.load(file)
    rules = {
        "local": make_fast_grad(neuron, trial, ntar, "trace"),
        "fitted": make_fast_grad(
            neuron,
            trial,
            ntar,
            "trace",
            spike_mix=np.array(f["spike_mix"]),
            cross=np.array(f["cross"]),
            offtarget=np.array(f["offtarget"]),
        ),
    }
    nw = N * (N - 1)
    step = max(args.every // int(trials[1] - trials[0]), 1)

    ### Evaluate recorded parameters
    out = {"sigmas": sig, "trials": trials[::step], "parts": np.array(PARTS)}
    t0 = time.time()
    for r, rule_fn in rules.items():
        if f"{r}_theta" not in d:
            continue

        def cosines(theta: Array, rule_fn=rule_fn) -> Array:
            w = theta[:nw].reshape(N, N - 1)
            x0 = theta[nw:].reshape(2, N)
            grad_exact = jax.grad(lambda ww: full(ww, x0, one))(w)
            grad_rule = rule_fn(w, x0, trial["spikes_in"])[1][0]
            return jnp.stack(
                [
                    cosine_similarity(grad_rule[:ntar], grad_exact[:ntar]),
                    cosine_similarity(grad_rule, grad_exact),
                    cosine_similarity(grad_rule[ntar:], grad_exact[ntar:]),
                ]
            )

        fn = jax.jit(jax.vmap(cosines))
        th = d[f"{r}_theta"][:, :, ::step]  # (S, R, B', P)
        res = np.zeros(th.shape[:3] + (3,))
        for s in range(len(sig)):
            for q in (
                range(1) if sig[s] == 0 else range(th.shape[1])
            ):  # sigma = 0: one run
                res[s, q] = np.asarray(fn(jnp.asarray(th[s, q])))
            if sig[s] == 0:
                res[s, 1:] = res[s, 0]
        out[r] = res
        print(f"  {r}: done ({time.time() - t0:.0f}s)", flush=True)
        for s_i, s in enumerate(sig):
            m = np.median(res[s_i].reshape(-1, 3), 0)
            print(
                "    sigma {:<5g} median cosine similarity: ".format(s)
                + ", ".join(f"{p} {v:.3f}" for p, v in zip(PARTS, m)),
                flush=True,
            )

    ### Save comparisons
    np.savez(args.out, **out)
    print(f"  -> {args.out}")

    return out


# %%
############################
### Command line
############################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=str(DATA_DIR / "train_results.npz"))
    parser.add_argument("--fit", default=str(DATA_DIR / "fit_rule.json"))
    parser.add_argument(
        "--every", type=int, default=250, help="trials between evaluations"
    )
    parser.add_argument("--out", default=str(DATA_DIR / "cosine_results.npz"))
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    args.out = output_file(args.out)
    run(args)


if __name__ == "__main__":
    main()
