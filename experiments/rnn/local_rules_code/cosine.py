"""
Per-neuron cosine similarity between each approximate rule's gradient and the
exact gradient, along the rule's own training runs from train.py.

At the parameters saved every --every trials (every sigma and seed), the
rule's gradient g and the exact gradient G = dL/dw (autodiff) are evaluated
with respect to the recurrent weights, at the same parameters and on the
inputs without jitter. For every neuron i (row i of w: its input weights),

    cos_i = sum_j g_ij G_ij / ( |g_i.| |G_i.| ),   j over the presynaptic neurons.

If the rule gives row i no update at that point (|g_i.| = 0: local rule on
off-target rows, or a neuron that did not spike), cos_i is undefined and
stored as NaN.

Usage (from experiments/rnn/local_rules_code/, after train.py):

    python cosine.py    # -> output/data/cosine_results.npz, output/data/cosine_per_neuron.csv
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
### Gradient comparison
############################


def row_products(g: Array, G: Array) -> Array:
    """
    Per-row norms of g and G and their inner products, shape (3, N).
    """
    return jnp.stack(
        [jnp.linalg.norm(g, axis=1), jnp.linalg.norm(G, axis=1), jnp.sum(g * G, axis=1)]
    )


def run(args: argparse.Namespace) -> dict:
    """
    Compares each rule with the exact gradient, neuron by neuron, along its
    recorded trajectory.
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
    out = {"sigmas": sig, "trials": trials[::step], "n_targets": ntar}
    t0 = time.time()
    for r, rule_fn in rules.items():
        if f"{r}_theta" not in d:
            continue

        def rows(theta: Array, rule_fn=rule_fn) -> Array:
            w = theta[:nw].reshape(N, N - 1)
            x0 = theta[nw:].reshape(2, N)
            grad_exact = jax.grad(lambda ww: full(ww, x0, one))(w)
            grad_rule = rule_fn(w, x0, trial["spikes_in"])[1][0]
            return row_products(grad_rule, grad_exact)

        fn = jax.jit(jax.vmap(rows))
        th = d[f"{r}_theta"][:, :, ::step]  # (S, R, B', P)
        v = np.zeros(th.shape[:3] + (3, N))
        for s in range(len(sig)):
            for q in (
                range(1) if sig[s] == 0 else range(th.shape[1])
            ):  # sigma = 0: one run
                v[s, q] = np.asarray(fn(jnp.asarray(th[s, q])))
            if sig[s] == 0:
                v[s, 1:] = v[s, 0]

        # Cosine per row; undefined where the rule (or the exact gradient) is zero
        norm_rule, norm_exact, dot = v[..., 0, :], v[..., 1, :], v[..., 2, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(
                (norm_rule > 0) & (norm_exact > 0), dot / (norm_rule * norm_exact), np.nan
            )
        out[f"{r}_cos"] = cos
        out[f"{r}_norm_rule"] = norm_rule
        out[f"{r}_norm_exact"] = norm_exact

        print(
            f"  {r}: done ({time.time() - t0:.0f}s); median cosine per neuron "
            "(over saved trials and seeds; '-' = never updated)",
            flush=True,
        )
        for s_i, s in enumerate(sig):
            c = cos[s_i, : 1 if s == 0 else None].reshape(-1, N)
            med = [
                np.nanmedian(c[:, i]) if np.any(np.isfinite(c[:, i])) else np.nan
                for i in range(N)
            ]
            print(
                f"    sigma {s:<5g}"
                + "".join(f"{m:7.2f}" if np.isfinite(m) else "      -" for m in med),
                flush=True,
            )

    ### Save comparisons
    np.savez(args.out, **out)
    print(f"  -> {args.out}")
    if args.csv:
        write_csv(args.csv, out, list(rules), ntar)
        print(f"  -> {args.csv}")

    return out


def write_csv(path: str, out: dict, rules: list[str], ntar: int) -> None:
    """
    Writes every per-neuron value: rule, sigma, seed, trial, neuron, cosine, norms.
    """
    sig = out["sigmas"]
    with open(path, "w") as fh:
        fh.write("rule,sigma,seed,trial,neuron,is_target,cosine,norm_rule,norm_exact\n")
        for r in rules:
            if f"{r}_cos" not in out:
                continue
            c, ng, nG = out[f"{r}_cos"], out[f"{r}_norm_rule"], out[f"{r}_norm_exact"]
            for s in range(len(sig)):
                for q in range(1 if sig[s] == 0 else c.shape[1]):
                    for b, t in enumerate(out["trials"]):
                        for i in range(c.shape[-1]):
                            cv = "" if np.isnan(c[s, q, b, i]) else f"{c[s, q, b, i]:.6g}"
                            fh.write(
                                f"{r},{sig[s]:g},{q},{int(t)},{i},{int(i < ntar)},"
                                f"{cv},{ng[s, q, b, i]:.6g},{nG[s, q, b, i]:.6g}\n"
                            )


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
    parser.add_argument(
        "--csv",
        default=str(DATA_DIR / "cosine_per_neuron.csv"),
        help="every per-neuron value as CSV ('' to skip)",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    args.out = output_file(args.out)
    if args.csv:
        args.csv = output_file(args.csv)
    run(args)


if __name__ == "__main__":
    main()
