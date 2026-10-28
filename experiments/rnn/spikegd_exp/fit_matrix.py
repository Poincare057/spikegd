"""
Fit the neuron-specific synapse-mixing matrices A^(i) of the working notes:

    dt_{k_i}/dw_ij  ~  sum_j' dt_{k_i}/dw_ij'|local * (delta_j'j + A^(i)_j'j),

so each postsynaptic neuron i's local gradient row (over its presynaptic
indices j') is multiplied by its own (I + A^(i)). A^(i) does not depend on the
spike index, so in terms of the summed gradient matrices (rows = postsynaptic
i, columns = presynaptic j, zero diagonal):

    E_i  ~  G_i (I + A^(i)),   i.e.  R_i = E_i - G_i  ~  G_i A^(i).

This is the intended form ("A_per_neuron" in the output, used for training).
The beta rule is the special case A^(i) = beta_i I. A single matrix shared by
all postsynaptic neurons (the notes' formula read without the index i) is also
fitted, as "A", for comparison.

Both are fitted by weighted least squares over every trial of the published
exact run (trial weight 1 / ||E^n||^2, as for alpha and beta); A^(i) uses
target neuron i's rows only. Entries with j = i are not weights and are
excluded from the fit and from the prediction.

Usage (from experiments/rnn/, after `lowrank.py collect --w-scale 0 --trials
10000 --data lowrank_data_km.npz`):

    python fit_matrix.py
"""

import argparse
import json

import numpy as np


def predict(G, A):
    """G (T, N, N) with zero diagonal -> G (I + A), diagonal set back to 0."""
    P = G + np.einsum("tij,jk->tik", G, A)
    idx = np.arange(G.shape[1])
    P[:, idx, idx] = 0.0
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="lowrank_data_km.npz")
    ap.add_argument("--out", default="A_fit_km.json")
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--ridge", type=float, default=1e-6,
                    help="ridge, relative to the mean diagonal of the normal matrix")
    args = ap.parse_args()

    d = np.load(args.data)
    E, G, ntar = d["exact"], d["local"], int(d["ntar"])
    T, N, _ = E.shape
    tar = list(range(ntar))
    blk = (np.arange(T) // args.block) % 2
    tr, te = blk == 0, blk == 1
    w = 1.0 / np.maximum(np.linalg.norm(E[:, tar], axis=(1, 2)), 1e-30) ** 2

    # samples: one per (trial, target row)
    X = G[tr][:, tar].reshape(-1, N)                     # local row
    Y = (E - G)[tr][:, tar].reshape(-1, N)               # non-local part of the row
    W = np.repeat(w[tr], ntar)
    rows = np.tile(np.array(tar), tr.sum())              # postsynaptic index of each sample
    A = np.zeros((N, N))
    for k in range(N):                                   # column k of A
        m = rows != k                                    # entry (i, k) with i == k is not a weight
        Xm, Ym, Wm = X[m], Y[m, k], W[m]
        M = Xm.T @ (Wm[:, None] * Xm)
        A[:, k] = np.linalg.solve(M + args.ridge * np.trace(M) / N * np.eye(N), Xm.T @ (Wm * Ym))

    def scores(P, m, r):
        x, p = E[m][:, r].reshape(m.sum(), -1), P[m][:, r].reshape(m.sum(), -1)
        nx = np.linalg.norm(x, axis=1)
        return (np.median(np.linalg.norm(x - p, axis=1) / nx),
                np.median(np.sum(x * p, 1) / (nx * np.maximum(np.linalg.norm(p, axis=1), 1e-30))))

    # neuron-specific variant: a separate A^(i) for each target neuron i, fitted
    # on that neuron's rows only (the notes call the parametrisation
    # neuron-specific; one shared A must compromise between the targets)
    A_i = np.zeros((N, N, N))
    for i in tar:
        Xi = G[tr][:, i]                                 # (Ttr, N)
        Yi = (E - G)[tr][:, i]
        Wi = w[tr]
        M = Xi.T @ (Wi[:, None] * Xi)
        sol = np.linalg.solve(M + args.ridge * np.trace(M) / N * np.eye(N), Xi.T @ (Wi[:, None] * Yi))
        sol[:, i] = 0.0                                  # column i is not a weight of row i
        A_i[i] = sol

    def predict_per_neuron(G):
        P = G + np.einsum("tij,ijk->tik", G, A_i)
        idx = np.arange(N)
        P[:, idx, idx] = 0.0
        return P

    models = {"local (A = 0)": G}
    try:   # optional comparison with the constant-beta fit, if it has been made
        beta = np.array(json.load(open("beta_fit_km.json"))["beta"])
        models["constant beta"] = G * (1 + beta)[None, :, None]
    except FileNotFoundError:
        pass
    models.update({"shared A": predict(G, A), "per-neuron A(i)": predict_per_neuron(G)})
    print(f"  fit on {tr.sum()} trials, tested on {te.sum()} (blocks of {args.block})\n")
    print(f"  {'model':<15} {'rows':<9} {'rel err (test)':>15} {'cos (test)':>11}")
    for name, P in models.items():
        for lab, r in (("neuron 0", [0]), ("neuron 1", [1])):
            rel, cos = scores(P, te, r)
            print(f"  {name:<15} {lab:<9} {rel:15.3f} {cos:11.3f}")
    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    for i in tar:
        keep = [j for j in range(N) if j != i]
        M = (np.eye(N) + A_i[i])[np.ix_(keep, keep)]
        print(f"\n  neuron {i}: I + A^({i}) over its 9 inputs: diagonal {np.diag(M)}")
        print(f"            singular values {np.linalg.svd(M, compute_uv=False)}")
    json.dump({"A": A.tolist(), "A_per_neuron": A_i.tolist(),
               "fitted_on": f"{args.data}: weighted LS; 'A' shared over targets, "
                            f"'A_per_neuron'[i] fitted on target i's rows only"},
              open(args.out, "w"), indent=1)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
