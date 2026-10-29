"""
Fit the coefficients of the A + C rule (with off-target rows) on one exact run.

    collect   replay the published noise-free exact-gradient training run and,
              every --stride trials, record from one forward trace pass
                DL  dL/dt of every target output (ordinary spikes, then pseudospikes)
                EL  eligibility e_i.(t_{k_i}) of every target output
                EH  eligibility e_l.(t_{k_l}) of the first --n-hidden spikes of
                    every off-target neuron l
                G   the exact gradient on all rows of w (autodiff)
              -> output/data/fit_data.npz

    fit       weighted least squares, trial weight q = 1/||G (target rows)||^2:
              targets i  (Eq. FitAC)  A^(i,k) and C^(m,i) jointly, ridges r_A, r_C
              off-target l (Eq. FitOff) C^(m,l), ridge r_C_off
              Ridges are relative to the mean diagonal of their block of the
              normal matrix. Recordings alternate in blocks of --block between
              the fit set and a held-out set, on which the relative error and
              cosine similarity of each row are reported.
              -> output/data/fit_rule.json

Usage (from experiments/rnn/local_rules_code/):

    python fit.py collect
    python fit.py fit
"""

import argparse
import json
import time

import experiment as ex  # sets JAX_PLATFORMS / JAX_ENABLE_X64 on import
import jax
import jax.numpy as jnp
import numpy as np
import optax
from spikegd.qif import QIFNeuron

from fastgrad import make_fast_grad
from output_paths import DATA_DIR, ensure_output_dirs, output_file

DATA = DATA_DIR / "fit_data.npz"
FIT = str(DATA_DIR / "fit_rule.json")


def setup():
    cfg = dict(ex.BASE_CONFIG)
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = ex.make_trial(cfg, cfg["Ntar"], 0.0)       # published: zero init
    return cfg, neuron, trial


def collect(args):
    cfg, neuron, trial = setup()
    ntar = cfg["Ntar"]
    one = jnp.array(1.0)
    full = ex.make_full_loss(neuron, trial, jnp.ones_like(trial["w0"]))
    pieces = jax.jit(make_fast_grad(neuron, trial, ntar, "trace",
                                    n_hidden_spikes=args.n_hidden, return_pieces=True))
    optim = optax.adabelief(optax.exponential_decay(1e-2, 2000, 1 / jnp.e))

    @jax.jit
    def step(p, s):
        loss, g = jax.value_and_grad(lambda q: full(q[0], q[1], one))(p)
        u, s = optim.update(g, s)
        return optax.apply_updates(p, u), s, loss, g[0]

    p = (trial["w0"], trial["x0"]); s = optim.init(p)
    rec = {k: [] for k in ("DL", "EL", "EH", "G", "trial")}
    t0 = time.time()
    for t in range(args.trials):
        p_before = p
        p, s, loss, g = step(p, s)
        if t % args.stride == 0:
            _, _, pc = pieces(p_before[0], p_before[1], trial["spikes_in"])
            for k in ("DL", "EL", "EH"):
                rec[k].append(np.asarray(pc[k]))
            rec["G"].append(np.asarray(g)); rec["trial"].append(t)
        if t % 1000 == 0:
            print(f"  trial {t:6d}  loss {float(loss):.4e}  ({time.time() - t0:.0f}s)", flush=True)
    np.savez(DATA, **{k: np.array(v) for k, v in rec.items()}, stride=args.stride)
    print(f"  {len(rec['trial'])} recordings -> {DATA}")


def wls(X, Y, q, pen):
    """argmin sum_n q_n |Y_n - X_n theta|^2 + sum pen * theta^2, X (S, P1, D), Y (S, P1)."""
    P1 = X.shape[1]
    Xt, Yt, Wt = X.reshape(-1, X.shape[2]), Y.reshape(-1), np.repeat(q, P1)
    M = Xt.T @ (Wt[:, None] * Xt)
    return np.linalg.solve(M + np.diag(pen(np.diag(M))), Xt.T @ (Wt * Yt))


def metrics(E, P):
    """Median relative error and cosine similarity of rows E (S, D) vs predictions P."""
    ne = np.linalg.norm(E, axis=1); ok = ne > 0
    rel = np.linalg.norm(E - P, axis=1)[ok] / ne[ok]
    cos = np.sum(E * P, 1)[ok] / (ne[ok] * np.maximum(np.linalg.norm(P, axis=1)[ok], 1e-30))
    return float(np.median(rel)), float(np.median(cos))


def fit(args):
    cfg, _, trial = setup()
    N, ntar = cfg["N"], cfg["Ntar"]
    ns = [len(t) for t in trial["t_targets"]]
    d = np.load(DATA)
    DL, EL, EH, G = d["DL"], d["EL"], d["EH"], d["G"]
    S, nmax, Kh, P1 = len(d["trial"]), DL.shape[2], EH.shape[2], N - 1
    OFF = np.array([[j for j in range(N) if j != i] for i in range(N)])
    blk = (np.arange(S) // args.block) % 2
    tr, te = blk == 0, blk == 1
    q = 1.0 / np.maximum(np.linalg.norm(G[:, :ntar], axis=(1, 2)), 1e-30) ** 2

    MIX = np.tile(np.eye(P1), (ntar, nmax, 1, 1))              # I + A^(i,k)
    CROSS = np.zeros((ntar, ntar, nmax, nmax))                 # C^(m,i)_{k_m k_i}
    OFFT = np.zeros((N - ntar, ntar, nmax, Kh))                # C^(m,l)_{k_m k_l}
    out = {"metrics": {}}
    print(f"  {S} recordings, every {int(d['stride'])} trials; held-out: alternate blocks of "
          f"{args.block}\n  ridges r_A = {args.r_A}, r_C = {args.r_C}, r_C_off = {args.r_C_off}\n")
    print(f"  {'row':<26} {'rel. error':>10} {'cosine':>8}   (held-out medians)")

    for i in range(ntar):                                       # target rows, Eq. FitAC
        n = ns[i]
        e = EL[:, i, :n][:, :, OFF[i]]                          # (S, n, P1)
        F = DL[:, i, :n, None] * e
        local = F.sum(1)
        XA = np.einsum("skp,jq->sjkpq", F, np.eye(P1)).reshape(S, P1, -1)
        others = [m for m in range(ntar) if m != i]
        XC = np.concatenate([np.einsum("sa,scj->sjac", DL[:, m, :ns[m]], e).reshape(S, P1, -1)
                             for m in others], axis=2)
        X = np.concatenate([XA, XC], axis=2)
        pa = XA.shape[2]
        theta = wls(X[tr], (G[:, i] - local)[tr], q[tr], lambda dg: np.concatenate(
            [np.full(pa, args.r_A * dg[:pa].mean()),
             np.full(len(dg) - pa, args.r_C * max(dg[pa:].mean(), 1e-300))]))
        MIX[i, :n] = np.eye(P1)[None] + theta[:pa].reshape(n, P1, P1)
        o = pa
        for m in others:
            CROSS[m, i, :ns[m], :n] = theta[o:o + ns[m] * n].reshape(ns[m], n)
            o += ns[m] * n
        for name, P in [("local", local), ("A + C", local + X @ theta)]:
            rel, cos = metrics(G[te, i], P[te])
            out["metrics"][f"target {i}|{name}"] = [rel, cos]
            print(f"  target {i}, {name:<16} {rel:10.3f} {cos:8.3f}")

    Ph = np.zeros((S, N - ntar, P1))
    for l in range(ntar, N):                                    # off-target rows, Eq. FitOff
        e = EH[:, l - ntar][:, :, OFF[l]]                       # (S, Kh, P1)
        X = np.einsum("smc,sdj->sjmcd", DL, e).reshape(S, P1, -1)
        theta = wls(X[tr], G[tr, l], q[tr],
                    lambda dg: np.full(len(dg), args.r_C_off * max(dg.mean(), 1e-300)))
        OFFT[l - ntar] = theta.reshape(ntar, nmax, Kh)
        Ph[:, l - ntar] = X @ theta
        rel, cos = metrics(G[te, l], Ph[te, l - ntar])
        out["metrics"][f"off-target {l}"] = [rel, cos]
        print(f"  off-target {l:<14} {rel:10.3f} {cos:8.3f}")
    rel, cos = metrics(G[te, ntar:].reshape(te.sum(), -1), Ph[te].reshape(te.sum(), -1))
    out["metrics"]["off-target, all rows"] = [rel, cos]
    print(f"  {'off-target, all rows':<26} {rel:10.3f} {cos:8.3f}")

    out.update({"spike_mix": MIX.tolist(), "cross": CROSS.tolist(), "offtarget": OFFT.tolist(),
                "n_outputs": ns, "n_hidden_spikes": int(Kh),
                "r_A": args.r_A, "r_C": args.r_C, "r_C_off": args.r_C_off})
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"  -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["collect", "fit"])
    ap.add_argument("--trials", type=int, default=10_000)
    ap.add_argument("--stride", type=int, default=10, help="collect: record every N trials")
    ap.add_argument("--n-hidden", type=int, default=8,
                    help="collect: spikes per off-target neuron with their own coefficients")
    ap.add_argument("--block", type=int, default=50, help="fit: recordings per fit/held-out block")
    ap.add_argument("--r-A", dest="r_A", type=float, default=1.0)
    ap.add_argument("--r-C", dest="r_C", type=float, default=1e-3)
    ap.add_argument("--r-C-off", dest="r_C_off", type=float, default=1.0)
    ap.add_argument("--out", default=FIT)
    args = ap.parse_args()
    ensure_output_dirs()
    args.out = output_file(args.out)
    {"collect": collect, "fit": fit}[args.stage](args)


if __name__ == "__main__":
    main()
