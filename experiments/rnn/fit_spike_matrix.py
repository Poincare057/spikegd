"""
Fit the spike-specific synapse-mixing matrices A^(i,k) of the working notes:

    dt_{k_i}/dw_ij  ~  sum_j' e_ij'(t_{k_i}) (delta_j'j + A^(i,k)_j'j),
    e_ij'(t_{k_i}) = dt_{k_i}/dw_ij' with all presynaptic spike times fixed,

one matrix per target neuron i and output index k (k counts the output times
that enter the loss: ordinary spikes first, then pseudospikes, as in rnn.outfn).
Because A^(i,k) depends on k it cannot be fitted from summed gradients, so the
per-output pieces are recorded along the published exact run:

    L_k   = dL/dt_{k_i}
    ex_k  = d t_{k_i} / d w_i.   (exact: autodiff through the full simulation)
    loc_k = e_i.(t_{k_i})        (local: frozen-surround resimulation)

The exact gradient on neuron i's input weights is not only the sum over
neuron i's own outputs: once the network is coupled, w_ij also moves the other
target's spikes,

    dL/dw_ij = sum_m sum_{k_m} dL/dt_{k_m} * dt_{k_m}/dw_ij     (m over all targets).

The local eligibility of target m != i with respect to w_ij is zero, so these
cross-target terms cannot be written as a correction of their own spikes'
eligibilities. All A^(i,k) of neuron i are therefore fitted jointly to the
total exact gradient row, keeping the spike-specific structure:

    dL/dw_i.  ~  sum_k L_k loc_k (I + A^(i,k)),

which is linear in the stacked matrices; weighted least squares over trials,
trial weight 1 / ||g_exact(target rows)||^2 as for the other fits.

Usage (from experiments/rnn/):

    python fit_spike_matrix.py collect     # -> spike_data_km.npz  (~10 min)
    python fit_spike_matrix.py fit         # -> spike_A_fit_km.json
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
from rnn import outfn

DATA = "spike_data_km.npz"
FIT = "spike_A_fit_km.json"


def setup():
    cfg = dict(ex.BASE_CONFIG)
    neuron = QIFNeuron(cfg["tau"], cfg["eps"], cfg["alpha"])
    trial = ex.make_trial(cfg, cfg["Ntar"], 0.0)       # published: zero init
    return cfg, neuron, trial


def make_pieces(neuron, trial):
    """f(w, x0) -> per target i: (L (n_i,), exact (n_i, N-1), local (n_i, N-1), ordinary (n_i,))."""
    cfg = trial["cfg"]
    ntar, tt, denom = cfg["Ntar"], trial["t_targets"], trial["denom"]
    ns = [len(t) for t in tt]
    ones = jnp.ones_like(trial["w0"]); one = jnp.array(1.0)
    single = dict(cfg); single["Ntar"] = 1

    def touts_exact(w, x0):
        out = neuron.event(x0, ex.w_to_weights_net(w), trial["weights_in"],
                           trial["spikes_in"], cfg)
        return outfn(neuron, out, [w, x0], tt, cfg), out

    def frozen_touts(w, x0, queue, rates, k):
        """Output times of target k re-simulated against a frozen surround
        (same construction as experiment.make_frozen_loss)."""
        n = ns[k]
        wn = ex.w_to_weights_net(w)
        weights_eff = jnp.concatenate([trial["weights_in"][k], wn[k], jnp.zeros(1)])[None]
        out = neuron.event(x0[:, k][:, None], jnp.zeros((1, 1)), weights_eff, queue, single)
        times, is_in, neurons, xs = out
        Kord = jnp.sum(neurons >= 0)
        Kout = jnp.sum((neurons == 0) & (is_in == False))  # noqa: E712
        t_ord = times[jnp.argsort(is_in)[:n]]
        t_ps = neuron.t_pseudo(xs[Kord][:, 0], neuron.linear(rates, wn[k]),
                               jnp.arange(n) + 1 - Kout, cfg)
        return jnp.where(jnp.arange(n) < Kout, t_ord, t_ps)

    def fn(w, x0):
        touts, out = touts_exact(w, x0)
        touts = [jax.lax.stop_gradient(t) for t in touts]
        L = jax.grad(lambda ts: ex.lossfn_shared(ts, tt, denom))(touts)
        _, is_in, neurons, _ = out
        res = []
        for k in range(ntar):
            jex = jax.jacrev(lambda ww, k=k: touts_exact(ww, x0)[0][k])(w)[:, k, :]
            queue, rates, _ = ex.record_surround(neuron, trial, w, x0, ones, one, k)
            jlo = jax.jacrev(lambda ww, k=k: frozen_touts(ww, x0, queue, rates, k))(w)[:, k, :]
            Kout = jnp.sum((neurons == k) & (~is_in))
            res.append((L[k], jex, jlo, jnp.arange(ns[k]) < Kout))
        return res

    return fn


def stage_collect(args):
    cfg, neuron, trial = setup()
    ntar = cfg["Ntar"]
    ones = jnp.ones_like(trial["w0"]); one = jnp.array(1.0)
    full = ex.make_full_loss(neuron, trial, ones)
    pieces = jax.jit(make_pieces(neuron, trial))
    trace = jax.jit(make_fast_grad(neuron, trial, ntar, "trace"))
    optim = optax.adabelief(optax.exponential_decay(1e-2, 2000, 1 / jnp.e))

    @jax.jit
    def step(p, s):
        loss, g = jax.value_and_grad(lambda q: full(q[0], q[1], one))(p)
        u, s = optim.update(g, s)
        return optax.apply_updates(p, u), s, loss, g[0]

    p = (trial["w0"], trial["x0"]); s = optim.init(p)
    rec = {f"{name}{k}": [] for k in range(ntar) for name in ("L", "EX", "LO", "ORD")}
    rec["Erows"] = []; rec["trial"] = []
    checks = []
    t0 = time.time()
    for t in range(args.trials):
        p_before = p
        p, s, loss, gE = step(p, s)
        if t % args.stride == 0:
            res = pieces(*p_before)
            for k, (L, jex, jlo, ordi) in enumerate(res):
                rec[f"L{k}"].append(np.asarray(L)); rec[f"EX{k}"].append(np.asarray(jex))
                rec[f"LO{k}"].append(np.asarray(jlo)); rec[f"ORD{k}"].append(np.asarray(ordi))
            rec["Erows"].append(np.asarray(gE[:ntar])); rec["trial"].append(t)
            if len(checks) < 5 and t % (args.stride * 200) == 0:
                _, (gL, _) = trace(p_before[0], p_before[1], trial["spikes_in"])
                for k, (L, jex, jlo, _) in enumerate(res):
                    se = np.asarray(jnp.einsum("c,cj->j", L, jex))
                    sl = np.asarray(jnp.einsum("c,cj->j", L, jlo))
                    re = np.linalg.norm(se - np.asarray(gE[k])) / max(np.linalg.norm(gE[k]), 1e-30)
                    rl = np.linalg.norm(sl - np.asarray(gL[k])) / max(np.linalg.norm(gL[k]), 1e-30)
                    checks.append((t, k, re, rl))
                    print(f"  check trial {t}: neuron {k}  local pieces vs trace gradient {rl:.1e}"
                          f"   (cross-target share of exact gradient: own-spike sum differs "
                          f"by {re:.2f})", flush=True)
        if t % 1000 == 0:
            print(f"  trial {t:6d}  loss {float(loss):.4e}  ({time.time() - t0:.0f}s)", flush=True)
    np.savez(DATA, **{k: np.array(v) for k, v in rec.items()}, stride=args.stride)
    print(f"  {len(rec['trial'])} snapshots -> {DATA}")


def stage_fit(args):
    cfg, _, _ = setup()
    N, ntar = cfg["N"], cfg["Ntar"]
    d = np.load(DATA)
    S = len(d["trial"])
    blk = (np.arange(S) // args.block) % 2
    tr, te = blk == 0, blk == 1
    w = 1.0 / np.maximum(np.linalg.norm(d["Erows"], axis=(1, 2)), 1e-30) ** 2
    ns = [d[f"L{k}"].shape[1] for k in range(ntar)]
    nmax = max(ns)
    MIX = np.tile(np.eye(N - 1), (ntar, nmax, 1, 1))          # I + A^(i,k)
    for i in range(ntar):
        L, LO, Erow = d[f"L{i}"], d[f"LO{i}"], d["Erows"][:, i]
        n = ns[i]
        F = (L[:, :, None] * LO).reshape(S, n * (N - 1))         # features [L_k loc_k]_k
        G = np.einsum("sc,scj->sj", L, LO)                        # local gradient row
        X, Y, Wt = F[tr], (Erow - G)[tr], w[tr]
        M = X.T @ (Wt[:, None] * X)
        lam = args.ridge * max(np.trace(M), 1e-300) / M.shape[0]
        Ablk = np.linalg.solve(M + lam * np.eye(M.shape[0]), X.T @ (Wt[:, None] * Y))
        MIX[i, :n] = np.eye(N - 1)[None] + Ablk.reshape(n, N - 1, N - 1)
        own = np.einsum("sc,scj->sj", L, d[f"EX{i}"])
        cross = np.median(np.linalg.norm(Erow - own, axis=1) /
                          np.maximum(np.linalg.norm(Erow, axis=1), 1e-30))
        print(f"  neuron {i}: median |cross-target part| / |exact gradient| = {cross:.2f}")

    # gradient-level comparison on held-out snapshots, per target neuron
    Aper = np.array(json.load(open(args.A_fit))["A_per_neuron"])   # net layout (N, N, N)
    print(f"  {S} snapshots (every {int(d['stride'])} trials); held-out = alternate blocks of "
          f"{args.block}\n")
    print(f"  {'model':<26} {'rows':<9} {'rel err (test)':>15} {'cos (test)':>11}")
    out = {"spike_mix": MIX.tolist(), "n_outputs": ns, "metrics": {}}
    for i in range(ntar):
        keep = [j for j in range(N) if j != i]
        L, LO = d[f"L{i}"], d[f"LO{i}"]
        E = d["Erows"][:, i]                                      # total exact gradient row
        G = np.einsum("sc,scj->sj", L, LO)
        preds = {"local": G,
                 "per-neuron A^(i)": G @ (np.eye(N) + Aper[i])[np.ix_(keep, keep)],
                 "spike-specific A^(i,k)": np.einsum("sc,scj,cjm->sm", L, LO, MIX[i, :ns[i]])}
        for name, P in preds.items():
            e, p = E[te], P[te]
            ne = np.linalg.norm(e, axis=1); ok = ne > 0
            rel = np.median(np.linalg.norm(e - p, axis=1)[ok] / ne[ok])
            cos = np.median(np.sum(e * p, 1)[ok] / (ne[ok] * np.maximum(np.linalg.norm(p, axis=1)[ok], 1e-30)))
            out["metrics"][f"{name}|neuron {i}"] = [rel, cos]
            print(f"  {name:<26} neuron {i}  {rel:15.3f} {cos:11.3f}")
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    for i in range(ntar):
        for k in range(ns[i]):
            sv = np.linalg.svd(MIX[i, k], compute_uv=False)
            kind = np.mean(d[f"ORD{i}"][:, k])
            print(f"  neuron {i}, output {k}: ordinary spike in {kind:4.0%} of snapshots; "
                  f"singular values of I + A: {sv[:3]} ... {sv[-1]:.2f}")
    out["ridge"] = args.ridge
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"  -> {args.out}")


def stage_fitcross(args):
    """Joint fit of A^(i,k) and explicit cross-target coefficients C^(m,i)_{c'c}:

        dL/dw_ij ~ sum_c [ dL/dt_{c_i} sum_j' (delta_j'j + A^(i,c)_j'j) e_ij'(t_{c_i})
                           + ( sum_{m != i} sum_c' dL/dt_{c'_m} C^(m,i)_{c'c} ) e_ij(t_{c_i}) ],

    weighted least squares over the training trials against the total exact
    gradient, ridge r_A on A (relative to the mean diagonal of its block of
    the normal matrix) and r_C on C."""
    cfg, _, _ = setup()
    N, ntar = cfg["N"], cfg["Ntar"]
    d = np.load(DATA)
    S = len(d["trial"])
    blk = (np.arange(S) // args.block) % 2
    tr, te = blk == 0, blk == 1
    w = 1.0 / np.maximum(np.linalg.norm(d["Erows"], axis=(1, 2)), 1e-30) ** 2
    ns = [d[f"L{k}"].shape[1] for k in range(ntar)]
    nmax, P1 = max(ns), N - 1
    MIX = np.tile(np.eye(P1), (ntar, nmax, 1, 1))
    CROSS = np.zeros((ntar, ntar, nmax, nmax))
    try:   # optional comparison with the A-only spike fit, if it has been made
        old = json.load(open(args.A_only_fit))["spike_mix"]
    except FileNotFoundError:
        old = None
    print(f"  ridge on A: {args.ridge}, on C: {args.ridge_c}\n")
    print(f"  {'model':<34} {'rows':<9} {'rel err (test)':>15} {'cos (test)':>11}")
    out = {"metrics": {}}
    for i in range(ntar):
        L, LO, Erow = d[f"L{i}"], d[f"LO{i}"], d["Erows"][:, i]
        n = ns[i]
        F = L[:, :, None] * LO                                       # (S, n, P1)
        G = F.sum(1)                                                 # local gradient row
        XA = np.einsum("skp,jq->sjkpq", F, np.eye(P1)).reshape(S, P1, n * P1 * P1)
        others = [m for m in range(ntar) if m != i]
        XC = np.concatenate([np.einsum("sa,scj->sjac", d[f"L{m}"], LO).reshape(S, P1, -1)
                             for m in others], axis=2)              # (S, P1, sum n_m * n)
        X = np.concatenate([XA, XC], axis=2)
        Y = Erow - G
        Xt, Yt, Wt = X[tr].reshape(-1, X.shape[2]), Y[tr].reshape(-1), np.repeat(w[tr], P1)
        M = Xt.T @ (Wt[:, None] * Xt)
        pa = XA.shape[2]
        dg = np.diag(M)
        if args.ridge == 0 and args.ridge_c == 0:
            # no regularisation: minimum-norm least squares (the ridge -> 0 limit),
            # which handles the exactly singular directions of M
            sw = np.sqrt(Wt)
            theta = np.linalg.lstsq(sw[:, None] * Xt, sw * Yt, rcond=None)[0]
        else:
            pen = np.concatenate([np.full(pa, args.ridge * dg[:pa].mean()),
                                  np.full(M.shape[0] - pa,
                                          args.ridge_c * max(dg[pa:].mean(), 1e-300))])
            theta = np.linalg.solve(M + np.diag(pen), Xt.T @ (Wt * Yt))
        MIX[i, :n] = np.eye(P1)[None] + theta[:pa].reshape(n, P1, P1)
        off = pa
        for m in others:
            CROSS[m, i, :ns[m], :n] = theta[off:off + ns[m] * n].reshape(ns[m], n)
            off += ns[m] * n
        # held-out comparison
        preds = {"local": G}
        if old is not None:
            preds["spike A^(i,k) only (fitted earlier)"] = np.einsum(
                "sc,scj,cjm->sm", L, LO, np.array(old)[i, :n])
        preds["spike A^(i,k) + cross-target C"] = np.einsum("sjp,p->sj", X, theta) + G
        for name, Pr in preds.items():
            e, p = Erow[te], Pr[te]
            ne = np.linalg.norm(e, axis=1); ok = ne > 0
            rel = np.median(np.linalg.norm(e - p, axis=1)[ok] / ne[ok])
            cos = np.median(np.sum(e * p, 1)[ok] / (ne[ok] * np.maximum(np.linalg.norm(p, axis=1)[ok], 1e-30)))
            out["metrics"][f"{name}|neuron {i}"] = [rel, cos]
            print(f"  {name:<34} neuron {i}  {rel:15.3f} {cos:11.3f}")
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    for i in range(ntar):
        for m in range(ntar):
            if m != i:
                print(f"\n  C^({m},{i}) [rows: outputs of target {m}, columns: outputs of target {i}]:\n"
                      f"{CROSS[m, i, :ns[m], :ns[i]]}")
    out.update({"spike_mix": MIX.tolist(), "cross": CROSS.tolist(), "n_outputs": ns,
                "ridge": args.ridge, "ridge_c": args.ridge_c})
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"  -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["collect", "fit", "fitcross"])
    ap.add_argument("--ridge-c", type=float, default=1e-3, help="ridge on C (fitcross)")
    ap.add_argument("--A-only-fit", dest="A_only_fit", default="spike_A_fit_km.json")
    ap.add_argument("--trials", type=int, default=10_000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--block", type=int, default=50, help="snapshots per train/test block")
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--A-fit", dest="A_fit", default="A_fit_km.json")
    ap.add_argument("--out", default=FIT, help="fit output file")
    args = ap.parse_args()
    {"collect": stage_collect, "fit": stage_fit, "fitcross": stage_fitcross}[args.stage](args)


if __name__ == "__main__":
    main()
