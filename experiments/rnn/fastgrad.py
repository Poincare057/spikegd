"""
One-pass local gradients for the QIF RNN: no backward pass, no re-simulation.

Both methods advance the network with spikegd's own `QIFNeuron.spike`, so the
dynamics are identical to `neuron.event`, and run a small extra update beside
it at every event. The loss gradient is accumulated online as each target
spike happens (dL/dt_c is known the moment spike c occurs), so only O(Ntar * N)
numbers are carried and nothing is stored per event.

  mode="trace"   forward eligibility traces. For each target neuron k, carry
                 d(V_k, I_k)/dp and d(next spike time of k)/dp for the N + 2
                 directions p = (row k of the weights, V_k(0), I_k(0)), with
                 every other spike time held fixed. This is exactly the
                 frozen-surround gradient of experiment.make_frozen_loss.
                 Cost per event: O(Ntar * N).

  mode="closed"  the closed form from the working notes, Eq. (wGrad):
                     dt_sp/dw_kj = dt_sp/dI(tau_m) * P_j(tau_m),
                     P_j(t) = sum_{t_j < t} exp(-2 (t - t_j)),
                 with the state at the last event tau_m held fixed (u_m fixed).
                 Carries one presynaptic trace per neuron, shared by all
                 targets. Cost per event: O(N), plus O(N) per target spike.
                 Initial states are not updated (dt/dx0 is zero once any
                 event has occurred).

Pseudospikes (target spikes that have not occurred by T) are differentiated
through the end state with the same convention as each mode.

    g = make_fast_grad(neuron, trial, ntar, "trace")
    loss, (gw, gx0) = g(w, x0, spikes_in)      # gw in spikegd's (N, N-1) layout
"""

import jax
import jax.numpy as jnp
from jax import lax

import experiment as ex


def make_fast_grad(neuron, trial, ntar, mode="trace", spike_mix=None, cross=None):
    """spike_mix (trace mode only): optional array (ntar, nmax, N-1, N-1) holding
    (I + A^(i,k)) for target i and output index k. Each output's eligibility over
    its neuron's input weights (spikegd layout, self entry dropped) is multiplied
    by its own matrix before it is accumulated; the initial-state directions are
    left unchanged. None gives the plain local gradient.

    cross (trace mode only): optional array (ntar, ntar, nmax, nmax) of
    coefficients C[m, i, c', c] for the explicit cross-target term
        g_ij += sum_c ( sum_{m != i} sum_c' dL/dt_{c'_m} C[m, i, c', c] ) e_ij(t_{c_i}).
    The other target's learning signals are only all known at the end of the
    trial, so each output's eligibility and dL/dt are stored (ntar * nmax
    vectors of length N + 2) and the term is added at trial end."""
    assert mode in ("trace", "closed")
    assert (spike_mix is None and cross is None) or mode == "trace"
    cfg = trial["cfg"]
    N, T, K, eps = cfg["N"], cfg["T"], cfg["K"], neuron.eps
    denom = trial["denom"]
    weights_in = trial["weights_in"]
    tars = [jnp.asarray(t) for t in trial["t_targets"][:ntar]]
    ns = [len(t) for t in tars]
    D = N + 2  # directions: row k of weights_net (N), V_k(0), I_k(0)
    # column indices that drop the diagonal: (N, N) -> spikegd's (N, N-1) layout
    OFFDIAG = jnp.array([[j for j in range(N) if j != i] for i in range(N)])

    def dloss_dt(k, c, t):
        """dL/dt for the c-th output spike of target k occurring at time t."""
        tar = tars[k][jnp.minimum(c, ns[k] - 1)]
        gate = jnp.where(c == ns[k] - 1, t < tar, True) & (c < ns[k])
        return jnp.where(gate, 2 * (t - tar) / (tar + 2) ** 2 / denom, 0.0)

    def loss_terms(k, ts, active):
        """Sum of loss terms of target k for output times ts (len n_k)."""
        tar = tars[k]
        cs = jnp.arange(ns[k])
        ts = jnp.where(active, ts, tar)          # keep inactive terms finite
        gate = jnp.where(cs == ns[k] - 1, ts < tar, True) & active
        return jnp.sum(jnp.where(gate, ((ts - tar) / (tar + 2)) ** 2, 0.0)) / denom

    def grad_dt_spike(xk):
        """d(time to next spike)/d(V, I) for one neuron; 0 if it never spikes."""
        dt = neuron.dt_spike(xk)
        g = jax.grad(neuron.dt_spike)(xk)
        return jnp.where(jnp.isfinite(dt) & jnp.all(jnp.isfinite(g)), g, 0.0)

    flow_jvp = jax.vmap(lambda xk, dt, z, ddt: jax.jvp(neuron.flow, (xk, dt), (z, ddt))[1],
                        in_axes=(None, None, 1, 0), out_axes=1)

    MIX = None if spike_mix is None else jnp.asarray(spike_mix)
    CROSS = None if cross is None else jnp.asarray(cross)

    def mixed(k, c, vec):
        """Apply (I + A^(k,c)) to the weight part of a (D,) eligibility vector."""
        if MIX is None:
            return vec
        c = jnp.clip(c, 0, MIX.shape[1] - 1)
        off = vec[:N][OFFDIAG[k]] @ MIX[k, c]
        return jnp.concatenate([jnp.zeros(N).at[OFFDIAG[k]].set(off), vec[N:]])

    def fn(w, x0, spikes_in):
        wn = ex.w_to_weights_net(w)
        order = jnp.argsort(spikes_in[0])
        spikes = (jnp.append(spikes_in[0][order], jnp.inf),
                  jnp.append(spikes_in[1][order], 0))
        state = (jnp.array(0.0), x0, neuron.dt_spike(x0), jnp.array(0))

        nmax = max(ns)
        tr = {"cnt": jnp.zeros(ntar, int), "g": jnp.zeros((ntar, D)),
              "tord": jnp.full((ntar, nmax), jnp.inf)}
        if mode == "trace":
            z0 = jnp.zeros((2, D)).at[0, N].set(1.0).at[1, N + 1].set(1.0)
            tr["z"] = jnp.stack([z0] * ntar)                      # (ntar, 2, D)
            tr["tau"] = jnp.zeros((ntar, D))                      # d(event time)
            tr["sig"] = jnp.stack([grad_dt_spike(x0[:, k]) @ z0   # d(next spike of k)
                                   for k in range(ntar)])
            if CROSS is not None:
                tr["EL"] = jnp.zeros((ntar, nmax, D))    # eligibility of each output
                tr["DL"] = jnp.zeros((ntar, nmax))       # dL/dt of each output
        else:
            tr["P"] = jnp.zeros(N)
            tr["s"] = jnp.stack([grad_dt_spike(x0[:, k])[1] for k in range(ntar)])

        def step(carry, _):
            state, tr = carry
            t_old, x_old, dtn_old, _ = state
            state_new, (t_new, spike_in, j, x_new) = neuron.spike(
                state, wn, weights_in, spikes, cfg)
            dt = t_new - t_old
            j_net = jnp.argmin(dtn_old)
            net_spike = (~spike_in) & (j >= 0)
            mask = jnp.abs(dtn_old - dt) < eps
            tr = dict(tr)
            cnt, g, tord = tr["cnt"], tr["g"], tr["tord"]

            if mode == "trace":
                zs, taus, sigs = [], [], []
                for k in range(ntar):
                    is_k = net_spike & (j_net == k)
                    tau_new = jnp.where(is_k, tr["sig"][k], 0.0)   # surround: fixed
                    ddt = tau_new - tr["tau"][k]
                    xk = x_old[:, k].at[0].set(jnp.where(mask[k], 0.0, x_old[0, k]))
                    zk = tr["z"][k]
                    zk = zk.at[0].set(jnp.where(mask[k], 0.0, zk[0]))
                    zn = flow_jvp(xk, dt, zk, ddt)
                    zn = zn.at[0].set(jnp.where(mask[k], 0.0, zn[0]))
                    send = net_spike & (j_net != k)
                    zn = zn.at[1].add(jnp.where(send, jax.nn.one_hot(j_net, D), 0.0))
                    zn = zn.at[0].set(jnp.where(is_k, 0.0, zn[0]))   # reset
                    recomputed = (~mask[k]) | ((j_net == k) & ~spike_in)
                    sig = jnp.where(recomputed,
                                    tau_new + grad_dt_spike(x_new[:, k]) @ zn,
                                    tr["sig"][k])
                    c = cnt[k]
                    dl = jnp.where(is_k, dloss_dt(k, c, t_new), 0.0)
                    g = g.at[k].add(dl * mixed(k, c, tau_new))
                    if CROSS is not None:
                        ok = is_k & (c < ns[k])
                        ci = jnp.minimum(c, nmax - 1)
                        tr["EL"] = tr["EL"].at[k, ci].set(jnp.where(ok, tau_new, tr["EL"][k, ci]))
                        tr["DL"] = tr["DL"].at[k, ci].set(jnp.where(ok, dl, tr["DL"][k, ci]))
                    zs.append(zn); taus.append(tau_new); sigs.append(sig)
                tr["z"], tr["tau"], tr["sig"] = jnp.stack(zs), jnp.stack(taus), jnp.stack(sigs)
            else:
                P_prev = tr["P"]
                ss = []
                for k in range(ntar):
                    is_k = net_spike & (j_net == k)
                    c = cnt[k]
                    coef = jnp.where(is_k, dloss_dt(k, c, t_new), 0.0) * tr["s"][k]
                    g = g.at[k, :N].add(coef * P_prev)
                    recomputed = (~mask[k]) | ((j_net == k) & ~spike_in)
                    ss.append(jnp.where(recomputed, grad_dt_spike(x_new[:, k])[1],
                                        tr["s"][k]))
                tr["s"] = jnp.stack(ss)
                P = P_prev * jnp.exp(-2 * dt / neuron.tau)
                tr["P"] = P + jnp.where(net_spike, jax.nn.one_hot(j_net, N), 0.0)

            for k in range(ntar):
                is_k = net_spike & (j_net == k)
                c = cnt[k]
                ok = is_k & (c < ns[k])
                tord = tord.at[k, jnp.minimum(c, nmax - 1)].set(
                    jnp.where(ok, t_new, tord[k, jnp.minimum(c, nmax - 1)]))
                cnt = cnt.at[k].add(is_k.astype(int))
            tr["cnt"], tr["g"], tr["tord"] = cnt, g, tord
            return (state_new, tr), None

        (state_end, tr), _ = lax.scan(step, (state, tr), None, length=K)
        x_end = state_end[1]
        rates = neuron.construct_ratefn(x_end)(input=-x_end[1])

        loss = 0.0
        g = tr["g"]
        for k in range(ntar):
            n, Kout = ns[k], tr["cnt"][k]
            cs = jnp.arange(n)
            ordinary = cs < Kout
            loss = loss + loss_terms(k, tr["tord"][k, :n], ordinary)

            def fpseudo(xk, wrow, k=k, n=n, Kout=Kout, cs=cs):
                tp = neuron.t_pseudo(xk, neuron.linear(rates, wrow), cs + 1 - Kout, cfg)
                return loss_terms(k, tp, cs >= Kout)

            lp, (gx, gwr) = jax.value_and_grad(fpseudo, argnums=(0, 1))(x_end[:, k], wn[k])
            loss = loss + lp
            if mode == "trace" and MIX is not None:
                # per-output pseudospike terms, so each gets its own (I + A^(k,c))
                def fterms(xk, wrow, k=k, n=n, Kout=Kout, cs=cs):
                    tp = neuron.t_pseudo(xk, neuron.linear(rates, wrow), cs + 1 - Kout, cfg)
                    return jnp.stack([loss_terms(k, tp, (cs >= Kout) & (cs == c))
                                      for c in range(n)])
                jx, jw = jax.jacrev(fterms, argnums=(0, 1))(x_end[:, k], wn[k])
                for c in range(n):
                    vec = jx[c] @ tr["z"][k] + jnp.concatenate([jw[c], jnp.zeros(2)])
                    g = g.at[k].add(mixed(k, jnp.array(c), vec))
            elif mode == "trace":
                g = g.at[k].add(gx @ tr["z"][k] + jnp.concatenate([gwr, jnp.zeros(2)]))
            else:
                g = g.at[k, :N].add(gx[1] * tr["P"] + gwr)

            if CROSS is not None:
                # store eligibility and dL/dt of this neuron's pseudospike outputs
                def ftimes(xk, wrow, n=n, Kout=Kout, cs=cs):
                    return neuron.t_pseudo(xk, neuron.linear(rates, wrow), cs + 1 - Kout, cfg)
                tp = ftimes(x_end[:, k], wn[k])
                jtx, jtw = jax.jacrev(ftimes, argnums=(0, 1))(x_end[:, k], wn[k])
                for c in range(n):
                    pseudo = c >= Kout
                    el = jtx[c] @ tr["z"][k] + jnp.concatenate([jtw[c], jnp.zeros(2)])
                    dl = dloss_dt(k, jnp.array(c), tp[c])
                    tr["EL"] = tr["EL"].at[k, c].set(jnp.where(pseudo, el, tr["EL"][k, c]))
                    tr["DL"] = tr["DL"].at[k, c].set(jnp.where(pseudo, dl, tr["DL"][k, c]))

        if CROSS is not None:
            # explicit cross-target term: sum_c S_ic e_i.(t_c), with
            # S_ic = sum_{m != i} sum_c' dL/dt_{c'_m} C[m, i, c', c]
            for i in range(ntar):
                S = sum(tr["DL"][m] @ CROSS[m, i] for m in range(ntar) if m != i)   # (nmax,)
                g = g.at[i, :N].add(S[:ns[i]] @ tr["EL"][i, :ns[i], :N])

        gw_net = jnp.zeros((N, N)).at[:ntar].set(g[:, :N])
        gw = jnp.take_along_axis(gw_net, OFFDIAG, axis=1)
        gx0 = jnp.zeros((2, N)).at[:, :ntar].set(g[:, N:].T)
        return loss, (gw, gx0)

    return fn


def make_eligibility(neuron, trial):
    """Summed spike-time eligibility of every neuron, in one forward pass.

    Returns ebar(w, x0, spikes_in) of shape (N, N-1) (spikegd's layout) with
        ebar_ij = sum over ordinary spikes k of neuron i of dt_{k_i}/dw_ij,
    computed with the same frozen-surround traces as make_fast_grad("trace"):
    every other neuron's spike times are held fixed, neuron i's own dynamics
    (including its earlier spikes and resets) are propagated.
    """
    cfg = trial["cfg"]
    N, K, eps = cfg["N"], cfg["K"], neuron.eps
    weights_in = trial["weights_in"]
    OFFDIAG = jnp.array([[j for j in range(N) if j != i] for i in range(N)])

    def grad_dt_spike(xk):
        dt = neuron.dt_spike(xk)
        g = jax.grad(neuron.dt_spike)(xk)
        return jnp.where(jnp.isfinite(dt) & jnp.all(jnp.isfinite(g)), g, 0.0)

    flow_jvp = jax.vmap(lambda xk, dt, z, ddt: jax.jvp(neuron.flow, (xk, dt), (z, ddt))[1],
                        in_axes=(None, None, 1, 0), out_axes=1)

    def per_neuron(k, xo, xn, mask_k, z, tau, sig, eb, dt, j_net, net_spike, spike_in):
        is_k = net_spike & (j_net == k)
        tau_new = jnp.where(is_k, sig, 0.0)        # surround spike times are fixed
        ddt = tau_new - tau
        xk = xo.at[0].set(jnp.where(mask_k, 0.0, xo[0]))
        z = z.at[0].set(jnp.where(mask_k, 0.0, z[0]))
        zn = flow_jvp(xk, dt, z, ddt)
        zn = zn.at[0].set(jnp.where(mask_k, 0.0, zn[0]))
        send = net_spike & (j_net != k)
        zn = zn.at[1].add(jnp.where(send, jax.nn.one_hot(j_net, N), 0.0))
        zn = zn.at[0].set(jnp.where(is_k, 0.0, zn[0]))  # reset
        recomputed = (~mask_k) | ((j_net == k) & ~spike_in)
        sig = jnp.where(recomputed, tau_new + grad_dt_spike(xn) @ zn, sig)
        eb = eb + jnp.where(is_k, tau_new, 0.0)
        return zn, tau_new, sig, eb

    step_all = jax.vmap(per_neuron, in_axes=(0, 1, 1, 0, 0, 0, 0, 0,
                                             None, None, None, None))

    def fn(w, x0, spikes_in):
        wn = ex.w_to_weights_net(w)
        order = jnp.argsort(spikes_in[0])
        spikes = (jnp.append(spikes_in[0][order], jnp.inf),
                  jnp.append(spikes_in[1][order], 0))
        state = (jnp.array(0.0), x0, neuron.dt_spike(x0), jnp.array(0))
        z0 = jnp.zeros((N, 2, N))
        tr = (z0, jnp.zeros((N, N)), jnp.zeros((N, N)), jnp.zeros((N, N)))

        def step(carry, _):
            state, (z, tau, sig, eb) = carry
            t_old, x_old, dtn_old, _ = state
            state_new, (t_new, spike_in, j, x_new) = neuron.spike(
                state, wn, weights_in, spikes, cfg)
            dt = t_new - t_old
            j_net = jnp.argmin(dtn_old)
            net_spike = (~spike_in) & (j >= 0)
            mask = jnp.abs(dtn_old - dt) < eps
            tr = step_all(jnp.arange(N), x_old, x_new, mask, z, tau, sig, eb,
                          dt, j_net, net_spike, spike_in)
            return (state_new, tr), None

        (_, (_, _, _, eb)), _ = lax.scan(step, (state, tr), None, length=K)
        return jnp.take_along_axis(eb, OFFDIAG, axis=1)

    return fn
