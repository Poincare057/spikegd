"""Exact gradients built only from the paper's closed forms (paper_forms.py): the event
maps of App. B, the injection (0,1),
the recurrent block of eq. (9) and the cross block of Prop. 7. No autodiff, no
finite differences.
"""

from __future__ import annotations

import numpy as np
import paper_forms as P

# %%
############################
### Single-neuron gradients
############################


def single_neuron(
    t_in: np.ndarray,
    syn: np.ndarray,
    w: np.ndarray,
    V0: float,
    b0: float,
    T: float,
    with_b0: bool = False,
) -> tuple:
    """RTRL of Sec. 2.2 for one neuron with external inputs. Parameters theta = (V0,
    w_1..w_l[, b0]).
    Returns output times and the time row of the trace at every output (list of
    arrays of length 1+l[+1]).
    """
    ### Initialize the simulation
    n_weights = len(w)
    M = np.zeros((2, 1 + n_weights + int(with_b0)))
    M[0, 0] = 1.0
    if with_b0:
        M[1, -1] = 1.0  # the initial current is a parameter: d b0 / d b0 = 1
    kind, s, t_last = "in", np.array([V0, b0], float), 0.0
    outs, rows, j = [], [], 0

    ### Event and gradient updates
    while True:
        t_next = t_in[j] if j < len(t_in) else np.inf
        if kind == "in":
            ts = t_last + P.tsp(*s) if _fires(s) else np.inf
        else:
            ts = s[0] + P.tsp_reset(s[1]) if s[1] > np.pi**2 else np.inf
        if ts < min(t_next, T):
            if kind == "in":
                J = P.J_in_out(*s)
                Tsp = P.tsp(*s)
                s = np.array([t_last + Tsp, s[1] * np.exp(-2 * Tsp)])
            else:
                J = P.J_out_out(s[1])
                Tsp = P.tsp_reset(s[1])
                s = np.array([s[0] + Tsp, s[1] * np.exp(-2 * Tsp)])
            M = J @ M
            kind = "out"
            outs.append(s[0])
            rows.append(M[0].copy())
        elif t_next < T:
            ww = w[syn[j]]
            if kind == "in":
                D = t_next - t_last
                J = P.J_in_in(s[0], s[1], D)
                s = np.array([P.V(s[0], s[1], D), s[1] * np.exp(-2 * D) + ww])
            else:
                D = t_next - s[0]
                J = P.J_out_in(s[1], D)
                s = np.array([P.V_reset(s[1], D), s[1] * np.exp(-2 * D) + ww])
            M = J @ M
            M[1, 1 + syn[j]] += 1.0
            kind = "in"
            t_last = t_next
            j += 1
        else:
            return np.array(outs), np.array(rows), (kind, s, t_last, M)


def _fires(s: np.ndarray) -> bool:
    V0, b = s
    if b > 0:
        return P.D(V0, b) > 0
    if b == 0:
        return V0 > 1
    return V0 > np.sqrt(-b) and P.D(V0, b) > 0


# %%
############################
### Recurrent gradients
############################


def r1(
    w_ext: float,
    t_ext: float,
    wj: float,
    d: float,
    T: float,
    V0: float = 0.0,
    b0: float = 0.0,
) -> dict:
    """R1 (Sec. 4.1): one external input, then every output re-enters after d through
    w_j. theta = (V0, w_ext, w_j).
    Exact trace with J_rec (eq. 9) and the e-prop trace without it.
    """
    out = {}
    for exact in (True, False):
        M = np.zeros((2, 3))
        M[0, 0] = 1.0
        D = t_ext
        s = np.array([P.V(V0, b0, D), b0 * np.exp(-2 * D) + w_ext])
        J = P.J_in_in(V0, b0, D)
        M = J @ M
        M[1, 1] += 1.0
        t_last = t_ext
        outs, rows = [], []
        while True:
            if not _fires(s):
                break
            Tsp = P.tsp(*s)
            if t_last + Tsp >= T:
                break
            J = P.J_in_out(*s)
            s = np.array([t_last + Tsp, s[1] * np.exp(-2 * Tsp)])
            M = J @ M
            outs.append(s[0])
            rows.append(M[0].copy())
            if s[0] + d >= T:
                break
            J = P.J_out_in(s[1], d) + (P.J_rec(wj) if exact else 0)
            s = np.array([P.V_reset(s[1], d), s[1] * np.exp(-2 * d) + wj])
            M = J @ M
            M[1, 2] += 1.0
            t_last = outs[-1] + d
        out["exact" if exact else "eprop"] = (np.array(outs), np.array(rows))
    return out


# %%
############################
### Network gradients
############################


def network(
    t_ext: np.ndarray,
    idx_ext: np.ndarray,
    W_in: np.ndarray,
    W_net: np.ndarray,
    x0: np.ndarray,
    T: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """N-neuron network with zero delays (spikegd's convention): a spike reaches every
    target at its own time.
    State a(k) stacks the two-vectors; J(k) has the single-neuron maps on the
    diagonal and the cross blocks of Prop. 7.
    theta = (W_in.ravel(), W_net.ravel()). Returns, per neuron, the output times and
    the time row of the trace at each output.
    """
    ### Initialize the network
    N, Nin = W_in.shape
    P_ = N * Nin + N * N
    kinds = ["in"] * N
    S = [np.array(x0[:, n], float) for n in range(N)]
    tl = [0.0] * N
    M = np.zeros((2 * N, P_))
    outs = [[] for _ in range(N)]
    rows = [[] for _ in range(N)]
    j = 0

    # Next output time
    def next_spike(n: int) -> float:
        s = S[n]
        if kinds[n] == "in":
            return tl[n] + P.tsp(*s) if _fires(s) else np.inf
        return s[0] + P.tsp_reset(s[1]) if s[1] > np.pi**2 else np.inf

    # State and Jacobian for a receiving neuron
    def recv(n: int, t: float, w: float) -> tuple[np.ndarray, np.ndarray]:
        """diagonal block and new state of neuron n receiving an input of weight w at
        time t (input time held fixed).
        """
        s = S[n]
        if kinds[n] == "in":
            D = t - tl[n]
            J = P.J_in_in(s[0], s[1], D)
            new = np.array([P.V(s[0], s[1], D), s[1] * np.exp(-2 * D) + w])
        else:
            D = t - s[0]
            J = P.J_out_in(s[1], D)
            new = np.array([P.V_reset(s[1], D), s[1] * np.exp(-2 * D) + w])
        return J, new

    while True:
        t_next = t_ext[j] if j < len(t_ext) else np.inf
        tsps = [next_spike(n) for n in range(N)]
        X = int(np.argmin(tsps))
        tX = tsps[X]
        Jk = np.eye(2 * N)
        if tX < min(t_next, T):  # neuron X fires; its spike reaches every Y != X now
            s = S[X]
            if kinds[X] == "in":
                JX = P.J_in_out(*s)
                Tsp = P.tsp(*s)
                newX = np.array([tl[X] + Tsp, s[1] * np.exp(-2 * Tsp)])
            else:
                JX = P.J_out_out(s[1])
                Tsp = P.tsp_reset(s[1])
                newX = np.array([s[0] + Tsp, s[1] * np.exp(-2 * Tsp)])
            Jk[2 * X : 2 * X + 2, 2 * X : 2 * X + 2] = JX
            grad_tX = JX[0]
            inj = []
            for Y in range(N):
                if Y == X:
                    continue
                w = W_net[Y, X]
                JY, newY = recv(Y, tX, w)
                Jk[2 * Y : 2 * Y + 2, 2 * Y : 2 * Y + 2] = JY
                Jk[2 * Y : 2 * Y + 2, 2 * X : 2 * X + 2] = P.J_cross(
                    w, grad_tX
                )  # Prop. 7
                S[Y] = newY
                kinds[Y] = "in"
                tl[Y] = tX
                inj.append((Y, N * Nin + Y * N + X))
            S[X] = newX
            kinds[X] = "out"
            M = Jk @ M
            for Y, col in inj:
                M[2 * Y + 1, col] += 1.0
            outs[X].append(tX)
            rows[X].append(M[2 * X].copy())
        elif t_next < T:
            # external input j reaches every neuron Y with weight W_in[Y, idx]; a zero
            # weight is an event with no jump
            Jk = np.eye(2 * N)
            cols = []
            for Y in range(N):
                JY, newY = recv(Y, t_next, W_in[Y, idx_ext[j]])
                Jk[2 * Y : 2 * Y + 2, 2 * Y : 2 * Y + 2] = JY
                S[Y] = newY
                kinds[Y] = "in"
                tl[Y] = t_next
                cols.append((Y, Y * Nin + idx_ext[j]))
            M = Jk @ M
            for Y, col in cols:
                M[2 * Y + 1, col] += 1.0
            j += 1
        else:
            break
    return [np.array(o) for o in outs], [np.array(r) for r in rows]
