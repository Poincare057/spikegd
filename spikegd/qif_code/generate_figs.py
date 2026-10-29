"""Generate the QIF paper figures and expose their exact event-driven simulator.

Run ``python generate_figs.py`` for every Python figure, or select figures with
``python generate_figs.py r1_train qr_train``. Use ``--recompute`` to refresh
pseudospike data. Importing this module does not generate figures or load JAX.

The event-driven simulator uses tau_m=1 and tau_s=1/2. Its state after an
input is (V0, b), with the external input time fixed; after an output it is
(t_out, b_out). Event maps use the closed forms of Sec. 2 and complex-step
Jacobians, f'(x) = Im f(x + i h) / h.
"""
import argparse
from pathlib import Path

from results import OUTPUT_DIR, capture_results

import numpy as np

H = 1e-30           # complex-step size
PI = np.pi

# ---------------------------------------------------------------- closed forms
def flow(V0, b, t):
    """V(t) from a finite V0 (eq. c-V). Branch chosen on the real parts."""
    br, Vr = np.real(b), np.real(V0)
    if br > 0:
        s = np.sqrt(b); return s*np.exp(-t)*np.tan(np.arctan(V0/s) + s*(1-np.exp(-t)))
    if br == 0:
        return V0/(V0-(V0-1)*np.exp(t))
    s = np.sqrt(-b)
    if abs(Vr) > np.real(s):   # coth branch, arcoth(x) = artanh(1/x)
        return s*np.exp(-t)/np.tanh(np.arctanh(s/V0) - s*(1-np.exp(-t)))
    return s*np.exp(-t)*np.tanh(np.arctanh(V0/s) - s*(1-np.exp(-t)))

def flow_reset(b, t):
    """V(t) a time t after a reset (V0 = -inf)."""
    br = np.real(b)
    if br > 0:
        s = np.sqrt(b); return s*np.exp(-t)*np.tan(-PI/2 + s*(1-np.exp(-t)))
    if br == 0:
        return -1/(np.exp(t)-1)
    s = np.sqrt(-b)
    return s*np.exp(-t)/np.tanh(-s*(1-np.exp(-t)))

def tsp(V0, b):
    """Time to the next spike from (V0,b) (eq. c-tsp); np.inf if there is none."""
    br, Vr = np.real(b), np.real(V0)
    if br > 0:
        s = np.sqrt(b); D = 1 - (PI/2 - np.arctan(V0/s))/s
    elif br == 0:
        if Vr <= 1: return np.inf
        D = 1 - 1/V0
    else:
        s = np.sqrt(-b)
        if Vr <= np.real(s): return np.inf
        D = 1 - np.arctanh(s/V0)/s
    return np.inf if np.real(D) <= 0 else -np.log(D)

def tsp_reset(b):
    """Time from a reset to the next spike (eq. c-reset); np.inf if b <= pi^2."""
    if np.real(b) <= PI**2: return np.inf
    return -np.log(1 - PI/np.sqrt(b))

# ---------------------------------------------------------------- event maps
def map_in_in(s, t_last, t_in, w):
    D = t_in - t_last
    return np.array([flow(s[0], s[1], D), s[1]*np.exp(-2*D) + w])

def map_in_out(s, t_last):
    T = tsp(s[0], s[1])
    return np.array([t_last + T, s[1]*np.exp(-2*T)])

def map_out_in(s, t_in, w):
    D = t_in - s[0]
    return np.array([flow_reset(s[1], D), s[1]*np.exp(-2*D) + w])

def map_out_out(s):
    T = tsp_reset(s[1])
    return np.array([s[0] + T, s[1]*np.exp(-2*T)])

def jac(f, s):
    """2x2 Jacobian of f at the real state s by the complex step."""
    J = np.empty((2, 2))
    for j in range(2):
        z = s.astype(complex); z[j] += 1j*H
        J[:, j] = np.imag(f(z))/H
    return J

# ---------------------------------------------------------------- simulation
def simulate(t_inputs, syn, weights, V0=0.0, b0=0.0, T_end=10.0, max_events=100000):
    """Event-driven run with RTRL traces.

    t_inputs : sorted input times (external, fixed);  syn : synapse index of each input;  weights : one weight per synapse.
    Parameters are theta = (V0, w_1, ..., w_l); the trace M is 2 x (1+l).
    Returns a dict with the event list (kind, time, state, Jacobian) and the traces at every event.
    """
    t_inputs = np.asarray(t_inputs, float); syn = np.asarray(syn, int); weights = np.asarray(weights, float)
    l = len(weights)
    kind, s, t_last = 'in', np.array([V0, b0], float), 0.0       # t=0 acts like an input with no jump
    M = np.zeros((2, 1+l)); M[0, 0] = 1.0
    ev = dict(kind=[], time=[], state=[], J=[], M=[], b_after_last_input=[])
    j = 0
    while len(ev['kind']) < max_events:
        t_next = t_inputs[j] if j < len(t_inputs) else np.inf
        t_spk = (t_last + tsp(*s)) if kind == 'in' else (s[0] + tsp_reset(s[1]))
        if t_spk < min(t_next, T_end):                              # an output comes first
            if kind == 'in':
                f = lambda z, tl=t_last: map_in_out(z, tl); new = 'in->out'
            else:
                f = map_out_out; new = 'out->out'
            Jk = jac(f, s); s = np.real(f(s)); kind = 'out'; inj = None
            M = Jk @ M
        elif t_next < T_end:                                        # an input comes first
            w = weights[syn[j]]
            if kind == 'in':
                f = lambda z, tl=t_last, tn=t_next, ww=w: map_in_in(z, tl, tn, ww); new = 'in->in'
            else:
                f = lambda z, tn=t_next, ww=w: map_out_in(z, tn, ww); new = 'out->in'
            Jk = jac(f, s); s = np.real(f(s)); kind = 'in'; t_last = t_next
            M = Jk @ M; M[1, 1+syn[j]] += 1.0                       # injection (0,1) in the column of the synapse
            j += 1
        else:
            break
        ev['kind'].append(new); ev['time'].append(s[0] if kind == 'out' else t_last)
        ev['state'].append(s.copy()); ev['J'].append(Jk); ev['M'].append(M.copy())
    for k in ev: ev[k] = list(ev[k])
    ev['outputs'] = [t for t, k in zip(ev['time'], ev['kind']) if k.endswith('out')]
    return ev

# ---------------------------------------------------------------- QR accumulation
def initial_basis(seed_ons=None):
    """Random orthonormal 2x2 basis (Gaussian QR, as in Toolkit 2); identity if seed_ons is None."""
    if seed_ons is None: return np.eye(2)
    Q, _ = np.linalg.qr(np.random.default_rng(seed_ons).standard_normal((2, 2)))
    return Q

def qr_accumulate(Js, Q0=None):
    """Benettin/QR along a product of event Jacobians, QR after every event, no alignment events discarded.
    Returns the running sums of log|R_ii| after every event (shape (K,2)), unsorted (Q is never reordered)."""
    Q = np.eye(2) if Q0 is None else np.array(Q0, float); logs = np.zeros(2); hist = []
    for Jk in Js:
        Q, R = np.linalg.qr(Jk @ Q)
        with np.errstate(divide='ignore'):
            logs = logs + np.log(np.abs(np.diag(R)))
        hist.append(logs.copy())
    return np.array(hist)

# ---------------------------------------------------------------- groups and the gap count
def groups(ev, t_inputs, T_end=np.inf):
    """Split the run at its outputs: for every maximal run of inputs with no output in between, return
    (index of last input, current B just after it, sqrt(B), c2, gap G to the next input or trial end, outputs in the gap)."""
    out = []
    t_inputs = np.asarray(t_inputs)
    kinds, times, states = ev['kind'], ev['time'], ev['state']
    for e, k in enumerate(kinds):
        if not k.endswith('in'): continue
        nxt = e+1
        if nxt < len(kinds) and kinds[nxt].endswith('in'): continue   # not the last input of its group
        V0, B = states[e]; tl = times[e]
        later = t_inputs[t_inputs > tl + 1e-12]
        G = min(later[0] if len(later) else np.inf, T_end) - tl
        n_out = 0
        while nxt < len(kinds) and kinds[nxt].endswith('out'):
            n_out += 1; nxt += 1
        c2 = np.sqrt(B) - PI/2 + np.arctan(V0/np.sqrt(B)) if B > 0 else -np.inf
        out.append(dict(event=e, t=tl, V0=V0, B=B, c2=c2, G=G, n_out=n_out))
    return out


WIDTH = 5.5                               # inches, NeurIPS text width
# colour-blind safe (Okabe-Ito)
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERM, PURPLE = '#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'
GREY = '0.6'

def configure_style():
    """Load the noninteractive plotting backend and apply the paper's style."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'font.family': 'serif', 'mathtext.fontset': 'cm',
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.2,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
    return plt


def save(fig, name, output_dir=OUTPUT_DIR):
    """Save a figure as SVG, then release it."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_dir / f"{name}.svg")
    finally:
        plt.close(fig)


def generate_r1_train(output_dir=OUTPUT_DIR):
    """Fig. 2(a): R1, current with and without the recurrent synapse.
    One external input (w=10 at t=0.3) starts the neuron from V0=b0=0; each output re-enters after d=0.15 through w_j.
    Uses the exact event-driven simulation of this module."""
    plt = configure_style()
    wj, d, T, t_ext, w_ext = 12.0, 0.15, 6.5, 0.3, 10.0

    def run(wj):
        """Event list, plus the current I(t) sampled between events (I decays as e^{-2t} and jumps by w at inputs)."""
        t_in, w_in = [t_ext], [w_ext]
        ev = None
        # re-entries depend on the outputs, so iterate: add the re-entry of every output found so far, rerun until stable
        for _ in range(50):
            ev = simulate(t_in, list(range(len(t_in))), w_in, V0=0.0, b0=0.0, T_end=T)
            re = [o + d for o in ev['outputs'] if o + d < T] if wj else []
            new_t = sorted([t_ext] + re)
            if new_t == t_in: break
            t_in = new_t; w_in = [w_ext if abs(t - t_ext) < 1e-12 else wj for t in t_in]
        ts = np.linspace(0, T, 4000); I = np.zeros_like(ts)
        for t, w in zip(t_in, w_in): I += w*np.exp(-2*(ts - t))*(ts >= t)
        return ev['outputs'], t_in, ts, I

    outs1, tin1, ts, I1 = run(wj)
    outs0, tin0, _, I0 = run(0.0)
    print('with w_j=12: outputs', np.round(outs1, 3), ' reset currents', np.round([np.interp(o, ts, I1) for o in outs1], 3))
    print('with w_j=0 : outputs', np.round(outs0, 3))

    fig, ax = plt.subplots(figsize=(WIDTH*0.92, 1.9))
    ax.axhline(PI**2, color=GREY, lw=0.8, ls='--'); ax.text(T + 0.05, PI**2, r'$\pi^2$', va='center', fontsize=9, color='0.35')
    ax.plot(ts, I0, color=GREY, lw=1.4)
    ax.plot(ts, I1, color=BLACK, lw=1.2)
    for o in outs1: ax.plot([o, o], [17.2, 18.6], color=BLACK, lw=1.4)
    for o in outs0: ax.plot([o, o], [15.2, 16.6], color=GREY, lw=1.4)
    for t in tin1[1:]: ax.plot(t, 13.9, 'v', color=VERM, ms=4)
    ax.plot(t_ext, 13.9, 'v', color=BLUE, ms=4)
    ax.text(T + 0.05, 17.9, 'outputs, $w_j=12$', va='center', fontsize=8)
    ax.text(T + 0.05, 15.9, 'outputs, $w_j=0$', va='center', fontsize=8, color='0.4')
    ax.text(T + 0.05, 13.9, 'inputs', va='center', fontsize=8)
    ax.text(6.0, 3.9, '$w_j=12$', fontsize=9, ha='center')
    ax.text(2.2, 0.9, '$w_j=0$', fontsize=9, color='0.4')
    ax.set_xlim(0, T); ax.set_ylim(0, 19); ax.set_yticks([0, 5, 10]); ax.set_xlabel('time $t$'); ax.set_ylabel('current $I(t)$')
    save(fig, 'r1_train', output_dir)


def generate_qr_train(output_dir=OUTPUT_DIR):
    """Fig. (App. G): along the outputs after one input, QR exponents go to 0 while the gradient magnitude grows.
    Case 1b: one input of weight w at t=0.1 from V0=b0=0, chosen so that c2/pi = 5.5, 10.5, 20.5 (k = 6, 11, 21 outputs).
    QR (Benettin) on the product of the out->out event Jacobians, QR after every map, Q0 = identity.
    Magnitude: time row of the RTRL trace for V0 and for w, relative to the first output."""
    plt = configure_style()
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.2))
    ax[0].axhline(0, color=GREY, lw=0.8, zorder=0)
    for col, target in zip([BLUE, ORANGE, GREEN], [5.5, 10.5, 20.5]):
        w = (target*PI + PI/2)**2
        ev = simulate([0.1], [0], [w], T_end=200.0)
        J_out = [J for J, k in zip(ev['J'], ev['kind']) if k.endswith('out')][1:]
        logs = qr_accumulate(J_out); i = np.arange(1, len(logs)+1)
        g = groups(ev, [0.1], 200.0)[0]; c2, k = g['c2'], g['n_out']; idx = np.arange(1, k+1); c = c2-(idx-1)*PI
        ax[0].plot(i+1, logs[:, 0]/i, '-', color=col)
        ax[0].plot(i+1, logs[:, 1]/i, '--', color=col)
        ax[0].text(k+0.4, logs[-1, 1]/i[-1], f'$k={k}$', color=col, va='center', fontsize=8)
        MV = np.array([M[0, 0] for M, kk in zip(ev['M'], ev['kind']) if kk.endswith('out')])
        Mw = np.array([M[0, 1] for M, kk in zip(ev['M'], ev['kind']) if kk.endswith('out')])
        ax[1].semilogy(idx, np.abs(Mw/Mw[0]), '-', color=col)
        ax[1].semilogy(idx, np.abs(MV/MV[0]), '--', color=col)
        print('c2/pi=%.2f k=%d | QR exponents per output %s, predicted (0, %.4f) | last-output amplification V0 %.1f (c2/c_{k+1}=%.1f >= k), w %.1f (=(2k-1)c2/c_{k+1}=%.1f)'
              % (c2/PI, k, np.round(logs[-1]/i[-1], 4), np.log(c[-1]/c2)/(k-1), MV[-1]/MV[0], c2/c[-1], Mw[-1]/Mw[0], (2*k-1)*c2/c[-1]))
    ax[0].set_xlabel('output $i$'); ax[0].set_ylabel('QR exponent per output')
    ax[0].text(14, 0.025, r'$\lambda_1=0$ (solid)', fontsize=8); ax[0].text(12.5, -0.42, r'$\lambda_2$ (dashed)', fontsize=8)
    ax[0].set_title('(a) exponents tend to 0'); ax[0].set_ylim(-0.52, 0.08); ax[0].set_xlim(1.5, 24)
    ax[1].set_xlabel('output $i$'); ax[1].set_ylabel(r'$|\partial t_i/\partial\theta\,/\,\partial t_1/\partial\theta|$')
    ax[1].text(16, 2.2e2, r'$\theta=w$ (solid)', fontsize=8); ax[1].text(14.5, 2.0, r'$\theta=V_0$ (dashed)', fontsize=8)
    ax[1].set_title('(b) magnitude grows')
    fig.tight_layout(w_pad=1.5); save(fig, 'qr_train', output_dir)


def generate_capacity(output_dir=OUTPUT_DIR):
    """Fig. (App. G): the number of learnable outputs is set by the current at the last input of each group and by the gap.
    Poisson inputs: n input times uniform on [0,T] (Poisson conditioned on n), one synapse of weight W=250, V0=b0=0, 300 draws per (n,T).
    (a) outputs in each gap against x = sqrt(B)(1-e^{-G})/pi, with the exact count of Prop. 4.
    (b) outputs in the trial against T, median and 10-90% band.
    (c) c2/pi at the last input of each group, T=4."""
    plt = configure_style()
    W, N_LIST, T_LIST, DRAWS, SEED_IN = 250.0, [1, 2, 4, 8], [1, 2, 4, 8, 16], 300, 11
    cols = [BLUE, ORANGE, GREEN, VERM]
    rng = np.random.default_rng(SEED_IN)
    c2_by_n = {n: [] for n in N_LIST}; cap = {n: [] for n in N_LIST}; gap = []
    for n in N_LIST:
        for T in T_LIST:
            tot = []
            for _ in range(DRAWS):
                t = np.sort(rng.uniform(0, T, n)); ev = simulate(t, np.zeros(n, int), [W], T_end=T)
                tot.append(len(ev['outputs']))
                for g in groups(ev, t, T):
                    gap.append((g['n_out'], g['c2'], g['B'], g['G']))
                    if T == 4: c2_by_n[n].append(g['c2']/PI)
            cap[n].append(np.percentile(tot, [10, 50, 90]))
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.1), gridspec_kw={'width_ratios': [1, 1.1, 1]})
    gd = np.array(gap); n_out, c2, B, G = gd.T
    x = np.sqrt(B)*(1-np.exp(-G))/PI; pred = np.maximum(0, np.ceil((c2-np.sqrt(B)*np.exp(-G))/PI))
    ax[0].plot(x, n_out, '.', ms=2, color=BLACK, alpha=0.25)
    xx = np.linspace(0, x.max(), 400); ax[0].plot(xx, np.ceil(xx), color=VERM, lw=0.9); ax[0].plot(xx, np.ceil(xx)-1, color=VERM, lw=0.9, ls='--')
    ax[0].set_xlabel(r'$x=\sqrt{B}\,(1-e^{-G})/\pi$'); ax[0].set_ylabel('outputs in the gap'); ax[0].set_title('(a) exact gap count'); ax[0].set_yticks([0, 2, 4, 6, 8, 10])
    print('(a) groups %d, mismatches with the exact count %d, outside {ceil(x)-1, ceil(x)} %d' % (len(gd), int(np.sum(pred != n_out)), int(np.sum((n_out > np.ceil(x)) | (n_out < np.ceil(x)-1)))))
    for col, n in zip(cols, N_LIST):
        q = np.array(cap[n]); ax[1].plot(T_LIST, q[:, 1], 'o-', ms=3, color=col); ax[1].fill_between(T_LIST, q[:, 0], q[:, 2], color=col, alpha=0.15)
        ax[1].text(17.5, q[-1, 1], f'$n={n}$', color=col, va='center', fontsize=8)
        print('(b) n=%d median outputs over T=%s: %s' % (n, T_LIST, q[:, 1]))
    ax[1].set_xscale('log', base=2); ax[1].set_xticks(T_LIST, [str(t) for t in T_LIST]); ax[1].set_xlim(0.9, 30)
    ax[1].set_xlabel('trial length $T$'); ax[1].set_ylabel('outputs in the trial'); ax[1].set_title('(b) outputs vs. $T$')
    bins = np.linspace(4, 9, 41)
    for col, n in zip(cols, N_LIST):
        ax[2].hist(c2_by_n[n], bins=bins, histtype='step', color=col, density=True, lw=1.1)
        v = np.array(c2_by_n[n]); print('(c) n=%d c2/pi median %.2f 10-90%% [%.2f, %.2f]' % (n, np.median(v), *np.percentile(v, [10, 90])))
    ax[2].set_xlabel(r'$c_2/\pi$ at the last input'); ax[2].set_ylabel('density'); ax[2].set_yticks([]); ax[2].set_title('(c) spread of $c_2$')
    ax[2].text(4.75, 7.0, '$n=1$', color=BLUE, fontsize=8); ax[2].text(6.6, 0.9, '$n=8$', color=VERM, fontsize=8)
    fig.tight_layout(w_pad=1.0); save(fig, 'capacity', output_dir)


def generate_qr_seeds(output_dir=OUTPUT_DIR):
    """Fig. (App. G): which sources of variation the QR exponents forget.
    Baseline trial: Poisson inputs at rate 1 on [0,60] over l=2 synapses, weights U(150,350), V0~U(-0.5,0.5), b0~U(0,3).
    Four seeds, varied one at a time with the other three at the baseline (as in Toolkit 2, Sec. 6):
    input times (seed 1 / 101..119), weights (2 / 201..219), initial state (3 / 301..319), initial QR basis (4 / 401..419); 20 runs each.
    QR after every event, no events discarded, exponents per event (natural log).
    (a) sample SD (ddof=1) of lambda_1 over the 20 runs against the number of events.  (b) mean +/- SD of lambda_1 at 200 events."""
    plt = configure_style()
    RATE, T, L, KMAX = 1.0, 60.0, 2, 170
    def run(seed_in, seed_w, seed_ic, seed_ons):
        ri = np.random.default_rng(seed_in); n = ri.poisson(RATE*T); t = np.sort(ri.uniform(0, T, n)); syn = ri.integers(0, L, n)
        w = np.random.default_rng(seed_w).uniform(150, 350, L)
        ic = np.random.default_rng(seed_ic); V0, b0 = ic.uniform(-0.5, 0.5), ic.uniform(0, 3)
        ev = simulate(t, syn, w, V0=V0, b0=b0, T_end=T)
        h = qr_accumulate(ev['J'], initial_basis(seed_ons)); K = np.arange(1, len(h)+1)
        return h/K[:, None]
    base = (1, 2, 3, 4)
    groups = [('input times', 0, [1]+list(range(101, 120)), BLUE), ('weights', 1, [2]+list(range(201, 220)), ORANGE),
              ('initial state', 2, [3]+list(range(301, 320)), GREEN), ('QR basis', 3, [4]+list(range(401, 420)), VERM)]
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.1), gridspec_kw={'width_ratios': [1.3, 1]})
    K = np.arange(1, KMAX+1); ends = []
    print('source          events   lambda_1 mean +/- SD      lambda_2 mean +/- SD')
    for j, (name, pos, seeds, col) in enumerate(groups):
        runs = []
        for sd in seeds:
            sv = list(base); sv[pos] = sd; runs.append(run(*sv)[:KMAX])
        R = np.array(runs)                       # (runs, KMAX, 2)
        sdev = R[:, :, 0].std(axis=0, ddof=1)
        ax[0].loglog(K[4:], sdev[4:], color=col); ends.append((sdev[-1], name, col))
        m, s = R[:, -1, 0].mean(), R[:, -1, 0].std(ddof=1)
        ax[1].errorbar(j, m, yerr=s, fmt='o', color=col, capsize=3)
        for c in [10, 50, KMAX]:
            print('%-14s  %4d     %.4f +/- %.4f        %.4f +/- %.4f' % (name, c, R[:, c-1, 0].mean(), R[:, c-1, 0].std(ddof=1), R[:, c-1, 1].mean(), R[:, c-1, 1].std(ddof=1)))
    ax[0].loglog(K[4:], 1.0/K[4:], color=GREY, ls=':', lw=1)
    ends.sort(); ys = np.geomspace(min(e[0] for e in ends)*0.8, max(e[0] for e in ends)*1.6, len(ends))
    for y, (v, name, col) in zip(ys, ends): ax[0].text(KMAX*1.1, y, name, color=col, va='center', fontsize=8)
    ax[0].set_xlim(5, KMAX*4.5); ax[0].set_xlabel('events $K$'); ax[0].set_ylabel(r'SD of $\lambda_1$ over 20 runs')
    ax[0].set_title('(a) initial state and basis are forgotten')
    ax[1].set_xticks(range(4), ['inputs', 'weights', 'initial\nstate', 'QR\nbasis'], fontsize=8)
    ax[1].set_ylabel(r'$\lambda_1$ per event, $K=%d$' % KMAX); ax[1].set_title(r'(b) mean $\pm$ SD')
    fig.tight_layout(w_pad=1.0); save(fig, 'qr_seeds', output_dir)


def generate_random_weights(output_dir=OUTPUT_DIR):
    """Fig. (App. F): with random weights the amplification is usually modest, but it has a heavy tail at the capacity boundary.
    l=100 synapses, one input each at a time uniform on [0,1], weights w_i ~ N(0, g^2/sqrt(l)) (g=1 is variance 1/sqrt(l)),
    V0=b0=0, 1000 draws per gain g in {1,10,30,100,300}, trial T=40 so trains run out.
    (a) distribution of the number of outputs.
    (b) amplification at the last output, voltage direction c2/c_{k+1} (exact), against u = c2/pi-(k-1), the place of c2 in its band.
    (c) survival P(amplification > A), voltage and weight directions, all g>=10 pooled; dashed: only outputs inside a trial of length 4."""
    plt = configure_style()
    L, GAINS, DRAWS, T, T_SHORT, SEED = 100, [1, 10, 30, 100, 300], 1000, 40.0, 4.0, 21
    rng = np.random.default_rng(SEED)
    cols = [GREY, BLUE, ORANGE, GREEN, VERM]
    kdist = {}; U, AV, KK = [], [], []; AW = []; AVs, AWs = [], []
    for g in GAINS:
        ks = []
        for _ in range(DRAWS):
            t = np.sort(rng.uniform(0, 1, L)); w = rng.normal(0, g/L**0.25, L)
            ev = simulate(t, np.arange(L), w, V0=0.0, b0=0.0, T_end=T)
            ks.append(len(ev['outputs']))
            gl = groups(ev, t, T)[-1]; k = gl['n_out']
            if k >= 2 and g >= 10:
                e = gl['event']; Mo = [ev['M'][j][0, 1:] for j in range(e+1, e+1+k)]
                to = [ev['time'][j] for j in range(e+1, e+1+k)]
                c2 = gl['c2']; U.append(c2/PI-(k-1)); KK.append(k)
                AV.append(c2/(c2-(k-1)*PI)); AW.append(np.linalg.norm(Mo[-1])/np.linalg.norm(Mo[0]))
                ks_in = sum(1 for x in to if x < T_SHORT)                      # outputs inside the short trial
                if ks_in >= 2:
                    AVs.append(c2/(c2-(ks_in-1)*PI)); AWs.append(np.linalg.norm(Mo[ks_in-1])/np.linalg.norm(Mo[0]))
        kdist[g] = np.bincount(ks, minlength=20)[:20]/DRAWS
        print('g=%3d: P(k=0)=%.2f P(k=1)=%.2f P(k>=2)=%.2f  median k=%d max k=%d' % (g, kdist[g][0], kdist[g][1], 1-kdist[g][:2].sum(), np.median(ks), max(ks)))
    U, AV, AW, KK = map(np.array, (U, AV, AW, KK)); AVs, AWs = np.array(AVs), np.array(AWs)
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.1))
    med = [np.average(np.arange(20), weights=kdist[g]) for g in GAINS]
    p2 = [1-kdist[g][:2].sum() for g in GAINS]
    ax[0].semilogx(GAINS, med, 'o-', color=BLACK, ms=3); ax[0].set_ylabel('mean outputs $k$')
    a0 = ax[0].twinx(); a0.spines['right'].set_visible(True); a0.semilogx(GAINS, p2, 's--', color=VERM, ms=3); a0.set_ylim(0, 1)
    a0.set_ylabel('$P(k\\geq2)$', color=VERM); a0.tick_params(axis='y', colors=VERM, labelsize=7)
    ax[0].set_xlabel('gain $g$'); ax[0].set_title('(a) $g=1$: $k\\leq1$')
    ax[1].scatter(U, AV, s=2, color=GREY, alpha=0.5, lw=0)
    uu = np.linspace(0.004, 1, 200)
    for k, col in [(2, BLUE), (6, ORANGE), (12, GREEN)]:
        ax[1].plot(uu, (k-1+uu)/uu, color=col, lw=1); ax[1].text(1.03, k, f'$k={k}$', color=col, fontsize=8, va='center')
    ax[1].set_yscale('log'); ax[1].set_xlim(0, 1.25); ax[1].set_xlabel(r'$u=c_2/\pi-(k-1)$'); ax[1].set_ylabel(r'$c_2/c_{k+1}$, last output')
    ax[1].set_title('(b) blow-up as $u\\to0$')
    for arr, col, ls, lab in [(AV, BLUE, '-', 'voltage'), (AW, ORANGE, '-', 'weights'), (AVs, BLUE, ':', None), (AWs, ORANGE, ':', None)]:
        s = np.sort(arr); p = 1 - np.arange(len(s))/len(s)
        ax[2].loglog(s, p, ls, color=col, lw=1.2)
    A = np.geomspace(5, 3e3, 50); ax[2].loglog(A, 4/A, color=GREY, lw=0.8, ls='--')
    ax[2].text(1e3, 2.2e-2, r'$\propto 1/A$', color='0.4', fontsize=8)
    ax[2].text(150, 0.5, 'voltage', color=BLUE, fontsize=8); ax[2].text(150, 0.18, 'weights', color=ORANGE, fontsize=8); ax[2].text(150, 0.065, 'dotted: $T=4$', color='0.3', fontsize=8)
    ax[2].set_xlabel('amplification $A$'); ax[2].set_ylabel(r'$P(\mathrm{amplification}>A)$'); ax[2].set_title('(c) heavy tail')
    print('trains with k>=2 (g>=10): %d; voltage amplification median %.1f, p90 %.1f, p99 %.1f, max %.1f' % (len(AV), np.median(AV), *np.percentile(AV, [90, 99]), AV.max()))
    print('                          weights amplification median %.1f, p90 %.1f, p99 %.1f, max %.1f' % (np.median(AW), *np.percentile(AW, [90, 99]), AW.max()))
    print('inside a trial of length %g: trains %d; voltage max %.1f, weights max %.1f' % (T_SHORT, len(AVs), AVs.max(), AWs.max()))
    fig.tight_layout(w_pad=0.6); save(fig, 'random_weights', output_dir)


def generate_pseudospike(output_dir=OUTPUT_DIR, *, recompute=False):
    """Fig. 5: a pseudospike added after the trial carries a gradient that decays like e^{-Delta T}.
    Numerical probe, outside (A4). Uses Klos & Memmesheimer's spikegd (https://github.com/chklos/spikegd) for the simulation, the
    pseudospike time and the gradients (JAX autodiff, float64):
      one input at t=0.1 from V0=b0=0 with c2/pi = K-1/2, so the train has K outputs and then stops;
      the trial ends Delta T after the last output;
      the pseudospike is spikegd's QIFNeuron.t_pseudo: t_ps = T + theta(I_ps) - phi(V(T), I_ps), I_ps = softplus_alpha(I(T)),
      i.e. after T the neuron runs on a constant suprathreshold current set by its current at T (their first type of pseudodynamics).
    Plotted: |d t_ps/d w| / |d t_K/d w| against Delta T.  (a) K=5, softplus scale alpha in {1, 10, 100} (alpha=10 is spikegd's default).
    (b) alpha=10, K in {2, 5, 10}.  Dashed: e^{-Delta T}.
    Recomputing the cache requires JAX and spikegd."""
    plt = configure_style()
    def ratio(K, dT, alpha):
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        from spikegd.qif import QIFNeuron
        neuron = QIFNeuron(tau=1.0, eps=1e-10, alpha=alpha)
        w0 = ((K-0.5)*np.pi + np.pi/2)**2; tin = 0.1
        def sim(w, T):
            return neuron.event(jnp.zeros((2, 1)), jnp.zeros((1, 1)), jnp.array([[w]]), (jnp.array([tin]), jnp.array([0])), {'T': T, 'K': 4*K+10})
        out = sim(w0, 200.0); ts, isin, js, xs = out
        tK = float(np.array(ts)[(~np.array(isin)) & (np.array(js) == 0)][-1]); T = tK + dT
        def last_out(w):
            ts, isin, js, xs = sim(w, T); m = (~isin) & (js == 0)
            return jnp.max(jnp.where(m, ts, -jnp.inf))
        def t_ps(w):
            ts, isin, js, xs = sim(w, T); xe = xs[jnp.sum(js >= 0)]
            return neuron.t_pseudo(xe[:, 0], 0.0, 1, {'T': T})
        return abs(float(jax.grad(t_ps)(w0))/float(jax.grad(last_out)(w0)))

    DT = np.array([0.02, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6])
    CACHE = Path(output_dir) / 'data' / 'pseudospike_data.npz'                 # delete it (or pass --recompute) to rerun the simulations
    if CACHE.exists() and not recompute:
        with np.load(CACHE) as cached:
            data = dict(cached)
    else:
        data = {f'a{al:g}': np.array([ratio(5, d, al) for d in DT]) for al in [1.0, 10.0, 100.0]}
        data.update({f'K{K}': np.array([ratio(K, d, 10.0) for d in DT]) for K in [2, 5, 10]})
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, **data)
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.1), sharey=True)
    for col, al in zip([BLUE, ORANGE, GREEN], [1.0, 10.0, 100.0]):
        r = data[f'a{al:g}']; ax[0].semilogy(DT, r, 'o-', ms=3, color=col)
        ax[0].text(6.2, r[-1], rf'$\alpha={al:g}$', color=col, va='center', fontsize=8)
        print('K=5 alpha=%5.1f: ratio at dT=0.02 %.3f, at dT=6 %.2e, decay per unit time over [4,6] %.3f' % (al, r[0], r[-1], (r[-1]/r[-3])**0.5))
    for col, K in zip([BLUE, ORANGE, GREEN], [2, 5, 10]):
        r = data[f'K{K}']; ax[1].semilogy(DT, r, 'o-', ms=3, color=col)
        print('alpha=10 K=%d: ratio at dT=0.02 %.3f, at dT=6 %.2e' % (K, r[0], r[-1]))
    for a in ax:
        a.semilogy(DT, np.exp(-DT), color=GREY, ls='--', lw=0.9); a.set_xlim(0, 7.6); a.set_xlabel(r'silence before the trial end, $\Delta T$')
    ax[1].text(0.3, 2e-4, r'dashed: $e^{-\Delta T}$', color='0.4', fontsize=8)
    ax[1].text(3.0, 0.15, '$K=2,5,10$ lie\non top of each other', fontsize=8)
    ax[0].set_ylabel(r'$|\partial t_{\mathrm{ps}}/\partial w|\,/\,|\partial t_K/\partial w|$')
    ax[0].set_title(r'(a) $K=5$, three drive scales $\alpha$'); ax[1].set_title(r'(b) $\alpha=10$: no dependence on $K$')
    fig.tight_layout(w_pad=1.0); save(fig, 'pseudospike', output_dir)


FIGURES = {
    "r1_train": generate_r1_train,
    "qr_train": generate_qr_train,
    "capacity": generate_capacity,
    "qr_seeds": generate_qr_seeds,
    "random_weights": generate_random_weights,
    "pseudospike": generate_pseudospike,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*", metavar="FIGURE",
                        help="Figures to generate (default: all): " + ", ".join(FIGURES))
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute the pseudospike cache using JAX and spikegd")
    args = parser.parse_args(argv)
    for name in args.figures:
        if name not in FIGURES:
            parser.error(f"unknown figure {name!r}; choose from: {', '.join(FIGURES)}")
    for name in args.figures or FIGURES:
        kwargs = {"recompute": args.recompute} if name == "pseudospike" else {}
        with capture_results(f"FIGURE_{name.upper()}", f"Figure: {name}"):
            FIGURES[name](OUTPUT_DIR, **kwargs)


if __name__ == "__main__":
    main()
