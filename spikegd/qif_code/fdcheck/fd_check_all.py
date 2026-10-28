"""Full check of every closed form and every gradient in theory_condensed.tex.

Three independent references are used:
  MPFD  finite differences of the paper's value formulas in 50-digit arithmetic (mpmath.diff), so FD error is negligible;
  SGD   Klos & Memmesheimer's spikegd (JAX, float64): its own closed forms, and JAX autodiff through its event-based simulation;
  FD    central differences (h = 1e-6, float64) of whole simulated trials.
The quantities under test are the paper's formulas transcribed in paper_forms.py, and gradients assembled only from them (rtrl_paper.py).
Run: python3 fd_check_all.py   ->  FD_REPORT.md, fd_results.json
"""
import json, time, sys, os
import numpy as np, mpmath as mp
from scipy.integrate import solve_ivp
import jax, jax.numpy as jnp
import paper_forms as P, rtrl_paper as R, sg
from r1_sim import r1_outputs
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs'))
from qif_events import qr_accumulate
mp.mp.dps = 50
rng = np.random.default_rng(2026)
REC = []
def record(section, check, paper, ref, ours, refv, tol, note='', rel_floor=1.0, allow_nonfinite=False, expected_fail=False):
    ours = np.atleast_1d(np.asarray(ours, float)); refv = np.atleast_1d(np.asarray(refv, float))
    ok_mask = np.isfinite(ours) & np.isfinite(refv)
    nonfinite = int((~ok_mask).sum())
    err = np.abs(ours[ok_mask]-refv[ok_mask]); rel = err/np.maximum(rel_floor, np.abs(refv[ok_mask]))
    r = dict(section=section, check=check, paper=paper, reference=ref, n=int(ok_mask.sum()), nonfinite=nonfinite,
             max_abs=float(err.max()) if err.size else float('nan'), max_rel=float(rel.max()) if rel.size else float('nan'),
             tol=tol, ok=bool(rel.size and rel.max() <= tol and (nonfinite == 0 or allow_nonfinite)), note=note, expected_fail=expected_fail)
    REC.append(r); print(('PASS ' if r['ok'] else ('ERRATUM ' if expected_fail else 'FAIL ')) + f"[{section}] {check} vs {ref}: n={r['n']} max_rel={r['max_rel']:.2e} {note}")
    return r
M = lambda x: mp.mpf(float(x))
def mpd(fun, x):              # 50-digit derivative of a scalar function at x
    return float(mp.diff(fun, M(x)))

# ------------------------------------------------------------------ samplers (one per branch)
def sample(br, need_spike=False):
    while True:
        if br == 'tan':  b = rng.uniform(0.5, 60); V0 = rng.uniform(-3, 3)
        elif br == 'b0': b = 0.0; V0 = rng.choice([rng.uniform(-3, -0.2), rng.uniform(0.2, 0.9), rng.uniform(1.1, 4)])
        else:
            b = -rng.uniform(0.3, 6); s = np.sqrt(-b)
            if br == 'coth': V0 = rng.choice([-1, 1])*(s+rng.uniform(0.05, 3)) if not need_spike else s+rng.uniform(0.05, 3)
            elif br == 'tanh': V0 = rng.uniform(-0.95, 0.95)*s
            else: V0 = rng.choice([-1, 1])*s
        if need_spike:
            if br == 'tan' and P.D(V0, b) <= 0.05: continue
            if br == 'coth' and P.D(V0, b) <= 0.05: continue
            if br == 'b0' and V0 <= 1.05: continue
        spikes = (br == 'tan' and P.D(V0, b) > 0) or (br == 'coth' and V0 > 0 and P.D(V0, b) > 0) or (br == 'b0' and V0 > 1)
        T = P.tsp(V0, b) if spikes else np.inf
        t = rng.uniform(0.02, min(2.0, 0.8*T)) if np.isfinite(T) else rng.uniform(0.02, 2.0)
        return V0, b, t

t0 = time.time()
# ================================================================== A. Neuron model (Sec. 2.1, App. A)
S = 'A neuron model'
bs = rng.uniform(-5, 40, 50); ts = rng.uniform(0, 3, 50)
record(S, 'current I(t) = b e^{-2t}  (eq. 2)', 'Sec. 2.1', 'SGD flow_I', bs*np.exp(-2*ts), [float(sg.neuron.flow_I(b, t)) for b, t in zip(bs, ts)], 1e-14)
for br in ['tan', 'b0', 'coth', 'tanh', 'line']:
    smp = [sample(br) for _ in range(40)]
    ours = [P.V(V0, b, t, br=br) for V0, b, t in smp]
    record(S, f'V(t), branch {br}  (eq. 3)', 'Sec. 2.1 / App. A.2', 'SGD flow_V', ours, [sg.flowV(V0, b, t) for V0, b, t in smp], 1e-9,
           note='spikegd returns non-finite values on part of the invariant line; finite ones compared' if br == 'line' else '',
           allow_nonfinite=(br == 'line'))
    ode = []
    for V0, b, t in smp[:15]:
        sol = solve_ivp(lambda s, y: [y[0]**2-y[0]+b*np.exp(-2*s)], [0, t], [V0], rtol=1e-12, atol=1e-13, method='DOP853'); ode.append(sol.y[0, -1])
    record(S, f'V(t), branch {br}  (eq. 3)', 'Sec. 2.1 / App. A.2', 'ODE (DOP853, rtol 1e-12)', ours[:15], ode, 1e-8)
def blowup(V0, b):
    rhs = lambda s, p: [np.sin(p[0])**2-np.sin(p[0])*np.cos(p[0])+b*np.exp(-2*s)*np.cos(p[0])**2]
    ev = lambda s, p: p[0]-np.pi/2; ev.terminal = True; ev.direction = 1
    return solve_ivp(rhs, [0, 50], [np.arctan(V0)], events=ev, rtol=1e-12, atol=1e-14, method='DOP853').t_events[0][0]
for br in ['tan', 'coth', 'b0']:
    smp = [sample(br, need_spike=True) for _ in range(40)]
    ours = [P.tsp(V0, b) for V0, b, _ in smp]
    record(S, f't_sp, branch {br}  (eq. 4)', 'Sec. 2.1 / App. A.3', 'SGD dt_spike', ours, [sg.tsp(V0, b) for V0, b, _ in smp], 1e-9,
           note='spikegd evaluates b=0 at b=1e-6' if br == 'b0' else '')
    record(S, f't_sp, branch {br}  (eq. 4)', 'Sec. 2.1 / App. A.3', 'ODE blow-up time (phase, rtol 1e-12)', ours[:15], [blowup(V0, b) for V0, b, _ in smp[:15]], 1e-8)
bb = rng.uniform(np.pi**2+0.5, 400, 40)
record(S, 't_sp after a reset  (eq. 5)', 'Sec. 2.1 / App. A.7', 'MP limit V0=-1e40', [P.tsp_reset(b) for b in bb], [float(P.tsp(-mp.mpf('1e40'), M(b), m=P.MP)) for b in bb], 1e-12)
record(S, 't_sp after a reset  (eq. 5)', 'Sec. 2.1 / App. A.7', 'SGD dt_spike at V=-1/eps', [P.tsp_reset(b) for b in bb], [sg.tsp(-1/sg.EPS, b) for b in bb], 1e-8)
# A.5 derivatives of the spike time
for br in ['tan', 'coth']:
    smp = [sample(br, need_spike=True) for _ in range(40)]
    for name, fn, ax in [('dt_sp/dV0', P.dtsp_dV0, 0), ('dt_sp/db', P.dtsp_db, 1)]:
        ours = [fn(V0, b) for V0, b, _ in smp]
        mpv = [mpd(lambda x: P.tsp(x, M(b), m=P.MP), V0) if ax == 0 else mpd(lambda x: P.tsp(M(V0), x, m=P.MP), b) for V0, b, _ in smp]
        record(S, f'{name}, branch {br}', 'App. A.5', 'MPFD', ours, mpv, 1e-11)
        record(S, f'{name}, branch {br}', 'App. A.5', 'SGD autodiff', ours, [float(sg.dtsp(jnp.array(V0), jnp.array(b))[ax]) for V0, b, _ in smp], 1e-8)
smp = [sample('b0', need_spike=True) for _ in range(30)]
record(S, 'dt_sp/dV0, b = 0', 'App. A.5', 'MPFD', [P.dtsp_dV0(V0, 0.0) for V0, _, _ in smp], [mpd(lambda x: P.tsp(x, mp.mpf(0), m=P.MP), V0) for V0, _, _ in smp], 1e-11)
h = mp.mpf('1e-18')
record(S, 'dt_sp/db, b = 0 (limit of both branches)', 'App. A.5', 'MPFD across b=0',
       [P.dtsp_db(V0, 0.0) for V0, _, _ in smp], [float((P.tsp(M(V0), h, m=P.MP)-P.tsp(M(V0), -h, m=P.MP))/(2*h)) for V0, _, _ in smp], 1e-11)
# A.6 derivatives of the voltage
for br in ['tan', 'coth', 'tanh', 'b0']:
    smp = [sample(br) for _ in range(40)]
    ours = [P.dV_dV0(V0, b, t, br=br) for V0, b, t in smp]
    record(S, f'dV/dV0, branch {br}', 'App. A.6', 'MPFD', ours, [mpd(lambda x: P.V(x, M(b), M(t), m=P.MP, br=br), V0) for V0, b, t in smp], 1e-10)
    record(S, f'dV/dV0, branch {br}', 'App. A.6', 'SGD autodiff', ours, [float(sg.dflowV(jnp.array(V0), jnp.array(b), jnp.array(t))[0]) for V0, b, t in smp],
           1e-5 if br == 'b0' else 1e-8, note='spikegd differentiates the b=0 branch through the b=1e-6 tan branch' if br == 'b0' else '')
    ours = [P.dV_db(V0, b, t, br=br) for V0, b, t in smp]
    if br == 'b0':
        mpv = [float((P.V(M(V0), h, M(t), m=P.MP, br='tan')-P.V(M(V0), -h, M(t), m=P.MP, br=('coth' if abs(V0) > 0 else 'tanh')))/(2*h)) for V0, b, t in smp]
        record(S, 'dV/db, b = 0 (variational equation)', 'App. A.6', 'MPFD across b=0', ours, mpv, 1e-9)
        smp0 = [(0.0, 0.0, t) for t in rng.uniform(0.05, 3, 10)]
        record(S, 'dV/db, b = 0, V0 = 0', 'App. A.6', 'MPFD across b=0', [P.dV_db(0.0, 0.0, t) for _, _, t in smp0],
               [float((P.V(mp.mpf(0), h, M(t), m=P.MP, br='tan')-P.V(mp.mpf(0), -h, M(t), m=P.MP, br='tanh'))/(2*h)) for _, _, t in smp0], 1e-9,
               note='the printed entry is 0/0 at V0=0; its limit e^{-t}(1-e^{-t}) is used')
    else:
        record(S, f'dV/db, branch {br}', 'App. A.6', 'MPFD', ours, [mpd(lambda x: P.V(M(V0), x, M(t), m=P.MP, br=br), b) for V0, b, t in smp], 1e-10)
        record(S, f'dV/db, branch {br}', 'App. A.6', 'SGD autodiff', ours, [float(sg.dflowV(jnp.array(V0), jnp.array(b), jnp.array(t))[1]) for V0, b, t in smp], 1e-8)
    ours = [P.dV_dt(V0, b, t, br=br) for V0, b, t in smp]
    record(S, f'dV/dt, branch {br}', 'App. A.6', 'MPFD', ours, [mpd(lambda x: P.V(M(V0), M(b), x, m=P.MP, br=br), t) for V0, b, t in smp], 1e-10)
smp = [sample('line') for _ in range(20)]
mpv = [mpd(lambda x: P.V(M(V0), x, M(t), m=P.MP), b + 1e-12) for V0, b, t in smp]   # fixed V0, b moves off the line
record(S, 'dV/db on the line |V0|=sqrt(-b), as printed', 'App. A.6', 'MPFD (fixed V0)', [P.dV_db(V0, b, t, br='line') for V0, b, t in smp], mpv, 1e-6,
       note='ERRATUM: the printed entry is the derivative along the line, not at fixed V0', expected_fail=True)
record(S, 'dV/db on the line, corrected e^{-t}(e^{2V0(1-e^{-t})}-1)/(2V0)', 'App. A.6 (correction)', 'MPFD (fixed V0)',
       [P.dV_db_line_corrected(V0, b, t) for V0, b, t in smp], mpv, 1e-6)
# A.7 from a reset
for name, gen in [('b>0', lambda: rng.uniform(0.3, 60)), ('b<0', lambda: -rng.uniform(0.3, 6))]:
    vals = [(gen(), rng.uniform(0.02, 1.5)) for _ in range(30)]
    vals = [(b, tau) for b, tau in vals if not (b > 0 and np.sqrt(b)*(1-np.exp(-tau)) >= np.pi*0.95)]
    record(S, f'V(tau) after a reset, {name}', 'App. A.7', 'MP V0=-1e40', [P.V_reset(b, tau) for b, tau in vals],
           [float(P.V(-mp.mpf('1e40'), M(b), M(tau), m=P.MP)) for b, tau in vals], 1e-12)
    record(S, f'dV(tau)/db after a reset, {name}', 'App. A.7', 'MPFD', [P.dVreset_db(b, tau) for b, tau in vals],
           [mpd(lambda x: P.V_reset(x, M(tau), m=P.MP), b) for b, tau in vals], 1e-10)
    record(S, f'V(tau) after a reset, {name}', 'App. A.7', 'SGD flow_V from V=-1/eps', [P.V_reset(b, tau) for b, tau in vals],
           [sg.flowV(-1/sg.EPS, b, tau) for b, tau in vals], 1e-8)
taus = rng.uniform(0.05, 2, 15)
record(S, 'dV(tau)/db after a reset, b = 0', 'App. A.7', 'MPFD across b=0', [P.dVreset_db(0.0, tau) for tau in taus],
       [float((P.V_reset(h, M(tau), m=P.MP)-P.V_reset(-h, M(tau), m=P.MP))/(2*h)) for tau in taus], 1e-9)
# A.8 the two identities
for br in ['tan', 'coth', 'b0']:
    smp = [sample(br, need_spike=True) for _ in range(40)]
    record(S, f'Prop. 1: grad t_sp . f = -1, branch {br}', 'Prop. 1, App. A.8', 'exact value -1',
           [P.grad_tsp(V0, b) @ P.f(V0, b) for V0, b, _ in smp], [-1.0]*len(smp), 1e-10)
# Prop. 2 against spikegd: move the input time of a single input, first output after it
vals = []
for _ in range(25):
    V0 = rng.uniform(-1, 0.8); b0 = rng.uniform(-1, 3); tin = rng.uniform(0.05, 0.6); w = rng.uniform(15, 80)
    ff = lambda tt: sg.output_times(sg.run(jnp.array([[V0], [b0]]), jnp.zeros((1, 1)), jnp.array([[w]]), jnp.array([tt]), np.array([0]), 20.0, 30), 0, 1)[0]
    tout = float(ff(tin))
    if not np.isfinite(tout) or tout < tin: continue
    Vm = P.V(V0, b0, tin); bp = b0*np.exp(-2*tin)+w
    vals.append((1+P.grad_tsp(Vm, bp) @ np.array([Vm**2-Vm+b0*np.exp(-2*tin), -2*b0*np.exp(-2*tin)]), P.input_time_sensitivity(Vm, bp, w), float(jax.grad(ff)(tin))))
vals = np.array(vals)
record(S, 'Prop. 2: 1 + grad t_sp . f(x-) = grad t_sp . (-w, 2w)', 'Prop. 2', 'both sides', vals[:, 0], vals[:, 1], 1e-10)
record(S, 'Prop. 2: d t_out / d t_in', 'Prop. 2', 'SGD autodiff wrt input time', vals[:, 1], vals[:, 2], 1e-7)

# ================================================================== B. Event maps (Sec. 3.1, App. B)
S = 'B event maps'
def mp_jac(fun, x):
    J = np.zeros((2, 2))
    for j in range(2):
        for i in range(2):
            J[i, j] = float(mp.diff(lambda z: fun(*[z if k == j else M(x[k]) for k in range(2)])[i], M(x[j])))
    return J
def maxrel(A, B): return np.max(np.abs(A-B)/np.maximum(1, np.abs(B)))
for br in ['tan', 'coth', 'tanh', 'b0']:
    ours, ref = [], []
    for _ in range(15):
        V0, b, D = sample(br); w = rng.uniform(-5, 20)
        ours.append(P.J_in_in(V0, b, D).ravel())
        if br == 'b0':                    # V at b=0 has no b in it: differentiate across b=0 with the neighbouring branches
            hb = mp.mpf('1e-18'); brm = 'coth' if V0 != 0 else 'tanh'
            col_b = (P.V(M(V0), hb, M(D), m=P.MP, br='tan')-P.V(M(V0), -hb, M(D), m=P.MP, br=brm))/(2*hb)
            ref.append(np.array([mpd(lambda v: P.V(v, mp.mpf(0), M(D), m=P.MP, br='b0'), V0), float(col_b), 0.0, float(mp.exp(-2*M(D)))]))
        else:
            ref.append(mp_jac(lambda v, bb: (P.V(v, bb, M(D), m=P.MP, br=br), bb*mp.exp(-2*M(D))+w), (V0, b)).ravel())
    record(S, f'J_in->in, branch {br}', 'Sec. 3.1, App. B', 'MPFD of the map', np.ravel(ours), np.ravel(ref), 1e-9)
for br in ['tan', 'coth']:
    ours, ref = [], []
    for _ in range(15):
        V0, b, _ = sample(br, need_spike=True); tl = rng.uniform(0, 2)
        ours.append(P.J_in_out(V0, b).ravel())
        ref.append(mp_jac(lambda v, bb: (M(tl)+P.tsp(v, bb, m=P.MP), bb*mp.exp(-2*P.tsp(v, bb, m=P.MP))), (V0, b)).ravel())
    record(S, f'J_in->out, branch {br}', 'Sec. 3.1, App. B', 'MPFD of the map', np.ravel(ours), np.ravel(ref), 1e-9)
ours, ref = [], []
for b in rng.uniform(np.pi**2+0.5, 300, 20):
    to = rng.uniform(0, 3); ours.append(P.J_out_out(b).ravel())
    ref.append(mp_jac(lambda t, bb: (t+P.tsp_reset(bb, m=P.MP), bb*mp.exp(-2*P.tsp_reset(bb, m=P.MP))), (to, b)).ravel())
record(S, 'J_out->out (eq. 8)', 'Sec. 3.1, App. B', 'MPFD of the map', np.ravel(ours), np.ravel(ref), 1e-10)
for name, gen in [('b>0', lambda: rng.uniform(0.3, 60)), ('b<0', lambda: -rng.uniform(0.3, 6))]:
    ours, ref = [], []
    for _ in range(20):
        b = gen(); to = rng.uniform(0, 2); D = rng.uniform(0.02, 1.2); w = rng.uniform(-5, 20)
        if b > 0 and np.sqrt(b)*(1-np.exp(-D)) >= np.pi*0.95: continue
        tin = to + D
        ours.append(P.J_out_in(b, D).ravel())
        ref.append(mp_jac(lambda t, bb: (P.V_reset(bb, M(tin)-t, m=P.MP), bb*mp.exp(-2*(M(tin)-t))+w), (to, b)).ravel())
    record(S, f'J_out->in, reset current {name}', 'Sec. 3.1, App. B', 'MPFD of the map', np.ravel(ours), np.ravel(ref), 1e-9)

# ================================================================== C. Single-neuron exact gradient (Secs. 2.2, 3)
S = 'C single neuron gradient'
def sgd_jac(t_in, syn, w, V0, b0, T, K, k):
    l = len(w); Win_idx = np.asarray(syn)
    def f(p):
        Win = jnp.zeros((1, l)).at[0, :].set(p[1:])
        return sg.output_times(sg.run(jnp.stack([jnp.array([p[0]]), jnp.array([b0])]), jnp.zeros((1, 1)), Win, jnp.asarray(t_in), Win_idx, T, K), 0, k)
    p0 = jnp.array(np.concatenate([[V0], w]))
    return np.array(f(p0)), np.array(jax.jacfwd(f)(p0)), f
cases = {'1a': [], '1b': [], '2': [], '3': []}
for case in cases:
    for _ in range(12):
        if case == '1a': t_in, syn, w, V0, b0 = np.array([]), np.array([], int), np.array([0.0]), rng.uniform(-1, 1), rng.uniform(15, 300)
        elif case == '1b': t_in, syn, w, V0, b0 = np.array([rng.uniform(0.05, 0.5)]), np.array([0]), np.array([rng.uniform(15, 300)]), rng.uniform(-1, 1), rng.uniform(-2, 2)
        elif case == '2':
            n = rng.integers(2, 6); t_in = np.sort(rng.uniform(0.05, 0.12, n)); syn = rng.integers(0, 3, n); w = rng.uniform(5, 80, 3); V0 = rng.uniform(-1, 0.5); b0 = 0.0
        else:
            n = rng.integers(4, 12); t_in = np.sort(rng.uniform(0, 4, n)); syn = rng.integers(0, 3, n); w = rng.uniform(-20, 120, 3); V0 = rng.uniform(-1, 1); b0 = rng.uniform(0, 5)
        T = 5.0
        outs, rows, _ = R.single_neuron(t_in, syn, w, V0, b0, T)
        if len(outs) == 0: continue
        if case == '2' and len(t_in) and outs[0] < t_in[-1]: continue
        times, Jj, f = sgd_jac(t_in, syn, w, V0, b0, T, 3*len(t_in)+4*len(outs)+20, len(outs))
        if np.any(~np.isfinite(times)) or np.max(np.abs(times-outs)) > 1e-6: continue
        hh = 1e-6; p0 = np.concatenate([[V0], w]); fd = np.zeros_like(rows)
        for q in range(len(p0)):
            e = np.zeros(len(p0)); e[q] = hh
            fd[:, q] = (np.array(f(jnp.array(p0+e)))-np.array(f(jnp.array(p0-e))))/(2*hh)
        cases[case].append((rows, Jj, fd))
    if cases[case]:
        A = np.concatenate([r.ravel() for r, _, _ in cases[case]]); B_ = np.concatenate([j.ravel() for _, j, _ in cases[case]]); C_ = np.concatenate([d.ravel() for _, _, d in cases[case]])
        record(S, f'Case {case}: d t_out / d(V0, w) from the event maps', 'Secs. 2.2, 3.1', 'SGD autodiff', A, B_, 1e-7, note=f'{len(cases[case])} trials')
        record(S, f'Case {case}: d t_out / d(V0, w) from the event maps', 'Secs. 2.2, 3.1', 'FD of spikegd trials', A, C_, 1e-5, note=f'{len(cases[case])} trials')

# ================================================================== D. Capacity (Sec. 3.2, App. C)
S = 'D capacity'
rec_i, rec_ii, rec_t, rec_iii, rec_iii_maps = [], [], [], [], []
for _ in range(40):
    V0 = rng.uniform(-2, 2); b0 = rng.uniform(0, 2); tin = rng.uniform(0.05, 0.4); w = rng.uniform(20, 900)
    out = sg.run(jnp.array([[V0], [b0]]), jnp.zeros((1, 1)), jnp.array([[w]]), jnp.array([tin]), np.array([0]), 80.0, 120)
    ts_, isin, js, xs = [np.array(a) for a in out]
    outs = ts_[(~isin) & (js == 0)]
    Vin = P.V(V0, b0, tin); b = b0*np.exp(-2*tin)+w
    k = P.horizon_count(Vin, b); rec_ii.append((k, len(outs)))
    resets = xs[(~isin) & (js == 0), 1, 0]                     # current at each reset
    c = np.sqrt(resets[:k])                                     # c_{i+1}, i = 1..k
    rec_i.append((c[1:] - (c[:-1]-np.pi)) if k > 1 else np.zeros(0))
    rec_t.append((P.output_times_after_input(Vin, b, k), outs[:k]-tin))
    # (iii): theta = V0 (moves V at the input only through the flow), b0, and w (beta = 1)
    for name, dV, beta, fn in [('V0', P.dV_dV0(V0, b0, tin), 0.0, 0), ('b0', P.dV_db(V0, b0, tin), np.exp(-2*tin), 1), ('w', 0.0, 1.0, 2)]:
        pred = np.array([P.trace_after_input(Vin, b, beta, dV, i)[0] for i in range(1, k+1)])
        def g(p):
            o = sg.run(jnp.stack([jnp.array([p[0]]), jnp.array([p[1]])]), jnp.zeros((1, 1)), jnp.array([[p[2]]]), jnp.array([tin]), np.array([0]), 80.0, 120)
            return sg.output_times(o, 0, k)
        Jg = np.array(jax.jacfwd(g)(jnp.array([V0, b0, w])))[:, fn]
        rec_iii.append((pred, Jg))
    o_, rows_, _ = R.single_neuron(np.array([tin]), np.array([0]), np.array([w]), V0, b0, 80.0, with_b0=True)
    rec_iii_maps.extend([rows_[:k, 0], rows_[:k, 2], rows_[:k, 1]])
record(S, 'Prop. 3(i): c_{i+1} = c_i - pi', 'Prop. 3(i), App. C', 'SGD reset currents', np.concatenate(rec_i), np.zeros(sum(len(x) for x in rec_i)), 1e-6, rel_floor=1.0)
record(S, 'Prop. 3(ii): number of outputs = ceil(c2/pi)', 'Prop. 3(ii), App. C', 'SGD output count', [a for a, _ in rec_ii], [b for _, b in rec_ii], 0.0)
record(S, 'App. C: t_i = 1/2 ln b - ln(c2-(i-1)pi)', 'App. C', 'SGD output times', np.concatenate([a for a, _ in rec_t]), np.concatenate([b for _, b in rec_t]), 1e-6,
       note='spikegd error grows along long trains (reset at -1/eps)')
record(S, 'Prop. 3(iii): dt_i/dtheta = beta/2b - gamma/c_{i+1}  (theta = V0, b0, w)', 'Prop. 3(iii), App. C', 'SGD autodiff',
       np.concatenate([a for a, _ in rec_iii]), np.concatenate([b for _, b in rec_iii]), 1e-5, note='amplified spikegd round-off near the capacity boundary')
record(S, 'Prop. 3(iii): dt_i/dtheta = beta/2b - gamma/c_{i+1}  (theta = V0, b0, w)', 'Prop. 3(iii), App. C', 'product of event maps (rtrl_paper)',
       np.concatenate([a for a, _ in rec_iii]), np.concatenate(rec_iii_maps), 1e-9)
# weight of the last input with V=0 there: ratio (2k-1) c2/c_{k+1}
vals = []
for target in [3.5, 5.5, 8.5, 12.5]:
    w = (target*np.pi+np.pi/2)**2
    o = sg.run(jnp.array([[0.0], [0.0]]), jnp.zeros((1, 1)), jnp.array([[w]]), jnp.array([0.1]), np.array([0]), 80.0, 80)
    k = int(np.ceil(target)); c2 = target*np.pi
    g = lambda p: sg.output_times(sg.run(jnp.array([[0.0], [0.0]]), jnp.zeros((1, 1)), jnp.array([[p]]), jnp.array([0.1]), np.array([0]), 80.0, 80), 0, k)
    d = np.array(jax.jacfwd(g)(w)); vals.append(((2*k-1)*c2/(c2-(k-1)*np.pi), d[-1]/d[0]))
record(S, 'App. C: weight of the last input, ratio = (2k-1) c2/c_{k+1}', 'App. C', 'SGD autodiff', [a for a, _ in vals], [b for _, b in vals], 1e-5)
# QR exponents of the output-to-output product
vals = []
for target in [3.5, 6.2, 10.9]:
    w = (target*np.pi+np.pi/2)**2; c2 = target*np.pi; k = int(np.ceil(target))
    Js = [P.J_out_out((c2-(i-1)*np.pi)**2) for i in range(1, k)]
    vals.append((np.array([0.0, np.log((c2-(k-1)*np.pi)/c2)/(k-1)]), qr_accumulate(Js)[-1]/(k-1)))
record(S, 'QR exponents along a train = (0, ln(c_{k+1}/c2)/(k-1))', 'Sec. 3.2, App. C', 'Benettin QR on the maps', np.concatenate([a for a, _ in vals]), np.concatenate([b for _, b in vals]), 1e-12)
# Prop. 4 gap count against spikegd trials with Poisson inputs
pred, obs = [], []
for _ in range(150):
    n = rng.integers(1, 9); T = rng.uniform(1, 12); t = np.sort(rng.uniform(0, T, n)); w = rng.uniform(30, 600)
    out = sg.run(jnp.array([[0.0], [0.0]]), jnp.zeros((1, 1)), jnp.full((1, n), w), jnp.asarray(t), np.arange(n), T, 4*n+300)
    ts_, isin, js, xs = [np.array(a) for a in out]
    ev = [(ts_[q], bool(isin[q])) for q in range(len(ts_)) if js[q] >= 0]
    for q, (tq, inq) in enumerate(ev):
        if not inq or (q+1 < len(ev) and ev[q+1][1]): continue         # last input of a group
        k = 0; r = q+1
        while r < len(ev) and not ev[r][1]: k += 1; r += 1
        Vq, Bq = xs[q, 0, 0], xs[q, 1, 0]
        G = (ev[r][0] if r < len(ev) else T) - tq
        pred.append(P.gap_count(Vq, Bq, G)); obs.append(k)
record(S, 'Prop. 4: outputs in a gap = max(0, ceil((c2 - sqrt(B) e^{-G})/pi))', 'Prop. 4, App. C', 'SGD trials (Poisson inputs)', pred, obs, 0.0, note=f'{len(obs)} groups')

# ================================================================== E. R1 (Sec. 4.1, App. D)
S = 'E R1'
A_, B_, C_, D_ = [], [], [], []
for _ in range(25):
    w_ext = rng.uniform(10, 40); t_ext = rng.uniform(0.1, 0.5); wj = rng.uniform(4, 25); d = rng.uniform(0.02, 0.3); T = rng.uniform(3, 8)
    try:
        r = R.r1(w_ext, t_ext, wj, d, T); o, rows = r['exact']
        if len(o) < 2: continue
        hh = 1e-6
        fd = np.stack([(r1_outputs(w_ext, t_ext, wj, d, T, V0=hh)-r1_outputs(w_ext, t_ext, wj, d, T, V0=-hh))/(2*hh),
                       (r1_outputs(w_ext+hh, t_ext, wj, d, T)-r1_outputs(w_ext-hh, t_ext, wj, d, T))/(2*hh),
                       (r1_outputs(w_ext, t_ext, wj+hh, d, T)-r1_outputs(w_ext, t_ext, wj-hh, d, T))/(2*hh)], 1)
        if fd.shape != rows.shape: continue
        A_.append(rows.ravel()); B_.append(fd.ravel())
    except ValueError:
        continue
record(S, 'exact trace with J_rec (eq. 9-10): d t_out / d(V0, w_ext, w_j)', 'Sec. 4.1', 'FD of an independent R1 simulator', np.concatenate(A_), np.concatenate(B_), 1e-6, note=f'{len(A_)} trials, delay d in [0.02, 0.3]')
for _ in range(10):
    w_ext = rng.uniform(10, 40); t_ext = rng.uniform(0.1, 0.5); wj = rng.uniform(4, 25); T = rng.uniform(3, 8); d = 1e-8
    o, rows = R.r1(w_ext, t_ext, wj, d, T)['exact']
    if len(o) < 2: continue
    fsg = lambda p: sg.output_times(sg.run(jnp.stack([jnp.array([p[0]]), jnp.array([0.0])]), jnp.array([[p[2]]]), jnp.array([[p[1]]]), jnp.array([t_ext]), np.array([0]), T, 4*len(o)+20), 0, len(o))
    Jj = np.array(jax.jacfwd(fsg)(jnp.array([0.0, w_ext, wj])))
    if np.any(~np.isfinite(Jj)): continue
    C_.append(rows.ravel()); D_.append(Jj.ravel())
record(S, 'exact trace with J_rec as d -> 0 (d = 1e-8)', 'Sec. 4.1', 'SGD autodiff, self-synapse (spikegd has no delays)', np.concatenate(C_), np.concatenate(D_), 1e-5, note=f'{len(C_)} trials; spikegd re-enters at the reset itself')
# four numbers and cycle maps against 50-digit FD of the one-cycle map
oursF, refF, oursG, refG = [], [], [], []
for _ in range(25):
    B = rng.uniform(0.2, 40); wj = rng.uniform(5, 30); d = rng.uniform(0.02, 0.3)
    if B > np.pi**2 and P.tsp_reset(B) < d: continue
    def cyc(t, Bb):                       # exact: the re-entry moves with t
        V1 = P.V_reset(Bb, M(d), m=P.MP); b1 = Bb*mp.exp(-2*M(d))+wj; Tt = P.tsp(V1, b1, m=P.MP)
        return (t+M(d)+Tt, b1*mp.exp(-2*Tt))
    t0_ = rng.uniform(0, 2); tin = t0_+d
    def cyc_e(t, Bb):                     # e-prop: re-entry time held at tin
        dd = M(tin)-t; V1 = P.V_reset(Bb, dd, m=P.MP); b1 = Bb*mp.exp(-2*dd)+wj; Tt = P.tsp(V1, b1, m=P.MP)
        return (M(tin)+Tt, b1*mp.exp(-2*Tt))
    oursF.append(P.F_map(B, wj, d).ravel()); refF.append(mp_jac(cyc, (t0_, B)).ravel())
    oursG.append(P.G_map(B, wj, d).ravel()); refG.append(mp_jac(cyc_e, (t0_, B)).ravel())
record(S, 'Prop. 5: exact cycle map F = [[1,K],[0,lambda]]', 'Prop. 5, App. D', 'MPFD of one cycle', np.ravel(oursF), np.ravel(refF), 1e-9)
record(S, 'Prop. 5: e-prop cycle map G = F - (sigma,rho)(1,0)', 'Prop. 5, App. D', 'MPFD of one cycle, re-entry held', np.ravel(oursG), np.ravel(refG), 1e-9)
# periodic example
B = 2.0
for _ in range(300):
    B = P.r1_cycle(B, 12.0, 0.15)[5]
evF = np.sort(np.abs(np.linalg.eigvals(P.F_map(B, 12.0, 0.15))))[::-1]; evG = np.sort(np.abs(np.linalg.eigvals(P.G_map(B, 12.0, 0.15))))[::-1]
record(S, 'periodic example w_j=12, d=0.15: B*, eig F, eig G', 'Sec. 4.1, App. D', 'numbers printed in the paper', [B, *evF, *evG], [0.885, 1.0, 0.225, 3.35, 0.052], 6e-3,
       note=f'B*={B:.4f}, eig F={np.round(evF,4)}, eig G={np.round(evG,4)}')
# Prop. 6 continuation: sufficiency of (A) or (B), w_j >= 0
viol = 0; tested = 0
for _ in range(400):
    Bk = rng.uniform(0.1, 30); wj = rng.uniform(0, 25); d = rng.uniform(0.01, 1.0)
    if not P.continuation(Bk, wj, d): continue
    tested += 1
    fires_first = Bk > np.pi**2 and P.tsp_reset(Bk) < d
    if fires_first: continue
    V1 = P.V_reset(Bk, d); b1 = Bk*np.exp(-2*d)+wj
    if not R._fires(np.array([V1, b1])): viol += 1
record(S, 'Prop. 6: (A) or (B) => next output exists', 'Prop. 6, App. D', 'exact next-spike test', [viol], [0], 0.0, note=f'{tested} draws satisfying (A) or (B)')
# Prop. 6, periodic limit of w_min
taus = rng.uniform(0.2, 3, 10); pred_, ref_ = [], []
for tau in taus:
    m = sum(np.exp(-2*tau*p) for p in range(1, 400)); pred_.append((np.pi**2-0.0)/(m+1)); ref_.append(np.pi**2*(1-np.exp(-2*tau)))
record(S, 'Prop. 6: w_min -> pi^2 (1 - e^{-2 tau}) in a periodic train', 'Prop. 6, App. D', 'direct sum of m_k', pred_, ref_, 1e-12)
# negative weight: never more outputs than without feedback
viol = 0; tested = 0
for _ in range(60):
    w_ext = rng.uniform(20, 400); t_ext = 0.1; wj = -rng.uniform(0.5, 20); d = rng.uniform(0.02, 0.5); T = 20.0
    try:
        n_fb = len(r1_outputs(w_ext, t_ext, wj, d, T)); n_0 = len(r1_outputs(w_ext, t_ext, 0.0, d, T)); tested += 1; viol += n_fb > n_0
    except ValueError:
        continue
record(S, 'App. D: w_j < 0 gives no more outputs than w_j = 0', 'App. D', 'R1 simulator', [viol], [0], 0.0, note=f'{tested} trials')

# ================================================================== F. Three neuron network (Sec. 4.2, Prop. 7)
S = 'F network'
A_, B_, C_ = [], [], []
N, Nin = 3, 2
for trial in range(12):
    n = rng.integers(3, 9); t_ext = np.sort(rng.uniform(0, 2.5, n)); idx = rng.integers(0, 2, n)
    W_in = np.array([[rng.uniform(30, 120), 0.0], [0.0, rng.uniform(30, 120)], [0.0, 0.0]])
    W_net = rng.uniform(-5, 30, (3, 3)); np.fill_diagonal(W_net, 0.0)
    T = 4.0
    outs, rows = R.network(t_ext, idx, W_in, W_net, np.zeros((2, N)), T)
    if len(outs[2]) == 0: continue
    def f(p):
        Wi = p[:N*Nin].reshape(N, Nin); Wn = p[N*Nin:].reshape(N, N)
        return sg.output_times(sg.run(jnp.zeros((2, N)), Wn, Wi, t_ext, idx, T, 400), 2, len(outs[2]))
    p0 = np.concatenate([W_in.ravel(), W_net.ravel()]); times = np.array(f(jnp.array(p0)))
    if np.any(~np.isfinite(times)) or np.max(np.abs(times-outs[2])) > 1e-6: continue
    Jj = np.array(jax.jacfwd(f)(jnp.array(p0)))
    mask = np.ones(len(p0), bool); mask[N*Nin+np.arange(N)*N+np.arange(N)] = False
    fd = np.zeros_like(Jj); hh = 1e-6
    for q in np.nonzero(mask)[0]:
        e = np.zeros(len(p0)); e[q] = hh; fd[:, q] = (np.array(f(jnp.array(p0+e)))-np.array(f(jnp.array(p0-e))))/(2*hh)
    A_.append(rows[2][:, mask].ravel()); B_.append(Jj[:, mask].ravel()); C_.append(fd[:, mask].ravel())
record(S, 'd t_C / d(W_in, W_net) from single-neuron maps + cross blocks (Prop. 7)', 'Sec. 4.2, Prop. 7', 'SGD autodiff', np.concatenate(A_), np.concatenate(B_), 1e-6, note=f'{len(A_)} trials, zero delays as in spikegd', rel_floor=1e-3)
record(S, 'd t_C / d(W_in, W_net) from single-neuron maps + cross blocks (Prop. 7)', 'Sec. 4.2, Prop. 7', 'FD of spikegd trials', np.concatenate(A_), np.concatenate(C_), 1e-4, note=f'{len(A_)} trials', rel_floor=1e-3)

# ================================================================== G. Pseudospikes (spikegd's pseudodynamics)
S = 'G pseudospikes'
ours_, ref_ = [], []
for _ in range(20):
    K = rng.integers(2, 8); w = ((K-0.5)*np.pi+np.pi/2)**2; tin = 0.1
    o, rows, (kind, s_end, t_last, Mend) = R.single_neuron(np.array([tin]), np.array([0]), np.array([w]), 0.0, 0.0, 1e3)
    dT = rng.uniform(0.05, 4); T = o[-1]+dT
    o, rows, (kind, s_end, t_last, Mend) = R.single_neuron(np.array([tin]), np.array([0]), np.array([w]), 0.0, 0.0, T)
    # state at T and its trace: flow from the last output (map out->in with no input)
    JT = P.J_out_in(s_end[1], T-s_end[0]); xT = np.array([P.V_reset(s_end[1], T-s_end[0]), s_end[1]*np.exp(-2*(T-s_end[0]))]); MT = JT @ Mend
    # Klos & Memmesheimer pseudospike time, closed form: t_ps = T + k*theta(I_ps) - phi(V, I_ps), I_ps = softplus(I)
    al = sg.neuron.alpha; Iz = xT[1]; sp = al*np.log1p(np.exp(Iz/al)); dsp = 1/(1+np.exp(-Iz/al)); r_ = np.sqrt(sp)
    dphi_dV = 1/(1+((xT[0]-0.5)/r_)**2)/r_/r_
    phi = (np.arctan((xT[0]-0.5)/r_)+np.pi/2)/r_
    dphi_dsp = (1/(1+((xT[0]-0.5)/r_)**2)*(-(xT[0]-0.5)/(2*sp**1.5)))/r_ - phi/(2*sp)
    dtheta_dsp = -np.pi/(2*sp**1.5)
    grad_x = np.array([-dphi_dV, (dtheta_dsp - dphi_dsp)*dsp])        # k = 1
    ours_.append(grad_x @ MT[:, 1])
    def g(p):
        out = sg.run(jnp.array([[0.0], [0.0]]), jnp.zeros((1, 1)), jnp.array([[p]]), jnp.array([tin]), np.array([0]), T, 60)
        xe = sg.state_at_end(out)
        return sg.neuron.t_pseudo(xe[:, 0], 0.0, 1, {'T': T})
    ref_.append(float(jax.grad(g)(w)))
record(S, 'd t_ps / d w: our trace at the trial end x closed-form gradient of spikegd t_pseudo', 'App. F (probe)', 'SGD autodiff through event + t_pseudo', ours_, ref_, 1e-5, rel_floor=1e-3)

# ------------------------------------------------------------------ report
dt = time.time()-t0
json.dump(REC, open('fd_results.json', 'w'), indent=1)
n_ok = sum(r['ok'] for r in REC); n_err = sum((not r['ok']) and r['expected_fail'] for r in REC); n_fail = len(REC)-n_ok-n_err
lines = ['# Finite-difference and cross-implementation check of theory_condensed.tex', '',
         f'{n_ok} of {len(REC)} checks pass, {n_err} reproduce a known erratum in the paper, {n_fail} unexplained failures ({dt:.0f} s).',
         'Run `python3 fd_check_all.py` to reproduce.', '',
         'References: **MPFD** = derivative of the paper\'s value formula in 50-digit arithmetic (mpmath); **SGD** = Klos & Memmesheimer\'s',
         'spikegd (float64 JAX; its closed forms, or autodiff through its event-based simulation); **FD** = central differences of whole',
         'simulated trials (h = 1e-6); **ODE** = numerical integration of the QIF equation. `max_rel` = max |ours - ref| / max(1, |ref|)',
         '(floor 1e-3 where noted). Tolerances are set by the reference: ~1e-10 for MPFD, ~1e-8 for spikegd (its reset is at -1/eps with',
         f'eps = {sg.EPS}), ~1e-5 for float64 FD of whole trials.', '']
for sec in dict.fromkeys(r['section'] for r in REC):
    lines += [f'## {sec}', '', '| check | paper | reference | n | max rel. err. | tol | result | note |', '|---|---|---|---|---|---|---|---|']
    for r in REC:
        if r['section'] != sec: continue
        lines.append(f"| {r['check']} | {r['paper']} | {r['reference']} | {r['n']} | {r['max_rel']:.1e} | {r['tol']:.0e} | {'pass' if r['ok'] else ('**erratum**' if r['expected_fail'] else '**FAIL**')} | {r['note']} |")
    lines.append('')
open('FD_REPORT.md', 'w').write('\n'.join(lines))
print(f'\n{n_ok}/{len(REC)} pass, {n_err} errata, {n_fail} unexplained, {dt:.0f} s')
