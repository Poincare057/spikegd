"""Fig. (App. G): which sources of variation the QR exponents forget.
Baseline trial: Poisson inputs at rate 1 on [0,60] over l=2 synapses, weights U(150,350), V0~U(-0.5,0.5), b0~U(0,3).
Four seeds, varied one at a time with the other three at the baseline (as in Toolkit 2, Sec. 6):
input times (seed 1 / 101..119), weights (2 / 201..219), initial state (3 / 301..319), initial QR basis (4 / 401..419); 20 runs each.
QR after every event, no events discarded, exponents per event (natural log).
(a) sample SD (ddof=1) of lambda_1 over the 20 runs against the number of events.  (b) mean +/- SD of lambda_1 at 200 events.
Run: python3 fig_qr_seeds.py -> qr_seeds.pdf/.png ; prints the table in the caption."""
import numpy as np
from style import plt, WIDTH, BLUE, ORANGE, GREEN, VERM, GREY, save
from qif_events import simulate, qr_accumulate, initial_basis
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
fig.tight_layout(w_pad=1.0); save(fig, 'qr_seeds')
