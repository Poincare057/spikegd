"""Fig. (App. G): the number of learnable outputs is set by the current at the last input of each group and by the gap.
Poisson inputs: n input times uniform on [0,T] (Poisson conditioned on n), one synapse of weight W=250, V0=b0=0, 300 draws per (n,T).
(a) outputs in each gap against x = sqrt(B)(1-e^{-G})/pi, with the exact count of Prop. 4.
(b) outputs in the trial against T, median and 10-90% band.
(c) c2/pi at the last input of each group, T=4.
Run: python3 fig_capacity.py -> capacity.pdf/.png ; prints the numbers in the caption."""
import numpy as np
from style import plt, WIDTH, BLUE, ORANGE, GREEN, VERM, GREY, BLACK, save
from qif_events import simulate, groups, PI
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
fig.tight_layout(w_pad=1.0); save(fig, 'capacity')
