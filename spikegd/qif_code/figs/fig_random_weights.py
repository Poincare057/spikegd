"""Fig. (App. F): with random weights the amplification is usually modest, but it has a heavy tail at the capacity boundary.
l=100 synapses, one input each at a time uniform on [0,1], weights w_i ~ N(0, g^2/sqrt(l)) (g=1 is variance 1/sqrt(l)),
V0=b0=0, 1000 draws per gain g in {1,10,30,100,300}, trial T=40 so trains run out.
(a) distribution of the number of outputs.
(b) amplification at the last output, voltage direction c2/c_{k+1} (exact), against u = c2/pi-(k-1), the place of c2 in its band.
(c) survival P(amplification > A), voltage and weight directions, all g>=10 pooled; dashed: only outputs inside a trial of length 4.
Run: python3 fig_random_weights.py -> random_weights.pdf/.png ; prints the numbers in the caption."""
import numpy as np
from style import plt, WIDTH, BLUE, ORANGE, GREEN, VERM, PURPLE, GREY, BLACK, save
from qif_events import simulate, groups, PI
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
fig.tight_layout(w_pad=0.6); save(fig, 'random_weights')
