"""Fig. (App. G): along the outputs after one input, QR exponents go to 0 while the gradient magnitude grows.
Case 1b: one input of weight w at t=0.1 from V0=b0=0, chosen so that c2/pi = 5.5, 10.5, 20.5 (k = 6, 11, 21 outputs).
QR (Benettin) on the product of the out->out event Jacobians, QR after every map, Q0 = identity.
Magnitude: time row of the RTRL trace for V0 and for w, relative to the first output.
Run: python3 fig_qr_train.py -> qr_train.pdf/.png ; prints the numbers in the caption."""
import numpy as np
from style import plt, WIDTH, BLUE, ORANGE, GREEN, GREY, save
from qif_events import simulate, qr_accumulate, groups, PI

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
fig.tight_layout(w_pad=1.5); save(fig, 'qr_train')
