"""Fig. 5: a pseudospike added after the trial carries a gradient that decays like e^{-Delta T}.
Numerical probe, outside (A4). Uses Klos & Memmesheimer's spikegd (https://github.com/chklos/spikegd) for the simulation, the
pseudospike time and the gradients (JAX autodiff, float64):
  one input at t=0.1 from V0=b0=0 with c2/pi = K-1/2, so the train has K outputs and then stops;
  the trial ends Delta T after the last output;
  the pseudospike is spikegd's QIFNeuron.t_pseudo: t_ps = T + theta(I_ps) - phi(V(T), I_ps), I_ps = softplus_alpha(I(T)),
  i.e. after T the neuron runs on a constant suprathreshold current set by its current at T (their first type of pseudodynamics).
Plotted: |d t_ps/d w| / |d t_K/d w| against Delta T.  (a) K=5, softplus scale alpha in {1, 10, 100} (alpha=10 is spikegd's default).
(b) alpha=10, K in {2, 5, 10}.  Dashed: e^{-Delta T}.
Requires: pip install jax; pip install -e <spikegd clone>.   Run: python3 fig_pseudospike.py -> pseudospike.pdf/.png"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from spikegd.qif import QIFNeuron
from style import plt, WIDTH, BLUE, ORANGE, GREEN, GREY, save

def ratio(K, dT, alpha):
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
import os, sys
CACHE = 'pseudospike_data.npz'                 # delete it (or pass --recompute) to rerun the simulations
if os.path.exists(CACHE) and '--recompute' not in sys.argv:
    data = dict(np.load(CACHE))
else:
    data = {f'a{al:g}': np.array([ratio(5, d, al) for d in DT]) for al in [1.0, 10.0, 100.0]}
    data.update({f'K{K}': np.array([ratio(K, d, 10.0) for d in DT]) for K in [2, 5, 10]})
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
fig.tight_layout(w_pad=1.0); save(fig, 'pseudospike')
