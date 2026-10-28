"""Fig. 2(a): R1, current with and without the recurrent synapse.
One external input (w=10 at t=0.3) starts the neuron from V0=b0=0; each output re-enters after d=0.15 through w_j.
Uses the exact event-driven simulation of qif_events.py. Run: python3 fig_r1_train.py -> r1_train.pdf/.png"""
import numpy as np
from style import plt, WIDTH, BLACK, GREY, VERM, BLUE, save
from qif_events import simulate, PI

wj, d, T, t_ext, w_ext = 12.0, 0.15, 6.5, 0.3, 10.0

def run(wj):
    """Event list, plus the current I(t) sampled between events (I decays as e^{-2t} and jumps by w at inputs)."""
    t_in, w_in = [t_ext], [w_ext]
    ev = None
    # re-entries depend on the outputs, so iterate: add the re-entry of every output found so far, rerun until stable
    for _ in range(50):
        ev = simulate(t_in, list(range(len(t_in))), w_in, V0=0.0, b0=0.0, T_end=T)
        re = [o + d for o in ev['outputs'] if o + d < T] if wj else []
        new_t = sorted([t_ext] + re); 
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
save(fig, 'r1_train')
