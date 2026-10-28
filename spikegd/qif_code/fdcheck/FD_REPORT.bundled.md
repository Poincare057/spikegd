# Finite-difference and cross-implementation check of theory_condensed.tex

98 of 99 checks pass, 1 reproduce a known erratum in the paper, 0 unexplained failures (667 s).
Run `python3 fd_check_all.py` to reproduce.

References: **MPFD** = derivative of the paper's value formula in 50-digit arithmetic (mpmath); **SGD** = Klos & Memmesheimer's
spikegd (float64 JAX; its closed forms, or autodiff through its event-based simulation); **FD** = central differences of whole
simulated trials (h = 1e-6); **ODE** = numerical integration of the QIF equation. `max_rel` = max |ours - ref| / max(1, |ref|)
(floor 1e-3 where noted). Tolerances are set by the reference: ~1e-10 for MPFD, ~1e-8 for spikegd (its reset is at -1/eps with
eps = 1e-10), ~1e-5 for float64 FD of whole trials.

## A neuron model

| check | paper | reference | n | max rel. err. | tol | result | note |
|---|---|---|---|---|---|---|---|
| current I(t) = b e^{-2t}  (eq. 2) | Sec. 2.1 | SGD flow_I | 50 | 2.2e-16 | 1e-14 | pass |  |
| V(t), branch tan  (eq. 3) | Sec. 2.1 / App. A.2 | SGD flow_V | 40 | 5.1e-15 | 1e-09 | pass |  |
| V(t), branch tan  (eq. 3) | Sec. 2.1 / App. A.2 | ODE (DOP853, rtol 1e-12) | 15 | 7.6e-14 | 1e-08 | pass |  |
| V(t), branch b0  (eq. 3) | Sec. 2.1 / App. A.2 | SGD flow_V | 40 | 1.9e-15 | 1e-09 | pass |  |
| V(t), branch b0  (eq. 3) | Sec. 2.1 / App. A.2 | ODE (DOP853, rtol 1e-12) | 15 | 4.5e-13 | 1e-08 | pass |  |
| V(t), branch coth  (eq. 3) | Sec. 2.1 / App. A.2 | SGD flow_V | 40 | 1.7e-15 | 1e-09 | pass |  |
| V(t), branch coth  (eq. 3) | Sec. 2.1 / App. A.2 | ODE (DOP853, rtol 1e-12) | 15 | 1.2e-13 | 1e-08 | pass |  |
| V(t), branch tanh  (eq. 3) | Sec. 2.1 / App. A.2 | SGD flow_V | 40 | 2.2e-16 | 1e-09 | pass |  |
| V(t), branch tanh  (eq. 3) | Sec. 2.1 / App. A.2 | ODE (DOP853, rtol 1e-12) | 15 | 1.7e-13 | 1e-08 | pass |  |
| V(t), branch line  (eq. 3) | Sec. 2.1 / App. A.2 | SGD flow_V | 21 | 1.3e-16 | 1e-09 | pass | spikegd returns non-finite values on part of the invariant line; finite ones compared |
| V(t), branch line  (eq. 3) | Sec. 2.1 / App. A.2 | ODE (DOP853, rtol 1e-12) | 15 | 2.1e-12 | 1e-08 | pass |  |
| t_sp, branch tan  (eq. 4) | Sec. 2.1 / App. A.3 | SGD dt_spike | 40 | 1.9e-16 | 1e-09 | pass |  |
| t_sp, branch tan  (eq. 4) | Sec. 2.1 / App. A.3 | ODE blow-up time (phase, rtol 1e-12) | 15 | 1.9e-12 | 1e-08 | pass |  |
| t_sp, branch coth  (eq. 4) | Sec. 2.1 / App. A.3 | SGD dt_spike | 40 | 3.9e-16 | 1e-09 | pass |  |
| t_sp, branch coth  (eq. 4) | Sec. 2.1 / App. A.3 | ODE blow-up time (phase, rtol 1e-12) | 15 | 1.1e-11 | 1e-08 | pass |  |
| t_sp, branch b0  (eq. 4) | Sec. 2.1 / App. A.3 | SGD dt_spike | 40 | 2.1e-16 | 1e-09 | pass | spikegd evaluates b=0 at b=1e-6 |
| t_sp, branch b0  (eq. 4) | Sec. 2.1 / App. A.3 | ODE blow-up time (phase, rtol 1e-12) | 15 | 5.3e-12 | 1e-08 | pass |  |
| t_sp after a reset  (eq. 5) | Sec. 2.1 / App. A.7 | MP limit V0=-1e40 | 40 | 2.2e-16 | 1e-12 | pass |  |
| t_sp after a reset  (eq. 5) | Sec. 2.1 / App. A.7 | SGD dt_spike at V=-1/eps | 40 | 2.7e-10 | 1e-08 | pass |  |
| dt_sp/dV0, branch tan | App. A.5 | MPFD | 40 | 5.6e-16 | 1e-11 | pass |  |
| dt_sp/dV0, branch tan | App. A.5 | SGD autodiff | 40 | 7.8e-16 | 1e-08 | pass |  |
| dt_sp/db, branch tan | App. A.5 | MPFD | 40 | 5.0e-16 | 1e-11 | pass |  |
| dt_sp/db, branch tan | App. A.5 | SGD autodiff | 40 | 3.1e-16 | 1e-08 | pass |  |
| dt_sp/dV0, branch coth | App. A.5 | MPFD | 40 | 4.9e-16 | 1e-11 | pass |  |
| dt_sp/dV0, branch coth | App. A.5 | SGD autodiff | 40 | 1.4e-15 | 1e-08 | pass |  |
| dt_sp/db, branch coth | App. A.5 | MPFD | 40 | 3.8e-16 | 1e-11 | pass |  |
| dt_sp/db, branch coth | App. A.5 | SGD autodiff | 40 | 1.3e-15 | 1e-08 | pass |  |
| dt_sp/dV0, b = 0 | App. A.5 | MPFD | 30 | 1.2e-16 | 1e-11 | pass |  |
| dt_sp/db, b = 0 (limit of both branches) | App. A.5 | MPFD across b=0 | 30 | 1.1e-16 | 1e-11 | pass |  |
| dV/dV0, branch tan | App. A.6 | MPFD | 40 | 1.7e-15 | 1e-10 | pass |  |
| dV/dV0, branch tan | App. A.6 | SGD autodiff | 40 | 1.0e-15 | 1e-08 | pass |  |
| dV/db, branch tan | App. A.6 | MPFD | 40 | 1.7e-15 | 1e-10 | pass |  |
| dV/db, branch tan | App. A.6 | SGD autodiff | 40 | 5.6e-16 | 1e-08 | pass |  |
| dV/dt, branch tan | App. A.6 | MPFD | 40 | 1.4e-15 | 1e-10 | pass |  |
| dV/dV0, branch coth | App. A.6 | MPFD | 40 | 1.7e-15 | 1e-10 | pass |  |
| dV/dV0, branch coth | App. A.6 | SGD autodiff | 40 | 3.0e-15 | 1e-08 | pass |  |
| dV/db, branch coth | App. A.6 | MPFD | 40 | 4.2e-15 | 1e-10 | pass |  |
| dV/db, branch coth | App. A.6 | SGD autodiff | 40 | 2.8e-15 | 1e-08 | pass |  |
| dV/dt, branch coth | App. A.6 | MPFD | 40 | 1.3e-15 | 1e-10 | pass |  |
| dV/dV0, branch tanh | App. A.6 | MPFD | 40 | 3.3e-16 | 1e-10 | pass |  |
| dV/dV0, branch tanh | App. A.6 | SGD autodiff | 40 | 1.0e-15 | 1e-08 | pass |  |
| dV/db, branch tanh | App. A.6 | MPFD | 40 | 2.2e-16 | 1e-10 | pass |  |
| dV/db, branch tanh | App. A.6 | SGD autodiff | 40 | 2.8e-16 | 1e-08 | pass |  |
| dV/dt, branch tanh | App. A.6 | MPFD | 40 | 6.7e-16 | 1e-10 | pass |  |
| dV/dV0, branch b0 | App. A.6 | MPFD | 40 | 1.4e-15 | 1e-10 | pass |  |
| dV/dV0, branch b0 | App. A.6 | SGD autodiff | 40 | 1.8e-06 | 1e-05 | pass | spikegd differentiates the b=0 branch through the b=1e-6 tan branch |
| dV/db, b = 0 (variational equation) | App. A.6 | MPFD across b=0 | 40 | 2.9e-15 | 1e-09 | pass |  |
| dV/db, b = 0, V0 = 0 | App. A.6 | MPFD across b=0 | 10 | 2.8e-17 | 1e-09 | pass | the printed entry is 0/0 at V0=0; its limit e^{-t}(1-e^{-t}) is used |
| dV/dt, branch b0 | App. A.6 | MPFD | 40 | 1.4e-15 | 1e-10 | pass |  |
| dV/db on the line |V0|=sqrt(-b), as printed | App. A.6 | MPFD (fixed V0) | 20 | 1.1e+00 | 1e-06 | **erratum** | ERRATUM: the printed entry is the derivative along the line, not at fixed V0 |
| dV/db on the line, corrected e^{-t}(e^{2V0(1-e^{-t})}-1)/(2V0) | App. A.6 (correction) | MPFD (fixed V0) | 20 | 1.7e-12 | 1e-06 | pass |  |
| V(tau) after a reset, b>0 | App. A.7 | MP V0=-1e40 | 20 | 1.6e-15 | 1e-12 | pass |  |
| dV(tau)/db after a reset, b>0 | App. A.7 | MPFD | 20 | 2.4e-15 | 1e-10 | pass |  |
| V(tau) after a reset, b>0 | App. A.7 | SGD flow_V from V=-1/eps | 20 | 4.9e-09 | 1e-08 | pass |  |
| V(tau) after a reset, b<0 | App. A.7 | MP V0=-1e40 | 30 | 8.5e-16 | 1e-12 | pass |  |
| dV(tau)/db after a reset, b<0 | App. A.7 | MPFD | 30 | 7.5e-16 | 1e-10 | pass |  |
| V(tau) after a reset, b<0 | App. A.7 | SGD flow_V from V=-1/eps | 30 | 1.9e-09 | 1e-08 | pass |  |
| dV(tau)/db after a reset, b = 0 | App. A.7 | MPFD across b=0 | 15 | 3.0e-15 | 1e-09 | pass |  |
| Prop. 1: grad t_sp . f = -1, branch tan | Prop. 1, App. A.8 | exact value -1 | 40 | 4.4e-16 | 1e-10 | pass |  |
| Prop. 1: grad t_sp . f = -1, branch coth | Prop. 1, App. A.8 | exact value -1 | 40 | 1.0e-14 | 1e-10 | pass |  |
| Prop. 1: grad t_sp . f = -1, branch b0 | Prop. 1, App. A.8 | exact value -1 | 40 | 6.7e-16 | 1e-10 | pass |  |
| Prop. 2: 1 + grad t_sp . f(x-) = grad t_sp . (-w, 2w) | Prop. 2 | both sides | 25 | 3.3e-16 | 1e-10 | pass |  |
| Prop. 2: d t_out / d t_in | Prop. 2 | SGD autodiff wrt input time | 25 | 4.4e-16 | 1e-07 | pass |  |

## B event maps

| check | paper | reference | n | max rel. err. | tol | result | note |
|---|---|---|---|---|---|---|---|
| J_in->in, branch tan | Sec. 3.1, App. B | MPFD of the map | 60 | 2.3e-15 | 1e-09 | pass |  |
| J_in->in, branch coth | Sec. 3.1, App. B | MPFD of the map | 60 | 3.8e-15 | 1e-09 | pass |  |
| J_in->in, branch tanh | Sec. 3.1, App. B | MPFD of the map | 60 | 2.1e-16 | 1e-09 | pass |  |
| J_in->in, branch b0 | Sec. 3.1, App. B | MPFD of the map | 60 | 2.9e-15 | 1e-09 | pass |  |
| J_in->out, branch tan | Sec. 3.1, App. B | MPFD of the map | 60 | 5.3e-16 | 1e-09 | pass |  |
| J_in->out, branch coth | Sec. 3.1, App. B | MPFD of the map | 60 | 8.1e-16 | 1e-09 | pass |  |
| J_out->out (eq. 8) | Sec. 3.1, App. B | MPFD of the map | 80 | 1.1e-16 | 1e-10 | pass |  |
| J_out->in, reset current b>0 | Sec. 3.1, App. B | MPFD of the map | 60 | 3.3e-15 | 1e-09 | pass |  |
| J_out->in, reset current b<0 | Sec. 3.1, App. B | MPFD of the map | 80 | 2.5e-15 | 1e-09 | pass |  |

## C single neuron gradient

| check | paper | reference | n | max rel. err. | tol | result | note |
|---|---|---|---|---|---|---|---|
| Case 1a: d t_out / d(V0, w) from the event maps | Secs. 2.2, 3.1 | SGD autodiff | 94 | 1.5e-08 | 1e-07 | pass | 12 trials |
| Case 1a: d t_out / d(V0, w) from the event maps | Secs. 2.2, 3.1 | FD of spikegd trials | 94 | 1.5e-08 | 1e-05 | pass | 12 trials |
| Case 1b: d t_out / d(V0, w) from the event maps | Secs. 2.2, 3.1 | SGD autodiff | 78 | 2.3e-09 | 1e-07 | pass | 12 trials |
| Case 1b: d t_out / d(V0, w) from the event maps | Secs. 2.2, 3.1 | FD of spikegd trials | 78 | 5.9e-09 | 1e-05 | pass | 12 trials |
| Case 2: d t_out / d(V0, w) from the event maps | Secs. 2.2, 3.1 | SGD autodiff | 152 | 2.7e-09 | 1e-07 | pass | 12 trials |
| Case 2: d t_out / d(V0, w) from the event maps | Secs. 2.2, 3.1 | FD of spikegd trials | 152 | 5.0e-09 | 1e-05 | pass | 12 trials |
| Case 3: d t_out / d(V0, w) from the event maps | Secs. 2.2, 3.1 | SGD autodiff | 340 | 6.0e-09 | 1e-07 | pass | 12 trials |
| Case 3: d t_out / d(V0, w) from the event maps | Secs. 2.2, 3.1 | FD of spikegd trials | 340 | 6.2e-09 | 1e-05 | pass | 12 trials |

## D capacity

| check | paper | reference | n | max rel. err. | tol | result | note |
|---|---|---|---|---|---|---|---|
| Prop. 3(i): c_{i+1} = c_i - pi | Prop. 3(i), App. C | SGD reset currents | 213 | 2.8e-09 | 1e-06 | pass |  |
| Prop. 3(ii): number of outputs = ceil(c2/pi) | Prop. 3(ii), App. C | SGD output count | 40 | 0.0e+00 | 0e+00 | pass |  |
| App. C: t_i = 1/2 ln b - ln(c2-(i-1)pi) | App. C | SGD output times | 253 | 1.3e-07 | 1e-06 | pass | spikegd error grows along long trains (reset at -1/eps) |
| Prop. 3(iii): dt_i/dtheta = beta/2b - gamma/c_{i+1}  (theta = V0, b0, w) | Prop. 3(iii), App. C | SGD autodiff | 759 | 1.0e-06 | 1e-05 | pass | amplified spikegd round-off near the capacity boundary |
| Prop. 3(iii): dt_i/dtheta = beta/2b - gamma/c_{i+1}  (theta = V0, b0, w) | Prop. 3(iii), App. C | product of event maps (rtrl_paper) | 759 | 9.8e-14 | 1e-09 | pass |  |
| App. C: weight of the last input, ratio = (2k-1) c2/c_{k+1} | App. C | SGD autodiff | 4 | 1.6e-08 | 1e-05 | pass |  |
| QR exponents along a train = (0, ln(c_{k+1}/c2)/(k-1)) | Sec. 3.2, App. C | Benettin QR on the maps | 6 | 0.0e+00 | 1e-12 | pass |  |
| Prop. 4: outputs in a gap = max(0, ceil((c2 - sqrt(B) e^{-G})/pi)) | Prop. 4, App. C | SGD trials (Poisson inputs) | 659 | 0.0e+00 | 0e+00 | pass | 659 groups |

## E R1

| check | paper | reference | n | max rel. err. | tol | result | note |
|---|---|---|---|---|---|---|---|
| exact trace with J_rec (eq. 9-10): d t_out / d(V0, w_ext, w_j) | Sec. 4.1 | FD of an independent R1 simulator | 366 | 2.2e-09 | 1e-06 | pass | 23 trials, delay d in [0.02, 0.3] |
| exact trace with J_rec as d -> 0 (d = 1e-8) | Sec. 4.1 | SGD autodiff, self-synapse (spikegd has no delays) | 138 | 3.3e-08 | 1e-05 | pass | 10 trials; spikegd re-enters at the reset itself |
| Prop. 5: exact cycle map F = [[1,K],[0,lambda]] | Prop. 5, App. D | MPFD of one cycle | 100 | 2.2e-16 | 1e-09 | pass |  |
| Prop. 5: e-prop cycle map G = F - (sigma,rho)(1,0) | Prop. 5, App. D | MPFD of one cycle, re-entry held | 100 | 6.1e-15 | 1e-09 | pass |  |
| periodic example w_j=12, d=0.15: B*, eig F, eig G | Sec. 4.1, App. D | numbers printed in the paper | 5 | 7.5e-04 | 6e-03 | pass | B*=0.8850, eig F=[1.     0.2251], eig G=[3.3475 0.0518] |
| Prop. 6: (A) or (B) => next output exists | Prop. 6, App. D | exact next-spike test | 1 | 0.0e+00 | 0e+00 | pass | 357 draws satisfying (A) or (B) |
| Prop. 6: w_min -> pi^2 (1 - e^{-2 tau}) in a periodic train | Prop. 6, App. D | direct sum of m_k | 10 | 2.3e-16 | 1e-12 | pass |  |
| App. D: w_j < 0 gives no more outputs than w_j = 0 | App. D | R1 simulator | 1 | 0.0e+00 | 0e+00 | pass | 40 trials |

## F network

| check | paper | reference | n | max rel. err. | tol | result | note |
|---|---|---|---|---|---|---|---|
| d t_C / d(W_in, W_net) from single-neuron maps + cross blocks (Prop. 7) | Sec. 4.2, Prop. 7 | SGD autodiff | 588 | 1.4e-07 | 1e-06 | pass | 12 trials, zero delays as in spikegd |
| d t_C / d(W_in, W_net) from single-neuron maps + cross blocks (Prop. 7) | Sec. 4.2, Prop. 7 | FD of spikegd trials | 588 | 1.1e-06 | 1e-04 | pass | 12 trials |

## G pseudospikes

| check | paper | reference | n | max rel. err. | tol | result | note |
|---|---|---|---|---|---|---|---|
| d t_ps / d w: our trace at the trial end x closed-form gradient of spikegd t_pseudo | App. F (probe) | SGD autodiff through event + t_pseudo | 20 | 1.3e-08 | 1e-05 | pass |  |
