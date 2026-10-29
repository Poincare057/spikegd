# fdcheck: finite-difference and cross-implementation check of every closed form

`fd_check_all.py` checks every closed form and every gradient in `theory_condensed.tex` and writes all FD output to
[`../RUN_RESULTS.md`](../RUN_RESULTS.md): tables by section, JSON records (one per check), and the execution log.
Each run replaces the marked FD section while preserving other results. The output path is relative to the script,
so it does not depend on the working directory. It takes about 10 minutes on a laptop CPU.

## What is under test

- `paper_forms.py` – the paper's formulas, transcribed literally from the equations (each function names its section/equation):
  current, voltage (5 branches), spike time, `D`, all derivatives of `t_sp` and `V` (App. A.5–A.7), the two identities (Props. 1–2),
  the four event maps (App. B), the recurrent and cross blocks (eq. 9, Prop. 7), the capacity results (Props. 3–4), and the R1 cycle
  quantities σ, ρ, K, λ, F, G and the continuation test (Props. 5–6).
- `rtrl_paper.py` – exact gradients assembled **only** from `paper_forms.py`: single neuron (any input pattern, parameters V0, w, b0),
  R1 with delay d (exact trace with `J_rec` and the e-prop trace without it), and an N-neuron network with zero delays
  (single-neuron maps on the diagonal, cross blocks off it).

## References (none shares code with the formulas)

| tag | what |
|---|---|
| MPFD | derivative of the paper's *value* formula in 50-digit arithmetic (`mpmath.diff`), so the FD error is negligible |
| SGD | Klos & Memmesheimer's **spikegd** (https://github.com/chklos/spikegd), float64 JAX: its own closed forms (`flow_V`, `dt_spike`, `t_pseudo`), and JAX autodiff through its event-based simulation `QIFNeuron.event` |
| FD | central differences (h = 1e-6) of whole simulated trials (spikegd's, or the independent R1 simulator `r1_sim.py`) |
| ODE | numerical integration of the QIF equation (DOP853, rtol 1e-12) |

spikegd has no synaptic delays, so R1 (delay d) is checked against FD of `r1_sim.py`, and against spikegd's self-synapse in the limit
d → 0. The three-neuron network is checked with zero delays, which is spikegd's convention and satisfies (T2).

## Setup

```bash
pip install numpy scipy mpmath "jax[cpu]" jaxtyping optax
git clone https://github.com/chklos/spikegd && pip install -e spikegd --no-deps
python3 fd_check_all.py          # imports helpers from ../figs/generate_figs.py and r1_sim.py
```

`sg.py` wraps spikegd (float64, `eps = 1e-10`). `r1_sim.py` is a values-only R1 simulator used for finite differences.

## Findings

- **Erratum (fixed in the paper):** the printed `∂V/∂b` on the invariant line |V0| = sqrt(-b) was the derivative *along* the line.
  The partial at fixed V0 is `e^{-t}(e^{2V0(1-e^{-t})}-1)/(2V0)`; this entry is reported as `erratum` in the report.
- **Removable 0/0:** the b = 0 entry of `∂V/∂b` is 0/0 at V0 = 0; its limit `e^{-t}(1-e^{-t})` is used (and now stated in the paper).
- Reference limitations, not errors: spikegd differentiates the b = 0 branch through b = 1e-6 (agreement 1e-6), returns non-finite
  values on part of the invariant line, and its reset at −1/eps lets round-off grow along long trains (agreement 1e-7 to 1e-6).
