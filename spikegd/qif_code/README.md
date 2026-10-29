# Code for "Exact Gradients of QIF Neurons and Online Learning Rules"

Figure generation, numerical checks, and full gradient validation share this directory and one output folder.
All output paths are resolved relative to the source files, independently of the working directory.

The Python scripts follow the sectioned style of `experiments/rnn/rnn.py` and
`experiments/singleneuron/single.py`, with annotated interfaces and explicit
initialization, event updates, analysis, and plotting blocks. Figure generation
exposes `parse_args()` and `main()`. Full validation starts only when
`fd_check_all.py` is executed or its `run()` function is called; importing it
does not run checks or write results. `run(output_path=...)` can direct the
validation report to a separate file and returns the check records.

```text
qif_code/
  README.md             Combined documentation
  generate_figs.py      Six figure generators and shared event-driven simulator
  checks.py             Simulator, gradient, and weight-bound checks
  fd_check_all.py       Full closed-form and gradient validation
  paper_forms.py        Paper formulas
  rtrl_paper.py         Gradients assembled from the paper formulas
  r1_sim.py             Independent R1 reference simulator
  sg.py                 Float64 spikegd wrappers
  results.py            Shared output paths and results writer
  build_figs.sh         Build all six figures
  output/
    *.svg               Six figures
    RUN_RESULTS.md      All reports, numerical summaries, and FD JSON records
    data/
      *.npz             Cached numerical data
```

## Run

From the repository root, with the project's Python environment activated:

```bash
cd spikegd/qif_code
python generate_figs.py                   # six SVGs and figure summaries
python checks.py                         # simulator, gradient, and weight-bound checks
python fd_check_all.py                    # full validation, about 10 minutes
```

Each command updates its own sections in [`output/RUN_RESULTS.md`](output/RUN_RESULTS.md).
The output folder contains only figures, that results file, and the `data/` subfolder for NPZ caches.
No separate TeX, JSON, environment, or log files are generated. Run commands sequentially when sharing the output folder.

## Numerical checks

`checks.py` runs both groups of checks using the simulator in `generate_figs.py`:

- Simulator checks: out-to-out Jacobian against eq. (8), the time-to-event identity on all branches,
  RTRL traces against whole-trial finite differences, and telescoping amplification.
- Weight-bound checks: the weight bounds of Prop. 4 for Cases 2 and 3. Bounds apply only to samples
  with no output inside the input window; rows with no eligible samples give vacuous results.

Run `python3 checks.py` from `qif_code`. It replaces the simulator and weight-bound sections
in `output/RUN_RESULTS.md`, preserving the other results.

## Full gradient validation

`fd_check_all.py` checks every closed form and every gradient in `theory_condensed.tex` and writes all FD output to
[`output/RUN_RESULTS.md`](output/RUN_RESULTS.md): tables by section, JSON records (one per check), and the execution log.
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
# Install this repository editable from its root: pip install -e . --no-deps
python3 fd_check_all.py          # uses generate_figs.py and r1_sim.py
```

`sg.py` wraps spikegd (float64, `eps = 1e-10`). `r1_sim.py` is a values-only R1 simulator used for finite differences.

## Findings

- **Erratum (fixed in the paper):** the printed `∂V/∂b` on the invariant line |V0| = sqrt(-b) was the derivative *along* the line.
  The partial at fixed V0 is `e^{-t}(e^{2V0(1-e^{-t})}-1)/(2V0)`; this entry is reported as `erratum` in the report.
- **Removable 0/0:** the b = 0 entry of `∂V/∂b` is 0/0 at V0 = 0; its limit `e^{-t}(1-e^{-t})` is used (and now stated in the paper).
- Reference limitations, not errors: spikegd differentiates the b = 0 branch through b = 1e-6 (agreement 1e-6), returns non-finite
  values on part of the invariant line, and its reset at −1/eps lets round-off grow along long trains (agreement 1e-7 to 1e-6).

## Figures

`generate_figs.py` contains all six Python figure generators, the shared plotting style (fonts and colours sized for the 5.5 in
text width), and the exact event-driven QIF simulator, complex-step event Jacobians, RTRL traces and Benettin/QR helpers.
Importing it exposes the helpers without generating plots or loading Matplotlib or JAX. Each generator prints the numbers quoted
in its caption into `output/RUN_RESULTS.md`; no separate log files are written.

| Figure | Source |
|---|---|
| Fig. 2(a), R1 current with and without the re-entry | `generate_figs.py r1_train` |
| App. F, QR exponents vs. magnitude along a train | `generate_figs.py qr_train` |
| App. F, capacity (gap count, outputs vs T, spread of c2) | `generate_figs.py capacity` |
| App. F, sources of variation of the QR exponents | `generate_figs.py qr_seeds` |
| App. F, random Gaussian weights (variance g^2/sqrt(l)) | `generate_figs.py random_weights` |
| Fig. 5, pseudospike probe (uses spikegd; cached in `pseudospike_data.npz`, `--recompute` to rerun) | `generate_figs.py pseudospike` |

From `qif_code`:

```bash
python3 generate_figs.py                              # all six Python figures
python3 generate_figs.py r1_train qr_train             # selected figures
python3 generate_figs.py pseudospike --recompute
```

Figures are saved as SVG in `output/`; the pseudospike cache is `output/data/pseudospike_data.npz`.
JAX and spikegd are needed only when the cache is missing or `--recompute` is passed.
`./build_figs.sh` rebuilds all six figures and records their numerical summaries in `output/RUN_RESULTS.md`.
