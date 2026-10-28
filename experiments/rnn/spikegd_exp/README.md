# Local gradients and fitted non-locality for QIF RNNs

This folder extends the recurrent-network experiment of Klos & Memmesheimer
(`../rnn.py`, `rnn.ipynb`) with an e-prop-style **local** gradient and a fitted
correction for the **non-local** part of the exact gradient. The derivation is
in the working notes (*Working Notes on Gradients*, A. Balivada); equation
names below follow those notes.

The result is a noise-sensitivity comparison of three training rules:
**exact** gradient, **local** gradient, and **local + matrix A + cross-target
terms**.

| figure | contents |
|---|---|
| `sensitivity_km_Akc.png` | full figure (panels A–D) |
| `sensitivity_km_Akc_loss_curves.png` | A: clean-input loss over training, one line per noise level σ |
| `sensitivity_km_Akc_drift_curves.png` | B: parameter drift from each rule's noise-free run |
| `sensitivity_km_Akc_final_loss.png` | C: final loss on held-out noisy inputs vs σ |
| `sensitivity_km_Akc_final_drift.png` | D: final drift vs σ |
| `sensitivity_km_Akc_cosine.png` | E–F: cosine between each rule's update and the exact gradient along the rule's own training |

---

## 1. Background

### Task and setting

The published RNN task (`../rnn.ipynb`, `config_qif`): N = 10 QIF neurons, 20
Poisson input channels, and two **target** neurons whose spike times are
trained towards target spike trains (7 and 2 target spikes, plus a final
"no extra spike before 1.1 T" entry). Missing spikes are scored through
pseudospike times. Recurrent weights and initial states start at zero;
training uses AdaBelief with `exponential_decay(1e-2, 2000, 1/e)` for 10,000
trials. A *trial* is one simulation of the window [0, T = 10], one gradient
evaluation and one parameter update. Task seed 7 fixes inputs and targets.

**Noise:** in each training trial every input spike time is jittered,
t → max(t + σξ, 0) with ξ ~ N(0, 1), with fresh noise every trial.

### Exact and local gradients

The exact gradient (Eq. LossChain, summed over **all** targets m):

$$\frac{d\mathcal L}{dw_{ij}} = \sum_{m}\sum_{k_m}\frac{\partial\mathcal L}{\partial t_{k_m}}\frac{dt_{k_m}}{dw_{ij}},$$

computed by autodiff through the event-based simulation, as in `../rnn.py`.

The **local** gradient drops the chain terms of Eq. TotalSpikeDeriv:

$$\frac{d\mathcal L}{dw_{ij}}\Big|_{\rm local} = \sum_{k_i}\frac{\partial\mathcal L}{\partial t_{k_i}}\,e_{ij}(t_{k_i}),\qquad e_{ij}(t_{k_i}) \equiv \frac{\partial t_{k_i}}{\partial w_{ij}}\ \text{(all other spike times fixed)}.$$

Cross-target terms (m ≠ i) have no local part: w_ij reaches neuron m only
through neuron i's spikes. The local gradient is computed in **one forward
pass** with eligibility traces (no backward pass); it matches the
frozen-surround autodiff reference to about 1e-13.

### The fitted rule (Eq. UpdateAC)

$$g_{ij} = \sum_{k_i}\Big[\frac{\partial\mathcal L}{\partial t_{k_i}}\sum_{j'}\big(\delta_{j'j}+A^{(i,k_i)}_{j'j}\big)e_{ij'}(t_{k_i}) + \Big(\sum_{m\ne i}\sum_{k_m}\frac{\partial\mathcal L}{\partial t_{k_m}}C^{(m,i)}_{k_m k_i}\Big)e_{ij}(t_{k_i})\Big]$$

- A⁽ⁱ'ᵏ⁾: one 9 × 9 matrix per target neuron i and output index k
  (ordinary spikes first, then pseudospikes, as in the loss).
- C⁽ᵐ'ⁱ⁾: first-order cross-target coefficients standing in for ∂t_{k_m}/∂t_{k_i}.

A and C are fitted jointly per target neuron (Eq. FitAC): weighted least
squares against the exact gradient, recorded every 10th trial of a published
exact-gradient run (training trials 0–499, 1,000–1,499, …, 9,000–9,499; the
rest held out), trial weight q⁽ⁿ⁾ = ‖dL/dw‖⁻², ridge factors r_A = 1 and
r_C = 1e-3. In training, g is computed online (each output's eligibility and
learning signal are stored; the cross-target term is added at trial end) and
passed to AdaBelief.

The ridge is essential: without it the fit is numerically singular (condition
numbers ~1e13) and training diverges (loss ~0.71 instead of 0.0035).

How the ridge factors were chosen (not a systematic search):

- r_A = 1 comes from the earlier A-only rule: one noise-free training run at
  each of r = 1e-3, 1e-2, 1e-1, 1 (final losses 0.024, 0.032, 0.021, 0.0043).
  It was kept unchanged when C was added.
- r_C = 1e-3 was compared only with r_C = 1, by held-out fit. At r_C = 1 the C
  term is shrunk to almost nothing.
- Only (1, 1e-3) and (0, 0) were trained with C.

### Results (held-out noisy loss, median over 20 noise seeds)

| σ | exact | local | local + A + C |
|---|---|---|---|
| 0 | 0.0119 | 0.0079 | **0.0035** |
| 0.025 | **0.0110** | 0.0156 | 0.0173 |
| 0.05 | **0.0166** | 0.0245 | 0.0223 |
| 0.1 | **0.0195** | 0.0259 | 0.0239 |
| 0.2 | **0.0265** | 0.0408 | 0.0333 |

---

## 2. Code files

| file | role |
|---|---|
| `../rnn.py` | published code, unchanged: inputs, targets, network simulation, output spike times (incl. pseudospikes), loss |
| `experiment.py` | trial setup (`make_trial`), full loss (`make_full_loss`), frozen-surround resimulation (`record_surround`, `make_frozen_loss`) used as the local-gradient reference. Includes the fix that drops neuron i's own recorded spikes from its frozen input queue. Sets the JAX CPU and float64 flags on import. |
| `fastgrad.py` | `make_fast_grad(..., "trace")`: one-pass local gradient by forward eligibility traces. Option `spike_mix` applies (I + A⁽ⁱ'ᵏ⁾) to each output's eligibility; option `cross` adds the C term at trial end. |
| `fit_spike_matrix.py` | `collect`: re-runs the published exact training and records, every 10 trials, ∂L/∂t, per-output local eligibilities and exact gradients. `fitcross`: fits A and C (Eq. FitAC). |
| `sensitivity_km.py` | `run`: trains each rule from the published initial state for every (σ, noise seed), all seeds in parallel, recording clean-input loss and parameters every 50 trials and the final loss on held-out noisy inputs. `plot`: draws the figure (`--separate` also saves each part). Rules used here: `exact`, `local`, `matkc`. |
| `cosine_km.py` | at the saved parameters of the `local` and `matkc` runs (every 250 trials, every σ and seed), evaluates the rule's gradient and the exact gradient on the clean inputs and computes their cosine over the recurrent weights, (a) on the target neurons' rows and (b) on all rows. Draws panels E–F. |
| `train_lowrank.py`, `train_noisy.py` | helpers imported by the above (`net_to_w`, a weight-layout conversion; `jitter`, the input noise) |

Plus the `spikegd` package itself (neuron model and event-based simulation).

### Data files produced

| file | made by | used by |
|---|---|---|
| `spike_data_km.npz` | `fit_spike_matrix.py collect` | `fit_spike_matrix.py fitcross` |
| `spike_AC_fit_km.json` | `fit_spike_matrix.py fitcross` | `sensitivity_km.py run` (rule `matkc`) |
| `sensitivity_km.npz` | `sensitivity_km.py run` | `sensitivity_km.py plot`, `cosine_km.py` |
| `cosine_km.npz` | `cosine_km.py` | `cosine_km.py --plot-only` |

---

## 3. Running instructions

### Environment

Python 3.11, JAX 0.10.2 (CPU), optax 0.2.8, NumPy 2.4.6, Matplotlib 3.11.2,
and `spikegd` installed from the repository root:

```bash
pip install -e .
```

All commands below are run from `..`.

### Steps

1. **Collect per-spike data** along the published exact run (~5 min;
   writes `spike_data_km.npz`):

   ```bash
   python fit_spike_matrix.py collect
   ```

2. **Fit A and C** (seconds; writes `spike_AC_fit_km.json`). The ridges must
   be given explicitly; the script's default `--ridge` is different.

   ```bash
   python fit_spike_matrix.py fitcross --ridge 1 --ridge-c 1e-3 --out spike_AC_fit_km.json
   ```

3. **Noise sweep**: σ ∈ {0, 0.025, 0.05, 0.1, 0.2}, 20 noise seeds each,
   10,000 trials, 30 held-out noisy inputs (writes `sensitivity_km.npz`;
   about 66 min for exact, 21 min for local, 30 min for A + C on a laptop
   CPU on mains power):

   ```bash
   python sensitivity_km.py run --rules exact local matkc
   ```

4. **Figures** (full figure plus the four parts):

   ```bash
   python sensitivity_km.py plot --plot-rules exact local matkc --out sensitivity_km_Akc.png --separate
   ```

5. **Cosine similarities** (needs `sensitivity_km.npz` and
   `spike_AC_fit_km.json`; writes `cosine_km.npz` and
   `sensitivity_km_Akc_cosine.png`):

   ```bash
   python cosine_km.py
   ```

To redraw the figures only, step 4 needs just `sensitivity_km.npz` and the
code, and `python cosine_km.py --plot-only` needs just `cosine_km.npz`.

---

## 4. Notes and caveats

- `run` adds to an existing `sensitivity_km.npz` (keeping other rules if the
  σ list matches); use a fresh folder for a clean reproduction.
- Every random draw is seeded, but training is chaotic: round-off differences
  (another machine, JAX version or thread count) can change individual runs.
  Expect similar medians and interquartile ranges, not identical numbers.
  Without noise, even the exact rule's result varies between ~0.003 and ~0.012.
- `run` trains all seeds of a rule in one compiled call and prints only when a
  rule finishes. Keep the machine on mains power and awake during steps 1 and
  3; battery throttling and sleep slow the runs severalfold.
- One task (seed 7) throughout; noise seeds vary only the input jitter.
- The cosines in `cosine_km.py` are measured on each rule's own trajectory,
  with clean inputs, so they differ from the fit diagnostics of
  `fit_spike_matrix.py` (measured on the exact run's states). At σ = 0 each
  rule is a single deterministic run, so that point has no spread.
- A and C are fixed coefficients fitted along one exact trajectory; they
  cannot follow spike-count transitions, where the exact gradient's large
  bursts occur.
