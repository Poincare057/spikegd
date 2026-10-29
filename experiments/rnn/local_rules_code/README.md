# Local gradients and fitted non-locality for QIF RNNs

This folder extends the recurrent-network experiment of Klos & Memmesheimer
(`../rnn.py`, `../rnn.ipynb`) with an e-prop-style **local** gradient and a fitted
correction for the **non-local** part of the exact gradient. The derivation is
in the notes *Empirics of SNN Local Learning Rules* (A. Balivada); equation
names below follow those notes.

Three update rules are trained under input spike-time jitter σ ∈ {0, 0.05, 0.2}:

- **exact**: the exact gradient, by autodiff;
- **local**: the local gradient;
- **fitted**: local + A + C, including the off-target rows.

For each rule the code records the loss over training, the final loss on
held-out jittered inputs, and the per-neuron cosine similarity of its gradient
with the exact gradient.

---

## 1. Background

### Task, initialisation, training

- **Network:** the published task (`rnn.ipynb`, `config_qif`, seed 7), N = 10
  QIF neurons on [0, T], T = 10.
- **Inputs:** each neuron receives its own excitatory and inhibitory Poisson
  input train (rate 1, input weights +5 and −5, not trained).
- **Targets:** two **target** neurons are trained towards 7 and 2 target
  spikes. Each target also carries a final entry that penalises an extra spike
  before 1.1 T, so N_tar,1 = 8 and N_tar,2 = 3 loss entries.
- **Missing spikes:** scored through pseudospike times.
- **Loss:** Eq. Loss, normalised by the total number of entries (11).

As in the published code, the recurrent weights w_ij and the initial states
V_i(0), I_i(0) all start at **zero**, so the network is initially uncoupled.

Training uses AdaBelief with learning rate 10⁻² e^(−n/2000) for 10,000 trials.
A trial is one simulation, one gradient evaluation and one update.

**Noise:** in every training trial each input spike time is jittered,
t → max(t + σξ, 0) with ξ ~ N(0, 1), with fresh noise every trial (Eq. Jitter).

### Exact and local gradients

The exact gradient sums over all targets m (Eq. LossChain):

$$\frac{d\mathcal L}{dw_{ij}} = \sum_{m}\sum_{k_m}\frac{\partial\mathcal L}{\partial t_{k_m}}\frac{dt_{k_m}}{dw_{ij}}.$$

It is computed by autodiff through the event-based simulation.

The **eligibility** is $e_{ij}(t_{k_i}) \equiv \partial t_{k_i}/\partial w_{ij}$.
It is evaluated with every spike time not emitted by neuron i held fixed,
while the dependence through neuron i's own earlier spikes is propagated. The
**local rule** keeps only this term for the target rows:

$$g_{ij} = \sum_{k_i}\frac{\partial\mathcal L}{\partial t_{k_i}}\,e_{ij}(t_{k_i}),$$

and likewise for $V_i(0)$ and $I_i(0)$. Off-target neurons are not updated.

### The fitted rule

**Target rows** (Eq. UpdateAC):

$$g_{ij} = \sum_{k_i}\Big[\frac{\partial\mathcal L}{\partial t_{k_i}}\sum_{j'}\big(\delta_{j'j}+A^{(i,k_i)}_{j'j}\big)e_{ij'}(t_{k_i}) + \Big(\sum_{m\ne i}\sum_{k_m}\frac{\partial\mathcal L}{\partial t_{k_m}}C^{(m,i)}_{k_m k_i}\Big)e_{ij}(t_{k_i})\Big]$$

- A⁽ⁱ'ᵏ⁾ is one full 9 × 9 matrix per target i and output k. Outputs are
  ordered with ordinary spikes first, then pseudospikes.
- C⁽ᵐ'ⁱ⁾ holds the first-order cross-target coefficients, standing in for
  ∂t_{k_m}/∂t_{k_i} (Eq. CrossFirstOrder).

**Off-target rows** i > N_tar have no loss term of their own, so their whole
gradient is cross-target (Eqs. CrossFirstOrderOffTarget, UpdateOff):

$$g_{ij} = \sum_{k_i}\Big(\sum_{m}\sum_{k_m}\frac{\partial\mathcal L}{\partial t_{k_m}}C^{(m,i)}_{k_m k_i}\Big)e_{ij}(t_{k_i}).$$

Here k_i counts neuron i's ordinary spikes in order of occurrence. Only the
first 8 spikes get coefficients; later spikes are ignored. Off-target neurons
have no pseudospikes.

**Initial states:** the target neurons' initial states get their local
gradient. The off-target neurons' initial states are not updated.

### Fitting (Eqs. FitAC, FitOff)

All coefficients are fitted by weighted least squares on **one noise-free
exact-gradient run**, recorded every 10th trial (1,000 recordings).

- **Fit and held-out sets:** blocks of 50 recordings alternate between the fit
  set and a held-out set.
- **Trial weight:** q⁽ⁿ⁾ = ‖dL/dw (target rows)‖⁻², which makes it a
  relative-error fit.
- **Target rows:** A and C are fitted jointly, separately for each target.
- **Off-target rows:** C is fitted separately for each neuron.
- **Ridge:** each ridge is set relative to the mean diagonal of its block of
  the normal matrix: λ = r × mean diagonal.

| coefficients | ridge factor |
|---|---|
| A⁽ⁱ'ᵏ⁾, target rows | r_A = 1 |
| C⁽ᵐ'ⁱ⁾, between targets | r_C = 10⁻³ |
| C⁽ᵐ'ⁱ⁾, off-target rows | r_C^off = 1 |

The fit is solved in closed form, and the coefficients are held fixed during
training.

### Computing the rules in one forward pass

The local and fitted rules need no backward pass and no re-simulation.
Alongside the dynamics, each traced neuron carries the derivatives of
(V_i, I_i) with respect to its own parameters, giving e_ij at each of its
spikes.

- **Which neurons are traced:** only the targets for the local rule, at cost
  O(N_tar N) per event. All N neurons for the fitted rule, at cost O(N²) per
  event.
- **When terms are added:** the A term is added as each output occurs. The C
  terms need every target's ∂L/∂t, so each output's eligibility and ∂L/∂t are
  stored and the C terms are added at the end of the trial.
- **Optimizer:** the result is passed to AdaBelief in place of the exact
  gradient.

---

## 2. Code files

| file | role |
|---|---|
| `../rnn.py` | shared published code, unchanged: inputs, targets, network simulation, output spike times (incl. pseudospikes), loss |
| `experiment.py` | trial setup (`make_trial`), full loss (`make_full_loss`), weight-layout helpers. Sets the JAX CPU and float64 flags on import. |
| `fastgrad.py` | `make_fast_grad(..., "trace")`: one-pass eligibility traces. Options: `spike_mix` (I + A), `cross` (target C), `offtarget` (off-target C, traces all neurons), `return_pieces` (per-output ∂L/∂t and eligibilities, used by `fit.py collect`). |
| `fit.py` | `collect`: replays the published exact run and records ∂L/∂t, per-output and per-spike eligibilities and the exact gradient. `fit`: fits A, C and the off-target C, and prints held-out relative error and cosine similarity per row. |
| `train.py` | trains `exact`, `local`, `fitted` for each (σ, noise seed). All seeds of a rule run in parallel. Records the clean-input loss and the parameters every 50 trials, and the final loss on 30 held-out jittered inputs. |
| `cosine.py` | at the parameters saved along each rule's own runs, every 250 trials, evaluates the rule's gradient g and the exact gradient G = ∂L/∂w on the inputs without jitter, and computes for every neuron i the cosine similarity of row i (its input weights): cos_i = Σ_j g_ij G_ij / (‖g_i·‖ ‖G_i·‖). Undefined (NaN) where the rule leaves row i unchanged (local rule on off-target rows; a neuron that did not spike). |
| `output_paths.py` | shared data and figure directories, anchored to this folder |
| `plot.py` | `output/figs/fig_loss.svg`, `output/figs/fig_cosine.svg` and `output/figs/fig_cosine_neurons.svg` |

Plus the `spikegd` package itself, which provides the neuron model and the
event-based simulation. `experiment.py` imports the shared
`experiments/rnn/rnn.py` from the repository; no local copy is needed.
Run the commands below from `experiments/rnn/local_rules_code/` with `spikegd`
installed.

The scripts follow the `rnn.py` and `single.py` organization: `# %%` sections
separate configuration, initialization, model functions, training and plotting.
Public functions have type annotations, and numerical routines keep their
setup, updates and result handling in explicit blocks. CLI scripts expose
`parse_args()` and `main()`; `train.run(args)` and `cosine.run(args)` can also
be called directly with an `argparse.Namespace` to run and save an experiment.
`fastgrad.py` shares the spike-time and flow derivative helpers between its
gradient and summed-eligibility implementations.

Outputs are created automatically under this folder, regardless of the working
directory (explicit CLI path overrides are still supported):

```text
output/
  data/  # NPZ, JSON, pickle and CSV results
  figs/  # SVG figures only
```

Bare output filenames (for example `--csv cosine_per_neuron.csv` or
`--neurons-out fig_cosine_neurons.svg`) are routed to these folders too,
including when writer functions are called directly. Paths with an explicit
directory keep that location, and missing parent directories are created.

The stages in `experiment.py` also use `output/data/` for snapshots,
decomposition, partitions, convergence and retraining results, and save
`output/figs/main_figure.svg`. Figure output overrides must end in `.svg`.

### Data files produced

| file | made by | used by |
|---|---|---|
| `output/data/fit_data.npz` | `fit.py collect` | `fit.py fit` |
| `output/data/fit_rule.json` | `fit.py fit` | `train.py`, `cosine.py` |
| `output/data/train_results.npz` | `train.py` | `cosine.py`, `plot.py` |
| `output/data/cosine_results.npz` | `cosine.py` | `plot.py` |
| `output/data/cosine_per_neuron.csv` | `cosine.py` | (every per-neuron value: rule, σ, seed, trial, neuron, cosine, row norms) |

### Figures

| file | contents |
|---|---|
| `output/figs/fig_loss.svg` | A: loss on inputs without jitter over training, one panel per rule, one line per σ. B: final held-out jittered loss vs σ. |
| `output/figs/fig_cosine.svg` | A: per-neuron cosine similarity for every neuron i, per σ and rule (median and interquartile range over saved trials and seeds; points without an update of row i left out). B: fraction of saved points at which the rule updates row i. |
| `output/figs/fig_cosine_neurons.svg` | per-neuron cosine similarity over training for targets 0, 1 and off-target neurons 4, 7, 8 (`plot.py --neurons` to choose), one row of panels per rule |

---

## 3. Running instructions

### Environment

Python 3.11, JAX 0.10.2 (CPU), optax 0.2.8, NumPy 2.4.6, Matplotlib 3.11.2,
plus `jaxtyping` and `tqdm` (imported by `rnn.py`), with `spikegd` installed
from the root of the spikegd repository:

```bash
pip install -e .
```

All commands below are run from this folder. The times are for a laptop
CPU on mains power.

1. **Collect** (about 6 min; writes `output/data/fit_data.npz`):

   ```bash
   python fit.py collect
   ```

2. **Fit** (seconds; writes `output/data/fit_rule.json`, with defaults r_A = 1, r_C = 10⁻³,
   r_C^off = 1):

   ```bash
   python fit.py fit
   ```

3. **Train**: σ ∈ {0, 0.05, 0.2}, 5 noise seeds, writes `output/data/train_results.npz`.
   It takes about 40 min in total: exact 6 min, local 4 min, fitted 30 min.

   ```bash
   python train.py
   ```

   Options: `--sigmas` (any subset of 0, 0.025, 0.05, 0.1, 0.2), `--n-seeds`,
   and `--rules` (rerun some rules and keep the others in the file).

4. **Cosine similarities** (about 2 min; writes `output/data/cosine_results.npz`
   and `output/data/cosine_per_neuron.csv`):

   ```bash
   python cosine.py
   ```

5. **Figures** (seconds; writes `output/figs/fig_loss.svg`, `output/figs/fig_cosine.svg`
   and `output/figs/fig_cosine_neurons.svg`):

   ```bash
   python plot.py
   ```

---

## 4. Notes and caveats

- **Import order.** `experiment.py` must be imported before `jax`: it switches
  on float64. Run in float32, the same seed produces a different task (7 and 4
  target spikes instead of 8 and 3). All scripts here import it first.
- **σ = 0 is a single deterministic run,** copied across seeds in the results.
  Noise-free training is chaotic: round-off differences (machine, JAX version,
  batched vs single runs) can change its final loss several-fold.
- **Training is chaotic at every σ.** Perturbations of order 1e-13 (in the
  fitted coefficients or the batch size) change individual runs. With 5 seeds
  the medians can shift noticeably between otherwise identical runs; for
  example the fitted rule's median at σ = 0.2 was 0.052 in one run and
  0.033 in another. Use more seeds (`--n-seeds`) for firm comparisons.
- `cosine.py` may print an XLA "constant folding ... taking > 1s" message
  while compiling. It is harmless.
- **Noise streams are paired.** The noise draws for a given σ and seed are the
  same in every rule and in any σ subset. They come from σ's position in the
  fixed grid (0, 0.025, 0.05, 0.1, 0.2).
- **Off-target rows depend on hidden spikes.** An off-target neuron that does
  not fire in a trial has no eligibility, so its row gets no update in that
  trial. Spikes beyond the 8th are ignored.
- **Fixed coefficients.** The coefficients are fitted along one exact
  trajectory and are not refitted. On the rule's own runs, the hidden neurons'
  spike counts change, so off-target coefficients indexed by spike order can
  be applied to different spikes than they were fitted on.
- **The ridges matter.** Without them the fits are numerically singular. With a
  weak off-target ridge (r_C^off = 10⁻³) the hidden weights run away in
  training.
