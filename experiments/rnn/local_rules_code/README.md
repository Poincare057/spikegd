# Reproducing the local learning rule figures

Compare exact, local, and fitted gradients in a 10-neuron QIF recurrent network.

## Dependencies

Use Python 3.10+ with JAX, jaxtyping, Optax, NumPy, Matplotlib, and tqdm.

If WSL is available, GPU acceleration is recommended.

From the repository root, install the package and shared requirements:

```bash
python -m pip install -r requirements.txt
cd experiments/rnn/local_rules_code
```

## Loss and gradient similarity figures

Run these commands in order from this folder:

```bash
python fit.py collect --trials 10000 --stride 10 --n-hidden 8
python fit.py fit --block 50 --r-A 1 --r-C 0.001 --r-C-off 1
python train.py --rules exact local fitted --sigmas 0 0.05 0.2 --n-seeds 5 --trials 10000 --lr 0.01 --decay-steps 2000 --record 50 --n-eval 30
python cosine.py --every 250
python plot.py --neurons 0 1 4 7 8
```

`fit.py` collects an exact-gradient trajectory and fits fixed correction
coefficients; 

`train.py` compares the three rules; 

`cosine.py` measures their per-neuron gradient similarity;

`plot.py` draws the figures. 

`fastgrad.py`
implements the local and fitted eligibility traces, using the shared task in
`../rnn.py`.

The task uses seed 7, trial duration 10, two target neurons, and input weights
+5 and -5 at rate 1. Training uses AdaBelief with exponentially decaying learning
rate and input spike-time jitter of standard deviation 0, 0.05, or 0.2.
The zero-jitter run is deterministic and copied across seeds. Floating-point
and JAX-version differences can change individual training trajectories.

Results go to `output/data/`; figures go to `output/figs/`:

- `fig_loss.svg`: training loss and final held-out jittered loss.
- `fig_cosine.svg`: per-neuron gradient similarity and update frequency.
- `fig_cosine_neurons.svg`: gradient similarity over training for selected neurons.

## Decomposition figure

`experiment.py` also produces `output/figs/main_figure.svg`, comparing sink and
acyclic architectures and downstream partitions. From this folder, run:

```bash
python experiment.py train --seed 7 --trials 400 --lr 0.01
python experiment.py decompose --arch sink --eps 0 0.25 0.5 1
python experiment.py decompose --arch acyclic --eps 0 0.25 0.5 1
python experiment.py partitions --n-down 5 --n-partitions 6 --part-seed 0
python experiment.py figure
```

These commands reference neuron 0, weight scale 3, and checkpoints at
trials 0, 10, 25, 50, 100, 200, and 400. Intermediate snapshots and measurements
are saved in `output/data/`.
