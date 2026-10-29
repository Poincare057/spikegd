# Reproducing the QIF figures

This folder generates the figures for *Exact Gradients of QIF Neurons and
Online Learning Rules* and checks the event-driven simulator and gradients.

## Dependencies

Use Python 3.10+ with NumPy and Matplotlib for figure generation, JAX and
jaxtyping for the pseudospike simulation, and SciPy and mpmath for validation.
From the repository root:

```bash
python -m pip install -r requirements.txt
cd spikegd/capacity_bound
```

## Generate the figures

Run from this folder to generate all six figures, recomputing the pseudospike
simulation instead of reusing its cache:

```bash
python generate_figs.py --recompute
```

To generate selected main figures or reuse cached pseudospike data:

```bash
python generate_figs.py r1_train
python generate_figs.py pseudospike --recompute
python generate_figs.py
```

`generate_figs.py` contains the simulator and all figure generators. Parameters
are fixed in the corresponding `generate_*` functions; the CLI selects figures
and controls cache reuse.

 `r1_train` Recurrent current, with and without re-entry: recurrent weight 12, delay 0.15, external weight 10 at time 0.3, trial duration 6.5. |

`pseudospike` Pseudospike gradient decay: spike counts 2, 5, 10; softplus scales 1, 10, 100; time after the last spike 0.02–6; float64, tau 1, eps 1e-10. |

`qr_train` QR exponents and gradient growth: one input at time 0.1, c2/pi = 5.5, 10.5, 20.5; trial duration 200. |

`capacity` Output capacity: weight 250; input counts 1, 2, 4, 8; durations 1, 2, 4, 8, 16; 300 draws per pair; seed 11. |

`qr_seeds` QR variability: input rate 1, duration 60, two synapses; vary input, weight, initial-state, and QR-basis seeds over 20 runs each. |

`random_weights` Gaussian weights: 100 synapses; gains 1, 10, 30, 100, 300; 1,000 draws per gain; duration 40; seed 21. |

SVGs are saved in `output/`, numerical summaries in `output/RUN_RESULTS.md`,
and the pseudospike cache in `output/data/pseudospike_data.npz`. A missing cache
is recomputed automatically.

## Validate the results

Run sequentially from this folder:

```bash
python checks.py
python fd_check_all.py
```

`checks.py` checks the simulator, Jacobians, RTRL traces, and capacity bounds.
`fd_check_all.py` validates closed forms and gradients using `paper_forms.py`,
`rtrl_paper.py`, the independent reference simulator `r1_sim.py`, and float64
spikegd wrappers in `sg.py`. Both commands update their sections in
`output/RUN_RESULTS.md`.
