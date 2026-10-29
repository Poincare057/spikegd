# Code for "Exact Gradients of QIF Neurons and Online Learning Rules"

Everything that produces a number or a figure in `theory_condensed.tex`, with one README per folder.

```
qif_code/
  figs/      SVG figures: generate_figs.py (numbers, plots and simulator); README.md maps figure -> command
  numerics/  quick validation checks of the simulator and of the weight bounds of Prop. 4
  fdcheck/   full check of every closed form and gradient against 50-digit FD, spikegd (Klos & Memmesheimer) and FD of trials
```


## Run

```bash
cd figs && ./build_figs.sh                 # six SVG figures in figs/
cd numerics && python3 checks.py && python3 check_wbound.py
cd fdcheck && python3 fd_check_all.py      # ~10 min; writes all FD output to ../RUN_RESULTS.md
```

## Core modules

- `figs/generate_figs.py` – all Python figure generators and the event-driven QIF neuron (tau_m = 2 tau_s = 1): closed forms,
  event maps differentiated by the complex step, RTRL traces, Benettin/QR, splitting a run into input groups.
- `fdcheck/paper_forms.py` – the paper's formulas transcribed literally, equation by equation.
- `fdcheck/rtrl_paper.py` – exact gradients assembled only from those formulas (single neuron, R1, N-neuron network).
- `fdcheck/sg.py` – float64 wrappers around spikegd.

