## Numerical checks

- Out-to-out Jacobian: maximum absolute difference 4.44e-16.
- Time-to-event identity: all four tested branches yield -1 to the printed precision.
- RTRL versus whole-trial finite differences: maximum reported relative error 5.25e-10 across three parameter comparisons.
- Telescoping amplification identity: maximum absolute difference 2.84e-14; 8 outputs; final amplification 24.572.
- All printed eligible weight-bound checks report true. The bounds require no spike within the input window. Rows reporting 100% window spikes have no eligible samples, so their true result is vacuous rather than additional validation.
## Figures

All six Python figure scripts completed. The figure outputs are SVG. TeX and log files have been removed, and `figs/build_figs.sh` builds the six SVG figures without separate logs.

| Output | Main result |
|---|---|
| [r1_train.svg](figs/figs/r1_train.svg) | Recurrent weight 12 produces spikes at approximately 0.987, 2.327, 3.769, 5.240; weight 0 produces only the first spike. |
| [qr_train.svg](figs/figs/qr_train.svg) | QR exponents match predictions; example voltage amplification factors are 11, 21, 41 and weight factors are 121, 441, 1681. |
| [capacity.svg](figs/figs/capacity.svg) | 19,837 groups; zero mismatches with the exact output-count prediction. |
| [qr_seeds.svg](figs/figs/qr_seeds.svg) | Quantifies variation from input timing, weights, initial state, and QR basis. |
| [random_weights.svg](figs/figs/random_weights.svg) | For sampled trains with at least two outputs, median voltage amplification is 7.3 and median weight amplification is 16.1. |
| [pseudospike.svg](figs/figs/pseudospike.svg) | Late-silence gradient decay per time unit is approximately 0.368, consistent with exp(-1). |

The pseudospike figure uses the supplied `pseudospike_data.npz` cache, the README's default behavior. It was not run with `--recompute`. Separately, the full gradient suite freshly computed and passed 20 pseudospike gradient comparisons.



## Environment and reproduction
From the repository root:

```bash
source .venv-wsl/bin/activate
cd spikegd/qif_code/figs
python -u generate_figs.py --output-dir figs
cd ../numerics
python -u checks.py
python -u check_wbound.py
cd ../fdcheck
python -u fd_check_all.py  # writes the full FD report, JSON records and log to ../RUN_RESULTS.md
```
