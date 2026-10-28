# QIF bundle execution results

Completed October 28, 2026 (America/Chicago). The ZIP was extracted into `spikegd/qif_code` in the existing spikegd repository. Supplied Python sources were run without changes.

## Full gradient validation

**98 of 99 checks pass; 1 known erratum; 0 unexplained failures. Runtime: 476 seconds.**

- [Fresh detailed report](fdcheck/FD_REPORT.md)
- [Fresh machine-readable results](fdcheck/fd_results.json)
- [Execution log](fdcheck/fd_check_all_log.txt)
- Original supplied results are preserved as `fdcheck/FD_REPORT.bundled.md` and `fdcheck/fd_results.bundled.json`.

The expected discrepancy is the printed voltage derivative with respect to current on the invariant line: it differentiates along that line rather than holding the initial voltage fixed. The supplied README says the paper has already corrected it. This run reproduces that discrepancy; no formulas were changed to suppress it.

The three-neuron network gradients agree with spikegd autodiff to a maximum scaled error of 1.35e-7 and finite differences to 5.19e-7. The final pseudospike gradient check passes at 1.31e-8.

## Numerical checks

- Out-to-out Jacobian: maximum absolute difference 4.44e-16.
- Time-to-event identity: all four tested branches yield -1 to the printed precision.
- RTRL versus whole-trial finite differences: maximum reported relative error 5.25e-10 across three parameter comparisons.
- Telescoping amplification identity: maximum absolute difference 2.84e-14; 8 outputs; final amplification 24.572.
- All printed eligible weight-bound checks report true. The bounds require no spike within the input window. Rows reporting 100% window spikes have no eligible samples, so their true result is vacuous rather than additional validation.

Logs: [simulator checks](numerics/checks_log.txt), [weight-bound checks](numerics/check_wbound_log.txt).

## Figures

All six Python figure scripts completed, producing PDF and PNG files and fresh `fig_*_log.txt` logs.

| Output | Main result |
|---|---|
| [r1_train.pdf](figs/r1_train.pdf) | Recurrent weight 12 produces spikes at approximately 0.987, 2.327, 3.769, 5.240; weight 0 produces only the first spike. |
| [qr_train.pdf](figs/qr_train.pdf) | QR exponents match predictions; example voltage amplification factors are 11, 21, 41 and weight factors are 121, 441, 1681. |
| [capacity.pdf](figs/capacity.pdf) | 19,837 groups; zero mismatches with the exact output-count prediction. |
| [qr_seeds.pdf](figs/qr_seeds.pdf) | Logs quantify variation from input timing, weights, initial state, and QR basis. |
| [random_weights.pdf](figs/random_weights.pdf) | For sampled trains with at least two outputs, median voltage amplification is 7.3 and median weight amplification is 16.1. |
| [pseudospike.pdf](figs/pseudospike.pdf) | Late-silence gradient decay per time unit is approximately 0.368, consistent with exp(-1). |

The pseudospike figure uses the supplied `pseudospike_data.npz` cache, the README's default behavior. It was not run with `--recompute`. Separately, the full gradient suite freshly computed and passed 20 pseudospike gradient comparisons.

All five TikZ sources (`toy_net.tex`, `r1_graph.tex`, `toy_graph.tex`, `M_slices.tex`, `M_horiz.tex`) compiled successfully using the Codex desktop LaTeX compiler. This verified compilation but did not export their PDFs into this directory. The shell script's `pdflatex` step could not run directly because the system has no `pdflatex`; its six Python steps were executed individually.

## Environment and reproduction

WSL Linux, Python 3.10.12, CPU JAX/JAXlib 0.6.2, NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.9, mpmath 1.4.1. The existing repository was installed editable with the dependencies required by this bundle. See [installed versions](run_environment.txt).

From the repository root in WSL:

```bash
source .venv-wsl/bin/activate
cd spikegd/qif_code/figs
for f in fig_r1_train fig_qr_train fig_capacity fig_qr_seeds fig_random_weights fig_pseudospike; do
  python -u "$f.py" > "${f}_log.txt" 2>&1
done
cd ../numerics
python -u checks.py > checks_log.txt 2>&1
python -u check_wbound.py > check_wbound_log.txt 2>&1
cd ../fdcheck
python -u fd_check_all.py > fd_check_all_log.txt 2>&1
```

The five standalone TeX files can be compiled in the desktop LaTeX editor or with `pdflatex` after installing a TeX distribution.
