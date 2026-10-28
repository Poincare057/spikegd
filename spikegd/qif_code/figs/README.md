# Figures for theory_condensed.tex

`generate_figs.py` contains all six Python figure generators, the shared plotting style (fonts and colours sized for the 5.5 in
text width), and the exact event-driven QIF simulator, complex-step event Jacobians, RTRL traces and Benettin/QR helpers.
Importing it exposes the helpers without generating plots or loading Matplotlib or JAX. Each generator prints the numbers quoted
in its caption; `--log-dir` saves those numbers in the corresponding `fig_<name>_log.txt` file.

| Figure | Source |
|---|---|
| Fig. 1, three neuron network | `toy_net_body.tex` (TikZ; standalone wrapper `toy_net.tex`) |
| Fig. 2(a), R1 current with and without the re-entry | `generate_figs.py r1_train` |
| Fig. 2(b), R1 computational graph | `r1_graph_body.tex` (`r1_graph.tex`) |
| Fig. 3, paths from A's weights to C's target, blocks of J(k) | `toy_graph_body.tex` (`toy_graph.tex`) |
| App. E, slices of the tensor M | `slices_body.tex`, `horiz_body.tex` (`M_slices.tex`, `M_horiz.tex`) |
| App. F, QR exponents vs. magnitude along a train | `generate_figs.py qr_train` |
| App. F, capacity (gap count, outputs vs T, spread of c2) | `generate_figs.py capacity` |
| App. F, sources of variation of the QR exponents | `generate_figs.py qr_seeds` |
| App. F, random Gaussian weights (variance g^2/sqrt(l)) | `generate_figs.py random_weights` |
| Fig. 5, pseudospike probe (uses spikegd; cached in `pseudospike_data.npz`, `--recompute` to rerun) | `generate_figs.py pseudospike` |

From this folder:

```bash
python3 generate_figs.py                              # all six Python figures
python3 generate_figs.py r1_train qr_train             # selected figures
python3 generate_figs.py --output-dir figs --log-dir figs
python3 generate_figs.py pseudospike --output-dir figs --recompute
```

Python figures are saved only as SVG (`r1_train.svg`, `qr_train.svg`, etc.). The output directory defaults to the current directory and also
holds `pseudospike_data.npz`; JAX and spikegd are needed only when this cache is missing or `--recompute` is passed.

`./build_figs.sh` rebuilds the Python figures and TikZ figures (numpy, matplotlib, pdflatex). Compile the paper from the folder above `figs/`.
